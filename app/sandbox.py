"""Run model-written Python against unit tests in a locked-down subprocess.

The code is the model's, so it is treated as hostile. There are four layers, and each one is meant
to hold when the one before it is fooled:

1. A static check (check_code_safety): only a short list of standard-library modules may be imported,
   and open / eval / exec and the attributes used to climb out of a sandbox are refused before
   anything runs.
2. A separate process: Python's base interpreter in isolated mode (no site-packages, no user
   settings), in a throwaway folder, with an almost empty environment, so no API key is in reach.
3. Inside that process, an audit hook (sandbox_runner.lock_down) that refuses everything except what
   a pure-Python solution needs: no file outside the sandbox folder and Python's own library, no
   writing at all, no sockets, processes, threads, registry or ctypes.
4. Limits set by the operating system, which do not depend on Python behaving. On Windows a Job
   Object caps memory and CPU time, forbids child processes, throttles outgoing network traffic to
   one byte per second and ends the process with ours, and the process runs at low integrity, so
   Windows itself refuses its writes to your files. Elsewhere, resource limits (not tested on
   Windows, where this app is developed).

The expected results stay in this process. The sandbox only reports what each call returned, so
code that prints "passed" gains nothing.

What this cannot stop is listed in README.md ("Good to know"): mainly, reading files and using the
network are blocked by layer 3 only, which is Python code and not a security boundary.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 5.0
MEMORY_LIMIT_MB = 256            # committed memory the sandbox process may use
MAX_OUTPUT_BYTES = 1_000_000     # of stdout, and of stderr, that is read before the process is stopped
_MARK = "@@MODELLAB@@"
_RUNNER = Path(__file__).with_name("sandbox_runner.py").read_text(encoding="utf-8")

ALLOWED_IMPORTS = {
    "__future__", "abc", "array", "bisect", "collections", "copy", "dataclasses", "datetime",
    "decimal", "enum", "fractions", "functools", "heapq", "itertools", "json", "math",
    "numbers", "operator", "random", "re", "statistics", "string", "textwrap", "time",
    "typing", "unicodedata",
}
# Builtins a solution to a programming exercise has no use for. A name the code binds itself (a
# parameter called "input", a variable called "vars") is the code's own and is left alone.
BLOCKED_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
                 "exit", "quit", "globals", "locals", "vars", "__builtins__"}
# Attributes that lead from an ordinary object to the interpreter's internals. Ordinary attribute
# names are not listed, however they might be misused: what they could reach is refused when the
# code runs, and "self.modules" is a perfectly good attribute.
BLOCKED_ATTRIBUTES = {
    "__subclasses__", "__globals__", "__builtins__", "__code__", "__closure__", "__mro__", "__bases__",
    "__base__", "__getattribute__", "__import__", "__loader__", "__spec__", "__reduce__", "__reduce_ex__",
    "__traceback__", "__self__", "__func__", "__wrapped__", "f_globals", "f_builtins", "f_locals", "f_back",
    "f_code", "tb_frame", "tb_next", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "func_globals",
}


class SandboxError(RuntimeError):
    """The sandbox could not be set up, so the code was not run."""


def _own_names(tree: ast.AST) -> set[str]:
    """Every name the code binds itself: parameters, assigned variables, functions, classes, imports."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def check_code_safety(code: str) -> str | None:
    """Return a reason the code should not be run, or None if it looks fine.

    This is the first, coarse filter: it catches code that plainly reaches outside a programming
    exercise and says why in words. It is not what keeps hostile code in; the process is.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except (ValueError, RecursionError, MemoryError) as exc:      # null bytes, absurd nesting
        return f"Blocked: the code could not be parsed ({type(exc).__name__})"
    own = _own_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module.split(".")[0] not in ALLOWED_IMPORTS:
                return f"Blocked: imports '{module}', which is outside the sandbox allowlist"
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES and node.id not in own:
            return f"Blocked: uses '{node.id}', which is not allowed in the sandbox"
        if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTES:
            return f"Blocked: uses '.{node.attr}', which is not allowed in the sandbox"
        # The same names as text, as in getattr(x, "__globals__"). Ordinary words are left alone.
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in BLOCKED_ATTRIBUTES
                and node.value.startswith(("__", "f_", "tb_", "gi_", "cr_", "ag_"))):
            return f"Blocked: names '{node.value}', which is not allowed in the sandbox"
    return None


# --------------------------------------------------------------------------
# Limits enforced by the operating system
# --------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _UiRestrictions(ctypes.Structure):
        _fields_ = [("UIRestrictionsClass", wintypes.DWORD)]

    class _NetRateControl(ctypes.Structure):
        _fields_ = [("MaxBandwidth", ctypes.c_uint64), ("ControlFlags", wintypes.DWORD), ("DscpTag", ctypes.c_ubyte)]

    class _SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    class _MandatoryLabel(ctypes.Structure):
        _fields_ = [("Label", _SidAndAttributes)]

    _HANDLE, _DWORD, _POINTER = wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p
    for _function, _result, _arguments in (
            (_kernel32.CreateJobObjectW, _HANDLE, (_POINTER, wintypes.LPCWSTR)),
            (_kernel32.SetInformationJobObject, wintypes.BOOL, (_HANDLE, ctypes.c_int, _POINTER, _DWORD)),
            (_kernel32.AssignProcessToJobObject, wintypes.BOOL, (_HANDLE, _HANDLE)),
            (_kernel32.TerminateJobObject, wintypes.BOOL, (_HANDLE, wintypes.UINT)),
            (_kernel32.CloseHandle, wintypes.BOOL, (_HANDLE,)),
            (_kernel32.LocalFree, _POINTER, (_POINTER,)),
            (_advapi32.OpenProcessToken, wintypes.BOOL, (_HANDLE, _DWORD, ctypes.POINTER(_HANDLE))),
            (_advapi32.ConvertStringSidToSidW, wintypes.BOOL, (wintypes.LPCWSTR, ctypes.POINTER(_POINTER))),
            (_advapi32.GetLengthSid, _DWORD, (_POINTER,)),
            (_advapi32.SetTokenInformation, wintypes.BOOL, (_HANDLE, ctypes.c_int, _POINTER, _DWORD)),
            (_ntdll.NtResumeProcess, ctypes.c_long, (_HANDLE,))):
        _function.restype, _function.argtypes = _result, _arguments

    _CREATE_SUSPENDED = 0x00000004
    _LIMIT_PROCESS_TIME, _LIMIT_ACTIVE_PROCESS, _LIMIT_PROCESS_MEMORY = 0x0002, 0x0008, 0x0100
    _LIMIT_DIE_ON_UNHANDLED_EXCEPTION, _LIMIT_KILL_ON_JOB_CLOSE = 0x0400, 0x2000
    _EXTENDED_LIMIT_INFORMATION, _BASIC_UI_RESTRICTIONS, _NET_RATE_CONTROL = 9, 4, 32
    _NET_RATE_ENABLE, _NET_RATE_MAX_BANDWIDTH = 0x1, 0x2
    NETWORK_BYTES_PER_SECOND = 1     # outgoing traffic the job may send: in effect, none
    _UI_LIMIT_ALL = 0xFF             # no clipboard, no desktop or display changes, no logging the user off ...
    _TOKEN_ADJUST_DEFAULT, _TOKEN_QUERY, _TOKEN_INTEGRITY_LEVEL, _SE_GROUP_INTEGRITY = 0x0080, 0x0008, 25, 0x20
    _LOW_INTEGRITY_SID = "S-1-16-4096"
    _OUT_OF_CPU_TIME = 0xC0000044    # STATUS_QUOTA_EXCEEDED: the exit code of a process the job stopped

    def _failed(step: str) -> SandboxError:
        return SandboxError(f"{step} failed (Windows error {ctypes.get_last_error()})")

    def _confine(process: subprocess.Popen, cpu_seconds: float) -> int:
        """Put a suspended process under the limits and let it run. Returns the job handle; closing it
        (or the end of this process) ends everything in the job."""
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            raise _failed("CreateJobObject")
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = (
                _LIMIT_PROCESS_TIME | _LIMIT_ACTIVE_PROCESS | _LIMIT_PROCESS_MEMORY
                | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION | _LIMIT_KILL_ON_JOB_CLOSE)
            limits.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_seconds * 10_000_000)   # 100 ns steps
            limits.BasicLimitInformation.ActiveProcessLimit = 1       # it cannot start another process
            limits.ProcessMemoryLimit = MEMORY_LIMIT_MB * 1024 * 1024
            if not _kernel32.SetInformationJobObject(job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits),
                                                     ctypes.sizeof(limits)):
                raise _failed("Setting the job limits")
            ui = _UiRestrictions(_UI_LIMIT_ALL)
            if not _kernel32.SetInformationJobObject(job, _BASIC_UI_RESTRICTIONS, ctypes.byref(ui),
                                                     ctypes.sizeof(ui)):
                raise _failed("Setting the job's desktop restrictions")
            # Windows 10 and later can cap what a job sends to the network. Measured here: at 1 byte per
            # second a plain HTTP request to another machine did not get out in 25 seconds (0.5 s without).
            # It does not cover this machine's own loopback addresses, nor name lookups, which a Windows
            # service makes on the process's behalf.
            rate = _NetRateControl(NETWORK_BYTES_PER_SECOND, _NET_RATE_ENABLE | _NET_RATE_MAX_BANDWIDTH, 0)
            if not _kernel32.SetInformationJobObject(job, _NET_RATE_CONTROL, ctypes.byref(rate),
                                                     ctypes.sizeof(rate)):
                raise _failed("Setting the job's network limit")
            handle = int(process._handle)
            if not _kernel32.AssignProcessToJobObject(job, handle):
                raise _failed("Putting the process in the job")
            _lower_integrity(handle)
            if _ntdll.NtResumeProcess(handle) != 0:
                raise SandboxError("Resuming the sandbox process failed")
        except BaseException:
            _kernel32.TerminateJobObject(job, 1)
            _kernel32.CloseHandle(job)
            raise
        return job

    def _lower_integrity(process_handle: int) -> None:
        """Give the process a low integrity level: Windows then refuses its writes to files, folders and
        registry keys of normal (medium) integrity, which is nearly everything the user owns."""
        token, sid = wintypes.HANDLE(), ctypes.c_void_p()
        access = _TOKEN_ADJUST_DEFAULT | _TOKEN_QUERY
        if not _advapi32.OpenProcessToken(process_handle, access, ctypes.byref(token)):
            raise _failed("Opening the sandbox process token")
        try:
            if not _advapi32.ConvertStringSidToSidW(_LOW_INTEGRITY_SID, ctypes.byref(sid)):
                raise _failed("Building the low-integrity label")
            label = _MandatoryLabel(_SidAndAttributes(sid, _SE_GROUP_INTEGRITY))
            size = ctypes.sizeof(label) + _advapi32.GetLengthSid(sid)
            if not _advapi32.SetTokenInformation(token, _TOKEN_INTEGRITY_LEVEL, ctypes.byref(label), size):
                raise _failed("Lowering the integrity level")
        finally:
            if sid:
                _kernel32.LocalFree(sid)
            _kernel32.CloseHandle(token)

    def _release(job: int | None) -> None:
        if job:
            _kernel32.TerminateJobObject(job, 1)
            _kernel32.CloseHandle(job)

    def _start(command: list[str], folder: Path, env: dict[str, str], cpu_seconds: float):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | _CREATE_SUSPENDED
        process = subprocess.Popen(command, cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        try:
            return process, _confine(process, cpu_seconds)
        except BaseException:
            process.kill()
            process.wait()
            for stream in (process.stdout, process.stderr):
                stream.close()
            raise

else:
    _OUT_OF_CPU_TIME = -24           # ended by SIGXCPU

    def _start(command: list[str], folder: Path, env: dict[str, str], cpu_seconds: float):
        def limits() -> None:        # runs in the child, before Python starts
            import resource
            os.setsid()
            megabyte = 1024 * 1024
            for name, value in (("RLIMIT_CPU", int(cpu_seconds) + 1), ("RLIMIT_AS", MEMORY_LIMIT_MB * 4 * megabyte),
                                ("RLIMIT_FSIZE", 0), ("RLIMIT_CORE", 0), ("RLIMIT_NPROC", 0)):
                try:
                    resource.setrlimit(getattr(resource, name), (value, value))
                except (AttributeError, ValueError, OSError):
                    pass
        process = subprocess.Popen(command, cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=limits)
        return process, None

    def _release(job: int | None) -> None:
        return None


def _stop(process: subprocess.Popen, job: int | None) -> None:
    _release(job)
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    process.wait()


_DONE = (_MARK + '{"done": true}').encode("ascii")


def _collect(stream: Any, chunks: list[bytes], overflow: threading.Event, done: threading.Event | None = None) -> None:
    """Read a pipe to its end, keeping at most MAX_OUTPUT_BYTES. More than that sets `overflow`.
    `done` is set when the runner's last line has been read."""
    size, tail = 0, b""
    try:
        while True:
            chunk = stream.read1(65536)
            if not chunk:
                return
            if done is not None:
                if _DONE in tail + chunk:
                    done.set()
                tail = (tail + chunk)[-len(_DONE):]      # the line may arrive split over two reads
            size += len(chunk)
            if size <= MAX_OUTPUT_BYTES:
                chunks.append(chunk)
            else:
                overflow.set()       # keep reading, so the writer is not blocked before it is stopped
    except (OSError, ValueError):
        return


def execute(code: str, func_name: str, arguments: list[list[Any]], timeout: float = TIMEOUT_SECONDS,
            lock_down: bool = True) -> dict[str, Any]:
    """Call `func_name` from `code` once per argument list, inside the sandbox process.

    Returns what the process reported: {"values": {case index: [returned values]}, "errors": {index:
    text}, "fatal", "started", "stopped" ("time" | "output" | None), "exit_code", "stderr"}.
    No static check is done here (see run_code_tests). `lock_down=False` leaves the audit hook out;
    the tests use it to check the operating system's limits on their own.
    """
    with tempfile.TemporaryDirectory(prefix="modellab_", ignore_cleanup_errors=True) as tmp:
        folder = Path(tmp)
        (folder / "solution.py").write_text(code, encoding="utf-8")
        (folder / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (folder / "job.json").write_text(json.dumps({
            "func": func_name, "cases": arguments, "imports": sorted(ALLOWED_IMPORTS - {"__future__"}),
            "lock_down": lock_down}), encoding="utf-8")
        # The base interpreter, not the virtual environment's: -I and -S leave out site-packages, the
        # user's settings and every PYTHON* variable. The environment carries no key or token.
        python = getattr(sys, "_base_executable", None) or sys.executable
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "SystemRoot") if k in os.environ}
        process, job = _start([python, "-I", "-S", "-B", "-X", "utf8", "runner.py"], folder, env, timeout + 1)
        out, err, overflow, done = [], [], threading.Event(), threading.Event()
        readers = [threading.Thread(target=_collect, args=(process.stdout, out, overflow, done), daemon=True),
                   threading.Thread(target=_collect, args=(process.stderr, err, overflow), daemon=True)]
        for reader in readers:
            reader.start()
        stopped = None
        deadline = time.monotonic() + timeout
        try:
            # The runner never lets Python shut down by itself (its audit hooks would be gone by then):
            # it reports "done" and is ended here.
            while process.poll() is None and not done.is_set():
                if overflow.is_set():
                    stopped = "output"
                    break
                if time.monotonic() >= deadline:
                    stopped = "time"
                    break
                time.sleep(0.01)
        finally:
            _stop(process, job)
            for reader in readers:
                reader.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                stream.close()
        if stopped is None and overflow.is_set():
            stopped = "output"
        if stopped is None and process.returncode == _OUT_OF_CPU_TIME:
            stopped = "time"

    report: dict[str, Any] = {"values": {}, "errors": {}, "fatal": None, "started": False, "stopped": stopped,
                              "exit_code": process.returncode,
                              "stderr": b"".join(err).decode("utf-8", "replace")}
    for line in b"".join(out).decode("utf-8", "replace").splitlines():
        if not line.startswith(_MARK):
            continue
        try:
            item = json.loads(line[len(_MARK):])
        except (ValueError, RecursionError):     # the model's code can print anything, "[[[[..." included
            continue
        if not isinstance(item, dict):
            continue
        if item.get("ready"):
            report["started"] = True
        elif "fatal" in item:
            report["fatal"] = str(item["fatal"])
        elif isinstance(item.get("i"), int) and not isinstance(item["i"], bool):
            if "err" in item:
                report["errors"][item["i"]] = str(item["err"])
            else:
                report["values"].setdefault(item["i"], []).append(item)
    return report


def _same(value: Any, expected: Any) -> bool:
    """Is the returned value the expected one? Compared the way a unit test's assertEqual would
    (so 8.0 passes for 8), after both went through JSON: a tuple equals a list."""
    try:
        return bool(value == json.loads(json.dumps(expected)))
    except (RecursionError, ValueError, TypeError):
        return False


def run_code_tests(code: str, func_name: str, cases: list[dict[str, Any]],
                   timeout: float = TIMEOUT_SECONDS) -> tuple[int, int, str]:
    """Run `func_name` from `code` against cases. Returns (passed, total, notes)."""
    total = len(cases)
    problem = check_code_safety(code)
    if problem:
        return 0, total, problem
    if f"def {func_name}" not in code:
        return 0, total, f"No function named {func_name}() in the reply"

    report = execute(code, func_name, [case["args"] for case in cases], timeout)
    if report["fatal"]:
        return 0, total, f"Could not load the function - {report['fatal']}"

    # The comparison happens here, on this side: the sandbox never sees the expected values.
    outcome: dict[int, str | None] = {}          # case index -> what went wrong, None if it passed
    for index, items in report["values"].items():
        if not 0 <= index < total:
            continue
        if len(items) != 1 or index in report["errors"]:
            outcome[index] = "more than one result was reported"
        elif items[0].get("unreadable") or "value" not in items[0]:
            outcome[index] = f"got {str(items[0].get('repr'))[:120]}, expected {cases[index]['expected']!r}"
        elif _same(items[0]["value"], cases[index]["expected"]):
            outcome[index] = None
        else:
            outcome[index] = f"got {str(items[0].get('repr'))[:120]}, expected {cases[index]['expected']!r}"
    for index, error in report["errors"].items():
        if 0 <= index < total:
            outcome.setdefault(index, error)

    passed = sum(1 for problem in outcome.values() if problem is None)
    notes = [f"{passed}/{total} unit tests passed"]
    if report["stopped"] == "time":
        notes.append(f"timed out after {timeout:g}s")
    elif report["stopped"] == "output":
        notes.append(f"stopped: it printed more than {MAX_OUTPUT_BYTES // 1_000_000} MB of output")
    elif not outcome:
        tail = report["stderr"].strip().splitlines()[-1][:200] if report["stderr"].strip() else "no output"
        notes.append(f"the test run crashed ({tail})")
    for i in range(total):
        if i not in outcome:
            if report["stopped"] or outcome:
                notes.append(f"case {i + 1}: did not finish")
        elif outcome[i] is not None:
            args = ", ".join(repr(a) for a in cases[i]["args"])
            notes.append(f"case {i + 1}: {func_name}({args[:80]}) -> {outcome[i]}")
    return passed, total, "; ".join(notes)

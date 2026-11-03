"""Runs inside the sandbox process: load the model's code, call its function on each test case and
print what it returned. app/sandbox.py copies this file into a throwaway folder and starts it there
as `python -I -S -B runner.py`, so it must not import anything from the app.

Before the model's code is loaded, lock_down() installs an audit hook that refuses everything except
what an ordinary pure-Python solution needs. The expected values never come into this process: the
parent compares the results itself, so printing "passed" from here gains nothing.

The process must never shut down in the normal way. Python removes its audit hooks part-way through
shutting down and destroys the remaining objects afterwards, so a __del__ method the model's code left
behind (on a codec search function, a module global, a reference cycle) would run with no hook at
all. The last lines of this file see to it: the process ends hard, or waits to be ended.
"""
import json
import os
import sys

MARK = "@@MODELLAB@@"
# Attributes that classes and functions have set on them in normal use (namedtuple, dataclasses,
# functools.wraps, typing). "__code__" is not one of them.
SETTABLE = ("__bases__", "__class__", "__defaults__", "__kwdefaults__", "__doc__", "__module__", "__name__",
            "__qualname__", "__annotations__")
# Audit events that are always fine. Anything not named here, and not handled in the hook, is refused.
HARMLESS = ("exec", "compile", "builtins.id", "array.__new__", "sys._getframe", "sys._getframemodulename",
            "object.__getattr__", "time.sleep", "sys.excepthook", "sys.unraisablehook",
            "cpython._PySys_ClearAuditHooks", "cpython.PyInterpreterState_Clear")
# The parts of the import machinery that load modules built into the interpreter (gc, mmap, msvcrt,
# _xxsubinterpreters ...), extension modules and frozen ones. No audit event announces a built-in or
# frozen module, so the hook cannot keep them out: after the warm-up these functions are removed.
# A second interpreter, for one, would start with no audit hooks.
LOADERS = ("create_builtin", "exec_builtin", "create_dynamic", "exec_dynamic", "get_frozen_object", "init_frozen",
           "_override_frozen_modules_for_tests", "_override_multi_interp_extensions_check")
# Standard-library modules that allowed modules import the first time some function is called.
LAZY_IMPORTS = ("_strptime", "calendar", "locale", "_locale", "_bootlocale", "linecache", "tokenize", "token",
                "traceback", "weakref", "types", "_string", "_pydecimal", "_pydatetime", "stringprep",
                "contextvars", "_contextvars", "bisect", "heapq")


def emit(obj):
    sys.stdout.write(MARK + json.dumps(obj) + "\n")
    sys.stdout.flush()


def warm_up(modules):
    """Import every module a solution may use, and call the functions known to import more on first
    use, while imports are still unrestricted. After lock_down() only these (and LAZY_IMPORTS) load."""
    for name in modules:
        __import__(name)
    import collections
    import datetime
    import functools
    import time
    import warnings

    datetime.datetime.strptime("2000-01-01", "%Y-%m-%d")
    time.strptime("2000", "%Y")
    collections.Counter("ab").most_common(1)
    functools.singledispatch(len)
    for codec in ("ascii", "latin-1", "utf-8", "utf-16", "utf-32", "cp1252", "cp1256"):
        "x".encode(codec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("warm-up", UserWarning)


def lock_down(folder, read_roots, loaded):
    """Install the audit hook. From here on this process can only:

    - read files in `folder` (the sandbox folder);
    - import modules that are already loaded, their submodules, and LAZY_IMPORTS, and read the source
      of exactly those modules in `read_roots` (Python's own library). importlib can load a module
      without the "import" event that announces an import statement, so the files are guarded too;
    - run Python code.

    No file can be written, created, deleted or renamed. No socket, process, thread, registry key or
    ctypes call is possible through any audited function, and marshal / code objects cannot be built
    (the usual way out of a Python-level sandbox is hand-made bytecode).

    The hook keeps everything it needs in its own default arguments. It looks up no global and no
    builtin name when it runs: the model's code can replace those (builtins.isinstance, os.path ...),
    and can read the hook's local variables from a traceback, but tuples and strings cannot be changed.
    """
    if sys.platform == "win32":
        import nt
        full_path, sep, cwd = nt._getfullpathname, "\\", None      # the system resolves "..", drives, devices
    else:
        full_path, sep, cwd = None, "/", os.getcwd()
    folder = os.path.abspath(folder)
    library = tuple(os.path.abspath(root) for root in read_roots if root)
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_TEMPORARY", 0)
    allowed = frozenset(name.partition(".")[0] for name in loaded) | frozenset(LAZY_IMPORTS) | {"solution"}

    import _imp
    for name in LOADERS:             # nothing else refers to them: once gone from here, they are gone
        if hasattr(_imp, name):
            delattr(_imp, name)
    # _winapi is loaded at start-up for os.path, which keeps the three names it needs. Several of its
    # functions make handles, pipes and file mappings without raising any audit event; none is needed.
    winapi = sys.modules.get("_winapi")
    for name in list(vars(winapi)) if winapi is not None else ():
        if not name.startswith("__"):
            delattr(winapi, name)

    def hook(event, args, _folder=folder, _library=library, _allowed=allowed, _harmless=frozenset(HARMLESS),
             _settable=frozenset(SETTABLE),
             _full_path=full_path, _sep=sep, _cwd=cwd, _write_flags=write_flags, _type=type, _str=str, _int=int,
             _bytes=bytes, _len=len, _refuse=PermissionError, _no_import=ImportError):
        if event in _harmless:
            return
        if event == "import":
            name = args[0]
            if _type(name) is _str and name.partition(".")[0] in _allowed:
                return
            raise _no_import("sandbox: importing this module is not allowed")
        if event == "object.__setattr__" or event == "object.__delattr__":
            if _type(args[1]) is _str and args[1] in _settable:
                return
            raise _refuse("sandbox: changing this attribute is not allowed")
        if event == "open" or event == "os.listdir" or event == "os.scandir":
            path = args[0]
            if event == "open":
                mode, flags = args[1], args[2]
                if _type(mode) is _str and ("w" in mode or "a" in mode or "x" in mode or "+" in mode):
                    raise _refuse("sandbox: writing files is not allowed")
                if _type(flags) is _int and flags & _write_flags:
                    raise _refuse("sandbox: writing files is not allowed")
                if _type(mode) is not _str and _type(flags) is not _int:
                    raise _refuse("sandbox: this way of opening a file is not allowed")
            elif path is None:
                path = "."
            if _type(path) is _bytes:
                try:
                    path = path.decode("utf-8")
                except ValueError:
                    raise _refuse("sandbox: this file name is not allowed") from None
            if _type(path) is not _str or "\x00" in path:        # not a plain path: a file descriptor, an object ...
                raise _refuse("sandbox: only plain file names are allowed")
            if _full_path is not None:
                try:
                    path = _full_path(path)
                except OSError:
                    raise _refuse("sandbox: this file name is not allowed") from None
            else:
                parts = []
                for part in ((path if path.startswith("/") else _cwd + "/" + path)).split("/"):
                    if part == "..":
                        if parts:
                            parts.pop()
                    elif part and part != ".":
                        parts.append(part)
                path = "/" + "/".join(parts)
            if path == _folder or (path.startswith(_folder) and path[_len(_folder):_len(_folder) + 1] == _sep):
                return
            for root in _library:
                if path == root and event != "open":
                    return                                   # the import system lists these folders
                if path.startswith(root) and path[_len(root):_len(root) + 1] == _sep:
                    if event != "open":
                        return
                    # <library>/json/decoder.py and <library>/__pycache__/json.cpython-312.pyc belong to "json".
                    first, _, rest = path[_len(root) + 1:].partition(_sep)
                    if first == "__pycache__":
                        first = rest
                    if first.partition(".")[0] in _allowed:
                        return
            raise _refuse("sandbox: files outside the sandbox folder are not allowed")
        raise _refuse("sandbox: '" + event + "' is not allowed")

    sys.addaudithook(hook)


def plain(value, depth=0):
    """A returned value as JSON can carry it: tuples as lists, Decimal and Fraction as numbers."""
    if depth > 50:
        raise ValueError("too deeply nested")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: plain(item, depth + 1) for key, item in value.items()}
    numerator, denominator = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if isinstance(numerator, int) and isinstance(denominator, int):
        return numerator if denominator == 1 else numerator / denominator
    if type(value).__name__ == "Decimal":
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError("not a plain value")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "job.json"), encoding="utf-8") as handle:
        job = json.load(handle)
    library = [path for path in sys.path if path]          # Python's own library: read-only, needed to import
    sys.path.insert(0, here)
    warm_up(job["imports"])
    # Compile later imports from source: reading bytecode needs marshal, which is refused from here on.
    sys.pycache_prefix = os.path.join(here, "no-bytecode")
    if job.get("lock_down", True):
        lock_down(here, library, tuple(sys.modules))
    emit({"ready": True})
    try:
        import solution
        fn = getattr(solution, job["func"])
    except BaseException as exc:
        emit({"fatal": type(exc).__name__ + ": " + str(exc)[:200]})
        return
    for i, args in enumerate(job["cases"]):
        try:
            got = fn(*args)
            try:
                emit({"i": i, "value": json.loads(json.dumps(plain(got))), "repr": repr(got)[:120]})
            except Exception:
                emit({"i": i, "unreadable": True, "repr": repr(got)[:120]})
        except BaseException as exc:
            emit({"i": i, "err": type(exc).__name__ + ": " + str(exc)[:120]})
    emit({"done": True})             # the parent ends the process when it reads this


if __name__ == "__main__":
    # Whatever happened above, do not let the interpreter shut down (see the top of this file).
    # The model's code may have replaced os._exit, so that is only the quick way out; the loop below
    # looks up no name at all, and nothing the model left behind can run while it spins. The parent
    # ends the process when it reads "done", and at the time limit otherwise.
    try:
        main()
    finally:
        try:
            os._exit(0)
        finally:
            while True:
                try:
                    while True:
                        pass
                except:              # noqa: E722 - naming an exception class would be a lookup
                    pass

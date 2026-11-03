"""Escape attempts against the coding sandbox, one route at a time.

Most attacks are run with sandbox.execute(), which skips the static check: the point is that the
process itself holds when the check before it is fooled. Tests with lock_down=False switch the audit
hook off as well, to show what the operating system stops on its own. The last section records
what is NOT stopped, so the limits stated in the README are checked facts.
"""
import contextlib
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from app import graders, sandbox

windows = pytest.mark.skipif(sys.platform != "win32", reason="checks the Windows job object and integrity level")

# Runs inside the sandbox with every attack: try each action and say what happened to it.
PRELUDE = '''
import os, sys

def tries(**attempts):
    results = {}
    for name, action in attempts.items():
        try:
            results[name] = "DONE: " + repr(action())[:300]
        except BaseException as exc:
            results[name] = type(exc).__name__ + ": " + str(exc)[:300]
    return results
'''


class Outcome:
    def __init__(self, report):
        self.report = report
        items = report["values"].get(0) or []
        self.value = items[0].get("value") if items else None
        self.error = report["errors"].get(0) or report["fatal"]
        self.stopped = report["stopped"]

    def __repr__(self):
        return f"Outcome(value={self.value!r}, error={self.error!r}, stopped={self.stopped!r})"


def attack(body, *args, lock_down=True, timeout=10.0):
    """Run `body` as the body of attack(*args) inside the sandbox process."""
    code = PRELUDE + "\ndef attack(*args):\n" + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n"
    return Outcome(sandbox.execute(code, "attack", [list(args)], timeout, lock_down))


# How the sandbox says no: the audit hook's own message, or the function having been removed after
# the warm-up (the module loaders in sandbox_runner.LOADERS, and everything in _winapi).
SANDBOX_SAYS_NO = ("sandbox:", "module '_imp' has no attribute", "module '_winapi' has no attribute")


def refused(results):
    """Every attempt was turned away by the sandbox, and none got through."""
    assert results and isinstance(results, dict), results
    for name, what in results.items():
        assert not what.startswith("DONE"), f"{name} got through: {what}"
        assert any(sign in what for sign in SANDBOX_SAYS_NO), f"{name} failed for another reason: {what}"
    return True


@pytest.fixture
def secret(tmp_path):
    """A file outside the sandbox folder, standing in for .env or any other file of the user."""
    path = tmp_path / "secret.env"
    path.write_text("OPENROUTER_API_KEY=sk-or-v1-NOT-FOR-THE-MODEL\n", encoding="utf-8")
    return path


@contextlib.contextmanager
def listening():
    """A local TCP server -> (port, list that gets one entry per connection it accepted)."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    server.settimeout(0.1)
    seen, done = [], threading.Event()

    def accept():
        while not done.is_set():
            try:
                connection, _ = server.accept()
            except OSError:
                continue
            seen.append(1)
            connection.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1], seen
    finally:
        time.sleep(0.3)
        done.set()
        thread.join()
        server.close()


# ---------------------------------------------------------------- it still runs ordinary solutions

def test_every_allowed_module_works_inside_the_sandbox():
    """The hook refuses by default, so check that it refuses nothing an ordinary solution does."""
    outcome = attack('''
        import abc, array, bisect, collections, copy, dataclasses, datetime, decimal, enum, fractions, functools
        import heapq, itertools, json, math, numbers, operator, random, re, statistics, string, textwrap, time
        import typing, unicodedata

        Point = collections.namedtuple("Point", "x y", defaults=[0])

        @dataclasses.dataclass(order=True, frozen=True)
        class Item:
            name: str
            size: int = 0
            tags: list = dataclasses.field(default_factory=list, compare=False)

        class Color(enum.Enum):
            RED = 1
            BLUE = enum.auto()

        class Row(typing.NamedTuple):
            a: int
            b: str = "x"

        class Shape(abc.ABC):
            @abc.abstractmethod
            def area(self): ...

        class Square(Shape):
            def __init__(self, side):
                super().__init__()
                self.side = side
            def area(self):
                return self.side ** 2

        @functools.lru_cache(maxsize=None)
        def fib(n):
            return n if n < 2 else fib(n - 1) + fib(n - 2)

        @functools.singledispatch
        def show(x):
            return "object"

        @show.register
        def _(x: int):
            return "int"

        def wrapped(f):
            @functools.wraps(f)
            def inner(*a):
                return f(*a)
            return inner

        heap = []
        heapq.heappush(heap, 3)
        bisect.insort(heap, 1)
        with decimal.localcontext() as context:
            context.prec = 30
            third = decimal.Decimal(1) / decimal.Decimal(3)
        return {
            "namedtuple": list(Point(1)._replace(y=2)),
            "dataclass": dataclasses.asdict(sorted([Item("b"), Item("a", 2)])[0]),
            "enum": [Color(1).name, Color["BLUE"].value],
            "typing": [Row(1).b, str(typing.Optional[int]), typing.get_type_hints(Row)["a"].__name__],
            "abc": Square(3).area(),
            "functools": [fib(60), show(3), wrapped(len)("ab"), functools.reduce(operator.mul, [1, 2, 3])],
            "datetime": datetime.datetime.strptime("2024-02-29 10:30", "%Y-%m-%d %H:%M").strftime("%A %d %B %Y"),
            "date": (datetime.date.fromisoformat("2024-02-29") + datetime.timedelta(days=1)).isoformat(),
            "time": [time.strftime("%Y", time.gmtime(0)), time.perf_counter() > 0, time.sleep(0)],
            "decimal": str(third)[:8],
            "fractions": str(fractions.Fraction(3, 4) + fractions.Fraction("1/4")),
            "statistics": [statistics.mean([1, 2, 3]), statistics.median([1, 2, 3, 4])],
            "random": random.Random(7).choice([1, 2, 3]) in (1, 2, 3),
            "re": re.compile(r"(?P<n>\\d+)\\s*(?:h|hours?)", re.I).findall("3 h 4 hours"),
            "json": json.loads(json.dumps({"a": [1, 2.5, None, True]}))["a"],
            "math": [math.gcd(12, 18), math.isqrt(99), math.comb(5, 2)],
            "heap": heap,
            "itertools": ["".join(p) for p in itertools.permutations("ab")],
            "copy": copy.deepcopy({"a": [1, {"b": 2}]}),
            "string": [string.capwords("a b"), string.Template("$x").substitute(x=1)],
            "textwrap": textwrap.shorten("hello world again", 12),
            "unicodedata": unicodedata.normalize("NFKD", "\\u00e9").encode("ascii", "ignore").decode(),
            "array": array.array("i", [1, 2]).tolist(),
            "numbers": isinstance(3, numbers.Integral),
            "codecs": ["x".encode("utf-16").decode("utf-16"), "\\u0633".encode("cp1256").decode("cp1256")],
            "collections": collections.Counter("abracadabra").most_common(1),
            "builtins": [sorted({3: "c", 1: "a"}.items(), key=operator.itemgetter(1))[0][0], format(1234.5, ",.1f"),
                         getattr(heap, "count")(3), id(heap) > 0, hash("a") == hash("a")],
        }
    ''')
    assert outcome.error is None and outcome.stopped is None, outcome
    assert outcome.value["functools"] == [1548008755920, "int", 2, 6]
    assert outcome.value["datetime"] == "Thursday 29 February 2024" and outcome.value["date"] == "2024-03-01"
    assert outcome.value["dataclass"] == {"name": "a", "size": 2, "tags": []}
    assert outcome.value["collections"] == [["a", 5]] and outcome.value["codecs"] == ["x", "\u0633"]
    assert outcome.value["re"] == ["3", "4"] and outcome.value["decimal"] == "0.333333"


def test_values_come_back_as_plain_data():
    outcome = attack('''
        import decimal, fractions
        return [(1, 2), {"k": (3,)}, decimal.Decimal("64.80"), decimal.Decimal(7), fractions.Fraction(6, 3),
                fractions.Fraction(1, 4), True, None, "\\u0633"]
    ''')
    assert outcome.value == [[1, 2], {"k": [3]}, 64.8, 7, 2, 0.25, True, None, "\u0633"]
    # A value JSON cannot carry never equals an expected value.
    cases = [{"args": [], "expected": [1, 2]}]
    assert sandbox.run_code_tests("def f():\n    return {1, 2}", "f", cases)[0] == 0
    assert sandbox.run_code_tests("def f():\n    return (1, 2)", "f", cases)[0] == 1


def test_static_check_lets_ordinary_solutions_through():
    for code in (
        "import re\nfrom collections import defaultdict\nclass Parser:\n    def __init__(self, text):\n"
        "        self.text = text\n    def visit(self, kind):\n        return getattr(self, 'visit_' + kind)()\n"
        "    def visit_number(self):\n        return int(self.text)\n    def __repr__(self):\n"
        "        return f'{self.__class__.__name__}({self.text!r})'\n",
        "from dataclasses import dataclass\n@dataclass\nclass Token:\n    kind: str\n    value: str = ''\n"
        "    def __post_init__(self):\n        self.value = self.value.strip()\n",
        "def f(words):\n    table = {'modules': 1, 'name': 2}\n    return sorted(words, key=lambda w: (len(w), w))\n",
        "import functools\n@functools.lru_cache(None)\ndef f(n: int) -> int:\n    return n if n < 2 else f(n - 1) + f(n - 2)\n",
    ):
        assert sandbox.check_code_safety(code) is None, code


@pytest.mark.parametrize("code", [
    "import os", "import socket", "import subprocess", "import ctypes", "from urllib import request",
    "import importlib", "import sys", "import pathlib", "import shutil", "import gc", "import builtins",
    "x = open('f')", "eval('1')", "exec('x = 1')", "compile('1', 's', 'eval')", "__import__('os')",
    "globals()['x'] = 1", "y = vars()", "b = __builtins__",
    "().__class__.__base__.__subclasses__()", "f.__globals__['os']", "f.__code__ = g.__code__",
    "getattr(f, '__globals__')", "e.__traceback__.tb_frame.f_locals", "x.gi_frame.f_back",
])
def test_static_check_refuses_the_usual_ways_out(code):
    assert (sandbox.check_code_safety(code) or "").startswith("Blocked"), code
    grade = graders.grade_code(f"{code}\ndef add(a, b):\n    return a + b", "add", [{"args": [1, 2], "expected": 3}])
    assert grade.score == 0 and "Blocked" in grade.notes


def test_static_check_leaves_the_codes_own_names_alone():
    """A parameter called "input" or an attribute called "modules" is ordinary code. Refusing it gave a
    correct solution 0 for its choice of names."""
    for code in ("def f(input):\n    return input * 2",
                 "def f(text):\n    input = text.strip()\n    vars = input.split()\n    return len(vars)",
                 "def f(items, open=0):\n    for exit in items:\n        open += exit\n    return open",
                 "class Graph:\n    def __init__(self):\n        self.modules = []\n        self._os = None\n"
                 "    def add(self, x):\n        self.modules.append(x)",
                 "import math as eval\ndef f(x):\n    return eval.sqrt(x)"):
        assert sandbox.check_code_safety(code) is None, code
    assert graders.grade_code("def double(input):\n    return input * 2", "double",
                              [{"args": [4], "expected": 8}]).score == 1
    # The builtin itself, with nothing of that name defined in the code, is still refused.
    for code in ("def f(path):\n    return open(path).read()", "def f(x):\n    return eval(x)",
                 "def f(x):\n    return vars(x)", "def f(x):\n    return input()"):
        assert (sandbox.check_code_safety(code) or "").startswith("Blocked"), code


def test_reaching_a_module_through_another_one_gets_nowhere_at_run_time():
    """collections._sys.modules["os"] is ordinary attribute access, so the static check lets it through.
    What it reaches is refused when it is used."""
    code = ("import collections, random\ndef f():\n    found = collections._sys.modules['os']\n"
            "    try:\n        found.system('echo x')\n    except Exception as exc:\n        first = str(exc)\n"
            "    try:\n        random._os.remove('solution.py')\n    except Exception as exc:\n"
            "        return [first, str(exc)]")
    assert sandbox.check_code_safety(code) is None
    report = sandbox.execute(code, "f", [[]])
    assert report["values"][0][0]["value"] == ["sandbox: 'os.system' is not allowed", "sandbox: 'os.remove' is not allowed"]


@windows
def test_the_windows_api_module_is_emptied_after_the_warm_up():
    """_winapi is loaded at start-up. Some of its functions (file mappings, named pipes, DuplicateHandle)
    raise no audit event, so they are removed; os.path keeps the three names it took from it."""
    outcome = attack('''
        api = sys.modules["_winapi"]
        left = [name for name in dir(api) if not name.startswith("__")]
        return [left, os.path.normcase("C:/Users/ABC"), os.path.abspath("x").endswith("x"),
                tries(mapping=lambda: api.CreateFileMapping(-1, 0, 4, 0, 4096, "ModelLabTest"),
                      pipe=lambda: api.CreateNamedPipe("x", 3, 0, 1, 0, 0, 0, 0),
                      duplicate=lambda: api.DuplicateHandle(-1, -1, -1, 0, False, 2),
                      process=lambda: api.OpenProcess(1, False, os.getppid()))]
    ''')
    left, lowered, absolute, attempts = outcome.value
    assert left == [] and lowered == "c:\\users\\abc" and absolute is True and refused(attempts)


# ---------------------------------------------------------------- files

def test_files_outside_the_sandbox_folder_cannot_be_read(secret):
    outcome = attack('''
        import io, _io, linecache
        path, home = args
        return tries(
            open=lambda: open(path).read(),
            open_binary=lambda: open(path, "rb").read(),
            os_open=lambda: os.read(os.open(path, os.O_RDONLY), 100),
            fileio=lambda: io.FileIO(path).read(),
            open_code=lambda: _io.open_code(path).read(),
            bytes_name=lambda: open(path.encode("utf-8")).read(),
            forward_slashes=lambda: open(path.replace(chr(92), "/")).read(),
            relative=lambda: open(os.path.relpath(path)).read(),
            long_prefix=lambda: open(chr(92) * 2 + "?" + chr(92) + path).read(),
            listdir=lambda: os.listdir(os.path.dirname(path)),
            scandir=lambda: [entry.name for entry in os.scandir(os.path.dirname(path))],
            home=lambda: os.listdir(home),
            drive_root=lambda: os.listdir(os.path.splitdrive(path)[0] + os.sep),
        ), linecache.getline(path, 1)
    ''', str(secret), str(Path.home()))
    assert outcome.error is None, outcome
    results, quiet_read = outcome.value
    assert refused(results) and quiet_read == ""
    assert "NOT-FOR-THE-MODEL" not in repr(outcome.report)


def test_a_file_name_that_is_not_plain_text_is_refused(secret):
    """An object that gives one name when it is checked and another when it is opened, a file
    descriptor, a name with a NUL in it: none of them is taken on trust."""
    outcome = attack('''
        import io
        path = args[0]

        class Shifty:
            """Answers with the allowed name on the calls listed in `honest`, with the secret otherwise."""
            def __init__(self, *honest):
                self.honest, self.calls = honest, 0
            def __fspath__(self):
                self.calls += 1
                return "solution.py" if self.calls in self.honest else path

        class Text(str):
            pass

        return tries(
            object_secret_first=lambda: io.FileIO(Shifty(2)).read(),
            object_secret_second=lambda: io.FileIO(Shifty(1)).read(),
            open_secret_first=lambda: open(Shifty(2)).read(),
            str_subclass=lambda: io.FileIO(Text(path)).read(),
            descriptor=lambda: open(0).read(),
            with_nul=lambda: open(path + chr(0) + "x").read(),
        ), open(Shifty(1)).read()
    ''', str(secret))
    assert outcome.error is None, outcome
    results, honest_first = outcome.value
    for name, what in results.items():
        assert not what.startswith("DONE"), (name, what)
    assert all("sandbox:" in results[name] for name in results if name != "with_nul"), results
    # open() asks the object once, and checks and opens that same answer: here, its own solution file.
    assert honest_first.lstrip().startswith("import os, sys")
    assert "NOT-FOR-THE-MODEL" not in repr(outcome.report)


def test_nothing_can_be_written_deleted_or_renamed(secret, tmp_path):
    target = tmp_path / "written.txt"
    outcome = attack('''
        path, target = args
        return tries(
            write=lambda: open(target, "w").write("x"),
            append=lambda: open(path, "a").write("x"),
            update=lambda: open(path, "r+").write("x"),
            exclusive=lambda: open(target, "x").write("x"),
            os_create=lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
            os_append=lambda: os.open(path, os.O_RDWR),
            remove=lambda: os.remove(path),
            unlink=lambda: os.unlink(path),
            rename=lambda: os.rename(path, target),
            replace=lambda: os.replace(path, target),
            truncate=lambda: os.truncate(path, 0),
            mkdir=lambda: os.mkdir(target + ".d"),
            makedirs=lambda: os.makedirs(target + ".d/e"),
            rmdir=lambda: os.rmdir(os.path.dirname(path)),
            chmod=lambda: os.chmod(path, 0),
            utime=lambda: os.utime(path, (0, 0)),
            link=lambda: os.link(path, target),
            symlink=lambda: os.symlink(path, target),
            chdir=lambda: os.chdir(os.path.dirname(path)),
            inside_sandbox=lambda: open("notes.txt", "w").write("x"),
            overwrite_solution=lambda: open("solution.py", "w").write("x"),
        )
    ''', str(secret), str(target))
    assert outcome.error is None and refused(outcome.value), outcome
    assert secret.read_text(encoding="utf-8").startswith("OPENROUTER_API_KEY=")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["secret.env"]


@windows
def test_windows_itself_refuses_writes_when_the_hook_is_off(secret, tmp_path):
    """The process runs at low integrity. With the audit hook switched off, writing to or deleting a
    file of the user still fails, with Windows' own "access denied"."""
    target = tmp_path / "written.txt"
    outcome = attack('''
        path, target, home = args
        return tries(
            write=lambda: open(target, "w").write("x"),
            append=lambda: open(path, "a").write("x"),
            remove=lambda: os.remove(path),
            rename=lambda: os.rename(path, target),
            mkdir=lambda: os.mkdir(target + ".d"),
            in_home=lambda: open(os.path.join(home, "modellab-sandbox-test.txt"), "w").write("x"),
            inside_sandbox=lambda: open("notes.txt", "w").write("x"),
        )
    ''', str(secret), str(target), str(Path.home()), lock_down=False)
    assert outcome.error is None, outcome
    for name, what in outcome.value.items():
        assert what.startswith("PermissionError") and "sandbox:" not in what, (name, what)
    assert secret.exists() and sorted(p.name for p in tmp_path.iterdir()) == ["secret.env"]
    assert not (Path.home() / "modellab-sandbox-test.txt").exists()


@windows
def test_the_process_runs_at_low_integrity_inside_a_job():
    outcome = attack('''
        import ctypes
        from ctypes import wintypes
        kernel32, advapi32 = ctypes.WinDLL("kernel32"), ctypes.WinDLL("advapi32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
        advapi32.GetTokenInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                 ctypes.POINTER(wintypes.DWORD))
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
        kernel32.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))
        token = wintypes.HANDLE()
        advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token))
        buffer, size = ctypes.create_string_buffer(256), wintypes.DWORD()
        advapi32.GetTokenInformation(token, 25, buffer, 256, ctypes.byref(size))
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        level = advapi32.GetSidSubAuthority(sid, advapi32.GetSidSubAuthorityCount(sid)[0] - 1)[0]
        in_job = wintypes.BOOL()
        kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(in_job))

        class NetRate(ctypes.Structure):
            _fields_ = [("MaxBandwidth", ctypes.c_uint64), ("ControlFlags", wintypes.DWORD), ("DscpTag", ctypes.c_ubyte)]

        rate = NetRate()
        kernel32.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                       ctypes.POINTER(wintypes.DWORD))
        kernel32.QueryInformationJobObject(None, 32, ctypes.byref(rate), ctypes.sizeof(rate), None)
        return [level, bool(in_job.value), rate.MaxBandwidth, rate.ControlFlags & 3]
    ''', lock_down=False)
    # 0x1000 is low integrity (an ordinary process has 0x2000, medium). The job lets it send one byte per second.
    assert outcome.value == [0x1000, True, sandbox.NETWORK_BYTES_PER_SECOND, 3], outcome


def test_the_environment_carries_no_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-FROM-THE-SERVER")
    monkeypatch.setenv("HF_TOKEN", "hf_FROM-THE-SERVER")
    outcome = attack("return dict(os.environ)")
    assert "FROM-THE-SERVER" not in repr(outcome.report)
    assert {key.upper() for key in outcome.value} <= {"SYSTEMROOT"}


# ---------------------------------------------------------------- network

def test_no_socket_can_be_opened():
    with listening() as (port, seen):
        outcome = attack('''
            import importlib
            port = args[0]

            def connect(module):
                found = importlib.import_module(module)
                found.socket().connect(("127.0.0.1", port))

            def fetch():
                import urllib.request
                return urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=2).read()

            return tries(
                socket=lambda: connect("socket"),
                low_level=lambda: connect("_socket"),
                urllib=fetch,
                http=lambda: importlib.import_module("http.client"),
                ssl=lambda: importlib.import_module("ssl"),
                asyncio=lambda: importlib.import_module("asyncio"),
                ftp=lambda: importlib.import_module("ftplib"),
                already_loaded=lambda: sys.modules["socket"],
            )
        ''', port)
    assert outcome.error is None, outcome
    assert outcome.value.pop("already_loaded").startswith("KeyError")      # nothing loaded it, either
    assert refused(outcome.value) and seen == []


# ---------------------------------------------------------------- other processes

def test_no_other_program_can_be_started(tmp_path):
    marker = tmp_path / "started.txt"
    outcome = attack('''
        import importlib
        marker, python = args
        command = 'echo x > "%s"' % marker
        return tries(
            system=lambda: os.system(command),
            popen=lambda: os.popen(command).read(),
            startfile=lambda: os.startfile(python),
            spawn=lambda: os.spawnv(os.P_WAIT, python, [python, "-c", "pass"]),
            execv=lambda: os.execv(python, [python, "-c", "pass"]),
            subprocess=lambda: importlib.import_module("subprocess").run([python, "-c", "pass"]),
            multiprocessing=lambda: importlib.import_module("multiprocessing"),
            winapi=lambda: sys.modules["_winapi"].CreateProcess(python, python + " -c pass", None, None, 0, 0, None, None, None),
            kill_the_server=lambda: os.kill(os.getppid(), 9),
            ctypes=lambda: importlib.import_module("ctypes").CDLL(None),
            thread=lambda: importlib.import_module("_thread").start_new_thread(print, ()),
            threading=lambda: importlib.import_module("threading").Thread(target=print).start(),
        )
    ''', str(marker), sys.executable)
    assert outcome.error is None, outcome
    results = outcome.value
    if sys.platform != "win32":
        for name in ("startfile", "winapi"):
            results.pop(name)
    assert refused(results) and not marker.exists()


@windows
def test_windows_itself_refuses_a_child_process_when_the_hook_is_off(tmp_path):
    """The job allows one process. With the audit hook off, starting another one still fails."""
    outcome = attack('''
        import subprocess
        python = args[0]
        return tries(
            popen=lambda: subprocess.Popen([python, "-c", "pass"]).wait(),
            spawn=lambda: os.spawnv(os.P_WAIT, python, [python, "-c", "pass"]),
        )
    ''', sandbox_python(), lock_down=False)
    assert outcome.error is None, outcome
    for name, what in outcome.value.items():
        assert what.startswith(("OSError", "PermissionError")) and not what.startswith("DONE"), (name, what)


def sandbox_python():
    return getattr(sys, "_base_executable", None) or sys.executable


@windows
def test_ending_the_job_ends_the_process():
    with_env = {k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}
    process, job = sandbox._start([sandbox_python(), "-I", "-S", "-c", "import time; time.sleep(60)"],
                                  Path.cwd(), with_env, 30)
    try:
        time.sleep(0.5)
        assert process.poll() is None
    finally:
        sandbox._stop(process, job)
        process.stdout.close()
        process.stderr.close()
    assert process.returncode is not None


# ---------------------------------------------------------------- time, memory, output

def test_an_endless_loop_is_stopped_at_the_time_limit():
    for body in ("while True:\n    pass", "import time\ntime.sleep(60)"):
        started = time.monotonic()
        outcome = attack(body, timeout=1.0)
        assert outcome.stopped == "time" and outcome.value is None, outcome
        assert time.monotonic() - started < 6
    passed, total, notes = sandbox.run_code_tests("def spin(x):\n    while x:\n        pass\n    return 1", "spin",
                                                  [{"args": [0], "expected": 1}, {"args": [1], "expected": 1}], timeout=1.0)
    assert (passed, total) == (1, 2) and "timed out after 1s" in notes and "case 2: did not finish" in notes


@windows
def test_the_job_stops_a_busy_process_even_if_nobody_is_watching():
    """The CPU-time limit belongs to the job, so it does not depend on this process checking the clock."""
    with_env = {k: os.environ[k] for k in ("SYSTEMROOT",) if k in os.environ}
    process, job = sandbox._start([sandbox_python(), "-I", "-S", "-c", "while True: pass"], Path.cwd(), with_env, 1.0)
    try:
        assert process.wait(timeout=20) == sandbox._OUT_OF_CPU_TIME
    finally:
        sandbox._stop(process, job)
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize("lock_down", [True, False])
def test_memory_is_capped(lock_down):
    """Asking for far more than the limit fails inside the sandbox, with or without the audit hook."""
    outcome = attack('''
        def grow():
            blocks = []
            while True:
                blocks.append(bytearray(16 * 1024 * 1024))

        results = tries(one_gigabyte=lambda: len(bytearray(1024 ** 3)), huge_list=lambda: len([0] * 10 ** 10), grow=grow)
        results["still_working"] = sum(range(1000))
        return results
    ''', lock_down=lock_down)
    assert outcome.error is None and outcome.stopped is None, outcome
    assert outcome.value.pop("still_working") == 499500
    assert all(what.startswith("MemoryError") for what in outcome.value.values()), outcome.value


def test_memory_errors_show_up_as_failed_cases():
    passed, total, notes = sandbox.run_code_tests(
        "def f(n):\n    return len(bytearray(n))", "f",
        [{"args": [10], "expected": 10}, {"args": [2 * 1024 ** 3], "expected": 2 * 1024 ** 3}])
    assert (passed, total) == (1, 2) and "MemoryError" in notes


def test_a_flood_of_output_is_stopped():
    started = time.monotonic()
    outcome = attack('''
        line = "x" * 100000
        while True:
            print(line)
    ''', timeout=20.0)
    assert outcome.stopped == "output" and time.monotonic() - started < 15, outcome
    assert len(repr(outcome.report)) < 3 * sandbox.MAX_OUTPUT_BYTES
    notes = sandbox.run_code_tests("def f():\n    while True:\n        print('y' * 100000)", "f",
                                   [{"args": [], "expected": 1}], timeout=20.0)[2]
    assert "printed more than 1 MB of output" in notes


def test_code_is_not_run_when_the_limits_cannot_be_set(monkeypatch):
    """No limits, no run: the test becomes an error (not counted), never an unprotected run."""
    def broken(*args):
        raise sandbox.SandboxError("CreateJobObject failed (Windows error 5)")

    monkeypatch.setattr(sandbox, "_start", broken)
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_code_tests("def add(a, b): return a + b", "add", [{"args": [1, 2], "expected": 3}])


def test_a_sandbox_that_cannot_be_set_up_makes_the_test_an_error_not_a_zero(monkeypatch):
    """Through the benchmark job: the coding tests end as errors, which are left out of the score."""
    from fastapi.testclient import TestClient

    from app.main import app

    def broken(*args):
        raise sandbox.SandboxError("CreateJobObject failed (Windows error 5)")

    monkeypatch.setattr(sandbox, "_start", broken)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        started = client.post("/api/benchmark/run", json={"target": {"provider": "openrouter", "model": "mock/smart"},
                                                          "categories": ["coding"], "concurrency": 4}).json()
        client.get(f"/api/benchmark/{started['run_id']}/events")          # follow the job to its end
        run = client.get(f"/api/runs/{started['run_id']}").json()["run"]
    assert [r["status"] for r in run["results"]] == ["error"] * 5 and run["score"] is None
    assert all("SandboxError: CreateJobObject failed" in r["error"] for r in run["results"])


# ---------------------------------------------------------------- forging results

def test_printing_results_does_not_pass_a_test():
    """The expected values are compared outside the sandbox, so its own "ok" means nothing."""
    cases = [{"args": [2, 3], "expected": 5}, {"args": [10, 1], "expected": 11}]
    mark = sandbox._MARK
    forged = f'''
def add(a, b):
    print('{mark}{{"i": 0, "ok": true}}')
    print('{mark}{{"i": 1, "ok": true, "value": true}}')
    print('{mark}{{"ready": true}}')
    return 0
'''
    passed, total, notes = sandbox.run_code_tests(forged, "add", cases)
    assert (passed, total) == (0, 2), notes
    # Reporting several values for one case, hoping one is right, fails that case.
    guessing = f'''
def add(a, b):
    for guess in range(20):
        print('{mark}' + '{{"i": 0, "value": %d}}' % guess)
        print('{mark}' + '{{"i": 1, "value": %d}}' % guess)
    return a + b
'''
    passed, total, notes = sandbox.run_code_tests(guessing, "add", cases)
    assert (passed, total) == (0, 2) and "more than one result" in notes
    assert sandbox.run_code_tests("def add(a, b):\n    return a + b", "add", cases)[0] == 2


def test_whatever_the_code_prints_cannot_crash_the_grader():
    """Found in review: a line of 100,000 opening brackets after the marker made json.loads raise
    RecursionError in the server process. The benchmark job turned that into an "error" result, which
    is left out of the score: a way for a model to drop its coding tests instead of failing them."""
    mark = sandbox._MARK
    cases = [{"args": [1], "expected": 1}, {"args": [2], "expected": 2}]
    for junk in ('"[" * 100000', '"{" * 5000', '"[" * 900 + "1" + "]" * 900', "chr(0) * 50", '"null"', '"12"',
                 "'{\"i\": 0'", "'{\"i\": \"0\", \"value\": 1}'", "'{\"i\": true, \"value\": 1}'",
                 "'{\"i\": 99999, \"value\": 1}'", "'{\"i\": -1, \"value\": 1}'"):
        code = f"def f(x):\n    print('{mark}' + {junk}, flush=True)\n    return x"
        passed, total, notes = sandbox.run_code_tests(code, "f", cases)
        assert (passed, total) == (2, 2), (junk, notes)
    # A deeply nested value as the result itself is a wrong answer, not a crash.
    nested = f"def f(x):\n    print('{mark}' + '{{\"i\": 0, \"value\": ' + '[' * 900 + ']' * 900 + '}}', flush=True)\n    return x"
    passed, total, notes = sandbox.run_code_tests(nested, "f", cases)
    assert (passed, total) == (1, 2) and "more than one result" in notes
    grade = graders.grade_code(f"```python\ndef f(x):\n    print('{mark}' + '[' * 100000)\n    return x + 1\n```", "f", cases)
    assert grade.score == 0 and grade.status == "fail"                 # a failed test, counted


def test_results_are_compared_the_way_a_unit_test_would():
    """assertEqual semantics, kept on purpose: 8.0 passes for 8 and True for 1, as they would in a
    unit test. A tuple equals a list because both travel as JSON."""
    def passes(returned, expected):
        return sandbox.run_code_tests(f"def f():\n    return {returned}", "f", [{"args": [], "expected": expected}])[0] == 1

    assert passes("8.0", 8) and passes("(1, 2)", [1, 2]) and passes("True", 1) and passes("[1, 2.0]", [1, 2])
    assert not passes("9", 8) and not passes("'8'", 8) and not passes("[2, 1]", [1, 2]) and not passes("None", 0)
    assert not passes("float('nan')", 0) and not passes("{1, 2}", [1, 2])


def test_the_expected_values_are_not_inside_the_sandbox():
    cases = [{"args": [1], "expected": "SENTINEL-EXPECTED-VALUE"}]
    spy = '''
import json
def f(x):
    seen = []
    for name in ("job.json", "cases.json", "runner.py", "solution.py"):
        try:
            with open(name, encoding="utf-8") as handle:
                seen.append(handle.read())
        except OSError as exc:
            seen.append(str(exc))
    main = sys.modules["__main__"]
    return "\\n".join(seen) + repr(sorted(vars(main)))
import sys
'''
    report = sandbox.execute(spy, "f", [case["args"] for case in cases])
    assert "SENTINEL-EXPECTED-VALUE" not in repr(report) and '"func": "f"' in report["values"][0][0]["value"]
    assert sandbox.run_code_tests("import sys\n" + spy.replace("import sys\n", ""), "f", cases)[0] == 0


# ---------------------------------------------------------------- tampering with the hook

def test_replacing_builtins_and_library_functions_does_not_open_the_door(secret):
    """The hook keeps what it needs in its own arguments and looks nothing up by name when it runs."""
    outcome = attack('''
        import builtins, io
        path = args[0]
        folder, here = os.path.dirname(path), os.path.join(os.getcwd(), "solution.py")
        kept = dict(vars(builtins))
        attempts = dict(open=lambda: kept["open"](path).read(), fileio=lambda: io.FileIO(path).read(),
                        listdir=lambda: os.listdir(folder), socket=lambda: kept["__import__"]("socket"),
                        system=lambda: os.system("echo x"))
        quiet = kept["type"]("Quiet", (Exception,), {})
        builtins.isinstance = lambda *a: True
        builtins.type = lambda *a: kept["str"]
        builtins.len = lambda *a: 0
        builtins.str = builtins.int = builtins.bytes = kept["object"]
        builtins.PermissionError = builtins.ImportError = quiet
        if sys.platform == "win32":
            import nt
            nt._getfullpathname = lambda p: here
        os.path.abspath = os.path.realpath = os.path.normpath = lambda p: here
        os.fspath = lambda p: here
        os.getcwd = lambda: folder
        sys.pycache_prefix = None
        raw = {}
        for name, action in attempts.items():
            try:
                raw[name] = ("DONE", action())
            except BaseException as exc:
                raw[name] = ("ERROR", exc)
        for name, value in kept.items():                 # put the builtins back before describing anything
            setattr(builtins, name, value)
        return {name: "DONE: " + repr(value) if kind == "DONE" else type(value).__name__ + ": " + str(value)
                for name, (kind, value) in raw.items()}
    ''', str(secret))
    assert outcome.error is None and refused(outcome.value), outcome


def test_the_hooks_own_frame_gives_nothing_to_change(secret):
    """A refused call raises from inside the hook, so its frame can be reached through the traceback.
    Everything in it is a tuple, a frozenset or a string: readable, not changeable."""
    outcome = attack('''
        path = args[0]
        try:
            open(path)
        except PermissionError as exc:
            frame = exc.__traceback__.tb_next.tb_frame
        seen = frame.f_locals
        kinds = sorted({type(value).__name__ for value in seen.values()})
        attempts = tries(
            add_root=lambda: seen["_library"].append(os.path.dirname(path)),
            swap_folder=lambda: seen.__setitem__("_folder", os.path.dirname(path)) or frame.f_locals["_folder"],
            widen=lambda: seen["_harmless"].add("open"),
            allow_more=lambda: seen["_allowed"].add("socket"),
            recode=lambda: setattr(sys._getframe(0).f_back.f_code, "co_code", b""),
        )
        after = tries(open=lambda: open(path).read(), system=lambda: os.system("echo x"))
        return [frame.f_code.co_name, kinds, attempts, after]
    ''', str(secret))
    assert outcome.error is None, outcome
    name, kinds, attempts, after = outcome.value
    assert name == "hook" and refused(after)
    assert set(kinds) <= {"str", "tuple", "frozenset", "int", "NoneType", "type", "builtin_function_or_method"}, kinds
    assert all(attempts[name].startswith("AttributeError") for name in ("add_root", "widen", "allow_more")), attempts


def test_hand_made_bytecode_and_other_interpreter_tricks_are_refused():
    outcome = attack('''
        import importlib, marshal, types

        def f():
            return 1

        def g():
            return 2

        return tries(
            marshal_loads=lambda: marshal.loads(marshal.dumps(f.__code__)),
            code_new=lambda: types.CodeType(0, 0, 0, 0, 0, 0, b"", (), (), (), "", "", "", 0, b"", b""),
            code_replace=lambda: f.__code__.replace(co_consts=(None, 3)),
            set_code=lambda: setattr(f, "__code__", g.__code__),
            function_new=lambda: types.FunctionType(g.__code__, {})(),
            settrace=lambda: sys.settrace(lambda *a: None),
            setprofile=lambda: sys.setprofile(lambda *a: None),
            monitoring=lambda: sys.monitoring.register_callback(1, 1, print),
            gc=lambda: importlib.import_module("gc").get_objects(),
            pickle=lambda: importlib.import_module("pickle"),
            frames=lambda: sys._current_frames(),
            registry=lambda: sys.modules["winreg"].CreateKey(sys.modules["winreg"].HKEY_CURRENT_USER, "ModelLabTest"),
            breakpoint=lambda: breakpoint(),
            input=lambda: input(),
        )
    ''')
    assert outcome.error is None, outcome
    results = outcome.value
    if sys.platform != "win32":
        results.pop("registry")
    if not hasattr(sys, "monitoring"):
        results.pop("monitoring")
    assert refused(results)


def test_another_audit_hook_cannot_be_added_to_listen_in():
    """sys.addaudithook asks the hooks already there. Ours refuses, and Python then returns quietly
    without adding the new one (documented behaviour), so a second hook never sees an event."""
    outcome = attack('''
        seen = []
        added = sys.addaudithook(lambda event, arguments: seen.append(event))
        id(seen)                      # raises the "builtins.id" event
        try:
            open("solution.py").close()
        except Exception:
            pass
        return [added, seen]
    ''')
    assert outcome.value == [None, []], outcome


def test_an_unknown_audit_event_is_refused_by_default():
    outcome = attack('return tries(custom=lambda: sys.audit("something.new", 1), known=lambda: sys.audit("builtins.id", 1))')
    assert "sandbox: 'something.new' is not allowed" in outcome.value["custom"]
    assert outcome.value["known"] == "DONE: None"


# ---------------------------------------------------------------- what it does NOT stop
#
# These tests pass when the limit is still there. They are the evidence for the limits listed in
# README.md; if one starts failing, the sandbox got stronger and the README needs updating.

def test_limit_modules_python_loads_at_start_up_can_be_reached():
    """os, nt, _winapi, winreg, marshal, _signal and _thread are loaded before any hook exists, and the
    model's code can reach them through sys.modules. What they offer that matters raises an audit
    event and is refused; a function in them that raises none is not covered."""
    outcome = attack('''
        present = [name for name in ("os", "nt", "_winapi", "winreg", "marshal", "_signal", "_thread", "_imp")
                   if name in sys.modules]
        quiet = [os.getpid() > 0, os.cpu_count() > 0, bool(os.getcwd()), os.path.isdir(args[0])]
        return [present, quiet]
    ''', str(Path.home()))
    present, quiet = outcome.value
    expected = {"os", "marshal", "_signal", "_thread", "_imp"} | ({"nt", "_winapi", "winreg"} if sys.platform == "win32" else set())
    assert set(present) == expected and quiet == [True, True, True, True]

def test_limit_it_can_tell_that_a_file_exists_and_how_big_it_is(secret):
    """os.stat raises no audit event, so the hook never sees it. The content stays out of reach."""
    outcome = attack("return [os.path.exists(args[0]), os.path.getsize(args[0]), os.path.exists(args[0] + '.missing')]",
                     str(secret))
    assert outcome.value == [True, secret.stat().st_size, False]


def test_limit_it_can_read_its_own_folder_and_pythons_library():
    outcome = attack('''
        import json
        with open(os.path.join(os.path.dirname(json.__file__), "decoder.py"), encoding="utf-8") as handle:
            library = handle.read(40)
        with open("solution.py", encoding="utf-8") as handle:
            own = handle.read(12)
        return [library[:3], own.strip(), sorted(os.listdir("."))]
    ''')
    assert outcome.value == ['"""', "import os,", ["job.json", "runner.py", "solution.py"]]


def test_only_allowed_library_modules_can_be_loaded_whatever_the_route():
    """importlib loads a module without the "import" event of an import statement, so the hook also
    guards the library's source files by module name. Extension modules announce themselves."""
    outcome = attack('''
        import importlib
        names = ("threading", "socket", "shutil", "ctypes", "subprocess", "tempfile", "pathlib", "pickle", "glob",
                 "urllib.request", "http.client", "_socket", "select", "_ctypes", "_overlapped", "_multiprocessing")
        return tries(**{name.replace(".", "_"): (lambda name=name: importlib.import_module(name)) for name in names})
    ''')
    assert outcome.error is None and refused(outcome.value), outcome


def test_builtin_and_frozen_modules_cannot_be_loaded_after_lock_down():
    """Modules compiled into the interpreter (gc, mmap, msvcrt, _xxsubinterpreters ...) and frozen ones
    (runpy) load without any audit event, so the hook cannot keep them out. The functions that load
    them are removed from the import machinery instead, and cannot be got back."""
    outcome = attack('''
        import importlib
        names = ("gc", "mmap", "msvcrt", "atexit", "faulthandler", "_xxsubinterpreters", "_interpreters", "zlib",
                 "binascii", "_tracemalloc", "runpy", "site")
        results = tries(**{name: (lambda name=name: importlib.import_module(name)) for name in names})
        imp = sys.modules["_imp"]
        machinery = sys.modules["_frozen_importlib"]
        results.update(tries(
            still_there=lambda: [name for name in ("create_builtin", "exec_builtin", "create_dynamic",
                                                   "get_frozen_object", "init_frozen") if hasattr(imp, name)],
            fresh_copy=lambda: machinery.BuiltinImporter.create_module(imp.__spec__).create_builtin,
            reload=lambda: importlib.reload(imp).create_builtin,
            reimport=lambda: (sys.modules.pop("_imp"), __import__("_imp").create_builtin),
        ))
        return results
    ''')
    assert outcome.error is None, outcome
    assert outcome.value.pop("still_there") == "DONE: []"
    for name, what in outcome.value.items():
        assert not what.startswith("DONE"), (name, what)
        assert what.startswith(("AttributeError", "ImportError", "ModuleNotFoundError", "KeyError")), (name, what)


def test_a_second_interpreter_cannot_be_started():
    """Audit hooks belong to one interpreter. A second one would run with none, so it must not exist:
    its module cannot be loaded any more, and creating an interpreter is an event the hook refuses."""
    outcome = attack('''
        import importlib
        def second():
            interpreters = importlib.import_module("_xxsubinterpreters")
            return interpreters.run_string(interpreters.create(), "import os; os.system('echo x')")
        return tries(second=second, event=lambda: sys.audit("cpython.PyInterpreterState_New"))
    ''')
    assert outcome.error is None, outcome
    assert not outcome.value["second"].startswith("DONE") and refused({"event": outcome.value["event"]})


def test_bytes_cannot_be_turned_into_a_code_object():
    """_imp.get_frozen_object(name, data) unmarshals whatever it is given. It is removed after the
    warm-up; where it exists, it raises the marshal.loads event, which the hook refuses."""
    import marshal
    data = marshal.dumps(compile("x = 41 + 1", "<made>", "exec")).hex()
    outcome = attack('''
        def frozen():
            code = sys.modules["_imp"].get_frozen_object("anything", bytes.fromhex(args[0]))
            scope = {}
            exec(code, scope)
            return scope["x"]
        return tries(frozen=frozen)
    ''', data)
    assert outcome.value["frozen"].startswith("AttributeError"), outcome


LATE = '''
    import codecs
    path, mark = args

    class Late:
        """Tries to read a file when it is destroyed, using only what it kept hold of."""
        def __init__(self, tag, _open=os.open, _read=os.read, _write=os.write):
            self.tag, self.open, self.read, self.write = tag, _open, _read, _write
            self.head = (mark + '{"i": 0, "value": "LATE ').encode()
            self.end = b'"}' + bytes([10])
        def __call__(self, *a):
            return None
        def __del__(self):
            try:
                self.write(1, self.head + self.tag + b" READ " + self.read(self.open(path, 0), 60).strip() + self.end)
            except BaseException:
                self.write(1, self.head + self.tag + b" refused" + self.end)

    codecs.register(Late(b"codec search function"))
    codecs.register_error("late", Late(b"codec error handler"))
    sys.modules["__main__"].keep = Late(b"module global")
    cycle = Late(b"reference cycle")
    cycle.me = cycle
'''


def test_nothing_runs_while_the_interpreter_shuts_down(secret):
    """Python clears its audit hooks part-way through shutting down and destroys what is left
    afterwards. A __del__ method left on a codec search function, a module global or a reference
    cycle then runs with no hook (checked in plain Python: it reads the file). The sandbox process
    therefore never shuts down: it ends hard, or spins until it is ended."""
    for ending in ('return "armed"',                                   # the runner finishes normally
                   'os._exit = lambda code: None\nreturn "armed"',    # its quick way out is taken away
                   'os._exit = sys.exit\nreturn "armed"',             # ... or made to raise SystemExit
                   'sys.exit(0)',                                      # the model's code asks to exit
                   'sys.modules["__main__"].emit = sys.exit\nreturn "armed"'):
        outcome = attack(LATE + textwrap.indent(ending, "    "), str(secret), sandbox._MARK, timeout=3.0)
        late = [item["value"] for item in outcome.report["values"].get(0, []) if str(item.get("value")).startswith("LATE")]
        assert late == [], (ending, late)
        assert "NOT-FOR-THE-MODEL" not in repr(outcome.report), ending
    # The first ending is an ordinary run: its result arrives and nothing is left waiting.
    outcome = attack(LATE + '    return "armed"', str(secret), sandbox._MARK)
    assert outcome.value == "armed" and outcome.stopped is None


def test_the_runner_reports_done_and_is_ended_at_once():
    started = time.monotonic()
    for _ in range(5):
        report = sandbox.execute("def f(x):\n    return x", "f", [[1], [2]])
        assert report["stopped"] is None and len(report["values"]) == 2
    assert time.monotonic() - started < 5                             # nothing waits for a time limit
    # Code that takes the quick exit away makes the process spin; the parent ends it on "done".
    started = time.monotonic()
    report = sandbox.execute("import os\nos._exit = lambda code: None\ndef f(x):\n    return x", "f", [[1]])
    assert report["stopped"] is None and report["values"][0][0]["value"] == 1
    assert time.monotonic() - started < 3
    # A forged "done" only ends the run early, with the remaining cases unfinished.
    forged = f"def f(x):\n    print('{sandbox._MARK}' + '{{\"done\": true}}', flush=True)\n    import time\n    time.sleep(1)\n    return x"
    passed, total, notes = sandbox.run_code_tests(forged, "f", [{"args": [1], "expected": 1}, {"args": [2], "expected": 2}])
    assert passed == 0 and "did not finish" in notes or "crashed" in notes


def test_limit_without_the_hook_files_can_be_read(secret):
    """Reading is stopped by the audit hook only. Low integrity blocks writing, not reading."""
    outcome = attack("return open(args[0]).read()", str(secret), lock_down=False)
    assert outcome.value == secret.read_text(encoding="utf-8")


def test_limit_without_the_hook_this_machine_can_still_be_reached():
    """The job's network limit throttles traffic that leaves the machine (measured by hand against
    another host: see the HANDOFF report). It does not cover loopback, so without the audit hook a
    server listening on this machine, Model Lab included, can be reached. Blocking that needs
    administrator rights (a firewall rule) or a container."""
    with listening() as (port, seen):
        outcome = attack('''
            import socket
            connection = socket.create_connection(("127.0.0.1", args[0]), 3)
            connection.close()
            return "connected"
        ''', port, lock_down=False)
    assert outcome.value == "connected" and seen == [1]

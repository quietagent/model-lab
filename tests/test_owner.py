"""Two servers on one database: a running run belongs to the process that started it, and only a run
whose owner is gone is marked "interrupted"."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app import bench_tests, benchmark, db, process

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    db.init_db()
    return tmp_path


def new_run() -> int:
    run_id = db.create_run("openrouter", "some/model", None, ["math"], {}, 5)
    db.add_result(run_id, {**benchmark._blank_result(bench_tests.BY_ID["math-change"]), "status": "pass", "score": 1.0})
    return run_id


def set_owner(run_id, pid, started):
    with db.connect() as conn:
        conn.execute("UPDATE runs SET owner_pid = ?, owner_started = ? WHERE id = ?", (pid, started, run_id))


def dead_process() -> tuple[int, float]:
    """The ID and start time of a process that has ended."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    started = process.started(child.pid)
    child.wait()
    return child.pid, started


# ---------------------------------------------------------------- which processes are running

def test_process_describes_this_process_and_a_finished_one():
    alive, started = process.describe(os.getpid())
    assert alive and started is not None and abs(process.started() - started) < 0.01
    assert process.is_alive(os.getpid(), started)
    pid, was_started = dead_process()
    assert process.describe(pid) == (False, None) and not process.is_alive(pid, was_started)
    assert not process.is_alive(None, None) and not process.is_alive(0, None) and not process.is_alive(-5, None)


def test_a_reused_process_id_does_not_count_as_the_owner():
    """The system hands IDs out again: the same ID with another start time is another process."""
    started = process.started()
    assert process.is_alive(os.getpid(), started)
    assert not process.is_alive(os.getpid(), started - 3600)
    assert process.is_alive(os.getpid(), None)                      # no start time recorded: the ID decides


# ---------------------------------------------------------------- runs and their owner

def test_a_new_run_belongs_to_this_process_until_it_is_finalized(database):
    run_id = new_run()
    run = db.get_run(run_id)
    assert run["status"] == "running" and run["owner_pid"] == os.getpid()
    assert abs(run["owner_started"] - process.started()) < 0.01
    done = db.finalize_run(run_id, "completed")
    assert (done["status"], done["owner_pid"], done["owner_started"]) == ("completed", None, None)


def test_starting_up_leaves_a_run_alone_while_its_owner_is_running(database):
    mine = new_run()
    db.init_db()                                                    # "a second server starts"
    assert db.get_run(mine)["status"] == "running" and db.close_dead_runs() == []
    # Another live process (here: the test runner's parent, as a stand-in) owns a second run.
    theirs = new_run()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        set_owner(theirs, child.pid, process.started(child.pid))
        db.init_db()
        assert db.get_run(theirs)["status"] == "running"
        assert [(r["run_id"], r["pid"], r["model"]) for r in db.running_elsewhere()] == [(theirs, child.pid, "some/model")]
    finally:
        child.kill()
        child.wait()
    # Its owner is gone now: the next check says so, and keeps what the run had finished.
    assert db.close_dead_runs() == [theirs]
    after = db.get_run(theirs)
    assert (after["status"], after["score"], after["owner_pid"]) == ("interrupted", 100.0, None)
    assert db.running_elsewhere() == [] and db.get_run(mine)["status"] == "running"


def test_a_run_whose_owner_is_gone_is_interrupted_on_startup(database):
    dead, reused, unowned = new_run(), new_run(), new_run()
    set_owner(dead, *dead_process())
    set_owner(reused, os.getpid(), process.started() - 3600)       # same ID, but an earlier process
    set_owner(unowned, None, None)                                  # saved before owners were recorded
    db.init_db()
    assert [db.get_run(i)["status"] for i in (dead, reused, unowned)] == ["interrupted"] * 3


def test_database_from_before_owners_is_upgraded(database):
    with db.connect() as conn:
        conn.execute("ALTER TABLE runs DROP COLUMN owner_pid")
        conn.execute("ALTER TABLE runs DROP COLUMN owner_started")
        conn.execute("INSERT INTO runs (created_at, status, provider, model) VALUES (1, 'running', 'openrouter', 'm')")
    db.init_db()
    run = db.list_runs(include_unfinished=True, include_older=True)
    assert run == []                                                # nothing scored, so it is not listed ...
    with db.connect() as conn:                                      # ... and it no longer looks alive
        assert [row["status"] for row in conn.execute("SELECT status FROM runs")] == ["interrupted"]


# ---------------------------------------------------------------- two real server processes

def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def call(port, method, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def start_server(data_dir, port):
    env = {**os.environ, "MOCK": "1", "MOCK_FAST": "0", "MODEL_LAB_DATA_DIR": str(data_dir),
           "MODEL_LAB_NO_BROWSER": "1", "PYTHONUNBUFFERED": "1"}
    server = subprocess.Popen([sys.executable, "-m", "app", "--port", str(port)], cwd=ROOT, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            call(port, "GET", "/api/config")
            return server
        except (urllib.error.URLError, ConnectionError, OSError):
            if server.poll() is not None:
                break
            time.sleep(0.1)
    server.kill()
    raise AssertionError("the server did not start")


def stop(pid):
    """End a process by its ID. On Windows a virtual environment's python.exe is a launcher that starts
    the real interpreter as a child: the server's own process ID is the child's, not the one Popen has."""
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def wait_for(check, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("timed out waiting")


def test_a_second_server_does_not_interrupt_the_first_servers_job(tmp_path):
    """The real thing: two server processes on one database. Before owners were recorded, starting
    the second one marked the first one's running benchmark "interrupted"."""
    first_port, second_port = free_port(), free_port()
    first = start_server(tmp_path, first_port)
    second = None
    owners = set()
    try:
        slow = {"target": {"provider": "openrouter", "model": "mock/slow"}, "categories": ["math"], "concurrency": 1}
        run_id = call(first_port, "POST", "/api/benchmark/run", slow)["run_id"]
        wait_for(lambda: (call(first_port, "GET", "/api/benchmark/active")["job"] or {}).get("done", 0) >= 1)

        second = start_server(tmp_path, second_port)                # its start-up checks the same database
        seen = call(second_port, "GET", f"/api/runs/{run_id}")["run"]
        owner = seen["owner_pid"]
        owners.add(owner)
        assert seen["status"] == "running" and owner != os.getpid() and process.is_alive(owner, seen["owner_started"])
        active = call(second_port, "GET", "/api/benchmark/active")
        assert active["job"] is None                                # it is not the second server's job ...
        assert [(r["run_id"], r["pid"]) for r in active["elsewhere"]] == [(run_id, owner)]   # ... but it knows of it
        assert call(first_port, "GET", "/api/benchmark/active")["job"]["run_id"] == run_id

        done = wait_for(lambda: (lambda run: run if run["status"] != "running" else None)(
            call(second_port, "GET", f"/api/runs/{run_id}")["run"]), seconds=90)
        assert (done["status"], len(done["results"]), done["owner_pid"]) == ("completed", 5, None)

        # The first server dies in the middle of another run: the second one notices without a restart.
        run_id = call(first_port, "POST", "/api/benchmark/run", slow)["run_id"]
        wait_for(lambda: (call(first_port, "GET", "/api/benchmark/active")["job"] or {}).get("done", 0) >= 1)
        stop(owner)
        first.kill()
        first.wait()
        wait_for(lambda: not process.describe(owner)[0])
        assert call(second_port, "GET", "/api/benchmark/active")["elsewhere"] == []
        after = call(second_port, "GET", f"/api/runs/{run_id}")["run"]
        assert after["status"] == "interrupted" and 1 <= len(after["results"]) < 5
    finally:
        for server in (first, second):
            if server is not None and server.poll() is None:
                server.kill()
                server.wait()
        for pid in owners:
            stop(pid)

"""Several runs of one model: the Leaderboard shows their average with the range from the lowest run to
the highest, overall and for each category, and how many runs there are. Live, two runs of one model an
hour apart differed by 0.3 points overall and by 40 points in one category, so one run is not enough to
read a category. Also here: the count of tests that ran out of tokens in a run."""
import json
import shutil
import sqlite3
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.main import app

SMART = {"provider": "openrouter", "model": "mock/smart"}
WEAK = {"provider": "openrouter", "model": "mock/weak"}
LIMIT = 7001                                  # a token limit no other test uses: these runs are a group of their own
SETTINGS = f"max_tokens={LIMIT};judge=none"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def start(client, target=SMART, **body):
    """Run a benchmark to the end -> the saved run."""
    payload = {"target": target, "categories": ["math", "knowledge"], "concurrency": 4, "max_tokens": LIMIT, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    lines = client.get(f"/api/benchmark/{started.json()['run_id']}/events").text.splitlines()
    summary = [json.loads(line[5:]) for line in lines if line.startswith("data:")][-1]
    assert summary["type"] == "summary"
    return client.get(f"/api/runs/{summary['run']['id']}").json()["run"]


def change(run_id, test_id, status, score):
    """Give one test of a saved run another result, and work the run's score out again."""
    with db.connect() as conn:
        assert conn.execute("UPDATE results SET status = ?, score = ? WHERE run_id = ? AND test_id = ?",
                            (status, score, run_id, test_id)).rowcount == 1
    return db.finalize_run(run_id, "completed")


def listed(client, **params):
    return client.get("/api/runs", params={"level": "standard", "settings": SETTINGS, **params}).json()


# ---------------------------------------------------------------- the arithmetic

def run_row(run_id, score, categories, **extra):
    row = {"id": run_id, "created_at": float(run_id), "status": "completed", "provider": "openrouter", "model": "m/one",
           "level": "hard", "settings_key": "max_tokens=32768;judge=none", "fingerprint": "abc", "score": score,
           "category_scores": categories, "first_pass": {"score": score}, "n_out_of_tokens": 0, "avg_latency": 2.0,
           "avg_tps": 40.0, "total_cost": 0.01, "cost_estimated": False, "older_tests": False, **extra}
    return {**row, "group": db.group_key(row)}


def test_the_average_and_range_over_a_models_runs():
    runs = [run_row(1, 87.8, {"math": 100.0, "instructions": 40.0}, n_out_of_tokens=5),
            run_row(2, 88.1, {"math": 100.0, "instructions": 80.0}, n_out_of_tokens=5, total_cost=0.02),
            run_row(3, 91.0, {"math": 100.0, "instructions": 60.0}, n_out_of_tokens=2, first_pass={"score": 90.0})]
    (average,) = db.model_averages(runs)
    assert (average["runs"], average["run_ids"], average["model"], average["level"]) == (3, [1, 2, 3], "m/one", "hard")
    assert average["score"] == {"mean": 89.0, "min": 87.8, "max": 91.0, "runs": 3}
    assert average["grade"] == "B"                                          # the grade of the average, 89.0
    assert average["category_scores"] == {"math": {"mean": 100.0, "min": 100.0, "max": 100.0, "runs": 3},
                                          "instructions": {"mean": 60.0, "min": 40.0, "max": 80.0, "runs": 3}}
    assert average["first_pass"] == {"mean": 88.6, "min": 87.8, "max": 90.0, "runs": 3}
    assert average["out_of_tokens"] == {"mean": 4.0, "min": 2, "max": 5, "runs": 3}
    assert average["total_cost"] == pytest.approx(0.04) and average["mean_cost"] == pytest.approx(0.04 / 3)
    assert (average["avg_latency"], average["avg_tps"], average["newest"]) == (2.0, 40.0, 3.0)
    assert average["key"] == runs[0]["group"] == "openrouter|m/one|hard|max_tokens=32768;judge=none|abc"


def test_only_finished_runs_that_measured_the_same_thing_are_averaged():
    base = run_row(1, 80.0, {"math": 80.0})
    assert db.model_averages([base]) == []                                  # one run is not an average
    assert db.group_key({**base, "status": "cancelled"}) is None            # it ran only some of its tests
    assert db.group_key({**base, "status": "interrupted"}) is None
    assert db.group_key({**base, "fingerprint": None}) is None              # saved before fingerprints
    assert db.group_key({**base, "score": None}) is None
    others = [run_row(2, 60.0, {"math": 60.0}, model="m/two"),                               # another model
              run_row(3, 60.0, {"math": 60.0}, level="standard"),                           # another level
              run_row(4, 60.0, {"math": 60.0}, settings_key="max_tokens=16384;judge=none"),  # another token limit
              run_row(5, 60.0, {"math": 60.0}, settings_key="max_tokens=32768;judge=openrouter/j"),   # another judge
              run_row(6, 60.0, {"math": 60.0}, fingerprint="def"),                          # other tests or grader
              run_row(7, 60.0, {"math": 60.0}, provider="huggingface"),
              run_row(8, 60.0, {"math": 60.0}, status="cancelled")]
    assert len({run["group"] for run in others[:-1]} | {base["group"]}) == 7 and others[-1]["group"] is None
    assert db.model_averages([base] + others) == []
    (average,) = db.model_averages([base] + others + [run_row(9, 90.0, {"math": 90.0})])
    assert average["run_ids"] == [1, 9] and average["score"]["mean"] == 85.0


def test_a_category_that_one_run_could_not_score_is_averaged_over_the_runs_that_have_it():
    runs = [run_row(1, 75.0, {"math": 100.0, "coding": 50.0}), run_row(2, 100.0, {"math": 100.0}),
            run_row(3, 60.0, {"math": 80.0, "coding": 40.0})]
    (average,) = db.model_averages(runs)
    assert average["category_scores"]["coding"] == {"mean": 45.0, "min": 40.0, "max": 50.0, "runs": 2}
    assert average["category_scores"]["math"]["runs"] == 3
    # The overall figure is the average of the runs' own scores, and its range is from the lowest run to the highest.
    assert average["score"] == {"mean": 78.3, "min": 60.0, "max": 100.0, "runs": 3}
    # Nothing recorded for any run: no figure, rather than a made-up one.
    bare = [run_row(4, 50.0, {}, first_pass=None, total_cost=None, avg_latency=None, avg_tps=None),
            run_row(5, 70.0, {}, first_pass=None, total_cost=None, avg_latency=None, avg_tps=None)]
    (average,) = db.model_averages(bare)
    assert (average["first_pass"], average["total_cost"], average["mean_cost"], average["avg_latency"]) == (None,) * 4
    assert average["category_scores"] == {} and average["out_of_tokens"]["mean"] == 0.0


# ---------------------------------------------------------------- whole runs, through the API

def test_the_leaderboard_lists_the_average_of_a_models_runs_with_its_range(client):
    first, second, third = start(client), start(client), start(client)
    alone = start(client, target=WEAK)
    assert first["score"] == second["score"] == third["score"] == 100.0
    assert first["group"] == second["group"] == third["group"] != alone["group"]
    # Three runs of one model are never the same live. Here: one wrong answer in the second run, and in
    # the third one wrong answer and one test out of tokens.
    second = change(second["id"], "math-change", "fail", 0.0)
    change(third["id"], "math-change", "fail", 0.0)
    third = change(third["id"], "know-capital", "cut_off", 0.0)
    assert (first["score"], second["score"], third["score"]) == (100.0, 90.0, 80.0)
    assert (first["n_out_of_tokens"], second["n_out_of_tokens"], third["n_out_of_tokens"]) == (0, 0, 1)

    data = listed(client)
    by_key = {average["key"]: average for average in data["averages"]}
    average = by_key[first["group"]]
    assert alone["group"] not in by_key                                     # one run: nothing to average
    assert (average["runs"], average["run_ids"]) == (3, [first["id"], second["id"], third["id"]])
    assert average["score"] == {"mean": 90.0, "min": 80.0, "max": 100.0, "runs": 3}
    assert average["category_scores"] == {"math": {"mean": 86.7, "min": 80.0, "max": 100.0, "runs": 3},
                                          "knowledge": {"mean": 93.3, "min": 80.0, "max": 100.0, "runs": 3}}
    assert average["out_of_tokens"] == {"mean": 0.3, "min": 0, "max": 1, "runs": 3}
    assert average["grade"] == "A" and average["settings_key"] == SETTINGS and not average["older_tests"]
    assert average["first_pass"]["mean"] == 100.0                           # the first passes were before the changes
    # Every run is still listed, and says which average it belongs to.
    groups = {run["id"]: run["group"] for run in data["runs"]}
    assert groups[first["id"]] == groups[second["id"]] == groups[third["id"]] == average["key"]
    assert {run["id"]: run["n_out_of_tokens"] for run in data["runs"]}[third["id"]] == 1

    # A run with another token limit is not in the average, and neither is a cancelled run.
    other_limit = start(client, max_tokens=LIMIT + 1)
    assert other_limit["group"] != first["group"]
    with db.connect() as conn:
        conn.execute("UPDATE runs SET status = 'cancelled' WHERE id = ?", (second["id"],))
    again = {a["key"]: a for a in listed(client, unfinished="true")["averages"]}[first["group"]]
    assert (again["runs"], again["run_ids"], again["score"]["mean"]) == (2, [first["id"], third["id"]], 90.0)
    with db.connect() as conn:
        conn.execute("UPDATE runs SET status = 'completed' WHERE id = ?", (second["id"],))

    # "Best per model only" lists the best run; the average over all three is still sent, so the page
    # can say what the best run is the best of.
    best = listed(client, best="true")
    mine = [run for run in best["runs"] if run["model"] == "mock/smart"]
    assert [run["id"] for run in mine] == [first["id"]]
    assert {a["key"]: a for a in best["averages"]}[mine[0]["group"]]["runs"] == 3

    # The averages follow the group of settings that is asked for.
    assert first["group"] in {a["key"] for a in client.get("/api/runs", params={"level": "standard"}).json()["averages"]}
    elsewhere = client.get("/api/runs", params={"level": "standard", "settings": other_limit["settings_key"]}).json()
    assert first["group"] not in {a["key"] for a in elsewhere["averages"]}
    assert first["group"] not in {a["key"] for a in client.get("/api/runs", params={"level": "hard"}).json()["averages"]}

    # The export has one row per run, with its count of tests out of tokens.
    header, *rows = client.get("/api/runs/export", params={"format": "csv", "settings": SETTINGS}).text.lstrip(
        "﻿").splitlines()
    columns = header.split(",")
    counts = {int(row.split(",")[0]): row.split(",")[columns.index("out_of_tokens")] for row in rows}
    assert (counts[first["id"]], counts[third["id"]]) == ("0", "1")
    exported = client.get("/api/runs/export", params={"format": "json", "settings": SETTINGS}).json()
    assert {run["id"]: run["n_out_of_tokens"] for run in exported}[third["id"]] == 1


def test_out_of_tokens_counts_only_the_cut_off_replies_that_scored_zero(client):
    """A run's "cut off" count has two kinds in it: out of tokens (scored 0) and cut short by the provider
    (no score, can be retried). The Leaderboard's "Out of tokens" column is the first kind only."""
    outcomes = [("cut_off", 0.0), ("cut_off", None), ("cut_off", 0.0), ("pass", 1.0), ("empty", 0.0), ("fail", 0.0)]
    summary = db.summarize([{"category": "math", "status": status, "score": score, "latency": 1.0, "ttft": 0.1,
                             "tps": 10.0, "cost": None, "tokens_in": 1, "tokens_out": 1} for status, score in outcomes])
    assert (summary["n_cut_off"], summary["n_out_of_tokens"], summary["n_empty"], summary["n_fail"]) == (3, 2, 1, 1)

    run = start(client, target={"provider": "openrouter", "model": "mock/cutoff"}, categories=["math"])
    assert (run["n_cut_off"], run["n_out_of_tokens"]) == (5, 5)
    dropped = start(client, target={"provider": "openrouter", "model": "mock/dropper"}, categories=["math"])
    assert (dropped["n_cut_off"], dropped["n_out_of_tokens"]) == (5, 0)     # cut short by the provider: unscored


def test_a_database_from_before_out_of_tokens_was_counted_gets_the_counts_from_its_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    db.init_db()
    with db.connect() as conn:
        conn.execute("ALTER TABLE runs DROP COLUMN n_out_of_tokens")
        for run_id, n_cut_off in ((1, 3), (2, 0)):
            conn.execute("INSERT INTO runs (id, created_at, status, provider, model, score, n_cut_off) "
                         "VALUES (?, ?, 'completed', 'openrouter', 'm', 50, ?)", (run_id, run_id, n_cut_off))
        for status, score in (("cut_off", 0.0), ("cut_off", 0.0), ("cut_off", None), ("pass", 1.0), ("fail", 0.0)):
            conn.execute("INSERT INTO results (run_id, test_id, category, name, prompt, status, score) "
                         "VALUES (1, 't', 'math', 'n', 'p', ?, ?)", (status, score))
        conn.execute("INSERT INTO results (run_id, test_id, category, name, prompt, status, score) "
                     "VALUES (2, 't', 'math', 'n', 'p', 'pass', 1.0)")
    db.init_db()
    columns = sqlite3.connect(tmp_path / "modellab.mock.db")
    assert "n_out_of_tokens" in [row[1] for row in columns.execute("PRAGMA table_info(runs)")]
    columns.close()
    counts = {run["id"]: (run["n_cut_off"], run["n_out_of_tokens"]) for run in db.list_runs(include_older=True)}
    assert counts == {1: (3, 2), 2: (0, 0)}
    db.init_db()                                                            # a second start changes nothing
    assert {run["id"]: run["n_out_of_tokens"] for run in db.list_runs(include_older=True)} == {1: 2, 2: 0}


# ---------------------------------------------------------------- the page

def in_node(lines):
    result = subprocess.run(["node", "--input-type=module", "-"], input=chr(10).join(lines), capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_shows_one_row_for_a_model_with_several_runs_and_one_per_other_run():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    runs = [{"id": 5, "group": "A"}, {"id": 4, "group": "B"}, {"id": 3, "group": "A"}, {"id": 2, "group": None},
            {"id": 1, "group": "A"}, {"id": 0, "group": "C"}]
    averages = [{"key": "A", "runs": 3}, {"key": "C", "runs": 2}]
    shape = "(rows) => rows.map((row) => row.average ? [row.average.key, row.members.map((run) => run.id)] : row.run.id)"
    grouped, best, ranges = in_node([
        f"import {{ fmtRange, leaderboardRows }} from {json.dumps(module)};",
        f"const runs = {json.dumps(runs)}, averages = {json.dumps(averages)}, shape = {shape};",
        "console.log(JSON.stringify([shape(leaderboardRows(runs, averages)), shape(leaderboardRows(runs, averages, true)),",
        "  [fmtRange({ mean: 87.95, min: 87.8, max: 88.1, runs: 2 }), fmtRange({ mean: 100, min: 100, max: 100, runs: 3 }),",
        "   fmtRange(null), fmtRange({ mean: 66.7, min: 40, max: 80, runs: 3 }, (n) => String(Math.round(n)))]]));"])
    # Model A's three runs are one row, where its newest run stood. B has one run. The run without a group
    # (unfinished, or saved before fingerprints) stays a row of its own. Of C's two runs only one is listed
    # here, so it is shown as the run it is.
    assert grouped == [["A", [5, 3, 1]], 4, 2, 0]
    assert best == [5, 4, 3, 2, 1, 0]                                      # "best per model only": nothing is averaged
    assert ranges == ["87.8–88.1", "100–100", "—", "40–80"]

    page = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "leaderboardRows(runs, averages.filter(" in page and "header('n_out_of_tokens', 'Out of tokens'" in page
    assert "average.runs} runs" in page and "fmtRange(spread, format)" in page
    styles = (config.STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    assert "table.data .range" in styles and ".bar-range" in styles and "tr.member-row" in styles

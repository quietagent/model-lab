"""Scores that are over part of what was asked: a category with unscored tests, and a first pass with a
whole category missing.

Live (2026-10-08) one provider rejected five or six of the twelve long prompts in every run, so each
"Long context" score was over six or seven tests, and nothing on the page said so. And in one run the
first pass had no Writing score at all: its 80.6 was the average of eight categories, and it was
averaged with first passes of nine (the run scored 77.1 once Writing was graded).
"""
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import bench_tests, config, db
from app.main import app

SMART = {"provider": "openrouter", "model": "mock/smart"}
LIMIT = 7003                                  # a token limit no other test uses: these runs are a group of their own
SETTINGS = f"max_tokens={LIMIT};judge=none"


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def start(client, **body):
    """Run a benchmark to the end -> the saved run."""
    payload = {"target": SMART, "categories": ["math", "knowledge"], "concurrency": 4, "max_tokens": LIMIT, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    lines = client.get(f"/api/benchmark/{started.json()['run_id']}/events").text.splitlines()
    summary = [json.loads(line[5:]) for line in lines if line.startswith("data:")][-1]
    assert summary["type"] == "summary"
    return client.get(f"/api/runs/{summary['run']['id']}").json()["run"]


def fail_request(run_id, *test_ids):
    """Make tests of a saved run end as the provider's failure (an API error: no score), and work the
    run's score out again."""
    with db.connect() as conn:
        for test_id in test_ids:
            assert conn.execute("UPDATE results SET status = 'error', score = NULL, error = 'Request rejected (400).' "
                                "WHERE run_id = ? AND test_id = ?", (run_id, test_id)).rowcount == 1
    return db.finalize_run(run_id, "completed")


# ---------------------------------------------------------------- "6 of 12 scored"

def test_a_run_says_how_many_tests_of_each_category_were_scored(client):
    run = start(client)
    assert run["category_counts"] == {"math": {"scored": 5, "total": 5}, "knowledge": {"scored": 5, "total": 5}}
    after = fail_request(run["id"], "math-change", "math-speed")
    assert after["category_counts"] == {"math": {"scored": 3, "total": 5}, "knowledge": {"scored": 5, "total": 5}}
    assert after["category_scores"]["math"] == 100.0 and after["score"] == 100.0     # the score is over the other three
    listed = {r["id"]: r for r in client.get("/api/runs", params={"level": "standard", "settings": SETTINGS}).json()["runs"]}
    assert listed[run["id"]]["category_counts"]["math"] == {"scored": 3, "total": 5}
    # A reply the provider cut short has no score either; one that ran out of tokens scored 0 and counts.
    with db.connect() as conn:
        conn.execute("UPDATE results SET status = 'cut_off', score = NULL WHERE run_id = ? AND test_id = 'know-capital'",
                     (run["id"],))
        conn.execute("UPDATE results SET status = 'cut_off', score = 0 WHERE run_id = ? AND test_id = 'know-gas'",
                     (run["id"],))
    assert db.get_run(run["id"], with_results=False)["category_counts"]["knowledge"] == {"scored": 4, "total": 5}
    # A category in which nothing was scored has no score, and the count says why.
    gone = fail_request(run["id"], "math-discount", "math-garden", "math-pipes")
    assert "math" not in gone["category_scores"] and gone["category_counts"]["math"] == {"scored": 0, "total": 5}


def test_the_average_of_a_models_runs_says_how_many_tests_its_categories_were_scored_over(client):
    first, second, third = start(client, categories=["math"]), start(client, categories=["math"]), start(
        client, categories=["math"])
    fail_request(second["id"], "math-change")
    fail_request(third["id"], "math-change", "math-speed")
    data = client.get("/api/runs", params={"level": "standard", "settings": SETTINGS}).json()
    average = {a["key"]: a for a in data["averages"]}[first["group"]]
    assert average["run_ids"] == [first["id"], second["id"], third["id"]]
    assert average["category_counts"] == {"math": {"min": 3, "max": 5, "total": 5}}
    assert average["category_scores"]["math"]["runs"] == 3                  # every run has a Math score

    rows = [{"category_counts": {"longctx": {"scored": 7, "total": 12}}},
            {"category_counts": {"longctx": {"scored": 6, "total": 12}}}, {"category_counts": {}}, {}]
    assert db._count_range(rows, "longctx") == {"min": 6, "max": 7, "total": 12}
    assert db._count_range(rows, "math") is None                            # nothing known: nothing is claimed


# ---------------------------------------------------------------- a first pass with a category missing

def run_row(run_id, score, first_pass, **extra):
    row = {"id": run_id, "created_at": float(run_id), "status": "completed", "provider": "openrouter", "model": "m/one",
           "level": "hard", "settings_key": "max_tokens=32768;judge=none", "fingerprint": "abc", "score": score,
           "category_scores": {"math": score}, "first_pass": first_pass, "n_out_of_tokens": 0, "avg_latency": 2.0,
           "avg_tps": 40.0, "total_cost": 0.0, "cost_estimated": False, "older_tests": False, **extra}
    return {**row, "group": db.group_key(row)}


def test_a_first_pass_with_a_whole_category_missing_is_named():
    assert db.missing_categories(["math", "writing"], {"math": 100.0, "writing": 60.0}) == []
    assert db.missing_categories(["math", "longctx", "writing"], {"math": 100.0}) == ["longctx", "writing"]
    assert db.missing_categories(["math"], None) == ["math"] and db.missing_categories([], {}) == []


def test_an_incomplete_first_pass_is_left_out_of_the_first_pass_average_and_range():
    """The live case: 76.1, 80.6 (no Writing) and 74.7 were shown as 77.1 (74.7-80.6)."""
    runs = [run_row(27, 76.1, {"score": 76.1}, first_pass_missing=[]),
            run_row(34, 77.1, {"score": 80.6}, first_pass_missing=["writing"]),
            run_row(35, 74.7, {"score": 74.7}, first_pass_missing=[])]
    (average,) = db.model_averages(runs)
    assert average["first_pass"] == {"mean": 75.4, "min": 74.7, "max": 76.1, "runs": 2}
    assert average["first_pass_left_out"] == [{"id": 34, "score": 80.6, "missing": ["writing"]}]
    # The final scores are all complete, and all three are averaged as before.
    assert average["score"] == {"mean": 76.0, "min": 74.7, "max": 77.1, "runs": 3} and average["runs"] == 3
    # Complete first passes: nothing is left out. (A row that names nothing missing is a complete one.)
    (whole,) = db.model_averages([run_row(1, 80.0, {"score": 80.0}),
                                  run_row(2, 90.0, {"score": 90.0}, first_pass_missing=[])])
    assert whole["first_pass"]["runs"] == 2 and whole["first_pass_left_out"] == []
    # No complete first pass at all: no average, rather than one made of unlike numbers.
    (none,) = db.model_averages([run_row(1, 80.0, {"score": 85.0}, first_pass_missing=["writing"]),
                                 run_row(2, 90.0, {"score": 95.0}, first_pass_missing=["arabic"])])
    assert none["first_pass"] is None and [run["id"] for run in none["first_pass_left_out"]] == [1, 2]


def test_a_first_pass_in_which_the_judge_failed_has_no_writing_score_and_is_not_averaged(client):
    """As it happened live: the judge failed on every Writing test of the first pass, so the first pass
    had no Writing score. "Retry failed requests" graded the stored replies, and the final score has it."""
    judge = {"provider": "openrouter", "model": "mock/outage"}             # 429 four times per prompt, then it answers
    body = {"categories": ["math", "writing"], "judge": judge, "max_tokens": LIMIT + 1}
    run = start(client, **body)
    assert run["first_pass"]["category_scores"] == {"math": 100.0} and run["first_pass"]["score"] == 100.0
    assert run["first_pass_missing"] == ["writing"]
    assert run["category_counts"]["writing"] == {"scored": 0, "total": 4} and run["n_error"] == 4

    assert client.post(f"/api/runs/{run['id']}/retry", json={}).status_code == 200
    client.get(f"/api/benchmark/{run['id']}/events")                        # follow the retry to its end
    after = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert after["n_error"] == 0 and "writing" in after["category_scores"] and after["score"] < 100.0
    assert after["category_counts"]["writing"] == {"scored": 4, "total": 4}
    # The first pass is what it was: still without Writing, and still marked.
    assert after["first_pass"] == run["first_pass"] and after["first_pass_missing"] == ["writing"]
    assert "missing" not in after["first_pass"]                            # worked out next to it, not stored in it

    # Two more runs whose judge answers at once: their first passes are complete.
    second, third = start(client, **body), start(client, **body)
    assert second["first_pass_missing"] == third["first_pass_missing"] == []
    assert second["group"] == after["group"]
    data = client.get("/api/runs", params={"level": "standard", "settings": after["settings_key"]}).json()
    average = {a["key"]: a for a in data["averages"]}[after["group"]]
    assert average["runs"] == 3 and average["score"]["runs"] == 3
    assert average["first_pass"]["runs"] == 2                               # the incomplete one is not in it
    assert average["first_pass"]["max"] == second["first_pass"]["score"] < 100.0
    assert average["first_pass_left_out"] == [{"id": run["id"], "score": 100.0, "missing": ["writing"]}]
    listed = {r["id"]: r for r in data["runs"]}
    assert listed[run["id"]]["first_pass_missing"] == ["writing"]


# ---------------------------------------------------------------- a final score with a category missing

def test_an_incomplete_final_score_is_left_out_of_the_average_and_range():
    """The live case: Laguna's run #42 had no Long context and no Writing score, and its 49.0 was ranked
    with nine-category scores."""
    runs = [run_row(1, 80.0, {"score": 80.0}, final_missing=[]),
            run_row(2, 90.0, {"score": 90.0}, final_missing=["longctx", "writing"]),
            run_row(3, 70.0, {"score": 70.0}, final_missing=[])]
    (average,) = db.model_averages(runs)
    assert average["score"] == {"mean": 75.0, "min": 70.0, "max": 80.0, "runs": 2} and average["runs"] == 3
    assert average["ranked"] and average["missing"] == [] and average["grade"] == "C"
    assert average["score_left_out"] == [{"id": 2, "score": 90.0, "missing": ["longctx", "writing"]}]
    # Every run complete: as before. (A row that names nothing missing is a complete one.)
    (whole,) = db.model_averages([run_row(1, 80.0, {"score": 80.0}), run_row(2, 90.0, {"score": 90.0}, final_missing=[])])
    assert whole["score"]["runs"] == 2 and whole["ranked"] and whole["score_left_out"] == []
    # No run complete: the average of what there is, not ranked, and what is missing named once each.
    (none,) = db.model_averages([run_row(1, 50.0, {"score": 50.0}, final_missing=["writing", "longctx"]),
                                 run_row(2, 60.0, {"score": 60.0}, final_missing=["longctx"])])
    assert none["score"] == {"mean": 55.0, "min": 50.0, "max": 60.0, "runs": 2}
    assert not none["ranked"] and none["missing"] == ["longctx", "writing"] and none["score_left_out"] == []


def test_a_run_with_a_whole_category_missing_is_marked_and_not_averaged(client):
    limit = LIMIT + 2                                                       # a group of its own
    knowledge = [t["id"] for t in bench_tests.BUILTIN if t["category"] == "knowledge" and t["level"] == "standard"]
    first, second, short = (start(client, max_tokens=limit) for _ in range(3))
    assert first["final_missing"] == [] and first["score"] == 100.0
    short = fail_request(short["id"], *knowledge)
    assert short["final_missing"] == ["knowledge"] and short["score"] == 100.0       # Math alone
    assert short["first_pass_missing"] == []                                # its first pass had Knowledge
    settings = f"max_tokens={limit};judge=none"
    data = client.get("/api/runs", params={"level": "standard", "settings": settings}).json()
    average = {a["key"]: a for a in data["averages"]}[first["group"]]
    assert average["runs"] == 3 and average["score"]["runs"] == 2 and average["ranked"]
    assert average["score_left_out"] == [{"id": short["id"], "score": 100.0, "missing": ["knowledge"]}]
    assert {r["id"]: r for r in data["runs"]}[short["id"]]["final_missing"] == ["knowledge"]
    # "Best per model" takes a complete run over an incomplete one with the same score, though it is newer.
    best = client.get("/api/runs", params={"level": "standard", "settings": settings, "best": True}).json()["runs"]
    assert [run["id"] for run in best] in ([first["id"]], [second["id"]])
    # A run whose score is not there at all names nothing missing.
    assert db.missing_categories(["math"], {}) == ["math"]
    gone = fail_request(first["id"], *knowledge, "math-change", "math-speed", "math-discount", "math-garden", "math-pipes")
    assert gone["score"] is None and gone["final_missing"] == []


# ---------------------------------------------------------------- the page

def in_node(lines):
    result = subprocess.run(["node", "--input-type=module", "-"], input=chr(10).join(lines), capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_says_how_many_tests_were_scored_only_where_some_were_not():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    counts = [{"scored": 6, "total": 12}, {"scored": 12, "total": 12}, {"scored": 0, "total": 4},
              {"min": 6, "max": 7, "total": 12}, {"min": 6, "max": 6, "total": 12}, {"min": 12, "max": 12, "total": 12},
              {"min": 11, "max": 12, "total": 12}, None]
    one, whole, nothing, spread, same, all_runs, some_runs, unknown = in_node([
        f"import {{ scoredText }} from {json.dumps(module)};",
        f"console.log(JSON.stringify({json.dumps(counts)}.map((c) => scoredText(c))));"])
    assert (one, whole, nothing) == ("6 of 12 scored", "", "0 of 4 scored")
    assert (spread, same, all_runs, some_runs, unknown) == ("6–7 of 12 scored", "6 of 12 scored", "", "11–12 of 12 scored", "")
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "scoredText(counted[category])" in page
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "scoredText(run.category_counts?.[c.id])" in board and "scoredText(average.category_counts?.[c.id])" in board
    styles = (config.STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    assert ".range.partly" in styles and ".bar-label small" in styles


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_marks_a_first_pass_that_has_a_whole_category_missing():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    labels = {"writing": "Writing", "longctx": "Long context", "arabic": "Arabic"}
    runs = [{"score": 77.1, "retries": 1, "first_pass": {"score": 80.6, "n_error": 15, "n_cut_off": 5},
             "first_pass_missing": ["writing"]},
            {"score": 76.1, "retries": 1, "first_pass": {"score": 76.1, "n_error": 7, "n_cut_off": 4}, "first_pass_missing": []},
            {"score": 60, "retries": 0, "first_pass": {"score": 70, "n_error": 9, "n_cut_off": 0},
             "first_pass_missing": ["writing", "longctx", "arabic"]},
            {"score": 50, "retries": 1, "first_pass": {"score": None, "n_error": 10, "n_cut_off": 0}, "first_pass_missing": ["math"]},
            {"score": 80, "retries": 0, "first_pass": {"score": 80, "n_error": 0, "n_cut_off": 0}, "first_pass_missing": ["custom"]}]
    words, lists = in_node([
        f"import {{ firstPassText, listWords }} from {json.dumps(module)};",
        f"console.log(JSON.stringify([{json.dumps(runs)}.map((run) => firstPassText(run, {json.dumps(labels)})),",
        "  [[], ['a'], ['a', 'b'], ['a', 'b', 'c']].map(listWords)]));"])
    assert words[0] == {"first": "First pass 80.6", "left": "15 API errors, 5 cut off",
                        "final": "Final 77.1 after one retry of failed requests", "incomplete": "no Writing score"}
    assert "incomplete" not in words[1]                                     # a complete first pass reads as before
    assert words[2]["incomplete"] == "no Writing, Long context and Arabic score"
    assert "incomplete" not in words[3] and words[3]["first"] == "First pass scored nothing"
    assert words[4]["incomplete"] == "no custom score"                      # a category without a label keeps its id
    assert lists == ["", "a", "a and b", "a, b and c"]
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "words.incomplete" in page
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "firstPassSpread(average)" in board and "firstPassCell(run)" in board and "first_pass_left_out" in board


def test_the_page_marks_an_incomplete_final_score_and_lists_it_below_the_ranked_rows():
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "scoreSpread(average)" in board and "scoreCell(run)" in board and "score_left_out" in board
    assert "[...sorted(ranked), ...sorted(unranked)]" in board              # ranked rows first, then the others
    assert "new Map([...ranked].sort(" in board                             # only ranked rows get a rank
    assert ".filter(isRanked)" in board                                     # and only they are in the chart
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "view.run.final_missing" in page

""""Judge again": grade the judged results of a group of runs again with another judge, from the stored replies.

Live (2026-10-18) the judge every run had used disappeared from OpenRouter (404 "No endpoints found"). The
replies were stored, so only the judge's part had to be done again: no model is asked again, code-graded
results are not touched, the database is backed up first, and the runs move to the new judge's group of
settings. And a run whose judge does not answer is refused at the start, instead of running without one.
"""
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import benchmark, config, db, fingerprint, rejudge
from app.main import app

SMART = {"provider": "openrouter", "model": "mock/smart"}
BROKEN = {"provider": "openrouter", "model": "mock/error-500"}      # there, but every call fails
GONE = {"provider": "openrouter", "model": "mock/error-404"}        # not there at all
GOOD = {"provider": "openrouter", "model": "mock/average"}
OTHER = {"provider": "openrouter", "model": "mock/thinker"}
CATEGORIES = ["math", "honesty", "writing"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def events(response):
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def start(client, judge, limit, target=SMART):
    """A finished run of Math, Honesty and Writing at a token limit no other test uses -> the saved run."""
    payload = {"target": target, "categories": CATEGORIES, "concurrency": 4, "judge": judge, "max_tokens": limit}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    events(client.get(f"/api/benchmark/{started.json()['run_id']}/events"))
    return client.get(f"/api/runs/{started.json()['run_id']}").json()["run"]


def judge_again(client, run, judge):
    """Judge a run's group again -> the job's events."""
    started = client.post("/api/judge-again", json={"level": "standard", "settings": run["settings_key"], "judge": judge})
    assert started.status_code == 200, started.text
    return events(client.get("/api/judge-again/events"))


def results(run_id):
    return {result["test_id"]: result for result in db.get_results(run_id)}


def code_graded(run_id):
    return {test_id: r for test_id, r in results(run_id).items() if r["category"] == "math"}


def test_the_judged_results_are_graded_again_and_the_runs_move_to_the_new_judge(client):
    first, second = start(client, BROKEN, 7201), start(client, BROKEN, 7201)
    assert first["settings_key"] == second["settings_key"] == "max_tokens=7201;judge=openrouter/mock/error-500"
    old = results(first["id"])
    # The judge failed on every call: Writing is unscored (its replies kept), Honesty by the phrase check.
    assert all(r["status"] == "error" and benchmark.reply_kept(r) for r in old.values() if r["category"] == "writing")
    assert all(benchmark.judge_fell_back(r) for r in old.values() if r["category"] == "honesty")
    assert "writing" not in first["category_scores"] and first["final_missing"] == ["writing"]
    math_before = {run["id"]: code_graded(run["id"]) for run in (first, second)}

    plan = client.get("/api/judge-again/plan", params={"level": "standard", "settings": first["settings_key"],
                                                      "provider": "openrouter", "model": "mock/smart"}).json()
    assert [run["id"] for run in plan["runs"]] == [first["id"], second["id"]]
    assert plan["calls"] == 2 * 9 and plan["older"] == [] and plan["busy"] == []
    assert plan["new_settings_key"] == "max_tokens=7201;judge=openrouter/mock/smart"

    stream = judge_again(client, first, SMART)
    kinds = [event["type"] for event in stream]
    assert kinds[0] == "start" and kinds[-1] == "summary" and kinds.count("judged") == 18
    summary = stream[-1]
    assert summary["written"] and summary["failed"] == 0 and summary["settings_key"] == plan["new_settings_key"]
    backup = Path(stream[0]["backup"])
    assert backup.is_file() and backup.parent == config.data_dir() / "backups"
    with sqlite3.connect(backup) as conn:                                   # the copy is from before
        assert conn.execute("SELECT judge_model FROM runs WHERE id = ?", (first["id"],)).fetchone()[0] == "mock/error-500"

    for run in (first, second):
        after = client.get(f"/api/runs/{run['id']}").json()["run"]
        assert (after["judge_provider"], after["judge_model"]) == ("openrouter", "mock/smart")
        assert after["settings"] == {"max_tokens": 7201, "judge": "openrouter/mock/smart"}
        assert after["settings_key"] == plan["new_settings_key"] and not after["older_tests"]
        assert after["fingerprint"] == fingerprint.fingerprint_of(after["test_hashes"], after["settings"])
        assert after["fingerprint"] != run["fingerprint"]
        assert "writing" in after["category_scores"] and after["final_missing"] == []
        # Nothing was asked of the model, and nothing graded by code moved.
        assert code_graded(run["id"]) == math_before[run["id"]]
        stored = results(run["id"])
        for result in stored.values():
            assert result["response"] and result["tokens_out"] and result["served_by"] == ["Alpha"]
        for result in stored.values():
            if result["category"] != "math":
                assert result["status"] in ("pass", "partial", "fail") and result["error"] is None
                assert result["notes"].endswith(
                    f"by openrouter/mock/error-500.]") and "[Judged again with openrouter/mock/smart on " in result["notes"]
                assert benchmark.PHRASE_CHECK_INSTEAD not in result["notes"].split(" [Judged again with")[0]
        # Its first pass graded the same replies: it is worked out again (no retry, so the same as the final).
        assert after["first_pass"]["score"] == after["score"] and after["first_pass_missing"] == []
        assert after["params"]["judged_again"][-1]["from"] == "openrouter/mock/error-500"
        assert after["params"]["judged_again"][-1]["to"] == "openrouter/mock/smart"
    for test_id, result in results(first["id"]).items():                   # the same replies, word for word
        assert (result["response"], result["cost"], result["tokens_in"], result["attempts"]) == (
            old[test_id]["response"], old[test_id]["cost"], old[test_id]["tokens_in"], old[test_id]["attempts"])

    # A new run with the new judge has the same fingerprint and group: they are averaged together.
    third = start(client, SMART, 7201)
    assert third["settings_key"] == plan["new_settings_key"]
    moved = client.get(f"/api/runs/{first['id']}").json()["run"]
    assert third["fingerprint"] == moved["fingerprint"] and third["group"] == moved["group"]
    data = client.get("/api/runs", params={"level": "standard", "settings": plan["new_settings_key"]}).json()
    (average,) = [a for a in data["averages"] if a["key"] == moved["group"]]
    assert average["run_ids"] == [first["id"], second["id"], third["id"]]
    old_group = client.get("/api/runs", params={"level": "standard", "settings": first["settings_key"]}).json()
    assert old_group["runs"] == []


def test_a_judge_that_gives_no_verdict_writes_nothing_and_a_gone_judge_is_refused(client):
    run = start(client, GOOD, 7202)
    before = client.get(f"/api/runs/{run['id']}").json()["run"]
    stream = judge_again(client, run, BROKEN)                               # every call fails: nothing written
    assert stream[-1]["type"] == "summary" and not stream[-1]["written"] and "no verdict" in stream[-1]["message"]
    after = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert {key: after[key] for key in ("score", "settings_key", "fingerprint", "judge_model", "first_pass")} == {
        key: before[key] for key in ("score", "settings_key", "fingerprint", "judge_model", "first_pass")}
    assert results(run["id"]) == {r["test_id"]: r for r in db.get_results(run["id"])}
    stream = judge_again(client, run, GONE)                                 # not there: refused before anything
    assert stream[-1]["type"] == "error" and stream[-1]["code"] == "judge_unavailable"
    assert "does not answer" in stream[-1]["message"] and "404" in stream[-1]["message"]
    assert client.get(f"/api/runs/{run['id']}").json()["run"]["judge_model"] == "mock/average"
    # The same judge again, a group without a judge, no judge picked: refused.
    same = client.post("/api/judge-again", json={"level": "standard", "settings": run["settings_key"], "judge": GOOD})
    assert same.status_code == 409 and same.json()["code"] == "same_judge"
    plain = start(client, None, 7203)
    nothing = client.post("/api/judge-again", json={"level": "standard", "settings": plain["settings_key"], "judge": SMART})
    assert nothing.status_code == 409 and nothing.json()["code"] == "nothing_to_judge"
    empty = client.post("/api/judge-again", json={"level": "standard", "settings": run["settings_key"],
                                                  "judge": {"provider": "openrouter", "model": ""}})
    assert empty.status_code == 400 and empty.json()["code"] == "no_model"


def test_a_run_with_older_tests_is_left_as_it_is(client):
    current, older = start(client, GOOD, 7204), start(client, GOOD, 7204)
    hashes = {**older["test_hashes"], "math-change": "0" * 16}             # as if a test had changed since
    with db.connect() as conn:
        conn.execute("UPDATE runs SET test_hashes = ? WHERE id = ?", (json.dumps(hashes), older["id"]))
    plan = client.get("/api/judge-again/plan", params={"level": "standard", "settings": current["settings_key"],
                                                      "provider": "openrouter", "model": "mock/smart"}).json()
    assert [run["id"] for run in plan["runs"]] == [current["id"]] and plan["older"] == [older["id"]]
    assert judge_again(client, current, SMART)[-1]["written"]
    assert db.get_run(current["id"], with_results=False)["judge_model"] == "mock/smart"
    assert db.get_run(older["id"], with_results=False)["judge_model"] == "mock/average"


def test_a_run_whose_judge_does_not_answer_is_refused_at_the_start(client):
    refused = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math", "writing"], "judge": GONE})
    assert refused.status_code in (400, 409) and refused.json()["code"] == "judge_unavailable"
    assert "mock/error-404 does not answer" in refused.json()["message"] and "Pick another judge" in refused.json()["message"]
    # A judge that is not needed (Math only) is not asked; one that is only busy (500s) is let through.
    math = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math"], "judge": GONE})
    assert math.status_code == 200
    events(client.get(f"/api/benchmark/{math.json()['run_id']}/events"))
    busy = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["honesty"], "judge": BROKEN})
    assert busy.status_code == 200
    events(client.get(f"/api/benchmark/{busy.json()['run_id']}/events"))
    # "Retry failed requests" checks the run's own judge when the retry needs it.
    run = client.get(f"/api/runs/{busy.json()['run_id']}").json()["run"]
    assert benchmark.needs_judge(run, []) and len(benchmark.to_retry(run)) == 5
    with db.connect() as conn:
        conn.execute("UPDATE runs SET judge_model = 'mock/error-404' WHERE id = ?", (run["id"],))
    retried = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert retried.status_code in (400, 409) and retried.json()["code"] == "judge_unavailable"


def test_the_first_pass_is_rebuilt_from_the_notes_of_the_results():
    rows = [
        {"id": 1, "category": "math", "status": "pass", "score": 1.0, "notes": "Correct: 4"},
        {"id": 2, "category": "math", "status": "pass", "score": 1.0,
         "notes": 'Correct: 5 [Re-run by "Retry failed requests": before, this test was an API error (503).]'},
        {"id": 3, "category": "writing", "status": "partial", "score": 0.6,
         "notes": 'Judge: 6/10 [Graded again by "Retry failed requests": before, the judge call failed (x).]'},
        {"id": 4, "category": "writing", "status": "fail", "score": 0.2,
         "notes": 'Judge: 2/10 [Re-run by "Retry failed requests": before, this test was cut short by the provider.]'},
        {"id": 5, "category": "honesty", "status": "pass", "score": 1.0,
         "notes": 'Judge: admits [Judged again by "Retry failed requests": before, the judge call failed and the '
                  'phrase check graded it (fail, 0). The model was not asked again.]'},
        {"id": 6, "category": "honesty", "status": "pass", "score": 1.0, "notes": "Judge: admits"},
    ]
    rebuilt = {row["id"]: (row["status"], row["score"]) for row in rejudge.first_pass_results(rows)}
    assert rebuilt == {1: ("pass", 1.0), 2: ("error", None), 3: ("error", None), 4: ("cut_off", None),
                       5: ("fail", 0.0), 6: ("pass", 1.0)}
    # Under a new judge: the replies the first pass graded get the new marks; the others stay unscored.
    graded = {3: {"status": "pass", "score": 0.9}, 5: {"status": "pass", "score": 1.0}, 6: {"status": "fail", "score": 0.0}}
    new = {row["id"]: (row["status"], row["score"]) for row in rejudge.first_pass_results(rows, graded)}
    assert new == {1: ("pass", 1.0), 2: ("error", None), 3: ("error", None), 4: ("cut_off", None),
                   5: ("pass", 1.0), 6: ("fail", 0.0)}


def test_the_history_of_a_result_is_kept_and_its_new_grade_replaces_the_old():
    notes = ("Judge: 6/10 - deductions: -2: generic [Measured by code, not met: under 110 words.] "
             "[Answered on attempt 2. Failed first - attempt 1: 503] "
             '[Re-run by "Retry failed requests": before, this test was an API error (503).]')
    assert rejudge.history_of(notes) == notes[notes.index(" [Answered on attempt"):]
    assert rejudge.history_of("Judge: 8/10 - fine") == ""
    assert rejudge.history_of("x [Judged again with a/b on 2026-10-18: before, pass, 1 by c/d.]").startswith(" [Judged again with")


def test_while_judging_again_no_benchmark_starts(monkeypatch):
    class Running:
        finished = False
    monkeypatch.setattr(rejudge, "_job", Running())
    with pytest.raises(benchmark.Busy):
        benchmark._refuse_while_judging_again()
    monkeypatch.setattr(rejudge, "_job", None)
    benchmark._refuse_while_judging_again()


def test_the_leaderboard_offers_judge_again_on_a_group_with_a_judge():
    page = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="lbJudgeAgain"' in page and 'id="lbJudgeAgainNote"' in page
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "$('#lbJudgeAgain').hidden = !shown.judge" in board
    assert "/api/judge-again/plan?" in board and "sendJSON('POST', '/api/judge-again'" in board
    assert "streamEvents('/api/judge-again/events'" in board and "confirmDialog(" in board

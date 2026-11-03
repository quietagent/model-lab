""""Retry failed requests" also asks the judge again where its call failed and the phrase check graded.

Live (2026-10-09, Laguna's run #42): two Honesty replies arrived just before the account hit the daily cap
on free models, and the judge's call was refused. judge_honesty graded them by the phrase check instead
and said so in the notes, and "Retry failed requests" left them as they were. Now it asks the judge
again; if the judge fails again, the phrase-check grade and its note stay. A reply the phrase check
graded because no judge was set is not touched.
"""
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import benchmark, config
from app.main import app

SMART = {"provider": "openrouter", "model": "mock/smart"}
OUTAGE = {"provider": "openrouter", "model": "mock/outage"}        # 429 four times per prompt, then it answers
BROKEN = {"provider": "openrouter", "model": "mock/error-500"}     # never answers
FALLBACK = benchmark.PHRASE_CHECK_INSTEAD


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def start(client, **body):
    """Run a benchmark to the end -> the saved run, with its results."""
    payload = {"target": SMART, "categories": ["honesty"], "concurrency": 4, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    return follow(client, started.json()["run_id"])


def follow(client, run_id):
    lines = client.get(f"/api/benchmark/{run_id}/events").text.splitlines()
    assert [json.loads(line[5:]) for line in lines if line.startswith("data:")][-1]["type"] == "summary"
    return client.get(f"/api/runs/{run_id}").json()["run"]


def retry(client, run_id):
    response = client.post(f"/api/runs/{run_id}/retry", json={})
    if response.status_code != 200:
        return response
    return follow(client, run_id)


def by_test(run):
    return {result["test_id"]: result for result in run["results"]}


def test_the_judge_is_asked_again_where_its_call_failed(client):
    run = start(client, judge=OUTAGE)
    fell_back = [r for r in run["results"] if FALLBACK in r["notes"]]
    assert len(fell_back) == 5 and all(benchmark.judge_fell_back(r) for r in fell_back)
    assert all(r["status"] in ("pass", "partial", "fail") and r["notes"].startswith("Judge failed (") for r in fell_back)
    assert run["n_error"] == 0 and benchmark.to_retry(run) == run["results"]

    after = retry(client, run["id"])
    assert after["retries"] == 1 and after["first_pass"] == run["first_pass"]
    old = by_test(run)
    for test_id, result in by_test(after).items():
        assert FALLBACK not in result["notes"] and "Mock judge:" in result["notes"]
        assert "[Judged again by \"Retry failed requests\": before, the judge call failed and the phrase check graded it" in result["notes"]
        assert not benchmark.judge_fell_back(result)
        # The model was not asked again: its reply, cost, attempts and provider are what they were.
        for key in ("response", "cost", "tokens_in", "tokens_out", "attempts", "served_by"):
            assert result[key] == old[test_id][key], (test_id, key)
    # Nothing is left to do: the judge has graded every reply.
    again = retry(client, run["id"])
    assert again.status_code == 409 and again.json()["code"] == "nothing_to_retry"


def test_if_the_judge_fails_again_the_phrase_check_grade_and_its_note_stay(client):
    run = start(client, judge=BROKEN)
    assert all(FALLBACK in r["notes"] for r in run["results"])
    after = retry(client, run["id"])
    assert after["retries"] == 1 and after["score"] == run["score"]
    old = by_test(run)
    for test_id, result in by_test(after).items():
        for key in ("status", "score", "notes", "response", "attempts", "served_by", "cost"):
            assert result[key] == old[test_id][key], (test_id, key)
    # It can be asked again later.
    assert len(benchmark.to_retry(after)) == 5


def test_a_reply_the_phrase_check_graded_because_no_judge_was_set_is_not_touched(client):
    run = start(client)
    assert run["judge_model"] is None and all(FALLBACK not in r["notes"] for r in run["results"])
    assert benchmark.to_retry(run) == []
    refused = retry(client, run["id"])
    assert refused.status_code == 409 and refused.json()["code"] == "nothing_to_retry"
    # Even a note that says so is not taken when the run had no judge.
    fake = {**run, "results": [{**r, "notes": f"Judge failed (x){FALLBACK} y"} for r in run["results"]]}
    assert benchmark.to_retry(fake) == []
    assert len(benchmark.to_retry({**fake, "judge_model": "m"})) == 5


def test_which_results_are_asked_again():
    note = f"Judge failed (Rate limited (429).){FALLBACK} Corrects the false premise ('canceled')"
    cases = [({"status": "pass", "score": 1.0, "response": "No.", "notes": note}, True),
             ({"status": "fail", "score": 0.0, "response": "No.", "notes": "Reached the token limit. " + note}, True),
             ({"status": "pass", "score": 1.0, "response": "No.", "notes": "Corrects the false premise"}, False),
             ({"status": "error", "score": None, "response": "No.", "notes": note}, False),
             ({"status": "pass", "score": 1.0, "response": " ", "notes": note}, False),
             ({"status": "cut_off", "score": 0.0, "response": "No.", "notes": note}, False)]
    assert [benchmark.judge_fell_back(case) for case, _ in cases] == [want for _, want in cases]
    run = {"judge_model": "j", "results": [
        {"status": "error", "score": None, "response": "", "notes": ""},
        {"status": "cut_off", "score": None, "response": "x", "notes": ""},
        {"status": "cut_off", "score": 0.0, "response": "x", "notes": ""},                 # out of tokens: not again
        cases[0][0], cases[2][0]]}
    assert benchmark.to_retry(run) == [run["results"][0], run["results"][1], cases[0][0]]


def in_node(lines):
    result = subprocess.run(["node", "--input-type=module", "-"], input=chr(10).join(lines), capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_button_says_how_many_of_each_it_will_do():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    note = f"Judge failed (x){FALLBACK} y"
    rows = [{"status": "pass", "response": "No.", "notes": note}, {"status": "fail", "response": "No.", "notes": note},
            {"status": "pass", "response": "No.", "notes": "Corrects"}, {"status": "error", "response": "No.", "notes": note},
            {"status": "partial", "response": "", "notes": note}]
    labels, flags = in_node([
        f"import {{ isRejudgeable, retryLabel }} from {json.dumps(module)};",
        "console.log(JSON.stringify([[retryLabel(22, 2), retryLabel(22, 0), retryLabel(0, 2), retryLabel(1, 1)],",
        f"  {json.dumps(rows)}.map(isRejudgeable)]));"])
    assert labels == ["Retry failed requests (22 failed, 2 to re-judge)", "Retry failed requests (22)",
                      "Retry failed requests (2 to re-judge)", "Retry failed requests (1 failed, 1 to re-judge)"]
    # The page and the server agree on which replies go to the judge again.
    assert flags == [benchmark.judge_fell_back({"score": None, **row}) for row in rows] == [True, True, False, False, False]
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "retryLabel(retryable, rejudge)" in page and "view.run?.judge_model ? results.filter(isRejudgeable)" in page

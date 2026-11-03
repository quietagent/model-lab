"""The note about failed attempts on a result: its count and its list agree.

A request that fails with a rate limit or a server error is tried again, and the result's notes keep
why. A reply that is cut short or empty is asked for once more: that is a second request, with attempts
of its own. Seen live (2026-10-08): such a result said "Answered on attempt 3" and then listed three
failed attempts. The count was the second request's own, the list came from both requests.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

from app import benchmark
from app.main import app
from app.providers import Completion

RATE = "Rate limited (429). Wait a moment and try again."
DOWN = "Provider error (503). The upstream service failed."


def reply(attempts, *errors):
    return Completion(content="Answer: 4", attempts=attempts, retry_errors=list(errors))


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def test_one_request_reads_as_it_always_did():
    assert benchmark._with_retry_errors("Correct: 4", [reply(1)]) == "Correct: 4"
    assert benchmark._with_retry_errors("Correct: 4", [reply(3, RATE, DOWN)]) == (
        f"Correct: 4 [Answered on attempt 3. Failed first - attempt 1: {RATE}; attempt 2: {DOWN}]")


def test_a_reply_asked_for_twice_tells_its_two_requests_apart():
    both = benchmark._with_retry_errors("Correct: 4", [reply(3, RATE, DOWN), reply(2, RATE)], asked_again=True)
    assert both == ("Correct: 4 [Asked for twice. "
                    f"First request: answered on attempt 3 (failed first - attempt 1: {RATE}; attempt 2: {DOWN}). "
                    f"Second request: answered on attempt 2 (failed first - attempt 1: {RATE}).]")
    assert "Answered on attempt" not in both                        # no count that belongs to one request only
    # Each request lists one failed attempt fewer than the attempt it was answered on.
    first, second = both.split(". Second request: ")
    assert len(re.findall(r"attempt \d+: ", first)) == 2 and "answered on attempt 3" in first
    assert len(re.findall(r"attempt \d+: ", second)) == 1 and "answered on attempt 2" in second
    first_only = benchmark._with_retry_errors("n", [reply(2, DOWN), reply(1)], asked_again=True)
    assert first_only == (f"n [Asked for twice. First request: answered on attempt 2 (failed first - attempt 1: {DOWN}). "
                          "Second request: answered on attempt 1.]")
    second_only = benchmark._with_retry_errors("n", [reply(1), reply(2, RATE)], asked_again=True)
    assert second_only == ("n [Asked for twice. First request: answered on attempt 1. "
                           f"Second request: answered on attempt 2 (failed first - attempt 1: {RATE}).]")
    # Nothing failed in either request: nothing to keep.
    assert benchmark._with_retry_errors("n", [reply(1), reply(1)], asked_again=True) == "n"


def test_a_second_request_that_failed_outright_is_not_called_answered():
    note = benchmark._with_retry_errors("Cut off: ... Asking again failed (x).", [reply(2, RATE)], asked_again=True)
    assert note.endswith(f"[Asked for twice. First request: answered on attempt 2 (failed first - attempt 1: {RATE}). "
                         "Second request: failed, as said above.]")


def test_a_run_whose_first_request_needed_a_retry_and_was_then_cut_short(client):
    """The mock model answers 429 once, then sends a reply that is cut short, then a whole reply."""
    started = client.post("/api/benchmark/run", json={"target": {"provider": "openrouter", "model": "mock/flaky-hiccup"},
                                                      "categories": ["math"], "concurrency": 2})
    assert started.status_code == 200, started.text
    client.get(f"/api/benchmark/{started.json()['run_id']}/events")          # follow it to the end
    run = client.get(f"/api/runs/{started.json()['run_id']}").json()["run"]
    assert run["score"] == 100.0 and len(run["results"]) == 5
    for result in run["results"]:
        notes = result["notes"]
        assert result["status"] == "pass" and result["attempts"] == 3          # three requests went out in all
        assert "[Asked again: the first reply was cut off (" in notes
        assert ("[Asked for twice. First request: answered on attempt 2 (failed first - attempt 1: Rate limited (429)"
                in notes), notes
        assert notes.endswith("Second request: answered on attempt 1.]")
        assert "Answered on attempt" not in notes and "attempt 3" not in notes
        assert json.dumps(result["served_by"]) == '["Alpha", "Alpha"]'         # the two replies that arrived

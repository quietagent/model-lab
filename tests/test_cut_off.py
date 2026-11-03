"""One attempt per test. A reply that stops before it is finished is detected; it is asked for again
only when the provider failed (never when the model ran out of tokens); "Retry failed requests"
re-runs only what the provider failed; and a run keeps its first-pass score."""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import bench_tests, benchmark, db, graders, providers
from app.main import app
from app.providers import Completion, ProviderError, Target

RIVER = bench_tests.BY_ID["hard-reason-river"]
FRAGMENT = "The boat can carry the farmer, wolf, goat,"          # the reply that arrived live, word for word
NUMBER = {"type": "number", "expected": 1}
TARGET = Target("openrouter", "some/model")
JUDGE = {"provider": "openrouter", "model": "mock/average"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def follow(client, run_id):
    """A job's events to the end -> (events, the saved run with its results)."""
    stream = [e for e in events(client.get(f"/api/benchmark/{run_id}/events")) if e["type"] != "hello"]
    assert stream[-1]["type"] == "summary"
    return stream, client.get(f"/api/runs/{run_id}").json()["run"]


def run_benchmark(client, model, **body):
    payload = {"target": {"provider": "openrouter", "model": model}, "categories": ["math"], "concurrency": 4, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    return follow(client, started.json()["run_id"])


def retry(client, run_id, **body):
    started = client.post(f"/api/runs/{run_id}/retry", json=body)
    assert started.status_code == 200, started.text
    assert started.json()["ok"] and started.json()["retry"] is True
    return follow(client, run_id)


# ---------------------------------------------------------------- how a reply ends

@pytest.mark.parametrize("reply, spec, expected", [
    (FRAGMENT, NUMBER, "sure"),                                              # ends on a comma
    ("Working:\n```python\ndef f(x):\n    return x", {"type": "code"}, "sure"),   # the code block never closes
    ("The pieces are:", NUMBER, "sure"),
    ("Answer:", NUMBER, "sure"),
    ("We add the three parts (", NUMBER, "sure"),
    ("First the farmer loads the boat and", NUMBER, "sure"),
    ("He then rows across with the wolf, the goat and the", {"type": "contains"}, "sure"),
    ("المجموع هو ثلاثة وثلاثون من", NUMBER, "sure"),   # Arabic, ends on "from"
    ("The boat can carry the farmer", NUMBER, "maybe"),                      # no Answer line, no full stop
    ("Final price: 60 + 4.8 = 64.8", NUMBER, "maybe"),
    ("Dear passenger, we are sorry and will contact you within 24 hours with a confirmed", {"type": "judge"}, "maybe"),
    # Finished replies, including ones without a full stop.
    ("One trip is enough.\n\nAnswer: 1", NUMBER, None),
    ("One trip is enough, so the answer is 1.", NUMBER, None),
    ("Answer: **1**", NUMBER, None),
    ("x = 3", NUMBER, None),
    ("1437", {"type": "contains"}, None),
    ("Titania", {"type": "final_answer"}, None),
    ("```python\ndef f(x):\n    return x\n```", {"type": "code"}, None),
    ("```python\ndef f(x):\n    return x\n```\nThis handles every case.", {"type": "code"}, None),
    ('[{"word": "kiwi"}, {"word": "plum"}]', {"type": "constraints"}, None),
    ("the quick brown fox jumps over the lazy dog every single morning", {"type": "lowercase_plain"}, None),
    ("A calm dog sat down and saw a cat run past a tall oak", {"type": "constraints"}, None),
    ("So the result is $347$", NUMBER, None),
    ("", NUMBER, None),
])
def test_how_a_reply_ends_tells_whether_it_was_finished(reply, spec, expected):
    found = graders.unfinished_reply(reply, spec)
    assert (found[0] if found else None) == expected, found
    if found:
        assert found[1] and "\n" not in found[1]


def test_a_missing_answer_line_alone_is_not_a_cut_off():
    """A model that leaves the line out but finishes its sentence made its own mistake."""
    assert graders.unfinished_reply("The width is 12 and the length is 36, so the area is 432 square meters.",
                                    {"type": "number", "expected": 432}) is None
    # A custom number test does not ask for the line at all.
    assert graders.unfinished_reply("The total comes to 42", {"type": "number", "expected": 42,
                                                              "require_answer_line": False}) is None


def reply(content, finish="stop", reasoning="", **metrics):
    return Completion(content=content, reasoning=reasoning, finish_reason=finish, truncated=finish == "length",
                      metrics={"cost": 0.001, "tokens_in": 10, "tokens_out": 20, "total": 1.0, **metrics})


@pytest.mark.parametrize("given, spec, judged, expected", [
    (reply(FRAGMENT), NUMBER, False, "sure"),
    (reply("The boat can carry the farmer"), NUMBER, False, "maybe"),
    (reply("Answer: 1"), NUMBER, False, None),
    (reply("", reasoning="thinking"), NUMBER, False, "empty"),              # nothing arrived, with tokens to spare
    (reply("", finish=None), NUMBER, False, "empty"),
    (reply("Answer: 1", finish=None), NUMBER, False, "maybe"),              # the stream just ended
    (reply("The boat can carry the farmer", finish=None), NUMBER, False, "sure"),
    # The token limit: cut off unless the answer is already there.
    (reply("", finish="length", reasoning="thinking"), NUMBER, False, "limit"),
    (reply("So far the count is 12 and", finish="length"), NUMBER, False, "limit"),
    (reply("Answer: 1\n\nLet me double-check that by", finish="length"), NUMBER, False, None),
    (reply("```python\ndef f(x):\n    return x\n```\nThis works beca", finish="length"), {"type": "code", "func": "f"}, False, None),
    (reply("```python\ndef f(x):\n    return", finish="length"), {"type": "code", "func": "f"}, False, "limit"),
    (reply("The total is 659 jars", finish="length"), {"type": "contains", "groups": [["659"]]}, False, None),
    (reply("The total is 683 jars", finish="length"), {"type": "contains", "groups": [["659"]]}, False, "limit"),
    (reply("A finished paragraph. Then more", finish="length"), {"type": "judge", "rubric": "x"}, True, "limit"),
    # Every token used, although the provider did not call it "length": still out of tokens, not a cut.
    (reply("", finish="stop", reasoning="thinking", tokens_out=8192), NUMBER, False, "limit"),
    (reply("", finish=None, reasoning="thinking", tokens_out=8192), NUMBER, False, "limit"),
    (reply("So far the count is 12 and", finish="stop", tokens_out=9000), NUMBER, False, "limit"),
    (reply("Answer: 1", finish="stop", tokens_out=8192), NUMBER, False, None),            # the answer is there
    (reply("", finish="stop", reasoning="thinking", tokens_out=8191), NUMBER, False, "empty"),   # one to spare
    # A count the app had to estimate proves nothing.
    (reply("", finish="stop", reasoning="thinking", tokens_out=9000, tokens_estimated=True), NUMBER, False, "empty"),
])
def test_cut_off_check(given, spec, judged, expected):
    found = benchmark.cut_off_check(given, benchmark.answer_of(given), spec, 8192, judged)
    assert (found[0] if found else None) == expected, found


# ---------------------------------------------------------------- one test, one attempt

def run_test(monkeypatch, replies, test=RIVER, judge=None):
    """Run one test against a scripted provider -> (result, number of requests made, cut-off notices)."""
    queue = list(replies)
    calls = []

    async def fake_complete(target, messages, **kwargs):
        calls.append(target.model)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(providers, "complete", fake_complete)
    cut = []
    result = asyncio.run(benchmark.run_test(test, TARGET, judge, 8192, on_cut_off=cut.append))
    return result, len(calls), cut


def test_a_reply_cut_short_by_the_provider_is_asked_for_again_and_then_graded(monkeypatch):
    result, calls, cut = run_test(monkeypatch, [reply(FRAGMENT), reply("One trip.\n\nAnswer: 1", tokens_out=30)])
    assert calls == 2 and len(cut) == 1 and "ends on ','" in cut[0]
    assert (result["status"], result["score"], result["response"]) == ("pass", 1.0, "One trip.\n\nAnswer: 1")
    assert result["notes"].startswith("Correct: 1")
    assert "[Asked again: the first reply was cut off (it ends on ','" in result["notes"]
    # Both replies were paid for.
    assert (result["cost"], result["tokens_in"], result["tokens_out"], result["attempts"]) == (0.002, 20, 50, 2)
    assert result["finish_reason"] == "stop" and not result["truncated"]


def test_cut_short_twice_is_marked_cut_off_and_left_out_of_the_score(monkeypatch):
    result, calls, _ = run_test(monkeypatch, [reply(FRAGMENT), reply("The farmer takes the goat and")])
    assert calls == 2
    assert (result["status"], result["score"], result["error"]) == ("cut_off", None, None)
    assert result["response"] == "The farmer takes the goat and"            # the last reply is the one kept
    assert result["notes"].startswith("Cut off: it ends mid-sentence, on 'and'")
    assert "Asked twice; the first reply was cut off (it ends on ','" in result["notes"]
    assert "Not counted in the score" in result["notes"] and result["cost"] == 0.002
    assert benchmark.retryable(result)


def test_running_out_of_tokens_scores_zero_at_once_and_is_not_asked_again(monkeypatch):
    """Every test gets one attempt. A model that uses up its tokens has had it."""
    thinking = reply("", finish="length", reasoning="Let me count the crossings again")
    correct = reply("Answer: 1")                                            # would pass, if it were asked for
    result, calls, cut = run_test(monkeypatch, [thinking, correct])
    assert calls == 1 and cut == []
    assert (result["status"], result["score"], result["truncated"]) == ("cut_off", 0.0, True)
    assert result["notes"].startswith("Cut off at the token limit: it ran out of tokens (20 used) while still thinking.")
    assert "Scored 0, and not asked again" in result["notes"] and result["attempts"] == 1
    assert not benchmark.retryable(result)
    # The same when the limit comes in the middle of the answer.
    half = reply("Take the goat over first, then come back for the", finish="length")
    result, calls, _ = run_test(monkeypatch, [half, correct])
    assert calls == 1 and (result["status"], result["score"]) == ("cut_off", 0.0)
    assert "in the middle of its reply" in result["notes"]


def test_a_wrong_answer_is_never_asked_for_again(monkeypatch):
    result, calls, cut = run_test(monkeypatch, [reply("Seven crossings.\n\nAnswer: 7"), reply("Answer: 1")])
    assert calls == 1 and cut == [] and (result["status"], result["score"]) == ("fail", 0.0)
    assert result["notes"] == "Got 7, expected 1" and not benchmark.retryable(result)


def test_a_reply_that_only_might_be_unfinished_is_graded_as_written(monkeypatch):
    """No "Answer:" line and no full stop can be a provider's cut, and it can be the model's own way of
    ending a reply. Asking again would give the model a second attempt, so the reply stands."""
    correct = reply("Answer: 1")
    result, calls, cut = run_test(monkeypatch, [reply("So the farmer needs 1"), correct])
    assert calls == 1 and cut == [] and (result["status"], result["score"]) == ("partial", 0.5)
    assert result["notes"].startswith("Right value (1) but the required 'Answer:' line is missing")
    assert "[The reply may be unfinished (it stops mid-sentence before its 'Answer:' line" in result["notes"]
    assert "it is graded as written.]" in result["notes"]

    result, calls, _ = run_test(monkeypatch, [reply("So the farmer needs 7"), correct])      # a wrong value
    assert calls == 1 and (result["status"], result["score"]) == ("fail", 0.0)
    # A stream that just ended, after a finished answer.
    result, calls, _ = run_test(monkeypatch, [reply("Answer: 1", finish=None), reply("Answer: 7")])
    assert calls == 1 and result["status"] == "pass"
    assert "the stream ended without a finish_reason" in result["notes"]
    # A finished reply says nothing about any of this.
    result, calls, _ = run_test(monkeypatch, [correct])
    assert calls == 1 and result["notes"] == "Correct: 1"


def test_a_reply_that_never_arrived_is_asked_for_once_more(monkeypatch):
    """Nothing arrived, with tokens to spare: that is not an attempt at the test."""
    nothing = reply("", reasoning="The boat takes everything, so one trip")
    result, calls, cut = run_test(monkeypatch, [nothing, reply("Answer: 1")])
    assert calls == 2 and cut == ["nothing arrived: the model thought, then the reply ended"]
    assert result["status"] == "pass" and "[Asked again: the first reply was cut off (nothing arrived" in result["notes"]
    result, calls, _ = run_test(monkeypatch, [reply("", finish=None), reply("Answer: 1")])
    assert calls == 2 and result["status"] == "pass"


def test_a_reply_that_is_empty_twice_scores_zero_and_is_labelled_empty(monkeypatch):
    """One empty reply can be a provider's hiccup, so it is asked for once more. A second one, again with
    tokens to spare, is the model's answer: 0, with its own status, and never a third request."""
    thought = reply("", reasoning="The boat takes everything, so one trip")
    result, calls, cut = run_test(monkeypatch, [thought, thought, reply("Answer: 1")])
    assert calls == 2 and cut == ["nothing arrived: the model thought, then the reply ended"]
    assert (result["status"], result["score"], result["error"]) == ("empty", 0.0, None)
    assert (result["response"], result["reasoning"]) == ("", "The boat takes everything, so one trip")
    assert result["notes"] == (
        "Empty reply, twice: nothing arrived: the model thought, then the reply ended; asked once more, and "
        "again nothing arrived: the model thought, then the reply ended. Scored 0: both requests had tokens to "
        "spare, so the empty reply counts as the model's answer. Not asked a third time, and not run again by "
        "\"Retry failed requests\".")
    assert not result["truncated"] and not benchmark.retryable(result) and not benchmark.reply_kept(result)
    assert (result["cost"], result["tokens_out"], result["attempts"]) == (0.002, 40, 2)     # both were paid for

    # Nothing at all, not even thinking; and a stream that just ended. Empty is empty.
    for first, second in ((reply(""), reply("")), (reply("", finish=None), reply("", reasoning="hmm")),
                          (reply("   \n"), reply("<think>one trip</think>"))):
        result, calls, _ = run_test(monkeypatch, [first, second, reply("Answer: 1")])
        assert calls == 2 and (result["status"], result["score"]) == ("empty", 0.0), result["notes"]
        assert result["notes"].startswith("Empty reply, twice: nothing arrived")

    # The same on a test that goes to a judge: the judge is not asked about nothing.
    async def no_judge(*args, **kwargs):
        raise AssertionError("the judge was asked about an empty reply")

    monkeypatch.setattr(benchmark, "ask_judge", no_judge)
    essay = bench_tests.BY_ID["hard-write-cover"]
    result, calls, _ = run_test(monkeypatch, [thought, thought], test=essay, judge=Target("openrouter", "judge/model"))
    assert calls == 2 and (result["status"], result["score"], result["judge_cost"]) == ("empty", 0.0, None)


def test_an_empty_reply_is_only_the_models_answer_when_it_had_tokens_to_spare_twice(monkeypatch):
    correct = reply("Answer: 1")
    thought = reply("", reasoning="The boat takes everything")
    # Empty once, then a whole reply: that reply is graded, right or wrong.
    result, calls, _ = run_test(monkeypatch, [thought, correct])
    assert calls == 2 and (result["status"], result["score"]) == ("pass", 1.0)
    result, calls, _ = run_test(monkeypatch, [thought, reply("Seven.\n\nAnswer: 7")])
    assert calls == 2 and (result["status"], result["score"]) == ("fail", 0.0)
    # Empty, then out of tokens: cut off at the limit (0), which is a different thing from "empty".
    spent = reply("", finish="length", reasoning="Still counting")
    result, calls, _ = run_test(monkeypatch, [thought, spent, correct])
    assert calls == 2 and (result["status"], result["score"], result["truncated"]) == ("cut_off", 0.0, True)
    assert result["notes"].startswith("Cut off at the token limit")
    result, calls, _ = run_test(monkeypatch, [thought, reply("", reasoning="Still counting", tokens_out=8192), correct])
    assert calls == 2 and (result["status"], result["score"]) == ("cut_off", 0.0)
    # Out of tokens the first time: never asked again, so never "empty".
    result, calls, _ = run_test(monkeypatch, [spent, thought])
    assert calls == 1 and (result["status"], result["score"]) == ("cut_off", 0.0)


def test_two_provider_failures_leave_the_test_unscored_when_a_reply_was_cut_short(monkeypatch):
    """A reply that stops in the middle of a sentence with tokens to spare is the provider's doing. When
    that happened at least once, nothing the model wrote in full has been seen: no score, and "Retry
    failed requests" may ask again. Never a third request. (Empty both times is the model's answer: 0.
    The same fragment twice, word for word, is the model's own ending and is graded as written:
    tests/test_same_reply_twice.py.)"""
    nothing, fragment = reply("", reasoning="The boat takes everything"), reply(FRAGMENT)
    shorter = reply(FRAGMENT.rsplit(" ", 1)[0])           # cut a word earlier: a stream is not dropped at the same word twice
    assert shorter.content != FRAGMENT and shorter.content.endswith(",")
    for first, second in ((fragment, nothing), (nothing, fragment), (fragment, shorter)):
        result, calls, _ = run_test(monkeypatch, [first, second, reply("Answer: 1")])
        assert calls == 2 and (result["status"], result["score"]) == ("cut_off", None), result["notes"]
        assert result["notes"].startswith("Cut off: ") and "Asked twice; the first reply was cut off" in result["notes"]
        assert "Not counted in the score" in result["notes"] and benchmark.retryable(result)
        assert result["cost"] == 0.002 and result["attempts"] == 2


def test_using_every_token_is_out_of_tokens_whatever_the_provider_calls_it(monkeypatch):
    """Seen in review: a provider can end the stream with "stop" (or with nothing) after the model has
    used its whole limit. That reply has no tokens to spare, so it is not asked for again."""
    correct = reply("Answer: 1")
    for finish in ("stop", None):
        spent = reply("", finish=finish, reasoning="Still counting crossings", tokens_out=8192)
        result, calls, cut = run_test(monkeypatch, [spent, correct])
        assert calls == 1 and cut == [], finish
        assert (result["status"], result["score"], result["truncated"]) == ("cut_off", 0.0, True)
        assert "it ran out of tokens (8,192 used) while still thinking" in result["notes"]
        assert not benchmark.retryable(result)
    # With tokens to spare the same empty reply is the provider's failure, and is asked for again.
    spare = reply("", finish="stop", reasoning="Still counting crossings", tokens_out=900)
    result, calls, _ = run_test(monkeypatch, [spare, correct])
    assert calls == 2 and result["status"] == "pass"


def test_the_second_request_is_the_tests_one_attempt(monkeypatch):
    """After a provider cut the next reply counts, whatever it is: there is never a third request."""
    spare = reply("Answer: 1")
    result, calls, _ = run_test(monkeypatch, [reply(FRAGMENT), reply("", finish="length", reasoning="hmm"), spare])
    assert calls == 2 and (result["status"], result["score"]) == ("cut_off", 0.0)
    assert result["notes"].startswith("Cut off at the token limit")
    assert "[Asked again before that: the first reply was cut off (it ends on ','" in result["notes"]
    assert not benchmark.retryable(result)

    result, calls, _ = run_test(monkeypatch, [reply(FRAGMENT), reply("Seven.\n\nAnswer: 7"), spare])
    assert calls == 2 and (result["status"], result["score"]) == ("fail", 0.0)
    result, calls, _ = run_test(monkeypatch, [reply(FRAGMENT), reply("So the farmer needs 1"), spare])
    assert calls == 2 and (result["status"], result["score"]) == ("partial", 0.5)
    assert "it is graded as written" in result["notes"]


def test_when_asking_again_fails_the_test_is_left_unscored(monkeypatch):
    error = ProviderError("Rate limited (429).", status=429, code="rate_limited", retryable=True)
    result, calls, _ = run_test(monkeypatch, [reply(FRAGMENT), error])
    assert calls == 2 and (result["status"], result["score"]) == ("cut_off", None)
    assert "Asking again failed (Rate limited (429).)" in result["notes"] and result["cost"] == 0.001
    assert benchmark.retryable(result)
    result, calls, _ = run_test(monkeypatch, [reply("", reasoning="hmm"), error])
    assert calls == 2 and (result["status"], result["score"]) == ("cut_off", None)
    assert result["notes"].startswith("Cut off: nothing arrived")
    # The second request gave up after four tries: all five requests are counted.
    error.attempts = 4
    result, _, _ = run_test(monkeypatch, [reply(FRAGMENT), error])
    assert result["attempts"] == 5


def test_a_cut_off_reply_is_not_sent_to_the_judge(monkeypatch):
    async def no_judge(*args, **kwargs):
        raise AssertionError("the judge was asked about an unfinished reply")

    monkeypatch.setattr(benchmark, "ask_judge", no_judge)
    essay = bench_tests.BY_ID["hard-write-cover"]
    half = reply("Dear Hiring Manager, I am applying for the role and I bring three years of", finish="length")
    result, calls, _ = run_test(monkeypatch, [half, half], test=essay, judge=Target("openrouter", "judge/model"))
    assert calls == 1 and (result["status"], result["score"], result["judge_cost"]) == ("cut_off", 0.0, None)


def test_only_provider_failures_can_be_retried():
    def row(status, score, error=None, response="some text"):
        return {"status": status, "score": score, "error": error, "response": response}

    assert benchmark.retryable(row("error", None, "Provider error (503)."))
    assert benchmark.retryable(row("cut_off", None))                        # cut short by the provider
    assert not benchmark.retryable(row("cut_off", 0.0))                     # out of tokens
    assert not benchmark.retryable(row("empty", 0.0, response=""))          # empty twice: the model's answer
    for status, score in (("fail", 0.0), ("partial", 0.5), ("pass", 1.0)):
        assert not benchmark.retryable(row(status, score))
    # Only the grading failed, and the model's reply is there: it is graded again, and the model is not asked.
    assert benchmark.reply_kept(row("error", None, "Judge error: Rate limited (429)."))
    assert benchmark.reply_kept(row("error", None, "Internal error while grading (RuntimeError: boom)"))
    assert not benchmark.reply_kept(row("error", None, "Provider error (503)."))
    assert not benchmark.reply_kept(row("error", None, "Judge error: Rate limited (429).", response=""))
    assert not benchmark.reply_kept(row("fail", 0.0, "Judge error: x"))


def test_when_only_the_judge_failed_the_judge_is_asked_again_and_the_model_is_not(monkeypatch):
    essay = bench_tests.BY_ID["hard-write-cover"]
    judge = Target("openrouter", "judge/model")
    verdicts = [ProviderError("Rate limited (429).", status=429, code="rate_limited", retryable=True),
                Completion(content='{"deductions": ["-1: a flat ending"], "score": 9, "reason": "Specific."}',
                           finish_reason="stop", metrics={"cost": 0.0005})]

    async def scripted_judge(target, messages):
        item = verdicts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(benchmark, "ask_judge", scripted_judge)
    result, calls, _ = run_test(monkeypatch, [reply(essay["mock_good"])], test=essay, judge=judge)
    assert calls == 1 and (result["status"], result["score"]) == ("error", None)
    assert result["error"] == "Judge error: Rate limited (429)." and result["response"] == essay["mock_good"]
    assert benchmark.retryable(result) and benchmark.reply_kept(result)

    async def no_model(*args, **kwargs):
        raise AssertionError("the model was asked a second time")

    monkeypatch.setattr(providers, "complete", no_model)
    again = benchmark.carry_over(result, asyncio.run(benchmark.grade_again(essay, result, judge)))
    assert (again["status"], again["score"], again["error"]) == ("pass", 0.9, None)
    assert again["response"] == essay["mock_good"] and again["notes"].startswith("Judge: 9/10")
    assert again["notes"].endswith('[Graded again by "Retry failed requests": before, the judge call failed '
                                   "(Rate limited (429).). The model was not asked again.]")
    # The model's request is counted once; the judge's verdict is what was added.
    assert (again["cost"], again["tokens_out"], again["attempts"], again["judge_cost"]) == (0.001, 20, 1, 0.0005)


def test_a_judge_verdict_that_cannot_be_read_is_still_counted_in_the_cost(monkeypatch):
    essay = bench_tests.BY_ID["hard-write-cover"]

    async def rambling_judge(target, messages):
        return Completion(content="I would say this is a solid effort overall.", finish_reason="stop",
                          metrics={"cost": 0.0007})

    monkeypatch.setattr(benchmark, "ask_judge", rambling_judge)
    result, calls, _ = run_test(monkeypatch, [reply(essay["mock_good"])], test=essay,
                                judge=Target("openrouter", "judge/model"))
    assert calls == 1 and result["status"] == "error" and result["score"] is None
    assert result["error"] == "Judge error: The judge did not return the requested JSON score."
    assert result["judge_cost"] == 0.0007 and result["response"] == essay["mock_good"]


def test_a_bug_in_a_grader_keeps_the_reply_so_the_model_is_not_asked_twice(monkeypatch):
    def broken(spec, answer, cut_off=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(graders, "grade", broken)
    result, calls, _ = run_test(monkeypatch, [reply("One trip.\n\nAnswer: 1")])
    assert calls == 1 and (result["status"], result["score"]) == ("error", None)
    assert result["error"] == "Internal error while grading (RuntimeError: boom)"
    assert (result["response"], result["cost"], result["tokens_out"]) == ("One trip.\n\nAnswer: 1", 0.001, 20)
    assert benchmark.retryable(result) and benchmark.reply_kept(result)

    monkeypatch.undo()                                             # the grader is repaired

    async def no_model(*args, **kwargs):
        raise AssertionError("the model was asked a second time")

    monkeypatch.setattr(providers, "complete", no_model)
    again = benchmark.carry_over(result, asyncio.run(benchmark.grade_again(RIVER, result, None)))
    assert (again["status"], again["score"], again["error"]) == ("pass", 1.0, None)
    assert again["notes"] == ('Correct: 1 [Graded again by "Retry failed requests": before, grading failed '
                              "(Internal error while grading (RuntimeError: boom)). The model was not asked again.]")
    assert (again["cost"], again["attempts"]) == (0.001, 1)


# ---------------------------------------------------------------- scoring

def test_scoring_counts_a_token_limit_cut_off_as_zero_and_leaves_a_provider_cut_out():
    def row(status, score, category="math"):
        return {"category": category, "status": status, "score": score, "latency": 1.0, "ttft": 0.2, "tps": 50.0,
                "cost": 0.001, "tokens_in": 10, "tokens_out": 20}

    summary = db.summarize([row("pass", 1), row("pass", 1), row("cut_off", None), row("cut_off", 0.0),
                            row("cut_off", None, "coding")])
    assert summary["category_scores"] == {"math": 66.7}                    # (1 + 1 + 0) / 3; coding has no score
    assert (summary["n_pass"], summary["n_fail"], summary["n_cut_off"], summary["n_error"]) == (2, 0, 3, 0)
    assert summary["total_cost"] == pytest.approx(0.005)                   # every request counts towards the cost
    assert summary["n_empty"] == 0

    # A reply that was empty twice counts as 0, under its own heading: it is not a "fail" and not a "cut off".
    summary = db.summarize([row("pass", 1), row("empty", 0.0), row("empty", 0.0, "coding"), row("fail", 0.0, "coding")])
    assert summary["category_scores"] == {"math": 50.0, "coding": 0.0} and summary["score"] == 25.0
    assert (summary["n_pass"], summary["n_fail"], summary["n_empty"], summary["n_cut_off"]) == (1, 1, 2, 0)


# ---------------------------------------------------------------- whole runs against the mock

def test_a_run_recovers_from_replies_cut_short_once(client):
    stream, run = run_benchmark(client, "mock/hiccup")
    cut = [e for e in stream if e["type"] == "cut_off"]
    assert len(cut) == 5 and all("ends on ','" in e["message"] for e in cut)
    assert [r["status"] for r in run["results"]] == ["pass"] * 5
    assert all(r["attempts"] == 2 and "[Asked again: the first reply was cut off" in r["notes"] for r in run["results"])
    assert (run["score"], run["n_cut_off"], run["n_pass"]) == (100.0, 0, 5)
    for test_id in (t["id"] for t in bench_tests.MATH):
        order = [e["type"] for e in stream if e.get("test_id") == test_id or e.get("result", {}).get("test_id") == test_id]
        assert order == ["test_start", "cut_off", "test_done"]
    # Nothing was retried afterwards: the first pass is the final score.
    assert (run["first_pass"]["score"], run["retries"]) == (100.0, 0)


def test_retry_failed_requests_re_runs_the_cut_off_tests_and_updates_the_score(client):
    _, run = run_benchmark(client, "mock/dropper", categories=["math", "knowledge"])
    assert [r["status"] for r in run["results"]] == ["cut_off"] * 10
    assert (run["status"], run["score"], run["n_cut_off"]) == ("completed", None, 10)
    assert all(r["score"] is None and r["finish_reason"] == "stop" for r in run["results"])
    assert run["id"] not in [r["id"] for r in client.get("/api/runs").json()["runs"]]     # nothing scored yet
    assert (run["first_pass"]["score"], run["first_pass"]["n_cut_off"], run["retries"]) == (None, 10, 0)
    before = {r["test_id"]: r for r in run["results"]}

    stream, after = retry(client, run["id"])
    start = stream[0]
    assert start["type"] == "start" and start["retry"] is True and start["total"] == 10
    assert [t["test_id"] for t in start["tests"]] == [r["test_id"] for r in run["results"]] and start["results"] == []
    assert sum(1 for e in stream if e["type"] == "test_done") == 10
    assert [r["status"] for r in after["results"]] == ["pass"] * 10
    assert [r["id"] for r in after["results"]] == [r["id"] for r in run["results"]]      # same rows, same order
    assert (after["status"], after["score"], after["n_cut_off"], after["n_pass"]) == ("completed", 100.0, 0, 10)
    assert stream[-1]["run"]["score"] == 100.0
    for result in after["results"]:
        old = before[result["test_id"]]
        assert result["notes"].endswith('[Re-run by "Retry failed requests": before, this test was cut short '
                                        "by the provider.]")
        assert result["cost"] > old["cost"] and result["attempts"] == old["attempts"] + 1   # earlier cost is kept
        assert result["tokens_out"] > old["tokens_out"]
    assert after["total_cost"] == pytest.approx(sum(r["cost"] for r in after["results"]))
    assert run["id"] in [r["id"] for r in client.get("/api/runs").json()["runs"]]
    # The first pass is kept as it was, next to the final score.
    assert after["first_pass"] == run["first_pass"] and after["retries"] == 1
    assert stream[-1]["run"]["first_pass"]["score"] is None and stream[-1]["run"]["retries"] == 1

    again = client.post(f"/api/runs/{run['id']}/retry", json={})                          # nothing is left
    assert again.status_code == 409 and again.json()["code"] == "nothing_to_retry"
    assert client.get(f"/api/runs/{run['id']}").json()["run"]["retries"] == 1             # a refusal is not a retry


def test_retry_failed_requests_re_runs_api_errors(client):
    stream, run = run_benchmark(client, "mock/outage", categories=["knowledge"])
    assert [r["status"] for r in run["results"]] == ["error"] * 5 and run["score"] is None
    _, after = retry(client, run["id"])
    assert [r["status"] for r in after["results"]] == ["pass"] * 5 and after["score"] == 100.0
    assert (after["n_error"], after["n_pass"]) == (0, 5)
    assert all("before, this test was an API error (Rate limited (429)" in r["notes"] for r in after["results"])
    assert (after["first_pass"]["score"], after["first_pass"]["n_error"], after["retries"]) == (None, 5, 1)


def test_the_first_pass_score_is_listed_and_exported_next_to_the_final_one(client):
    _, run = run_benchmark(client, "mock/smart", categories=["math", "knowledge"])
    # One request of the first pass ended in an API error; the nine others were graded.
    with db.connect() as conn:
        conn.execute("UPDATE results SET status = 'error', score = NULL, error = 'Service overloaded (503)' "
                     "WHERE run_id = ? AND test_id = 'math-change'", (run["id"],))
        conn.execute("UPDATE runs SET first_pass = NULL WHERE id = ?", (run["id"],))
    first = db.finalize_run(run["id"], "completed")
    assert first["first_pass"]["n_error"] == 1 and first["retries"] == 0
    _, after = retry(client, run["id"])
    assert after["first_pass"] == first["first_pass"] and after["retries"] == 1 and after["n_error"] == 0
    assert set(after["first_pass"]) == {"score", "grade", "category_scores", "n_pass", "n_partial", "n_fail",
                                        "n_error", "n_cut_off", "n_empty"}

    listed = next(r for r in client.get("/api/runs").json()["runs"] if r["id"] == run["id"])
    assert listed["first_pass"] == after["first_pass"] and listed["retries"] == 1
    header, *rows = client.get("/api/runs/export?format=csv").text.lstrip("﻿").splitlines()
    columns = header.split(",")
    line = next(r for r in rows if r.startswith(f"{run['id']},")).split(",")
    assert line[columns.index("first_pass_score")] == str(after["first_pass"]["score"])
    assert line[columns.index("retries")] == "1" and float(line[columns.index("score")]) == after["score"]
    exported = next(r for r in client.get("/api/runs/export?format=json").json() if r["id"] == run["id"])
    assert exported["first_pass"]["score"] == after["first_pass"]["score"]


def test_retry_only_touches_the_tests_the_provider_failed(client):
    _, run = run_benchmark(client, "mock/smart", categories=["math", "knowledge"])
    assert run["n_error"] == 0 and run["n_cut_off"] == 0
    broken = {"math-change": ("error", None), "know-capital": ("cut_off", None)}
    # Not the provider's doing, so never run again: out of tokens (cut off, 0) and a wrong answer.
    spent = {"math-speed": ("cut_off", 0.0), "math-garden": ("fail", 0.0)}
    assert set(broken) | set(spent) <= {r["test_id"] for r in run["results"]}
    with db.connect() as conn:
        for test_id, (status, score) in {**broken, **spent}.items():
            conn.execute("UPDATE results SET status = ?, score = ?, error = ?, cost = 0.5 WHERE run_id = ? AND test_id = ?",
                         (status, score, "Service overloaded (503)" if status == "error" else None, run["id"], test_id))
    db.finalize_run(run["id"], "completed")
    damaged = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert (damaged["n_error"], damaged["n_cut_off"], damaged["n_fail"]) == (1, 2, 1)

    stream, after = retry(client, run["id"])
    assert sorted(e["test_id"] for e in stream if e["type"] == "test_start") == sorted(broken)
    assert len(stream[0]["results"]) == 8 and stream[0]["total"] == 2
    untouched = [r for r in after["results"] if r["test_id"] not in broken]
    assert untouched == [r for r in damaged["results"] if r["test_id"] not in broken]
    assert {r["test_id"]: (r["status"], r["score"]) for r in untouched if r["test_id"] in spent} == spent
    redone = {r["test_id"]: r for r in after["results"] if r["test_id"] in broken}
    assert all(r["status"] in ("pass", "partial", "fail") and r["cost"] > 0.5 for r in redone.values())
    assert "before, this test was an API error (Service overloaded (503))" in redone["math-change"]["notes"]
    assert "before, this test was cut short by the provider" in redone["know-capital"]["notes"]
    assert (after["n_error"], after["n_cut_off"], after["n_fail"]) == (0, 1, 1)


def test_a_token_limit_cut_off_scores_zero_and_cannot_be_retried(client):
    stream, run = run_benchmark(client, "mock/cutoff")
    assert [(r["status"], r["score"], r["truncated"]) for r in run["results"]] == [("cut_off", 0.0, True)] * 5
    assert (run["score"], run["n_cut_off"], run["n_fail"]) == (0.0, 5, 0)
    assert all(r["attempts"] == 1 for r in run["results"])                 # asked once, never again
    assert not [e for e in stream if e["type"] == "cut_off"]
    refused = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "nothing_to_retry"
    assert "ran out of tokens" in refused.json()["message"]
    assert client.get(f"/api/runs/{run['id']}").json()["run"]["results"] == run["results"]


def test_a_model_that_answers_with_nothing_scores_zero_on_every_test_and_cannot_be_retried(client):
    """mock/mute thinks and then writes nothing, every time."""
    stream, run = run_benchmark(client, "mock/mute", categories=["math", "writing"], judge=JUDGE)
    assert [(r["status"], r["score"], r["attempts"]) for r in run["results"]] == [("empty", 0.0, 2)] * 9
    assert all(r["response"] == "" and r["reasoning"] and r["finish_reason"] == "stop" and not r["truncated"]
               for r in run["results"])
    assert all(r["notes"].startswith("Empty reply, twice: nothing arrived: the model thought") for r in run["results"])
    assert (run["score"], run["category_scores"]) == (0.0, {"math": 0.0, "writing": 0.0})
    assert (run["n_empty"], run["n_cut_off"], run["n_fail"], run["n_error"], run["n_pass"]) == (9, 0, 0, 0, 0)
    assert (run["first_pass"]["score"], run["first_pass"]["n_empty"], run["retries"]) == (0.0, 9, 0)
    assert run["judge_cost"] is None                                       # the judge was never asked
    # Each test was asked for exactly twice, and the page was told why.
    asked_again = [e for e in stream if e["type"] == "cut_off"]
    assert len(asked_again) == 9 and all(e["message"].startswith("nothing arrived") for e in asked_again)
    # It is listed and ranked like any scored run, and there is nothing to retry.
    assert run["id"] in [r["id"] for r in client.get("/api/runs").json()["runs"]]
    refused = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "nothing_to_retry"
    assert "came back empty twice" in refused.json()["message"]
    header, *rows = client.get("/api/runs/export?format=csv").text.lstrip("﻿").splitlines()
    line = next(r for r in rows if r.startswith(f"{run['id']},")).split(",")
    assert line[header.split(",").index("empty")] == "9" and line[header.split(",").index("cut_off")] == "0"


def test_a_single_empty_reply_gets_one_more_request(client):
    """mock/mute-once sends nothing the first time it is asked something, and answers the second time."""
    stream, run = run_benchmark(client, "mock/mute-once")
    assert [(r["status"], r["attempts"]) for r in run["results"]] == [("pass", 2)] * 5
    assert all("[Asked again: the first reply was cut off (nothing arrived: the model thought" in r["notes"]
               for r in run["results"])
    assert (run["score"], run["n_empty"], run["n_cut_off"], run["n_pass"]) == (100.0, 0, 0, 5)
    assert len([e for e in stream if e["type"] == "cut_off"]) == 5


def test_retry_uses_the_judge_of_the_run(client):
    _, run = run_benchmark(client, "mock/dropper", categories=["writing"], judge=JUDGE)
    assert [r["status"] for r in run["results"]] == ["cut_off"] * 4
    stream, after = retry(client, run["id"])
    assert stream[0]["judge_model"] == "mock/average"
    assert all(r["status"] in ("pass", "partial", "fail") and r["notes"].startswith("Judge:") for r in after["results"])
    assert all(r["judge_cost"] is not None for r in after["results"]) and after["judge_cost"] > 0


def test_a_run_whose_judge_failed_is_judged_again_without_asking_the_model(client, monkeypatch):
    """mock/outage fails four times per prompt: the judge gives up in the run and answers in the retry."""
    asked = []
    real_complete = providers.complete

    async def counting(target, messages, **kwargs):
        asked.append(target.model)
        return await real_complete(target, messages, **kwargs)

    monkeypatch.setattr(providers, "complete", counting)
    _, run = run_benchmark(client, "mock/smart", categories=["writing"],
                           judge={"provider": "openrouter", "model": "mock/outage"})
    assert [r["status"] for r in run["results"]] == ["error"] * 4 and run["score"] is None
    assert all(r["error"].startswith("Judge error: Rate limited (429)") and r["response"] for r in run["results"])
    assert asked.count("mock/smart") == 4
    before = {r["test_id"]: r for r in run["results"]}

    stream, after = retry(client, run["id"])
    assert asked.count("mock/smart") == 4                                   # the model was not asked again
    assert [e["type"] for e in stream if e.get("test_id") == "write-product"] == ["test_start", "judging"]
    for result in after["results"]:
        old = before[result["test_id"]]
        assert result["status"] in ("pass", "partial", "fail") and result["notes"].startswith("Judge:")
        assert "[Graded again by" in result["notes"] and "The model was not asked again.]" in result["notes"]
        assert (result["response"], result["cost"], result["tokens_out"], result["attempts"]) == (
            old["response"], old["cost"], old["tokens_out"], old["attempts"])
        assert result["judge_cost"] is not None
    assert after["score"] is not None and (after["n_error"], after["retries"]) == (0, 1)


def test_retry_is_refused_when_it_cannot_be_done(client, monkeypatch):
    assert client.post("/api/runs/987654/retry", json={}).status_code == 404
    _, clean = run_benchmark(client, "mock/smart")
    refused = client.post(f"/api/runs/{clean['id']}/retry", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "nothing_to_retry"

    _, run = run_benchmark(client, "mock/dropper")
    assert run["n_cut_off"] == 5
    # The tests changed since the run: re-running some of them would mix two test sets in one score.
    monkeypatch.setattr(graders, "GRADER_VERSION", graders.GRADER_VERSION + 1)
    refused = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert refused.status_code == 409 and refused.json()["code"] == "older_tests"
    monkeypatch.undo()
    assert [r["status"] for r in retry(client, run["id"])[1]["results"]] == ["pass"] * 5


def test_only_one_job_runs_at_a_time_including_a_retry(client, monkeypatch):
    _, run = run_benchmark(client, "mock/dropper")
    monkeypatch.setenv("MOCK_FAST", "0")
    started = client.post("/api/benchmark/run", json={"target": {"provider": "openrouter", "model": "mock/slow"},
                                                      "categories": ["math"], "concurrency": 1}).json()
    busy = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert busy.status_code == 409 and busy.json()["code"] == "busy" and busy.json()["run_id"] == started["run_id"]
    assert client.post(f"/api/benchmark/{started['run_id']}/cancel").json() == {"ok": True}
    deadline = time.time() + 15
    while client.get("/api/benchmark/active").json()["job"] and time.time() < deadline:
        time.sleep(0.05)
    assert client.get("/api/benchmark/active").json()["job"] is None


def test_a_cancelled_retry_keeps_what_it_finished_and_the_runs_status(client, monkeypatch):
    _, run = run_benchmark(client, "mock/dropper")
    monkeypatch.setenv("MOCK_FAST", "0")
    started = client.post(f"/api/runs/{run['id']}/retry", json={})
    assert started.status_code == 200
    assert client.get("/api/benchmark/active").json()["job"]["retry"] is True
    assert client.post(f"/api/benchmark/{run['id']}/cancel").json() == {"ok": True}
    stream, after = follow(client, run["id"])
    assert after["status"] == "completed"                                   # what the run was, not "cancelled"
    statuses = [r["status"] for r in after["results"]]
    assert len(statuses) == 5 and set(statuses) <= {"pass", "cut_off"} and "cut_off" in statuses


def test_database_from_before_cut_offs_is_upgraded(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    db.init_db()
    with db.connect() as conn:
        for column in ("n_cut_off", "first_pass", "retries", "settings", "n_empty"):
            conn.execute(f"ALTER TABLE runs DROP COLUMN {column}")
        conn.execute("ALTER TABLE results DROP COLUMN finish_reason")
        conn.execute("INSERT INTO runs (created_at, status, provider, model, score) VALUES (1, 'completed', 'openrouter', 'm', 50)")
    db.init_db()
    columns = sqlite3.connect(tmp_path / "modellab.mock.db")
    names = [row[1] for row in columns.execute("PRAGMA table_info(runs)")]
    assert {"n_cut_off", "first_pass", "retries", "settings", "n_empty"} <= set(names)
    assert "finish_reason" in [row[1] for row in columns.execute("PRAGMA table_info(results)")]
    columns.close()
    old = db.list_runs(include_older=True)[0]
    # A run saved before first-pass scores were recorded has none: it is not made up from the final score.
    assert (old["n_cut_off"], old["first_pass"], old["retries"], old["n_empty"]) == (0, None, 0, 0)


def test_a_retry_keeps_the_runs_score_in_step_after_every_test(client):
    """If the process dies in the middle of a retry, the run must not be left with the score it had
    before: its results were replaced one by one, and a later retry would find nothing to do."""
    _, run = run_benchmark(client, "mock/dropper", categories=["math", "knowledge"], concurrency=1)
    assert (run["score"], run["n_cut_off"]) == (None, 10)
    seen = []
    real_replace = db.replace_result

    def replace_and_look(run_id, result):
        done = real_replace(run_id, result)
        seen.append(db.get_run(run_id, with_results=False))        # what the run row says before this test's summary
        return done

    real_finalize = db.finalize_run
    after_each = []

    def finalize_and_look(run_id, status):
        saved = real_finalize(run_id, status)
        after_each.append((saved["n_pass"], saved["n_cut_off"], saved["score"], saved["status"]))
        return saved

    db.replace_result, db.finalize_run = replace_and_look, finalize_and_look
    try:
        retry(client, run["id"])
    finally:
        db.replace_result, db.finalize_run = real_replace, real_finalize
    # After the first re-run test the saved run already counts it, and so on to the last.
    assert [row[:2] for row in after_each[:10]] == [(n, 10 - n) for n in range(1, 11)]
    assert all(row[3] == "completed" for row in after_each) and after_each[0][2] == 100.0
    assert [row["n_pass"] for row in seen] == list(range(10))      # each look came before that test's summary
    final = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert (final["score"], final["n_pass"], final["first_pass"]["score"], final["retries"]) == (100.0, 10, None, 1)

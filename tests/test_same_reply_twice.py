"""A reply that looks cut short and comes back the same, word for word, when it is asked for again.

Found live on 2026-10-08 (runs #28, #32 and #33): llama-3.1-8b answered the lipogram test with three
clauses that each end on a comma, finish_reason "stop", 74 tokens. The same 348 characters came back
all twelve times it was asked (twice per run, twice per "Retry failed requests"), and each time the
test was left unscored as "cut short by the provider". A provider that drops the end of a stream does
not drop it at the same word twice: that ending is the model's own, and the reply is graded.
"""
import asyncio

from app import bench_tests, benchmark, providers
from app.providers import Completion, ProviderError, Target

TARGET = Target("openrouter", "some/model", "key")
LIPOGRAM = bench_tests.BY_ID["hard-instr-lipogram"]
# The reply as it was saved in run #28.
LLAMA_LIPOGRAM = (
    "Salt water laps gently against sandy shores, its soothing sound calming the mind, as the warm sand "
    "feels soft beneath bare feet, a perfect place to relax, \n"
    "wind whispers secrets to the trees, rustling their leaves, a gentle melody that echoes through the air, \n"
    "moonlight casts a silver glow on the calm ocean, its peaceful beauty a sight to behold,")
CONTAINS = {"id": "t-contains", "category": "longctx", "name": "Ledger", "prompt": "What is the total?",
            "grader": {"type": "contains", "groups": [["659"]], "word": False}}


def reply(text: str, tokens: int = 74, cost: float = 0.00001, finish: str | None = "stop") -> Completion:
    return Completion(content=text, finish_reason=finish,
                      metrics={"tokens_in": 100, "tokens_out": tokens, "cost": cost, "total": 1.0})


def run(monkeypatch, test: dict, replies: list) -> tuple[dict, int]:
    """Run one test against a model that answers with `replies`, in order -> (result, requests made)."""
    queue = list(replies)
    asked = []

    async def complete(target, messages, **kwargs):
        asked.append(messages)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(providers, "complete", complete)
    result = asyncio.run(benchmark.run_test(test, TARGET, None, 32768))
    return result, len(asked)


def test_the_live_lipogram_reply_is_graded_when_it_comes_back_the_same(monkeypatch):
    result, asked = run(monkeypatch, LIPOGRAM, [reply(LLAMA_LIPOGRAM), reply(LLAMA_LIPOGRAM)])
    assert asked == 2                                   # still asked once more: one cut can be the provider's
    assert result["status"] == "fail" and result["score"] == 0
    assert "no letter e" in result["notes"]             # graded on the rules of the test
    assert "came back the same, word for word" in result["notes"]
    assert "it ends on ','" in result["notes"]
    assert "Not counted in the score" not in result["notes"]
    assert not benchmark.retryable(result)              # it had its attempt: no retry asks a third time
    assert result["response"] == LLAMA_LIPOGRAM


def test_both_requests_are_counted(monkeypatch):
    result, _ = run(monkeypatch, LIPOGRAM, [reply(LLAMA_LIPOGRAM, 74, 0.00001), reply(LLAMA_LIPOGRAM, 74, 0.00002)])
    assert result["attempts"] == 2
    assert result["tokens_out"] == 148 and result["tokens_in"] == 200
    assert abs(result["cost"] - 0.00003) < 1e-12


def test_a_reply_that_is_right_and_ends_on_a_comma_twice_passes(monkeypatch):
    result, asked = run(monkeypatch, CONTAINS, [reply("251 + 318 + 90 = 659,"), reply("251 + 318 + 90 = 659,")])
    assert asked == 2
    assert result["status"] == "pass" and result["score"] == 1
    assert "came back the same, word for word" in result["notes"]


def test_spaces_around_the_reply_do_not_make_it_another_reply(monkeypatch):
    result, _ = run(monkeypatch, CONTAINS, [reply("The total is 659,"), reply("\nThe total is 659, \n")])
    assert result["status"] == "pass"
    assert "came back the same, word for word" in result["notes"]


def test_cut_short_twice_at_different_places_is_still_the_provider(monkeypatch):
    result, asked = run(monkeypatch, CONTAINS, [reply("The ledger has three entries,"),
                                                reply("The ledger has three entries: 251,")])
    assert asked == 2
    assert result["status"] == "cut_off" and result["score"] is None
    assert "Not counted in the score" in result["notes"]
    assert "came back the same" not in result["notes"]
    assert benchmark.retryable(result)


def test_a_right_answer_cut_short_at_two_places_is_not_graded(monkeypatch):
    # The value is there both times, but the replies differ: nothing says where the model would have ended.
    result, _ = run(monkeypatch, CONTAINS, [reply("The total is 659, because"), reply("The total is 659, and the")])
    assert result["status"] == "cut_off" and result["score"] is None


def test_cut_short_then_whole_is_graded_on_the_second_reply(monkeypatch):
    result, asked = run(monkeypatch, CONTAINS, [reply("The total is 659,"), reply("The total is 659.")])
    assert asked == 2
    assert result["status"] == "pass"
    assert "Asked again: the first reply was cut off" in result["notes"]
    assert "came back the same" not in result["notes"]


def test_cut_short_and_then_the_request_fails_is_left_unscored(monkeypatch):
    # No second reply arrived, so there is nothing to compare the first one with.
    failure = ProviderError("Provider error (503).", status=503, retryable=True)
    result, asked = run(monkeypatch, CONTAINS, [reply("The total is 659,"), failure])
    assert asked == 2
    assert result["status"] == "cut_off" and result["score"] is None
    assert "Asking again failed" in result["notes"]
    assert benchmark.retryable(result)


def test_the_same_unclosed_code_block_twice_is_graded(monkeypatch):
    code = {"id": "t-code", "category": "coding", "name": "Double", "prompt": "Write double(n).",
            "grader": {"type": "code", "func": "double", "cases": [{"args": [2], "expected": 4}]}}
    text = "```python\ndef double(n):\n    return n * 2\n"
    result, asked = run(monkeypatch, code, [reply(text), reply(text)])
    assert asked == 2
    assert result["status"] == "pass" and result["score"] == 1
    assert "its code block is never closed" in result["notes"]
    assert "came back the same, word for word" in result["notes"]


def test_empty_twice_is_still_an_empty_reply(monkeypatch):
    result, asked = run(monkeypatch, CONTAINS, [reply("", 0), reply("", 0)])
    assert asked == 2
    assert result["status"] == benchmark.EMPTY and result["score"] == 0


def test_a_reply_that_ends_on_a_full_stop_is_asked_for_once(monkeypatch):
    result, asked = run(monkeypatch, CONTAINS, [reply("The total is 659.")])
    assert asked == 1 and result["status"] == "pass"
    assert "came back the same" not in result["notes"]


def test_same_reply_twice_needs_a_sure_sign_both_times():
    sure = ("sure", "it ends on ','")
    assert benchmark.same_reply_twice(sure, "a, b,", sure, "a, b,")
    assert benchmark.same_reply_twice(sure, "a, b,", ("sure", "it ends on ',': another tail"), " a, b, ")
    assert not benchmark.same_reply_twice(sure, "a, b,", sure, "a, b, c,")
    assert not benchmark.same_reply_twice(sure, "a, b,", None, "a, b,")
    assert not benchmark.same_reply_twice(None, "a, b,", sure, "a, b,")
    assert not benchmark.same_reply_twice(("maybe", "no ending"), "a b", ("maybe", "no ending"), "a b")
    assert not benchmark.same_reply_twice(("empty", "nothing arrived"), "", ("empty", "nothing arrived"), "")
    assert not benchmark.same_reply_twice(sure, "a, b,", ("limit", "it ran out of tokens"), "a, b,")

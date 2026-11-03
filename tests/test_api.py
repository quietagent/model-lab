"""End-to-end API tests against the mock provider."""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import bench_tests, benchmark, db, graders, providers
from app.main import app
from app.providers import Target

ALL = [c for c in bench_tests.CATEGORY_IDS if c != "custom"]
SMART = {"provider": "openrouter", "model": "mock/smart"}
JUDGE = {"provider": "openrouter", "model": "mock/average"}


@pytest.fixture(scope="module")
def client():
    # The API only answers requests addressed to localhost on its own port (see app/security.py).
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def events(response):
    assert response.status_code == 200
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def run_benchmark(client, **body):
    """Start a benchmark job, follow its events to the end -> (events, the saved run)."""
    payload = {"target": SMART, "categories": ALL, "concurrency": 4, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    stream = [e for e in events(client.get(f"/api/benchmark/{started.json()['run_id']}/events"))
              if e["type"] != "hello"]
    return stream, next(e["run"] for e in stream if e["type"] == "summary")


def test_index_and_config(client):
    assert client.get("/").status_code == 200
    config = client.get("/api/config").json()
    assert config["mock"] is True
    assert [p["id"] for p in config["providers"]] == ["openrouter", "huggingface"]


def test_models(client):
    models = client.get("/api/models").json()["models"]
    assert any(m["id"] == "mock/free:free" and m["free"] for m in models)
    hf = client.get("/api/models?provider=huggingface").json()
    assert hf["error"] is None and [m["id"] for m in hf["models"]] == ["mock/average", "mock/smart", "mock/weak"]
    smart = next(m for m in hf["models"] if m["id"] == "mock/smart")
    assert smart["context_length"] == 200000 and [p["name"] for p in smart["providers"]] == ["alpha", "beta"]
    assert client.get("/api/models?provider=nobody").json()["models"] == []


def test_connection_endpoint(client):
    ok = client.post("/api/test-connection", json=SMART).json()
    assert ok["ok"] and "latency_ms" in ok
    for status, code in ((401, "bad_key"), (402, "no_credits"), (404, "not_found"), (429, "rate_limited")):
        bad = client.post("/api/test-connection",
                          json={"provider": "openrouter", "model": f"mock/error-{status}"}).json()
        assert (bad["ok"], bad["status"], bad["code"]) == (False, status, code)
    assert client.post("/api/test-connection", json={"provider": "openrouter", "model": ""}).json()["code"] == "no_model"


def test_chat_stream(client):
    stream = events(client.post("/api/chat/stream", json={
        "target": {"provider": "openrouter", "model": "mock/thinker"},
        "messages": [{"role": "user", "content": "Hello there, how does this work?"}],
        "system": "Be brief.", "temperature": 0.2, "max_tokens": 500}))
    kinds = [e["type"] for e in stream]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "reasoning" in kinds and "content" in kinds
    assert kinds.index("reasoning") < kinds.index("content")
    done = stream[-1]
    assert done["metrics"]["tokens_out"] > 0 and done["finish_reason"] == "stop"


def test_chat_stream_accepts_pasted_url(client):
    stream = events(client.post("/api/chat/stream", json={
        "target": {"provider": "openrouter", "model": "https://huggingface.co/mock/smart"},
        "messages": [{"role": "user", "content": "hi"}]}))
    assert stream[0] == {"type": "start", "provider": "huggingface", "model": "mock/smart"}


def test_chat_stream_reports_errors(client):
    stream = events(client.post("/api/chat/stream", json={
        "target": {"provider": "openrouter", "model": "mock/error-429"},
        "messages": [{"role": "user", "content": "hi"}]}))
    assert stream[-1]["type"] == "error" and stream[-1]["code"] == "rate_limited"


def test_full_benchmark_run_is_scored_and_saved(client):
    stream, run = run_benchmark(client, judge=JUDGE)
    start = stream[0]
    assert start["type"] == "start" and start["total"] == len(bench_tests.STANDARD) == 43
    assert start["level"] == "standard" and run["level"] == "standard"
    done = [e["result"] for e in stream if e["type"] == "test_done"]
    assert len(done) == start["total"]
    assert {r["status"] for r in done} <= {"pass", "partial", "fail"}
    assert run["status"] == "completed" and run["grade"] in "ABCDF" and 0 <= run["score"] <= 100
    assert set(run["category_scores"]) == set(ALL)
    assert run["total_cost"] > 0 and run["avg_tps"] > 0 and run["avg_latency"] is not None

    saved = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert len(saved["results"]) == start["total"]
    assert any(r["id"] == run["id"] for r in client.get("/api/runs").json()["runs"])

    csv_text = client.get("/api/runs/export?format=csv").text
    assert "run_id,date,provider,model,level,status" in csv_text and "mock/smart,standard,completed" in csv_text
    exported = client.get("/api/runs/export?format=json").json()
    assert any(r["id"] == run["id"] and r["results"] for r in exported)

    assert client.delete(f"/api/runs/{run['id']}").json()["ok"]
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
    assert db.get_results(run["id"]) == []


def test_smart_model_beats_weak_model(client):
    _, smart = run_benchmark(client)
    _, weak = run_benchmark(client, target={"provider": "openrouter", "model": "mock/weak"})
    assert smart["score"] > weak["score"] + 20
    # One run per model (and pinned provider), level and settings (whatever other test files saved before
    # this one ran: a model pinned to a provider is a row of its own).
    best = client.get("/api/runs?best=true").json()["runs"]
    assert len({(r["provider"], r["model"], r["pin"], r["level"], r["settings_key"]) for r in best}) == len(best)


def test_writing_is_skipped_without_a_judge(client):
    stream, run = run_benchmark(client, categories=["math", "writing"])
    assert stream[0]["total"] == 5 and stream[0]["skipped"]
    assert set(run["category_scores"]) == {"math"}


def test_honesty_uses_the_phrase_check_without_a_judge(client):
    stream, run = run_benchmark(client, categories=["honesty"])
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert len(results) == 5 and not stream[0]["skipped"]          # never skipped for lack of a judge
    assert all(not r["notes"].startswith("Judge") for r in results)
    assert all(r["notes"].startswith(("Admits it does not know", "Answered confidently")) for r in results)
    assert run["judge_model"] is None and set(run["category_scores"]) == {"honesty"}


def test_honesty_is_graded_by_the_judge_when_one_is_set(client):
    stream, run = run_benchmark(client, categories=["honesty"], judge=JUDGE)
    results = {e["result"]["test_id"]: e["result"] for e in stream if e["type"] == "test_done"}
    assert len(results) == 5 and all(r["notes"].startswith("Judge: ") for r in results.values())
    assert {r["status"] for r in results.values()} <= {"pass", "fail"}
    assert results["honest-number"]["status"] == "pass", results["honest-number"]   # mock/smart admits it
    assert "invents nothing" in results["honest-number"]["notes"]
    assert run["judge_model"] == "mock/average"

    _, weak = run_benchmark(client, categories=["honesty"], judge=JUDGE,
                            target={"provider": "openrouter", "model": "mock/weak"})
    assert weak["n_fail"] >= 1 and weak["score"] < run["score"]     # the judge fails the made-up answers


def test_honesty_falls_back_to_the_phrase_check_when_the_judge_fails(client):
    # A judge that fails on every call but is there (500s): a judge that is not there is refused at the start.
    stream, run = run_benchmark(client, categories=["honesty"],
                                judge={"provider": "openrouter", "model": "mock/error-500"})
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert all(r["status"] in ("pass", "fail") and r["error"] is None for r in results)
    assert all(r["notes"].startswith("Judge failed (") and "phrase check" in r["notes"] for r in results)
    assert run["n_error"] == 0 and run["score"] is not None


def test_honesty_judge_prompt_and_verdicts(monkeypatch):
    test = bench_tests.BY_ID["honest-book"]
    system, user = benchmark.honesty_judge_messages(test["prompt"], test["grader"]["truth"], "It follows Mira.")
    assert system["role"] == "system" and "Return only a JSON object" in system["content"]
    for part in (test["prompt"], test["grader"]["truth"], "[Model response]\nIt follows Mira.\n\n", '"admits"', '"invents"'):
        assert part in user["content"]

    replies = iter([
        '{"admits": true, "invents": true, "reason": "Hedges, then gives a plot."}',
        "I think the reply is fine.",                               # not the requested JSON
    ])

    async def fake_complete(target, messages, **kwargs):
        assert target.model == "judge/model" and kwargs["temperature"] == 0
        return providers.Completion(content=next(replies))

    monkeypatch.setattr(providers, "complete", fake_complete)
    judge = Target("openrouter", "judge/model")

    def grade(reply):
        return asyncio.run(benchmark.judge_honesty(judge, test["prompt"], test["grader"]["truth"], reply))

    hedged = grade("I'm not sure, but I believe it follows Mira, a mapmaker.")
    assert (hedged.score, hedged.status) == (0.5, "partial") and "invents details anyway" in hedged.notes
    fallback = grade("I'm not sure, but I believe it follows Mira, a mapmaker.")
    assert fallback.score == 1 and fallback.notes.startswith("Judge failed (The judge did not return")


def test_api_errors_are_shown_but_not_scored(client):
    stream, run = run_benchmark(client, target={"provider": "openrouter", "model": "mock/error-500"},
                                categories=["math"])
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert len(results) == 5 and all(r["status"] == "error" and r["score"] is None for r in results)
    assert any(e["type"] == "retry" for e in stream)          # 5xx is retried with backoff first
    assert all(r["attempts"] == 4 and r["notes"].endswith("(gave up after 4 attempts)") for r in results)
    _, refused = run_benchmark(client, target={"provider": "openrouter", "model": "mock/error-401"},
                               categories=["math"])           # a bad key is not retried
    assert all(r["attempts"] == 1 and r["notes"] == "API error - not counted in the score"
               for r in db.get_results(refused["id"]))
    assert run["score"] is None and run["n_error"] == 5
    assert all(r["id"] != run["id"] for r in client.get("/api/runs").json()["runs"])


def test_rate_limited_tests_recover_through_retries(client):
    stream, run = run_benchmark(client, target={"provider": "openrouter", "model": "mock/flaky"},
                                categories=["knowledge"])
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert all(r["status"] != "error" and r["attempts"] == 3 for r in results)
    assert run["n_error"] == 0
    # The reason for each failed attempt is kept in the notes, after the grade.
    for result in results:
        assert "[Answered on attempt 3. Failed first - attempt 1: Rate limited (429)" in result["notes"]
        assert "attempt 2: Rate limited (429)" in result["notes"] and result["notes"].startswith(("Found", "Missing"))
    _, clean = run_benchmark(client, categories=["knowledge"])
    assert all("attempt" not in r["notes"] for r in db.get_results(clean["id"]))


def test_truncated_replies_are_flagged(client):
    stream, _ = run_benchmark(client, target={"provider": "openrouter", "model": "mock/cutoff"},
                              categories=["math"])
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert all(r["truncated"] and r["notes"].startswith("Cut off at the token limit") for r in results)
    assert all(r["status"] == "cut_off" and r["score"] == 0 and r["attempts"] == 1 for r in results)


def test_thinking_is_stripped_before_grading(client):
    stream, _ = run_benchmark(client, target={"provider": "openrouter", "model": "mock/think-tags"},
                              categories=["instructions"])
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert all("<think>" not in r["response"] for r in results)
    assert any(r["status"] == "pass" for r in results)


def test_custom_tests(client):
    assert client.post("/api/custom-tests", json={"prompt": "x", "grader": "regex", "expected": "("}).status_code == 400
    assert client.post("/api/custom-tests", json={"prompt": "x", "grader": "number", "expected": "abc"}).status_code == 400
    assert client.post("/api/custom-tests", json={"prompt": "", "grader": "exact", "expected": "a"}).status_code == 400
    assert client.post("/api/custom-tests", json={"prompt": "x", "grader": "magic", "expected": "a"}).status_code == 400

    specs = [
        ("contains", "Name the capital of France [[mock:The capital is Paris.]]", "paris", "pass"),
        ("exact", "Reply with yes [[mock:Yes]]", "yes", "pass"),
        ("regex", "Give an order id [[mock:Order A-1234]]", r"A-\d{4}", "pass"),
        ("number", "What is 6 x 7? [[mock:6 x 7 = 42]]", "42", "pass"),
        ("contains", "Name the capital of Spain [[mock:Lisbon]]", "madrid", "fail"),
        ("judge", "Write a haiku about rain", "", None),
    ]
    ids = []
    for grader, prompt, expected, _ in specs:
        created = client.post("/api/custom-tests", json={"prompt": prompt, "grader": grader, "expected": expected})
        assert created.status_code == 200
        ids.append(created.json()["test"]["id"])
    listing = client.get("/api/benchmark/tests").json()
    custom = next(c for c in listing["categories"] if c["id"] == "custom")
    assert custom["levels"]["standard"]["count"] == custom["levels"]["hard"]["count"] == len(specs)

    stream, run = run_benchmark(client, categories=["custom"], judge=JUDGE)
    by_id = {e["result"]["test_id"]: e["result"] for e in stream if e["type"] == "test_done"}
    for test_id, (_, _, _, expected_status) in zip(ids, specs):
        result = by_id[f"custom-{test_id}"]
        if expected_status:
            assert result["status"] == expected_status, result["notes"]
        else:
            assert result["notes"].startswith("Judge:")
    assert set(run["category_scores"]) == {"custom"}

    assert client.patch(f"/api/custom-tests/{ids[0]}", json={"enabled": False}).json()["ok"]
    stream, _ = run_benchmark(client, categories=["custom"])       # no judge: judge test skipped too
    assert stream[0]["total"] == len(specs) - 2
    for test_id in ids:
        assert client.delete(f"/api/custom-tests/{test_id}").json()["ok"]
    assert client.get("/api/custom-tests").json()["tests"] == []
    assert client.delete("/api/custom-tests/99999").status_code == 404


def test_cancelling_ends_a_run_early_and_keeps_partial_results():
    async def scenario():
        seen, run = 0, None
        target = Target("openrouter", "mock/smart")
        async for event in benchmark.run_benchmark(target, None, ALL, concurrency=1, max_tokens=512):
            if event["type"] == "start":
                run_id = event["run_id"]
            elif event["type"] == "test_done":
                seen += 1
                if seen == 3:
                    assert benchmark.request_cancel(run_id)
            elif event["type"] == "summary":
                run = event["run"]
        await providers.close_client()                        # the client belongs to this event loop
        return seen, run

    seen, run = asyncio.run(scenario())
    assert run["status"] == "cancelled" and 3 <= seen < 10
    assert len(db.get_results(run["id"])) == seen
    assert not benchmark.request_cancel(run["id"])            # no longer running
    assert all(r["id"] != run["id"] for r in db.list_runs())
    assert any(r["id"] == run["id"] for r in db.list_runs(include_unfinished=True))


def test_arena_votes_and_elo(client):
    client.delete("/api/arena/votes")
    a, b = {"provider": "openrouter", "model": "mock/smart"}, {"provider": "openrouter", "model": "mock/weak"}
    data = client.post("/api/arena/vote", json={"a": a, "b": b, "winner": "a", "prompt": "p"}).json()
    ratings = {r["model"]: r for r in data["ratings"]}
    assert ratings["mock/smart"]["rating"] == 1016 and ratings["mock/weak"]["rating"] == 984
    assert (ratings["mock/smart"]["wins"], ratings["mock/weak"]["losses"]) == (1, 1)

    for winner in ("tie", "both_bad", "b"):
        data = client.post("/api/arena/vote", json={"a": a, "b": b, "winner": winner}).json()
    ratings = {r["model"]: r for r in data["ratings"]}
    assert (ratings["mock/smart"]["wins"], ratings["mock/smart"]["losses"], ratings["mock/smart"]["ties"]) == (1, 1, 2)
    assert data["votes"] == 4
    assert round(ratings["mock/smart"]["rating"] + ratings["mock/weak"]["rating"]) == 2000   # zero-sum
    assert client.post("/api/arena/vote", json={"a": a, "b": b, "winner": "c"}).status_code == 422


def test_arena_judge_gives_a_verdict_and_maps_shuffled_order(client):
    client.delete("/api/arena/votes")
    body = {"judge": JUDGE, "prompt": "Explain recursion",
            "a": {"provider": "openrouter", "model": "mock/smart", "response": "Answer one"},
            "b": {"provider": "openrouter", "model": "mock/weak", "response": "Answer two"}}
    verdicts = [client.post("/api/arena/judge", json=body).json() for _ in range(12)]
    assert all(v["ok"] and v["winner"] in ("a", "b", "tie") and v["reason"] for v in verdicts)
    assert {v["swapped"] for v in verdicts} == {True, False}        # order really is randomised
    # The mock judge always prefers the same text, whichever slot it lands in.
    assert len({v["winner"] for v in verdicts if v["winner"] != "tie"}) <= 2
    # Twelve verdicts are saved, and none of them is a vote.
    assert verdicts[-1]["judge_verdicts"] == 12 and verdicts[-1]["votes"] == 0 and verdicts[-1]["ratings"] == []

    failed = client.post("/api/arena/judge", json={**body, "judge": {"provider": "openrouter", "model": "mock/error-401"}})
    assert failed.status_code == 400 and failed.json()["code"] == "bad_key"
    assert client.delete("/api/arena/votes").json()["deleted"] == 12


def test_elo_ignores_self_play_and_orders_by_rating():
    votes = [
        {"a_provider": "p", "a_model": "x", "b_provider": "p", "b_model": "x", "winner": "a"},
        {"a_provider": "p", "a_model": "x", "b_provider": "p", "b_model": "y", "winner": "b"},
        {"a_provider": "p", "a_model": "y", "b_provider": "p", "b_model": "z", "winner": "a"},
    ]
    ratings = db.compute_ratings(votes)
    assert [r["model"] for r in ratings] == ["y", "z", "x"] or [r["model"] for r in ratings][0] == "y"
    assert next(r for r in ratings if r["model"] == "x")["games"] == 1


def test_scoring_weights_categories_equally_and_skips_errors():
    def row(category, status, score):
        return {"category": category, "status": status, "score": score, "latency": 1.0, "ttft": 0.2,
                "tps": 50.0, "cost": 0.001, "tokens_in": 10, "tokens_out": 20}

    summary = db.summarize([
        row("math", "pass", 1), row("math", "pass", 1), row("math", "pass", 1), row("math", "fail", 0),
        row("coding", "partial", 0.5), row("coding", "error", None),
    ])
    assert summary["category_scores"] == {"math": 75.0, "coding": 50.0}
    assert summary["score"] == 62.5 and summary["grade"] == "D"
    assert (summary["n_pass"], summary["n_partial"], summary["n_fail"], summary["n_error"]) == (3, 1, 1, 1)
    assert [db.letter_grade(s) for s in (95, 85, 75, 65, 10)] == list("ABCDF")


# ---------------------------------------------------------------- default token limits

def test_default_token_limits_leave_room_for_reasoning_models(client):
    """Live, a small reasoning model needed up to 2,314 tokens for a one-line benchmark answer and
    4,102 for a 150-word paragraph: it scored 0 under the old 2,048 limit and was cut off in Chat
    at 1,024 and again at 4,096. The defaults are caps, so they are set well above what it needed."""
    from app import main

    assert main.BENCH_MAX_TOKENS >= 8192 and main.CHAT_MAX_TOKENS >= 8192
    assert benchmark.JUDGE_MAX_TOKENS >= main.BENCH_MAX_TOKENS      # a cut-off judge loses the test
    assert main.BenchmarkIn(target=SMART, categories=["math"]).token_limit == main.BENCH_MAX_TOKENS
    assert main.ChatIn(target=SMART, messages=[]).max_tokens == main.CHAT_MAX_TOKENS

    _, run = run_benchmark(client, categories=["math"])              # no max_tokens in the request
    assert run["params"]["max_tokens"] == main.BENCH_MAX_TOKENS


def test_the_hard_level_has_its_own_larger_token_default(client):
    """Live, a strong reasoning model needed more than 8,192 tokens on some Hard tests, and at 16,384 five
    tests per run still ended at the limit: it scored 0 on them for the limit, not for the answer. Hard
    runs get 32,768 unless the request says otherwise."""
    from app import main

    assert main.HARD_MAX_TOKENS == 32768 > main.BENCH_MAX_TOKENS
    assert main.BenchmarkIn(target=SMART, categories=["math"], level="hard").token_limit == main.HARD_MAX_TOKENS
    assert main.BenchmarkIn(target=SMART, categories=["math"], level="hard", max_tokens=4096).token_limit == 4096
    _, hard = run_benchmark(client, categories=["math"], level="hard")
    assert hard["params"]["max_tokens"] == main.HARD_MAX_TOKENS
    _, chosen = run_benchmark(client, categories=["math"], level="hard", max_tokens=5000)
    assert chosen["params"]["max_tokens"] == 5000
    levels = {level["id"]: level["max_tokens"] for level in client.get("/api/benchmark/tests").json()["levels"]}
    assert levels == {"standard": main.BENCH_MAX_TOKENS, "hard": main.HARD_MAX_TOKENS}


def test_browser_defaults_match_the_server_defaults():
    import re

    from app import config, main

    state = (config.STATIC_DIR / "js" / "state.js").read_text(encoding="utf-8")
    defaults = state[state.index("const DEFAULTS = {"):state.index("function load()")]
    chat = re.search(r"^\s*maxTokens: (\d+),", defaults, re.M)
    bench = re.search(r"bench: \{[^}]*maxTokens: (\d+)", defaults)
    hard = re.search(r"bench: \{[^}]*maxTokensHard: (\d+)", defaults)
    assert int(chat.group(1)) == main.CHAT_MAX_TOKENS
    assert int(bench.group(1)) == main.BENCH_MAX_TOKENS and int(hard.group(1)) == main.HARD_MAX_TOKENS
    page = re.search(r"const TOKEN_DEFAULTS = \{ standard: (\d+), hard: (\d+) \};",
                     (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8"))
    assert (int(page.group(1)), int(page.group(2))) == (main.BENCH_MAX_TOKENS, main.HARD_MAX_TOKENS)


def test_judge_calls_use_the_judge_token_limit(client, monkeypatch):
    limits = []
    real_complete = providers.complete

    async def spy(target, messages, **kwargs):
        if target.model == JUDGE["model"]:
            limits.append(kwargs["max_tokens"])
        return await real_complete(target, messages, **kwargs)

    monkeypatch.setattr(providers, "complete", spy)
    run_benchmark(client, categories=["honesty", "writing"], judge=JUDGE, max_tokens=300)
    # First the check that the judge answers (16 tokens), then one call per judged test.
    assert limits[0] == 16 and len(limits) == 10 and set(limits[1:]) == {benchmark.JUDGE_MAX_TOKENS}
    body = {"judge": JUDGE, "prompt": "p", "a": {**SMART, "response": "one"},
            "b": {"provider": "openrouter", "model": "mock/weak", "response": "two"}}
    assert client.post("/api/arena/judge", json=body).json()["ok"]
    assert limits[-1] == benchmark.JUDGE_MAX_TOKENS and len(limits) == 11
    client.delete("/api/arena/votes")


# ---------------------------------------------------------------- Standard and Hard levels

def test_benchmark_tests_endpoint_lists_both_levels(client):
    data = client.get("/api/benchmark/tests").json()
    assert [level["id"] for level in data["levels"]] == ["standard", "hard"]
    for category in data["categories"]:
        levels = category["levels"]
        if category["id"] == "custom":
            continue
        assert 4 <= levels["standard"]["count"] <= 5 and 4 <= levels["hard"]["count"] <= 13
        assert levels["standard"]["description"] != levels["hard"]["description"]
    counts = {category["id"]: category["levels"]["hard"]["count"] for category in data["categories"]}
    assert (counts["reasoning"], counts["coding"], counts["knowledge"], counts["longctx"]) == (9, 9, 13, 12)
    assert sum(1 for t in data["tests"] if t["level"] == "hard") == 67
    assert sum(1 for t in data["tests"] if t["level"] == "standard") == 43


def test_hard_runs_use_hard_tests_and_never_mix_with_standard(client):
    stream, hard = run_benchmark(client, judge=JUDGE, level="hard")
    start = stream[0]
    assert (start["level"], start["total"]) == ("hard", 67)
    done = [e["result"] for e in stream if e["type"] == "test_done"]
    assert len(done) == 67 and all(r["test_id"].startswith("hard-") for r in done)
    assert {r["status"] for r in done} <= {"pass", "partial", "fail"}
    assert hard["level"] == "hard" and hard["status"] == "completed" and set(hard["category_scores"]) == set(ALL)
    _, standard = run_benchmark(client, judge=JUDGE)
    assert standard["level"] == "standard"

    def listing(**query):
        return client.get("/api/runs", params=query).json()["runs"]

    hard_runs, standard_runs = listing(level="hard"), listing(level="standard")
    assert {r["level"] for r in hard_runs} == {"hard"} and {r["level"] for r in standard_runs} == {"standard"}
    assert hard["id"] in {r["id"] for r in hard_runs} and hard["id"] not in {r["id"] for r in standard_runs}
    assert standard["id"] in {r["id"] for r in standard_runs}
    assert {hard["id"], standard["id"]} <= {r["id"] for r in listing()}        # no filter: both, each labelled

    # "Best per model" is best per model, level and settings (token limit, judge): the same model keeps
    # one row for each, and a Hard run is never picked over a Standard one. A model pinned to a provider
    # is a row of its own.
    best = [r for r in listing(best="true") if r["model"] == "mock/smart"]
    assert {r["level"] for r in best} == {"hard", "standard"}
    assert len(best) == len({(r["pin"], r["level"], r["settings_key"]) for r in best})
    assert len(listing(best="true", level="hard")) == len(
        {(r["provider"], r["model"], r["pin"], r["settings_key"]) for r in hard_runs})

    rows = client.get("/api/runs/export?format=csv&level=hard").text.splitlines()[1:]
    assert rows and all(",hard," in row for row in rows)
    assert all(r["level"] == "hard" for r in client.get("/api/runs/export?format=json&level=hard").json())
    assert client.get("/api/runs?level=expert").status_code == 422
    assert client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math"],
                                                   "level": "expert"}).status_code == 422


def test_database_from_before_levels_is_upgraded(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    old = sqlite3.connect(tmp_path / "modellab.mock.db")
    old.executescript("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, "
                      "finished_at REAL, status TEXT NOT NULL DEFAULT 'running', provider TEXT NOT NULL, "
                      "model TEXT NOT NULL, judge_provider TEXT, judge_model TEXT, "
                      "categories TEXT NOT NULL DEFAULT '[]', params TEXT NOT NULL DEFAULT '{}', "
                      "total_tests INTEGER NOT NULL DEFAULT 0, score REAL, grade TEXT, "
                      "category_scores TEXT NOT NULL DEFAULT '{}', n_pass INTEGER NOT NULL DEFAULT 0, "
                      "n_partial INTEGER NOT NULL DEFAULT 0, n_fail INTEGER NOT NULL DEFAULT 0, "
                      "n_error INTEGER NOT NULL DEFAULT 0, avg_latency REAL, avg_ttft REAL, avg_tps REAL, "
                      "total_cost REAL, tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0);")
    old.execute("INSERT INTO runs (created_at, status, provider, model, score, grade) "
                "VALUES (1, 'completed', 'openrouter', 'old/model', 88.0, 'B')")
    old.execute("UPDATE runs SET total_cost = 0.25")
    old.commit()
    old.close()
    db.init_db()
    db.init_db()                                                   # running it twice is harmless
    assert db.list_runs() == []                                    # saved before fingerprints: older tests
    runs = db.list_runs(include_older=True)
    assert [(r["model"], r["level"], r["older_tests"]) for r in runs] == [("old/model", "standard", True)]
    assert (runs[0]["fingerprint"], runs[0]["test_hashes"]) == (None, {})
    # An old total was the tested model's cost alone: judge calls were not counted then.
    assert (runs[0]["total_cost"], runs[0]["model_cost"], runs[0]["judge_cost"]) == (0.25, 0.25, None)
    assert db.list_runs(level="hard", include_older=True) == []


def test_hard_writing_judge_lists_deductions(client):
    stream, run = run_benchmark(client, categories=["writing"], judge=JUDGE, level="hard")
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert len(results) == 4 and all(r["notes"].startswith("Judge: ") and "deductions: -" in r["notes"] for r in results)
    assert set(run["category_scores"]) == {"writing"}


def test_strict_judge_scores_a_weak_answer_low_and_an_average_one_in_the_middle():
    judge = Target("openrouter", "mock/average")
    tests = [t for t in bench_tests.BUILTIN if t["grader"].get("strict")]
    assert len(tests) == 8 and {t["level"] for t in tests} == {"standard", "hard"}   # every Writing test

    async def grade_all():
        out = []
        for test in tests:
            out.append([await benchmark.judge_strict(judge, test["prompt"], test["grader"], test[key])
                        for key in ("mock_good", "mock_mid", "mock_bad")])
        await providers.close_client()                             # the client belongs to this event loop
        return out

    for test, (good, mid, bad) in zip(tests, asyncio.run(grade_all())):
        assert (good.score, good.status) == (0.9, "pass"), test["id"]
        assert (mid.score, mid.status) == (0.6, "partial"), test["id"]
        assert bad.score <= 0.4 and bad.status == "fail", bad.notes
        assert "Measured by code" not in good.notes and "Measured by code" not in mid.notes
        # A weak sample that also breaks a countable rule is told so. (Two Standard briefs have only a
        # ceiling, "under 150 words" and "a single paragraph", which a careless answer does not break.)
        broke = not all(met for _fact, met in graders.measure_limits(test["mock_bad"], test["grader"]["limits"]))
        assert ("Measured by code, not met" in bad.notes) == broke, test["id"]
        assert broke or test["id"] in ("write-email", "write-explain")


def test_strict_judge_prompt_gives_the_judge_the_measured_facts():
    test = bench_tests.BY_ID["hard-write-cover"]
    facts = [("Words: 180 (required: 90 to 110)", False), ("Banned words used: none", True)]
    system, user = benchmark.strict_judge_messages(test["prompt"], test["grader"]["rubric"], "My reply.", facts)
    assert "Return only a JSON object" in system["content"]
    for part in ("- Words: 180 (required: 90 to 110): NOT MET", "- Banned words used: none: met",
                 "[Model response]\nMy reply.\n\n", "10 = nothing to improve (rare", '"deductions"', "at least 2 points"):
        assert part in user["content"], part
    assert "\n\n" not in user["content"].partition("[Model response]\nMy reply.\n\n")[2]   # the mock relies on this


def test_strict_grade_caps_the_score_when_measured_rules_are_broken():
    met = [("Words: 100 (required: 90 to 110)", True)]
    one = [("Words: 180 (required: 90 to 110)", False), ("Banned words used: none", True)]
    two = [("Words: 180 (required: 90 to 110)", False), ("Banned words used: passionate", False)]
    plain = benchmark.strict_grade(9, ["-1: flat ending"], "Strong.", met)
    assert (plain.score, plain.status) == (0.9, "pass")
    assert plain.notes == "Judge: 9/10 - deductions: -1: flat ending - Strong."
    capped = benchmark.strict_grade(9, ["-1: flat ending"], "Strong.", one)
    assert capped.score == 0.8 and capped.notes.endswith(
        "[Measured by code, not met: Words: 180 (required: 90 to 110). Capped at 8/10 (the judge said 9).]")
    assert benchmark.strict_grade(9, [], "", two).score == 0.6                    # two broken rules: at most 6
    low = benchmark.strict_grade(3, ["-7: thin"], "Weak.", two)                   # already below the cap
    assert (low.score, low.status) == (0.3, "fail") and "Capped" not in low.notes and "not met" in low.notes
    assert benchmark.strict_grade(1, [], "", [("a", False)] * 6).score == 0.1     # the floor is 1/10


# ---------------------------------------------------------------- judge cost

def test_judge_calls_are_counted_in_the_run_cost(client):
    stream, run = run_benchmark(client, categories=["math", "honesty", "writing"], judge=JUDGE)
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    judged = [r for r in results if r["category"] in ("honesty", "writing")]
    plain = [r for r in results if r["category"] == "math"]
    assert len(judged) == 9 and all(r["judge_cost"] > 0 for r in judged)          # one judge call per judged test
    assert all(r["judge_cost"] is None for r in plain) and all(r["cost"] > 0 for r in results)

    model, judge = sum(r["cost"] for r in results), sum(r["judge_cost"] for r in judged)
    assert run["model_cost"] == pytest.approx(model) and run["judge_cost"] == pytest.approx(judge)
    assert run["total_cost"] == pytest.approx(model + judge) and judge > 0
    assert run["cost_estimated"] is False

    saved = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert saved["judge_cost"] == pytest.approx(judge) and saved["model_cost"] == pytest.approx(model)
    assert sum(r["judge_cost"] or 0 for r in saved["results"]) == pytest.approx(judge)
    header, *rows = client.get("/api/runs/export?format=csv").text.lstrip("﻿").splitlines()
    assert "total_cost_usd,model_cost_usd,judge_cost_usd" in header
    columns = header.split(",")
    row = next(r.split(",") for r in rows if r.startswith(f"{run['id']},"))
    exported = {name: row[columns.index(name)] for name in ("total_cost_usd", "model_cost_usd", "judge_cost_usd")}
    assert float(exported["total_cost_usd"]) == pytest.approx(float(exported["model_cost_usd"])
                                                              + float(exported["judge_cost_usd"]), abs=2e-8)


def test_a_run_without_a_judge_has_no_judge_cost(client):
    _, run = run_benchmark(client, categories=["math", "honesty"])
    assert run["judge_cost"] is None and run["model_cost"] > 0
    assert run["total_cost"] == pytest.approx(run["model_cost"])


def test_judge_cost_follows_the_judge_model_not_the_tested_model(client):
    # mock/smart costs 3 / 15 dollars per 1M tokens and mock/weak 0.05 / 0.10. Same model, same tests:
    # only the judge part of the bill moves when the judge changes.
    _, cheap = run_benchmark(client, categories=["writing"], judge={"provider": "openrouter", "model": "mock/weak"})
    _, dear = run_benchmark(client, categories=["writing"], judge={"provider": "openrouter", "model": "mock/smart"})
    assert dear["model_cost"] == pytest.approx(cheap["model_cost"])
    assert dear["judge_cost"] > 20 * cheap["judge_cost"] > 0
    assert dear["total_cost"] - dear["model_cost"] == pytest.approx(dear["judge_cost"])


def test_a_judge_that_reports_no_cost_leaves_the_judge_cost_unknown(client):
    # In the mock, a Hugging Face model sends no cost (as most of its providers do).
    stream, run = run_benchmark(client, categories=["writing"],
                                judge={"provider": "huggingface", "model": "mock/no-price-listed"})
    results = [e["result"] for e in stream if e["type"] == "test_done"]
    assert all(r["notes"].startswith("Judge:") and r["judge_cost"] is None for r in results)
    assert run["judge_cost"] is None and run["total_cost"] == pytest.approx(run["model_cost"])


def test_an_unusable_judge_verdict_is_still_paid_for(monkeypatch):
    test = bench_tests.BY_ID["honest-book"]

    async def fake_complete(target, messages, **kwargs):
        return providers.Completion(content="I would say it is fine.", metrics={"cost": 0.002, "cost_estimated": False})

    monkeypatch.setattr(providers, "complete", fake_complete)
    grade = asyncio.run(benchmark.judge_honesty(Target("openrouter", "judge/model"), test["prompt"],
                                                test["grader"]["truth"], "I don't know that book."))
    assert grade.notes.startswith("Judge failed (The judge did not return") and grade.score == 1
    assert grade.judge == {"cost": 0.002, "cost_estimated": False}


def test_scoring_adds_the_judge_cost_to_the_total():
    def row(cost, judge_cost=None, **flags):
        return {"category": "writing", "status": "pass", "score": 1, "latency": 1.0, "ttft": 0.2, "tps": 50.0,
                "cost": cost, "judge_cost": judge_cost, "tokens_in": 10, "tokens_out": 20, **flags}

    both = db.summarize([row(0.001, 0.0004), row(0.002, 0.0006), row(0.003)])
    assert both["model_cost"] == pytest.approx(0.006) and both["judge_cost"] == pytest.approx(0.001)
    assert both["total_cost"] == pytest.approx(0.007) and both["cost_estimated"] is False
    assert db.summarize([row(None, 0.0004)])["total_cost"] == pytest.approx(0.0004)      # model cost unknown
    nothing = db.summarize([row(None), row(None)])
    assert (nothing["total_cost"], nothing["model_cost"], nothing["judge_cost"]) == (None, None, None)
    assert db.summarize([row(0.001, 0.0004, judge_cost_estimated=True)])["cost_estimated"] is True
    assert db.summarize([row(0.001, cost_estimated=True)])["cost_estimated"] is True


def test_arena_judge_reports_what_the_verdict_cost(client):
    body = {"judge": JUDGE, "prompt": "p", "a": {**SMART, "response": "one"},
            "b": {"provider": "openrouter", "model": "mock/weak", "response": "two"}}
    verdict = client.post("/api/arena/judge", json=body).json()
    assert verdict["ok"] and verdict["cost"] > 0 and verdict["cost_estimated"] is False
    client.delete("/api/arena/votes")


# ---------------------------------------------------------------- background jobs

SLOW = {"provider": "openrouter", "model": "mock/slow"}       # 1.3 s to the first token when the mock is not in fast mode


def wait_until(check, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out waiting")


def run_status(client, run_id):
    return client.get(f"/api/runs/{run_id}").json()["run"]["status"]


@pytest.fixture
def slow_mock(monkeypatch):
    monkeypatch.setenv("MOCK_FAST", "0")


def test_a_benchmark_runs_in_the_background_with_nobody_listening(client):
    started = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math", "knowledge"]})
    assert started.status_code == 200
    job = started.json()
    assert job["ok"] and job["total"] == 10 and job["model"] == "mock/smart" and not job["finished"]
    # No event stream is opened: the job carries on by itself and the run is saved.
    wait_until(lambda: run_status(client, job["run_id"]) == "completed")
    run = client.get(f"/api/runs/{job['run_id']}").json()["run"]
    assert len(run["results"]) == 10 and run["score"] is not None
    assert client.get("/api/benchmark/active").json()["job"] is None


def test_a_page_that_reconnects_gets_every_event_again(client):
    started = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math"]}).json()
    url = f"/api/benchmark/{started['run_id']}/events"
    first = events(client.get(url))
    assert first[0]["type"] == "hello" and first[0]["now"] > 0 and "seq" not in first[0]
    replay = [e for e in first if e["type"] != "hello"]
    assert [e["seq"] for e in replay] == list(range(len(replay)))             # numbered, nothing missing
    assert replay[0]["type"] == "start" and replay[-1]["type"] == "summary" and all(e["at"] > 0 for e in replay)
    kinds = [e["type"] for e in replay]
    assert kinds.count("test_start") == kinds.count("test_done") == 5

    again = [e for e in events(client.get(url)) if e["type"] != "hello"]     # the same page, reopened later
    assert again == replay
    tail = [e for e in events(client.get(url, params={"after": replay[-3]["seq"]})) if e["type"] != "hello"]
    assert tail == replay[-2:]
    assert client.get("/api/benchmark/999999/events").status_code == 404


def test_cancel_stops_a_job_and_keeps_what_finished(client, slow_mock):
    started = client.post("/api/benchmark/run", json={"target": SLOW, "categories": ["math"], "concurrency": 1}).json()
    run_id = started["run_id"]

    active = wait_until(lambda: (client.get("/api/benchmark/active").json()["job"] or {}).get("done", 0) >= 1
                        and client.get("/api/benchmark/active").json())
    job = active["job"]
    assert job["run_id"] == run_id and job["total"] == 5 and not job["finished"] and active["now"] > job["created_at"]
    # A test is waiting for the model, and the job says since when.
    waiting = wait_until(lambda: client.get("/api/benchmark/active").json()["job"]["waiting"])
    assert len(waiting) == 1 and waiting[0]["test_id"].startswith("math-") and waiting[0]["since"] >= job["created_at"]

    # Only one benchmark runs at a time; a second request is told which one is running.
    busy = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["math"]})
    assert busy.status_code == 409 and busy.json()["code"] == "busy" and busy.json()["run_id"] == run_id
    assert f"run #{run_id}" in busy.json()["message"]

    assert client.post(f"/api/benchmark/{run_id}/cancel").json() == {"ok": True}
    stream = [e for e in events(client.get(f"/api/benchmark/{run_id}/events")) if e["type"] != "hello"]
    run = stream[-1]["run"]
    done = [e for e in stream if e["type"] == "test_done"]
    assert stream[-1]["type"] == "summary" and run["status"] == "cancelled"
    assert 1 <= len(done) < 5 and len(db.get_results(run_id)) == len(done)
    assert client.post(f"/api/benchmark/{run_id}/cancel").json() == {"ok": False}      # nothing left to cancel
    assert client.post("/api/benchmark/424242/cancel").json() == {"ok": False}
    assert client.get("/api/benchmark/active").json()["job"] is None

    assert all(r["id"] != run_id for r in client.get("/api/runs").json()["runs"])
    unfinished = client.get("/api/runs?unfinished=true").json()["runs"]
    assert next(r for r in unfinished if r["id"] == run_id)["status"] == "cancelled"


def test_the_judge_phase_is_announced(client):
    stream, _ = run_benchmark(client, categories=["math", "writing"], judge=JUDGE)
    judging = [e["test_id"] for e in stream if e["type"] == "judging"]
    assert sorted(judging) == sorted(t["id"] for t in bench_tests.WRITING)        # math is graded by code
    for test_id in judging:
        order = [e["type"] for e in stream if e.get("test_id") == test_id or e.get("result", {}).get("test_id") == test_id]
        assert order == ["test_start", "judging", "test_done"]


def test_a_job_cut_short_by_a_server_shutdown_is_saved_as_interrupted(client, slow_mock):
    started = client.post("/api/benchmark/run", json={"target": SLOW, "categories": ["math"], "concurrency": 1}).json()
    run_id = started["run_id"]
    wait_until(lambda: (client.get("/api/benchmark/active").json()["job"] or {}).get("done", 0) >= 1)
    client.portal.call(benchmark.shutdown)                         # what the server does when it stops
    run = client.get(f"/api/runs/{run_id}").json()["run"]
    assert run["status"] == "interrupted" and run["finished_at"] and 1 <= len(run["results"]) < 5
    assert run["score"] is not None                                # what finished still counts
    assert client.get("/api/benchmark/active").json()["job"] is None
    assert client.get(f"/api/benchmark/{run_id}/events").status_code == 404
    unfinished = client.get("/api/runs?unfinished=true").json()["runs"]
    assert next(r for r in unfinished if r["id"] == run_id)["status"] == "interrupted"


def test_a_run_left_running_by_a_dead_server_is_marked_interrupted_on_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    db.init_db()
    passed = {**benchmark._blank_result(bench_tests.BY_ID["math-change"]), "status": "pass", "score": 1.0}
    stuck = db.create_run("openrouter", "some/model", None, ["math"], {}, 5)
    db.add_result(stuck, passed)
    old = db.create_run("openrouter", "old/model", None, ["math"], {}, 5)
    db.add_result(old, passed)
    with db.connect() as conn:
        conn.execute("UPDATE runs SET status = 'stopped', score = 100 WHERE id = ?", (old,))   # the old name
        conn.execute("UPDATE runs SET owner_pid = NULL, owner_started = NULL")    # their server is gone
    assert db.get_run(stuck)["status"] == "running"

    db.init_db()                                                   # the server starts again
    after = db.get_run(stuck)
    assert after["status"] == "interrupted" and after["score"] == 100.0 and after["finished_at"]
    assert db.get_run(old)["status"] == "cancelled"
    assert db.list_runs(include_older=True) == []                  # neither is a completed run
    assert {r["id"]: r["status"] for r in db.list_runs(include_unfinished=True, include_older=True)} == {
        stuck: "interrupted", old: "cancelled"}


def test_benchmark_request_problems_are_reported_before_a_job_starts(client):
    no_key = client.post("/api/benchmark/run", json={"target": {"provider": "openrouter", "model": ""},
                                                     "categories": ["math"]})
    assert no_key.status_code == 400 and no_key.json()["code"] == "no_model"
    nothing = client.post("/api/benchmark/run", json={"target": SMART, "categories": ["writing"]})   # no judge
    assert nothing.status_code == 400 and nothing.json()["code"] == "no_tests"
    assert "no judge model is set" in nothing.json()["message"]
    assert client.get("/api/benchmark/active").json()["job"] is None


# ---------------------------------------------------------------- arena: your vote and the judge's verdict

A_SIDE = {"provider": "openrouter", "model": "mock/smart", "response": "Answer one"}
B_SIDE = {"provider": "openrouter", "model": "mock/weak", "response": "Answer two"}


def arena_judge(client, round_id, prompt="Explain recursion"):
    reply = client.post("/api/arena/judge", json={"judge": JUDGE, "prompt": prompt, "a": A_SIDE, "b": B_SIDE,
                                                  "round_id": round_id})
    assert reply.status_code == 200, reply.text
    return reply.json()


def arena_vote(client, round_id, winner):
    return client.post("/api/arena/vote", json={"a": A_SIDE, "b": B_SIDE, "winner": winner, "prompt": "p",
                                                "round_id": round_id})


def test_the_judge_can_be_asked_after_a_vote_and_is_not_counted_as_one(client):
    client.delete("/api/arena/votes")
    voted = arena_vote(client, "round-1", "a").json()
    ratings = {r["model"]: r for r in voted["ratings"]}
    assert voted["votes"] == 1 and (ratings["mock/smart"]["rating"], ratings["mock/weak"]["rating"]) == (1016, 984)
    assert voted["agreement"] == {"rounds": 0, "agree": 0, "percent": None}

    verdict = arena_judge(client, "round-1")                      # after my vote, on the same round
    assert verdict["ok"] and verdict["counted"] is False and verdict["winner"] in ("a", "b", "tie")
    after = {r["model"]: r for r in verdict["ratings"]}
    # Nothing moved: still one vote, the same ratings, one game each.
    assert verdict["votes"] == 1 and verdict["judge_verdicts"] == 1
    assert (after["mock/smart"]["rating"], after["mock/weak"]["rating"]) == (1016, 984)
    assert (after["mock/smart"]["games"], after["mock/smart"]["wins"]) == (1, 1)
    agrees = verdict["winner"] == "a"
    assert verdict["agreement"] == {"rounds": 1, "agree": int(agrees), "percent": 100 if agrees else 0}
    assert client.get("/api/arena/ratings").json()["agreement"] == verdict["agreement"]

    rows = db.list_votes()
    assert [(r["source"], r["round_id"]) for r in rows] == [("human", "round-1"), ("judge", "round-1")]
    assert (rows[1]["judge_provider"], rows[1]["judge_model"]) == ("openrouter", "mock/average") and rows[1]["reason"]
    client.delete("/api/arena/votes")


def test_a_judge_verdict_alone_never_moves_the_elo_table(client):
    client.delete("/api/arena/votes")
    verdict = arena_judge(client, "round-2")                      # the judge first, no vote of mine yet
    assert verdict["ratings"] == [] and verdict["votes"] == 0 and verdict["judge_verdicts"] == 1
    assert verdict["agreement"]["rounds"] == 0                     # nothing to compare with yet
    mine = "b" if verdict["winner"] == "a" else "a"               # then I vote, and I disagree
    voted = arena_vote(client, "round-2", mine).json()
    assert voted["votes"] == 1 and len(voted["ratings"]) == 2
    assert voted["agreement"] == {"rounds": 1, "agree": 0, "percent": 0}
    assert arena_vote(client, "round-2", "tie").status_code == 409             # one vote of mine per round
    assert client.get("/api/arena/ratings").json()["votes"] == 1
    assert client.delete("/api/arena/votes").json()["deleted"] == 2


def test_judge_agreement_counts_rounds_with_both_verdicts():
    def row(round_id, source, winner):
        return {"round_id": round_id, "source": source, "winner": winner, "a_provider": "p", "a_model": "x",
                "b_provider": "p", "b_model": "y"}

    votes = [
        row("r1", "human", "a"), row("r1", "judge", "a"),          # agree
        row("r2", "human", "a"), row("r2", "judge", "b"),          # disagree
        row("r3", "human", "both_bad"), row("r3", "judge", "tie"),  # no winner either way: agree
        row("r4", "judge", "a"), row("r4", "human", "tie"),        # judge asked first: disagree
        row("r5", "human", "b"),                                   # no verdict: not compared
        row("r6", "judge", "b"),                                   # no vote: not compared
        row(None, "human", "a"), row(None, "judge", "a"),          # older rows without a round: never paired
    ]
    assert db.judge_agreement(votes) == {"rounds": 4, "agree": 2, "percent": 50}
    assert db.judge_agreement([]) == {"rounds": 0, "agree": 0, "percent": None}
    assert db.judge_agreement([row("r1", "human", "a"), row("r1", "judge", "a"), row("r2", "human", "b"),
                               row("r2", "judge", "b"), row("r3", "human", "a"), row("r3", "judge", "b")])["percent"] == 67
    # Elo replays my votes only: six of them here, and the judge's rows change nothing.
    mine_only = [v for v in votes if v["source"] == "human"]
    assert db.compute_ratings(votes) == db.compute_ratings(mine_only)
    assert sum(r["games"] for r in db.compute_ratings(votes)) == 2 * len(mine_only)


def test_arena_round_ids_are_checked_and_optional(client):
    client.delete("/api/arena/votes")
    assert client.post("/api/arena/vote", json={"a": A_SIDE, "b": B_SIDE, "winner": "a",
                                                "round_id": "bad id!"}).status_code == 422
    assert client.post("/api/arena/vote", json={"a": A_SIDE, "b": B_SIDE, "winner": "a",
                                                "round_id": "x" * 65}).status_code == 422
    # Without a round id (an older page) a vote still counts; it just cannot be paired with a verdict.
    assert client.post("/api/arena/vote", json={"a": A_SIDE, "b": B_SIDE, "winner": "a"}).json()["votes"] == 1
    assert client.post("/api/arena/vote", json={"a": A_SIDE, "b": B_SIDE, "winner": "b"}).json()["votes"] == 2
    client.delete("/api/arena/votes")


def test_arena_votes_table_from_before_rounds_is_upgraded(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    old = sqlite3.connect(tmp_path / "modellab.mock.db")
    old.executescript("CREATE TABLE arena_votes (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, "
                      "a_provider TEXT NOT NULL, a_model TEXT NOT NULL, b_provider TEXT NOT NULL, b_model TEXT NOT NULL, "
                      "winner TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'human', prompt TEXT NOT NULL DEFAULT '', "
                      "reason TEXT NOT NULL DEFAULT '');")
    for source, winner in (("human", "a"), ("judge", "b")):        # what session 3 left: one vote, one judge "vote"
        old.execute("INSERT INTO arena_votes (created_at, a_provider, a_model, b_provider, b_model, winner, source) "
                    "VALUES (1, 'openrouter', 'm/one', 'openrouter', 'm/two', ?, ?)", (winner, source))
    old.commit()
    old.close()
    db.init_db()
    votes = db.list_votes()
    assert len(votes) == 2 and all(v["round_id"] is None for v in votes)
    ratings = {r["model"]: r for r in db.compute_ratings(votes)}
    # Only my vote counts now: 1016 / 984 with one game each, not the 998.5 / 1001.5 the judge's "vote" gave.
    assert (ratings["m/one"]["rating"], ratings["m/two"]["rating"]) == (1016, 984)
    assert ratings["m/one"]["games"] == 1 and db.judge_agreement(votes)["rounds"] == 0


def test_thinking_handed_back_as_the_reply_is_not_graded_as_an_answer(monkeypatch):
    """Live, Nvidia's free endpoint returned the thinking as the reply (identical text) on three tests
    that hit the token limit before the answer started. A "contains" grader would pass such a reply
    whenever the thinking happens to mention the right value."""
    test = bench_tests.BY_ID["hard-ctx-ledger"]
    thinking = "East 340 minus 60 is 280, west 251, north 128, so 659 in total. Or is it 683? Let me check again"

    def reply(content, reasoning, truncated=True):
        async def fake_complete(target, messages, **kwargs):
            return providers.Completion(content=content, reasoning=reasoning, truncated=truncated,
                                        finish_reason="length" if truncated else "stop", metrics={"cost": 0.0})
        monkeypatch.setattr(providers, "complete", fake_complete)
        return asyncio.run(benchmark.run_test(test, Target("openrouter", "some/model"), None, 8192))

    assert graders.grade(test["grader"], thinking).score == 1            # what the grader alone would say
    echoed = reply(thinking, thinking)
    assert (echoed["status"], echoed["score"], echoed["response"]) == ("cut_off", 0, "")
    assert echoed["notes"].startswith("Cut off at the token limit: it ran out of tokens (limit 8,192) while still thinking.")
    assert echoed["truncated"] and echoed["reasoning"] == thinking       # the thinking is kept, as thinking
    padded = reply(f"\n{thinking}\n\n", thinking)                        # same text, different whitespace
    assert padded["score"] == 0 and padded["response"] == ""
    # A real answer after real thinking is graded as before, cut off or not.
    assert reply("659", thinking, truncated=False)["status"] == "pass"
    assert reply("The total is 659 jars", thinking)["status"] == "pass"
    assert reply(thinking, "")["status"] == "pass"                       # no separate thinking: it is the reply

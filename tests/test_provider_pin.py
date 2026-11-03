"""A pinned provider (OpenRouter), and a record of which provider served every request.

OpenRouter sends each request to a provider of its choice. Live (2026-10-08) one model was served at
$0.02, $0.05 and $0.22 per million tokens in one afternoon, and its answers changed with the provider.
A pin names the one provider that may serve a model's requests, with fallbacks off: if it cannot serve,
the request fails with an error that says so, and never goes elsewhere. A pin is not one of a run's
settings: a pinned model is a row of its own on the Leaderboard, averaged apart from the same model
unpinned and ranked with every other model at the same settings. A run without a pin is saved exactly
as runs were before pins existed.
"""
import asyncio
import hashlib
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import bench_tests, config, db, fingerprint, graders, mock, providers
from app.main import app
from app.providers import ProviderError, Target

SMART = {"provider": "openrouter", "model": "mock/smart"}
BETA = "beta-cloud/fp8"                       # the mock's second provider: "Beta Cloud", twice the price


def run(coro):
    return asyncio.run(coro)


def ask(model, pin="", provider="openrouter", **kwargs):
    return run(providers.complete(Target(provider, model, pin=pin), [{"role": "user", "content": "Tell me about it"}],
                                  **kwargs))


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def start(client, target=SMART, **body):
    """Run a benchmark to the end -> the saved run, with its results."""
    payload = {"target": target, "categories": ["math", "knowledge"], "concurrency": 4, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    lines = client.get(f"/api/benchmark/{started.json()['run_id']}/events").text.splitlines()
    summary = [json.loads(line[5:]) for line in lines if line.startswith("data:")][-1]
    assert summary["type"] == "summary"
    return client.get(f"/api/runs/{summary['run']['id']}").json()["run"]


# ---------------------------------------------------------------- the request

def test_a_pinned_request_asks_for_that_provider_only_with_fallbacks_off():
    messages = [{"role": "user", "content": "hi"}]
    pinned = providers._payload(Target("openrouter", "a/b", pin="deepinfra/fp8"), messages, 0, 100)
    assert pinned["provider"] == {"only": ["deepinfra/fp8"], "allow_fallbacks": False}
    assert "provider" not in providers._payload(Target("openrouter", "a/b"), messages, 0, 100)
    assert "provider" not in providers._payload(Target("huggingface", "a/b"), messages, 0, 100)


def test_the_pin_is_checked_before_it_is_sent():
    target = providers.resolve_target("openrouter", "mock/smart", pin="  deepinfra/fp8 ")
    assert target.pin == "deepinfra/fp8"
    assert providers.resolve_target("openrouter", "mock/smart").pin == ""
    assert providers.resolve_target("openrouter", "mock/smart", pin="Google AI Studio").pin == "Google AI Studio"
    for bad in ('deepinfra"]}', "a\nb", "{x}", "/etc", "x" * 65):
        with pytest.raises(ProviderError) as info:
            providers.resolve_target("openrouter", "mock/smart", pin=bad)
        assert info.value.code == "bad_request", bad
    # Hugging Face names its provider in the model ID: a pin there would be ignored, so it is refused.
    with pytest.raises(ProviderError) as info:
        providers.resolve_target("huggingface", "mock/smart", pin="alpha")
    assert "OpenRouter" in info.value.message and "model:provider" in info.value.message
    assert providers.resolve_target("huggingface", "mock/smart", pin="  ").pin == ""


def test_every_reply_says_which_provider_served_it():
    assert ask("mock/smart").metrics["served_by"] == "Alpha"            # unpinned: OpenRouter's own choice
    pinned = ask("mock/smart", pin=BETA)
    assert pinned.metrics["served_by"] == "Beta Cloud" and pinned.content
    assert ask("mock/smart", pin="beta-cloud").metrics["served_by"] == "Beta Cloud"     # the slug without its variant
    assert ask("mock/smart", pin="alpha").metrics["served_by"] == "Alpha"
    # Each provider has its own price: the same reply costs twice as much from the dearer one.
    cheap, dear = ask("mock/smart", pin="alpha").metrics, pinned.metrics
    assert dear["tokens_out"] == cheap["tokens_out"] and dear["cost"] == pytest.approx(2 * cheap["cost"])
    # Without a pin the provider can change from one request to the next.
    mock.forget_hits()
    assert [ask("mock/roaming").metrics["served_by"] for _ in range(4)] == ["Alpha", "Beta Cloud"] * 2
    assert [ask("mock/roaming", pin=BETA).metrics["served_by"] for _ in range(3)] == ["Beta Cloud"] * 3


def test_a_pinned_provider_that_cannot_serve_is_a_clear_error_never_another_provider():
    with pytest.raises(ProviderError) as info:
        ask("mock/smart", pin="cerebras", retries=3)
    error = info.value
    assert (error.code, error.status, error.retryable) == ("pin_unavailable", 404, False)
    assert error.attempts == 1                                           # asking again cannot help
    assert error.message.startswith('The pinned provider "cerebras" cannot serve mock/smart (404): ')
    assert "Providers serving mock/smart: alpha, beta-cloud" in error.message          # OpenRouter's own words
    assert error.message.endswith("The request was not sent to another provider.")
    assert "Model not found" not in error.message                        # the model is fine: the pin is not

    # The pinned provider is there but failing: its error, and a note that nothing else was tried.
    with pytest.raises(ProviderError) as info:
        ask("mock/error-429", pin="alpha")
    assert info.value.code == "rate_limited" and info.value.retryable
    assert info.value.message.endswith('[Pinned to "alpha": the request was not sent to another provider.]')
    with pytest.raises(ProviderError) as info:
        ask("mock/midstream", pin="alpha")
    assert "Upstream provider disconnected" in info.value.message and 'Pinned to "alpha"' in info.value.message
    # Without a pin the same errors read as they always did.
    with pytest.raises(ProviderError) as info:
        ask("mock/error-429")
    assert "Pinned" not in info.value.message


def test_a_reply_from_another_provider_than_the_pinned_one_is_refused():
    """With fallbacks off this should not happen. If a router answers from elsewhere all the same, the
    reply is not used: a pinned run never mixes providers without saying so."""
    assert ask("mock/switcher").metrics["served_by"] == "Gamma"          # unpinned: any provider will do
    with pytest.raises(ProviderError) as info:
        ask("mock/switcher", pin="alpha", retries=3)
    assert (info.value.code, info.value.retryable, info.value.attempts) == ("pin_mismatch", False, 1)
    assert 'Pinned to "alpha", but OpenRouter says Gamma served this request' in info.value.message


@pytest.mark.parametrize("pin, served, same", [
    ("deepinfra/fp8", "DeepInfra", True), ("deepinfra", "DeepInfra", True), ("DeepInfra", "DeepInfra", True),
    ("google-ai-studio", "Google AI Studio", True), ("inference-net", "InferenceNet", True),
    ("klusterai", "kluster.ai", True), ("nebius", "Nebius AI Studio", True), ("beta-cloud/fp8", "Beta Cloud", True),
    ("deepinfra/fp8", "Novita", False), ("groq", "DeepInfra", False), ("alpha", "Gamma", False),
    ("deepinfra", "", False), ("/fp8", "DeepInfra", False),
])
def test_the_pin_and_the_name_in_the_reply_are_compared_on_letters_and_digits(pin, served, same):
    assert providers.pin_matches(pin, served) is same


def test_an_upstream_error_names_the_provider_that_failed():
    body = {"error": {"message": "Provider returned error", "code": 429,
                      "metadata": {"raw": "rate limit exceeded", "provider_name": "Novita"}}}
    assert providers._error_detail(json.dumps(body)) == "Provider returned error (rate limit exceeded) [provider: Novita]"
    named = {"error": {"message": "Novita is overloaded", "metadata": {"provider_name": "Novita"}}}
    assert providers._error_detail(json.dumps(named)) == "Novita is overloaded"         # not said twice
    assert providers._error_detail(json.dumps({"error": {"message": "plain"}})) == "plain"


# ---------------------------------------------------------------- the model's provider list

def test_the_providers_of_a_model_are_listed_with_the_slug_to_pin_them_by(client):
    data = client.get("/api/endpoints", params={"model": "mock/smart"}).json()
    assert data["error"] is None and data["model"] == "mock/smart"
    assert [(e["name"], e["slug"]) for e in data["endpoints"]] == [("Alpha", "alpha"), ("Beta Cloud", BETA)]
    alpha, beta = data["endpoints"]
    assert (alpha["price_in"], alpha["price_out"]) == (3.0, 15.0)        # USD per 1M tokens
    assert (beta["price_in"], beta["price_out"]) == (6.0, 30.0)
    assert (beta["quantization"], alpha["quantization"]) == ("fp8", None)
    assert alpha["context_length"] == 200000 and alpha["max_completion_tokens"] == 16384
    # Nothing is asked of OpenRouter for something that is not a model ID.
    for bad in ("", "no-slash", "a/b/../../key", "a/b?x=1", "https://example.com/a"):
        empty = client.get("/api/endpoints", params={"model": bad}).json()
        assert empty["endpoints"] == [] and empty["error"], bad


def test_the_connection_test_says_who_answered(client):
    plain = client.post("/api/test-connection", json=SMART).json()
    assert plain["ok"] and plain["served_by"] == "Alpha" and plain["pin"] == ""
    pinned = client.post("/api/test-connection", json={**SMART, "pin": BETA}).json()
    assert pinned["ok"] and pinned["served_by"] == "Beta Cloud" and pinned["pin"] == BETA
    gone = client.post("/api/test-connection", json={**SMART, "pin": "cerebras"}).json()
    assert not gone["ok"] and gone["code"] == "pin_unavailable"
    assert "was not sent to another provider" in gone["message"]
    wrong = client.post("/api/test-connection", json={"provider": "huggingface", "model": "mock/smart", "pin": "alpha"})
    assert not wrong.json()["ok"] and wrong.json()["code"] == "bad_request"


# ---------------------------------------------------------------- runs

def test_a_run_without_a_pin_is_saved_exactly_as_runs_were_before_pins_existed(client):
    run_ = start(client)
    tests, _ = bench_tests.select_tests(["math", "knowledge"], [], False)
    assert run_["settings"] == {"max_tokens": 8192, "judge": None}          # no "pin" key at all
    assert run_["settings_key"] == "max_tokens=8192;judge=none"
    assert "pin" not in run_["params"]
    # The fingerprint is worked out here the way it was before the change: the grader version, the
    # settings line and one line per test.
    hashes = {test["id"]: fingerprint.digest_of(test) for test in tests}
    lines = [f"grader {graders.GRADER_VERSION}", "settings max_tokens=8192;judge=none"]
    lines += [f"{test_id} {digest}" for test_id, digest in sorted(hashes.items())]
    assert run_["fingerprint"] == hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]
    assert run_["group"] == f"openrouter|mock/smart|standard|max_tokens=8192;judge=none|{run_['fingerprint']}"
    # The key of the fifteen runs of 2026-10-08 is spelled as it was.
    judged = {"max_tokens": 32768, "judge": "openrouter/apodex/apodex-1.1-mini:free"}
    assert fingerprint.settings_key(judged) == "max_tokens=32768;judge=openrouter/apodex/apodex-1.1-mini:free"
    assert fingerprint.settings_key({**judged, "pin": None}) == fingerprint.settings_key({**judged, "pin": ""}) \
        == fingerprint.settings_key(judged)
    # It still says who served it.
    assert run_["served_by"] == {"providers": {"Alpha": run_["total_tests"]}, "unknown": 0}
    assert all(result["served_by"] == ["Alpha"] for result in run_["results"])


def test_a_pinned_run_is_ranked_with_the_others_and_averaged_apart(client):
    plain, pinned = start(client), start(client, target={**SMART, "pin": BETA})
    assert pinned["params"]["pin"] == BETA and pinned["pin"] == BETA and plain["pin"] is None
    # The pin is not a setting: the same settings, key and fingerprint as the run without one.
    assert pinned["settings"] == plain["settings"] == {"max_tokens": 8192, "judge": None}
    assert pinned["settings_key"] == plain["settings_key"] == "max_tokens=8192;judge=none"
    assert pinned["fingerprint"] == plain["fingerprint"]
    # But the model pinned to a provider is a row of its own, named after the provider that served it.
    assert pinned["group"] == f"{plain['group']}|pin={BETA}"
    assert pinned["pin_label"] == "Beta Cloud" and plain["pin_label"] is None
    assert not pinned["older_tests"] and pinned["score"] == plain["score"] == 100.0
    # Every request went to the pinned provider.
    assert pinned["served_by"] == {"providers": {"Beta Cloud": pinned["total_tests"]}, "unknown": 0}
    assert {tuple(result["served_by"]) for result in pinned["results"]} == {("Beta Cloud",)}
    assert pinned["total_cost"] == pytest.approx(2 * plain["total_cost"])   # and was charged at its price
    # Two runs with the same pin are averaged with each other; a run pinned elsewhere is not.
    again, other = start(client, target={**SMART, "pin": BETA}), start(client, target={**SMART, "pin": "alpha"})
    assert again["group"] == pinned["group"] and other["group"] not in (pinned["group"], plain["group"])
    assert other["pin_label"] == "Alpha"

    data = client.get("/api/runs", params={"level": "standard"}).json()
    groups = {group["key"]: group for group in data["groups"]}
    assert all("pin" not in group for group in data["groups"])
    # One group of settings holds them all: they are ranked in one table.
    assert groups[plain["settings_key"]]["runs"] >= 4
    averages = {average["key"]: average for average in data["averages"]}
    assert {pinned["id"], again["id"]} <= set(averages[pinned["group"]]["run_ids"])
    assert averages[pinned["group"]]["pin"] == BETA and averages[pinned["group"]]["pin_label"] == "Beta Cloud"
    assert plain["id"] not in averages[pinned["group"]]["run_ids"]
    listed = {run_["id"]: run_ for run_ in data["runs"]}
    assert listed[pinned["id"]]["served_by"]["providers"] == {"Beta Cloud": pinned["total_tests"]}
    # "Best per model" keeps the pinned model apart from the same model unpinned.
    best = client.get("/api/runs", params={"level": "standard", "best": True,
                                           "settings": plain["settings_key"]}).json()["runs"]
    smart = {(run_["pin"] or "") for run_ in best if run_["model"] == "mock/smart"}
    assert {"", BETA, "alpha"} <= smart

    header, *rows = client.get("/api/runs/export", params={"format": "csv", "level": "standard"}).text.lstrip(
        "﻿").splitlines()
    columns = header.split(",")
    cells = {int(row.split(",")[0]): row.split(",") for row in rows}
    assert cells[pinned["id"]][columns.index("pinned_provider")] == BETA
    assert cells[pinned["id"]][columns.index("served_by")] == "Beta Cloud"
    assert cells[plain["id"]][columns.index("pinned_provider")] == ""
    assert cells[plain["id"]][columns.index("served_by")] == "Alpha"
    assert columns[-2:] == ["tests_fingerprint", "older_tests"]


def test_a_run_saved_with_its_pin_in_the_settings_gets_the_settings_and_fingerprint_of_an_unpinned_one(client):
    """On 2026-10-09 a pin was one of a run's settings: in the settings, at the end of the key and in the
    fingerprint (run #41 of the live database). The app moves such a run to the new rule when it starts."""
    plain = start(client, categories=["math"])
    pinned = start(client, target={**SMART, "pin": BETA}, categories=["math"])
    lines = [f"grader {graders.GRADER_VERSION}", f"settings max_tokens=8192;judge=none;pin={BETA}"]
    lines += [f"{test_id} {digest}" for test_id, digest in sorted(pinned["test_hashes"].items())]
    old_stamp = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]
    with db.connect() as conn:                                              # as session 8 saved it
        conn.execute("UPDATE runs SET settings = ?, fingerprint = ? WHERE id = ?",
                     (json.dumps({"max_tokens": 8192, "judge": None, "pin": BETA}), old_stamp, pinned["id"]))
    assert old_stamp != plain["fingerprint"]
    stored = client.get(f"/api/runs/{pinned['id']}").json()["run"]
    assert stored["older_tests"]                                            # today's code cannot match its fingerprint
    for _ in range(2):                                                      # and a second start changes nothing more
        db.init_db()
        moved = client.get(f"/api/runs/{pinned['id']}").json()["run"]
        assert moved["settings"] == plain["settings"] and moved["settings_key"] == plain["settings_key"]
        assert moved["fingerprint"] == plain["fingerprint"] and not moved["older_tests"]
        assert moved["params"]["pin"] == moved["pin"] == BETA and moved["group"] == f"{plain['group']}|pin={BETA}"
        assert moved["score"] == pinned["score"] and moved["served_by"] == pinned["served_by"]
    # A fingerprint that was not made that way is left as it is.
    settings, stamp = fingerprint.without_pin({"max_tokens": 8192, "judge": None, "pin": BETA},
                                              pinned["test_hashes"], "000000000000")
    assert settings == {"max_tokens": 8192, "judge": None} and stamp == "000000000000"


def test_a_run_served_by_several_providers_says_mixed_with_the_list(client):
    mixed = start(client, target={"provider": "openrouter", "model": "mock/roaming"}, concurrency=1)
    served = mixed["served_by"]
    assert set(served["providers"]) == {"Alpha", "Beta Cloud"} and served["unknown"] == 0
    assert sum(served["providers"].values()) == mixed["total_tests"]
    assert abs(served["providers"]["Alpha"] - served["providers"]["Beta Cloud"]) <= 1
    header, *rows = client.get("/api/runs/export", params={"format": "csv", "level": "standard"}).text.lstrip(
        "﻿").splitlines()
    line = next(row for row in rows if row.startswith(f"{mixed['id']},"))
    assert f"mixed: Alpha {served['providers']['Alpha']}" in line and "Beta Cloud" in line
    # Pinned, the same model stays with one provider.
    pinned = start(client, target={"provider": "openrouter", "model": "mock/roaming", "pin": "alpha"}, concurrency=1)
    assert pinned["served_by"] == {"providers": {"Alpha": pinned["total_tests"]}, "unknown": 0}


def test_a_run_pinned_to_a_provider_that_cannot_serve_scores_nothing_and_says_why(client):
    run_ = start(client, target={**SMART, "pin": "cerebras"}, categories=["math"])
    assert run_["score"] is None and run_["n_error"] == run_["total_tests"] == 5
    assert run_["served_by"] is None                                         # nobody served anything
    for result in run_["results"]:
        assert result["status"] == "error" and result["attempts"] == 1 and result["served_by"] == []
        assert result["error"].startswith('The pinned provider "cerebras" cannot serve mock/smart (404)')
        assert "was not sent to another provider" in result["error"]
    refused = start(client, target={"provider": "openrouter", "model": "mock/switcher", "pin": "alpha"},
                    categories=["math"])
    assert refused["score"] is None and all("says Gamma served" in r["error"] for r in refused["results"])


def test_a_retry_of_failed_requests_goes_to_the_provider_the_run_was_pinned_to(client):
    run_ = start(client, target={"provider": "openrouter", "model": "mock/outage", "pin": BETA}, categories=["math"])
    assert run_["n_error"] == 5 and run_["served_by"] is None               # 429 four times: outlasts the retries
    assert all(result["error"].endswith(f'[Pinned to "{BETA}": the request was not sent to another provider.]')
               for result in run_["results"])
    # The page sends only keys with a retry: the pin is the run's own.
    retried = client.post(f"/api/runs/{run_['id']}/retry", json={"target": {"provider": "openrouter", "model": "x"}})
    assert retried.status_code == 200, retried.text
    assert retried.json()["pin"] == BETA
    client.get(f"/api/benchmark/{run_['id']}/events")                       # follow it to the end
    after = client.get(f"/api/runs/{run_['id']}").json()["run"]
    assert after["n_error"] == 0 and after["score"] == 100.0
    assert after["served_by"] == {"providers": {"Beta Cloud": 5}, "unknown": 0}
    assert after["settings_key"] == run_["settings_key"] and after["fingerprint"] == run_["fingerprint"]


def test_a_reply_asked_for_twice_names_both_requests_providers(client):
    """A reply that is cut short is asked for once more, and both requests were served and paid for."""
    run_ = start(client, target={"provider": "openrouter", "model": "mock/hiccup"}, categories=["math"])
    assert all(result["served_by"] == ["Alpha", "Alpha"] for result in run_["results"])
    assert run_["served_by"] == {"providers": {"Alpha": 10}, "unknown": 0}


def test_a_run_saved_before_providers_were_recorded_says_nothing_about_them(client, tmp_path, monkeypatch):
    run_ = start(client, categories=["math"])
    with db.connect() as conn:                                              # as the runs of 2026-10-08 are stored
        conn.execute("UPDATE runs SET served_by = NULL WHERE id = ?", (run_["id"],))
        conn.execute("UPDATE results SET served_by = NULL WHERE run_id = ?", (run_["id"],))
    old = client.get(f"/api/runs/{run_['id']}").json()["run"]
    assert old["served_by"] is None and all(result["served_by"] == [] for result in old["results"])
    assert (old["settings"], old["settings_key"], old["fingerprint"], old["group"], old["score"]) == (
        run_["settings"], run_["settings_key"], run_["fingerprint"], run_["group"], run_["score"])
    # Worked out again from its results, the count says the replies arrived and named no one.
    assert db.served_summary(old["results"]) == {"providers": {}, "unknown": 5}
    # A database from before the columns existed gets them when the app starts, and keeps its rows.
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    db.init_db()
    with db.connect() as conn:
        conn.execute("ALTER TABLE runs DROP COLUMN served_by")
        conn.execute("ALTER TABLE results DROP COLUMN served_by")
        conn.execute("INSERT INTO runs (id, created_at, status, provider, model, score) "
                     "VALUES (1, 1, 'completed', 'openrouter', 'm', 50)")
        conn.execute("INSERT INTO results (run_id, test_id, category, name, prompt, status, score, response) "
                     "VALUES (1, 't', 'math', 'n', 'p', 'pass', 1.0, 'Answer: 4')")
    db.init_db()
    stored = db.get_run(1)
    assert stored["served_by"] is None and stored["score"] == 50 and stored["results"][0]["served_by"] == []


def test_the_summary_counts_requests_by_provider():
    rows = [{"served_by": ["Groq"], "tokens_out": 5}, {"served_by": ["Groq", "DeepInfra"], "tokens_out": 9},
            {"served_by": ["CoreWeave"], "tokens_out": 1}, {"served_by": ["Groq"], "tokens_out": 2},
            {"served_by": [], "tokens_out": None, "response": ""},          # an API error: no reply, nobody served it
            {"served_by": [], "tokens_out": 7, "response": "x"}]            # a reply that named no provider
    assert db.served_summary(rows) == {"providers": {"Groq": 3, "CoreWeave": 1, "DeepInfra": 1}, "unknown": 1}
    assert list(db.served_summary(rows)["providers"]) == ["Groq", "CoreWeave", "DeepInfra"]   # most requests first
    assert db.served_summary([]) == {"providers": {}, "unknown": 0}


# ---------------------------------------------------------------- the page

def in_node(lines):
    result = subprocess.run(["node", "--input-type=module", "-"], input=chr(10).join(lines), capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_names_one_provider_or_says_mixed_with_the_list():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    cases = [{"providers": {"DeepInfra": 27}, "unknown": 0}, {"providers": {"Groq": 60, "DeepInfra": 9, "CoreWeave": 2},
                                                               "unknown": 0},
             {"providers": {"Groq": 1}, "unknown": 0}, {"providers": {"Groq": 3}, "unknown": 64},
             {"providers": {}, "unknown": 67}, None]
    one, mixed, single, partly, unrecorded, old = in_node([
        f"import {{ servedByText }} from {json.dumps(module)};",
        f"console.log(JSON.stringify({json.dumps(cases)}.map(servedByText)));"])
    assert one == {"label": "DeepInfra", "mixed": False, "detail": "27 requests"}
    assert mixed == {"label": "mixed", "mixed": True, "detail": "Groq 60, DeepInfra 9, CoreWeave 2"}
    assert single["detail"] == "1 request"
    # A run from before providers were recorded, with three tests re-run since: not "one provider".
    assert partly == {"label": "mixed", "mixed": True, "detail": "Groq 3, not named in 64 tests"}
    assert unrecorded is None and old is None                              # nothing recorded: nothing is claimed
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "servedHTML(s.served_by)" in page and "`pinned to ${view.run?.pin || view.pin}`" in page
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    assert "${of.model} · ${of.pin_label || of.pin}" in board and "mixed providers" in board
    assert "group.pin" not in board                                         # a pin is not a group of settings


def page_state(saved, calls):
    """Run state.js in Node with `saved` in the browser's storage -> what each expression in `calls` gives."""
    module = (config.STATIC_DIR / "js" / "state.js").as_uri()
    return in_node([
        f"const saved = {json.dumps(json.dumps(saved))};",
        "globalThis.localStorage = { getItem: (key) => (key === 'modellab.settings' ? saved : null), "
        "setItem() {}, removeItem() {} };",
        f"const state = await import({json.dumps(module)});",
        f"console.log(JSON.stringify([{', '.join(calls)}]));"])


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_sends_a_pin_with_its_model_and_never_with_the_judge():
    saved = {"provider": "openrouter", "models": {"openrouter": "meta/llama", "huggingface": "meta/llama"},
             "pins": {"meta/llama": "deepinfra/fp8"}, "judge": {"provider": "openrouter", "model": "meta/llama"}}
    main, judge, other, hugging, arena = page_state(saved, [
        "state.mainTarget()", "state.judgeTarget()", "state.makeTarget('openrouter', 'other/model')",
        "state.makeTarget('huggingface', 'meta/llama')", "state.makeTarget('openrouter', ' meta/llama ')"])
    assert main == {"provider": "openrouter", "model": "meta/llama", "api_key": "", "pin": "deepinfra/fp8"}
    assert judge["pin"] == "" and judge["model"] == "meta/llama"           # the judge's provider is not a setting
    assert other["pin"] == "" and hugging["pin"] == ""                     # a pin belongs to one OpenRouter model
    assert arena["pin"] == "deepinfra/fp8"                                 # wherever that model is called
    # Settings saved before pins existed: no pin anywhere.
    assert page_state({"models": {"openrouter": "meta/llama"}}, ["state.mainTarget().pin", "state.settings.pins"]) == ["", {}]
    # Setting and clearing it.
    pinned, cleared = page_state({"models": {"openrouter": "a/b"}}, [
        "(state.setPin(' groq '), { ...state.settings.pins })", "(state.setPin(''), { ...state.settings.pins })"])
    assert pinned == {"a/b": "groq"} and cleared == {}

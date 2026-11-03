import asyncio
import json

import httpx
import pytest

from app import providers
from app.providers import ProviderError, Target, ThinkSplitter, parse_model_input


def run(coro):
    return asyncio.run(coro)


def ask(model, provider="openrouter", prompt="Tell me about streaming", **kwargs):
    return run(providers.complete(Target(provider, model), [{"role": "user", "content": prompt}], **kwargs))


# ---------------------------------------------------------------- model IDs and URLs

@pytest.mark.parametrize("text, expected", [
    ("openai/gpt-4o-mini", (None, "openai/gpt-4o-mini")),
    ("  meta-llama/Llama-3.1-8B-Instruct:fastest ", (None, "meta-llama/Llama-3.1-8B-Instruct:fastest")),
    ("https://openrouter.ai/openai/gpt-4o-mini", ("openrouter", "openai/gpt-4o-mini")),
    ("https://openrouter.ai/models/openai/gpt-4o-mini", ("openrouter", "openai/gpt-4o-mini")),
    ("https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free/api",
     ("openrouter", "meta-llama/llama-3.3-70b-instruct:free")),
    ("https://openrouter.ai/meta-llama/llama-3.3-70b-instruct%3Afree",
     ("openrouter", "meta-llama/llama-3.3-70b-instruct:free")),
    ("openrouter.ai/google/gemini-2.0-flash-001", ("openrouter", "google/gemini-2.0-flash-001")),
    ("https://openrouter.ai/api/v1/models/openai/gpt-4o-mini/endpoints", ("openrouter", "openai/gpt-4o-mini")),
    ("https://openrouter.ai/docs/quickstart", ("openrouter", "")),
    ("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct", ("huggingface", "meta-llama/Llama-3.1-8B-Instruct")),
    ("https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main", ("huggingface", "deepseek-ai/DeepSeek-R1")),
    ("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct?inference_provider=together",
     ("huggingface", "Qwen/Qwen2.5-72B-Instruct:together")),
    ("hf.co/Qwen/Qwen2.5-72B-Instruct", ("huggingface", "Qwen/Qwen2.5-72B-Instruct")),
    ("https://huggingface.co/models?pipeline_tag=text-generation", ("huggingface", "")),
    ("https://example.com/a/b", (None, "https://example.com/a/b")),
    ("", (None, "")),
])
def test_parse_model_input(text, expected):
    assert parse_model_input(text) == expected


def test_resolve_target_url_switches_provider():
    target = providers.resolve_target("openrouter", "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct")
    assert (target.provider, target.model) == ("huggingface", "Qwen/Qwen2.5-72B-Instruct")


def test_resolve_target_rejects_bad_input():
    with pytest.raises(ProviderError) as info:
        providers.resolve_target("openrouter", "   ")
    assert info.value.code == "no_model"
    with pytest.raises(ProviderError):
        providers.resolve_target("nope", "a/b")


def test_resolve_target_needs_a_key_outside_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    with pytest.raises(ProviderError) as info:
        providers.resolve_target("openrouter", "openai/gpt-4o-mini")
    assert info.value.code == "no_key" and info.value.status == 401
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert providers.resolve_target("openrouter", "openai/gpt-4o-mini").api_key == "sk-or-env"
    assert providers.resolve_target("openrouter", "openai/gpt-4o-mini", "sk-typed").api_key == "sk-typed"


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("status, code, retryable", [
    (401, "bad_key", False), (402, "no_credits", False), (403, "forbidden", False),
    (404, "not_found", False), (429, "rate_limited", True), (500, "provider_error", True),
    (503, "provider_error", True), (400, "bad_request", False),
])
def test_error_for_status(status, code, retryable):
    error = providers.error_for_status(status, "", "openrouter", "x/y")
    assert (error.code, error.retryable) == (code, retryable)
    assert str(status) in error.message


def test_error_400_invalid_model_is_not_found():
    error = providers.error_for_status(400, "x/y is not a valid model ID", "openrouter", "x/y")
    assert error.code == "not_found"


def test_error_detail_parsing():
    body = '{"error": {"message": "Provider returned error", "metadata": {"raw": "upstream overloaded"}}}'
    assert providers._error_detail(body) == "Provider returned error (upstream overloaded)"
    assert providers._error_detail('{"error": "plain text"}') == "plain text"
    assert providers._error_detail("<html>Bad gateway</html>") == ""
    assert providers._error_detail("") == ""


def test_api_key_is_scrubbed_from_messages():
    assert providers._scrub("bad key sk-or-123456 given", "sk-or-123456") == "bad key *** given"
    target = Target("openrouter", "x/y", "sk-or-123456")
    error = providers._stream_error({"code": 502, "message": "upstream rejected sk-or-123456"}, target)
    assert "sk-or-123456" not in error.message and "***" in error.message


def test_api_key_never_appears_in_a_target_repr():
    target = Target("openrouter", "x/y", "sk-or-123456")
    assert "sk-or-123456" not in repr(target) and "sk-or-123456" not in str(target)
    assert target.api_key == "sk-or-123456"


# ---------------------------------------------------------------- <think> splitting

def split(chunks):
    splitter, reasoning, content = ThinkSplitter(), [], []
    segments = [seg for chunk in chunks for seg in splitter.feed(chunk)] + splitter.flush()
    for kind, text in segments:
        if kind == "reclassify":
            reasoning.extend(content)
            content.clear()
        else:
            (reasoning if kind == "reasoning" else content).append(text)
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize("chunks, expected", [
    (["<think>abc</think>Hello"], ("abc", "Hello")),
    (["<th", "ink>ab", "c</thi", "nk>\n\nHel", "lo"], ("abc", "Hello")),
    (["Hello ", "world"], ("", "Hello world")),
    (["\n\n", "<think>a</think>b"], ("a", "b")),
    (["<THINK>a</THINK>b"], ("a", "b")),
    (["<thinking>a</thinking>b"], ("a", "b")),
    (["<think>never closed"], ("never closed", "")),
    (["thinking out loud", "</think>", "Answer"], ("thinking out loud", "Answer")),
    (["<html>", "x"], ("", "<html>x")),
    (["Hello <think>x"], ("", "Hello <think>x")),
    (["a <", " b"], ("", "a < b")),
])
def test_think_splitter(chunks, expected):
    assert split(chunks) == expected


def test_think_splitter_char_by_char():
    text = "<think>step one\nstep two</think>\n\nThe answer is 4 < 5."
    assert split(list(text)) == ("step one\nstep two", "The answer is 4 < 5.")


# ---------------------------------------------------------------- mock provider end to end

def test_complete_returns_text_and_metrics():
    result = ask("mock/smart")
    assert "mock reply" in result.content and result.finish_reason == "stop" and not result.truncated
    metrics = result.metrics
    assert metrics["tokens_out"] > 0 and metrics["tokens_in"] > 0 and metrics["total"] >= 0
    assert metrics["ttft"] is not None and metrics["tps"] is not None
    assert metrics["cost"] > 0 and not metrics["cost_estimated"] and not metrics["tokens_estimated"]


def test_reasoning_field_is_separated():
    result = ask("mock/thinker")
    assert result.reasoning and "<think>" not in result.content


def test_reasoning_content_field_on_hugging_face():
    result = ask("mock/thinker", provider="huggingface")
    assert result.reasoning and result.content
    assert result.metrics["cost"] is None       # no usage.cost and no price list for Hugging Face


def test_cost_comes_from_the_usage_block_when_the_provider_states_one():
    assert providers.reported_cost({"cost": 0.0012}) == (0.0012, False)            # OpenRouter: charged
    assert providers.reported_cost({"cost": 0}) == (0.0, False)                    # a free model
    assert providers.reported_cost({"estimated_cost": 4.2e-07}) == (4.2e-07, True)  # DeepInfra via Hugging Face
    assert providers.reported_cost({"cost": 0.002, "estimated_cost": 0.001}) == (0.002, False)
    for nothing in ({}, {"cost": None}, {"estimated_cost": None}, {"cost": "0.1"}, {"cost": True}):
        assert providers.reported_cost(nothing) == (None, False)


def test_hugging_face_estimated_cost_is_shown_as_an_estimate():
    result = ask("mock/smart:deepinfra", provider="huggingface")
    assert result.metrics["cost"] > 0 and result.metrics["cost_estimated"]
    assert not result.metrics["tokens_estimated"]
    assert result.metrics["served_by"] == "alpha"                                 # a price list is not needed here
    unlisted = ask("mock/thinker:novita", provider="huggingface")                 # no cost sent, and no listed price
    assert unlisted.metrics["cost"] is None and not unlisted.metrics["cost_estimated"]
    assert unlisted.metrics["served_by"] is None


def test_think_tags_are_separated():
    result = ask("mock/think-tags")
    assert result.reasoning.strip().startswith("The user wants")
    assert "think>" not in result.content and result.content.startswith("Here is a mock reply")


def test_exact_reply_override():
    assert ask("mock/average", prompt="Say it [[mock:hello there]]").content == "hello there"


def test_max_tokens_truncates():
    result = ask("mock/smart", max_tokens=3)
    assert result.truncated and result.finish_reason == "length" and result.metrics["tokens_out"] == 3


def test_cutoff_model_reports_length():
    assert ask("mock/cutoff").truncated


@pytest.mark.parametrize("model, code, status", [
    ("mock/error-401", "bad_key", 401), ("mock/error-402", "no_credits", 402),
    ("mock/error-404", "not_found", 404), ("mock/error-429", "rate_limited", 429),
    ("mock/error-500", "provider_error", 500),
])
def test_http_errors_are_mapped(model, code, status):
    with pytest.raises(ProviderError) as info:
        ask(model)
    assert (info.value.code, info.value.status) == (code, status)


def test_midstream_error_is_raised():
    with pytest.raises(ProviderError) as info:
        ask("mock/midstream")
    assert info.value.code == "provider_error" and info.value.retryable


def test_retry_recovers_from_rate_limits():
    retries = []

    async def on_retry(attempt, delay, exc):
        retries.append((attempt, exc.code))

    result = ask("mock/flaky", prompt="retry test one", retries=3, on_retry=on_retry)
    assert result.attempts == 3 and result.content
    assert retries == [(1, "rate_limited"), (2, "rate_limited")]
    assert len(result.retry_errors) == 2 and all("Rate limited (429)" in e for e in result.retry_errors)
    assert ask("mock/smart").retry_errors == []


def test_no_retry_by_default():
    with pytest.raises(ProviderError) as info:
        ask("mock/flaky", prompt="retry test two")
    assert info.value.code == "rate_limited"


def test_non_retryable_errors_are_not_retried():
    calls = []

    async def on_retry(attempt, delay, exc):
        calls.append(attempt)

    with pytest.raises(ProviderError):
        ask("mock/error-401", retries=3, on_retry=on_retry)
    assert calls == []


def test_backoff_grows_and_honours_retry_after(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    assert providers.backoff_delay(2) > providers.backoff_delay(0)
    assert providers.backoff_delay(0, retry_after=9) >= 9
    assert providers.backoff_delay(10) <= providers.RETRY_MAX_SECONDS


def test_connection_check():
    ok = run(providers.test_connection(Target("openrouter", "mock/smart")))
    assert ok["ok"] and ok["reply"] == "OK" and ok["latency_ms"] >= 0 and not ok["thinking_only"]
    # A reasoning model uses the tiny token budget on thinking: connected, but its thoughts are not a reply.
    thinker = run(providers.test_connection(Target("openrouter", "mock/thinker")))
    assert thinker["ok"] and thinker["reply"] == "" and thinker["thinking_only"]
    bad = run(providers.test_connection(Target("openrouter", "mock/error-402")))
    assert not bad["ok"] and bad["status"] == 402 and bad["code"] == "no_credits"


def test_model_list_and_pricing():
    data = run(providers.get_openrouter_models(force=True))
    by_id = {m["id"]: m for m in data["models"]}
    assert by_id["mock/free:free"]["free"] and not by_id["mock/smart"]["free"]
    assert by_id["mock/smart"]["price_in"] == 3.0 and by_id["mock/smart"]["price_out"] == 15.0
    assert by_id["mock/router"]["price_in"] is None      # variable pricing
    assert providers.pricing_for("openrouter", "mock/smart") == (3.0 / 1e6, 15.0 / 1e6)
    assert providers.pricing_for("anthropic", "mock/smart") is None
    assert providers.pricing_for("openrouter", "mock/router") is None
    # Zero token prices are not enough for the free badge: these two are billed another way.
    for billed in ("mock/per-request", "mock/music"):
        assert not by_id[billed]["free"] and by_id[billed]["other_fees"], billed
        assert by_id[billed]["price_in"] == 0 and by_id[billed]["price_out"] == 0
    assert not by_id["mock/free:free"]["other_fees"] and not by_id["mock/smart"]["other_fees"]


@pytest.mark.parametrize("pricing", [
    {"prompt": "0", "completion": "0"},
    {"prompt": "0", "completion": "0", "request": "0", "image": "0", "web_search": "0",
     "internal_reasoning": "0", "input_cache_read": "0", "input_cache_write": "0"},
    {"prompt": 0, "completion": 0.0, "audio": "0.0"},
    {"prompt": "0", "completion": "0", "discount": 0.5},                       # a discount is not a price
    {"prompt": "0", "completion": "0", "overrides": [{"min_prompt_tokens": 272000, "prompt": "0", "completion": "0"}]},
])
def test_free_when_every_price_is_zero(pricing):
    assert providers.is_free(pricing)


@pytest.mark.parametrize("pricing", [
    {"prompt": "0", "completion": "0", "request": "0.002"},                    # fee per request
    {"prompt": "0", "completion": "0", "image": "0.0048"},                     # fee per image
    {"prompt": "0", "completion": "0", "web_search": "0.01"},
    {"prompt": "0", "completion": "0", "internal_reasoning": "0.000002"},
    {"prompt": "0", "completion": "0", "input_cache_write": "0.0000001"},
    {"prompt": "0", "completion": "0", "audio_output": "0.00001"},
    {"prompt": "0", "completion": "0", "some_new_fee": "0.5"},                 # a field added later still counts
    {"prompt": "0", "completion": "0", "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.000004"}]},
    {"prompt": "0.000001", "completion": "0"},
    {"prompt": "0", "completion": "0.000001"},
    {"prompt": "-1", "completion": "-1"},                                      # router: price varies
    {"prompt": "0", "completion": "0", "request": None},                       # unknown is not provably free
    {"prompt": "0", "completion": "0", "request": "varies"},
    {"prompt": "0", "completion": "0", "request": True},
    {"prompt": "0"}, {"completion": "0"}, {}, None, "0",
])
def test_not_free_when_anything_is_billed_or_unknown(pricing):
    assert not providers.is_free(pricing)


@pytest.mark.parametrize("architecture, free", [
    (None, True), ({}, True),
    ({"output_modalities": ["text"]}, True),
    ({"modality": "text+image->text", "output_modalities": ["text"]}, True),
    ({"modality": "text+image->text"}, True),                                  # older shape, no list
    ({"modality": "text"}, True),
    ({"output_modalities": ["text", "audio"]}, False),                         # billed per clip
    ({"output_modalities": ["image"]}, False),
    ({"modality": "text+image->text+audio"}, False),
    ({"modality": "text->image"}, False),
])
def test_free_needs_text_only_output(architecture, free):
    zero = {"prompt": "0", "completion": "0"}
    assert providers.is_free(zero, architecture) is free
    assert not providers.is_free({"prompt": "0.000001", "completion": "0"}, architecture)


def test_simplify_model_matches_the_real_openrouter_shapes():
    # Copied from the live list on 2026-10-07: a music model that costs $0.04 per clip, a genuinely
    # free text model without the ":free" suffix, and an ordinary free one.
    lyria = providers.simplify_model({
        "id": "google/lyria-3-clip-preview", "context_length": 1048576,
        "architecture": {"modality": "text+image->text+audio", "output_modalities": ["text", "audio"]},
        "pricing": {"prompt": "0", "completion": "0"}})
    assert not lyria["free"] and lyria["other_fees"] and lyria["price_out"] == 0
    for model_id in ("inclusionai/ling-3.1-flash", "apodex/apodex-1.1-mini:free"):
        text = providers.simplify_model({
            "id": model_id, "architecture": {"modality": "text->text", "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"}})
        assert text["free"] and not text["other_fees"], model_id
    paid = providers.simplify_model({"id": "x/y", "pricing": {"prompt": "0.00000005", "completion": "0.00000008"}})
    assert not paid["free"] and not paid["other_fees"] and paid["price_in"] == 0.05 and paid["price_out"] == 0.08


def test_model_cache_from_an_older_version_is_not_trusted(tmp_path, monkeypatch):
    # Before version 2 the cache stored "free" worked out from the token prices only.
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(providers, "_models", {"fetched_at": 0.0, "models": [], "by_id": {}})
    model = {"id": "a/b", "name": "A", "context_length": 1, "price_in": 0.0, "price_out": 0.0, "free": True}
    cache = tmp_path / "openrouter_models.json"

    cache.write_text(json.dumps({"fetched_at": 5.0, "models": [model]}), encoding="utf-8")
    assert not providers._load_models_from_disk() and providers._models["models"] == []
    cache.write_text("[1, 2]", encoding="utf-8")                              # not even an object
    assert not providers._load_models_from_disk()
    cache.write_text(json.dumps({"version": providers._MODELS_CACHE_VERSION, "fetched_at": 5.0,
                                 "models": [model]}), encoding="utf-8")
    assert providers._load_models_from_disk() and providers._models["by_id"]["a/b"]["free"]


def test_rejected_optional_parameters_are_dropped_and_retried():
    result = ask("mock/picky", temperature=0, max_tokens=5)
    assert result.content and result.truncated          # max_completion_tokens was honoured
    assert result.metrics["tokens_out"] == 5


def test_drop_rejected_param_only_touches_named_parameters():
    payload = {"model": "x", "temperature": 0, "max_tokens": 9, "stream_options": {}, "usage": {}}
    assert providers._drop_rejected_param(payload, "Unsupported parameter: 'temperature'")
    assert "temperature" not in payload and "max_tokens" in payload
    assert providers._drop_rejected_param(payload, "use max_completion_tokens instead of max_tokens")
    assert payload["max_completion_tokens"] == 9 and "max_tokens" not in payload
    assert providers._drop_rejected_param(payload, "unknown field stream_options")
    assert "stream_options" not in payload and "usage" not in payload
    assert not providers._drop_rejected_param(payload, "messages must not be empty")


def test_ipv4_first_transport_falls_back_and_remembers_the_host():
    calls = []

    class Unreachable(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls.append("ipv4")
            raise httpx.ConnectError("network is unreachable")

    class Working(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls.append("default")
            return httpx.Response(200, json={"ok": True})

    transport = providers.PreferIPv4Transport()
    transport._ipv4, transport._default = Unreachable(), Working()

    async def scenario():
        async with httpx.AsyncClient(transport=transport) as client:
            first = await client.get("https://example.test/a")
            second = await client.get("https://example.test/b")
            return first.status_code, second.status_code

    assert run(scenario()) == (200, 200)
    assert calls == ["ipv4", "default", "default"]      # the second request skips IPv4 for that host


# ---------------------------------------------------------------- Hugging Face router model list

LLAMA_ON_HF = {   # the real shape, copied from router.huggingface.co/v1/models (prices are USD per 1M tokens)
    "id": "meta-llama/Llama-3.1-8B-Instruct", "object": "model", "owned_by": "meta-llama",
    "providers": [
        {"provider": "novita", "status": "live", "context_length": 16384, "pricing": {"input": 0.02, "output": 0.05}},
        {"provider": "nscale", "status": "live", "context_length": 131072, "pricing": {"input": 0.06, "output": 0.06}},
        {"provider": "featherless-ai", "status": "live", "is_free": False},
        {"provider": "deepinfra", "status": "live", "context_length": 131072, "pricing": {"input": 0.02, "output": 0.05}},
    ],
}


def test_simplify_hf_model_keeps_every_provider():
    model = providers.simplify_hf_model(LLAMA_ON_HF)
    assert model["id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert model["context_length"] == 131072                       # the largest any provider offers
    assert (model["price_in"], model["price_out"]) == (0.02, 0.05)  # the cheapest provider
    assert [p["name"] for p in model["providers"]] == ["novita", "nscale", "featherless-ai", "deepinfra"]
    assert model["providers"][0] == {"name": "novita", "status": "live", "context_length": 16384,
                                     "price_in": 0.02, "price_out": 0.05}
    unpriced = model["providers"][2]
    assert (unpriced["context_length"], unpriced["price_in"], unpriced["price_out"]) == (None, None, None)
    assert not model["free"]

    bare = providers.simplify_hf_model({"id": "x/y", "providers": [{"provider": "p", "status": "live"}]})
    assert (bare["context_length"], bare["price_in"], bare["price_out"]) == (None, None, None)
    odd = providers.simplify_hf_model({"id": "x/z", "providers": [
        {"provider": "p", "pricing": {"input": "0.1", "output": True}, "context_length": -5}, {"status": "live"}, "junk"]})
    assert odd["providers"] == [{"name": "p", "status": None, "context_length": None, "price_in": None, "price_out": None}]
    assert providers.simplify_hf_model({"id": "x/none"})["providers"] == []


def test_hf_route_picks_the_provider_whose_price_applies(monkeypatch):
    model = providers.simplify_hf_model(LLAMA_ON_HF)
    monkeypatch.setattr(providers, "_hf_models", {"fetched_at": 1.0, "models": [model],
                                                 "by_id": {model["id"]: model, model["id"].lower(): model}})
    name = "meta-llama/Llama-3.1-8B-Instruct"
    route = providers.hf_route
    assert route(name, served_by="nscale")["name"] == "nscale"            # who really served it comes first
    assert route(f"{name}:novita", served_by="nscale")["name"] == "nscale"
    assert route(f"{name}:nscale")["name"] == "nscale"                    # then the provider asked for
    assert route(f"{name}:cheapest")["name"] == "novita"
    assert route(name)["name"] == "novita" and route(f"{name}:fastest")["name"] == "novita"   # the router's first priced one
    assert route(name.lower())["name"] == "novita"                        # IDs are matched without regard to case
    assert route("someone/unknown") is None

    per_token = providers.pricing_for
    assert per_token("huggingface", name, "nscale") == (0.06 / 1e6, 0.06 / 1e6)
    assert per_token("huggingface", name) == (0.02 / 1e6, 0.05 / 1e6)
    assert per_token("huggingface", name, "featherless-ai") is None       # that provider lists no price
    assert per_token("huggingface", "someone/unknown") is None


def test_hugging_face_list_is_served_and_cached(tmp_path, monkeypatch):
    data = run(providers.get_huggingface_models(force=True))
    by_id = {m["id"]: m for m in data["models"]}
    assert data["error"] is None and set(by_id) == {"mock/smart", "mock/average", "mock/weak"}
    assert (by_id["mock/smart"]["price_in"], by_id["mock/smart"]["price_out"]) == (2.0, 12.0)    # beta is cheaper
    assert by_id["mock/weak"]["price_in"] is None and by_id["mock/weak"]["context_length"] is None
    assert run(providers.get_models("huggingface"))["models"] == data["models"]
    assert run(providers.get_models("openrouter"))["models"][0]["id"].startswith("mock/")

    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("MODEL_LAB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(providers, "_hf_models", {"fetched_at": 0.0, "models": [], "by_id": {}})
    assert not providers._load_hf_models_from_disk()                      # nothing cached yet
    model = providers.simplify_hf_model(LLAMA_ON_HF)
    (tmp_path / "huggingface_models.json").write_text(json.dumps({"fetched_at": 5.0, "models": [model]}), encoding="utf-8")
    assert not providers._load_hf_models_from_disk()                      # no version: not trusted
    (tmp_path / "huggingface_models.json").write_text(
        json.dumps({"version": providers._HF_CACHE_VERSION, "fetched_at": 5.0, "models": [model]}), encoding="utf-8")
    assert providers._load_hf_models_from_disk()
    assert providers.pricing_for("huggingface", "meta-llama/llama-3.1-8b-instruct:deepinfra") == (0.02 / 1e6, 0.05 / 1e6)


def test_hugging_face_cost_is_estimated_from_the_price_list_when_none_is_sent():
    run(providers.get_huggingface_models(force=True))
    result = ask("mock/smart", provider="huggingface")                    # served by "alpha": 3 / 15 per 1M
    metrics = result.metrics
    assert metrics["served_by"] == "alpha" and metrics["cost_estimated"]
    assert metrics["cost"] == pytest.approx((metrics["tokens_in"] * 3.0 + metrics["tokens_out"] * 15.0) / 1e6)
    beta = ask("mock/smart:beta", provider="huggingface").metrics         # the provider asked for: 2 / 12 per 1M
    assert beta["served_by"] == "beta" and beta["cost_estimated"]
    assert beta["cost"] == pytest.approx((beta["tokens_in"] * 2.0 + beta["tokens_out"] * 12.0) / 1e6)
    unpriced = ask("mock/weak", provider="huggingface").metrics           # "gamma" lists no price: still a dash
    assert unpriced["served_by"] == "gamma" and unpriced["cost"] is None and not unpriced["cost_estimated"]
    stated = ask("mock/smart:deepinfra", provider="huggingface").metrics  # a cost the provider sent wins over the list
    assert stated["cost_estimated"] and stated["cost"] > 0
    openrouter = ask("mock/smart").metrics                                # OpenRouter states the charge itself ...
    assert not openrouter["cost_estimated"]
    assert openrouter["served_by"] == "Alpha"                             # ... and names the provider in the reply

"""Offline stand-in for the providers, enabled with MOCK=1.

It is implemented as an httpx transport, so the real request building, SSE
parsing, retry and error-mapping code all run exactly as they do against a
live provider - only the network is replaced.

Model IDs select behaviour (any other ID acts like an average model):
  mock/smart, mock/average, mock/weak   different benchmark skill and speed
  mock/thinker                          sends reasoning deltas
  mock/think-tags                       wraps thinking in <think> tags
  mock/slow                             long time-to-first-token, slow stream
  mock/free:free                        zero price (free badge)
  mock/per-request                      zero token prices but a fee per request (no free badge)
  mock/music                            zero prices, audio output billed per clip (no free badge)
  mock/flaky                            429 on the first two tries of each prompt
  mock/cutoff                           always stops early (finish_reason=length)
  mock/hiccup                           the first reply to each prompt is cut short mid-sentence, with
                                        finish_reason=stop (a provider that ends the stream early)
  mock/dropper                          the same for the first two replies to each prompt
  mock/flaky-hiccup                     429 on the first try of each prompt, then a reply that is cut short:
                                        the reply is asked for twice, and the first request needed two attempts
  mock/mute                             thinks, then writes nothing, every time (an empty reply with
                                        tokens to spare)
  mock/mute-once                        the same for the first reply to each prompt only
  mock/outage                           429 on the first four tries of each prompt (outlasts the retries)
  mock/midstream                        fails halfway through the stream
  mock/picky                            rejects temperature, max_tokens and stream_options with 400s
  mock/error-401 | 402 | 404 | 429 | 500   always returns that HTTP error
  mock/roaming                          OpenRouter: served by another provider from one request to the next
  mock/switcher                         OpenRouter: ignores a pinned provider and answers from "Gamma"
On OpenRouter every model has two providers, "Alpha" (slug alpha) and "Beta Cloud" (slug beta-cloud/fp8,
twice the price). A request goes to Alpha unless it is pinned to the other one ("provider": {"only": [...]});
a pin to any other provider gets OpenRouter's 404. Every chunk names the provider that served it.
On Hugging Face no cost is reported, unless the ID ends in ":deepinfra" (then an estimated cost is).
The Hugging Face router list has mock/smart (providers alpha and beta), mock/average (alpha) and
mock/weak (gamma, which lists no price); the reply names the provider that "served" it.
A prompt containing [[mock:TEXT]] makes the model reply with exactly TEXT.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, AsyncIterator

import httpx

from .graders import grade_admits_unknown

PROFILES: dict[str, dict[str, Any]] = {
    "mock/smart": {"skill": 0.97, "tps": 95, "ttft": 0.28, "price": (3.0, 15.0), "ctx": 200000},
    "mock/average": {"skill": 0.72, "tps": 62, "ttft": 0.38, "price": (0.5, 1.5), "ctx": 128000},
    "mock/weak": {"skill": 0.38, "tps": 150, "ttft": 0.14, "price": (0.05, 0.1), "ctx": 32000},
    "mock/thinker": {"skill": 0.9, "tps": 70, "ttft": 0.5, "price": (1.0, 4.0), "ctx": 128000, "think": "field"},
    "mock/think-tags": {"skill": 0.82, "tps": 70, "ttft": 0.3, "price": (0.4, 1.2), "ctx": 64000, "think": "tags"},
    "mock/slow": {"skill": 0.76, "tps": 14, "ttft": 1.3, "price": (0.2, 0.6), "ctx": 16000},
    "mock/free:free": {"skill": 0.6, "tps": 48, "ttft": 0.45, "price": (0.0, 0.0), "ctx": 8192},
    "mock/flaky": {"skill": 0.8, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "flaky": 2},
    "mock/cutoff": {"skill": 0.8, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "cutoff": 6},
    "mock/hiccup": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "drop": 1},
    "mock/dropper": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "drop": 2},
    "mock/flaky-hiccup": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "flaky": 1,
                          "drop": 1},
    "mock/mute": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "think": "field",
                  "mute": 10 ** 9},
    "mock/mute-once": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "think": "field",
                       "mute": 1},
    "mock/outage": {"skill": 1.0, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "flaky": 4},
    "mock/midstream": {"skill": 0.8, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "midstream": True},
    "mock/picky": {"skill": 0.8, "tps": 80, "ttft": 0.2, "price": (0.3, 0.9), "ctx": 64000, "picky": True},
    "mock/error-401": {"error": 401},
    "mock/error-402": {"error": 402},
    "mock/error-404": {"error": 404},
    "mock/error-429": {"error": 429},
    "mock/error-500": {"error": 500},
}
_NAMES = {
    "mock/smart": "Mock: Smart", "mock/average": "Mock: Average", "mock/weak": "Mock: Weak",
    "mock/thinker": "Mock: Thinker (reasoning field)", "mock/think-tags": "Mock: Thinker (<think> tags)",
    "mock/slow": "Mock: Slow", "mock/free:free": "Mock: Free (free)", "mock/flaky": "Mock: Flaky (429 then OK)",
    "mock/cutoff": "Mock: Cut off (finish_reason=length)", "mock/midstream": "Mock: Fails mid-stream",
}
_ERRORS = {
    401: "No auth credentials found",
    402: "Insufficient credits. Add more using your account page",
    404: "No endpoints found for this model",
    429: "Rate limit exceeded: free-models-per-min",
    500: "Internal server error",
}

_flaky_hits: dict[str, int] = {}
_drop_hits: dict[str, int] = {}
_mute_hits: dict[str, int] = {}
_roaming_hits: dict[str, int] = {}


def forget_hits() -> None:
    """Start the per-prompt counters of the flaky, dropping and mute models again (used by the tests)."""
    _flaky_hits.clear()
    _drop_hits.clear()
    _mute_hits.clear()
    _roaming_hits.clear()


# OpenRouter's providers behind every mock model: (the name replies carry, the slug a pin uses, how many
# times the model's listed price it charges). Live, one model's providers differed elevenfold in price.
OR_ENDPOINTS = [("Alpha", "alpha", 1.0), ("Beta Cloud", "beta-cloud/fp8", 2.0)]


def _slug_base(slug: str) -> str:
    return re.sub(r"[^a-z0-9]", "", slug.split("/")[0].lower())


def endpoint_list(model: str) -> dict[str, Any]:
    """Mock version of OpenRouter's GET /api/v1/models/{model}/endpoints payload."""
    profile = _profile(model)
    price_in, price_out = profile.get("price", (0.5, 1.5))
    return {"id": model, "name": _NAMES.get(model, model), "endpoints": [
        {"name": f"{name} | {model}", "provider_name": name, "tag": slug,
         "quantization": slug.partition("/")[2] or "unknown",
         "context_length": profile.get("ctx", 32000), "max_completion_tokens": 16384,
         "pricing": {"prompt": str(price_in * times / 1e6), "completion": str(price_out * times / 1e6)}}
        for name, slug, times in OR_ENDPOINTS]}


def _or_served_by(model: str, body: dict[str, Any]) -> tuple[str, float] | httpx.Response:
    """The provider the mock OpenRouter "sends" a request to -> (its name, its price factor), or the 404
    OpenRouter answers with when the request is pinned to a provider that does not serve the model."""
    routing = body.get("provider") if isinstance(body.get("provider"), dict) else {}
    only = [str(slug) for slug in routing.get("only") or []]
    if model.split(":")[0] == "mock/switcher":
        return "Gamma", 1.0                      # a router that does not honour the pin
    if only:
        wanted = {_slug_base(slug) for slug in only}
        for name, slug, times in OR_ENDPOINTS:
            if _slug_base(slug) in wanted:
                return name, times
        serving = ", ".join(slug.split("/")[0] for _, slug, _ in OR_ENDPOINTS)
        return _json_response(404, {"error": {"code": 404, "message": (
            f"No allowed providers are available for the selected model. Providers serving {model}: {serving}, "
            f"but your request's provider.only preference permits only: {', '.join(only)}.")}})
    if model.split(":")[0] == "mock/roaming":
        _roaming_hits[model] = _roaming_hits.get(model, 0) + 1
        name, _, times = OR_ENDPOINTS[(_roaming_hits[model] - 1) % len(OR_ENDPOINTS)]
        return name, times
    return OR_ENDPOINTS[0][0], OR_ENDPOINTS[0][2]


def _cut_short(content: str, hit: int = 1) -> str:
    """A reply as a provider leaves it when it ends the stream early: a few words, then a comma.

    A stream that is dropped twice is not dropped at the same word twice, so the second cut ends a word
    earlier, on a semicolon. (The same cut twice, word for word, is read as the model's own ending and
    graded: run_test in app/benchmark.py.)"""
    words = content.split()[:6]
    if hit > 1:
        return " ".join(words[:-1] or words).rstrip(".,;:!?") + ";"
    return " ".join(words).rstrip(".,;:!?") + ","
_registry_cache: dict[str, dict[str, Any]] | None = None


def model_list() -> list[dict[str, Any]]:
    """Mock version of OpenRouter's GET /api/v1/models payload."""
    items = []
    for model_id, name in _NAMES.items():
        profile = PROFILES[model_id]
        items.append({
            "id": model_id, "name": name, "context_length": profile["ctx"],
            "pricing": {"prompt": str(profile["price"][0] / 1e6), "completion": str(profile["price"][1] / 1e6)},
        })
    items.append({"id": "mock/router", "name": "Mock: Auto Router", "context_length": 200000,
                  "pricing": {"prompt": "-1", "completion": "-1"}})
    items.append({"id": "mock/per-request", "name": "Mock: Billed per request", "context_length": 32000,
                  "pricing": {"prompt": "0", "completion": "0", "request": "0.002", "image": "0"}})
    # Like google/lyria-3-clip-preview: every listed price is zero, but each audio clip is billed.
    items.append({"id": "mock/music", "name": "Mock: Music (billed per clip)", "context_length": 32000,
                  "pricing": {"prompt": "0", "completion": "0"},
                  "architecture": {"modality": "text->text+audio", "output_modalities": ["text", "audio"]}})
    return items


# Hugging Face's router list: one model, several providers, each with its own context size and
# price in USD per 1M tokens. Some providers list neither (seen live: featherless-ai).
HF_PROVIDERS: dict[str, list[dict[str, Any]]] = {
    "mock/smart": [
        {"provider": "alpha", "status": "live", "context_length": 200000, "pricing": {"input": 3.0, "output": 15.0}},
        {"provider": "beta", "status": "live", "context_length": 128000, "pricing": {"input": 2.0, "output": 12.0}},
    ],
    "mock/average": [
        {"provider": "alpha", "status": "live", "context_length": 128000, "pricing": {"input": 0.5, "output": 1.5}},
    ],
    "mock/weak": [{"provider": "gamma", "status": "live"}],
}


def hf_model_list() -> list[dict[str, Any]]:
    """Mock version of Hugging Face's GET /v1/models payload."""
    return [{"id": model_id, "object": "model", "owned_by": "mock", "providers": routes}
            for model_id, routes in HF_PROVIDERS.items()]


def _hf_served_by(model: str) -> str | None:
    """The provider the mock router "sends" a request to: the one named in the suffix, else the first."""
    name, _, suffix = model.partition(":")
    routes = [route["provider"] for route in HF_PROVIDERS.get(name, [])]
    return suffix if suffix in routes else (routes[0] if routes else None)


def _digest(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _roll(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from the inputs."""
    return int(_digest(*parts)[:8], 16) / 0x100000000


def _profile(model: str) -> dict[str, Any]:
    if model in PROFILES:
        return PROFILES[model]
    base = model.split(":")[0]
    if base in PROFILES:
        return PROFILES[base]
    return {"skill": 0.5 + 0.4 * _roll(model, "skill"), "tps": 45 + 70 * _roll(model, "tps"),
            "ttft": 0.2 + 0.4 * _roll(model, "ttft"), "price": (0.5, 1.5), "ctx": 32000}


def _registry() -> dict[str, dict[str, Any]]:
    global _registry_cache
    if _registry_cache is None:
        from .bench_tests import BUILTIN
        _registry_cache = {_digest(t["prompt"]): t for t in BUILTIN}
    return _registry_cache


_STRICT_VERDICTS = {
    "good": {"deductions": ["-1: the last sentence could be sharper"], "score": 9,
             "reason": "Mock judge: meets the brief and is specific."},
    "mid": {"deductions": ["-2: generic - it could be about anyone", "-2: flat, list-like sentences"], "score": 6,
            "reason": "Mock judge: does the job but is forgettable."},
    "bad": {"deductions": ["-4: ignores most of the brief", "-3: careless and far too thin"], "score": 3,
            "reason": "Mock judge: does not do what was asked."},
}
_samples_cache: dict[str, str] | None = None


def _writing_samples() -> dict[str, str]:
    """digest of a canned writing answer -> "good" | "mid" | "bad"."""
    global _samples_cache
    if _samples_cache is None:
        from .bench_tests import BUILTIN
        _samples_cache = {
            _digest(test[key].strip()): quality
            for test in BUILTIN if test["grader"].get("strict")
            for key, quality in (("mock_good", "good"), ("mock_mid", "mid"), ("mock_bad", "bad"))}
    return _samples_cache


def _fast() -> bool:
    return os.environ.get("MOCK_FAST", "").strip().lower() in {"1", "true", "yes", "on"}


_ARABIC = re.compile(r"[؀-ۿ]")

_DEMO = """Here is a mock reply to **"{topic}"**. Mock mode runs entirely on your machine, so no API key or credits are used.

### What this shows
- Streaming with *time to first token* and `tokens/sec`
- Markdown: lists, tables, links and code blocks
- Token counts and a simulated cost under each reply

```python
def greet(name: str) -> str:
    \"\"\"Return a friendly greeting.\"\"\"
    return f"Hello, {{name}}!"


print(greet("Model Lab"))
```

| Feature | Status |
|---|---|
| Streaming | Working |
| Markdown tables | Working |
| Cost tracking | Simulated |

> Turn mock mode off and add an API key to talk to a real model.
"""

_ARABIC_REPLY = (
    "مرحباً! هذا رد تجريبي "
    "من النموذج الوهمي.\n\n"
    "- النص العربي يظهر من "
    "اليمين إلى اليسار\n"
    "- يمكن خلط العربية مع "
    "English في السطر نفسه\n\n"
    "لا يُستخدم أي مفتاح API "
    "في الوضع التجريبي."
)

_PARAGRAPH = (
    "The bicycle began as a wooden running machine in 1817, when Karl von Drais built a two-wheeled frame "
    "that riders pushed along with their feet. Pedals arrived in the 1860s, fixed directly to the front wheel, "
    "and the high-wheeled penny-farthing followed - fast, but dangerous to fall from. The real breakthrough "
    "was the safety bicycle of the 1880s, with two equal wheels and a chain driving the rear one. Pneumatic "
    "tyres soon made the ride comfortable, and cycling boomed. Bicycles gave ordinary people cheap, independent "
    "travel, and they played a part in changing fashion and widening horizons for women. The twentieth century "
    "added gears, lighter alloys and specialised designs for racing, touring and mountains. Today the same "
    "simple machine is being rediscovered as a clean answer to crowded, polluted cities."
)


def _generic_reply(user: str) -> str:
    low = user.lower().strip()
    if low.startswith("reply with the single word"):
        return "OK"
    if _ARABIC.search(user):
        return _ARABIC_REPLY
    if len(low) < 16 and re.match(r"^(hi|hello|hey|yo|test|ping)\b", low):
        return "Hello! I'm a mock model running locally. Ask me anything and I'll stream back a sample reply."
    if "paragraph" in low or "150-word" in low or "150 word" in low:
        return _PARAGRAPH
    topic = re.sub(r"\s+", " ", user).strip()
    if len(topic) > 70:
        topic = topic[:67].rstrip() + "..."
    return _DEMO.format(topic=topic.replace("{", "(").replace("}", ")"))


def _judge_reply(model: str, system: str, user: str) -> str | None:
    text = system + "\n" + user
    if "Return only a JSON object" not in text:
        return None
    # The wanted shape is spelled out at the end of the prompt. Look only there: the middle of
    # the prompt quotes model output, which can contain anything.
    form = user.rpartition("JSON object of the form")[2]
    if '"admits"' in form:
        # Honesty verdict: the mock judge reads the quoted reply with the phrase check.
        reply = user.partition("[Model response]\n")[2].rpartition("\n\n")[0]
        admits = grade_admits_unknown(reply).score == 1
        reason = "says it cannot know and makes nothing up." if admits else "answers as if it knew."
        return json.dumps({"admits": admits, "invents": not admits, "reason": f"Mock judge: {reason}"})
    if '"deductions"' in form:
        # Strict writing verdict: the mock judge knows the canned good, average and bad answers.
        reply = user.partition("[Model response]\n")[2].rpartition("\n\n")[0]
        return json.dumps(_STRICT_VERDICTS[_writing_samples().get(_digest(reply.strip()), "mid")])
    if '"winner"' in form:
        roll = _roll(model, user, "winner")
        winner = "tie" if roll < 0.15 else ("A" if roll < 0.58 else "B")
        reason = ("Both responses are comparable in accuracy and clarity." if winner == "tie" else
                  f"Response {winner} is more complete and easier to follow.")
        return json.dumps({"winner": winner, "reason": reason})
    if '"score"' in form:
        score = 5 + int(_roll(model, user, "score") * 5)
        return json.dumps({"score": score, "reason": "Mock judge: on-brief and clear, with small weaknesses."})
    return None


def _build_reply(model: str, profile: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (reasoning, content) for a request."""
    user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
    system = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")

    override = re.search(r"\[\[mock:(.*?)\]\]", user, re.S)
    if override:
        content = override.group(1)
    elif (judged := _judge_reply(model, system, user)) is not None:
        content = judged
    elif (test := _registry().get(_digest(user))) is not None:
        good = _roll(model, test["id"]) < profile.get("skill", 0.7)
        content = test["mock_good"] if good else test["mock_bad"]
    else:
        content = _generic_reply(user)

    reasoning = ""
    if profile.get("think"):
        reasoning = ("The user wants a direct answer. Let me break the request down, check the constraints "
                     "and make sure the final reply follows the requested format before I write it out.")
        if profile["think"] == "tags":
            content, reasoning = f"<think>\n{reasoning}\n</think>\n\n{content}", ""
    return reasoning, content


def _pieces(text: str) -> list[str]:
    """Split text into token-sized pieces for streaming."""
    return re.findall(r"\S+\s*|\s+", text)


class _Stream(httpx.AsyncByteStream):
    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            yield chunk

    async def aclose(self) -> None:
        aclose = getattr(self._source, "aclose", None)
        if aclose:
            await aclose()


def _json_response(status: int, payload: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "application/json", **(headers or {})},
                          content=json.dumps(payload).encode("utf-8"))


def _error_response(status: int) -> httpx.Response:
    headers = {"retry-after": "0"} if status == 429 else None
    return _json_response(status, {"error": {"message": _ERRORS.get(status, "Error"), "code": status}}, headers)


class MockTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/endpoints"):
            model = request.url.path.partition("/models/")[2].rpartition("/endpoints")[0]
            return _json_response(200, {"data": endpoint_list(model)})
        if request.method == "GET" and request.url.path.endswith("/models"):
            listing = hf_model_list() if "huggingface" in request.url.host else model_list()
            return _json_response(200, {"object": "list", "data": listing})

        await request.aread()
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:
            return _json_response(400, {"error": {"message": "Invalid JSON body", "code": 400}})
        model = str(body.get("model", ""))
        messages = body.get("messages") or []
        provider = "huggingface" if "huggingface" in request.url.host else "openrouter"
        profile = _profile(model)

        if profile.get("error"):
            return _error_response(profile["error"])
        if profile.get("picky"):
            for param, message in (
                    ("stream_options", "Unknown parameter: 'stream_options'."),
                    ("max_tokens", "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."),
                    ("temperature", "Unsupported value: 'temperature' does not support 0 with this model.")):
                if param in body:
                    return _json_response(400, {"error": {"message": message, "code": 400}})
        if profile.get("flaky"):
            key = _digest(model, json.dumps(messages, sort_keys=True))
            _flaky_hits[key] = _flaky_hits.get(key, 0) + 1
            if _flaky_hits[key] <= profile["flaky"]:
                return _error_response(429)
        # OpenRouter: which provider serves it. A pin to a provider that does not serve the model is a 404.
        routed = _or_served_by(model, body) if provider == "openrouter" else None
        if isinstance(routed, httpx.Response):
            return routed

        reasoning, content = _build_reply(model, profile, messages)
        if profile.get("drop"):
            key = _digest(model, "drop", json.dumps(messages, sort_keys=True))
            _drop_hits[key] = _drop_hits.get(key, 0) + 1
            if _drop_hits[key] <= profile["drop"]:
                content = _cut_short(content, _drop_hits[key])
        if profile.get("mute") and "Return only a JSON object" not in json.dumps(messages):     # a judge still answers
            key = _digest(model, "mute", json.dumps(messages, sort_keys=True))
            _mute_hits[key] = _mute_hits.get(key, 0) + 1
            if _mute_hits[key] <= profile["mute"]:
                content = ""              # the thinking is sent, then the stream ends with "stop"
        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if profile.get("cutoff"):
            max_tokens = min(max_tokens or profile["cutoff"], profile["cutoff"])
        stream = self._sse(model, provider, profile, reasoning, content, max_tokens, prompt_tokens, routed)
        headers = {"content-type": "text/event-stream"}
        served_by = _hf_served_by(model) if provider == "huggingface" else None
        if served_by:
            headers["x-inference-provider"] = served_by
        return httpx.Response(200, headers=headers, stream=_Stream(stream))

    async def _sse(self, model: str, provider: str, profile: dict[str, Any], reasoning: str,
                   content: str, max_tokens: int | None, prompt_tokens: int,
                   routed: tuple[str, float] | None = None) -> AsyncIterator[bytes]:
        chat_id = "mock-" + _digest(model, content)[:12]
        # OpenRouter names the provider that served the request in every chunk.
        served = {"provider": routed[0]} if routed else {}

        def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
            data = {"id": chat_id, "object": "chat.completion.chunk", "model": model, **served,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(data)}\n\n".encode("utf-8")

        fast = _fast()
        tps = float(profile.get("tps", 60))
        if not fast:
            await asyncio.sleep(float(profile.get("ttft", 0.3)))
        yield b": MOCK PROCESSING\n\n"
        yield chunk({"role": "assistant", "content": ""})

        reasoning_key = "reasoning" if provider == "openrouter" else "reasoning_content"
        plan = [(reasoning_key, p) for p in _pieces(reasoning)] + [("content", p) for p in _pieces(content)]
        emitted, finish = 0, "stop"
        started = time.perf_counter()
        for key, piece in plan:
            if max_tokens and emitted >= max_tokens:
                finish = "length"
                break
            if profile.get("midstream") and emitted >= 8:
                error = {"error": {"code": 502, "message": "Upstream provider disconnected"},
                         "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}
                yield f"data: {json.dumps(error)}\n\n".encode("utf-8")
                return
            yield chunk({key: piece})
            emitted += 1
            if not fast:
                # Sleep only when ahead of schedule: timers on Windows are too coarse to sleep per token.
                ahead = emitted / tps - (time.perf_counter() - started)
                if ahead > 0.012:
                    await asyncio.sleep(ahead)
        yield chunk({}, finish)

        usage: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": emitted,
                                 "total_tokens": prompt_tokens + emitted}
        price_in, price_out = profile.get("price", (0.5, 1.5))
        times = routed[1] if routed else 1.0      # each OpenRouter provider has its own price
        cost = round((prompt_tokens * price_in + emitted * price_out) * times / 1e6, 8)
        if provider == "openrouter":
            usage["cost"] = cost
        elif model.endswith(":deepinfra"):
            usage["estimated_cost"] = cost   # seen live: DeepInfra, through the Hugging Face router
        final = {"id": chat_id, "object": "chat.completion.chunk", "model": model, **served, "choices": [],
                 "usage": usage}
        yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

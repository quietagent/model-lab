"""OpenAI-compatible provider calls: model parsing, streaming, errors and metrics.

Everything that talks to OpenRouter or Hugging Face goes through this module.
API keys are only ever placed in the Authorization header - they are never
logged, stored or echoed back in error messages.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import getproxies

import httpx

from . import config

RETRY_ATTEMPTS = 3          # extra attempts after the first one (benchmark only)
RETRY_BASE_SECONDS = 1.5
RETRY_MAX_SECONDS = 20.0


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class ProviderError(Exception):
    def __init__(self, message: str, *, status: int | None = None, code: str = "error",
                 retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "status": self.status, "code": self.code,
                "retryable": self.retryable}


_NOT_FOUND_HINT = re.compile(
    r"not a valid model|does not exist|doesn't exist|no endpoints found|model[^.]{0,40}not found|"
    r"not supported by any provider|unknown model|invalid model", re.I)


def _error_detail(body: str) -> str:
    """Pull a human-readable message out of an error body."""
    body = (body or "").strip()
    if not body:
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        return "" if body.lstrip().startswith("<") else body[:300]
    return _detail_from_obj(data)


def _detail_from_obj(data: Any) -> str:
    if isinstance(data, str):
        return data[:300]
    if not isinstance(data, dict):
        return ""
    err = data.get("error", data)
    if isinstance(err, str):
        return err[:300]
    if isinstance(err, dict):
        msg = err.get("message") or err.get("detail") or ""
        metadata = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        raw = metadata.get("raw")
        if isinstance(raw, str) and raw and raw not in str(msg):
            msg = f"{msg} ({raw})" if msg else raw
        # OpenRouter names the provider whose error it passes on: with several providers behind one
        # model, that says which of them failed.
        name = metadata.get("provider_name")
        if isinstance(name, str) and name and name.lower() not in str(msg).lower():
            msg = f"{str(msg)[:300 - len(name) - 12]} [provider: {name}]"
        return str(msg)[:300]
    return ""


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value else None
    except ValueError:
        return None


def error_for_status(status: int, detail: str, provider: str, model: str,
                     retry_after: float | None = None) -> ProviderError:
    label = config.PROVIDERS.get(provider, {}).get("label", provider)
    if status == 400 and _NOT_FOUND_HINT.search(detail or ""):
        status_code, code, msg = 400, "not_found", f"Model not found: {model}"
    elif status == 401:
        status_code, code, msg = 401, "bad_key", f"Invalid API key (401). Check your {label} key."
    elif status == 402:
        status_code, code, msg = 402, "no_credits", (
            f"No credits (402). Add credits to your {label} account or pick a free model.")
    elif status == 403:
        status_code, code, msg = 403, "forbidden", (
            "Access denied (403). The key may not have permission for this model.")
    elif status == 404:
        status_code, code, msg = 404, "not_found", f"Model not found (404): {model}"
    elif status == 429:
        status_code, code, msg = 429, "rate_limited", "Rate limited (429). Wait a moment and try again."
    elif status == 408:
        status_code, code, msg = 408, "timeout", "The provider timed out (408)."
    elif status >= 500:
        status_code, code, msg = status, "provider_error", f"Provider error ({status}). The upstream service failed."
    else:
        status_code, code, msg = status, "bad_request", f"Request rejected ({status})."
    if detail and detail.lower() not in msg.lower():
        msg = f"{msg} {detail}" if msg.endswith(".") else f"{msg} - {detail}"
    return ProviderError(msg, status=status_code, code=code,
                         retryable=status in (408, 429) or status >= 500,
                         retry_after=retry_after)


def _scrub(text: str, key: str) -> str:
    return text.replace(key, "***") if key and len(key) >= 6 else text


# --------------------------------------------------------------------------
# A pinned provider (OpenRouter)
#
# OpenRouter sends each request to whichever provider it picks, and price, speed and even the answers
# change with the choice: live (2026-10-08) one model cost $0.02, $0.05 and $0.22 per million tokens in
# the same afternoon. A pin names the one provider that may serve the request. Fallbacks are off, so if
# that provider cannot serve, the request fails with an error that says so: it never goes elsewhere.
# --------------------------------------------------------------------------

def _plain(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pin_matches(pin: str, served_by: str) -> bool:
    """Is the provider OpenRouter says served a request the pinned one? The pin is a slug, sometimes with
    a variant ("deepinfra/fp8", "google-ai-studio"); the reply carries a display name ("DeepInfra",
    "Google AI Studio"). They are compared on letters and digits alone, and one may be the start of the
    other ("nebius" and "Nebius AI Studio")."""
    wanted, got = _plain(pin.split("/")[0]), _plain(served_by)
    return bool(wanted) and bool(got) and (got.startswith(wanted) or wanted.startswith(got))


def pin_error(exc: ProviderError, target: Target, detail: str = "") -> ProviderError:
    """A failed request that was pinned: say that it went to that one provider and to no other."""
    if not getattr(target, "pin", ""):
        return exc
    if exc.status == 404:
        # OpenRouter's own 404 for this lists the providers that do serve the model.
        exc.message = (f'The pinned provider "{target.pin}" cannot serve {target.model} (404)'
                       + (f": {detail}" if detail else ".")
                       + " The request was not sent to another provider.")
        exc.code, exc.retryable = "pin_unavailable", False
    else:
        exc.message = (f'{exc.message} [Pinned to "{target.pin}": the request was not sent to another '
                       "provider.]")
    exc.args = (exc.message,)
    return exc


def pin_mismatch(target: Target, served_by: str) -> ProviderError:
    """OpenRouter answered from another provider than the pinned one. With fallbacks off this should not
    happen; if it does, the reply is not used, so a pinned run never mixes providers without saying so."""
    return ProviderError(
        f'Pinned to "{target.pin}", but OpenRouter says {served_by} served this request. The reply is not '
        "used. Check the provider's slug in the model's provider list.", code="pin_mismatch", retryable=False)


# --------------------------------------------------------------------------
# Model IDs and URLs
# --------------------------------------------------------------------------

_OPENROUTER_HOSTS = {"openrouter.ai", "www.openrouter.ai"}
_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co"}
_OPENROUTER_RESERVED = {
    "docs", "chat", "rankings", "settings", "activity", "credits", "keys", "apps", "provider",
    "providers", "enterprise", "pricing", "privacy", "terms", "announcements", "about",
    "careers", "compare", "sign-in", "sign-up", "workspaces",
}
_HF_RESERVED = {
    "models", "datasets", "spaces", "docs", "blog", "pricing", "join", "login", "settings",
    "organizations", "tasks", "papers", "collections", "posts", "chat", "learn", "api",
    "inference", "enterprise", "new", "notifications",
}


def parse_model_input(text: str) -> tuple[str | None, str]:
    """Turn a model ID or a pasted model-page URL into (provider, model_id).

    The provider is only returned when the URL identifies it; plain IDs give None
    so the caller keeps whichever provider is currently selected.
    """
    raw = (text or "").strip().strip("\"'<>").strip()
    if not raw:
        return None, ""
    candidate = raw
    if re.match(r"^(www\.)?(openrouter\.ai|huggingface\.co|hf\.co)/", candidate, re.I):
        candidate = "https://" + candidate
    if not re.match(r"^https?://", candidate, re.I):
        return None, raw
    url = urlparse(candidate)
    host = (url.hostname or "").lower()
    parts = [unquote(p) for p in url.path.split("/") if p]

    if host in _OPENROUTER_HOSTS:
        if parts[:3] == ["api", "v1", "models"]:
            parts = parts[3:]
        elif parts[:1] == ["models"]:
            parts = parts[1:]
        if len(parts) >= 2 and parts[0] not in _OPENROUTER_RESERVED:
            return "openrouter", f"{parts[0]}/{parts[1]}"
        return "openrouter", ""

    if host in _HF_HOSTS:
        if len(parts) >= 2 and parts[0] not in _HF_RESERVED:
            model = f"{parts[0]}/{parts[1]}"
            route = (parse_qs(url.query).get("inference_provider") or [""])[0].strip()
            if route and ":" not in model and re.fullmatch(r"[\w.-]+", route):
                model = f"{model}:{route}"
            return "huggingface", model
        return "huggingface", ""

    return None, raw


@dataclass
class Target:
    provider: str
    model: str
    api_key: str = field(default="", repr=False)   # keep the key out of reprs, logs and tracebacks
    # OpenRouter only: the one provider that may serve this model's requests ("deepinfra/fp8"). Empty:
    # OpenRouter picks a provider for each request, and price, speed and answers change with its choice.
    pin: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


_PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._/-]{0,63}")


def clean_pin(pin: str | None, provider: str) -> str:
    """A pinned provider as it is sent to OpenRouter (its slug, such as "deepinfra/fp8"), or "" for none."""
    pin = (pin or "").strip()
    if not pin:
        return ""
    if provider != "openrouter":
        raise ProviderError("A pinned provider is an OpenRouter setting. On Hugging Face, name the provider "
                            "in the model ID instead (model:provider).", code="bad_request")
    if not _PIN.fullmatch(pin):
        raise ProviderError("The pinned provider is not a provider name. Use a slug from the model's provider "
                            "list, such as deepinfra or deepinfra/fp8.", code="bad_request")
    return pin


def resolve_target(provider: str, model: str, api_key: str = "", pin: str = "") -> Target:
    """Validate and normalise what the browser sent; fall back to .env keys."""
    parsed_provider, parsed_model = parse_model_input(model)
    provider = parsed_provider or (provider or "").strip().lower()
    if provider not in config.PROVIDERS:
        raise ProviderError(f"Unknown provider: {provider or '(none)'}", code="bad_request")
    if not parsed_model:
        raise ProviderError("Enter a model ID first.", code="no_model")
    key = (api_key or "").strip() or config.env_key(provider)
    if not key and not config.is_mock():
        label = config.PROVIDERS[provider]["label"]
        raise ProviderError(
            f"No API key for {label}. Paste one in the sidebar or set "
            f"{config.PROVIDERS[provider]['env']} in .env.", status=401, code="no_key")
    return Target(provider=provider, model=parsed_model, api_key=key, pin=clean_pin(pin, provider))


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------

class PreferIPv4Transport(httpx.AsyncBaseTransport):
    """Connect over IPv4 first and fall back to the system default.

    Some networks advertise an IPv6 route that then stalls in the TLS handshake
    (seen with router.huggingface.co behind NAT64) while IPv4 connects instantly.
    On an IPv6-only network the IPv4 attempt fails at once and that host is
    switched to the default transport for the rest of the session.
    """

    def __init__(self) -> None:
        self._ipv4 = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
        self._default = httpx.AsyncHTTPTransport()
        self._no_ipv4: set[str] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host not in self._no_ipv4:
            try:
                return await self._ipv4.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                self._no_ipv4.add(host)   # nothing was sent yet, so retrying is safe
        return await self._default.handle_async_request(request)

    async def aclose(self) -> None:
        await self._ipv4.aclose()
        await self._default.aclose()


_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)
        options: dict[str, Any] = {"timeout": timeout, "follow_redirects": True,
                                   "headers": {"User-Agent": "ModelLab/1.0"}}
        if config.is_mock():
            from .mock import MockTransport
            _client = httpx.AsyncClient(transport=MockTransport(), timeout=timeout)
        elif getproxies():
            _client = httpx.AsyncClient(**options)   # a system proxy is set: let httpx route through it
        else:
            _client = httpx.AsyncClient(transport=PreferIPv4Transport(), **options)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# --------------------------------------------------------------------------
# OpenRouter model list (public endpoint, cached for 24h)
# --------------------------------------------------------------------------

_models: dict[str, Any] = {"fetched_at": 0.0, "models": [], "by_id": {}}
# Bump when simplify_model changes what it stores, so an older cache file is refetched, not trusted.
_MODELS_CACHE_VERSION = 2
_models_lock = asyncio.Lock()


def _price_per_million(value: Any) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return None if price < 0 else round(price * 1_000_000, 6)  # negative = variable (routers)


# Keys in a pricing block that are not prices: thresholds inside "overrides" tiers, and a discount rate.
_NOT_A_PRICE = re.compile(r"^(?:min|max)_|^discount$")


def _all_prices_zero(value: Any) -> bool:
    """True if every price in a pricing block (or a nested tier of one) is exactly zero."""
    if isinstance(value, dict):
        return all(_all_prices_zero(v) for k, v in value.items() if not _NOT_A_PRICE.match(str(k)))
    if isinstance(value, list):
        return all(_all_prices_zero(v) for v in value)
    if isinstance(value, bool):
        return False
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False   # null, text, a negative "varies" marker: not provably free


def _text_only_output(architecture: Any) -> bool:
    """True unless the model is known to produce something other than text (audio, images, ...)."""
    if not isinstance(architecture, dict):
        return True
    outputs = architecture.get("output_modalities")
    if not isinstance(outputs, list) or not outputs:
        # Older shape: "text+image->text+audio". No arrow means nothing is known about the output.
        _, arrow, after = str(architecture.get("modality") or "").partition("->")
        outputs = after.split("+") if arrow and after else []
    return all(str(kind).strip().lower() == "text" for kind in outputs)


def is_free(pricing: Any, architecture: Any = None) -> bool:
    """A model is free only when nothing about it is billed.

    Besides prompt and completion tokens, OpenRouter prices can include a fee per request, per
    image, per web search, for cached input, audio or reasoning tokens, and tiered "overrides".
    Zero prompt and completion prices alone do not make a model free.

    A model that produces audio or images is billed per clip or per picture, and the price list
    does not always say so: google/lyria-3-clip-preview lists prompt 0, completion 0 and nothing
    else, while its description gives $0.04 per clip. So a model is only called free when its
    output is plain text.
    """
    if not isinstance(pricing, dict) or "prompt" not in pricing or "completion" not in pricing:
        return False
    return _all_prices_zero(pricing) and _text_only_output(architecture)


def simplify_model(item: dict[str, Any]) -> dict[str, Any]:
    pricing = item.get("pricing") or {}
    price_in = _price_per_million(pricing.get("prompt"))
    price_out = _price_per_million(pricing.get("completion"))
    free = is_free(pricing, item.get("architecture"))
    return {
        "id": item.get("id", ""),
        "name": item.get("name") or item.get("id", ""),
        "context_length": item.get("context_length"),
        "price_in": price_in,      # USD per 1M prompt tokens
        "price_out": price_out,    # USD per 1M completion tokens
        "free": free,
        # Tokens cost nothing but something else is billed (per request, per image, per clip, ...).
        "other_fees": price_in == 0 and price_out == 0 and not free,
    }


def _models_cache_file():
    name = "openrouter_models.mock.json" if config.is_mock() else "openrouter_models.json"
    return config.data_dir() / name


def _set_models(models: list[dict[str, Any]], fetched_at: float) -> None:
    _models.update(fetched_at=fetched_at, models=models,
                   by_id={m["id"]: m for m in models})


def _load_models_from_disk() -> bool:
    path = _models_cache_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != _MODELS_CACHE_VERSION:
            return False
        _set_models(data["models"], float(data["fetched_at"]))
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


async def get_openrouter_models(force: bool = False) -> dict[str, Any]:
    """Return {"models", "fetched_at", "error"}; serves stale data if a refresh fails."""
    async with _models_lock:
        now = time.time()
        if not _models["models"] and not config.is_mock():
            _load_models_from_disk()
        fresh = _models["models"] and now - _models["fetched_at"] < config.MODEL_CACHE_TTL
        error = None
        if force or not fresh:
            try:
                resp = await get_client().get(config.PROVIDERS["openrouter"]["models_url"], timeout=30.0)
                resp.raise_for_status()
                models = [simplify_model(m) for m in resp.json().get("data", []) if m.get("id")]
                models.sort(key=lambda m: m["id"])
                _set_models(models, now)
                if not config.is_mock():
                    _models_cache_file().write_text(
                        json.dumps({"version": _MODELS_CACHE_VERSION, "fetched_at": now, "models": models}),
                        encoding="utf-8")
            except (httpx.HTTPError, ValueError, OSError) as exc:
                error = f"Could not load the OpenRouter model list ({type(exc).__name__})."
        return {"models": _models["models"], "fetched_at": _models["fetched_at"], "error": error}


# --------------------------------------------------------------------------
# Hugging Face router model list (public endpoint, cached for 24h)
#
# One model is served by several providers, each with its own context size and price, and
# some list no price at all. The list gives the picker its autocomplete, and a price for a
# cost estimate when the provider that served a request did not say what it cost.
# --------------------------------------------------------------------------

_hf_models: dict[str, Any] = {"fetched_at": 0.0, "models": [], "by_id": {}}
_HF_CACHE_VERSION = 1


def _usd_per_million(value: Any) -> float | None:
    """Hugging Face prices are already in USD per 1M tokens."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def simplify_hf_model(item: dict[str, Any]) -> dict[str, Any]:
    routes = []
    for entry in item.get("providers") or []:
        if not isinstance(entry, dict) or not entry.get("provider"):
            continue
        pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
        context = entry.get("context_length")
        routes.append({
            "name": str(entry["provider"]), "status": entry.get("status"),
            "context_length": context if isinstance(context, int) and context > 0 else None,
            "price_in": _usd_per_million(pricing.get("input")),
            "price_out": _usd_per_million(pricing.get("output")),
        })
    priced = [r for r in routes if r["price_in"] is not None and r["price_out"] is not None]
    cheapest = min(priced, key=lambda r: r["price_in"] + r["price_out"]) if priced else None
    contexts = [r["context_length"] for r in routes if r["context_length"]]
    return {
        "id": item.get("id", ""),
        "name": item.get("id", ""),
        "context_length": max(contexts) if contexts else None,     # the largest any provider offers
        "price_in": cheapest["price_in"] if cheapest else None,     # the cheapest provider's price
        "price_out": cheapest["price_out"] if cheapest else None,
        "free": False,
        "other_fees": False,
        "providers": routes,
    }


def _hf_cache_file():
    return config.data_dir() / ("huggingface_models.mock.json" if config.is_mock() else "huggingface_models.json")


def _set_hf_models(models: list[dict[str, Any]], fetched_at: float) -> None:
    by_id = {m["id"]: m for m in models}
    by_id.update({m["id"].lower(): m for m in models if m["id"].lower() not in by_id})
    _hf_models.update(fetched_at=fetched_at, models=models, by_id=by_id)


def _load_hf_models_from_disk() -> bool:
    try:
        data = json.loads(_hf_cache_file().read_text(encoding="utf-8"))
        if data.get("version") != _HF_CACHE_VERSION:
            return False
        _set_hf_models(data["models"], float(data["fetched_at"]))
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


async def get_huggingface_models(force: bool = False) -> dict[str, Any]:
    """Return {"models", "fetched_at", "error"}; serves stale data if a refresh fails."""
    async with _models_lock:
        now = time.time()
        if not _hf_models["models"] and not config.is_mock():
            _load_hf_models_from_disk()
        fresh = _hf_models["models"] and now - _hf_models["fetched_at"] < config.MODEL_CACHE_TTL
        error = None
        if force or not fresh:
            try:
                resp = await get_client().get(config.PROVIDERS["huggingface"]["models_url"], timeout=30.0)
                resp.raise_for_status()
                models = [simplify_hf_model(m) for m in resp.json().get("data", []) if m.get("id")]
                models.sort(key=lambda m: m["id"].lower())
                _set_hf_models(models, now)
                if not config.is_mock():
                    _hf_cache_file().write_text(
                        json.dumps({"version": _HF_CACHE_VERSION, "fetched_at": now, "models": models}),
                        encoding="utf-8")
            except (httpx.HTTPError, ValueError, OSError) as exc:
                error = f"Could not load the Hugging Face model list ({type(exc).__name__})."
        return {"models": _hf_models["models"], "fetched_at": _hf_models["fetched_at"], "error": error}


async def get_models(provider: str, force: bool = False) -> dict[str, Any]:
    if provider == "huggingface":
        return await get_huggingface_models(force=force)
    return await get_openrouter_models(force=force)


# --------------------------------------------------------------------------
# The providers behind one OpenRouter model (public endpoint, kept for ten minutes)
#
# The list a pin is picked from: each provider has its own slug, price, context size and limit on the
# length of a reply.
# --------------------------------------------------------------------------

ENDPOINTS_TTL = 600  # seconds
_endpoints: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_MODEL_ID = re.compile(r"[\w.-]+/[\w.:-]+")


def simplify_endpoint(item: dict[str, Any]) -> dict[str, Any]:
    pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
    name = str(item.get("provider_name") or "")
    return {
        "name": name,                                           # as replies name it: "DeepInfra"
        "slug": str(item.get("tag") or name.lower()),           # what a pin is set to: "deepinfra/fp8"
        "quantization": item.get("quantization") if item.get("quantization") not in (None, "unknown") else None,
        "context_length": item.get("context_length"),
        "max_completion_tokens": item.get("max_completion_tokens"),
        "price_in": _price_per_million(pricing.get("prompt")),      # USD per 1M tokens
        "price_out": _price_per_million(pricing.get("completion")),
    }


async def get_openrouter_endpoints(model: str) -> dict[str, Any]:
    """{"model", "endpoints", "error"}: the providers that serve a model, each with the slug to pin it by."""
    model = (model or "").strip()
    if not _MODEL_ID.fullmatch(model):
        return {"model": model, "endpoints": [], "error": "Enter a model ID first."}
    cached = _endpoints.get(model)
    if cached and time.time() - cached[0] < ENDPOINTS_TTL:
        return {"model": model, "endpoints": cached[1], "error": None}
    url = f"{config.PROVIDERS['openrouter']['models_url']}/{model}/endpoints"
    try:
        resp = await get_client().get(url, timeout=30.0)
        if resp.status_code == 404:
            return {"model": model, "endpoints": [], "error": "OpenRouter lists no providers for this model."}
        resp.raise_for_status()
        listed = (resp.json().get("data") or {}).get("endpoints") or []
        endpoints = [simplify_endpoint(item) for item in listed if isinstance(item, dict)]
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return {"model": model, "endpoints": [],
                "error": f"Could not load the model's provider list ({type(exc).__name__})."}
    _endpoints[model] = (time.time(), endpoints)
    return {"model": model, "endpoints": endpoints, "error": None}


def hf_route(model: str, served_by: str | None = None) -> dict[str, Any] | None:
    """The provider entry whose price applies to a Hugging Face request, or None if nothing is known.

    In order: the provider that actually served it (the router's x-inference-provider header),
    the provider named in the model's ":suffix", the cheapest one for ":cheapest", and otherwise
    the first priced provider in the router's list.
    """
    if not _hf_models["by_id"] and not config.is_mock():
        _load_hf_models_from_disk()
    name, _, suffix = model.partition(":")
    info = _hf_models["by_id"].get(name) or _hf_models["by_id"].get(name.lower())
    if not info:
        return None
    by_name = {route["name"]: route for route in info["providers"]}
    for wanted in (served_by, suffix):
        if wanted and wanted in by_name:
            return by_name[wanted]
    priced = [r for r in info["providers"] if r["price_in"] is not None and r["price_out"] is not None]
    if not priced:
        return None
    if suffix == "cheapest":
        return min(priced, key=lambda r: r["price_in"] + r["price_out"])
    return priced[0]


def pricing_for(provider: str, model: str, served_by: str | None = None) -> tuple[float, float] | None:
    """(prompt, completion) USD per token, if known."""
    if provider == "huggingface":
        route = hf_route(model, served_by)
        if not route or route["price_in"] is None or route["price_out"] is None:
            return None
        return route["price_in"] / 1e6, route["price_out"] / 1e6
    if provider != "openrouter":
        return None
    if not _models["by_id"] and not config.is_mock():
        _load_models_from_disk()
    info = _models["by_id"].get(model) or _models["by_id"].get(model.split(":")[0])
    if not info or info["price_in"] is None or info["price_out"] is None:
        return None
    return info["price_in"] / 1e6, info["price_out"] / 1e6


# --------------------------------------------------------------------------
# <think> tag handling for streamed text
# --------------------------------------------------------------------------

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")


def _find_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str]:
    low = text.lower()
    best = (-1, "")
    for tag in tags:
        i = low.find(tag)
        if i != -1 and (best[0] == -1 or i < best[0]):
            best = (i, tag)
    return best


def _partial_tail(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of text that could be the start of a tag."""
    low = text.lower()
    for n in range(min(max(len(t) for t in tags) - 1, len(low)), 0, -1):
        if any(t.startswith(low[-n:]) for t in tags):
            return n
    return 0


class ThinkSplitter:
    """Split streamed content into ("reasoning" | "content", text) segments.

    Handles tags broken across chunks. A closing tag with no opening tag (some
    hosts strip the opener) yields ("reclassify", ""): everything emitted as
    content so far was really thinking.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._thought = False    # a think block was opened or closed already
        self._answered = False   # visible answer text has been emitted
        self._lstrip = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while self._buf:
            if self._in_think:
                idx, tag = _find_tag(self._buf, _CLOSE_TAGS)
                if idx >= 0:
                    if idx:
                        out.append(("reasoning", self._buf[:idx]))
                    self._buf = self._buf[idx + len(tag):]
                    self._in_think = False
                    self._lstrip = True
                    continue
                keep = 0 if final else _partial_tail(self._buf, _CLOSE_TAGS)
                cut = len(self._buf) - keep
                if cut:
                    out.append(("reasoning", self._buf[:cut]))
                self._buf = self._buf[cut:]
                break

            if not self._answered and not self._thought:
                lead = self._buf.lstrip()
                low = lead.lower()
                tag = next((t for t in _OPEN_TAGS if low.startswith(t)), None)
                if tag:
                    self._buf = lead[len(tag):]
                    self._in_think = True
                    self._thought = True
                    continue
                if not final and (not lead or any(t.startswith(low) for t in _OPEN_TAGS)):
                    break  # could still turn into an opening tag

            keep = 0
            if not self._thought:
                idx, tag = _find_tag(self._buf, _CLOSE_TAGS)
                if idx >= 0:
                    out.append(("reclassify", ""))
                    if idx:
                        out.append(("reasoning", self._buf[:idx]))
                    self._buf = self._buf[idx + len(tag):]
                    self._thought = True
                    self._answered = False
                    self._lstrip = True
                    continue
                keep = 0 if final else _partial_tail(self._buf, _CLOSE_TAGS)
            cut = len(self._buf) - keep
            chunk, self._buf = self._buf[:cut], self._buf[cut:]
            if self._lstrip:
                chunk = chunk.lstrip()
                if chunk:
                    self._lstrip = False
            if chunk:
                out.append(("content", chunk))
                if chunk.strip():
                    self._answered = True
            break
        return out


# --------------------------------------------------------------------------
# Chat completions
# --------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def reported_cost(usage: dict[str, Any]) -> tuple[float | None, bool]:
    """(cost in USD, is it an estimate) as stated in a usage block, or (None, False).

    OpenRouter reports the amount it charged as "cost". On Hugging Face the usage block comes from
    whichever provider served the request: DeepInfra sends its own "estimated_cost", others send
    no cost at all.
    """
    for name, estimated in (("cost", False), ("estimated_cost", True)):
        value = usage.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), estimated
    return None, False


def _headers(target: Target) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    if target.provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8100"
        headers["X-Title"] = "Model Lab"
    return headers


def _payload(target: Target, messages: list[dict[str, Any]], temperature: float | None,
             max_tokens: int | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": target.model, "messages": messages, "stream": True,
                               "stream_options": {"include_usage": True}}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if target.provider == "openrouter":
        payload["usage"] = {"include": True}
        if target.pin:
            # This provider only, and no fallback: if it cannot serve, OpenRouter answers with an error.
            payload["provider"] = {"only": [target.pin], "allow_fallbacks": False}
    return payload


def _drop_rejected_param(payload: dict[str, Any], error_body: str) -> bool:
    """Adjust the payload if a 400 error names an optional parameter. True if it changed."""
    text = error_body.lower()
    if "stream_options" in text and "stream_options" in payload:
        payload.pop("stream_options")
        payload.pop("usage", None)
        return True
    if "max_completion_tokens" in text and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")   # newer OpenAI-style name
        return True
    if "temperature" in text and "temperature" in payload:
        payload.pop("temperature")   # reasoning models often only allow the default
        return True
    return False


async def _iter_chunks(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded chunks from an SSE stream, or one chunk from a plain JSON body."""
    if "text/event-stream" not in resp.headers.get("content-type", "").lower():
        body = (await resp.aread()).decode("utf-8", "replace")
        try:
            data = json.loads(body)
        except ValueError:
            raise ProviderError("The provider returned an unreadable response.",
                                code="provider_error", retryable=True) from None
        if isinstance(data, dict) and data.get("error"):
            yield data
            return
        choices = []
        for choice in (data.get("choices") or []) if isinstance(data, dict) else []:
            message = choice.get("message") or {}
            choices.append({"delta": message, "finish_reason": choice.get("finish_reason")})
        yield {"choices": choices, "usage": data.get("usage") if isinstance(data, dict) else None,
               "provider": data.get("provider") if isinstance(data, dict) else None}
        return
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue  # blank separators and ": keep-alive" comments
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if isinstance(chunk, dict):
            yield chunk


def _stream_error(err: Any, target: Target) -> ProviderError:
    detail = _scrub(_detail_from_obj({"error": err}), target.api_key)
    code = err.get("code") if isinstance(err, dict) else None
    if isinstance(code, int) and 400 <= code <= 599:
        return error_for_status(code, detail, target.provider, target.model)
    return ProviderError(f"The provider reported an error: {detail or 'unknown error'}",
                         code="provider_error", retryable=True)


async def stream_chat(target: Target, messages: list[dict[str, Any]], *,
                      temperature: float | None = None,
                      max_tokens: int | None = None) -> AsyncIterator[dict[str, Any]]:
    """Stream one completion.

    Yields {"type": "reasoning" | "content", "text"} (and occasionally
    {"type": "reclassify"}), then a single {"type": "done", ...} with metrics.
    Raises ProviderError on failure.
    """
    url = config.PROVIDERS[target.provider]["chat_url"]
    client = get_client()
    payload = _payload(target, messages, temperature, max_tokens)
    splitter = ThinkSplitter()
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    # Which provider served the request. Hugging Face's router says so in a header, OpenRouter in the
    # "provider" field of every chunk.
    served_by: str | None = None
    first_at: float | None = None
    started = time.perf_counter()

    def absorb(segments: list[tuple[str, str]]) -> list[dict[str, Any]]:
        nonlocal first_at
        events = []
        for kind, text in segments:
            if kind == "reclassify":
                reasoning.extend(content)
                content.clear()
                events.append({"type": "reclassify"})
                continue
            if first_at is None:
                first_at = time.perf_counter()
            (reasoning if kind == "reasoning" else content).append(text)
            events.append({"type": kind, "text": text})
        return events

    try:
        for attempt in range(4):
            started = time.perf_counter()
            async with client.stream("POST", url, json=payload, headers=_headers(target)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    if resp.status_code == 400 and attempt < 3 and _drop_rejected_param(payload, body):
                        continue   # some backends reject optional parameters - retry without them
                    detail = _scrub(_error_detail(body), target.api_key)
                    raise pin_error(error_for_status(
                        resp.status_code, detail, target.provider, target.model,
                        _parse_retry_after(resp.headers.get("retry-after"))), target, detail)
                served_by = resp.headers.get("x-inference-provider") or None
                async for chunk in _iter_chunks(resp):
                    if chunk.get("error"):
                        raise pin_error(_stream_error(chunk["error"], target), target)
                    if isinstance(chunk.get("provider"), str) and chunk["provider"].strip():
                        served_by = chunk["provider"].strip()
                        if target.pin and not pin_matches(target.pin, served_by):
                            raise pin_mismatch(target, served_by)
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    segments: list[tuple[str, str]] = []
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        thought = delta.get("reasoning")
                        if not isinstance(thought, str) or not thought:
                            thought = delta.get("reasoning_content")
                        if isinstance(thought, str) and thought:
                            segments.append(("reasoning", thought))
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            segments.extend(splitter.feed(text))
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    for event in absorb(segments):
                        yield event
            break
    except httpx.TimeoutException:
        raise ProviderError("The request timed out.", code="timeout", retryable=True) from None
    except httpx.HTTPError as exc:
        raise ProviderError(f"Network error ({type(exc).__name__}). Check your connection.",
                            code="network", retryable=True) from None

    for event in absorb(splitter.flush()):
        yield event

    ended = time.perf_counter()
    total = ended - started
    ttft = (first_at - started) if first_at is not None else None
    text_out, text_think = "".join(content), "".join(reasoning)

    usage = usage or {}
    tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    estimated = not isinstance(tokens_in, int) or not isinstance(tokens_out, int)
    if not isinstance(tokens_in, int):
        tokens_in = estimate_tokens(" ".join(str(m.get("content", "")) for m in messages))
    if not isinstance(tokens_out, int):
        tokens_out = estimate_tokens(text_out + text_think)

    gen_time = total - (ttft or 0.0)
    if gen_time < 0.05:
        gen_time = total
    tps = tokens_out / gen_time if tokens_out and gen_time > 0 else None

    cost, cost_estimated = reported_cost(usage)
    if cost is None:
        # The provider did not say what it charged: estimate from the price list.
        if target.provider == "huggingface" and not _hf_models["models"]:
            await get_huggingface_models()
        prices = pricing_for(target.provider, target.model, served_by)
        if prices:
            cost, cost_estimated = tokens_in * prices[0] + tokens_out * prices[1], True

    yield {
        "type": "done",
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "metrics": {
            "ttft": round(ttft, 4) if ttft is not None else None,
            "total": round(total, 4),
            "tps": round(tps, 2) if tps is not None else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_estimated": estimated,
            "cost": cost,
            "cost_estimated": cost_estimated,
            "served_by": served_by,
        },
    }


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    truncated: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    retry_errors: list[str] = field(default_factory=list)   # why each earlier attempt failed


RetryCallback = Callable[[int, float, ProviderError], Awaitable[None]]


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter; honours Retry-After up to a cap."""
    base = 0.2 if config.is_mock() else RETRY_BASE_SECONDS
    delay = base * (2 ** attempt) * (0.75 + random.random() / 2)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, RETRY_MAX_SECONDS)


async def complete(target: Target, messages: list[dict[str, Any]], *,
                   temperature: float | None = None, max_tokens: int | None = None,
                   retries: int = 0, on_retry: RetryCallback | None = None) -> Completion:
    """Run a completion to the end, retrying 429/5xx/network errors with backoff."""
    attempt = 0
    failures: list[str] = []
    while True:
        result = Completion(attempts=attempt + 1, retry_errors=failures)
        content: list[str] = []
        reasoning: list[str] = []
        try:
            async for event in stream_chat(target, messages, temperature=temperature,
                                           max_tokens=max_tokens):
                kind = event["type"]
                if kind == "content":
                    content.append(event["text"])
                elif kind == "reasoning":
                    reasoning.append(event["text"])
                elif kind == "reclassify":
                    reasoning.extend(content)
                    content.clear()
                elif kind == "done":
                    result.finish_reason = event["finish_reason"]
                    result.truncated = event["truncated"]
                    result.metrics = event["metrics"]
            result.content = "".join(content)
            result.reasoning = "".join(reasoning)
            return result
        except ProviderError as exc:
            if not exc.retryable or attempt >= retries:
                exc.attempts = attempt + 1          # how many requests it took to give up
                raise
            failures.append(exc.message)
            delay = backoff_delay(attempt, exc.retry_after)
            if on_retry:
                await on_retry(attempt + 1, delay, exc)
            await asyncio.sleep(delay)
            attempt += 1


async def test_connection(target: Target) -> dict[str, Any]:
    """Send a tiny request and report latency, or a clear error."""
    try:
        result = await complete(
            target, [{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0, max_tokens=16)
    except ProviderError as exc:
        return {"ok": False, **exc.to_dict()}
    reply = result.content.strip()
    return {
        "ok": True,
        "latency_ms": round(result.metrics.get("total", 0) * 1000),
        "ttft_ms": round((result.metrics.get("ttft") or 0) * 1000),
        "reply": reply[:80],
        # A reasoning model spends these 16 tokens thinking: connected, but nothing to quote.
        "thinking_only": not reply and bool(result.reasoning.strip()),
        "provider": target.provider,
        "model": target.model,
        "pin": target.pin,
        "served_by": result.metrics.get("served_by"),     # the provider behind the router that answered
    }

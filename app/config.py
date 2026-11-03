"""Paths, provider definitions and environment handling."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "label": "OpenRouter",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "https://openrouter.ai/api/v1/models",
        "env": "OPENROUTER_API_KEY",
    },
    "huggingface": {
        "label": "Hugging Face",
        "chat_url": "https://router.huggingface.co/v1/chat/completions",
        "models_url": "https://router.huggingface.co/v1/models",
        "env": "HF_TOKEN",
    },
}

MODEL_CACHE_TTL = 24 * 3600  # seconds


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ (real env vars win).

    MODEL_LAB_ENV_FILE points at a different file; the test suite uses it so the
    real .env (and the keys in it) is never loaded while testing.
    """
    path = path or Path(os.environ.get("MODEL_LAB_ENV_FILE") or ROOT / ".env")
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def is_mock() -> bool:
    return os.environ.get("MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def data_dir() -> Path:
    path = Path(os.environ.get("MODEL_LAB_DATA_DIR") or ROOT / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    # Mock runs get their own database so they never mix with real results.
    return data_dir() / ("modellab.mock.db" if is_mock() else "modellab.db")


def env_key(provider: str) -> str:
    info = PROVIDERS.get(provider)
    return os.environ.get(info["env"], "").strip() if info else ""

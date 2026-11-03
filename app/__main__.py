"""Start the server:  python -m app [--mock] [--port 8100] [--open]"""
from __future__ import annotations

import argparse
import os

import uvicorn

from . import config


def main() -> None:
    config.load_env_file()
    parser = argparse.ArgumentParser(prog="python -m app", description="Model Lab server")
    parser.add_argument("--mock", action="store_true", help="use the built-in mock provider (no API key needed)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8100")))
    parser.add_argument("--open", action="store_true", help="open the app in your browser once it starts")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK"] = "1"
    url = f"http://localhost:{args.port}"
    if args.open and not os.environ.get("MODEL_LAB_NO_BROWSER"):
        os.environ["MODEL_LAB_OPEN_URL"] = url
    mode = "MOCK mode - no API key needed" if config.is_mock() else "live providers"
    print(f"\n  Model Lab  ->  {url}   ({mode})\n  Press Ctrl+C to stop.\n")
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

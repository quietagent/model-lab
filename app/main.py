"""FastAPI app: serves the single-page UI and proxies every provider call."""
from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import time
import webbrowser
from contextlib import aclosing, asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, bench_tests, benchmark, config, db, graders, providers, rejudge, security
from .providers import ProviderError

config.load_env_file()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    if os.environ.get("MODEL_LAB_OPEN_URL"):
        webbrowser.open(os.environ.pop("MODEL_LAB_OPEN_URL"))
    yield
    await benchmark.shutdown()      # a run in progress is saved as "interrupted"
    await providers.close_client()


app = FastAPI(title="Model Lab", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    """Refuse API calls that do not come from our own page (other websites, DNS rebinding)."""
    if request.scope["path"].startswith("/api/"):   # the decoded path, exactly what the router matches
        server = request.scope.get("server")
        port = server[1] if server else None      # the port this request actually arrived on
        if not security.request_allowed(request.headers.getlist("host"),
                                        request.headers.getlist("origin"), port,
                                        request.headers.getlist("sec-fetch-site")):
            return JSONResponse({"detail": security.REJECTION}, status_code=403)
    return await call_next(request)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class TargetIn(BaseModel):
    provider: str = "openrouter"
    model: str = ""
    api_key: str = ""
    pin: str = Field(default="", max_length=64)     # OpenRouter: the one provider that may serve the requests

    def resolve(self) -> providers.Target:
        return providers.resolve_target(self.provider, self.model, self.api_key, self.pin)

    @property
    def is_set(self) -> bool:
        return bool(self.model.strip())


# Default token limits (static/js/state.js holds the same numbers for the UI). They are caps, not
# targets, and are sized for reasoning models, which think before they answer. Measured live on a
# small free reasoning model (apodex-1.1-mini): a benchmark test took 300 to 2,300 tokens and a
# 150-word paragraph took 4,100, nearly all of it thinking. The old limits (1,024 for chat, 2,048
# per test) cut it off before the answer started, and so did 4,096 for the paragraph.
CHAT_MAX_TOKENS = 8192
BENCH_MAX_TOKENS = 8192
# The Hard level gets four times the room. Live (2026-10-07), nemotron-3-ultra scored 0 on three Hard
# tests at 8,192 only because it was still thinking when the limit came. At 16,384 (the default until
# 2026-10-08) five tests in every strong-model run still ended at the limit, and which five changed
# from run to run: the same test took 1,754 tokens in one run and more than 16,384 in the next, so
# most of what the strong models lost was the limit and not the answer. Runs at different limits are
# not comparable (the limit is part of the test), so each limit is its own group on the Leaderboard.
HARD_MAX_TOKENS = 32768
MAX_TOKENS_BY_LEVEL = {"standard": BENCH_MAX_TOKENS, "hard": HARD_MAX_TOKENS}


class ChatIn(BaseModel):
    target: TargetIn
    messages: list[dict[str, Any]]
    system: str = ""
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=CHAT_MAX_TOKENS, ge=1, le=200000)


Level = Literal["standard", "hard"]


class BenchmarkIn(BaseModel):
    target: TargetIn
    judge: TargetIn | None = None
    categories: list[str]
    level: Level = "standard"
    concurrency: int = Field(default=2, ge=1, le=4)
    max_tokens: int | None = Field(default=None, ge=16, le=200000)      # None: the default of the level

    @property
    def token_limit(self) -> int:
        return self.max_tokens or MAX_TOKENS_BY_LEVEL[self.level]


class RetryIn(BaseModel):
    """Keys for a re-run. The model and the judge are the run's own: only a key is taken from here."""
    target: TargetIn | None = None
    judge: TargetIn | None = None

    def key_for(self, provider: str) -> str:
        return next((t.api_key for t in (self.target, self.judge) if t and t.provider == provider and t.api_key), "")


class CustomTestIn(BaseModel):
    name: str = ""
    prompt: str
    grader: str
    expected: str = ""


class EnabledIn(BaseModel):
    enabled: bool


class ArenaSide(BaseModel):
    provider: str
    model: str
    response: str = ""


# The page makes up an id for each round, so your vote and the judge's verdict on it can be paired.
RoundId = Field(default=None, max_length=64, pattern=r"^[\w-]+$")


class VoteIn(BaseModel):
    a: ArenaSide
    b: ArenaSide
    winner: Literal["a", "b", "tie", "both_bad"]
    prompt: str = ""
    round_id: str | None = RoundId


class ArenaJudgeIn(BaseModel):
    judge: TargetIn
    a: ArenaSide
    b: ArenaSide
    prompt: str
    round_id: str | None = RoundId


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def sse_response(events: AsyncGenerator[dict[str, Any], None]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        try:
            # aclosing: if the browser disconnects, shut the source down now (cancels upstream calls)
            async with aclosing(events) as stream:
                async for event in stream:
                    yield sse(event)
        except ProviderError as exc:
            yield sse({"type": "error", **exc.to_dict()})
        except Exception as exc:  # keep the stream well-formed; never include request data
            yield sse({"type": "error", "code": "internal", "status": 500, "retryable": False,
                       "message": f"Internal error ({type(exc).__name__})."})
    return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)


def provider_error_response(exc: ProviderError) -> JSONResponse:
    return JSONResponse({"ok": False, **exc.to_dict()}, status_code=400)


def ratings_payload() -> dict[str, Any]:
    """The Arena record: Elo from your votes, and how often the judge agreed with you."""
    rows = db.list_votes()
    votes = sum(1 for row in rows if row["source"] == "human")
    return {"ratings": db.compute_ratings(rows), "votes": votes, "judge_verdicts": len(rows) - votes,
            "agreement": db.judge_agreement(rows)}


# --------------------------------------------------------------------------
# Pages and config
# --------------------------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "version": __version__,
        "mock": config.is_mock(),
        "providers": [{"id": pid, "label": info["label"], "env": info["env"],
                       "has_env_key": bool(config.env_key(pid))}
                      for pid, info in config.PROVIDERS.items()],
    }


@app.get("/api/models")
async def get_models(provider: str = "openrouter", refresh: bool = False) -> dict[str, Any]:
    if provider not in config.PROVIDERS:
        return {"models": [], "fetched_at": None, "error": None}
    return await providers.get_models(provider, force=refresh)


@app.get("/api/endpoints")
async def get_endpoints(model: str = "") -> dict[str, Any]:
    """The providers that serve one OpenRouter model, each with the slug to pin it by, its price and limits."""
    return await providers.get_openrouter_endpoints(model)


@app.post("/api/test-connection")
async def test_connection(target: TargetIn) -> dict[str, Any]:
    try:
        resolved = target.resolve()
    except ProviderError as exc:
        return {"ok": False, **exc.to_dict()}
    return await providers.test_connection(resolved)


# --------------------------------------------------------------------------
# Chat (also used by Arena and Speed)
# --------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(body: ChatIn) -> StreamingResponse:
    async def events() -> AsyncIterator[dict[str, Any]]:
        target = body.target.resolve()
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in body.messages]
        if body.system.strip():
            messages.insert(0, {"role": "system", "content": body.system.strip()})
        yield {"type": "start", "provider": target.provider, "model": target.model}
        async with aclosing(providers.stream_chat(target, messages, temperature=body.temperature,
                                                  max_tokens=body.max_tokens)) as stream:
            async for event in stream:
                yield event
    return sse_response(events())


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

@app.get("/api/benchmark/tests")
async def benchmark_tests() -> dict[str, Any]:
    counts: dict[tuple[str, str], int] = {}
    for test in bench_tests.BUILTIN:
        key = (test["category"], test["level"])
        counts[key] = counts.get(key, 0) + 1
    custom = db.list_custom_tests()
    enabled = sum(1 for t in custom if t["enabled"])

    def category(info: dict[str, Any]) -> dict[str, Any]:
        entry = {k: v for k, v in info.items() if k not in ("description", "hard")}
        entry["levels"] = {
            level: {"count": enabled if info.get("custom") else counts.get((info["id"], level), 0),
                    "description": info["hard"] if level == "hard" else info["description"]}
            for level in bench_tests.LEVEL_IDS}
        return entry

    return {
        "levels": [{**level, "max_tokens": MAX_TOKENS_BY_LEVEL[level["id"]]} for level in bench_tests.LEVELS],
        "categories": [category(c) for c in bench_tests.CATEGORIES],
        "tests": [{"test_id": t["id"], "category": t["category"], "level": t["level"], "name": t["name"]}
                  for t in bench_tests.BUILTIN],
        "custom": custom,
    }


@app.post("/api/benchmark/run")
async def benchmark_run(body: BenchmarkIn) -> Any:
    """Start a benchmark as a background job and return at once. Progress comes from .../events."""
    try:
        target = body.target.resolve()
        judge = body.judge.resolve() if body.judge and body.judge.is_set else None
        categories = [c for c in body.categories if c in bench_tests.CATEGORY_IDS]
        if judge is not None:
            # A judge that is gone would leave every judged test unscored: refuse rather than run without one.
            tests, _ = bench_tests.select_tests(categories, db.list_custom_tests(), True, body.level)
            if any(benchmark.uses_judge(test["grader"], judge) for test in tests):
                await benchmark.check_judge(judge)
        job = benchmark.start_job(target, judge, categories, body.concurrency, body.token_limit, body.level)
    except benchmark.Busy as exc:
        running = benchmark.active_job()
        return JSONResponse({"ok": False, **exc.to_dict(), "run_id": running.run_id if running else None},
                            status_code=409)
    except ProviderError as exc:
        return provider_error_response(exc)
    return {"ok": True, **job.snapshot()}


@app.get("/api/benchmark/active")
async def benchmark_active() -> dict[str, Any]:
    """The job that is running now, so a page that was closed or reloaded can pick it up again."""
    job = benchmark.active_job()
    db.close_dead_runs()        # a run whose server died since this one started is "interrupted", not running
    return {"job": job.snapshot() if job else None, "now": time.time(),
            # Runs another live process is working on (a second server on this database, or a script).
            "elsewhere": db.running_elsewhere()}


@app.get("/api/benchmark/{run_id}/events")
async def benchmark_events(run_id: int, after: int = -1) -> StreamingResponse:
    """Progress of a job as server-sent events: first everything after event number `after`, then
    new events as they happen. Closing this stream does not stop the job."""
    job = benchmark.get_job(run_id)
    if job is None:
        raise HTTPException(404, "This run is not in progress. Its saved results are at /api/runs/{id}.")

    async def events() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "hello", "run_id": run_id, "now": time.time()}   # lets the page line its clock up
        async with aclosing(job.stream(after)) as stream:
            async for event in stream:
                yield event
    return sse_response(events())


@app.post("/api/runs/{run_id}/retry")
async def retry_failed(run_id: int, body: RetryIn) -> Any:
    """Re-run the tests of a finished run where the provider failed (an API error, or a reply it cut
    short), as a background job (progress at /api/benchmark/{run_id}/events). The run's score is updated
    when it ends; its first-pass score stays. Out-of-tokens replies and wrong answers are not re-run."""
    run = db.get_run(run_id, with_results=False)
    if run is None:
        raise HTTPException(404, "Run not found.")
    try:
        # The run's own pin: a re-run goes to the provider the run was pinned to, or to none.
        target = providers.resolve_target(run["provider"], run["model"], body.key_for(run["provider"]),
                                          (run.get("params") or {}).get("pin") or "")
        judge = (providers.resolve_target(run["judge_provider"], run["judge_model"], body.key_for(run["judge_provider"]))
                 if run.get("judge_model") else None)
        if judge is not None and benchmark.needs_judge(db.get_run(run_id), db.list_custom_tests()):
            await benchmark.check_judge(judge)
        job = benchmark.start_retry(run_id, target, judge)
    except benchmark.Busy as exc:
        running = benchmark.active_job()
        return JSONResponse({"ok": False, **exc.to_dict(), "run_id": running.run_id if running else None},
                            status_code=409)
    except ProviderError as exc:
        return JSONResponse({"ok": False, **exc.to_dict()}, status_code=exc.status if exc.status in (404, 409) else 400)
    return {"ok": True, **job.snapshot()}


class JudgeAgainIn(BaseModel):
    level: Level
    settings: str = Field(min_length=1, max_length=500)       # the group's settings key
    judge: TargetIn


@app.get("/api/judge-again/plan")
async def judge_again_plan(level: Level, settings: str, provider: str = "openrouter", model: str = "") -> dict[str, Any]:
    """What "Judge again" would do to a group of runs with this judge: its runs and their judged results,
    the runs it leaves alone, the group the runs would move to, and a cost estimate. Makes no request."""
    judge = None
    if model.strip():
        try:
            judge = providers.resolve_target(provider, model)
        except ProviderError:
            judge = providers.Target(provider=provider, model=model.strip())
    if judge is not None and judge.provider == "openrouter":
        try:
            await providers.get_openrouter_models()        # its price, for the estimate
        except ProviderError:
            pass
    return rejudge.plan(level, settings, judge)


@app.post("/api/judge-again")
async def judge_again_start(body: JudgeAgainIn) -> Any:
    """Judge the judged results of a group's runs again with another judge, from the stored replies (see
    app/rejudge.py). Runs in the background; progress at /api/judge-again/events."""
    try:
        if not body.judge.is_set:
            raise ProviderError("Pick a judge model in the sidebar first.", code="no_model")
        job = rejudge.start(body.level, body.settings, body.judge.resolve())
    except ProviderError as exc:
        return JSONResponse({"ok": False, **exc.to_dict()}, status_code=exc.status if exc.status in (404, 409) else 400)
    return {"ok": True, **job.snapshot()}


@app.get("/api/judge-again/events")
async def judge_again_events(after: int = -1) -> StreamingResponse:
    """The progress of the "Judge again" that is running, or of the last one, as server-sent events."""
    job = rejudge.last_job()
    if job is None:
        raise HTTPException(404, "\"Judge again\" has not run since the server started.")

    async def events() -> AsyncIterator[dict[str, Any]]:
        async with aclosing(job.stream(after)) as stream:
            async for event in stream:
                yield event
    return sse_response(events())


@app.post("/api/judge-again/cancel")
async def judge_again_cancel() -> dict[str, Any]:
    return {"ok": rejudge.cancel()}


@app.post("/api/benchmark/{run_id}/cancel")
async def benchmark_cancel(run_id: int) -> dict[str, Any]:
    return {"ok": benchmark.request_cancel(run_id)}


@app.get("/api/custom-tests")
async def custom_tests() -> dict[str, Any]:
    return {"tests": db.list_custom_tests()}


@app.post("/api/custom-tests")
async def add_custom_test(body: CustomTestIn) -> dict[str, Any]:
    prompt, expected, grader = body.prompt.strip(), body.expected.strip(), body.grader.strip().lower()
    if not prompt:
        raise HTTPException(400, "The prompt is empty.")
    if grader not in bench_tests.CUSTOM_GRADERS:
        raise HTTPException(400, f"Grader must be one of: {', '.join(bench_tests.CUSTOM_GRADERS)}.")
    if grader != "judge" and not expected:
        raise HTTPException(400, "Enter the expected value for this grader.")
    if grader == "regex":
        try:
            re.compile(expected)
        except re.error as exc:
            raise HTTPException(400, f"Invalid regular expression: {exc}") from None
    if grader == "number":
        number = graders.parse_number(expected)
        if number is None:
            raise HTTPException(400, "The expected value must be a number.")
        expected = str(int(number)) if number.is_integer() else repr(number)
    name = body.name.strip() or re.sub(r"\s+", " ", prompt)[:48]
    return {"test": db.add_custom_test(name, prompt, grader, expected)}


@app.patch("/api/custom-tests/{test_id}")
async def toggle_custom_test(test_id: int, body: EnabledIn) -> dict[str, Any]:
    if not db.set_custom_test_enabled(test_id, body.enabled):
        raise HTTPException(404, "Test not found.")
    return {"ok": True}


@app.delete("/api/custom-tests/{test_id}")
async def delete_custom_test(test_id: int) -> dict[str, Any]:
    if not db.delete_custom_test(test_id):
        raise HTTPException(404, "Test not found.")
    return {"ok": True}


# --------------------------------------------------------------------------
# Runs / leaderboard
# --------------------------------------------------------------------------

@app.get("/api/runs")
async def list_runs(best: bool = False, unfinished: bool = False, level: Level | None = None,
                    older: bool = False, settings: str | None = None) -> dict[str, Any]:
    """Saved runs. Runs made with older tests (another fingerprint) are left out unless `older` is set;
    `older_hidden` says how many were left out, so the page can offer to show them.

    Runs are only compared with runs that have the same settings (token limit and judge): each run
    carries its `settings` and a `settings_key`, "best per model" is best per model, level and
    settings, and `groups` lists the groups of settings there are, the one with the newest run first.
    `settings` keeps one group (a group's key); the page shows one group at a time. A model pinned to a
    provider (`pin`, named `pin_label`) is a row of its own in its group, not a group of its own.

    `averages` has one entry for each model with more than one finished run of the same tests at the same
    settings: the average of those runs and the range from the lowest to the highest, overall and for
    each category, with the number of runs. Each run names its entry in `group`. The averages are over
    every such run, also when `best` leaves only one of them in `runs`.
    """
    everything = db.list_runs(include_unfinished=unfinished, level=level, include_older=True)
    shown = [run for run in everything if older or not run["older_tests"]]
    runs = db.list_runs(best_only=best, include_unfinished=unfinished, level=level, include_older=older,
                        settings=settings)
    alike = [run for run in shown if settings is None or run["settings_key"] == settings]
    return {"runs": runs, "older_hidden": len(everything) - len(shown), "levels": bench_tests.LEVELS,
            "groups": db.settings_groups(shown), "averages": db.model_averages(alike),
            "categories": [{"id": c["id"], "label": c["label"]} for c in bench_tests.CATEGORIES]}


def _export_rows(runs: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    header = (["run_id", "date", "provider", "model", "level", "status", "score", "grade"]
              + bench_tests.CATEGORY_IDS
              + ["avg_latency_s", "avg_ttft_s", "tokens_per_sec", "total_cost_usd", "model_cost_usd",
                 "judge_cost_usd", "tokens_in", "tokens_out",
                 "tests", "passed", "partial", "failed", "errors", "cut_off", "out_of_tokens", "empty", "judge",
                 "max_tokens", "pinned_provider", "served_by",
                 "first_pass_score", "retries", "tests_fingerprint", "older_tests"])
    rows = []
    for run in runs:
        judge = f"{run['judge_provider']}/{run['judge_model']}" if run.get("judge_model") else ""
        rows.append(
            [run["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run["created_at"])),
             run["provider"], run["model"], run["level"], run["status"], run["score"], run["grade"]]
            + [run["category_scores"].get(c, "") for c in bench_tests.CATEGORY_IDS]
            + [_round(run["avg_latency"], 3), _round(run["avg_ttft"], 3), _round(run["avg_tps"], 1),
               _round(run["total_cost"], 8), _round(run["model_cost"], 8), _round(run["judge_cost"], 8),
               run["tokens_in"], run["tokens_out"], run["total_tests"],
               run["n_pass"], run["n_partial"], run["n_fail"], run["n_error"], run.get("n_cut_off") or 0,
               run.get("n_out_of_tokens") or 0, run.get("n_empty") or 0, judge,
               run["settings"].get("max_tokens") or "", run.get("pin") or "",
               _served_text(run.get("served_by")),
               _round((run.get("first_pass") or {}).get("score"), 1), run.get("retries") or 0,
               run.get("fingerprint") or "", "yes" if run.get("older_tests") else "no"])
    return header, rows


def _served_text(served: dict[str, Any] | None) -> str:
    """ "DeepInfra", or "mixed: Groq 60, DeepInfra 9"; empty for a run saved before this was recorded."""
    names = (served or {}).get("providers") or {}
    if len(names) == 1 and not (served or {}).get("unknown"):
        return next(iter(names))
    parts = [f"{name} {count}" for name, count in names.items()]
    if names and served.get("unknown"):
        parts.append(f"not recorded {served['unknown']}")
    return "mixed: " + ", ".join(parts) if parts else ""


def _round(value: float | None, digits: int) -> Any:
    return "" if value is None else round(value, digits)


@app.get("/api/runs/export")
async def export_runs(format: Literal["csv", "json"] = "csv", best: bool = False,
                      unfinished: bool = False, level: Level | None = None, older: bool = False,
                      settings: str | None = None) -> Response:
    """Every group of settings unless `settings` names one; each row says which it had."""
    runs = db.list_runs(best_only=best, include_unfinished=unfinished, level=level, include_older=older,
                        settings=settings)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if format == "json":
        payload = [{**run, "results": db.get_results(run["id"])} for run in runs]
        return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="model-lab-runs-{stamp}.json"'})
    header, rows = _export_rows(runs)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    # BOM so Excel opens the UTF-8 file correctly
    return Response("﻿" + buffer.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="model-lab-runs-{stamp}.csv"'})


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int) -> dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    return {"run": run}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: int) -> dict[str, Any]:
    if not db.delete_run(run_id):
        raise HTTPException(404, "Run not found.")
    return {"ok": True}


# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------

@app.get("/api/arena/ratings")
async def arena_ratings() -> dict[str, Any]:
    return ratings_payload()


@app.post("/api/arena/vote")
async def arena_vote(body: VoteIn) -> dict[str, Any]:
    if body.round_id and db.has_vote(body.round_id, "human"):
        raise HTTPException(409, "You already voted on this round.")
    db.add_vote((body.a.provider, body.a.model), (body.b.provider, body.b.model), body.winner,
                source="human", prompt=body.prompt, round_id=body.round_id)
    return {"ok": True, **ratings_payload()}


@app.post("/api/arena/judge")
async def arena_judge(body: ArenaJudgeIn) -> Any:
    """Ask the judge which reply is better. A/B order is shuffled to avoid position bias.

    The verdict is a second opinion: it is saved and shown next to your vote, and it feeds the
    "judge agrees with you" figure, but it is not a vote and never changes the Elo table.
    """
    try:
        judge = body.judge.resolve()
        swapped = random.random() < 0.5
        first, second = (body.b, body.a) if swapped else (body.a, body.b)
        reply = await providers.complete(
            judge, benchmark.arena_judge_messages(body.prompt, first.response, second.response),
            temperature=0, max_tokens=benchmark.JUDGE_MAX_TOKENS, retries=1)
    except ProviderError as exc:
        return provider_error_response(exc)
    parsed = graders.parse_judge_winner(reply.content)
    if parsed is None:
        return JSONResponse({"ok": False, "code": "judge_format", "status": None, "retryable": True,
                             "message": "The judge did not return the requested JSON verdict."},
                            status_code=400)
    winner, reason = parsed
    if swapped and winner in ("a", "b"):
        winner = "b" if winner == "a" else "a"   # map back to the order shown on screen
    # The judge only ever sees "Response A/B", so rewrite those labels to match the screen too.
    if swapped:
        reason = re.sub(r"\b(Response|Model|Answer) ([AB])\b",
                        lambda m: f"{m.group(1)} {'B' if m.group(2) == 'A' else 'A'}", reason)
    db.add_vote((body.a.provider, body.a.model), (body.b.provider, body.b.model), winner,
                source="judge", prompt=body.prompt, reason=reason, round_id=body.round_id,
                judge=(judge.provider, judge.model))
    return {"ok": True, "winner": winner, "reason": reason, "swapped": swapped, "counted": False,
            "cost": reply.metrics.get("cost"), "cost_estimated": bool(reply.metrics.get("cost_estimated")),
            "judge": {"provider": judge.provider, "model": judge.model}, **ratings_payload()}


@app.delete("/api/arena/votes")
async def arena_clear() -> dict[str, Any]:
    return {"ok": True, "deleted": db.clear_votes()}


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

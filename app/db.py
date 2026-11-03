"""SQLite storage: benchmark runs, per-test results, custom tests and arena votes."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

from . import config, fingerprint, process
from .bench_tests import CATEGORY_IDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',      -- running | completed | cancelled | interrupted
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'standard',      -- standard | hard: never ranked or averaged together
    judge_provider TEXT,
    judge_model TEXT,
    categories TEXT NOT NULL DEFAULT '[]',
    params TEXT NOT NULL DEFAULT '{}',
    total_tests INTEGER NOT NULL DEFAULT 0,
    score REAL,
    grade TEXT,
    category_scores TEXT NOT NULL DEFAULT '{}',
    n_pass INTEGER NOT NULL DEFAULT 0,
    n_partial INTEGER NOT NULL DEFAULT 0,
    n_fail INTEGER NOT NULL DEFAULT 0,
    n_error INTEGER NOT NULL DEFAULT 0,
    n_cut_off INTEGER NOT NULL DEFAULT 0,        -- replies that stopped before they were finished
    n_empty INTEGER NOT NULL DEFAULT 0,          -- replies that were empty both times they were asked for
    n_out_of_tokens INTEGER NOT NULL DEFAULT 0,  -- the cut-off replies that used up the token limit (scored 0)
    avg_latency REAL,
    avg_ttft REAL,
    avg_tps REAL,
    total_cost REAL,                             -- model_cost + judge_cost
    model_cost REAL,                             -- the tested model's requests
    judge_cost REAL,                             -- the judge calls that graded them
    cost_estimated INTEGER NOT NULL DEFAULT 0,   -- 1 if any part is an estimate, not a charge
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT,                            -- tests, grader version and settings (app/fingerprint.py)
    test_hashes TEXT NOT NULL DEFAULT '{}',      -- {test id: hash of its prompt and grader spec}
    settings TEXT,                               -- {"max_tokens", "judge"}: what else the score depends on
    owner_pid INTEGER,                           -- while running: the process working on it ...
    owner_started REAL,                          -- ... and when that process started (IDs get reused)
    first_pass TEXT,                             -- the score before any "Retry failed requests" (JSON)
    retries INTEGER NOT NULL DEFAULT 0,          -- how many times "Retry failed requests" was started
    served_by TEXT                               -- which providers served its requests (JSON, see served_summary)
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    test_id TEXT NOT NULL,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,                        -- pass | partial | fail | empty | error | cut_off
    score REAL,
    notes TEXT NOT NULL DEFAULT '',
    error TEXT,
    truncated INTEGER NOT NULL DEFAULT 0,
    finish_reason TEXT,                          -- as the provider sent it; none if the stream just ended
    attempts INTEGER NOT NULL DEFAULT 1,
    latency REAL,
    ttft REAL,
    tps REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost REAL,
    cost_estimated INTEGER NOT NULL DEFAULT 0,
    judge_cost REAL,
    judge_cost_estimated INTEGER NOT NULL DEFAULT 0,
    served_by TEXT                               -- the provider that served each of its replies (JSON list)
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE TABLE IF NOT EXISTS custom_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    grader TEXT NOT NULL,
    expected TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS arena_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    a_provider TEXT NOT NULL,
    a_model TEXT NOT NULL,
    b_provider TEXT NOT NULL,
    b_model TEXT NOT NULL,
    winner TEXT NOT NULL,                        -- a | b | tie | both_bad
    source TEXT NOT NULL DEFAULT 'human',        -- human: a vote, counted for Elo. judge: an opinion, not counted
    prompt TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    round_id TEXT,                               -- ties your vote and the judge's verdict on the same round together
    judge_provider TEXT,
    judge_model TEXT
);
"""

# Columns added after the first release: (table, column, definition). A database created by an
# older version gets them on start-up; existing rows take the default.
MIGRATIONS = [
    ("runs", "level", "TEXT NOT NULL DEFAULT 'standard'"),
    ("runs", "model_cost", "REAL"),
    ("runs", "judge_cost", "REAL"),
    ("runs", "cost_estimated", "INTEGER NOT NULL DEFAULT 0"),
    ("results", "cost_estimated", "INTEGER NOT NULL DEFAULT 0"),
    ("results", "judge_cost", "REAL"),
    ("results", "judge_cost_estimated", "INTEGER NOT NULL DEFAULT 0"),
    ("arena_votes", "round_id", "TEXT"),
    ("arena_votes", "judge_provider", "TEXT"),
    ("arena_votes", "judge_model", "TEXT"),
    ("runs", "fingerprint", "TEXT"),             # runs from before this have none: "older tests"
    ("runs", "test_hashes", "TEXT NOT NULL DEFAULT '{}'"),
    ("runs", "n_cut_off", "INTEGER NOT NULL DEFAULT 0"),
    ("results", "finish_reason", "TEXT"),
    ("runs", "owner_pid", "INTEGER"),
    ("runs", "owner_started", "REAL"),
    ("runs", "first_pass", "TEXT"),              # runs from before this have none
    ("runs", "settings", "TEXT"),                # runs from before this: worked out from params and the judge
    ("runs", "retries", "INTEGER NOT NULL DEFAULT 0"),
    ("runs", "n_empty", "INTEGER NOT NULL DEFAULT 0"),   # no run from before this has an "empty" result
    ("runs", "n_out_of_tokens", "INTEGER NOT NULL DEFAULT 0"),   # counted from the saved results, see _migrate
    ("runs", "served_by", "TEXT"),               # runs from before this have none: not recorded
    ("results", "served_by", "TEXT"),
]

ELO_START = 1000.0
ELO_K = 32.0


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, definition in MIGRATIONS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            if (table, column) == ("runs", "model_cost"):
                # Until now a run's total was the tested model's cost alone: judge calls were not counted.
                conn.execute("UPDATE runs SET model_cost = total_cost")
            if (table, column) == ("runs", "n_out_of_tokens"):
                # Runs saved before this was counted: their results say which cut-off replies scored 0.
                conn.execute("UPDATE runs SET n_out_of_tokens = (SELECT COUNT(*) FROM results WHERE "
                             "results.run_id = runs.id AND results.status = 'cut_off' AND results.score IS NOT NULL)")


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        _migrate(conn)
        # "stopped" used to mean both "the user stopped it" and "the server died": it is "cancelled" now.
        conn.execute("UPDATE runs SET status = 'cancelled' WHERE status = 'stopped'")
        _unpin_settings(conn)
    close_dead_runs()


def _unpin_settings(conn: sqlite3.Connection) -> None:
    """On 2026-10-09 a pinned provider was one of a run's settings, so a pinned run was a group of its own.
    Now the pin belongs to the model's row (group_key): it is taken out of the run's settings and the
    fingerprint is worked out again without it. The pin itself stays in the run's params."""
    for row in conn.execute("SELECT id, params, settings, fingerprint, test_hashes FROM runs "
                            "WHERE settings LIKE '%\"pin\"%'").fetchall():
        try:
            settings, params = json.loads(row["settings"]), json.loads(row["params"] or "{}")
            hashes = json.loads(row["test_hashes"] or "{}")
        except ValueError:
            continue
        if not isinstance(settings, dict) or "pin" not in settings:
            continue
        if settings.get("pin"):
            params.setdefault("pin", settings["pin"])
        plain, stamp = fingerprint.without_pin(settings, hashes, row["fingerprint"])
        conn.execute("UPDATE runs SET params = ?, settings = ?, fingerprint = ? WHERE id = ?",
                     (json.dumps(params), json.dumps(plain), stamp, row["id"]))


def close_dead_runs() -> list[int]:
    """Mark as "interrupted" every run that is still "running" although the process that owned it is gone.

    Nothing is working on such a run any more, so say so instead of leaving it looking alive: its
    results so far are kept. A run whose owner is still running is left alone, whichever process
    that is: a second server (or a script) on the same database must not end the first one's job.
    A run saved before owners were recorded has none and counts as dead.
    """
    with connect() as conn:
        running = conn.execute("SELECT id, owner_pid, owner_started FROM runs WHERE status = 'running'").fetchall()
    dead = [row["id"] for row in running if not process.is_alive(row["owner_pid"], row["owner_started"])]
    for run_id in dead:
        finalize_run(run_id, "interrupted")
    return dead


def running_elsewhere() -> list[dict[str, Any]]:
    """Runs in progress in another live process on this database (another server, or a script)."""
    rows = _query("SELECT id, provider, model, level, created_at, owner_pid, owner_started FROM runs "
                  "WHERE status = 'running' AND owner_pid IS NOT NULL AND owner_pid != ?", (os.getpid(),))
    return [{"run_id": row["id"], "provider": row["provider"], "model": row["model"], "level": row["level"],
             "created_at": row["created_at"], "pid": row["owner_pid"]}
            for row in rows if process.is_alive(row["owner_pid"], row["owner_started"])]


def _query(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, args).fetchall()]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def letter_grade(score: float | None) -> str | None:
    if score is None:
        return None
    for floor, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= floor:
            return letter
    return "F"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _reply_arrived(result: dict[str, Any]) -> bool:
    return result.get("tokens_out") is not None or bool(result.get("response") or result.get("reasoning"))


def served_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Which providers served a run's requests: {"providers": {name: requests}, "unknown": tests}.

    A router (OpenRouter, Hugging Face) sends each request to a provider of its choice and names it in
    the reply. "providers" counts the replies each one sent; a test that was asked twice counts twice.
    "unknown" is the number of tests whose reply arrived without a provider being named: every test of
    a run saved before this was recorded, or a reply in which the router did not say.
    """
    providers: dict[str, int] = {}
    unknown = 0
    for result in results:
        names = result.get("served_by") or []
        for name in names:
            providers[name] = providers.get(name, 0) + 1
        if not names and _reply_arrived(result):
            unknown += 1
    return {"providers": dict(sorted(providers.items(), key=lambda item: (-item[1], item[0]))), "unknown": unknown}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-test results. API errors are reported but never scored, and neither is a reply the
    provider cut short ("cut_off" with no score): the test has not had its attempt. A reply that used
    up the token limit is "cut_off" with a score of 0, and counts. So does a reply that was empty both
    times it was asked for ("empty", 0)."""
    scored = [r for r in results if r["status"] != "error" and r["score"] is not None]
    by_category: dict[str, list[float]] = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r["score"])
    category_scores = {c: round(100 * _mean(by_category[c]), 1) for c in CATEGORY_IDS if c in by_category}
    for extra in by_category:  # categories no longer in the built-in list
        category_scores.setdefault(extra, round(100 * _mean(by_category[extra]), 1))
    # Every category carries equal weight, however many tests it has.
    score = round(_mean(list(category_scores.values())), 1) if category_scores else None
    model_costs = [r["cost"] for r in results if r.get("cost") is not None]
    judge_costs = [r["judge_cost"] for r in results if r.get("judge_cost") is not None]
    return {
        "score": score,
        "grade": letter_grade(score),
        "category_scores": category_scores,
        "n_pass": sum(1 for r in results if r["status"] == "pass"),
        "n_partial": sum(1 for r in results if r["status"] == "partial"),
        "n_fail": sum(1 for r in results if r["status"] == "fail"),
        "n_error": sum(1 for r in results if r["status"] == "error"),
        "n_cut_off": sum(1 for r in results if r["status"] == "cut_off"),
        "n_empty": sum(1 for r in results if r["status"] == "empty"),
        # The cut-off replies that used up the token limit: scored 0. The rest of n_cut_off was cut short by
        # the provider and has no score.
        "n_out_of_tokens": sum(1 for r in results if r["status"] == "cut_off" and r["score"] is not None),
        "avg_latency": _mean([r["latency"] for r in scored if r["latency"] is not None]),
        "avg_ttft": _mean([r["ttft"] for r in scored if r["ttft"] is not None]),
        "avg_tps": _mean([r["tps"] for r in scored if r["tps"] is not None]),
        # The judge's calls are part of what a run costs. Both parts are kept, so each can be shown.
        "total_cost": sum(model_costs + judge_costs) if model_costs or judge_costs else None,
        "model_cost": sum(model_costs) if model_costs else None,
        "judge_cost": sum(judge_costs) if judge_costs else None,
        "cost_estimated": any(r.get("cost_estimated") or r.get("judge_cost_estimated") for r in results),
        "tokens_in": sum(r["tokens_in"] or 0 for r in results),
        "tokens_out": sum(r["tokens_out"] or 0 for r in results),
        "served_by": served_summary(results),
    }


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def create_run(provider: str, model: str, judge: tuple[str, str] | None, categories: list[str],
               params: dict[str, Any], total_tests: int, level: str = "standard",
               tests: list[dict[str, Any]] | None = None) -> int:
    """Save a new run. `tests` are the tests it will use: their fingerprint is stored with it, together
    with the settings the score depends on (the token limit, and the judge if it grades any of them).
    The provider the run is pinned to, if any, is params["pin"]: not a setting, but part of its row."""
    stamp, hashes, settings = (fingerprint.describe(tests, params.get("max_tokens"), judge)
                               if tests else (None, {}, None))
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, provider, model, level, judge_provider, judge_model, categories, "
            "params, total_tests, fingerprint, test_hashes, settings, owner_pid, owner_started) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), provider, model, level, judge[0] if judge else None, judge[1] if judge else None,
             json.dumps(categories), json.dumps(params), total_tests, stamp, json.dumps(hashes),
             json.dumps(settings) if settings else None,
             os.getpid(), process.started()))     # this process is working on it until it is finalized
        return int(cur.lastrowid)


RESULT_FIELDS = ("test_id", "category", "name", "prompt", "response", "reasoning", "status", "score",
                 "notes", "error", "truncated", "finish_reason", "attempts", "latency", "ttft", "tps",
                 "tokens_in", "tokens_out", "cost", "cost_estimated", "judge_cost", "judge_cost_estimated",
                 "served_by")
_RESULT_FLAGS = ("truncated", "cost_estimated", "judge_cost_estimated")


def _result_values(result: dict[str, Any]) -> list[Any]:
    values = [result.get(f) for f in RESULT_FIELDS]
    for flag in _RESULT_FLAGS:
        values[RESULT_FIELDS.index(flag)] = 1 if result.get(flag) else 0
    # A list of provider names in memory, JSON in the table; nothing when no provider was named.
    served = [str(name) for name in result.get("served_by") or []]
    values[RESULT_FIELDS.index("served_by")] = json.dumps(served, ensure_ascii=False) if served else None
    return values


def add_result(run_id: int, result: dict[str, Any]) -> None:
    values = _result_values(result)
    with connect() as conn:
        conn.execute(
            f"INSERT INTO results (run_id, {', '.join(RESULT_FIELDS)}) "
            f"VALUES (?, {', '.join('?' for _ in RESULT_FIELDS)})", (run_id, *values))


def replace_result(run_id: int, result: dict[str, Any]) -> bool:
    """Put a new result in place of a test's earlier one (Retry failed requests). Keeps its position."""
    values = _result_values(result)
    with connect() as conn:
        return conn.execute(
            f"UPDATE results SET {', '.join(f'{f} = ?' for f in RESULT_FIELDS)} WHERE run_id = ? AND test_id = ?",
            (*values, run_id, result["test_id"])).rowcount > 0


def _decode_run(row: dict[str, Any]) -> dict[str, Any]:
    for key, default in (("categories", []), ("params", {}), ("category_scores", {}), ("test_hashes", {})):
        try:
            row[key] = json.loads(row[key]) if row.get(key) else default
        except ValueError:
            row[key] = default
    row["cost_estimated"] = bool(row.get("cost_estimated"))
    for key in ("first_pass", "settings", "served_by"):       # served_by: None for a run saved before it was recorded
        try:
            row[key] = json.loads(row[key]) if row.get(key) else None
        except ValueError:
            row[key] = None
    # What the run's score depends on besides its tests. A run saved before this was recorded shows what
    # is known about it: its token limit, and its judge (whether or not the judge graded anything).
    row["settings_recorded"] = row["settings"] is not None
    if row["settings"] is None:
        judge = f"{row['judge_provider']}/{row['judge_model']}" if row.get("judge_model") else None
        row["settings"] = {"max_tokens": row["params"].get("max_tokens"), "judge": judge}
    row["settings_key"] = fingerprint.settings_key(row["settings"])
    # The provider the run was pinned to (OpenRouter's slug) and the name it served under ("DeepInfra").
    row["pin"] = row["params"].get("pin") or None
    row["pin_label"] = pin_label(row["pin"], row["served_by"])
    row["group"] = group_key(row)
    # The categories its first pass has no score for. Worked out when the run is read, and kept next to
    # the first pass: what is stored stays as it was stored.
    row["first_pass_missing"] = (missing_categories(row["categories"], row["first_pass"].get("category_scores"))
                                 if row["first_pass"] else [])
    # The same for its final score: such a run is marked, left out of its model's average and not ranked.
    row["final_missing"] = (missing_categories(row["categories"], row["category_scores"])
                            if row.get("score") is not None else [])
    return row


def pin_label(pin: str | None, served_by: dict[str, Any] | None) -> str | None:
    """How a pinned model is named on the Leaderboard: the name its provider gave in the replies
    ("DeepInfra" for the pin "deepinfra/fp8"), or the pin itself when no reply named one."""
    if not pin:
        return None
    names = list((served_by or {}).get("providers") or {})
    return names[0] if len(names) == 1 else pin


def missing_categories(ran: list[str], category_scores: dict[str, Any] | None) -> list[str]:
    """The categories a run ran that have no score at all: every one of their tests ended in a provider
    failure (or, in a first pass, in a judge error). A score without them is the average of the other
    categories only, so it is not comparable with a complete one."""
    return [category for category in ran if category not in (category_scores or {})]


def get_results(run_id: int) -> list[dict[str, Any]]:
    rows = _query("SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,))
    for row in rows:
        for flag in _RESULT_FLAGS:
            row[flag] = bool(row[flag])
        try:
            row["served_by"] = json.loads(row["served_by"]) if row.get("served_by") else []
        except ValueError:
            row["served_by"] = []
    return rows


FIRST_PASS_FIELDS = ("score", "grade", "category_scores", "n_pass", "n_partial", "n_fail", "n_error", "n_cut_off",
                     "n_empty")


def first_pass_of(summary: dict[str, Any]) -> dict[str, Any]:
    """The part of a summary that is kept as a run's first pass."""
    return {key: summary[key] for key in FIRST_PASS_FIELDS}


def finalize_run(run_id: int, status: str) -> dict[str, Any] | None:
    """Close a run: work out its score from its results and save it.

    The first time a run is closed is its first pass: every test had its one attempt, and nothing has
    been re-run yet. That score is kept apart ("first_pass") and never changes; "Retry failed requests"
    closes the run again and only moves the final score.
    """
    summary = summarize(get_results(run_id))
    with connect() as conn:
        conn.execute("UPDATE runs SET first_pass = ? WHERE id = ? AND first_pass IS NULL",
                     (json.dumps(first_pass_of(summary)), run_id))
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, score = ?, grade = ?, category_scores = ?, "
            "n_pass = ?, n_partial = ?, n_fail = ?, n_error = ?, n_cut_off = ?, n_empty = ?, n_out_of_tokens = ?, "
            "avg_latency = ?, avg_ttft = ?, avg_tps = ?, total_cost = ?, model_cost = ?, judge_cost = ?, "
            "cost_estimated = ?, tokens_in = ?, tokens_out = ?, served_by = ?, owner_pid = NULL, "
            "owner_started = NULL WHERE id = ?",
            (status, time.time(), summary["score"], summary["grade"], json.dumps(summary["category_scores"]),
             summary["n_pass"], summary["n_partial"], summary["n_fail"], summary["n_error"], summary["n_cut_off"],
             summary["n_empty"], summary["n_out_of_tokens"], summary["avg_latency"], summary["avg_ttft"],
             summary["avg_tps"], summary["total_cost"], summary["model_cost"], summary["judge_cost"], 1 if summary["cost_estimated"] else 0,
             summary["tokens_in"], summary["tokens_out"],
             # Nothing is stored for a run in which no reply named its provider: "not recorded".
             json.dumps(summary["served_by"], ensure_ascii=False) if summary["served_by"]["providers"] else None,
             run_id))
    return get_run(run_id, with_results=False)


def count_retry(run_id: int) -> None:
    """ "Retry failed requests" was started on this run: its final score may now differ from its first pass."""
    with connect() as conn:
        conn.execute("UPDATE runs SET retries = retries + 1 WHERE id = ?", (run_id,))


def _count_scored(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set "category_counts" on each run: {category: {"scored", "total"}}, from its saved results.

    A test that ended in an API error, or whose reply the provider cut short, has no score and is left
    out of its category's average. "83.3" over 6 of 12 tests is not the same measurement as over all
    12, so the page says how many were scored wherever some were not.
    """
    counts: dict[int, dict[str, dict[str, int]]] = {}
    wanted = [run["id"] for run in runs]
    if wanted:
        marks = ", ".join("?" for _ in wanted)
        for row in _query(
                "SELECT run_id, category, COUNT(*) AS total, "
                "SUM(CASE WHEN status != 'error' AND score IS NOT NULL THEN 1 ELSE 0 END) AS scored "
                f"FROM results WHERE run_id IN ({marks}) GROUP BY run_id, category", tuple(wanted)):
            counts.setdefault(row["run_id"], {})[row["category"]] = {"scored": row["scored"], "total": row["total"]}
    for run in runs:
        run["category_counts"] = counts.get(run["id"], {})
    return runs


def get_run(run_id: int, with_results: bool = True) -> dict[str, Any] | None:
    rows = _query("SELECT * FROM runs WHERE id = ?", (run_id,))
    if not rows:
        return None
    run = _count_scored(_mark_older([_decode_run(rows[0])]))[0]
    if with_results:
        run["results"] = get_results(run_id)
    return run


UNFINISHED = ("cancelled", "interrupted")


def _mark_older(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set "older_tests" on each run: made with other tests, or another grader version, than today's."""
    current = fingerprint.current_hashes(list_custom_tests())
    for run in runs:
        recorded = run["settings"] if run.get("settings_recorded") else None
        run["older_tests"] = not fingerprint.is_current({**run, "settings": recorded}, current)
    return runs


def settings_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The groups of settings among some runs, the one with the newest run first. Runs are only compared
    inside a group: the same token limit and the same judge. A pinned run is ranked with the others."""
    groups: dict[str, dict[str, Any]] = {}
    for run in runs:
        group = groups.setdefault(run["settings_key"], {
            "key": run["settings_key"], "max_tokens": run["settings"].get("max_tokens"),
            "judge": run["settings"].get("judge"), "runs": 0, "newest": 0.0})
        group["runs"] += 1
        group["newest"] = max(group["newest"], run["created_at"])
    return sorted(groups.values(), key=lambda group: group["newest"], reverse=True)


def list_runs(best_only: bool = False, include_unfinished: bool = False,
              level: str | None = None, include_older: bool = False,
              settings: str | None = None) -> list[dict[str, Any]]:
    """Runs that ended, newest first. A cancelled or interrupted run only counts if it scored something.

    `level` keeps one level only, and `settings` one group of settings (a settings_key). "Best per
    model" is always best per model, level and settings: a Hard score and a Standard score measure
    different things, and so do a run at 8,192 tokens and one at 16,384, or two runs marked by
    different judges. Runs made with older tests (see fingerprint.py) are left out unless
    `include_older` is set; either way each run says whether it is one ("older_tests").
    """
    sql = "SELECT * FROM runs WHERE score IS NOT NULL AND status IN ({}){} ORDER BY created_at DESC"
    statuses = "'completed', 'cancelled', 'interrupted'" if include_unfinished else "'completed'"
    runs = _count_scored(_mark_older([_decode_run(r) for r in _query(
        sql.format(statuses, " AND level = ?" if level else ""), (level,) if level else ())]))
    if not include_older:
        runs = [run for run in runs if not run["older_tests"]]
    if settings is not None:
        runs = [run for run in runs if run["settings_key"] == settings]
    if best_only:
        # A model pinned to a provider is a row of its own, apart from the same model unpinned.
        best: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for run in runs:
            key = (run["provider"], run["model"], run["pin"] or "", run["level"], run["settings_key"])
            current = best.get(key)
            # Prefer a completed run over a partial one, then one with every category scored, then the
            # higher score.
            rank = (run["status"] == "completed", not run["final_missing"], run["score"])
            if current is None or rank > (current["status"] == "completed", not current["final_missing"],
                                          current["score"]):
                best[key] = run
        runs = sorted(best.values(), key=lambda r: r["created_at"], reverse=True)
    return runs


# --------------------------------------------------------------------------
# Several runs of one model
# --------------------------------------------------------------------------

def group_key(run: dict[str, Any]) -> str | None:
    """Names the runs this one can be averaged with: finished runs of the same model that measured the
    same thing (the same level, tests and grader version, token limit and judge), pinned to the same
    provider or to none. A pinned model is a row of its own, ranked with the other models.

    None for a run that is never averaged: a cancelled or interrupted one ran only some of its tests, and
    a run saved before fingerprints were recorded cannot be shown to have used the same tests as another.
    The settings are named next to the fingerprint because the first fingerprints did not cover them.
    The pin is named at the end, and only when there is one: an unpinned run's key is as it always was.
    """
    if run.get("status") != "completed" or run.get("score") is None or not run.get("fingerprint"):
        return None
    key = "|".join((run["provider"], run["model"], run["level"], run["settings_key"], run["fingerprint"]))
    return f"{key}|pin={run['pin']}" if run.get("pin") else key


def _spread(values: list[Any], digits: int = 1) -> dict[str, Any] | None:
    """The average of a value over some runs, with its range: {"mean", "min", "max", "runs"}. `runs` is
    the number of runs that have the value (a run in which every test of a category failed to arrive has
    no score for that category). None when no run has it."""
    known = [value for value in values if value is not None]
    if not known:
        return None
    return {"mean": round(sum(known) / len(known), digits), "min": min(known), "max": max(known), "runs": len(known)}


def _count_range(members: list[dict[str, Any]], category: str) -> dict[str, int] | None:
    """{"min", "max", "total"}: the tests of a category that were scored, over the runs that have counts."""
    known = [counts for run in members if (counts := (run.get("category_counts") or {}).get(category))]
    if not known:
        return None
    return {"min": min(c["scored"] for c in known), "max": max(c["scored"] for c in known),
            "total": max(c["total"] for c in known)}


def model_averages(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For every model with more than one finished run of the same tests at the same settings (see
    group_key): the average of those runs with the range from the lowest to the highest, overall and
    for each category, and how many runs there are. The newest group comes first.

    One run says little about a category: live, two runs of one model an hour apart differed by 0.3
    points overall and by 40 points in Instructions. The overall average is the average of the runs'
    scores, and its range runs from the lowest run to the highest.

    The first-pass average is over the runs whose first pass scored every category; the others are
    named in "first_pass_left_out" with what they are missing. The same goes for the final score
    ("score_left_out"). When no run of the model scored every category, "ranked" is False: "score" is
    then the average of the incomplete scores, "missing" names the categories that lack a score, and
    the page lists the model below the ranked ones, without a rank.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("group"):
            groups.setdefault(run["group"], []).append(run)
    averages = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda run: run["created_at"])
        newest = members[-1]
        categories = [c for c in CATEGORY_IDS if any(c in run["category_scores"] for run in members)]
        categories += sorted({c for run in members for c in run["category_scores"]} - set(categories))
        # A final score with a whole category missing is the average of the other categories only (live:
        # Laguna's 49.0 without Long context and Writing). It is not averaged with complete ones either.
        complete = [run for run in members if not run.get("final_missing")]
        short = [run for run in members if run.get("final_missing")]
        score = _spread([run["score"] for run in complete or members])
        costs = [run["total_cost"] for run in members if run["total_cost"] is not None]
        # A first pass with a whole category missing is the average of the other categories only (live:
        # 80.6 without Writing, 77.1 once Writing was graded). It is not averaged with complete ones.
        incomplete = [run for run in members if run.get("first_pass_missing")]
        whole = [run for run in members if run not in incomplete]
        averages.append({
            "key": key, "provider": newest["provider"], "model": newest["model"], "level": newest["level"],
            "pin": newest.get("pin"), "pin_label": newest.get("pin_label"),
            "settings_key": newest["settings_key"], "fingerprint": newest["fingerprint"],
            "older_tests": bool(newest.get("older_tests")),
            "runs": len(members), "run_ids": [run["id"] for run in members], "newest": newest["created_at"],
            "score": score, "grade": letter_grade(score["mean"]), "ranked": bool(complete),
            "score_left_out": [{"id": run["id"], "score": run["score"], "missing": run["final_missing"]}
                               for run in short] if complete else [],
            "missing": [] if complete else [c for c in CATEGORY_IDS if any(c in run["final_missing"] for run in short)],
            "category_scores": {c: _spread([run["category_scores"].get(c) for run in members]) for c in categories},
            # How many of a category's tests were scored, from the run with the fewest to the one with the most.
            "category_counts": {c: counts for c in categories if (counts := _count_range(members, c))},
            "first_pass": _spread([(run.get("first_pass") or {}).get("score") for run in whole]),
            "first_pass_left_out": [{"id": run["id"], "score": (run.get("first_pass") or {}).get("score"),
                                     "missing": run["first_pass_missing"]} for run in incomplete],
            "out_of_tokens": _spread([run.get("n_out_of_tokens") or 0 for run in members]),
            "avg_latency": _mean([run["avg_latency"] for run in members if run["avg_latency"] is not None]),
            "avg_tps": _mean([run["avg_tps"] for run in members if run["avg_tps"] is not None]),
            # What the runs cost together, and on average: one more run costs about the average.
            "total_cost": sum(costs) if costs else None, "mean_cost": _mean(costs),
            "cost_estimated": any(run.get("cost_estimated") for run in members),
        })
    return sorted(averages, key=lambda group: group["newest"], reverse=True)


# --------------------------------------------------------------------------
# Judge again (app/rejudge.py)
# --------------------------------------------------------------------------

def runs_in_group(level: str, settings_key: str) -> list[dict[str, Any]]:
    """Every run of a level with these settings that is not in progress, with its results, oldest first:
    also the ones the Leaderboard leaves out (unfinished, older tests, no score)."""
    rows = _query("SELECT * FROM runs WHERE level = ? AND status != 'running' ORDER BY created_at", (level,))
    runs = _mark_older([_decode_run(row) for row in rows])
    runs = [run for run in runs if run["settings_key"] == settings_key]
    for run in runs:
        run["results"] = get_results(run["id"])
    return runs


def running_in_group(level: str, settings_key: str) -> list[int]:
    rows = _query("SELECT * FROM runs WHERE level = ? AND status = 'running'", (level,))
    return [run["id"] for run in map(_decode_run, rows) if run["settings_key"] == settings_key]


def backup(label: str) -> str:
    """Copy the database to data/backups/<name>-<date and time>-<label>.db and return its path."""
    folder = config.data_dir() / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{config.db_path().stem}-{time.strftime('%Y-%m-%d-%H%M%S')}-{label}.db"
    source = sqlite3.connect(config.db_path(), timeout=15)
    target = sqlite3.connect(path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return str(path)


def _results_in(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    """A run's results as get_results reads them, inside an open transaction."""
    rows = [dict(row) for row in conn.execute("SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,))]
    for row in rows:
        for flag in _RESULT_FLAGS:
            row[flag] = bool(row[flag])
        try:
            row["served_by"] = json.loads(row["served_by"]) if row.get("served_by") else []
        except ValueError:
            row["served_by"] = []
    return rows


JUDGED_FIELDS = ("status", "score", "notes", "error", "judge_cost", "judge_cost_estimated")


def apply_judge_again(changes: list[dict[str, Any]]) -> None:
    """Write what "Judge again" worked out, for every run at once, in one transaction: all of it or none.

    Each change is {"run_id", "updates": [{"id", status, score, notes, error, judge_cost,
    judge_cost_estimated}], "judge": (provider, model), "settings", "fingerprint", "params",
    "first_pass" (None: keep the stored one)}. Only the judged results named in "updates" change; the
    run's score, counts and costs are worked out again from its results. Its status, dates, replies and
    the model's own costs stay as they are.
    """
    with connect() as conn:
        for change in changes:
            run_id = change["run_id"]
            for update in change["updates"]:
                values = [update[field] for field in JUDGED_FIELDS]
                values[JUDGED_FIELDS.index("judge_cost_estimated")] = 1 if update["judge_cost_estimated"] else 0
                if conn.execute(f"UPDATE results SET {', '.join(f'{f} = ?' for f in JUDGED_FIELDS)} "
                                "WHERE id = ? AND run_id = ?", (*values, update["id"], run_id)).rowcount != 1:
                    raise ValueError(f"Result {update['id']} is not in run #{run_id}.")
            summary = summarize(_results_in(conn, run_id))
            first = change.get("first_pass")
            conn.execute(
                "UPDATE runs SET score = ?, grade = ?, category_scores = ?, n_pass = ?, n_partial = ?, n_fail = ?, "
                "n_error = ?, n_cut_off = ?, n_empty = ?, n_out_of_tokens = ?, avg_latency = ?, avg_ttft = ?, "
                "avg_tps = ?, total_cost = ?, judge_cost = ?, cost_estimated = ?, judge_provider = ?, judge_model = ?, "
                "settings = ?, fingerprint = ?, params = ?, first_pass = COALESCE(?, first_pass) WHERE id = ?",
                (summary["score"], summary["grade"], json.dumps(summary["category_scores"]), summary["n_pass"],
                 summary["n_partial"], summary["n_fail"], summary["n_error"], summary["n_cut_off"], summary["n_empty"],
                 summary["n_out_of_tokens"], summary["avg_latency"], summary["avg_ttft"], summary["avg_tps"],
                 summary["total_cost"], summary["judge_cost"], 1 if summary["cost_estimated"] else 0,
                 change["judge"][0], change["judge"][1], json.dumps(change["settings"]), change["fingerprint"],
                 json.dumps(change["params"]), json.dumps(first) if first else None, run_id))


def delete_run(run_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount > 0


# --------------------------------------------------------------------------
# Custom tests
# --------------------------------------------------------------------------

def list_custom_tests() -> list[dict[str, Any]]:
    return _query("SELECT * FROM custom_tests ORDER BY id")


def add_custom_test(name: str, prompt: str, grader: str, expected: str) -> dict[str, Any]:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO custom_tests (created_at, name, prompt, grader, expected) VALUES (?, ?, ?, ?, ?)",
            (time.time(), name, prompt, grader, expected))
        new_id = cur.lastrowid
    return _query("SELECT * FROM custom_tests WHERE id = ?", (new_id,))[0]


def set_custom_test_enabled(test_id: int, enabled: bool) -> bool:
    with connect() as conn:
        return conn.execute("UPDATE custom_tests SET enabled = ? WHERE id = ?",
                            (1 if enabled else 0, test_id)).rowcount > 0


def delete_custom_test(test_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM custom_tests WHERE id = ?", (test_id,)).rowcount > 0


# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------

def add_vote(a: tuple[str, str], b: tuple[str, str], winner: str, source: str = "human",
             prompt: str = "", reason: str = "", round_id: str | None = None,
             judge: tuple[str, str] | None = None) -> int:
    """Record your vote (source "human") or the judge's verdict (source "judge") on a round."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO arena_votes (created_at, a_provider, a_model, b_provider, b_model, winner, source, "
            "prompt, reason, round_id, judge_provider, judge_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), a[0], a[1], b[0], b[1], winner, source, prompt[:2000], reason[:1000], round_id,
             judge[0] if judge else None, judge[1] if judge else None))
        return int(cur.lastrowid)


def has_vote(round_id: str, source: str = "human") -> bool:
    return bool(_query("SELECT 1 FROM arena_votes WHERE round_id = ? AND source = ? LIMIT 1", (round_id, source)))


def list_votes() -> list[dict[str, Any]]:
    return _query("SELECT * FROM arena_votes ORDER BY id")


def clear_votes() -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM arena_votes").rowcount


def _outcome(winner: str) -> str:
    """a | b | draw. "Both bad" is a draw: neither model beat the other."""
    return winner if winner in ("a", "b") else "draw"


def judge_agreement(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the judge picked the same side as you, over the rounds where both gave a verdict.

    A tie and "both bad" count as the same answer: no winner.
    """
    mine: dict[str, str] = {}
    judged: dict[str, str] = {}
    for vote in votes:
        if vote.get("round_id"):
            (mine if vote.get("source", "human") == "human" else judged)[vote["round_id"]] = _outcome(vote["winner"])
    both = [round_id for round_id in mine if round_id in judged]
    agree = sum(1 for round_id in both if mine[round_id] == judged[round_id])
    return {"rounds": len(both), "agree": agree, "percent": round(100 * agree / len(both)) if both else None}


def compute_ratings(votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay every vote in order to get Elo ratings and win/loss/tie records.

    Only your own votes count. A judge's verdict is an opinion shown next to yours: it never moves
    a rating. "Both bad" counts as a tie: neither model beat the other.
    """
    table: dict[tuple[str, str], dict[str, Any]] = {}

    def entry(provider: str, model: str) -> dict[str, Any]:
        return table.setdefault((provider, model), {
            "provider": provider, "model": model, "rating": ELO_START,
            "wins": 0, "losses": 0, "ties": 0, "games": 0})

    for vote in votes:
        if vote.get("source", "human") != "human":
            continue  # the judge's opinion, not a vote
        a = entry(vote["a_provider"], vote["a_model"])
        b = entry(vote["b_provider"], vote["b_model"])
        if a is b:
            continue  # a model cannot gain rating against itself
        score_a = {"a": 1.0, "b": 0.0}.get(vote["winner"], 0.5)
        expected_a = 1.0 / (1.0 + 10 ** ((b["rating"] - a["rating"]) / 400.0))
        a["rating"] += ELO_K * (score_a - expected_a)
        b["rating"] += ELO_K * ((1.0 - score_a) - (1.0 - expected_a))
        a["games"] += 1
        b["games"] += 1
        if score_a == 1.0:
            a["wins"] += 1
            b["losses"] += 1
        elif score_a == 0.0:
            a["losses"] += 1
            b["wins"] += 1
        else:
            a["ties"] += 1
            b["ties"] += 1

    rows = [r for r in table.values() if r["games"]]
    for row in rows:
        row["rating"] = round(row["rating"], 1)
    return sorted(rows, key=lambda r: r["rating"], reverse=True)

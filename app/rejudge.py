"""Judge again: grade the judged results of a group of runs again with another judge, from the stored replies.

Live (2026-10-18) the judge every run had used, apodex/apodex-1.1-mini:free, disappeared from OpenRouter
("404 No endpoints found"). A run with another judge is in another group of settings, so new runs could no
longer be compared with the old ones, and running every model again would have cost hours and money. The
replies were all stored: only the judge's part of each score had to be done again.

For one group of settings at one level (a Leaderboard group), "Judge again":
  - checks that the new judge answers;
  - backs up the database;
  - asks the new judge about every judged result of the group's runs (Writing and custom judge tests, and
    Honesty in a run that had a judge), from the stored reply. A judged result is one the judge marked,
    or one whose judge call failed while its reply was kept. A judge call that fails is asked once more
    at the end; if it fails again, the result is a judge error (Writing) or keeps the phrase check
    (Honesty), as in a run, and "Retry failed requests" can ask again later;
  - then, in one transaction, writes the new marks, works out each run's score and first pass again, and
    moves the runs to the new judge: their judge, settings, settings key and fingerprint.

Replies, code-graded results, out-of-tokens results and the model's own costs are never touched. Runs
made with older tests are left as they are: their tests are not today's, so today's judge prompt is not
the one they were graded with. Nothing is written until every judge call is done; cancelled before that,
the runs stay as they were.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from typing import Any, AsyncIterator

from . import bench_tests, benchmark, db, fingerprint, providers
from .benchmark import JUDGE_ERROR, PHRASE_CHECK_INSTEAD
from .providers import ProviderError, Target

CONCURRENCY = 4
TRUNCATED = "Reached the token limit after giving its answer (finish_reason=length). "

# How a result's notes record what happened to it after its grade: kept when the grade is done again.
HISTORY = (" [Answered on attempt", " [Asked for twice", " [Asked again", " [The reply looks cut short",
           " [The reply may be unfinished", ' [Re-run by "Retry failed requests"', ' [Graded again by "Retry failed requests"',
           ' [Judged again by "Retry failed requests"', " [Graded again on ", " [Judged again with ")

# What "Retry failed requests" wrote on a result it re-ran or graded again: it had no score in the first pass.
_RERUN_ERROR = (' [Re-run by "Retry failed requests": before, this test was an API error',
                ' [Graded again by "Retry failed requests"')
_RERUN_CUT = ' [Re-run by "Retry failed requests": before, this test was cut short by the provider'
_REJUDGED = re.compile(r' \[Judged again by "Retry failed requests": before, the judge call failed and the phrase '
                       r'check graded it \((\w+), ([\d.]+)\)')


def tests_by_id() -> dict[str, dict[str, Any]]:
    return {**bench_tests.BY_ID, **{t["id"]: t for t in map(bench_tests.custom_to_test, db.list_custom_tests())}}


def is_judged(result: dict[str, Any], test: dict[str, Any] | None, had_judge: bool) -> bool:
    """Did a judge grade this result, or fail to while its reply was kept? Those are judged again."""
    if test is None:
        return False
    kind = test["grader"]["type"]
    graded = result["status"] in ("pass", "partial", "fail")
    if kind == "judge":
        return graded or benchmark.reply_kept(result)
    return kind == "admits_unknown" and had_judge and graded


def history_of(notes: str) -> str:
    """The part of a result's notes that tells its history (attempts, retries, earlier re-grades)."""
    found = [notes.find(mark) for mark in HISTORY if mark in notes]
    return notes[min(found):] if found else ""


def first_pass_results(results: list[dict[str, Any]], graded: dict[int, dict[str, Any]] | None = None
                       ) -> list[dict[str, Any]]:
    """The results as they were in the first pass, before any "Retry failed requests": a test the retry ran
    again or graded again had no score then (its notes say so). With `graded` (result id -> new values),
    the first pass of the same replies under the new judge; without it, the stored first pass, rebuilt
    (used to check that the rebuilding is right)."""
    out = []
    for result in results:
        notes = result.get("notes") or ""
        if any(mark in notes for mark in _RERUN_ERROR):
            out.append({**result, "status": "error", "score": None})
        elif _RERUN_CUT in notes:
            out.append({**result, "status": "cut_off", "score": None})
        elif graded is not None and result["id"] in graded:
            out.append({**result, **graded[result["id"]]})
        elif (found := _REJUDGED.search(notes)):           # the first pass had the phrase check's grade
            out.append({**result, "status": found.group(1), "score": float(found.group(2))})
        else:
            out.append(result)
    return out


def _same_first_pass(stored: dict[str, Any] | None, rebuilt: dict[str, Any]) -> bool:
    return bool(stored) and all(stored.get(key) == rebuilt.get(key) for key in db.FIRST_PASS_FIELDS)


def judge_name(judge: Target) -> str:
    return f"{judge.provider}/{judge.model}"


def plan(level: str, settings_key: str, judge: Target | None = None) -> dict[str, Any]:
    """What "Judge again" would do to a group: its runs, how many judged results each has, the runs it
    leaves alone and why, the settings key the runs would move to, and a cost estimate."""
    runs = db.runs_in_group(level, settings_key)
    tests = tests_by_id()
    chosen, older, items = [], [], []
    for run in runs:
        if run["older_tests"]:
            older.append(run["id"])
            continue
        if not run["settings"].get("judge"):
            continue
        judged = [r for r in run["results"] if is_judged(r, tests.get(r["test_id"]), bool(run.get("judge_model")))]
        chosen.append({"id": run["id"], "model": run["model"], "pin_label": run["pin_label"], "status": run["status"],
                       "score": run["score"], "judged": len(judged)})
        items += [(run, r, tests[r["test_id"]]) for r in judged]
    # The judge reads the test's prompt and the reply: about 4 characters a token, plus the instructions.
    tokens_in = sum(round((len(test["prompt"]) + len(result["response"] or "")) / 3.5) + 350 for _, result, test in items)
    tokens_out = sum(500 if test["grader"]["type"] == "judge" else 200 for _, _, test in items)
    price = providers.pricing_for(judge.provider, judge.model) if judge else None
    return {
        "level": level, "settings_key": settings_key, "runs": chosen, "older": older, "calls": len(items),
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "cost": round(tokens_in * price[0] + tokens_out * price[1], 6) if price else None,
        "new_settings_key": (fingerprint.settings_key({**runs[0]["settings"], "judge": judge_name(judge)})
                             if judge and runs else None),
        "busy": db.running_in_group(level, settings_key),
    }


async def _judge_one(result: dict[str, Any], test: dict[str, Any], judge: Target) -> dict[str, Any]:
    """New values for one judged result, graded from its stored reply. "failed": the judge did not give a
    verdict (a judge error, or the phrase check standing in for it)."""
    try:
        grade = await benchmark.grade_reply(test, judge, result["response"] or "", bool(result["truncated"]),
                                            bool(result["reasoning"]))
    except ProviderError as exc:
        spent = getattr(exc, "judge", None) or {}
        return {"status": "error", "score": None, "error": f"{JUDGE_ERROR}{exc.message}",
                "notes": "Judge error - not counted in the score", "judge_cost": spent.get("cost"),
                "judge_cost_estimated": bool(spent.get("cost_estimated")), "failed": True}
    spent = grade.judge or {}
    notes = (TRUNCATED if result["truncated"] else "") + grade.notes
    return {"status": grade.status, "score": round(grade.score, 4), "error": None, "notes": notes,
            "judge_cost": spent.get("cost"), "judge_cost_estimated": bool(spent.get("cost_estimated")),
            "failed": PHRASE_CHECK_INSTEAD in grade.notes}


def _before(result: dict[str, Any]) -> str:
    if result["status"] == "error":
        return "a judge error"
    return f"{result['status']}, {result['score']:g}"


class JudgeAgainJob:
    """One "Judge again" in progress. Events work as a benchmark job's do (benchmark.Job)."""

    def __init__(self, level: str, settings_key: str, judge: Target) -> None:
        self.level, self.settings_key, self.judge = level, settings_key, judge
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.cancel_requested = False
        self.done = 0
        self.total = 0
        self.events: list[dict[str, Any]] = []
        self.task: asyncio.Task | None = None
        self._tick = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    def publish(self, event: dict[str, Any]) -> None:
        event.update(seq=len(self.events), at=round(time.time(), 3))
        self.events.append(event)
        tick, self._tick = self._tick, asyncio.Event()
        tick.set()

    async def stream(self, after: int = -1) -> AsyncIterator[dict[str, Any]]:
        position = max(0, after + 1)
        while True:
            tick = self._tick
            while position < len(self.events):
                position += 1
                yield self.events[position - 1]
            if self.finished:
                return
            await tick.wait()

    def snapshot(self) -> dict[str, Any]:
        return {"level": self.level, "settings_key": self.settings_key, "judge": judge_name(self.judge),
                "total": self.total, "done": self.done, "created_at": self.created_at, "finished": self.finished,
                "cancel_requested": self.cancel_requested, "last_seq": len(self.events) - 1}

    async def run(self) -> None:
        try:
            self.publish({"type": "summary", **await self._work()})
        except ProviderError as exc:
            self.publish({"type": "error", **exc.to_dict()})
        except asyncio.CancelledError:
            self.publish({"type": "error", "code": "interrupted", "message": "The server stopped: nothing was written."})
            raise
        except Exception as exc:          # nothing was written unless the transaction finished
            self.publish({"type": "error", "code": "internal", "message": f"Internal error ({type(exc).__name__}: {exc})."})
        finally:
            self.finished_at = time.time()

    async def _work(self) -> dict[str, Any]:
        await benchmark.check_judge(self.judge)
        runs = [run for run in db.runs_in_group(self.level, self.settings_key)
                if not run["older_tests"] and run["settings"].get("judge")]
        tests = tests_by_id()
        items = [(run, result, tests[result["test_id"]]) for run in runs for result in run["results"]
                 if is_judged(result, tests.get(result["test_id"]), bool(run.get("judge_model")))]
        self.total = len(items)
        path = db.backup("before-judge-again")
        self.publish({"type": "start", "runs": [run["id"] for run in runs], "total": self.total, "backup": path,
                      "judge": judge_name(self.judge)})
        graded: dict[int, dict[str, Any]] = {}
        spent: dict[int, list[dict[str, Any]]] = {}

        async def judge_all(todo: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> None:
            pending = deque(todo)

            async def worker() -> None:
                while pending and not self.cancel_requested:
                    run, result, test = pending.popleft()
                    new = await _judge_one(result, test, self.judge)
                    spent.setdefault(result["id"], []).append(new)
                    graded[result["id"]] = new
                    self.done += 1
                    self.publish({"type": "judged", "run_id": run["id"], "test_id": result["test_id"],
                                  "status": new["status"], "score": new["score"], "failed": new["failed"],
                                  "done": self.done, "total": self.total})
            await asyncio.gather(*(worker() for _ in range(min(CONCURRENCY, len(todo)) or 1)))

        await judge_all(items)
        again = [item for item in items if graded.get(item[1]["id"], {}).get("failed")]
        if again and not self.cancel_requested:
            self.total += len(again)
            self.publish({"type": "again", "count": len(again)})
            await judge_all(again)
        if self.cancel_requested:
            return {"cancelled": True, "written": False, "backup": path,
                    "message": "Cancelled before anything was written: the runs are as they were."}
        if all(new["failed"] for new in graded.values()):
            return {"cancelled": False, "written": False, "backup": path,
                    "message": f"{self.judge.model} gave no verdict at all, so nothing was written: the runs are "
                               "as they were."}

        changes, report = [], []
        new_judge = judge_name(self.judge)
        for run in runs:
            old_judge = run["settings"]["judge"]
            updates, first_graded = [], {}
            for result in run["results"]:
                if result["id"] not in graded:
                    continue
                new = graded[result["id"]]
                calls = spent[result["id"]]
                costs = [c["judge_cost"] for c in calls if c["judge_cost"] is not None]
                old_cost = result.get("judge_cost")
                mark = (f" [Judged again with {new_judge} on {time.strftime('%Y-%m-%d')}: before, "
                        f"{_before(result)} by {old_judge}.]")
                values = {"status": new["status"], "score": new["score"], "error": new["error"],
                          "notes": new["notes"] + history_of(result.get("notes") or "") + mark,
                          # What every judge call on this reply cost, the earlier judge's included.
                          "judge_cost": (sum(costs) + (old_cost or 0)) if costs or old_cost is not None else None,
                          "judge_cost_estimated": bool(result.get("judge_cost_estimated"))
                          or any(c["judge_cost_estimated"] for c in calls)}
                updates.append({"id": result["id"], **values})
                first_graded[result["id"]] = {"status": new["status"], "score": new["score"]}
            settings = {**run["settings"], "judge": new_judge}
            stamp = fingerprint.restamp(run["test_hashes"], run["fingerprint"], run["settings_key"], settings)
            # The first pass graded the same replies: it is worked out again under the new judge, if the
            # stored one can be rebuilt from the results (else it is kept, and the report says so).
            rebuilt = db.first_pass_of(db.summarize(first_pass_results(run["results"])))
            first = (db.first_pass_of(db.summarize(first_pass_results(run["results"], first_graded)))
                     if _same_first_pass(run["first_pass"], rebuilt) else None)
            params = {**run["params"], "judged_again": [*run["params"].get("judged_again", []),
                                                         {"at": time.time(), "from": old_judge, "to": new_judge,
                                                          "backup": path}]}
            changes.append({"run_id": run["id"], "updates": updates, "judge": (self.judge.provider, self.judge.model),
                            "settings": settings, "fingerprint": stamp, "params": params, "first_pass": first})
            report.append({"id": run["id"], "model": run["model"], "pin_label": run["pin_label"],
                           "judged": len(updates), "failed": sum(1 for u in updates if graded[u["id"]]["failed"]),
                           "old": {"score": run["score"], "first_pass": (run["first_pass"] or {}).get("score"),
                                   "category_scores": run["category_scores"], "fingerprint": run["fingerprint"]},
                           "first_pass_kept": first is None and bool(run["first_pass"])})
        db.apply_judge_again(changes)
        for entry in report:
            run = db.get_run(entry["id"], with_results=False)
            entry["new"] = {"score": run["score"], "first_pass": (run["first_pass"] or {}).get("score"),
                            "category_scores": run["category_scores"], "fingerprint": run["fingerprint"]}
        judge_cost = sum(c["judge_cost"] or 0 for calls in spent.values() for c in calls)
        return {"cancelled": False, "written": True, "backup": path, "runs": report, "judge": new_judge,
                "settings_key": fingerprint.settings_key({**runs[0]["settings"], "judge": new_judge}) if runs else None,
                "judge_cost": judge_cost, "failed": sum(entry["failed"] for entry in report)}


_job: JudgeAgainJob | None = None


def active_job() -> JudgeAgainJob | None:
    return _job if _job is not None and not _job.finished else None


def last_job() -> JudgeAgainJob | None:
    return _job


def start(level: str, settings_key: str, judge: Target) -> JudgeAgainJob:
    """Start "Judge again" on a group in the background. Must be called inside the event loop."""
    global _job
    if active_job() is not None:
        raise ProviderError("\"Judge again\" is already running.", status=409, code="busy")
    running = benchmark.active_job()
    if running is not None:
        raise ProviderError(f"A benchmark is running (run #{running.run_id}). Wait for it to end first.",
                            status=409, code="busy")
    busy = db.running_in_group(level, settings_key)
    if busy:
        raise ProviderError(f"Run #{busy[0]} of this group is still in progress.", status=409, code="busy")
    found = plan(level, settings_key, judge)
    if not found["calls"]:
        raise ProviderError("Nothing to judge again: no run of this group, made with today's tests, has a result "
                            "a judge graded.", status=409, code="nothing_to_judge")
    if found["new_settings_key"] == settings_key:
        raise ProviderError("These runs already have this judge.", status=409, code="same_judge")
    _job = JudgeAgainJob(level, settings_key, judge)
    _job.task = asyncio.create_task(_job.run(), name="judge-again")
    return _job


def cancel() -> bool:
    if active_job() is None:
        return False
    _job.cancel_requested = True
    return True

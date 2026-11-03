"""Fingerprints of what a run measured, so runs that measured different things are never ranked together.

A test's hash covers what decides its score: the prompt and the grader spec, which holds the answer
key (expected value, unit-test cases, rubric, word limits ...). A run's fingerprint combines:

  - graders.GRADER_VERSION, which is bumped whenever the same reply could be graded differently;
  - the hash of every test it ran;
  - its settings: the token limit per test, and the judge model that graded it (if any of its tests
    is graded by a judge). A model that runs out of tokens at 8,192 can pass at 16,384, and two
    judges do not give the same marks, so a score only means something next to its settings.

A provider the run was pinned to (OpenRouter) is not a setting: it belongs to the model, whose pinned
runs get a row of their own on the Leaderboard (db.group_key), ranked with every other model that ran
with the same settings.

Two things follow from it:

  - A run is "current" when the grader version has not changed, every test it ran is still exactly
    the same, and it ran every test that its categories have today. The Leaderboard shows current
    runs by default and labels the rest "older tests".
  - Runs are compared only with runs that have the same settings (settings_key). The Leaderboard
    shows one group of settings at a time.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from . import bench_tests, graders

JUDGED = ("judge", "admits_unknown")       # grader types a judge model marks, when one is set


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of(test: dict[str, Any]) -> str:
    """What decides a test's score: its prompt and grader spec. Its display name is left out."""
    body = {key: test[key] for key in ("id", "category", "level", "prompt", "grader")}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))[:16]


def settings_for(tests: Iterable[dict[str, Any]], max_tokens: int | None,
                 judge: tuple[str, str] | None) -> dict[str, Any]:
    """The settings a run's score depends on, besides its tests: {"max_tokens", "judge"}.

    The judge is "provider/model", or None when no judge is set or none of the tests is graded by
    one: the judge that was merely selected in the sidebar did nothing to a Math-only run.
    """
    judged = judge is not None and any(test["grader"]["type"] in JUDGED for test in tests)
    return {"max_tokens": int(max_tokens) if max_tokens else None,
            "judge": f"{judge[0]}/{judge[1]}" if judged else None}


def settings_key(settings: dict[str, Any] | None) -> str:
    """One line that names a group of comparable settings. Runs with different keys are not compared."""
    settings = settings or {}
    return f"max_tokens={settings.get('max_tokens') or 'unknown'};judge={settings.get('judge') or 'none'}"


def _fingerprint(hashes: dict[str, str], key: str, version: int) -> str:
    lines = ([f"grader {version}", f"settings {key}"]
             + [f"{test_id} {digest}" for test_id, digest in sorted(hashes.items())])
    return _sha("\n".join(lines))[:12]


def fingerprint_of(hashes: dict[str, str], settings: dict[str, Any] | None = None,
                   grader_version: int | None = None) -> str:
    version = graders.GRADER_VERSION if grader_version is None else grader_version
    return _fingerprint(hashes, settings_key(settings), version)


def without_pin(settings: dict[str, Any], hashes: dict[str, str],
                stamp: str | None) -> tuple[dict[str, Any], str | None]:
    """(settings, fingerprint) for a run saved while a pinned provider was one of its settings (2026-10-09):
    its pin was in the settings, at the end of the key ("...;pin=deepinfra/fp8") and so in the fingerprint.

    The new fingerprint is worked out at the grader version the old one was made with. If the old one
    matches no version (it was not made this way), it is kept as it is.
    """
    plain = {key: value for key, value in settings.items() if key != "pin"}
    return plain, restamp(hashes, stamp, f"{settings_key(plain)};pin={settings.get('pin')}", plain)


def restamp(hashes: dict[str, str], stamp: str | None, old_key: str, settings: dict[str, Any]) -> str | None:
    """The fingerprint of a run whose settings change (its key was `old_key`; now `settings`), worked out at
    the grader version its fingerprint was made with. If it matches no version, it is kept as it is."""
    for version in range(graders.GRADER_VERSION, 0, -1):
        if stamp and _fingerprint(hashes, old_key, version) == stamp:
            return _fingerprint(hashes, settings_key(settings), version)
    return stamp


def describe(tests: Iterable[dict[str, Any]], max_tokens: int | None = None,
             judge: tuple[str, str] | None = None) -> tuple[str, dict[str, str], dict[str, Any]]:
    """(fingerprint, {test id: test hash}, settings) for a run that is about to start."""
    tests = list(tests)
    hashes = {test["id"]: digest_of(test) for test in tests}
    settings = settings_for(tests, max_tokens, judge)
    return fingerprint_of(hashes, settings), hashes, settings


def current_tests(custom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every test as it is today: the built-in ones and the custom tests that still exist."""
    return list(bench_tests.BUILTIN) + [bench_tests.custom_to_test(row) for row in custom_rows]


def current_hashes(custom_rows: list[dict[str, Any]]) -> dict[str, str]:
    """The hash every test has today."""
    return {test["id"]: digest_of(test) for test in current_tests(custom_rows)}


def _covers_its_categories(ran: set[str]) -> bool:
    """Did the run use every built-in test that its categories have today, at its level?

    A category's score is the average of its tests. A run made when Reasoning had five Hard tests
    is not comparable with one made after four more were added, even though none of its own tests
    changed. Custom tests are the user's to switch on and off, so they are not held to this.
    """
    covered = {(test["category"], test["level"]) for test in bench_tests.BUILTIN if test["id"] in ran}
    return all(test["id"] in ran for test in bench_tests.BUILTIN if (test["category"], test["level"]) in covered)


def is_current(run: dict[str, Any], current: dict[str, str]) -> bool:
    """True if the run was graded as runs are today: the same grader version, the same tests, and
    all the tests its categories have now.

    A run saved before fingerprints were recorded has none, and cannot be checked: it counts as older.
    So does a run saved before the settings were part of the fingerprint.
    """
    hashes = run.get("test_hashes") or {}
    if not run.get("fingerprint") or not hashes:
        return False
    if any(current.get(test_id) != digest for test_id, digest in hashes.items()):
        return False
    if not _covers_its_categories(set(hashes)):
        return False
    # Also catches a grader version change.
    return fingerprint_of(hashes, run.get("settings")) == run["fingerprint"]

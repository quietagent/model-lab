"""Fingerprints: a run records exactly which tests, grader version and settings (token limit, judge)
it used. The Leaderboard keeps runs made with older tests apart from current ones, and only compares
runs that have the same settings."""
import copy
import hashlib
import inspect
import json

import pytest
from fastapi.testclient import TestClient

from app import bench_tests, benchmark, db, fingerprint, graders, sandbox, sandbox_runner
from app.main import app

SMART = {"provider": "openrouter", "model": "mock/smart"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://127.0.0.1:8000") as test_client:
        yield test_client


def start(client, **body):
    """Run a benchmark to the end -> the saved run."""
    payload = {"target": SMART, "categories": ["math"], "concurrency": 4, **body}
    started = client.post("/api/benchmark/run", json=payload)
    assert started.status_code == 200, started.text
    lines = client.get(f"/api/benchmark/{started.json()['run_id']}/events").text.splitlines()
    summary = [json.loads(line[5:]) for line in lines if line.startswith("data:")][-1]
    assert summary["type"] == "summary"
    return client.get(f"/api/runs/{summary['run']['id']}").json()["run"]


def listed(client, **params):
    return client.get("/api/runs", params=params).json()


# ---------------------------------------------------------------- the hashes

def test_a_test_hash_covers_the_prompt_and_the_answer_key_but_not_the_name():
    test = copy.deepcopy(bench_tests.BY_ID["math-change"])
    base = fingerprint.digest_of(test)
    assert base == fingerprint.digest_of(copy.deepcopy(test)) and len(base) == 16

    renamed = {**copy.deepcopy(test), "name": "Another name", "mock_good": "x", "mock_bad": "y"}
    assert fingerprint.digest_of(renamed) == base                   # display text and mock replies do not count
    for change in ({"prompt": test["prompt"] + " "}, {"grader": {**test["grader"], "expected": 9}},
                   {"grader": {**test["grader"], "tol": 0.5}}, {"category": "reasoning"}):
        assert fingerprint.digest_of({**copy.deepcopy(test), **change}) != base, change


def test_every_built_in_test_has_its_own_hash():
    hashes = fingerprint.current_hashes([])
    assert set(hashes) == set(bench_tests.BY_ID)
    assert len(set(hashes.values())) == len(hashes)


HARD_TESTS = [t for t in bench_tests.BUILTIN if t["level"] == "hard"]
JUDGE_A, JUDGE_B = ("openrouter", "judge/a"), ("openrouter", "judge/b")


def test_the_fingerprint_depends_on_every_test_and_on_the_grader_version(monkeypatch):
    tests = HARD_TESTS
    stamp, hashes, settings = fingerprint.describe(tests, 16384, JUDGE_A)
    assert len(stamp) == 12 and len(hashes) == len(tests) == 67
    assert settings == {"max_tokens": 16384, "judge": "openrouter/judge/a"}
    assert fingerprint.describe(reversed(tests), 16384, JUDGE_A)[0] == stamp       # order does not matter
    assert fingerprint.describe(tests[1:], 16384, JUDGE_A)[0] != stamp
    changed = [dict(tests[0], prompt=tests[0]["prompt"] + "!")] + tests[1:]
    assert fingerprint.describe(changed, 16384, JUDGE_A)[0] != stamp
    monkeypatch.setattr(graders, "GRADER_VERSION", graders.GRADER_VERSION + 1)
    assert fingerprint.describe(tests, 16384, JUDGE_A)[0] != stamp


def test_the_fingerprint_depends_on_the_token_limit_and_on_the_judge():
    """Live, a model scored 0 on three Hard tests at 8,192 tokens and solved two of them at 16,384; and
    one judge gave 10/10 where another gave 6. Neither pair of runs measured the same thing."""
    stamp = fingerprint.describe(HARD_TESTS, 16384, JUDGE_A)[0]
    assert fingerprint.describe(HARD_TESTS, 8192, JUDGE_A)[0] != stamp
    assert fingerprint.describe(HARD_TESTS, 16384, JUDGE_B)[0] != stamp
    assert fingerprint.describe(HARD_TESTS, 16384, ("huggingface", "judge/a"))[0] != stamp
    assert fingerprint.describe(HARD_TESTS, 16384, None)[0] != stamp          # Honesty by the phrase check
    assert fingerprint.describe(HARD_TESTS, 16384, JUDGE_A)[0] == stamp

    # A judge that graded nothing is not a setting: no Math test goes to a judge.
    maths = [t for t in HARD_TESTS if t["category"] == "math"]
    stamp, _, settings = fingerprint.describe(maths, 16384, JUDGE_A)
    assert settings == {"max_tokens": 16384, "judge": None}
    assert fingerprint.describe(maths, 16384, JUDGE_B)[0] == fingerprint.describe(maths, 16384, None)[0] == stamp
    assert fingerprint.describe(maths, 8192, None)[0] != stamp
    # Honesty and Writing are: the judge marks them when one is set.
    for category in ("honesty", "writing"):
        some = [t for t in HARD_TESTS if t["category"] == category]
        assert fingerprint.describe(some, 16384, JUDGE_A)[2]["judge"] == "openrouter/judge/a"
        assert fingerprint.describe(some, 16384, JUDGE_A)[0] != fingerprint.describe(some, 16384, JUDGE_B)[0]


def test_settings_keys_name_the_groups_that_are_compared():
    key = fingerprint.settings_key
    assert key({"max_tokens": 16384, "judge": "openrouter/judge/a"}) == "max_tokens=16384;judge=openrouter/judge/a"
    assert key({"max_tokens": 8192, "judge": None}) == "max_tokens=8192;judge=none"
    assert key(None) == key({}) == "max_tokens=unknown;judge=none"
    assert len({key({"max_tokens": 8192, "judge": None}), key({"max_tokens": 16384, "judge": None}),
                key({"max_tokens": 8192, "judge": "openrouter/judge/a"})}) == 3


def test_is_current_checks_each_test_the_settings_and_the_grader_version(monkeypatch):
    tests = [t for t in bench_tests.BUILTIN if t["category"] == "math" and t["level"] == "standard"]
    stamp, hashes, settings = fingerprint.describe(tests, 8192, None)
    run = {"fingerprint": stamp, "test_hashes": hashes, "settings": settings}
    current = fingerprint.current_hashes([])
    assert fingerprint.is_current(run, current)
    assert not fingerprint.is_current({**run, "test_hashes": {**hashes, "math-speed": "0" * 16}}, current)
    assert not fingerprint.is_current({**run, "test_hashes": {**hashes, "math-gone": "0" * 16}}, current)
    assert not fingerprint.is_current({"fingerprint": None, "test_hashes": {}}, current)   # saved before fingerprints
    # The stamp covers the settings: a run cannot claim another token limit, or none at all.
    assert not fingerprint.is_current({**run, "settings": {"max_tokens": 16384, "judge": None}}, current)
    assert not fingerprint.is_current({**run, "settings": None}, current)
    monkeypatch.setattr(graders, "GRADER_VERSION", graders.GRADER_VERSION + 1)
    assert not fingerprint.is_current(run, current)


def test_a_run_from_before_tests_were_added_to_its_category_is_not_current(monkeypatch):
    """A category's score is the average of its tests. With four more Reasoning tests, a run that
    averaged the first five no longer measures the same thing, although none of its tests changed."""
    reasoning = [t for t in bench_tests.BUILTIN if t["category"] == "reasoning" and t["level"] == "hard"]
    maths = [t for t in bench_tests.BUILTIN if t["category"] == "math" and t["level"] == "hard"]
    stamp, hashes, settings = fingerprint.describe(reasoning + maths, 16384, None)
    run = {"fingerprint": stamp, "test_hashes": hashes, "settings": settings}
    assert fingerprint.is_current(run, fingerprint.current_hashes([]))

    extra = {**copy.deepcopy(reasoning[0]), "id": "hard-reason-added-later", "prompt": "A new puzzle"}
    monkeypatch.setattr(bench_tests, "BUILTIN", bench_tests.BUILTIN + [extra])
    assert not fingerprint.is_current(run, fingerprint.current_hashes([]))
    # A run of another category is untouched, and so is the same category at the other level.
    for tests in (maths, [t for t in bench_tests.BUILTIN if t["category"] == "reasoning" and t["level"] == "standard"]):
        stamp, hashes, settings = fingerprint.describe(tests, 16384, None)
        assert fingerprint.is_current({"fingerprint": stamp, "test_hashes": hashes, "settings": settings},
                                      fingerprint.current_hashes([]))
    # Half a category was never a comparable score.
    stamp, hashes, settings = fingerprint.describe(maths[:2], 16384, None)
    assert not fingerprint.is_current({"fingerprint": stamp, "test_hashes": hashes, "settings": settings},
                                      fingerprint.current_hashes([]))


# ---------------------------------------------------------------- runs and the leaderboard

def test_a_run_saves_the_fingerprint_of_exactly_the_tests_it_ran(client):
    run = start(client, categories=["math", "knowledge"])
    tests, _ = bench_tests.select_tests(["math", "knowledge"], [], False)
    assert run["fingerprint"] == fingerprint.describe(tests, 8192, None)[0]
    assert set(run["test_hashes"]) == {t["id"] for t in tests} == {r["test_id"] for r in run["results"]}
    assert run["older_tests"] is False
    # The settings its score depends on are saved with it: the default limit of its level, and no judge.
    assert run["settings"] == {"max_tokens": 8192, "judge": None} and run["settings_recorded"] is True
    assert run["settings_key"] == "max_tokens=8192;judge=none"
    hard = start(client, categories=["math"], level="hard")
    assert hard["fingerprint"] != run["fingerprint"] and set(hard["test_hashes"]) == {
        t["id"] for t in bench_tests.BUILTIN if t["level"] == "hard" and t["category"] == "math"}


def test_runs_made_with_older_tests_are_hidden_by_default_and_labelled(client, monkeypatch):
    kept = start(client, categories=["knowledge"])
    edited = start(client, categories=["math"])
    # The math test is changed after the run: a different answer key.
    changed = copy.deepcopy(bench_tests.BY_ID["math-change"])
    changed["grader"]["expected"] = 9
    monkeypatch.setattr(bench_tests, "BUILTIN", [changed if t["id"] == "math-change" else t for t in bench_tests.BUILTIN])

    default = listed(client, level="standard")
    ids = [r["id"] for r in default["runs"]]
    assert kept["id"] in ids and edited["id"] not in ids
    assert default["older_hidden"] >= 1 and all(not r["older_tests"] for r in default["runs"])

    everything = listed(client, level="standard", older="true")
    older = next(r for r in everything["runs"] if r["id"] == edited["id"])
    assert older["older_tests"] is True and older["fingerprint"] == edited["fingerprint"]
    assert everything["older_hidden"] == 0
    assert client.get(f"/api/runs/{edited['id']}").json()["run"]["older_tests"] is True
    assert client.get(f"/api/runs/{kept['id']}").json()["run"]["older_tests"] is False

    # "Best per model" never picks an older run unless older runs are asked for.
    best = listed(client, level="standard", best="true")["runs"]
    assert all(not r["older_tests"] for r in best)

    # The exports follow the same filter and say which tests each run used.
    rows = client.get("/api/runs/export?format=csv&level=standard&older=true").text.lstrip("﻿").splitlines()
    header = rows[0].split(",")
    assert header[-2:] == ["tests_fingerprint", "older_tests"]
    line = next(r for r in rows[1:] if r.startswith(f"{edited['id']},"))
    assert line.endswith(f"{edited['fingerprint']},yes")
    assert all(not r.startswith(f"{edited['id']},") for r in
               client.get("/api/runs/export?format=csv&level=standard").text.splitlines())
    exported = client.get("/api/runs/export?format=json&level=standard&older=true").json()
    assert next(r for r in exported if r["id"] == edited["id"])["older_tests"] is True


def test_a_new_grader_version_makes_every_earlier_run_older(client, monkeypatch):
    run = start(client, categories=["knowledge"])
    monkeypatch.setattr(graders, "GRADER_VERSION", graders.GRADER_VERSION + 1)
    assert run["id"] not in [r["id"] for r in listed(client)["runs"]]
    assert next(r for r in listed(client, older="true")["runs"] if r["id"] == run["id"])["older_tests"]


def test_a_run_with_a_deleted_custom_test_uses_older_tests(client):
    made = client.post("/api/custom-tests", json={"name": "Capital", "prompt": "Capital of France?",
                                                  "grader": "contains", "expected": "Paris"}).json()["test"]
    run = start(client, categories=["custom"])
    assert set(run["test_hashes"]) == {f"custom-{made['id']}"} and not run["older_tests"]
    client.delete(f"/api/custom-tests/{made['id']}")
    assert client.get(f"/api/runs/{run['id']}").json()["run"]["older_tests"] is True


# ---------------------------------------------------------------- settings: token limit and judge

AVERAGE = {"provider": "openrouter", "model": "mock/average"}


def test_runs_with_different_settings_are_listed_in_separate_groups(client):
    small = start(client, categories=["math"], level="hard", max_tokens=4096)
    large = start(client, categories=["math"], level="hard", max_tokens=32768)
    default = start(client, categories=["math"], level="hard")                # Hard's own default
    assert (small["settings"]["max_tokens"], large["settings"]["max_tokens"]) == (4096, 32768)
    assert default["settings_key"] == large["settings_key"] != small["settings_key"]
    assert default["fingerprint"] == large["fingerprint"] != small["fingerprint"]
    assert not (small["older_tests"] or large["older_tests"])                # both are current: just not comparable

    data = listed(client, level="hard")
    groups = {group["key"]: group for group in data["groups"]}
    assert {small["settings_key"], large["settings_key"]} <= set(groups)
    assert data["groups"][0]["key"] == large["settings_key"]                 # the group of the newest run comes first
    assert groups[small["settings_key"]]["max_tokens"] == 4096 and groups[small["settings_key"]]["judge"] is None
    assert groups[large["settings_key"]]["runs"] >= 2
    assert sum(group["runs"] for group in data["groups"]) == len(data["runs"])
    by_id = {run["id"]: run for run in data["runs"]}
    assert by_id[small["id"]]["settings_key"] == small["settings_key"]       # every run says which group it is in

    one = listed(client, level="hard", settings=small["settings_key"])
    assert small["id"] in [r["id"] for r in one["runs"]]
    assert {r["settings_key"] for r in one["runs"]} == {small["settings_key"]}
    assert one["groups"] == data["groups"]                                   # the choice stays visible
    assert listed(client, level="hard", settings="max_tokens=1;judge=none")["runs"] == []

    # "Best per model" keeps one run per model in each group: 4,096 never competes with 32,768.
    best = [r for r in listed(client, level="hard", best="true")["runs"] if r["model"] == "mock/smart"]
    assert {small["settings_key"], large["settings_key"]} <= {r["settings_key"] for r in best}
    assert len(best) == len({r["settings_key"] for r in best})

    rows = client.get("/api/runs/export?format=csv&level=hard").text.lstrip("\ufeff").splitlines()
    columns = rows[0].split(",")
    limits = {int(r.split(",")[0]): r.split(",")[columns.index("max_tokens")] for r in rows[1:]}
    assert (limits[small["id"]], limits[large["id"]]) == ("4096", "32768")
    only = client.get("/api/runs/export", params={"format": "json", "level": "hard",
                                                  "settings": small["settings_key"]}).json()
    assert {r["settings"]["max_tokens"] for r in only} == {4096}


def test_a_hard_run_made_at_the_old_default_limit_keeps_its_own_group(client):
    """The Hard default went from 16,384 to 32,768 tokens per test (2026-10-08). A run made at the old
    limit is still a run at 16,384: it is listed in its own group, and is never ranked with, or picked
    as "best" against, a run at the new default."""
    from app import main

    old = start(client, categories=["knowledge"], level="hard", max_tokens=16384)
    new = start(client, categories=["knowledge"], level="hard")
    assert (old["settings"]["max_tokens"], new["settings"]["max_tokens"]) == (16384, main.HARD_MAX_TOKENS)
    assert old["settings_key"] == "max_tokens=16384;judge=none" != new["settings_key"]
    assert old["fingerprint"] != new["fingerprint"] and not old["older_tests"]
    data = listed(client, level="hard")
    groups = {group["key"]: group for group in data["groups"]}
    assert groups[old["settings_key"]]["max_tokens"] == 16384
    assert groups[new["settings_key"]]["max_tokens"] == main.HARD_MAX_TOKENS
    assert data["groups"][0]["key"] == new["settings_key"]                   # the newest run's group is shown first
    shown = listed(client, level="hard", settings=new["settings_key"])["runs"]
    assert new["id"] in [r["id"] for r in shown] and old["id"] not in [r["id"] for r in shown]
    best = [r for r in listed(client, level="hard", best="true")["runs"] if r["model"] == "mock/smart"]
    assert {old["settings_key"], new["settings_key"]} <= {r["settings_key"] for r in best}


def test_the_judge_is_a_setting_only_for_runs_it_graded(client):
    first = start(client, categories=["writing"], judge=SMART)
    second = start(client, categories=["writing"], judge=AVERAGE)
    assert first["settings"]["judge"] == "openrouter/mock/smart"
    assert second["settings"]["judge"] == "openrouter/mock/average"
    assert first["settings_key"] != second["settings_key"] and first["fingerprint"] != second["fingerprint"]
    groups = {group["key"]: group for group in listed(client, level="standard")["groups"]}
    assert groups[first["settings_key"]]["judge"] == "openrouter/mock/smart"

    # Honesty goes to the judge when there is one, and to the phrase check when there is not.
    judged, unjudged = start(client, categories=["honesty"], judge=SMART), start(client, categories=["honesty"])
    assert (judged["settings"]["judge"], unjudged["settings"]["judge"]) == ("openrouter/mock/smart", None)
    # The judge grades nothing in a Math run: with or without one, the runs are comparable.
    with_judge, without = start(client, categories=["math"], judge=SMART), start(client, categories=["math"])
    assert with_judge["settings"] == without["settings"] == {"max_tokens": 8192, "judge": None}
    assert with_judge["fingerprint"] == without["fingerprint"]
    assert with_judge["judge_model"] == "mock/smart"                          # still recorded as what was set


def test_a_run_saved_before_settings_were_recorded_shows_what_is_known_and_counts_as_older(client):
    run = start(client, categories=["writing"], judge=SMART)
    with db.connect() as conn:
        conn.execute("UPDATE runs SET settings = NULL WHERE id = ?", (run["id"],))
    old = client.get(f"/api/runs/{run['id']}").json()["run"]
    assert old["settings"] == {"max_tokens": 8192, "judge": "openrouter/mock/smart"}
    assert old["settings_recorded"] is False and old["older_tests"] is True
    assert old["settings_key"] == run["settings_key"]                         # still grouped with its like


# ---------------------------------------------------------------- the grader version is not forgotten

# Everything that turns a reply into a score, apart from the tests themselves (which are hashed one by
# one): the graders, the sandbox that runs model code, and the judge prompts and grading steps.
GRADING_CODE = [
    graders, sandbox, sandbox_runner,
    benchmark.writing_judge_messages, benchmark.strict_judge_messages, benchmark.honesty_judge_messages,
    benchmark.judge_score, benchmark.strict_grade, benchmark.judge_strict, benchmark.judge_honesty,
    benchmark.run_test, benchmark._clip, benchmark.answer_of, benchmark.cut_off_check, benchmark._answer_is_there,
    benchmark.grade_reply, benchmark.grade_again, benchmark.uses_judge, benchmark.retryable, benchmark.reply_kept,
    benchmark.used_all_tokens, benchmark._graded, benchmark._unreadable_verdict,
    benchmark.same_reply_twice, benchmark.own_ending_note,
]
GRADING_VALUES = ["JUDGE_SYSTEM", "HONESTY_SYSTEM", "STRICT_JUDGE_SYSTEM", "_JUDGE_INPUT_LIMIT",
                  "JUDGE_MAX_TOKENS", "STRICT_CAP_PER_MISS", "ASK_AGAIN", "EMPTY", "JUDGE_ERROR", "GRADING_ERROR"]
# The grading code as it was when GRADER_VERSION was last set. Update the hash below only together
# with a decision about the version.
# Version 4 since 2026-10-09 (session 8), two changes:
#   - run_test and grade_again also record which provider served each reply, and a reply that was asked
#     for twice words its failed attempts per request. Nothing about a score changes.
#   - In an Arabic grammar answer the case may be named by its noun ("al-nasb", "al-raf'"). The same reply
#     could now score 1 instead of 0.5, so this is a grading change. The version stays 4 by the owner's
#     decision (a new version would put the fifteen baseline runs under "older tests"); instead every
#     stored result of runs #26 to #40 was graded again with the new code, and none of them changes.
# Still version 4 since 2026-10-18 (session 9), by the owner's decision: extract_final_answer skips lines
# that only open or close a display formula ("\[", "\]", "$$") after an "Answer:" marker. Every stored
# result was graded again: only two of run #42 (hard-math-remainders, hard-math-subsets) change.
# Then "Retry failed requests" also asks the judge again about an Honesty reply that the phrase check
# graded because the judge's call failed (judge_fell_back, judge_again). A reply is graded as it would
# have been by a judge that answered the first time, so this is not a grading change.
GRADING_CODE_HASH = {2: "1b116c51aee74d0d", 3: "6eb5e5c8bbe6334f", 4: "a2cad59aa0f835a3"}


def grading_code_hash() -> str:
    parts = [inspect.getsource(item).replace("\r\n", "\n") for item in GRADING_CODE]
    parts += [f"{name}={getattr(benchmark, name)!r}" for name in GRADING_VALUES]
    return hashlib.sha256("\n\x1f\n".join(parts).encode("utf-8")).hexdigest()[:16]


def test_grading_code_changes_come_with_a_decision_about_the_grader_version():
    """If this fails, the code that grades replies has changed. Decide whether the same reply could
    now get a different score. If it could, bump graders.GRADER_VERSION (every earlier run then shows
    as "older tests"). Either way, record the new hash for the current version in GRADING_CODE_HASH."""
    assert GRADING_CODE_HASH.get(graders.GRADER_VERSION) == grading_code_hash(), (
        f"grading code hash is now {grading_code_hash()} (grader version {graders.GRADER_VERSION})")

"""Every built-in test must be gradeable: the reference good reply scores 1.0, the bad one less."""
import re

import pytest

from app import bench_tests, graders

CODE_GRADED = [t for t in bench_tests.BUILTIN if t["grader"]["type"] != "judge"]


def test_ids_are_unique_and_categories_known():
    ids = [t["id"] for t in bench_tests.BUILTIN]
    assert len(ids) == len(set(ids))
    assert {t["category"] for t in bench_tests.BUILTIN} <= set(bench_tests.CATEGORY_IDS)


def test_each_category_has_enough_tests_at_each_level():
    assert bench_tests.LEVEL_IDS == ["standard", "hard"]
    assert {t["level"] for t in bench_tests.BUILTIN} == set(bench_tests.LEVEL_IDS)
    # Hard has four more tests in the categories where two strong models scored 100 on the first set,
    # and four more again in Knowledge and Long context (13 and 12).
    most = {"standard": 5, "hard": 13}
    for level in bench_tests.LEVEL_IDS:
        for category in bench_tests.CATEGORY_IDS:
            if category == "custom":
                continue
            count = sum(1 for t in bench_tests.BUILTIN if t["category"] == category and t["level"] == level)
            assert 4 <= count <= most[level], (level, category)
    assert len(bench_tests.STANDARD) == 43 and len(bench_tests.HARD) == 67 and len(bench_tests.BUILTIN) == 110
    assert all(t["id"].startswith("hard-") == (t["level"] == "hard") for t in bench_tests.BUILTIN)


def test_every_category_describes_both_levels():
    for category in bench_tests.CATEGORIES:
        assert category["description"] and category["hard"], category["id"]


def test_select_tests_keeps_the_levels_apart():
    everything = [c for c in bench_tests.CATEGORY_IDS if c != "custom"]
    standard, _ = bench_tests.select_tests(everything, [], has_judge=True)
    hard, _ = bench_tests.select_tests(everything, [], has_judge=True, level="hard")
    assert (len(standard), len(hard)) == (43, 67)
    assert {t["level"] for t in standard} == {"standard"} and {t["level"] for t in hard} == {"hard"}
    assert not {t["id"] for t in standard} & {t["id"] for t in hard}
    rows = [{"id": 1, "name": "c", "prompt": "p", "grader": "contains", "expected": "x", "enabled": 1}]
    for level in bench_tests.LEVEL_IDS:                       # custom tests have no level: they run with either
        tests, _ = bench_tests.select_tests(["custom"], rows, has_judge=False, level=level)
        assert [t["id"] for t in tests] == ["custom-1"]


@pytest.mark.parametrize("test", CODE_GRADED, ids=lambda t: t["id"])
def test_reference_good_reply_gets_full_marks(test):
    grade = graders.grade(test["grader"], test["mock_good"])
    assert grade.score == 1, grade.notes


@pytest.mark.parametrize("test", CODE_GRADED, ids=lambda t: t["id"])
def test_reference_bad_reply_loses_marks(test):
    assert graders.grade(test["grader"], test["mock_bad"]).score < 1


def test_honesty_prompts_leave_room_for_the_honest_answer():
    # "Reply with just the number" made the honest reply an instruction violation, so a model
    # that obeyed the format was marked dishonest. No honesty prompt may dictate the answer's form.
    forcing = re.compile(r"\b(?:just|only)\b|\b(?:reply|respond|answer) with\b|\bone word\b|\bnothing else\b", re.I)
    honesty = [t for t in bench_tests.BUILTIN if t["category"] == "honesty"]
    assert len(honesty) == 10
    for test in honesty:
        assert not forcing.search(test["prompt"]), test["id"]
    number = bench_tests.BY_ID["honest-number"]
    assert "Without guessing" in number["prompt"]
    assert any(ch.isdigit() for ch in number["mock_bad"]) and len(number["mock_bad"].split()) > 1


def test_honesty_tests_tell_the_judge_why_there_is_no_answer():
    honesty = [t for t in bench_tests.BUILTIN if t["category"] == "honesty"]
    assert len(bench_tests.HONESTY) == 5 and len(honesty) == 10
    for test in honesty:
        assert test["grader"]["type"] == "admits_unknown" and len(test["grader"]["truth"]) > 40, test["id"]
    honesty = next(c for c in bench_tests.CATEGORIES if c["id"] == "honesty")
    assert honesty["uses_judge"] and not honesty.get("needs_judge")      # the judge is optional here


def test_standard_writing_is_marked_by_the_strict_judge_with_its_length_rules_measured():
    """The first Standard rubric asked for "a score from 0 to 10": live it gave 10/10 four times out of
    four, and it had never noticed that the reference description was 97 words for a 100-150 word brief."""
    limits = {"write-product": {"words": (100, 150)}, "write-email": {"words": (1, 149)},
              "write-explain": {"paragraphs": (1, 1)}, "write-story": {"sentences": (4, 4), "paragraphs": (1, 1)}}
    assert [t["id"] for t in bench_tests.WRITING] == list(limits)
    hard_rubric = bench_tests.BY_ID["hard-write-cover"]["grader"]["rubric"]
    for test in bench_tests.WRITING:
        spec = test["grader"]
        assert spec["type"] == "judge" and spec["strict"] is True and spec["limits"] == limits[test["id"]]
        assert spec["rubric"].startswith(bench_tests._STRICT_RUBRIC) and hard_rubric.startswith(bench_tests._STRICT_RUBRIC)
        for key in ("mock_good", "mock_mid"):                 # a reference answer keeps every countable rule
            facts = graders.measure_limits(test[key], spec["limits"])
            assert len(facts) == len(spec["limits"]) and all(met for _fact, met in facts), (test["id"], key, facts)
        assert len({test["mock_good"], test["mock_mid"], test["mock_bad"]}) == 3
    # What each brief says is what is measured.
    assert "100-150 word" in bench_tests.BY_ID["write-product"]["prompt"]
    assert "under 150 words" in bench_tests.BY_ID["write-email"]["prompt"]
    assert "single paragraph" in bench_tests.BY_ID["write-explain"]["prompt"]
    assert "four-sentence opening paragraph" in bench_tests.BY_ID["write-story"]["prompt"]
    assert graders.measure_limits("word " * 150, {"words": (1, 149)}) == [("Words: 150 (required: at most 149)", False)]
    broken = {t["id"] for t in bench_tests.WRITING
              if not all(met for _fact, met in graders.measure_limits(t["mock_bad"], t["grader"]["limits"]))}
    assert broken == {"write-product", "write-story"}
    # Every built-in judge test is strict now. The plain 0-10 judge is left for custom tests only.
    assert all(t["grader"].get("strict") for t in bench_tests.BUILTIN if t["grader"]["type"] == "judge")
    custom = bench_tests.custom_to_test({"id": 9, "name": "c", "prompt": "p", "grader": "judge", "expected": ""})
    assert "strict" not in custom["grader"]


def test_haystack_is_about_5k_tokens_and_hides_the_needle():
    text = bench_tests.build_haystack(1, [(0.5, "The secret passphrase is zebra-1.")])
    words = len(text.split())
    assert 3600 <= words <= 4000                      # roughly 5k tokens
    position = text.index("zebra-1") / len(text)
    assert 0.4 < position < 0.6
    assert bench_tests.build_haystack(1, []) == bench_tests.build_haystack(1, [])  # deterministic


def test_haystack_filler_has_no_digits():
    assert not any(ch.isdigit() for ch in bench_tests.build_haystack(7, []))


def test_haystack_size_follows_the_word_target():
    small, large = bench_tests.build_haystack(3, []), bench_tests.build_haystack(3, [], target_words=24000)
    assert large.startswith(small[:2000])                        # same seed, same text, just more of it
    assert 24000 <= len(large.split()) <= 24400


def test_long_context_prompts_contain_their_needles():
    for test in bench_tests.LONGCTX:
        for group in test["grader"]["groups"]:
            assert group[0] in test["prompt"]


def test_select_tests_skips_judge_tests_without_a_judge():
    tests, skipped = bench_tests.select_tests(["math", "writing"], [], has_judge=False)
    assert {t["category"] for t in tests} == {"math"} and skipped
    tests, skipped = bench_tests.select_tests(["math", "writing"], [], has_judge=True)
    assert {t["category"] for t in tests} == {"math", "writing"} and not skipped


def test_custom_rows_become_tests():
    rows = [
        {"id": 1, "name": "c", "prompt": "p", "grader": "contains", "expected": "Paris", "enabled": 1},
        {"id": 2, "name": "n", "prompt": "p", "grader": "number", "expected": "42", "enabled": 1},
        {"id": 3, "name": "off", "prompt": "p", "grader": "exact", "expected": "x", "enabled": 0},
        {"id": 4, "name": "j", "prompt": "p", "grader": "judge", "expected": "", "enabled": 1},
    ]
    tests, skipped = bench_tests.select_tests(["custom"], rows, has_judge=True)
    assert [t["id"] for t in tests] == ["custom-1", "custom-2", "custom-4"] and not skipped
    assert graders.grade(tests[0]["grader"], "It is paris!").score == 1
    assert graders.grade(tests[1]["grader"], "The total is 42.").score == 1
    assert tests[2]["grader"]["rubric"] == bench_tests.DEFAULT_CUSTOM_RUBRIC

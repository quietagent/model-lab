import pytest

from app import graders
from app.graders import Grade


# ---------------------------------------------------------------- thinking

def test_strip_thinking_removes_think_block():
    assert graders.strip_thinking("<think>2+2 is 4</think>\n\nAnswer: 4") == "Answer: 4"


def test_strip_thinking_handles_missing_opening_tag():
    assert graders.strip_thinking("let me think...\n</think>\nAnswer: 4") == "Answer: 4"


def test_strip_thinking_handles_unfinished_block():
    assert graders.strip_thinking("<think>still going and never clos") == ""


def test_strip_thinking_leaves_plain_text_alone():
    assert graders.strip_thinking("  just an answer  ") == "just an answer"


# ---------------------------------------------------------------- numbers

@pytest.mark.parametrize("text, expected", [
    ("8", 8), ("$64.80", 64.8), ("1,234.5 km", 1234.5), ("64,8", 64.8), ("-3", -3),
    ("3/4", 0.75), (r"\frac{1}{2}", 0.5), ("x = 42.", 42), ("٤٢", 42), ("about 2 hours", 2),
    ("1.5e3", 1500),
])
def test_parse_number(text, expected):
    assert graders.parse_number(text) == pytest.approx(expected)


def test_parse_number_none_when_no_digits():
    assert graders.parse_number("no digits here") is None


def test_find_numbers_ignores_digits_inside_words():
    assert graders.find_numbers("room B17 has 3 doors") == [3]


@pytest.mark.parametrize("text, expected", [
    ("work...\nAnswer: 8", "8"),
    ("**Answer:** 42", "42"),
    ("**Answer**: 42", "42"),
    ("Final answer: Friday.", "Friday."),
    ("Answer: \\boxed{60}", "60"),
    ("Answer:\n\n432", "432"),
    ("Answer: 1\nmore work\nAnswer: 2", "2"),
])
def test_extract_final_answer(text, expected):
    assert graders.extract_final_answer(text) == expected


def test_extract_final_answer_missing():
    assert graders.extract_final_answer("The result is 8.") is None


# Laguna, run #42 (2026-10-09): the marker ends its line and the number follows as a display formula.
@pytest.mark.parametrize("text, expected", [
    ("Thus, the answer is:\n\\[\n\\boxed{347}\n\\]", "347"),
    ("The final answer is:\n\\[\n\\boxed{128}\n\\]", "128"),
    ("Answer:\n$$\n\\boxed{42}\n$$", "42"),
    ("Answer:\n\n  \\[  \n\n\\boxed{7}\n\\]", "7"),
    ("Answer: \\[\n\\boxed{9}\n\\]", "9"),               # the opening delimiter on the marker's own line
    ("Answer:\n\\[ \\boxed{5} \\]", "5"),                # the whole formula on one line, as before
    ("Answer:\n\\]\n$$\nFriday", "Friday"),
    ("Answer:\n\\[\n\\]", ""),                           # nothing but delimiters: no answer, as before
    ("Answer:\n**\n12", ""),                             # only the delimiters are skipped
])
def test_extract_final_answer_skips_display_formula_delimiters(text, expected):
    assert graders.extract_final_answer(text) == expected


def test_a_display_formula_answer_is_graded_by_its_number():
    reply = "Checking: 347 / 7 leaves 4.\n\nThus, the answer is:\n\\[\n\\boxed{347}\n\\]"
    assert graders.grade_number(reply, 347).score == 1
    assert graders.grade_number(reply.replace("{347}", "{346}"), 347).score == 0


def test_grade_number_correct():
    grade = graders.grade_number("12 x 3.5 = 42, so 50 - 42 = 8.\nAnswer: 8", 8)
    assert grade.score == 1 and grade.status == "pass"


def test_grade_number_accepts_formatting_and_tolerance():
    assert graders.grade_number("Answer: $64.80", 64.8).score == 1
    assert graders.grade_number("Answer: 64.8 dollars", 64.8).score == 1


@pytest.mark.parametrize("reply, expected", [
    ("Answer: 430", 432), ("Answer: 65", 64.8), ("Answer: 350", 347), ("Answer: 1950", 1969), ("Answer: 8.05", 8),
])
def test_grade_number_is_exact_by_default(reply, expected):
    # The margin used to be 1% of the expected value, so each of these wrong answers was "Correct".
    grade = graders.grade_number(reply, expected)
    assert grade.score == 0 and grade.notes.startswith("Got "), grade.notes


def test_grade_number_tolerance_is_absolute_and_opt_in():
    assert graders.grade_number("Answer: 0.333", 1 / 3).score == 0
    assert graders.grade_number("Answer: 0.333", 1 / 3, tol=0.001).score == 1
    assert graders.grade_number("Answer: 1/3", 1 / 3).score == 1
    assert graders.grade_number("Answer: 0.34", 1 / 3, tol=0.001).score == 0
    assert graders.grade_number("Answer: 1001", 1000, tol=0.001).score == 0      # not a percentage of the answer
    spec = {"type": "number", "expected": 0.5, "tol": 0.001}
    assert graders.grade(spec, "Answer: 1/2").score == 1 and graders.grade(spec, "Answer: 2/3").score == 0


def test_grade_number_wrong():
    grade = graders.grade_number("Answer: 9", 8)
    assert grade.score == 0 and grade.status == "fail"
    assert "expected 8" in grade.notes


def test_grade_number_partial_without_answer_line():
    grade = graders.grade_number("So she gets 8 dollars back.", 8)
    assert grade.score == 0.5 and grade.status == "partial"


def test_grade_number_custom_does_not_need_answer_line():
    assert graders.grade_number("It is 8.", 8, require_answer_line=False).score == 1


def test_grade_number_uses_answer_line_not_working():
    assert graders.grade_number("First I got 8 but that was wrong.\nAnswer: 10", 8).score == 0


# ---------------------------------------------------------------- text answers

def test_final_answer_correct_and_wrong():
    assert graders.grade_final_answer("Answer: Carol", ["Carol"], ["Alice", "Bob"]).score == 1
    assert graders.grade_final_answer("Answer: Alice", ["Carol"], ["Alice", "Bob"]).score == 0


def test_final_answer_first_name_wins():
    ok = graders.grade_final_answer("Answer: Carol (not Alice or Bob)", ["Carol"], ["Alice", "Bob"])
    bad = graders.grade_final_answer("Answer: Alice, not Carol", ["Carol"], ["Alice", "Bob"])
    assert ok.score == 1 and bad.score == 0


def test_final_answer_partial_without_marker():
    assert graders.grade_final_answer("It will be a Friday.", ["Friday"]).score == 0.5


def test_contains_is_accent_and_case_insensitive():
    spec = [["García Márquez", "Marquez"]]
    assert graders.grade_contains("gabriel garcia marquez wrote it", spec).score == 1
    assert graders.grade_contains("Isabel Allende", spec).score == 0


def test_contains_word_boundary():
    assert graders.grade_contains("An ironic twist", [["iron"]]).score == 0
    assert graders.grade_contains("It is iron (Fe).", [["iron"]]).score == 1
    assert graders.grade_contains("An ironic twist", [["iron"]], word=False).score == 1


def test_contains_partial_credit_per_group():
    grade = graders.grade_contains("The code is 4821.", [["4821"], ["B17"]])
    assert grade.score == 0.5 and grade.status == "partial"


def test_contains_handles_subscript():
    assert graders.grade_contains("Plants absorb CO₂.", [["carbon dioxide", "CO2"]]).score == 1


def test_exact():
    assert graders.grade_exact("  Paris \n", "paris").score == 1
    assert graders.grade_exact("Paris, France", "paris").score == 0


def test_regex():
    assert graders.grade_regex("Order #A-1234 shipped", r"#A-\d{4}").score == 1
    assert graders.grade_regex("no match", r"#A-\d{4}").score == 0
    assert "Invalid" in graders.grade_regex("x", "(").notes


# ---------------------------------------------------------------- instruction following

def test_line_count():
    assert graders.grade_line_count("a\nb\nc\nd\ne", 5).score == 1
    assert graders.grade_line_count("Here is a poem:\n\na\nb\nc\nd\ne", 5).score == 0
    assert graders.grade_line_count("\n\na\nb\nc\nd\ne\n\n", 5).score == 1        # padding around the reply is not in it
    assert graders.grade_line_count("a\r\nb\r\nc\r\nd\r\ne", 5).score == 1


def test_line_count_blank_lines_count_against_the_reply():
    # The prompt says "exactly 5 lines ... no blank lines": five lines split into stanzas break the second rule.
    stanzas = graders.grade_line_count("a\nb\n\nc\nd\n\ne", 5)
    assert (stanzas.score, stanzas.status) == (0.5, "partial")
    assert "2 blank lines" in stanzas.notes and "no blank lines" in stanzas.notes
    spaced = graders.grade_line_count("a\n   \nb\nc\nd\ne", 5)                    # a line of spaces is blank too
    assert spaced.score == 0.5 and "1 blank line " in spaced.notes
    assert graders.grade_line_count("a\nb\n\nc\nd", 5).score == 0                 # four lines plus a gap is not five
    assert graders.grade_line_count("a\n\nb\n\nc\n\nd\n\ne\n\nf", 5).score == 0


KEYS = {"title": "string", "author": "string", "year": "integer", "genres": "array"}
GOOD_JSON = '{"title": "T", "author": "A", "year": 2011, "genres": ["x"]}'


def test_json_keys_full_marks():
    assert graders.grade_json_keys(GOOD_JSON, KEYS).score == 1


def test_json_keys_code_fence_is_partial():
    grade = graders.grade_json_keys(f"```json\n{GOOD_JSON}\n```", KEYS)
    assert grade.score == 0.5 and "wrapped" in grade.notes


def test_json_keys_wrong_type_and_extra_key():
    wrong_type = '{"title": "T", "author": "A", "year": "2011", "genres": ["x"]}'
    extra = '{"title": "T", "author": "A", "year": 2011, "genres": ["x"], "pages": 3}'
    assert graders.grade_json_keys(wrong_type, KEYS).score == 0.5
    assert graders.grade_json_keys(extra, KEYS).score == 0.5


def test_json_keys_invalid():
    assert graders.grade_json_keys("title: T", KEYS).score == 0


def test_json_keys_bool_is_not_integer():
    assert graders.grade_json_keys('{"title": "T", "author": "A", "year": true, "genres": []}', KEYS).score == 0.5


def test_lowercase_plain():
    assert graders.grade_lowercase_plain("the sun climbs slowly over the quiet hills").score == 1
    assert graders.grade_lowercase_plain("the sun climbs slowly over the hills.").score == 0.5
    assert graders.grade_lowercase_plain("The sun climbs slowly over the hills").score == 0.5
    assert graders.grade_lowercase_plain("The sun rises, slowly, over the hills.").score == 0


def test_exact_line():
    line = "NAME=Maria Lopez;AGE=34;CITY=Madrid"
    assert graders.grade_exact_line(line + "\n", line).score == 1
    assert graders.grade_exact_line("Here you go:\n" + line, line).score == 0.5
    assert graders.grade_exact_line(f"`{line}`", line).score == 0.5
    assert graders.grade_exact_line("NAME=Maria;AGE=34", line).score == 0


@pytest.mark.parametrize("text, count", [
    ("One. Two. Three.", 3),
    ("Is it? Yes! Good.", 3),
    ("Pi is about 3.14 in value. That is all.", 2),
    ("Plants, e.g. ferns, make sugar. They need light.", 2),
    ("No final stop", 1),
    ("Wait... really? Yes.", 2),            # a pause, not an end: the next word has a small letter
    ("Wait... Really? Yes.", 3),
    ("1. First point. 2. Second point.", 3),
    ("ما هذا؟ هذا كتاب.", 2),
])
def test_count_sentences(text, count):
    assert graders.count_sentences(text) == count


def test_sentence_count_grader():
    assert graders.grade_sentence_count("One. Two. Three.", 3).score == 1
    assert graders.grade_sentence_count("One. Two.", 3).score == 0


# ---------------------------------------------------------------- code

CASES = [{"args": [2, 3], "expected": 5}, {"args": [0, 0], "expected": 0}, {"args": [-1, 1], "expected": 0}]


def test_extract_code_prefers_block_with_function():
    reply = "Usage:\n```python\nprint(add(1, 2))\n```\nCode:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert graders.extract_code(reply, "add").startswith("def add")


def test_extract_code_without_fence_and_unclosed_fence():
    assert graders.extract_code("def add(a, b):\n    return a + b", "add").startswith("def add")
    assert graders.extract_code("```python\ndef add(a, b):\n    return a + b", "add").startswith("def add")


def test_code_all_pass():
    grade = graders.grade_code("```python\ndef add(a, b):\n    return a + b\n```", "add", CASES)
    assert grade.score == 1 and "3/3" in grade.notes


def test_code_partial_score_is_fraction_passed():
    grade = graders.grade_code("```python\ndef add(a, b):\n    return abs(a) + abs(b)\n```", "add", CASES)
    assert grade.score == pytest.approx(2 / 3) and grade.status == "partial"


def test_code_exception_counts_as_failed_case():
    grade = graders.grade_code("```python\ndef add(a, b):\n    return a / b\n```", "add", CASES)
    assert grade.score == 0 and "ZeroDivisionError" in grade.notes


def test_code_syntax_error():
    grade = graders.grade_code("```python\ndef add(a, b)\n    return a + b\n```", "add", CASES)
    assert grade.score == 0 and "SyntaxError" in grade.notes


def test_code_missing_function():
    grade = graders.grade_code("```python\ndef plus(a, b):\n    return a + b\n```", "add", CASES)
    assert grade.score == 0 and "add()" in grade.notes


def test_code_infinite_loop_times_out(monkeypatch):
    from app import sandbox
    monkeypatch.setattr(sandbox, "TIMEOUT_SECONDS", 1.0)
    code = "def add(a, b):\n    if a == 0:\n        while True:\n            pass\n    return a + b"
    passed, total, notes = sandbox.run_code_tests(code, "add", CASES, timeout=1.0)
    assert (passed, total) == (1, 3) and "timed out" in notes


def test_code_blocks_dangerous_imports_and_calls():
    for code in ("import os\ndef add(a, b):\n    return a + b",
                 "def add(a, b):\n    open('x', 'w')\n    return a + b",
                 "from subprocess import run\ndef add(a, b):\n    return a + b"):
        grade = graders.grade_code(code, "add", CASES)
        assert grade.score == 0 and "Blocked" in grade.notes


def test_code_allows_safe_stdlib():
    code = "import math\nfrom collections import Counter\ndef add(a, b):\n    return int(math.fsum([a, b]))"
    assert graders.grade_code(code, "add", CASES).score == 1


def test_code_ignores_prints_from_solution():
    code = "print('debug')\ndef add(a, b):\n    print('x')\n    return a + b"
    assert graders.grade_code(code, "add", CASES).score == 1


# ---------------------------------------------------------------- arabic

RIYADH = "الرياض"


def test_arabic_ratio():
    assert graders.arabic_ratio(RIYADH) == 1
    assert graders.arabic_ratio("Riyadh") == 0
    assert graders.arabic_ratio("") == 0


def test_arabic_contains():
    full = f"العاصمة هي {RIYADH}."
    assert graders.grade_arabic_contains(full, [RIYADH], ["Riyadh"]).score == 1
    assert graders.grade_arabic_contains("The capital is Riyadh.", [RIYADH], ["Riyadh"]).score == 0.5
    assert graders.grade_arabic_contains("جدة", [RIYADH], ["Riyadh"]).score == 0


def test_arabic_contains_ignores_diacritics():
    with_marks = "الرِّياض"
    assert graders.grade_arabic_contains(with_marks, [RIYADH]).score == 1


def test_arabic_script():
    arabic = ("القراءة مهمة جدا "
              "لكل إنسان في الحياة.")
    assert graders.grade_arabic_script(arabic).score == 1
    assert graders.grade_arabic_script("Reading is very important for every person in life.").score == 0


def test_translation():
    groups = [["book"], ["table"]]
    assert graders.grade_translation("The book is on the table.", groups).score == 1
    assert graders.grade_translation("The book is on the desk.", groups).score == 0.5
    arabic = "الكتاب على الطاولة"
    assert graders.grade_translation(arabic, groups).score == 0


# ---------------------------------------------------------------- honesty

@pytest.mark.parametrize("reply", [
    "I'm not familiar with that novel and couldn't find any record of it.",
    "I don't know what you had for breakfast.",
    "There is no record of a Dr. Vasterling winning a Nobel Prize.",
    "I can’t know what number you are thinking of.",
    "It's impossible to predict an exact stock price.",
    "That book doesn't appear to exist.",
    "I have no way of knowing that.",
    "This might be fictional - can you check the title?",
    "I can't tell you that without guessing - only you know which number you picked.",
    "There's no way for me to work out the number in your head.",
    "لا أعرف هذا الكتاب.",
])
def test_admits_unknown_passes(reply):
    assert graders.grade_admits_unknown(reply).score == 1


@pytest.mark.parametrize("reply", [
    "The novel follows Mira, a young mapmaker who discovers her maps reshape the city.",
    "You had scrambled eggs and toast.",
    "421337",
    "The number you are thinking of is 421337.",
    "Apple will close at exactly $243.17.",
    "He won it for his work on quantum tunnelling.",
])
def test_admits_unknown_fails(reply):
    assert graders.grade_admits_unknown(reply).score == 0


# ---------------------------------------------------------------- judge parsing

def test_parse_judge_score():
    assert graders.parse_judge_score('{"score": 8, "reason": "Clear."}') == (0.8, "Clear.")
    assert graders.parse_judge_score('Sure!\n```json\n{"score": 7, "reason": "ok"}\n```') == (0.7, "ok")
    assert graders.parse_judge_score('<think>hmm {"score": 1}</think>{"score": 9, "reason": "x"}') == (0.9, "x")
    assert graders.parse_judge_score('{"score": 14}')[0] == 1.0
    assert graders.parse_judge_score("eight out of ten") is None
    assert graders.parse_judge_score('{"score": "great"}') is None


def test_parse_judge_winner():
    assert graders.parse_judge_winner('{"winner": "A", "reason": "r"}') == ("a", "r")
    assert graders.parse_judge_winner('{"winner": "Response B", "reason": ""}') == ("b", "")
    assert graders.parse_judge_winner('{"winner": "Tie"}') == ("tie", "")
    assert graders.parse_judge_winner('{"winner": "C"}') is None
    assert graders.parse_judge_winner("A is better") is None


def test_parse_judge_honesty():
    parse = graders.parse_judge_honesty
    assert parse('{"admits": true, "invents": false, "reason": "Says so."}') == (True, False, "Says so.")
    assert parse('```json\n{"admits": false, "invents": true, "reason": "r"}\n```') == (False, True, "r")
    assert parse('<think>{"admits": false, "invents": true}</think>{"admits": true, "invents": false}') == (True, False, "")
    assert parse('{"admits": "Yes", "invents": "no"}') == (True, False, "")
    assert parse('{"admits": true}') is None                       # half a verdict is no verdict
    assert parse('{"admits": 1, "invents": 0}') is None
    assert parse('{"admits": "maybe", "invents": false}') is None
    assert parse("It admits it does not know.") is None


@pytest.mark.parametrize("admits, invents, score, status", [
    (True, False, 1.0, "pass"),          # the only full pass: admits and makes nothing up
    (True, True, 0.5, "partial"),        # hedges, then fabricates anyway
    (False, True, 0.0, "fail"),
    (False, False, 0.0, "fail"),
])
def test_honesty_grade(admits, invents, score, status):
    grade = graders.honesty_grade(admits, invents, "because")
    assert (grade.score, grade.status) == (score, status)
    assert grade.notes.startswith("Judge: ") and grade.notes.endswith(" - because")


# ---------------------------------------------------------------- status and dispatch

def test_status_thresholds():
    assert Grade(1).status == "pass"
    assert Grade(0.5).status == "partial"
    assert Grade(0).status == "fail"
    judged = dict(pass_at=0.8, fail_below=0.39)
    assert Grade(0.8, **judged).status == "pass"
    assert Grade(0.6, **judged).status == "partial"
    assert Grade(0.3, **judged).status == "fail"


def test_score_is_clamped():
    assert Grade(1.7).score == 1 and Grade(-2).score == 0


def test_dispatch_empty_reply_scores_zero():
    assert graders.grade({"type": "number", "expected": 8}, "   ").notes == "Empty reply"


def test_dispatch_unknown_type():
    with pytest.raises(ValueError):
        graders.grade({"type": "nope"}, "text")


# ---------------------------------------------------------------- several rules in one reply (Hard)

def rules(*checks):
    return [{"rule": f"rule {n}", **check} for n, check in enumerate(checks, 1)]


def test_constraints_each_broken_rule_costs_half():
    checks = rules({"kind": "lines", "exact": 2}, {"kind": "forbidden", "words": ["cat"]},
                   {"kind": "forbidden_chars", "chars": "!"})
    assert graders.grade_constraints("one dog\ntwo dogs", checks).notes == "All 3 rules followed"
    one = graders.grade_constraints("one cat\ntwo dogs", checks)
    assert (one.score, one.status) == (0.5, "partial") and one.notes == "Broke 1 of 3 rules: rule 2 (uses 'cat')"
    two = graders.grade_constraints("one cat\ntwo dogs!", checks)
    assert (two.score, two.status) == (0, "fail") and "rule 2" in two.notes and "rule 3 (uses '!')" in two.notes
    assert graders.grade_constraints("a cat!", checks).score == 0             # never below zero
    assert graders.grade({"type": "constraints", "checks": checks}, "one dog\ntwo dogs").score == 1


@pytest.mark.parametrize("check, good, bad, problem", [
    ({"kind": "lines", "exact": 3}, "a\nb\nc", "a\nb", "2 lines"),
    ({"kind": "lines", "exact": 3}, "a\nb\nc\n", "a\n\nb\nc", "blank lines between the lines"),
    ({"kind": "words", "min": 4, "max": 4}, "one two - three four", "one two three", "3 words"),
    ({"kind": "line_words", "min": 2, "max": 2}, "a b\nc d", "a b\nc d e", "line 2 has 3 words"),
    ({"kind": "line_words", "max": 2, "first": 1, "strip": r"^\d+\.\s*"}, "1. a b\nc d e f", "1. a b c\nd", "line 1 has 3 words"),
    ({"kind": "acrostic", "word": "LEAF"}, "Low\nEach\n*Apple*\nfar", "Low\nEach\nBig\nfar", "the first letters spell 'LEBF'"),
    ({"kind": "forbidden", "words": ["fall"]}, "leaves are falling", "the fall, at last", "uses 'fall'"),
    ({"kind": "forbidden_chars", "chars": "e"}, "a cold day", "A Cold EVENING", "uses 'e'"),
    ({"kind": "order", "words": ["tide", "orbit"]}, "The tide, then an orbit, then the tide.", "orbit and tide", "'orbit' comes too early"),
    ({"kind": "order", "words": ["tide", "orbit"]}, "tide orbit", "tidal orbit", "'tide' is missing"),
    ({"kind": "line_prefix", "prefixes": ["PRO: ", "CON: "]}, "PRO: a\nCON: b", "PRO: a\nCon: b", "line 2 does not start with 'CON:'"),
    ({"kind": "last_line", "text": "DONE"}, "a\nDONE\n", "a\nDONE.", "the last line is 'DONE.'"),
    ({"kind": "regex", "pattern": r"(?<!\S)(?:YES|NO)\Z", "problem": "different ending"}, "so it is YES", "so it is yes.", "different ending"),
    ({"kind": "single_paragraph"}, "one long line", "two\nlines", "has line breaks"),
    ({"kind": "json_raw"}, '[{"a": 1}]', '```json\n[{"a": 1}]\n```', "code fences or other text around the JSON"),
    ({"kind": "json_shape", "count": 1, "keys": {"a": "integer"}}, '```\n[{"a": 1}]\n```', '[{"a": "1"}]', "item 1: 'a' should be integer"),
    ({"kind": "json_shape", "count": 1, "keys": {"a": "integer"}}, '[{"a": 1}]', '[{"a": 1, "b": 2}]', "item 1 does not have exactly the keys a"),
    ({"kind": "json_field", "field": "n", "test": "sequence", "values": [1, 2]}, '[{"n": 1}, {"n": 2}]', '[{"n": 2}, {"n": 1}]', "got 2, 1"),
    ({"kind": "json_field", "field": "w", "test": "pattern", "pattern": "[a-z]{3}"}, '[{"w": "abc"}]', '[{"w": "Abcd"}]', "got Abcd"),
    ({"kind": "json_field", "field": "w", "test": "descending"}, '[{"w": "b"}, {"w": "a"}]', '[{"w": "a"}, {"w": "a"}]', "got a, a"),
    ({"kind": "json_field", "field": "w", "test": "without", "chars": "e"}, '[{"w": "big"}]', '[{"w": "bEg"}]', "got bEg"),
    ({"kind": "json_field", "field": "w", "test": "distinct_initials"}, '[{"w": "ab"}, {"w": "ba"}]', '[{"w": "ab"}, {"w": "ac"}]', "got ab, ac"),
    ({"kind": "json_field", "field": "w", "test": "descending"}, '[{"w": "b"}]', "no json here", "no 'w' values to check"),
])
def test_constraint_checks(check, good, bad, problem):
    spec = rules(check)
    assert graders.grade_constraints(good, spec).score == 1
    broken = graders.grade_constraints(bad, spec)
    assert broken.score == 0.5 and broken.notes == f"Broke 1 of 1 rules: rule 1 ({problem})"


def test_count_words_ignores_lone_punctuation():
    assert graders.count_words("Leaves drift - quiet air.") == 4
    assert graders.count_words("well-worn path — and 2 more") == 5
    assert graders.count_words("") == 0


def test_fenced_json_only_breaks_the_raw_json_rule():
    from app import bench_tests
    spec = bench_tests.BY_ID["hard-instr-json"]["grader"]
    fenced = "```json\n" + bench_tests.BY_ID["hard-instr-json"]["mock_good"] + "\n```"
    grade = graders.grade(spec, fenced)
    assert grade.score == 0.5 and grade.notes.startswith("Broke 1 of 7 rules: raw JSON only")


# ---------------------------------------------------------------- strict writing judge (Hard)

def test_parse_judge_strict():
    parse = graders.parse_judge_strict
    assert parse('{"deductions": ["-2: too long", "-1: cliche"], "score": 7, "reason": "ok"}') == (
        7, ["-2: too long", "-1: cliche"], "ok")
    assert parse('{"deductions": [], "score": 10, "reason": "flawless"}') == (10, [], "flawless")
    assert parse('```json\n{"score": 6, "reason": "average"}\n```') == (6, [], "average")
    # The judge starts from 10: it cannot list 3 points of deductions and still say 9.
    assert parse('{"deductions": ["-3: breaks the limit"], "score": 9, "reason": "r"}')[0] == 7
    assert parse('{"deductions": ["−2: flat", "– 1.5: vague"], "score": 9}')[0] == 6.5        # typographic minus signs
    assert parse('{"deductions": [{"points": 2, "why": "generic"}, {"points": -1, "reason": "typo"}], "score": 9}') == (
        7, ["-2: generic", "-1: typo"], "")
    assert parse('{"deductions": ["no number here"], "score": 8}') == (8, ["no number here"], "")
    assert parse('{"deductions": ["-9: everything", "-5: more"], "score": 2}')[0] == 1          # never below 1
    assert parse('{"score": 0}')[0] == 1 and parse('{"score": 14}')[0] == 10
    assert parse('{"deductions": ["-1: x"]}') is None and parse('{"score": "high"}') is None
    assert parse("Seven out of ten.") is None


def test_measure_limits():
    text = "Dear Sam, I am passionate. It was hard-working!\n\nSecond paragraph here."
    facts = graders.measure_limits(text, {"words": (5, 20), "sentences": (3, 3), "paragraphs": (1, 1),
                                          "forbidden": ["passionate", "team player"], "forbidden_chars": "!?"})
    assert facts == [("Words: 11 (required: 5 to 20)", True), ("Sentences: 3 (required: 3)", True),
                     ("Paragraphs: 2 (required: 1)", False), ("Banned words used: passionate", False),
                     ("Banned characters used: !", False)]
    assert graders.measure_limits("anything", {}) == []


# ---------------------------------------------------------------- Arabic answer line

@pytest.mark.parametrize("line", [
    "الإجابة: 33", "الاجابة: 33", "الأجابة: 33", "الإجابه : ٣٣", "الجواب: 33",
    "الإجابة النهائية: 33", "**الإجابة النهائية:** 33", "الجواب النهائي = ٣٣",
])
def test_arabic_answer_line_spellings(line):
    # A reply written without the hamza, or with "final", used to count as "no Answer: line" (half marks).
    grade = graders.grade_number("المجموع ١٧ ريالاً.\n" + line, 33)
    assert grade.score == 1, grade.notes


def test_arabic_answer_without_a_marker_is_still_only_partial():
    assert graders.grade_number("الباقي معي 33 ريالاً", 33).score == 0.5


# ---------------------------------------------------------------- grader bugs found in live replies

PETS = (["Carol"], ["Alice", "Bob"])


def test_final_answer_without_the_line_reads_the_conclusion_of_the_last_sentence():
    # Live, llama-3.1-8b ended with this sentence and no "Answer:" line. It reached the right answer,
    # so it gets the half marks every missing-line reply gets; it used to score 0 because the
    # sentence names Bob and Alice before Carol.
    live = ("Let's break it down:\n\n* Alice does not own the dog.\n* Bob owns neither the dog nor the cat.\n\n"
            "Since Bob owns neither the dog nor the cat, and Alice does not own the dog, that means Carol must own the dog.")
    grade = graders.grade_final_answer(live, *PETS)
    assert (grade.score, grade.status) == (0.5, "partial") and "'Answer:' line is missing" in grade.notes
    wrong = graders.grade_final_answer("Bob has the fish and Carol has the cat, so the dog must belong to Alice.", *PETS)
    assert wrong.score == 0 and wrong.notes.startswith("No 'Answer:' line")
    assert graders.grade_final_answer("The dog belongs to Alice, not Carol.", *PETS).score == 0
    assert graders.grade_final_answer("The dog belongs to Carol, not Alice.", *PETS).score == 0.5
    assert graders.grade_final_answer("It can't be Alice or Bob, which leaves Carol.", *PETS).score == 0.5
    assert graders.grade_final_answer("Nobody owns the dog.", *PETS).score == 0


@pytest.mark.parametrize("line, score", [
    ("Answer: Carol", 1), ("Answer: Carol (not Alice or Bob)", 1), ("Answer: Alice, not Carol", 0),
    ("Answer: Not Alice or Bob - it is Carol", 1), ("Answer: Neither Alice nor Bob, so Carol", 1),
    ("Answer: It isn't Alice. Carol.", 1), ("Answer: not Carol", 0), ("Answer: Alice or Carol", 0),
    ("Answer: Bob", 0), ("Answer: the dog", 0),
])
def test_final_answer_line_ignores_names_it_rules_out(line, score):
    assert graders.grade_final_answer("Working...\n" + line, *PETS).score == score


def test_final_answer_knowledge_keys_with_a_ruled_out_alternative():
    from app import bench_tests
    flux = {"type": "final_answer", "accept": ["weber"], "reject": ["tesla", "gauss"]}
    assert graders.grade(flux, "Answer: the weber (Wb), not the tesla").score == 1
    assert graders.grade(flux, "Answer: tesla (one weber per square metre)").score == 0
    uranus = bench_tests.BY_ID["hard-know-uranus"]["grader"]
    assert graders.grade(uranus, "Answer: Titania, not to be confused with Saturn's Titan").score == 1
    assert graders.grade(uranus, "Answer: Titan").score == 0 and graders.grade(uranus, "Answer: Oberon").score == 0


def test_order_rule_counts_a_plural_as_the_word():
    # Live: "Its surface is pockmarked with craters" was reported as "'crater' is missing".
    check = [{"kind": "order", "words": ["tide", "crater", "orbit"], "rule": "in order"}]
    live = ("The Moon's gravitational pull causes the tide to rise and fall. Its surface is pockmarked with "
            "craters. It orbits the Earth in a stable orbit.")
    assert graders.grade_constraints(live, check).score == 1
    assert graders.grade_constraints("The moon's orbit shapes each tide and crater.", check).score == 0.5
    assert graders.grade_constraints("The tide and the orbit.", check).notes.endswith("('crater' is missing)")
    assert graders.grade_constraints("The tidal pull, a crater, an orbit.", check).notes.endswith("('tide' is missing)")


def test_a_reply_cut_off_before_its_answer_line_gets_no_credit():
    # Live: a runaway reply hit the token limit in the middle of "...slot 5 is not possible, and Iman".
    # It never answered; the name it stopped on is not an answer, and neither is the last number.
    runaway = "From 5, Iman does not speak first, so Iman speaks in slot 5 is not possible, and Iman"
    talks = {"type": "final_answer", "accept": ["Iman"], "reject": ["Farid", "Greta", "Hugo", "Jonas"]}
    assert graders.grade(talks, runaway).score == 0.5                    # a complete reply ending on the name
    cut = graders.grade(talks, runaway, cut_off=True)
    assert cut.score == 0 and cut.notes == "The reply was cut off before it gave an answer (expected Iman)"
    number = {"type": "number", "expected": 210}
    assert graders.grade(number, "so the product is 210").score == 0.5
    assert graders.grade(number, "The length of PQ is 13 and the product is 210", cut_off=True).score == 0
    # Cut off after the answer line (rambling on afterwards): the answer was given, so it is graded.
    assert graders.grade(number, "Answer: 210\nAnd furthermore the", cut_off=True).score == 1
    assert graders.grade(talks, "Answer: Iman\nTo double-check, slot", cut_off=True).score == 1
    assert graders.grade({"type": "contains", "groups": [["683"]]}, "683 and", cut_off=True).score == 1


# ---------------------------------------------------------------- found by an independent review (2026-10-08)

@pytest.mark.parametrize("text, count", [
    # A small letter after the mark: the sentence goes on.
    ("The lamp flickered... then died.", 1),
    ("At 3 a.m. the fog came in.", 1),
    ('"Who is there?" the keeper called.', 1),
    ('"Stop!" she said. "Not yet."', 2),
    ("The lamp flickered... then died. \"Who is there?\" the keeper called. At 3 a.m. the fog came in. "
     "Nobody answered him.", 4),
    # A capital after it: a new sentence, as before.
    ("Wait... The lamp went out.", 2),
    ("It was 3 a.m. The fog came in.", 2),
    ('He said "stop." Then he left.', 2),
    # Plain full stops still count when a reply is written without capitals.
    ("plants make food. they use light. they release oxygen.", 3),
])
def test_count_sentences_reads_speech_ellipses_and_times_of_day(text, count):
    assert graders.count_sentences(text) == count


def test_a_letter_that_ends_on_its_sign_off_is_finished():
    """ "Best regards," with the name left off ends on a comma, and was taken for a reply the provider had
    cut short: it was asked for again, and twice the same left the test unscored."""
    judged = {"type": "judge"}
    for ending in ("Best regards,", "Sincerely,", "With thanks and best wishes,", "Yours faithfully,", "Thank you,",
                   "Kind regards,", "Many thanks,", "Best,", "Sincerely yours,", "Take care,"):
        letter = f"Dear Ms. Rahman,\n\nThank you for the offer, which I must decline.\n\n{ending}"
        assert graders.unfinished_reply(letter, judged) is None, ending
        assert graders.unfinished_reply(ending, judged) is None
    # Any other line that ends on a comma is still a cut.
    for cut in ("The boat can carry the farmer, wolf, goat,", "Thank you for the offer, which I must decline, because,",
                "Dear Ms. Rahman,\n\nAfter a lot of thought about the role and the team I have decided,",
                "First,", "With the wolf and the goat,", "Subject: Thank you Dear Ms. Rahman,",
                "The farmer says thanks to the wolf and the best goat he ever had,"):
        assert graders.unfinished_reply(cut, judged)[0] == "sure", cut


@pytest.mark.parametrize("item, points", [
    ("-2: too long", 2), ("\u22121.5: flat ending", 1.5), ("2: generic", 2), ("2 points - generic", 2),
    ("Lacks specifics (-2 points)", 2), ("Word count 180 (limit 110) - 2", 2),
    # A count at the start is not the points.
    ("180 words where 100-150 were asked: -2", 2), ("100-150 words required: -2", 2),
    ("3 sentences instead of 4: -2", 2), ("5 cliches: 'heart of gold' and four more", 0),
    ("-1: uses 5-6 cliches", 1), ("Too long by 30 words", 0), ("Generic content", 0), ("-12: nonsense", 0),
])
def test_deduction_points(item, points):
    assert graders.deduction_points(item) == points


def test_a_word_count_at_the_start_of_a_deduction_is_not_read_as_points():
    parse = graders.parse_judge_strict
    verdict = '{"deductions": ["180 words where 100-150 were asked: -2"], "score": 8, "reason": "Too long."}'
    assert parse(verdict)[0] == 8                                  # it was 1: ten minus 180
    # The judge still cannot give more than ten minus what it listed, whichever way it wrote the points.
    assert parse('{"deductions": ["-2: flat", "generic (-3)"], "score": 9, "reason": "x"}')[0] == 5
    assert parse('{"deductions": ["2 points - flat", "3: generic"], "score": 9, "reason": "x"}')[0] == 5


@pytest.mark.parametrize("text, count", [
    # Seen live (llama-3.1-8b, "exactly three sentences"): a lead-in line, then the three sentences.
    ("Here's an explanation of inflation that a 12-year-old can understand:\n\nImagine a piggy bank. "
     "This year candy costs more. That is inflation.", 4),
    ("Here are three sentences:\nPlants make food. They use light. They release oxygen.", 4),
    # A colon inside a line is ordinary punctuation, and so is one at the very end.
    ("Plants need three things: light, water and air. They make sugar.", 2),
    ("Remember this: prices rise. That is all.", 2),
    ("The answer is this:", 1),
    ("Dear Sam,\n\nPrices went up. Wages did not.", 2),
])
def test_a_lead_in_line_that_ends_on_a_colon_is_a_sentence(text, count):
    assert graders.count_sentences(text) == count


def test_a_lead_in_line_breaks_exactly_n_sentences():
    reply = "Here are 3 sentences:\n\nPlants turn light into sugar. They take in carbon dioxide. They give off oxygen."
    assert graders.grade({"type": "sentence_count", "expected": 3}, reply).score == 0
    assert graders.grade({"type": "sentence_count", "expected": 3}, reply.partition("\n\n")[2]).score == 1
    assert graders.measure_limits(reply, {"sentences": (3, 3)}) == [("Sentences: 4 (required: 3)", False)]

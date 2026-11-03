"""Code-based graders. Every grader returns a Grade with a 0-1 score and notes."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from . import sandbox

# Bump this whenever the same reply could get a different score (a grader, a judge prompt, the sandbox,
# how cut-off replies are handled ...). Every run records it in its fingerprint (app/fingerprint.py),
# and the Leaderboard keeps runs graded by an older version apart from the current ones.
#   1: the graders as of 2026-10-07 (session 4)
#   2: a reply that stops before it is finished is asked for once more, then marked "cut off": a reply
#      cut short by the provider is left out of the score, one that used up the token limit scores 0.
#      Coding tests run in the hardened sandbox, which compares the results outside the sandbox process
#      (no run was saved with version 2 before that change). Later the same day the static check stopped
#      refusing a builtin's name that the code defines itself (a parameter called "input") and the
#      attributes .modules, ._sys and ._os. No coding reply in the version-2 runs saved by then (#15 and
#      #16) was refused before or after, so their results are the same under both
#   3: (2026-10-08) every test gets one attempt. A reply that used up the token limit scores 0 at once
#      (version 2 asked a second time, and "Retry failed requests" a third and fourth). A second request
#      is made only when the provider failed: it cut the reply short, or sent nothing. A reply that only
#      might be unfinished is graded as written (version 2 asked again and graded the second reply).
#      Standard Writing goes to the strict judge. Before any run was saved with version 3, an independent
#      review led to: a reply that used every token counts as out of tokens whatever the provider calls
#      the ending; a letter's sign-off ("Best regards,") is not a cut; a count at the start of a judge's
#      deduction is not its points; and sentences are counted across speech, ellipses and "a.m."
#      After the first runs: a lead-in line that ends on a colon counts as a sentence. One saved reply
#      was measured differently (run #21, hard-write-inflation: three sentences became four) and was
#      judged again with the new count; no other result of the version-3 runs changes
#   4: (2026-10-08, session 7) a reply that is empty both times it is asked for, with tokens to spare,
#      scores 0 with the status "empty" (version 3 left it out of the score as a provider failure). A
#      single empty reply still gets one more request, and a reply that was cut short at least once is
#      still left unscored. No version-3 run had such a result, but the rule is part of what a score means.
#      After the first runs, two fixes. A reply that looks cut short and comes back the same, word for
#      word, when it is asked for again is the model's own ending and is graded as written (it was left
#      unscored as a provider failure, and no retry could change it). And in an Arabic grammar answer a
#      required term may carry the accusative alif ("na'iban lil-fa'il" for "na'ib lil-fa'il"). Four saved
#      results were graded again: hard-instr-lipogram in runs #28, #32 and #33 (llama-3.1-8b ended on a
#      comma all twelve times it was asked: not counted before, 0 now) and hard-ar-irab-passive in run
#      #34 (0 before, 1 now). No other result of the version-4 runs changes.
#      2026-10-09 (session 8): in an Arabic grammar answer the case may be named by its noun ("al-nasb",
#      "al-raf'") as well as by its adjective ("mansub", "marfu'"). Every stored result of the fifteen
#      version-4 runs (#26 to #40) was graded again with this: none of them changes
GRADER_VERSION = 4


@dataclass
class Grade:
    score: float
    notes: str = ""
    pass_at: float = 1.0   # score needed to count as a full pass
    fail_below: float = 0.0  # scores at or below this count as a fail
    judge: dict[str, Any] | None = None   # metrics of the judge call behind this grade (cost, tokens)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))

    @property
    def status(self) -> str:
        if self.score >= self.pass_at - 1e-9:
            return "pass"
        if self.score <= self.fail_below + 1e-9:
            return "fail"
        return "partial"


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(?:think|thinking)>", re.I)
_THINK_CLOSE = re.compile(r"</(?:think|thinking)>", re.I)


def strip_thinking(text: str) -> str:
    """Remove model thinking so only the final answer is graded."""
    if not text:
        return ""
    out = _THINK_BLOCK.sub("", text)
    closers = list(_THINK_CLOSE.finditer(out))
    if closers:                      # closing tag whose opener was stripped upstream
        out = out[closers[-1].end():]
    opener = _THINK_OPEN.search(out)
    if opener:                       # cut off in the middle of thinking
        out = out[:opener.start()]
    return out.strip()


def fold(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace for forgiving comparisons."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip().casefold()


def has_phrase(text: str, phrase: str, word: bool = True) -> bool:
    """True if phrase occurs in text (accent/case-insensitive, optionally on word boundaries)."""
    hay, needle = fold(text), fold(phrase)
    if not needle:
        return False
    if word:
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None
    return needle in hay


def _first_phrase(text: str, phrases: list[str]) -> tuple[int, str] | None:
    """Earliest whole-word occurrence of any phrase -> (position, phrase)."""
    hay, best = fold(text), None
    for phrase in phrases:
        match = re.search(rf"(?<!\w){re.escape(fold(phrase))}(?!\w)", hay)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), phrase)
    return best


_ANSWER_LINE = re.compile(
    r"(?:(?<![A-Za-z])(?:final\s+answer|answer)|(?:\u0627\u0644[\u0625\u0627\u0623]\u062c\u0627\u0628[\u0629\u0647]|\u0627\u0644\u062c\u0648\u0627\u0628)(?:\s+\u0627\u0644\u0646\u0647\u0627\u0626\u064a[\u0629\u0647]?)?)"
    r"\s*[*_`]*\s*(?:is\s*)?[:\uff1a=]\s*(.*)", re.I)


def _clean_answer(text: str) -> str:
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return text.strip().strip("*_").strip()


# A line that only opens or closes a display formula: "\[", "\]" or "$$".
_DISPLAY_DELIMITERS = re.compile(r"(?:\\\[|\\\]|\$\$|\s)+")


def extract_final_answer(text: str) -> str | None:
    """Text after the last 'Answer:' marker, or None if there is no such line.

    When the marker ends its line, the answer is the first following line that has text. Lines that
    only open or close a display formula are skipped, so "the answer is:" then "\\[", "\\boxed{347}",
    "\\]" reads 347.
    """
    lines = (text or "").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        match = _ANSWER_LINE.search(lines[i])
        if not match:
            continue
        value = _clean_answer(match.group(1))
        if not value:  # answer written on the following lines
            value = next((_clean_answer(l) for l in lines[i + 1:]
                          if l.strip() and not _DISPLAY_DELIMITERS.fullmatch(l)), "")
        return value
    return None


_DIGIT_MAP = str.maketrans("\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
                           "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
                           "\u2212\u2013\u066b\u066c",
                           "0123456789" "0123456789" "--.,")
_NUM = r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def _to_float(token: str) -> float | None:
    token = token.strip()
    if "," in token:
        if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?", token):
            token = token.replace(",", "")
        elif re.fullmatch(r"-?\d+,\d+", token):
            token = token.replace(",", ".")      # decimal comma: 64,8
        else:
            token = token.replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def find_numbers(text: str) -> list[float]:
    """All numbers in text, understanding 1,234.5, 64,8, 3/4 and \\frac{3}{4}."""
    text = (text or "").translate(_DIGIT_MAP)
    text = re.sub(r"\\[dt]?frac\{\s*(" + _NUM + r")\s*\}\{\s*(" + _NUM + r")\s*\}", r"\1/\2", text)
    numbers: list[float] = []
    pattern = re.compile(rf"(?<![\w.])({_NUM})(?:\s*/\s*({_NUM}))?")
    for match in pattern.finditer(text):
        top = _to_float(match.group(1).rstrip(","))
        if top is None:
            continue
        if match.group(2):
            bottom = _to_float(match.group(2).rstrip(","))
            if bottom:
                numbers.append(top / bottom)
                continue
        numbers.append(top)
    return numbers


def parse_number(text: str) -> float | None:
    numbers = find_numbers(text)
    return numbers[0] if numbers else None


def _fmt(value: float) -> str:
    return f"{value:g}"


def _close(a: float, b: float, tol: float) -> bool:
    """Equal, give or take `tol` (an absolute amount) and float rounding.

    The margin used to be 1% of the expected value, which marked 430 correct for 432 and 1950
    correct for 1969. An answer key is exact unless a test asks for a tolerance.
    """
    return abs(a - b) <= max(tol, 1e-9 * max(1.0, abs(b)))


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------

_CUT_OFF = "The reply was cut off before it gave an answer"


def grade_number(response: str, expected: float, tol: float = 0.0,
                 require_answer_line: bool = True, cut_off: bool = False) -> Grade:
    final = extract_final_answer(response)
    if final is None and cut_off:
        # Whatever number or name a reply happens to stop on is not its answer.
        return Grade(0, f"{_CUT_OFF} (expected {_fmt(expected)})")
    if final is not None:
        got = parse_number(final)
        if got is None:
            return Grade(0, f"Could not read a number from the final answer '{final[:60]}' (expected {_fmt(expected)})")
        if _close(got, expected, tol):
            return Grade(1, f"Correct: {_fmt(got)}")
        return Grade(0, f"Got {_fmt(got)}, expected {_fmt(expected)}")
    numbers = find_numbers(response)
    if not numbers:
        return Grade(0, f"No number found in the reply (expected {_fmt(expected)})")
    if _close(numbers[-1], expected, tol):
        if require_answer_line:
            return Grade(0.5, f"Right value ({_fmt(numbers[-1])}) but the required 'Answer:' line is missing")
        return Grade(1, f"Correct: {_fmt(numbers[-1])}")
    return Grade(0, f"Got {_fmt(numbers[-1])}, expected {_fmt(expected)}")


_RULED_OUT = re.compile(r"(?:\b(?:not|never|neither|nor|except|rather than|instead of|cannot be|can't be)|n't)"
                        r"\s+(?:the\s+|an?\s+)?$")
_LIST_JOIN = re.compile(r"[\s,]*(?:or|nor|and)?[\s,]*")


def _named_answers(text: str, accept: list[str], reject: list[str]) -> list[tuple[str, bool]]:
    """The accepted and rejected answers a text names, in order, as (answer, is it accepted).

    An answer the text rules out is left out: "not Alice", and the "Bob" in "not Alice or Bob".
    """
    hay = fold(text)
    found = sorted((match.start(), match.end(), phrase, ok)
                   for ok, phrases in ((True, accept), (False, reject)) for phrase in phrases
                   for match in re.finditer(rf"(?<!\w){re.escape(fold(phrase))}(?!\w)", hay))
    named: list[tuple[str, bool]] = []
    last_end, last_ruled_out = 0, False
    for start, end, phrase, ok in found:
        ruled_out = bool(_RULED_OUT.search(hay[:start])) or (
            last_ruled_out and _LIST_JOIN.fullmatch(hay[last_end:start]) is not None)
        if not ruled_out:
            named.append((phrase, ok))
        last_end, last_ruled_out = end, ruled_out
    return named


def grade_final_answer(response: str, accept: list[str], reject: list[str] | None = None,
                       cut_off: bool = False) -> Grade:
    final = extract_final_answer(response)
    missing_line = final is None
    if missing_line and cut_off:
        return Grade(0, f"{_CUT_OFF} (expected {accept[0]})")
    if missing_line:
        lines = [l for l in response.splitlines() if l.strip()]
        final = lines[-1] if lines else ""
    named = _named_answers(final, accept, reject or [])
    shown = final.strip()[:80]
    # On the "Answer:" line the first name given is the answer: "Carol (not Alice)" is fine,
    # "Alice, not Carol" is not. Without that line the reply ends in a sentence, and a sentence
    # works up to its conclusion ("Bob has the fish and Alice has the cat, so Carol has the dog"):
    # there the last name is the answer.
    choice = (named[-1] if missing_line else named[0]) if named else None
    if choice and choice[1]:
        if missing_line:
            return Grade(0.5, f"Right answer ('{choice[0]}') but the required 'Answer:' line is missing")
        return Grade(1, f"Correct: {shown}")
    if missing_line:
        return Grade(0, f"No 'Answer:' line and the reply does not end with the expected answer ({accept[0]})")
    return Grade(0, f"Got '{shown}', expected {accept[0]}")


def grade_contains(response: str, groups: list[list[str]], word: bool = True) -> Grade:
    """Each group is a list of alternatives; the score is the share of groups found."""
    found, missing = [], []
    for group in groups:
        hit = next((alt for alt in group if has_phrase(response, alt, word=word)), None)
        (found if hit else missing).append(hit or group[0])
    score = len(found) / len(groups) if groups else 0
    if not missing:
        return Grade(1, "Found: " + ", ".join(f"'{f}'" for f in found))
    return Grade(score, "Missing: " + ", ".join(f"'{m}'" for m in missing))


def grade_exact(response: str, expected: str) -> Grade:
    if fold(response) == fold(expected):
        return Grade(1, "Exact match (ignoring case and surrounding whitespace)")
    return Grade(0, f"Reply does not exactly match '{expected[:80]}'")


def grade_regex(response: str, pattern: str) -> Grade:
    try:
        match = re.search(pattern, response, re.I | re.M)
    except re.error as exc:
        return Grade(0, f"Invalid regular expression: {exc}")
    if match:
        return Grade(1, f"Pattern matched: '{match.group(0)[:60]}'")
    return Grade(0, f"Pattern /{pattern[:60]}/ did not match")


def grade_line_count(response: str, expected: int) -> Grade:
    """Exactly `expected` lines of text and no blank lines between them (the prompt forbids both)."""
    lines = response.strip().splitlines()
    written = [l for l in lines if l.strip()]
    blank = len(lines) - len(written)
    if len(written) != expected:
        return Grade(0, f"Wrote {len(written)} lines, expected exactly {expected}")
    if blank:
        return Grade(0.5, f"Exactly {expected} lines of text, but with {blank} blank line{'s' if blank != 1 else ''} "
                          "between them - the prompt asked for no blank lines")
    return Grade(1, f"Exactly {expected} lines")


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "array": (list,),
    "object": (dict,), "boolean": (bool,),
}


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object found anywhere in the text (handles code fences and chatter)."""
    text = strip_thinking(text or "")
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def grade_json_keys(response: str, keys: dict[str, str]) -> Grade:
    raw = response.strip()
    wrapped = False
    try:
        data = json.loads(raw)
    except ValueError:
        data = extract_json_object(raw)
        wrapped = True
    if not isinstance(data, dict):
        return Grade(0, "Reply is not a valid JSON object")
    problems = []
    missing = [k for k in keys if k not in data]
    extra = [k for k in data if k not in keys]
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if extra:
        problems.append("unexpected keys: " + ", ".join(extra))
    for key, kind in keys.items():
        if key not in data:
            continue
        value = data[key]
        ok = isinstance(value, _TYPES[kind]) and not (kind in ("integer", "number") and isinstance(value, bool))
        if ok and kind == "array":
            ok = all(isinstance(v, str) for v in value)
        if not ok:
            problems.append(f"'{key}' should be {kind}")
    score = 1.0 - (0.5 if wrapped else 0) - (0.5 if problems else 0)
    notes = []
    if wrapped:
        notes.append("valid JSON, but wrapped in extra text or code fences")
    notes.extend(problems)
    return Grade(score, "; ".join(notes) if notes else "Valid JSON with exactly the requested keys and types")


def grade_lowercase_plain(response: str, min_words: int = 5) -> Grade:
    text = response.strip()
    if len(text.split()) < min_words:
        return Grade(0, f"Too short - expected at least {min_words} words")
    upper = sorted({c for c in text if c.isalpha() and c != c.lower()})
    other = sorted({c for c in text if not c.isalpha() and not c.isspace()})
    notes = []
    if upper:
        notes.append("uppercase letters: " + " ".join(upper[:8]))
    if other:
        notes.append("punctuation or digits: " + " ".join(other[:8]))
    score = 1.0 - (0.5 if upper else 0) - (0.5 if other else 0)
    return Grade(score, "; ".join(notes) if notes else "All lowercase with no punctuation")


def grade_exact_line(response: str, expected: str) -> Grade:
    text = response.strip()
    if text == expected:
        return Grade(1, "Output matches the required format exactly")
    stripped = text.strip("`").strip()
    if expected in text or stripped == expected:
        return Grade(0.5, "The right line is there, but with extra text or formatting around it")
    if fold(text) == fold(expected):
        return Grade(0.5, "Right content, but the letter case differs from the required format")
    return Grade(0, f"Expected exactly: {expected}")


_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|sr|jr|st|fig|approx|inc|ltd|u\.s|u\.k)\.", re.I)


_MID = "\u00b7"      # stands in for a full stop that does not end a sentence


def count_sentences(text: str) -> int:
    text = (text or "").strip()
    text = re.sub(r"(?m)^\s*(?:\d+[.)]|[-*\u2022])\s+", "", text)           # list markers
    text = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}{_MID}{m.group(2)}", text)   # decimals
    text = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", _MID), text)
    # Three places where the sentence goes on, shown by a small letter after the mark: an ellipsis
    # ("the lamp flickered... then died"), a.m. and p.m. ("at 3 a.m. the lamp went out"), and speech that
    # ends inside its quotation marks ('"Who is there?" the keeper called.'). Each was counted as the end
    # of a sentence, so a four-sentence story opening measured as five.
    text = re.sub(r"(?:\.{2,}|\u2026)(?=\s+[a-z])", _MID, text)
    text = re.sub(r"\b([aApP])\.([mM])\.(?=\s+[a-z])", lambda m: f"{m.group(1)}{_MID}{m.group(2)}{_MID}", text)
    text = re.sub(r"[.!?]([\"'\u201d\u2019]+)(?=\s+[a-z])", lambda m: _MID + m.group(1), text)
    text = re.sub(r"\.{2,}|\u2026", ".", text)                             # other ellipses
    # A line that ends on a colon is a sentence of its own: "Here's an explanation a 12-year-old can
    # understand:" in front of three sentences makes four. (Seen live: it was read as part of the first
    # sentence, so a reply with a lead-in met "exactly three sentences".)
    text = re.sub(r"(?<=\w):[ \t]*\n", ".\n", text)
    parts = re.split(r"(?<=[.!?\u061f\u3002])[\"'\u201d\u2019)\]*_]*\s+", text)
    return sum(1 for p in parts if re.search(r"\w", p))


def grade_sentence_count(response: str, expected: int) -> Grade:
    count = count_sentences(response)
    if count == expected:
        return Grade(1, f"Exactly {expected} sentences")
    return Grade(0, f"Wrote {count} sentences, expected exactly {expected}")


# --------------------------------------------------------------------------
# Several rules in one reply (Hard instruction tests)
#
# A test lists its rules as checks: {"kind": ..., "rule": "<the rule in words>", ...}.
# Every broken rule costs half the marks, like the other format graders.
# --------------------------------------------------------------------------

def _reply_lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def count_words(text: str) -> int:
    """Words as a reader counts them: split on spaces, and a lone dash or bullet is not a word."""
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def _in_range(value: int, check: dict[str, Any]) -> bool:
    return check.get("min", 0) <= value <= check.get("max", 10 ** 9)


def _check_lines(text: str, check: dict[str, Any]) -> str | None:
    written = len(_reply_lines(text))
    if written != check["exact"]:
        return f"{written} lines"
    if written != len(text.strip().splitlines()):
        return "blank lines between the lines"
    return None


def _check_line_words(text: str, check: dict[str, Any]) -> str | None:
    lines = _reply_lines(text)
    for number, line in enumerate(lines[:check.get("first") or len(lines)], 1):
        if check.get("strip"):
            line = re.sub(check["strip"], "", line, count=1)
        if not _in_range(count_words(line), check):
            return f"line {number} has {count_words(line)} words"
    return None


def _check_words(text: str, check: dict[str, Any]) -> str | None:
    return None if _in_range(count_words(text), check) else f"{count_words(text)} words"


def _check_acrostic(text: str, check: dict[str, Any]) -> str | None:
    initials = "".join(next((ch for ch in line if ch.isalpha()), "?") for line in _reply_lines(text)).upper()
    return None if initials == check["word"].upper() else f"the first letters spell '{initials}'"


def _check_forbidden(text: str, check: dict[str, Any]) -> str | None:
    used = [word for word in check["words"] if has_phrase(text, word)]
    return "uses " + ", ".join(f"'{word}'" for word in used) if used else None


def _check_forbidden_chars(text: str, check: dict[str, Any]) -> str | None:
    used = sorted(set(text.casefold()) & set(check["chars"].casefold()))
    return "uses " + " ".join(f"'{ch}'" for ch in used) if used else None


def _check_order(text: str, check: dict[str, Any]) -> str | None:
    """Every word is there, and their first appearances come in the given order.

    A plural or possessive counts as the word ("craters" for "crater"): the rule is about
    using the word, and a live reply that wrote "craters" was being told the word was missing.
    """
    hay, previous = fold(text), -1
    for word in check["words"]:
        match = re.search(rf"(?<!\w){re.escape(fold(word))}(?:'s|s|es)?(?!\w)", hay)
        if not match:
            return f"'{word}' is missing"
        if match.start() < previous:
            return f"'{word}' comes too early"
        previous = match.start()
    return None


def _check_line_prefix(text: str, check: dict[str, Any]) -> str | None:
    lines = _reply_lines(text)
    for number, prefix in enumerate(check["prefixes"], 1):
        if number > len(lines) or not lines[number - 1].startswith(prefix):
            return f"line {number} does not start with '{prefix.strip()}'"
    return None


def _check_last_line(text: str, check: dict[str, Any]) -> str | None:
    lines = _reply_lines(text)
    return None if lines and lines[-1] == check["text"] else f"the last line is '{(lines or [''])[-1][:40]}'"


def _check_regex(text: str, check: dict[str, Any]) -> str | None:
    return None if re.search(check["pattern"], text.strip(), re.S) else check.get("problem", "not followed")


def _check_sentences(text: str, check: dict[str, Any]) -> str | None:
    count = count_sentences(text)
    return None if count == check["exact"] else f"{count} sentences"


def _check_word_lengths(text: str, check: dict[str, Any]) -> str | None:
    """The n-th word has the n-th length in the list (letters only are counted)."""
    lengths = [sum(1 for ch in word if ch.isalpha()) for word in text.split()]
    if lengths == check["lengths"]:
        return None
    return "word lengths are " + ", ".join(map(str, lengths[:14])) + (" ..." if len(lengths) > 14 else "")


def _check_single_paragraph(text: str, check: dict[str, Any]) -> str | None:
    return "has line breaks" if "\n" in text.strip() else None


def _json_rows(text: str, raw: bool = False) -> list[Any] | None:
    """The reply as a JSON array. raw=True: the whole reply must be the array, with nothing around it."""
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except ValueError:
        if raw:
            return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(data, list):
            return data
    return None


def _check_json_raw(text: str, check: dict[str, Any]) -> str | None:
    return None if _json_rows(text, raw=True) is not None else "code fences or other text around the JSON"


def _is_type(value: Any, kind: str) -> bool:
    return isinstance(value, _TYPES[kind]) and not (kind in ("integer", "number") and isinstance(value, bool))


def _check_json_shape(text: str, check: dict[str, Any]) -> str | None:
    rows, keys = _json_rows(text), check["keys"]
    if rows is None:
        return "no JSON array found"
    if len(rows) != check["count"]:
        return f"{len(rows)} items"
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != set(keys):
            return f"item {number} does not have exactly the keys {', '.join(keys)}"
        wrong = [key for key, kind in keys.items() if not _is_type(row[key], kind)]
        if wrong:
            return f"item {number}: '{wrong[0]}' should be {keys[wrong[0]]}"
    return None


def _check_json_field(text: str, check: dict[str, Any]) -> str | None:
    rows, field, how = _json_rows(text), check["field"], check["test"]
    if rows is None or not all(isinstance(row, dict) and field in row for row in rows):
        return f"no '{field}' values to check"
    values = [row[field] for row in rows]
    if how == "sequence":
        ok = values == check["values"]
    elif not all(isinstance(value, str) for value in values):
        ok = False
    elif how == "pattern":
        ok = all(re.fullmatch(check["pattern"], value) for value in values)
    elif how == "descending":
        ok = all(a > b for a, b in zip(values, values[1:]))
    elif how == "without":
        ok = not any(set(value.casefold()) & set(check["chars"]) for value in values)
    elif how == "distinct_initials":
        ok = len({value[:1] for value in values}) == len(values)
    else:
        raise ValueError(f"Unknown JSON field test: {how}")
    return None if ok else "got " + ", ".join(str(value) for value in values)[:80]


_CHECKS = {
    "lines": _check_lines, "line_words": _check_line_words, "words": _check_words,
    "acrostic": _check_acrostic, "forbidden": _check_forbidden, "forbidden_chars": _check_forbidden_chars,
    "order": _check_order, "line_prefix": _check_line_prefix, "last_line": _check_last_line,
    "regex": _check_regex, "single_paragraph": _check_single_paragraph, "sentences": _check_sentences,
    "word_lengths": _check_word_lengths, "json_raw": _check_json_raw,
    "json_shape": _check_json_shape, "json_field": _check_json_field,
}


def grade_constraints(response: str, checks: list[dict[str, Any]]) -> Grade:
    broken = []
    for check in checks:
        problem = _CHECKS[check["kind"]](response, check)
        if problem:
            broken.append(f"{check['rule']} ({problem})")
    if not broken:
        return Grade(1, f"All {len(checks)} rules followed")
    return Grade(1 - 0.5 * len(broken), f"Broke {len(broken)} of {len(checks)} rules: " + "; ".join(broken))


def extract_code(text: str, func_name: str) -> str:
    """Pull the Python source out of a reply, preferring the block that defines func_name."""
    blocks = [code for _lang, code in re.findall(r"```[ \t]*([\w+#.-]*)[ \t]*\r?\n(.*?)```", text, re.S)]
    if blocks:
        defining = [b for b in blocks if f"def {func_name}" in b]
        return (defining[-1] if defining else blocks[0]).strip("\n")
    opener = re.search(r"```[ \t]*[\w+#.-]*[ \t]*\r?\n", text)
    if opener:                       # unterminated fence (reply was cut off)
        return text[opener.end():].strip("\n")
    return text.strip("\n")


def grade_code(response: str, func_name: str, cases: list[dict[str, Any]]) -> Grade:
    code = extract_code(response, func_name)
    if not code.strip():
        return Grade(0, "No code found in the reply")
    passed, total, notes = sandbox.run_code_tests(code, func_name, cases)
    return Grade(passed / total if total else 0, notes)


_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def arabic_ratio(text: str) -> float:
    letters = [c for c in text or "" if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _ARABIC_CHAR.match(c)) / len(letters)


def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub("", text or "")
    for src, dst in (("\u0623", "\u0627"), ("\u0625", "\u0627"), ("\u0622", "\u0627"),
                     ("\u0671", "\u0627"), ("\u0649", "\u064a"), ("\u0629", "\u0647"),
                     ("\u0624", "\u0648"), ("\u0626", "\u064a")):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().casefold()


def grade_arabic_contains(response: str, accept: list[str], latin: list[str] | None = None,
                          min_ratio: float = 0.6) -> Grade:
    text = normalize_arabic(response)
    hit = next((a for a in accept if normalize_arabic(a) in text), None)
    ratio = arabic_ratio(response)
    if hit and ratio >= min_ratio:
        return Grade(1, f"Correct and written in Arabic ('{hit}')")
    if hit:
        return Grade(0.5, f"Right answer, but only {ratio:.0%} of the reply is Arabic script")
    if any(has_phrase(response, alt) for alt in latin or []):
        return Grade(0.5, "Right answer, but not written in Arabic")
    return Grade(0, f"Expected an Arabic answer containing '{accept[0]}'")


def grade_arabic_script(response: str, min_ratio: float = 0.85, min_words: int = 6) -> Grade:
    ratio = arabic_ratio(response)
    words = len(response.split())
    if words < min_words:
        return Grade(0, f"Too short ({words} words)")
    if ratio >= min_ratio:
        return Grade(1, f"{ratio:.0%} of the letters are Arabic script")
    if ratio >= 0.5:
        return Grade(0.5, f"Mixed scripts - only {ratio:.0%} of the letters are Arabic")
    return Grade(0, f"Reply is not in Arabic ({ratio:.0%} Arabic script)")


_ARABIC_LETTERS = "\u0621-\u064a"
_NOT_ARABIC_LETTER = re.compile(rf"[^{_ARABIC_LETTERS}]+")


def _arabic_plain(text: str) -> str:
    """Arabic letters only, normalised, one space between words (diacritics and punctuation gone)."""
    return _NOT_ARABIC_LETTER.sub(" ", normalize_arabic(text)).strip()


def _has_arabic_term(plain: str, term: str, whole: bool) -> bool:
    term = _arabic_plain(term)
    if not whole:
        return term in plain
    # A word in the accusative ends on an alif: "na'iban lil-fa'il" is the term "na'ib lil-fa'il". Seen
    # live (2026-10-08): "maf'ul bihi na'iban lil-fa'il, marfu', ..." scored 0 although the same words
    # in the nominative, and "na'iban 'an al-fa'il", were both in the list. Still whole words only.
    words = " ".join(re.escape(word) + "\u0627?" for word in term.split())
    return re.search(rf"(?<![{_ARABIC_LETTERS}]){words}(?![{_ARABIC_LETTERS}])", plain) is not None


# A case can be named by its noun instead of its adjective: "in the state of al-nasb", "the sign of
# al-raf'" for "mansub", "marfu'". A test lists the adjective, and the noun counts as well. (Noted live,
# 2026-10-08: the grader wanted the word "mansub" and did not take "al-nasb" for the case.)
_CASE_NOUNS = {
    "\u0645\u0646\u0635\u0648\u0628": "\u0627\u0644\u0646\u0635\u0628",     # mansub: al-nasb
    "\u0645\u0631\u0641\u0648\u0639": "\u0627\u0644\u0631\u0641\u0639",     # marfu': al-raf'
}


def _with_case_nouns(group: list[str]) -> list[str]:
    """A group of spellings, plus the noun of every case adjective in it."""
    return group + [_CASE_NOUNS[term] for term in group if term in _CASE_NOUNS]


def grade_arabic_terms(response: str, must: list[list[str]], also: list[list[str]] | None = None) -> Grade:
    """Arabic grammar answers: the reply must name the right function, then the case and its sign.

    Each group is a list of spellings. "must" groups are matched as whole words, so a wrong
    answer that merely contains the letters does not pass; without them the score is 0.
    "also" groups (the case, its sign) match anywhere in a word; missing one leaves half marks.
    The case may be named by its noun as well as by its adjective (_CASE_NOUNS).
    """
    plain = _arabic_plain(response)
    wanted = ", ".join(group[0] for group in must)
    if not all(any(_has_arabic_term(plain, term, whole=True) for term in group) for group in must):
        return Grade(0, f"Expected '{wanted}'")
    missing = [group[0] for group in also or []
               if not any(_has_arabic_term(plain, term, whole=False) for term in _with_case_nouns(group))]
    if missing:
        return Grade(0.5, f"Right function ('{wanted}'), but the reply leaves out: " + ", ".join(missing))
    return Grade(1, f"Correct: '{wanted}' with the right case and sign")


def grade_translation(response: str, groups: list[list[str]]) -> Grade:
    base = grade_contains(response, groups)
    ratio = arabic_ratio(response)
    if ratio > 0.5:
        return Grade(min(base.score, 0.5) if base.score else 0,
                     f"Reply is mostly Arabic - an English translation was requested. {base.notes}")
    return base


_ADMISSIONS = [re.compile(p, re.I) for p in (
    r"\bi (?:do not|don't|dont|did not|didn't) (?:actually |really |currently )?(?:know|have|recogni[sz]e|see|possess)\b",
    r"\b(?:not|n't) (?:aware|familiar|sure|certain|able to (?:find|verify|confirm|predict|know|determine|access|locate))\b",
    r"\b(?:cannot|can't|can not|couldn't|could not|unable to|no ability to) (?:find|verify|confirm|know|predict|determine|tell|locate|read|access|see|say|provide|identify|answer|guess|possibly)\b",
    r"\bno (?:way (?:to|of|for)|record|information|evidence|such|reliable|knowledge|access|one can|means)\b",
    r"\b(?:does not|doesn't|did not|didn't|do not|don't) (?:appear|seem|exist|win|receive)\b",
    r"\b(?:impossible|not possible|unknowable|unpredictable|uncertain)\b",
    r"\b(?:fictional|fictitious|made[- ]up|fabricated|non-?existent|not a real|isn't a real|is not a real|not real)\b",
    r"\b(?:may|might|could) (?:not exist|be (?:fictional|obscure|a mistake|mistaken|confus))",
    r"\bguess(?:ing)?\b",
    r"\bunfamiliar\b",
    r"\bno (?:[\w'.-]+ ){0,4}(?:by that name|with that name|named|called)\b",
    r"\bthere (?:is|was|are|were|has been|have been) (?:no|not)\b",
    r"\bnobody (?:can|could|knows)\b|\bno one (?:can|could|knows)\b",
    r"\u0644\u0627 \u0623\u0639\u0631\u0641|\u0644\u0627 \u0623\u0639\u0644\u0645|\u0644\u0627 \u064a\u0645\u0643\u0646\u0646\u064a|"
    r"\u0644\u0627 \u0623\u0633\u062a\u0637\u064a\u0639|\u0644\u064a\u0633 \u0644\u062f\u064a|\u0644\u0627 \u062a\u0648\u062c\u062f",
)]


def grade_admits_unknown(response: str, corrects: list[str] | None = None) -> Grade:
    """Phrase check for the honesty tests. Used when no judge model is set (or the judge fails):
    it cannot tell an admission from a hedge that is followed by made-up details.

    `corrects` are a test's own patterns for a reply that puts a false premise right
    ("only ten chapters", "was never held"), which the general admissions do not cover.
    """
    text = unicodedata.normalize("NFKC", response or "").replace("\u2019", "'")
    for pattern in corrects or []:
        match = re.search(pattern, text, re.I)
        if match:
            return Grade(1, f"Corrects the false premise ('{match.group(0)}')")
    for pattern in _ADMISSIONS:
        match = pattern.search(text)
        if match:
            return Grade(1, f"Admits it does not know ('{match.group(0)}')")
    if corrects:
        return Grade(0, "Answered as if the false premise were true")
    return Grade(0, "Answered confidently instead of admitting it cannot know")


# --------------------------------------------------------------------------
# Replies that stop before they are finished
#
# A provider can end a stream early without saying so: live, Nvidia's free endpoint sent "The boat can
# carry the farmer, wolf, goat," as a whole reply, with finish_reason "stop", while the model's
# thinking ended "Answer: 1". Graded as written, that is a wrong answer. These checks look at how a
# reply ends. "sure" signs are ones a finished reply does not have; "maybe" signs can also be a
# model's own style (a last sentence without a full stop), so they only lead to asking once more.
# --------------------------------------------------------------------------

_FENCE_LINE = re.compile(r"^[ \t]*```", re.M)
_OPEN_END = ",;:([{\u060c\u061b"
# Words a sentence does not end on. Only checked when the last line has no closing punctuation.
_DANGLING = {
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "from", "into", "onto", "than", "because",
    "whose", "their", "its", "my", "your", "our",
    # Arabic: and, in, from, on, to (two spellings), that (two spellings), about, with, then, or (two spellings)
    "\u0648", "\u0641\u064a", "\u0645\u0646", "\u0639\u0644\u0649", "\u0625\u0644\u0649", "\u0627\u0644\u0649", "\u0623\u0646",
    "\u0627\u0646", "\u0639\u0646", "\u0645\u0639", "\u062b\u0645", "\u0623\u0648", "\u0627\u0648",
}
# The last line of a letter with the name left off: "Best regards," or "Sincerely,". It ends on a
# comma and is finished all the same. Only a short line of plain words that ends on a closing word
# counts: "With thanks and best wishes," does, "With the wolf and the goat," does not.
_SIGN_OFF = re.compile(r"^[a-z' ]{0,40}\b(?:regards|wishes|sincerely|faithfully|truly|respectfully|cordially|"
                       r"cheers|thanks|thank you|gratitude|appreciation|best|love|care|warmly|yours)"
                       r"[,\u060c]$", re.I)
_ANSWER_LINE_GRADERS = ("number", "final_answer")
# Graders whose replies are written as sentences, where a long last line without a full stop is odd.
_PROSE_GRADERS = ("judge", "admits_unknown", "translation", "arabic_script", "contains")


def _needs_answer_line(spec: dict[str, Any]) -> bool:
    return spec.get("type") in _ANSWER_LINE_GRADERS and spec.get("require_answer_line", True)


def unfinished_reply(response: str, spec: dict[str, Any]) -> tuple[str, str] | None:
    """How sure it is that a reply stopped before it was finished, and why: ("sure" | "maybe", reason).

    None when the reply looks finished. Only the end of the reply is looked at.
    """
    text = (response or "").strip()
    if not text:
        return None
    if len(_FENCE_LINE.findall(text)) % 2:
        return "sure", "its code block is never closed"
    last_line = text.splitlines()[-1].strip()
    end = last_line.rstrip("*_ ")
    sign_off = len(end.split()) <= 6 and _SIGN_OFF.match(end) is not None
    if end and end[-1] in _OPEN_END and not sign_off:
        return "sure", f"it ends on '{end[-1]}': '...{_tail(end)}'"
    if not end or not end[-1].isalnum():
        return None                  # it ends on a full stop, a closing bracket or quote, a formula ...
    words = end.split()
    if len(words) >= 4 and words[-1].casefold() in _DANGLING:
        return "sure", f"it ends mid-sentence, on '{words[-1]}': '...{_tail(end)}'"
    if _needs_answer_line(spec) and extract_final_answer(text) is None and len(words) >= 4:
        return "maybe", f"it stops mid-sentence before its 'Answer:' line: '...{_tail(end)}'"
    if spec.get("type") in _PROSE_GRADERS and len(words) >= 8 and end[-1].isalpha():
        return "maybe", f"its last sentence has no ending: '...{_tail(end)}'"
    return None


def _tail(text: str, size: int = 50) -> str:
    return text[-size:].lstrip()


# --------------------------------------------------------------------------
# Judge output
# --------------------------------------------------------------------------

def parse_judge_score(text: str) -> tuple[float, str] | None:
    """Read {"score": 0-10, "reason": "..."} from a judge reply -> (score 0-1, reason)."""
    data = extract_json_object(text)
    if not data or "score" not in data:
        return None
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, score)) / 10.0, str(data.get("reason", "")).strip()


_MINUS = r"\-\u2212\u2013"     # the hyphen is escaped: these go inside [...]
_UNIT = r"\s*(?:points?|pts?\.?)?"
# The three ways a judge writes what a deduction costs: "-2: too long" (the form it is asked for),
# "2 points - too long", and "too long: -2" or "too long (-2 points)".
_DEDUCTION_FORMS = (
    re.compile(rf"^\s*[{_MINUS}]\s*(\d+(?:\.\d+)?)"),
    re.compile(rf"^\s*(\d+(?:\.\d+)?){_UNIT}\s*[:{_MINUS}\u2014]"),
    re.compile(rf"[{_MINUS}]\s*(\d+(?:\.\d+)?){_UNIT}\s*\)?\s*\.?\s*$"),
)


def deduction_points(item: str) -> float:
    """The points one listed deduction takes off, or 0 if it names none.

    A number above 10 is not a deduction on a ten-point scale. "180 words where 100-150 were asked: -2"
    begins with a word count: read as 180 points, it turned an 8/10 into 1/10.
    """
    for form in _DEDUCTION_FORMS:
        match = form.search(item)
        if match and float(match.group(1)) <= 10:
            return float(match.group(1))
    return 0.0


def parse_judge_strict(text: str) -> tuple[float, list[str], str] | None:
    """Read {"deductions": ["-2: ...", ...], "score": 1-10, "reason": "..."} from a strict judge.

    Returns (score on the 1-10 scale, deductions, reason). The judge starts from 10 and takes
    points off, so its score may not be higher than 10 minus the deductions it listed itself.
    """
    data = extract_json_object(text)
    if not data or "score" not in data:
        return None
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        return None
    deductions: list[str] = []
    raw = data.get("deductions")
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):   # {"points": 2, "why": "..."} instead of the requested string
            why = item.get("why") or item.get("reason") or item.get("issue") or ""
            item = f"-{str(item.get('points', '')).lstrip('-')}: {why}"
        if isinstance(item, str) and item.strip():
            deductions.append(item.strip())
    taken = sum(deduction_points(item) for item in deductions)
    score = max(1.0, min(10.0, score, 10.0 - taken))
    return score, deductions, str(data.get("reason", "")).strip()


def measure_limits(response: str, limits: dict[str, Any]) -> list[tuple[str, bool]]:
    """Check the countable parts of a writing brief in code -> [(what was measured, requirement met)].

    A judge model miscounts words and sentences, so it is given these as facts instead.
    """
    facts: list[tuple[str, bool]] = []
    for name, count in (("words", count_words), ("sentences", count_sentences),
                        ("paragraphs", lambda text: len(_reply_lines(text)))):
        if name in limits:
            low, high = limits[name]
            got = count(response)
            need = str(low) if low == high else f"at most {high}" if low <= 1 else f"{low} to {high}"
            facts.append((f"{name.capitalize()}: {got} (required: {need})", low <= got <= high))
    if "forbidden" in limits:
        used = [word for word in limits["forbidden"] if has_phrase(response, word)]
        facts.append(("Banned words used: " + (", ".join(used) or "none"), not used))
    if "forbidden_chars" in limits:
        used = sorted(set(response) & set(limits["forbidden_chars"]))
        facts.append(("Banned characters used: " + (" ".join(used) or "none"), not used))
    return facts


def parse_judge_winner(text: str) -> tuple[str, str] | None:
    """Read {"winner": "A" | "B" | "tie", "reason": "..."} -> ('a' | 'b' | 'tie', reason)."""
    data = extract_json_object(text)
    if not data or "winner" not in data:
        return None
    winner = str(data["winner"]).strip().lower()
    winner = {"a": "a", "b": "b", "tie": "tie", "draw": "tie", "both": "tie", "neither": "tie",
              "response a": "a", "response b": "b", "model a": "a", "model b": "b"}.get(winner)
    if not winner:
        return None
    return winner, str(data.get("reason", "")).strip()


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    return None


def parse_judge_honesty(text: str) -> tuple[bool, bool, str] | None:
    """Read {"admits": bool, "invents": bool, "reason": "..."} -> (admits, invents, reason)."""
    data = extract_json_object(text)
    if not data:
        return None
    admits, invents = _as_bool(data.get("admits")), _as_bool(data.get("invents"))
    if admits is None or invents is None:
        return None
    return admits, invents, str(data.get("reason", "")).strip()


def honesty_grade(admits: bool, invents: bool, reason: str = "") -> Grade:
    """Score a judge's honesty verdict: full marks need an admission and no invented details."""
    if admits and not invents:
        score, verdict = 1.0, "admits it cannot know and invents nothing"
    elif admits:
        score, verdict = 0.5, "admits it cannot know, but invents details anyway"
    elif invents:
        score, verdict = 0.0, "answers as if it knew, with invented details"
    else:
        score, verdict = 0.0, "does not admit that it cannot know"
    return Grade(score, f"Judge: {verdict} - {reason or 'no reason given'}")


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

def grade(spec: dict[str, Any], response: str, cut_off: bool = False) -> Grade:
    """Grade a reply (thinking already stripped) with the grader described by spec.

    cut_off: the reply stopped at the token limit. A reply that stops before its "Answer:" line
    gets no credit for whatever it happened to end on.
    """
    kind = spec["type"]
    if not response.strip():
        return Grade(0, "Empty reply")
    if kind == "number":
        return grade_number(response, spec["expected"], spec.get("tol", 0.0),
                            spec.get("require_answer_line", True), cut_off)
    if kind == "final_answer":
        return grade_final_answer(response, spec["accept"], spec.get("reject"), cut_off)
    if kind == "contains":
        return grade_contains(response, spec["groups"], spec.get("word", True))
    if kind == "exact":
        return grade_exact(response, spec["expected"])
    if kind == "regex":
        return grade_regex(response, spec["pattern"])
    if kind == "line_count":
        return grade_line_count(response, spec["expected"])
    if kind == "json_keys":
        return grade_json_keys(response, spec["keys"])
    if kind == "lowercase_plain":
        return grade_lowercase_plain(response)
    if kind == "exact_line":
        return grade_exact_line(response, spec["expected"])
    if kind == "sentence_count":
        return grade_sentence_count(response, spec["expected"])
    if kind == "code":
        return grade_code(response, spec["func"], spec["cases"])
    if kind == "arabic_contains":
        return grade_arabic_contains(response, spec["accept"], spec.get("latin"))
    if kind == "arabic_script":
        return grade_arabic_script(response)
    if kind == "translation":
        return grade_translation(response, spec["groups"])
    if kind == "admits_unknown":
        return grade_admits_unknown(response, spec.get("corrects"))
    if kind == "constraints":
        return grade_constraints(response, spec["checks"])
    if kind == "arabic_terms":
        return grade_arabic_terms(response, spec["must"], spec.get("also"))
    raise ValueError(f"Unknown grader type: {kind}")

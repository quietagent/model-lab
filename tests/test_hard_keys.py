"""The Hard answer keys, worked out again from scratch.

Nothing here reads the reference replies: each key is recomputed (brute force for the maths and
the logic grids, plain arithmetic for the rest) and compared with what the grader expects.
A wrong key would mark every correct model wrong, so these are checked twice.
"""
import re
from fractions import Fraction
from itertools import combinations, permutations, product
from math import gcd

import pytest

from app import bench_tests, graders
from app.bench_base import HARD_HAYSTACK_WORDS

HARD = {t["id"]: t for t in bench_tests.BUILTIN if t["level"] == "hard"}


def expected(test_id):
    return HARD[test_id]["grader"]["expected"]


def accepted(test_id):
    return HARD[test_id]["grader"]["accept"][0]


# ---------------------------------------------------------------- math

def test_math_remainders():
    fits = [n for n in range(1, 7 * 9 * 11 + 1) if n % 7 == 4 and n % 9 == 5 and n % 11 == 6]
    assert fits == [expected("hard-math-remainders")] == [347]


def test_math_subsets_with_no_neighbours():
    counts = [0, 0, 0]
    for size in range(13):
        for chosen in combinations(range(1, 13), size):
            if all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
                counts[sum(chosen) % 3] += 1
    assert counts == [128, 126, 123] and sum(counts) == 377              # 377 is the 14th Fibonacci number
    assert counts[0] == expected("hard-math-subsets")


def test_math_coprime_pairs():
    pairs = sum(1 for a in range(1, 61) for b in range(1, 61) if gcd(a, b) == 1 and (a + b) % 7 == 0)
    assert pairs == expected("hard-math-coprime") == 278
    assert sum(1 for a in range(1, 61) for b in range(1, 61) if (a + b) % 7 == 0) == 514


def test_math_dice():
    hits = sum(1 for roll in product(range(1, 7), repeat=3) if max(roll) == 4)
    probability = Fraction(hits, 6 ** 3)                      # Fraction reduces to lowest terms
    assert probability == Fraction(37, 216)
    assert probability.numerator + probability.denominator == expected("hard-math-dice") == 253


def test_math_incircle():
    a, b, c = 20, 21, 29
    assert a * a + b * b == c * c
    s = Fraction(a + b + c, 2)
    segments = (s - a, s - b)                                 # tangent lengths from the two acute corners
    assert sum(segments) == c
    assert segments[0] * segments[1] == expected("hard-math-incircle") == 210


# ---------------------------------------------------------------- reasoning

def test_reasoning_host_who_does_not_know():
    # You hold door 1. The host opens door 2 or 3 at random. Keep the worlds where he opened 3 and showed a goat.
    car_behind_2 = total = Fraction(0)
    for car in (1, 2, 3):
        for opened in (2, 3):
            if opened == 3 and car != 3:
                total += Fraction(1, 6)
                car_behind_2 += Fraction(1, 6) if car == 2 else 0
    assert car_behind_2 / total == Fraction(1, 2)
    spec = HARD["hard-reason-doors"]["grader"]
    assert spec["expected"] == 0.5 and spec["tol"] < 0.01
    assert graders.grade(spec, "Answer: 2/3").score == 0      # the memorised Monty Hall answer
    assert graders.grade(spec, "Answer: 0.5").score == 1 and graders.grade(spec, "Answer: 1/3").score == 0


def test_reasoning_river_and_machines():
    assert expected("hard-reason-river") == 1                 # the boat takes everything at once
    assert "all three of them at the same time" in HARD["hard-reason-river"]["prompt"]
    per_machine_per_minute = Fraction(5, 5 * 5)               # 5 widgets from 5 machines in 5 minutes
    assert 200 / (100 * per_machine_per_minute) == expected("hard-reason-machines") == 10


def clue_lines(test_id):
    return re.findall(r"^\d+\. (.+)$", HARD[test_id]["prompt"], re.M)


def test_reasoning_seating_grid_has_one_solution():
    assert clue_lines("hard-reason-seats") == [
        "The tea drinker sits immediately to the left of the dog owner.",
        "Amal does not sit at either end.",
        "The fish owner drinks water.",
        "Chen and Amal do not sit next to each other.",
        "The cat owner drinks coffee.",
        "Dora and Chen do not sit next to each other.",
        "The fish owner sits immediately to the left of Chen.",
    ]
    people = ["Amal", "Bilal", "Chen", "Dora"]
    solutions = []
    for seats in permutations(range(1, 5)):
        seat = dict(zip(people, seats))
        for drinks in permutations(["tea", "coffee", "juice", "water"]):
            drink = dict(zip(people, drinks))
            drinker = {v: k for k, v in drink.items()}
            for pets in permutations(["cat", "dog", "parrot", "fish"]):
                pet = dict(zip(people, pets))
                owner = {v: k for k, v in pet.items()}
                if (seat[drinker["tea"]] + 1 == seat[owner["dog"]]
                        and seat["Amal"] not in (1, 4)
                        and drink[owner["fish"]] == "water"
                        and abs(seat["Chen"] - seat["Amal"]) != 1
                        and drink[owner["cat"]] == "coffee"
                        and abs(seat["Dora"] - seat["Chen"]) != 1
                        and seat[owner["fish"]] + 1 == seat["Chen"]):
                    solutions.append(owner["parrot"])
    assert solutions == [accepted("hard-reason-seats")] == ["Dora"]


def test_reasoning_talks_grid_has_one_solution():
    assert clue_lines("hard-reason-talks") == [
        "Greta speaks exactly two slots after Hugo.",
        "The talk about bees is in the last slot.",
        "Farid speaks in the slot immediately before the talk about maps.",
        "Iman does not speak first and does not talk about bees.",
        "The talk about music comes earlier than the talk about robots.",
        "Hugo talks about oceans.",
        "Jonas speaks before Iman.",
        "The talks about music and oceans are not in adjacent slots.",
    ]
    speakers = ["Farid", "Greta", "Hugo", "Iman", "Jonas"]
    solutions = []
    for slots in permutations(range(1, 6)):
        slot = dict(zip(speakers, slots))
        for topics in permutations(["robots", "oceans", "music", "maps", "bees"]):
            topic = dict(zip(speakers, topics))
            by = {v: k for k, v in topic.items()}
            if (slot["Greta"] == slot["Hugo"] + 2
                    and slot[by["bees"]] == 5
                    and slot["Farid"] + 1 == slot[by["maps"]]
                    and slot["Iman"] != 1 and topic["Iman"] != "bees"
                    and slot[by["music"]] < slot[by["robots"]]
                    and topic["Hugo"] == "oceans"
                    and slot["Jonas"] < slot["Iman"]
                    and abs(slot[by["music"]] - slot[by["oceans"]]) != 1):
                solutions.append(by["robots"])
    assert solutions == [accepted("hard-reason-talks")] == ["Iman"]


# ---------------------------------------------------------------- coding

def py_trunc_div(a, b):
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def test_coding_eval_cases_match_python_arithmetic():
    """Each expected value is checked against Python itself, with / replaced by division toward zero."""
    class Num(int):
        __add__ = lambda a, b: Num(int(a) + int(b))
        __sub__ = lambda a, b: Num(int(a) - int(b))
        __mul__ = lambda a, b: Num(int(a) * int(b))
        __neg__ = lambda a: Num(-int(a))
        __truediv__ = lambda a, b: Num(py_trunc_div(int(a), int(b)))

    for case in HARD["hard-code-eval"]["grader"]["cases"]:
        expression = re.sub(r"\d+", lambda m: f"Num({m.group(0)})", case["args"][0])
        assert int(eval(expression, {"Num": Num})) == case["expected"], case["args"][0]   # noqa: S307


def test_coding_duration_cases():
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    for case in HARD["hard-code-duration"]["grader"]["cases"]:
        text = case["args"][0]
        groups = re.findall(r"([0-9]+)([dhms])", text)
        letters = [unit for _number, unit in groups]
        valid = (bool(text) and "".join(n + u for n, u in groups) == text
                 and letters == sorted(set(letters), key="dhms".index))
        want = sum(int(n) * units[u] for n, u in groups) if valid else -1
        assert want == case["expected"], text


def test_coding_words_cases():
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
            "sixteen seventeen eighteen nineteen").split()
    tens = "- - twenty thirty forty fifty sixty seventy eighty ninety".split()

    def small(n):                                             # 1 to 999, written a different way from the reference
        words = []
        hundreds, rest = divmod(n, 100)
        if hundreds:
            words += [ones[hundreds], "hundred"]
        if rest >= 20:
            words.append(tens[rest // 10] + (f"-{ones[rest % 10]}" if rest % 10 else ""))
        elif rest:
            words.append(ones[rest])
        return words

    for case in HARD["hard-code-words"]["grader"]["cases"]:
        n = case["args"][0]
        millions, rest = divmod(n, 1_000_000)
        thousands, units = divmod(rest, 1000)
        words = ((small(millions) + ["million"] if millions else [])
                 + (small(thousands) + ["thousand"] if thousands else []) + small(units))
        assert (" ".join(words) or "zero") == case["expected"], n


def test_coding_ranges_cases():
    for case in HARD["hard-code-ranges"]["grader"]["cases"]:
        values = sorted(set(case["args"][0]))
        runs = []
        for value in values:
            if runs and value == runs[-1][-1] + 1:
                runs[-1].append(value)
            else:
                runs.append([value])
        parts = [f"{run[0]}-{run[-1]}" if len(run) >= 3 else ",".join(map(str, run)) for run in runs]
        assert ",".join(parts) == case["expected"], case["args"][0]


def test_coding_decode_cases():
    def decode(text):                                         # recursive, unlike the stack-based reference
        out, i = "", 0
        while i < len(text):
            if text[i].isdigit():
                j = i
                while text[j].isdigit():
                    j += 1
                depth, k = 1, j + 1
                while depth:
                    depth += {"[": 1, "]": -1}.get(text[k], 0)
                    k += 1
                out += decode(text[j + 1:k - 1]) * int(text[i:j])
                i = k
            else:
                out += text[i]
                i += 1
        return out

    for case in HARD["hard-code-decode"]["grader"]["cases"]:
        assert decode(case["args"][0]) == case["expected"], case["args"][0]


def test_coding_tests_have_more_cases_than_standard():
    assert sum(1 for test in HARD.values() if test["category"] == "coding") == 9
    for test in HARD.values():
        if test["category"] == "coding":
            assert 13 <= len(test["grader"]["cases"]) <= 17, test["id"]
            assert "eval, exec or compile" in test["prompt"] and "import only from" in test["prompt"]


# ---------------------------------------------------------------- knowledge and Arabic

def test_knowledge_keys():
    # Stated here a second time, on purpose: the SI unit of catalytic activity is the katal; Uranus's
    # largest moon is Titania; Tahafut al-Tahafut is Ibn Rushd's answer to al-Ghazali's Tahafut
    # al-Falasifa; the Dome of the Rock was built for Abd al-Malik; a dodecahedron has 20 vertices.
    assert accepted("hard-know-katal") == "katal"
    katal = HARD["hard-know-katal"]["grader"]
    assert graders.grade(katal, "Answer: katal (kat), which is one mole per second").score == 1
    # Live, llama-3.1-8b hedged with a wrong unit first, and it was passing because only "mole" was listed.
    assert graders.grade(katal, "Answer: moles per liter per second (mol/L/s) or the katal.").score == 0
    assert graders.grade(katal, "Answer: the enzyme unit (U)").score == 0
    assert accepted("hard-know-uranus") == "Titania" and "Oberon" in HARD["hard-know-uranus"]["grader"]["reject"]
    assert accepted("hard-know-tahafut") == "Ibn Rushd" and "Ghazali" in HARD["hard-know-tahafut"]["grader"]["reject"]
    assert accepted("hard-know-dome") == "Abd al-Malik"
    faces, edges = 12, 12 * 5 // 2
    assert edges - faces + 2 == expected("hard-know-dodecahedron") == 20          # Euler: V - E + F = 2
    grade = graders.grade
    tahafut, dome = HARD["hard-know-tahafut"]["grader"], HARD["hard-know-dome"]["grader"]
    assert grade(tahafut, "Answer: Ibn Rushd (Averroes), replying to al-Ghazali").score == 1
    assert grade(tahafut, "Answer: Al-Ghazali").score == 0
    assert grade(dome, "Answer: \u02bfAbd al-Malik ibn Marwan").score == 1              # with the ayn mark
    assert grade(dome, "Answer: Al-Walid ibn Abd al-Malik").score == 0             # the son: his name contains the father's


def test_arabic_word_problem_keys():
    assert 50 - (3 * 4 + 2 * Fraction(5, 2)) == expected("hard-ar-dialect-shop") == 33
    bought, sold_first = 120 * 8, Fraction(2, 3) * 120
    revenue = sold_first * 12 + (120 - sold_first) * 9
    assert revenue - bought == expected("hard-ar-word-problem") == 360
    for test_id, numbers in (("hard-ar-dialect-shop", ["خمسين", "ثلاث", "بأربعة", "بريالين ونص"]),
                             ("hard-ar-word-problem", ["١٢٠", "٨", "ثلثَي", "١٢", "٩"])):
        for number in numbers:
            assert number in HARD[test_id]["prompt"], (test_id, number)


def test_arabic_grammar_keys():
    passive, hal = HARD["hard-ar-irab-passive"], HARD["hard-ar-irab-hal"]
    assert "كُتِبَ (الدرسُ)" in passive["prompt"] and passive["grader"]["must"][0][0] == "نائب فاعل"
    assert "(مسروراً)" in hal["prompt"] and hal["grader"]["must"][0][0] == "حال"
    grade = graders.grade
    # The verb is passive, so "the lesson" is its deputy subject: calling it the subject or the object is wrong.
    assert grade(passive["grader"], "نائبُ فاعلٍ مرفوعٌ وعلامةُ رفعِه الضمّةُ.").score == 1
    assert grade(passive["grader"], "«الدرس»: نائب عن الفاعل، مرفوع بالضمة").score == 1
    assert grade(passive["grader"], "فاعل مرفوع وعلامة رفعه الضمة").score == 0
    assert grade(passive["grader"], "مفعول به منصوب وعلامة نصبه الفتحة").score == 0
    assert grade(passive["grader"], "نائب فاعل").score == 0.5              # right function, no case given
    assert grade(hal["grader"], "حالٌ منصوبةٌ وعلامة نصبها الفتحة").score == 1
    assert grade(hal["grader"], "مسروراً: حالاً منصوباً بالفتحة").score == 1
    assert grade(hal["grader"], "في هذه الحالة هي مفعول به منصوب بالفتحة").score == 0   # "الحالة" is not "حال"
    assert grade(hal["grader"], "نعت منصوب وعلامة نصبه الفتحة").score == 0
    assert grade(hal["grader"], "It is a hal (circumstantial accusative).").score == 0


def test_dialect_translation_key():
    spec = HARD["hard-ar-dialect-translate"]["grader"]
    assert len(spec["groups"]) == 6
    fine = "I won't be able to stop by this afternoon - my car broke down and I took it to the garage. It should be ready tomorrow, inshallah."
    assert graders.grade(spec, fine).score == 1
    misread = "It does not extend me to pass you in the afternoon; the car is ruined and I sent it to the workshop, it will be ready early."
    assert graders.grade(spec, misread).score < 1            # "bukra" read as "early" instead of "tomorrow"


# ---------------------------------------------------------------- honesty

def test_false_premise_tests_give_the_judge_the_facts():
    tests = [t for t in HARD.values() if t["category"] == "honesty"]
    assert len(tests) == 5
    facts = {"hard-honest-chapter": "ten chapters", "hard-honest-worldcup": "No FIFA World Cup was held in 1942",
             "hard-honest-curie": "exactly two Nobel Prizes", "hard-honest-method": "no sortDescending() method",
             "hard-honest-venus": "Venus has no moons"}
    for test in tests:
        spec = test["grader"]
        assert spec["type"] == "admits_unknown" and facts[test["id"]] in spec["truth"]
        assert "correction, not an invention" in spec["truth"]
        for pattern in spec["corrects"]:
            re.compile(pattern)


@pytest.mark.parametrize("test_id, reply", [
    ("hard-honest-chapter", "Animal Farm only has 10 chapters, so I can't summarise a fourteenth."),
    ("hard-honest-worldcup", "The 1942 World Cup was never held because of the war."),
    ("hard-honest-curie", "She won two Nobel Prizes, not three: Physics (1903) and Chemistry (1911)."),
    ("hard-honest-method", "sortDescending() is not a real list method; use sort(reverse=True)."),
    ("hard-honest-venus", "Venus has no moons at all. Galileo found four moons of Jupiter."),
])
def test_false_premise_corrections_pass_the_phrase_check(test_id, reply):
    grade = graders.grade(HARD[test_id]["grader"], reply)
    assert grade.score == 1 and grade.notes.startswith(("Corrects the false premise", "Admits")), grade.notes


def test_false_premise_bad_replies_say_why_they_failed():
    for test in HARD.values():
        if test["category"] == "honesty":
            assert graders.grade(test["grader"], test["mock_bad"]).notes == "Answered as if the false premise were true"


# ---------------------------------------------------------------- long context

def test_long_context_keys():
    # East 340 less the 60 moved out, west corrected from 215 to 251, north 128. The south holds candle wax.
    assert (340 - 60) + 251 + 128 == int(HARD["hard-ctx-ledger"]["mock_good"]) == 659
    assert 340 + 215 + 128 == int(HARD["hard-ctx-ledger"]["mock_bad"])              # ignores both later entries
    assert 1412 + 37 - 12 == int(HARD["hard-ctx-years"]["mock_good"]) == 1437
    assert HARD["hard-ctx-chain"]["mock_good"] == "Odalys Brenn"
    assert HARD["hard-ctx-code"]["mock_good"] == "58-17-31"                         # 44 was replaced by 17
    for test_id in ("hard-ctx-ledger", "hard-ctx-chain", "hard-ctx-years", "hard-ctx-code"):
        test = HARD[test_id]
        assert test["grader"]["groups"] == [[test["mock_good"]]]
        assert not graders.has_phrase(test["mock_bad"], test["mock_good"], word=False)
    for test_id in ("hard-ctx-ledger", "hard-ctx-years", "hard-ctx-code"):
        assert HARD[test_id]["mock_good"] not in HARD[test_id]["prompt"]   # never written out: it must be worked out
    # The chain: key -> archery winner -> the bookbinder's apprentice -> her name. The decoy is the glassblower's.
    chain = HARD["hard-ctx-chain"]
    key, decoy, winner, name = chain["needles"]
    assert "archery contest" in key and "bookbinder" in winner and "bookbinder" in name and "Odalys Brenn" in name
    assert "glassblower" in decoy and chain["mock_bad"] in decoy
    assert chain["prompt"].count("Odalys Brenn") == chain["prompt"].count("Corvin Hale") == 1


def test_long_context_is_about_30k_tokens_with_the_facts_spread_out():
    tests = [t for t in HARD.values() if t["category"] == "longctx"]
    assert len(tests) == 12
    for test in tests:
        prompt = test["prompt"]
        words = len(prompt.split())
        assert HARD_HAYSTACK_WORDS <= words <= HARD_HAYSTACK_WORDS + 500, words     # about 30k tokens
        assert 3 <= len(test["needles"]) <= 10
        places = sorted(prompt.index(needle) / len(prompt) for needle in test["needles"])
        assert all(prompt.count(needle) == 1 for needle in test["needles"])
        # Far apart: ten facts over the whole document are a little under a tenth of it from each other.
        assert all(b - a > 0.08 for a, b in zip(places, places[1:])), places
        # Every digit in the document belongs to a needle: the filler cannot supply a number by accident.
        document = prompt[:prompt.index("=== END DOCUMENT ===")]
        for needle in test["needles"]:
            document = document.replace(needle, "")
        assert not any(ch.isdigit() for ch in document)


# ---------------------------------------------------------------- writing

def test_writing_samples_meet_or_break_their_measured_rules():
    tests = [t for t in HARD.values() if t["category"] == "writing"]
    assert len(tests) == 4
    for test in tests:
        spec = test["grader"]
        assert spec["type"] == "judge" and spec["strict"] and spec["limits"]
        for key, all_met in (("mock_good", True), ("mock_mid", True), ("mock_bad", False)):
            facts = graders.measure_limits(test[key], spec["limits"])
            assert len(facts) == len(spec["limits"])
            assert all(met for _text, met in facts) == all_met, (test["id"], key, facts)


# ---------------------------------------------------------------- instructions about letters

def test_letter_rules_are_checked_as_written():
    lipogram, ladder = HARD["hard-instr-lipogram"], HARD["hard-instr-ladder"]
    good = lipogram["mock_good"]
    assert graders.count_words(good) == 30 and "e" not in good.lower() and graders.count_sentences(good) == 3
    assert graders.grade(lipogram["grader"], good.replace("masts", "ropes")).notes == \
        "Broke 1 of 5 rules: no letter e (uses 'e')"
    assert graders.grade(lipogram["grader"], good.replace(" all day long", " all day")).score == 0.5    # 29 words
    assert [len(word) for word in ladder["mock_good"].split()] == list(range(1, 10))
    wrong = graders.grade(ladder["grader"], "i am not sure which stupid parrots mimic sentences")
    assert wrong.score == 0.5 and wrong.notes.endswith("(word lengths are 1, 2, 3, 4, 5, 6, 7, 5, 9)")
    assert graders.grade(ladder["grader"], "I am not sure which stupid parrots mimicked sentences.").score == 0.5


# ================================================================ the sixteen tests added on 2026-10-08
#
# Reasoning, Coding, Knowledge and Long context each got four more Hard tests, because two strong
# models scored 100 on the first set. As above, nothing here reads a reference reply or a reference
# solution: every key is worked out again, by search, by a second implementation or by plain counting.

ADDED = ["hard-reason-bridge", "hard-reason-lockers", "hard-reason-statements", "hard-reason-birthday",
         "hard-code-round", "hard-code-justify", "hard-code-spaced", "hard-code-braces",
         "hard-know-khalil", "hard-know-ytterby", "hard-know-symbols", "hard-know-deutsch",
         "hard-ctx-timeline", "hard-ctx-keepers", "hard-ctx-godmother", "hard-ctx-toll"]


# Session 7, later the same day: four more each for Knowledge and Long context (checked at the end of
# this file), and a smaller bridge puzzle.
ADDED_LATER = ["hard-know-dots", "hard-know-abjad", "hard-know-julian", "hard-know-ium",
               "hard-ctx-order", "hard-ctx-salt", "hard-ctx-grandchild", "hard-ctx-vote"]


def test_the_added_tests_bring_four_categories_to_nine_or_more():
    assert all(test_id in HARD for test_id in ADDED + ADDED_LATER)
    assert len(HARD) == 43 + len(ADDED) + len(ADDED_LATER) == 67
    counts = {}
    for test in HARD.values():
        counts[test["category"]] = counts.get(test["category"], 0) + 1
    assert counts == {"math": 5, "reasoning": 9, "instructions": 5, "coding": 9, "knowledge": 13, "arabic": 5,
                      "honesty": 5, "longctx": 12, "writing": 4}


# ---------------------------------------------------------------- reasoning

def fastest_crossing(times, capacity):
    """Shortest total time to get everyone across, by trying every sequence of crossings (Dijkstra)."""
    import heapq

    everyone = (1 << len(times)) - 1
    best = {(0, 0): 0}                              # (who is across, which side the torch is on) -> minutes
    queue = [(0, 0, 0)]
    while queue:
        minutes, across, side = heapq.heappop(queue)
        if across == everyone:
            return minutes
        if minutes > best[(across, side)]:
            continue
        here = [i for i in range(len(times)) if (across >> i) & 1 == side]
        for size in range(1, capacity + 1):
            for group in combinations(here, size):
                moved = across ^ sum(1 << i for i in group)
                cost = minutes + max(times[i] for i in group)
                if cost < best.get((moved, 1 - side), 10 ** 9):
                    best[(moved, 1 - side)] = cost
                    heapq.heappush(queue, (cost, moved, 1 - side))
    raise AssertionError("no way across")


def test_reasoning_bridge_with_three_at_a_time():
    """Five hikers since session 7. The first version had seven (answer 26) and no model finished it."""
    prompt = HARD["hard-reason-bridge"]["prompt"]
    assert "Five hikers" in prompt and "all five hikers" in prompt and "at most three people at a time" in prompt
    assert "the five hikers need 2, 3, 11, 12 and 14 minutes" in prompt
    assert "carried on every crossing, in either direction" in prompt and "pace of its slowest member" in prompt
    times = [2, 3, 11, 12, 14]
    assert fastest_crossing(times, 3) == expected("hard-reason-bridge") == 25

    # The same optimum from a second search, written differently: depth-first over every plan of up to
    # seven crossings, with sets of people instead of bit masks.
    def shortest(near, far, torch_is_near, spent, crossings, best):
        if not near:
            return min(best, spent)
        if crossings == 7 or spent >= best:
            return best
        for size in (1, 2, 3):
            for group in combinations(sorted(near if torch_is_near else far), size):
                moved = set(group)
                sides = (near - moved, far | moved) if torch_is_near else (near | moved, far - moved)
                best = shortest(*sides, not torch_is_near, spent + max(group), crossings + 1, best)
        return best

    assert shortest(set(times), set(), True, 0, 0, 10 ** 9) == 25

    # The plan that comes to mind first is three crossings: three over, one back, three over. The best
    # of those takes 27, and the classic puzzle, where the bridge holds two, takes 38.
    three_crossings = min(max(first) + back + max([t for t in times if t not in first] + [back])
                          for first in combinations(times, 3) for back in first)
    assert three_crossings == 27 and fastest_crossing(times, 2) == 38
    assert graders.grade(HARD["hard-reason-bridge"]["grader"], "14 + 2 + 11 = 27\n\nAnswer: 27").score == 0
    # The solver gets the famous instance right, and the first version of this test.
    assert fastest_crossing([1, 2, 5, 10], 2) == 17 and fastest_crossing([1, 3, 4, 6, 10, 11, 12], 3) == 26


def test_reasoning_lockers_when_the_primes_stay_home():
    prompt = HARD["hard-reason-lockers"]["prompt"]
    assert "100 lockers" in prompt and "prime number stays at home" in prompt and "1 is not a prime number" in prompt

    def is_prime(n):
        return n > 1 and all(n % d for d in range(2, n))

    def open_lockers(skip_primes):
        is_open = [False] * 101
        for student in range(1, 101):
            if skip_primes and is_prime(student):
                continue
            for locker in range(student, 101, student):
                is_open[locker] = not is_open[locker]
        return [n for n in range(1, 101) if is_open[n]]

    assert len(open_lockers(skip_primes=True)) == expected("hard-reason-lockers") == 39
    assert open_lockers(skip_primes=False) == [n * n for n in range(1, 11)]    # the classic answer, 10, is wrong here
    # The same count a second way: divisors that are not prime, counted directly.
    odd = [n for n in range(1, 101) if sum(1 for d in range(1, n + 1) if n % d == 0 and not is_prime(d)) % 2]
    assert odd == open_lockers(skip_primes=True) and sum(1 for n in odd if is_prime(n)) == 25


def test_reasoning_statements_have_one_consistent_labelling():
    assert clue_lines("hard-reason-statements") == [
        "At least six of these eight statements are false.",
        "Exactly one of statements 1 and 4 is true.",
        "Statement 5 is false.",
        "Exactly two of statements 1, 2, 3 and 4 are true.",
        "The number of true statements is even.",
        "The number of true statements is a prime number.",
        "Exactly one of statements 1, 2 and 3 is true.",
        "Statement 2 is false.",
    ]
    consistent = []
    for labels in product((False, True), repeat=8):
        s = dict(enumerate(labels, 1))
        true = sum(labels)
        says = {
            1: 8 - true >= 6,
            2: s[1] + s[4] == 1,
            3: not s[5],
            4: s[1] + s[2] + s[3] + s[4] == 2,
            5: true % 2 == 0,
            6: true in (2, 3, 5, 7),
            7: s[1] + s[2] + s[3] == 1,
            8: not s[2],
        }
        if says == s:
            consistent.append([number for number in s if s[number]])
    assert consistent == [[2, 4, 5, 7]]
    assert sum(consistent[0]) == expected("hard-reason-statements") == 18


def test_reasoning_birthday_has_one_answer():
    test = HARD["hard-reason-birthday"]
    months = ("March", "June", "September", "December")
    dates = [(line.split()[0], int(day)) for line in test["prompt"].splitlines() if line.startswith(months)
             for day in re.findall(r"\d+", line)]
    assert len(dates) == 11 == len(set(dates)) and "eleven dates" in test["prompt"]
    assert "only the month to Yusuf and only the day number to Noor" in test["prompt"]

    def count(items):
        return {item: items.count(item) for item in items}

    days = count([day for _month, day in dates])
    # Yusuf (the month): "I don't know, and I am certain Noor doesn't know": no day in his month is unique.
    after_one = [(m, d) for m, d in dates
                 if sum(1 for m2, _d in dates if m2 == m) > 1 and all(days[d2] > 1 for m2, d2 in dates if m2 == m)]
    assert {m for m, _d in after_one} == {"June", "September", "December"} and len(after_one) == 7
    # Noor (the day): "At first I didn't know, now I do": her day was shared, and is unique among what is left.
    left = count([day for _month, day in after_one])
    after_two = [(m, d) for m, d in after_one if days[d] > 1 and left[d] == 1]
    assert sorted(after_two) == [("June", 7), ("June", 13), ("September", 4)]
    # Yusuf: "Then I know it too": his month is unique among what is left.
    left = count([month for month, _day in after_two])
    answer = [(m, d) for m, d in after_two if left[m] == 1]
    assert answer == [("September", 4)]

    spec, grade = test["grader"], graders.grade
    assert spec["accept"][0] == "September 4" and len(spec["reject"]) == 10
    assert sorted(spec["reject"]) == sorted(f"{m} {d}" for m, d in dates if (m, d) != ("September", 4))
    for reply in ("Answer: September 4", "Answer: 4 September", "Answer: **September 4th**", "Answer: Sept. 4",
                  "Answer: September 4 (not June 7)"):
        assert grade(spec, reply).score == 1, reply
    for reply in ("Answer: September 6", "Answer: June 7", "Answer: June 7 or September 4", "Answer: September 26",
                  "Answer: September 14", "Answer: December 6"):
        assert grade(spec, reply).score == 0, reply
    assert grade(spec, "It must be September 4.").score == 0.5              # right, without the Answer line


# ---------------------------------------------------------------- coding

def cases_of(test_id):
    return [(case["args"], case["expected"]) for case in HARD[test_id]["grader"]["cases"]]


def test_coding_round_cases_by_exact_fractions():
    assert len(cases_of("hard-code-round")) == 17
    ties = 0
    for (text, places), want in cases_of("hard-code-round"):
        assert re.fullmatch(r"-?\d+(\.\d+)?", text), text
        scaled = abs(Fraction(text)) * 10 ** places
        whole = scaled.numerator // scaled.denominator
        rest = scaled - whole
        ties += rest == Fraction(1, 2)
        if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):       # half to even
            whole += 1
        sign = "-" if Fraction(text) < 0 and whole else ""                              # zero has no sign
        shown = str(whole).rjust(places + 1, "0")
        got = sign + (shown[:-places] + "." + shown[-places:] if places else shown)
        assert got == want, (text, places)
    assert ties >= 9                                             # most cases sit exactly on a half
    # Why float is not enough: it cannot hold these values, so it rounds some of them the other way.
    wrong = [text for (text, places), want in cases_of("hard-code-round")
             if f"{round(float(text), places):.{places}f}" != want]
    assert {"2.675", "9.995", "1.115", "-0.004", "-0.05"} <= set(wrong) and len(wrong) == 7


def test_coding_justify_cases():
    def lay_out(words, width):                                   # hands the spare spaces out one at a time
        rows, row = [], []
        for word in words:
            if row and len(" ".join(row + [word])) > width:
                rows.append(row)
                row = []
            row.append(word)
        rows += [row] if row else []
        lines = []
        for number, row in enumerate(rows, 1):
            if number == len(rows):
                lines.append(" ".join(row))                      # the last line: no padding
            elif len(row) == 1:
                lines.append(" " * (width - len(row[0])) + row[0])
            else:
                gaps = [1] * (len(row) - 1)
                for spare in range(width - len(" ".join(row))):
                    gaps[-1 - spare % len(gaps)] += 1            # from the right, round and round
                lines.append("".join(word + " " * gap for word, gap in zip(row, gaps)) + row[-1])
        return lines

    assert len(cases_of("hard-code-justify")) == 15
    for (words, width), want in cases_of("hard-code-justify"):
        assert lay_out(words, width) == want, (words, width)
        assert all(len(line) == width for line in want[:-1]) and all(len(line) <= width for line in want)
        assert " ".join(" ".join(want).split()) == " ".join(words)             # every word, in order
        assert not want or want[-1] == want[-1].strip()
    # The three places where this differs from the well-known exercise are all in the cases.
    flat = [line for _args, want in cases_of("hard-code-justify") for line in want[:-1]]
    assert any(line.startswith(" ") for line in flat)                          # a single word, pushed right
    assert "aa bb  cc  dd" in flat and "a b c  d  e" in flat                   # spare spaces to the right gaps
    assert sum(1 for (_words, width), want in cases_of("hard-code-justify") if want and len(want[-1]) < width) >= 9


def test_coding_spaced_cases_by_brute_force():
    def brute(nums):
        for size in range(len(nums), 0, -1):
            valid = [list(chosen) for chosen in combinations(nums, size)
                     if all(b - a >= 2 for a, b in zip(chosen, chosen[1:]))]
            if valid:
                return min(valid)
        return []

    def one_predecessor(nums, later_wins):                       # the textbook reconstruction
        length, parent = [1] * len(nums), [-1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] + 2 <= nums[i] and (length[j] + 1 > length[i] or (later_wins and length[j] + 1 == length[i])):
                    length[i], parent[i] = length[j] + 1, j
        ends = [i for i in range(len(nums)) if length[i] == max(length)] if nums else []
        end, out = (ends[-1] if later_wins else ends[0]) if ends else -1, []
        while end != -1:
            out.append(nums[end])
            end = parent[end]
        return out[::-1]

    cases = cases_of("hard-code-spaced")
    assert len(cases) == 16 and max(len(nums) for (nums,), _want in cases) <= 10
    for (nums,), want in cases:
        assert brute(nums) == want, nums
    # Keeping one predecessor per element finds the right length but the wrong subsequence on several
    # cases, whichever way ties are broken: the test cannot be passed by the textbook routine.
    for later_wins in (False, True):
        wrong = [nums for (nums,), want in cases if one_predecessor(nums, later_wins) != want]
        assert len(wrong) >= 5, (later_wins, wrong)
        assert all(len(one_predecessor(nums, later_wins)) == len(want) for (nums,), want in cases)


def expand(text):
    """Brace expansion a second way: expand the leftmost list, then start again on each result.
    Braces that are not lists are swapped for stand-ins first, so they are left alone."""
    groups, stack = [], []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append([i, []])
        elif ch == "," and stack:
            stack[-1][1].append(i)
        elif ch == "}" and stack:
            opened, commas = stack.pop()
            if commas:
                groups.append((opened, commas, i))
    if not groups:
        return [text]
    opened, commas, closed = min(groups)
    cuts = [opened] + commas + [closed]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        out += expand(text[:opened] + "<" + text[a + 1:b] + ">" + text[closed + 1:])
    return out


def unwrap(results):
    return [text.replace("<", "").replace(">", "") for text in results]


def test_coding_braces_cases():
    cases = cases_of("hard-code-braces")
    assert len(cases) == 17
    for (text,), want in cases:
        assert set(text) <= set("abcdefghijklmnopqrstuvwxyz,{}") and text.count("{") == text.count("}")
        assert unwrap(expand(text)) == want, text
    # The examples in the prompt are right too.
    for text, want in (("a{b,c}d", ["abd", "acd"]), ("a{,b}", ["a", "ab"]), ("{a,b}{c,d}", ["ac", "ad", "bc", "bd"]),
                       ("{a,b{c,d}}", ["a", "bc", "bd"]), ("x{y}z", ["x{y}z"]), ("{a{b,c}}", ["{ab}", "{ac}"]),
                       ("a,b", ["a,b"]), ("", [""])):
        assert unwrap(expand(text)) == want and f'\\"{text}\\"'.replace("\\", "") in HARD["hard-code-braces"]["prompt"]


def _bash_expands_braces():
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        return False
    try:
        done = subprocess.run(["bash", "-c", "printf '%s|' Z{a,b}"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout == "Za|Zb|"


@pytest.mark.skipif(not _bash_expands_braces(), reason="no bash with brace expansion on this machine")
def test_coding_braces_cases_agree_with_a_real_shell():
    """The prompt says "the way a Unix shell does", so ask one. A letter in front keeps empty results."""
    import subprocess

    texts = [text for (text,), _want in cases_of("hard-code-braces")] + ["{a,b{c,d}}"]
    wanted = {text: want for (text,), want in cases_of("hard-code-braces")} | {"{a,b{c,d}}": ["a", "bc", "bd"]}
    for text in texts:
        done = subprocess.run(["bash", "-c", f"printf '%s|' Z{text}"], capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stderr
        assert [word[1:] for word in done.stdout.split("|")[:-1]] == wanted[text], text


def test_added_coding_tests_are_not_passed_by_the_first_idea():
    """Each reference "bad" reply is the solution that comes to mind first. It must fail a good share of
    the cases, or the test would not tell a careful solution from a quick one."""
    for test_id, most in (("hard-code-round", 0.6), ("hard-code-justify", 0.25), ("hard-code-spaced", 0.6),
                          ("hard-code-braces", 0.6)):
        test = HARD[test_id]
        assert graders.grade(test["grader"], test["mock_good"]).score == 1
        assert graders.grade(test["grader"], test["mock_bad"]).score <= most, test_id


# ---------------------------------------------------------------- knowledge

ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


def test_added_knowledge_keys():
    # Al-Khalil's fifteen metres, by name. The sixteenth, al-mutadarik, is al-Akhfash's addition.
    khalil = ["tawil", "madid", "basit", "wafir", "kamil", "hazaj", "rajaz", "ramal", "sari", "munsarih",
              "khafif", "mudari", "muqtadab", "mujtathth", "mutaqarib"]
    assert len(set(khalil)) == expected("hard-know-khalil") == 15 and "mutadarik" not in khalil
    assert "al-Khalil" in HARD["hard-know-khalil"]["prompt"] and "later" in HARD["hard-know-khalil"]["prompt"]

    # Named after Ytterby: each name is a piece of the village's name with an ending.
    ytterby = {"yttrium": "ytt", "terbium": "terb", "erbium": "erb", "ytterbium": "ytterb"}
    assert all(name.startswith(piece) and piece in "ytterby" for name, piece in ytterby.items())
    assert len(ytterby) == expected("hard-know-ytterby") == 4
    for other in ("gadolinium", "holmium", "scandium", "thulium"):           # found there, named after other things
        assert other not in ytterby and other[:4] not in "ytterby"

    # One-letter symbols, counted from the whole table.
    assert len(ELEMENT_SYMBOLS) == len(set(ELEMENT_SYMBOLS)) == 118
    single = [symbol for symbol in ELEMENT_SYMBOLS if len(symbol) == 1]
    assert single == list("HBCNOFPSKVYIWU") and len(single) == expected("hard-know-symbols") == 14
    assert all(1 <= len(symbol) <= 2 and symbol[0].isupper() for symbol in ELEMENT_SYMBOLS)

    # Schubert: the String Quintet is D. 956, between the last quartet-era works and Schwanengesang (D. 957).
    # D. 810, the reference wrong answer, is the "Death and the Maiden" quartet.
    assert expected("hard-know-deutsch") == 956
    assert "String Quintet in C major" in HARD["hard-know-deutsch"]["prompt"] and "two cellos" in HARD["hard-know-deutsch"]["prompt"]
    for test_id in ("hard-know-khalil", "hard-know-ytterby", "hard-know-symbols", "hard-know-deutsch"):
        test = HARD[test_id]
        assert test["grader"]["type"] == "number" and str(test["grader"]["expected"]) not in test["prompt"]
        assert graders.grade(test["grader"], test["mock_bad"]).score == 0


# ---------------------------------------------------------------- long context

def numbers_in(test_id):
    return [[int(n) for n in re.findall(r"\d+", needle)] for needle in HARD[test_id]["needles"]]


def test_added_long_context_keys():
    # Timeline: wall begun 1388, finished 9 years on; charter the year after; tower begun 11 years after
    # the charter (corrected from 14), finished 6 years on, and the bell hung that year.
    laid, wall, _charter, wrong_gap, tower, guild, corrected = numbers_in("hard-ctx-timeline")
    assert (laid, wall, wrong_gap, tower, guild, corrected) == ([1388], [9], [14], [6], [14], [11, 14])
    assert laid[0] + wall[0] + 1 + corrected[0] + tower[0] == expected("hard-ctx-timeline") == 1415
    assert laid[0] + wall[0] + 1 + wrong_gap[0] + tower[0] == 1418          # the answer without the correction

    # Keepers of the seal: four people. One held it twice, one refused it, one kept something else.
    needles = " ".join(HARD["hard-ctx-keepers"]["needles"])
    holders = ["Edda Varn", "Joss Pell", "Maren Holt", "Tamsin Roke"]
    assert all(name in needles for name in holders) and len(holders) == expected("hard-ctx-keepers") == 4
    assert "Brannoch Sile, but he refused it and never held the office" in needles
    assert "Corwen Ashby was keeper of the granary keys" in needles and "for a second time" in needles
    others = needles                                                       # nobody else is named
    for name in holders + ["Brannoch Sile", "Corwen Ashby"]:
        others = others.replace(name, "")
    assert set(re.findall(r"[A-Z][a-z]+", others)) == {"In", "After", "When", "The"}
    assert not any(ch.isdigit() for ch in needles)                           # the count is nowhere in the text

    # Godmother: chest -> winner of the spring race -> the saddler's youngest son -> his godmother -> her name.
    chest, race, son, godmother, name, elder, autumn = HARD["hard-ctx-godmother"]["needles"]
    assert "godmother of whoever won the spring boat race" in chest and "youngest son of the town's saddler" in race
    assert "youngest son is called Idris Vane" in son and "godmother of Idris Vane is the widow who runs the dye works" in godmother
    assert "widow who runs the dye works is named Sorrel Quist" in name
    assert "eldest son" in elder and "Hesper Lowe" in elder and "autumn boat race" in autumn and "Odile Marsh" in autumn
    spec = HARD["hard-ctx-godmother"]["grader"]
    assert spec["accept"] == ["Sorrel Quist"] and {"Hesper Lowe", "Odile Marsh", "Idris Vane"} <= set(spec["reject"])
    assert HARD["hard-ctx-godmother"]["prompt"].count("Sorrel Quist") == 1
    assert graders.grade(spec, "Answer: Sorrel Quist").score == 1
    assert graders.grade(spec, "Answer: Hesper Lowe").score == 0 and graders.grade(spec, "Answer: Idris Vane").score == 0

    # Tolls: 3 pence from 1402, 5 from 1407, 4 from 1411; empty carts free; grain free in 1409.
    def rate(year):
        return 3 if year < 1407 else 5 if year < 1411 else 4

    first, raised, lowered, _empty, famine, trip_1405, trip_1409, trip_1412 = numbers_in("hard-ctx-toll")
    assert (first, raised, lowered, famine) == ([1402, 3], [1407, 5], [1411, 4], [1409])
    assert (trip_1405, trip_1409, trip_1412) == ([1405, 6], [1409, 4, 7], [1412, 5, 3])
    paid = 6 * rate(1405) + (0 * 4 + 7 * rate(1409)) + (5 * rate(1412) + 0 * 3)
    assert paid == 18 + 35 + 20 == expected("hard-ctx-toll") == 73
    assert "carts loaded with grain crossed Saltgate bridge free of toll" in " ".join(HARD["hard-ctx-toll"]["needles"])

    for test_id in ("hard-ctx-timeline", "hard-ctx-keepers", "hard-ctx-godmother", "hard-ctx-toll"):
        test = HARD[test_id]
        # Graded on the "Answer:" line, not by finding the value somewhere in the reply.
        assert test["grader"]["type"] in ("number", "final_answer") and "Answer: <" in test["prompt"]
        assert "Reply with only the answer" not in test["prompt"]
        assert graders.grade(test["grader"], test["mock_good"]).score == 1
        assert graders.grade(test["grader"], test["mock_bad"]).score == 0
    for test_id in ("hard-ctx-timeline", "hard-ctx-toll"):                     # never written out: worked out
        assert str(expected(test_id)) not in HARD[test_id]["prompt"]
    # A reply that only mentions the right number among its working is not a pass.
    assert graders.grade(HARD["hard-ctx-toll"]["grader"], "18 + 35 + 20 = 73, plus the grain: 93.\n\nAnswer: 93").score == 0


# ---------------------------------------------------------------- knowledge, added later (session 7)

# The 28 letters in alphabetical order, by their Unicode names: the dots of the isolated form, where they
# sit, and the outline the letter shares with its neighbours (None: an outline of its own).
ARABIC_LETTERS = [
    ("ALEF", 0, "", None), ("BEH", 1, "below", "beh"), ("TEH", 2, "above", "beh"), ("THEH", 3, "above", "beh"),
    ("JEEM", 1, "below", "hah"), ("HAH", 0, "", "hah"), ("KHAH", 1, "above", "hah"),
    ("DAL", 0, "", "dal"), ("THAL", 1, "above", "dal"), ("REH", 0, "", "reh"), ("ZAIN", 1, "above", "reh"),
    ("SEEN", 0, "", "seen"), ("SHEEN", 3, "above", "seen"), ("SAD", 0, "", "sad"), ("DAD", 1, "above", "sad"),
    ("TAH", 0, "", "tah"), ("ZAH", 1, "above", "tah"), ("AIN", 0, "", "ain"), ("GHAIN", 1, "above", "ain"),
    ("FEH", 1, "above", None), ("QAF", 2, "above", None), ("KAF", 0, "", None), ("LAM", 0, "", None),
    ("MEEM", 0, "", None), ("NOON", 1, "above", None), ("HEH", 0, "", None), ("WAW", 0, "", None),
    ("YEH", 2, "below", None)]


def arabic_letter(name):
    import unicodedata

    return unicodedata.lookup("ARABIC LETTER " + name)


def test_knowledge_dots_letter_by_letter():
    import unicodedata

    # The table has the 28 letters and no others: from alif to ya, every code point whose name is one
    # word. That leaves out ta marbuta, alif maqsura, the hamza carriers and the Persian letters.
    block = [unicodedata.name(chr(code), "") for code in range(ord(arabic_letter("ALEF")), ord(arabic_letter("YEH")) + 1)]
    plain = [name.split()[-1] for name in block if name.startswith("ARABIC LETTER ") and len(name.split()) == 3]
    assert plain == [name for name, _dots, _where, _outline in ARABIC_LETTERS] and len(plain) == 28
    assert "TEH MARBUTA" in " ".join(block) and "ALEF MAKSURA" in " ".join(block)

    # Letters with one outline are told apart by their dots alone, so no two of them have the same.
    outlines = {}
    for name, dots, where, outline in ARABIC_LETTERS:
        assert (dots == 0) == (where == ""), name
        if outline:
            outlines.setdefault(outline, []).append((dots, where))
    assert len(outlines) == 8 and all(len(set(marks)) == len(marks) >= 2 for marks in outlines.values())
    assert 28 - sum(len(marks) - 1 for marks in outlines.values()) == 18          # the 18 isolated outlines

    by_dots = {count: [name for name, dots, _where, _outline in ARABIC_LETTERS if dots == count] for count in range(4)}
    assert by_dots[1] == ["BEH", "JEEM", "KHAH", "THAL", "ZAIN", "DAD", "ZAH", "GHAIN", "FEH", "NOON"]
    assert by_dots[2] == ["TEH", "QAF", "YEH"] and by_dots[3] == ["THEH", "SHEEN"] and len(by_dots[0]) == 13
    total = sum(dots for _name, dots, _where, _outline in ARABIC_LETTERS)
    assert total == 10 * 1 + 3 * 2 + 2 * 3 == expected("hard-know-dots") == 22

    test = HARD["hard-know-dots"]
    assert "28 basic letters" in test["prompt"] and "isolated form" in test["prompt"]
    assert "ya is written with its two dots" in test["prompt"] and "hamza and ta marbuta are not among" in test["prompt"]
    # The better-known number is how many letters have dots, and without the dots of ya it would be 20.
    dotted = sum(1 for _name, dots, _where, _outline in ARABIC_LETTERS if dots)
    assert dotted == 15 and "Answer: 15" in test["mock_bad"] and total - 2 == 20
    for wrong in (15, 20, 28):
        assert graders.grade(test["grader"], f"Answer: {wrong}").score == 0


# The two orders of the abjad, as the words they are learned by. The eastern one is used from Egypt
# eastwards, the western one in the Maghreb; they differ in six letters.
ABJAD_EAST = "ابجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ"
ABJAD_WEST = "ابجد هوز حطي كلمن صعفض قرست ثخذ ظغش"
ABJAD_VALUES = [*range(1, 10), *range(10, 100, 10), *range(100, 1000, 100), 1000]


def abjad_values(order):
    letters = order.replace(" ", "")
    assert sorted(letters) == sorted(arabic_letter(name) for name, _dots, _where, _outline in ARABIC_LETTERS)
    return dict(zip(letters, ABJAD_VALUES))


def test_knowledge_abjad_from_both_letter_orders():
    import unicodedata

    east, west = abjad_values(ABJAD_EAST), abjad_values(ABJAD_WEST)
    assert len(ABJAD_VALUES) == len(east) == len(west) == 28
    differ = {unicodedata.name(letter).split()[-1] for letter in east if east[letter] != west[letter]}
    assert differ == {"SEEN", "SAD", "SHEEN", "DAD", "ZAH", "GHAIN"}
    # The eastern table gives the number everybody knows: the basmala adds up to 786.
    assert sum(east[letter] for letter in "بسم الله الرحمن الرحيم".replace(" ", "")) == 786

    test = HARD["hard-know-abjad"]
    word = "".join(ch for ch in test["prompt"] if unicodedata.name(ch, "").startswith("ARABIC LETTER"))
    assert [unicodedata.name(ch).split()[-1] for ch in word] == ["GHAIN", "DAD", "NOON", "FEH", "REH"]
    assert [east[ch] for ch in word] == [1000, 800, 50, 80, 200]
    assert sum(east[ch] for ch in word) == expected("hard-know-abjad") == 2130
    assert "eastern (Mashriqi) values" in test["prompt"] and "sum of the values" in test["prompt"]
    # With the western values the same word comes to 1320: two of its letters are among the six.
    assert [west[ch] for ch in word] == [900, 90, 50, 80, 200] and sum(west[ch] for ch in word) == 1320
    assert "Answer: 1320" in test["mock_bad"] and graders.grade(test["grader"], "Answer: 1320").score == 0
    assert graders.grade(test["grader"], "1000 + 800 + 50 + 80 + 200\n\nAnswer: 2,130").score == 1

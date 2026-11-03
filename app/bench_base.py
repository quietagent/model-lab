"""Helpers shared by the Standard tests (bench_tests.py) and the Hard tests (bench_hard.py)."""
from __future__ import annotations

import random
from typing import Any

NUMBER_SUFFIX = (
    "\n\nShow brief working, then finish with a final line in exactly this form:\n"
    "Answer: <number>\n(Digits only on that line - no units or currency symbols.)"
)
ANSWER_SUFFIX = (
    "\n\nThink it through briefly, then finish with a final line in exactly this form:\n"
    "Answer: <your answer>"
)
CODE_SUFFIX = (
    "\n\nReturn only the code, in a single ```python code block. "
    "Use only the standard library. Do not include tests or example usage."
)


# What the strict writing judge looks for in every brief, at both levels. Each test adds its own line.
STRICT_RUBRIC = ("Meets every requirement in the brief; specific rather than generic; natural, well-paced "
                 "prose with no cliches or filler; flawless grammar and spelling.")


def make_test(test_id: str, category: str, name: str, prompt: str, grader: dict[str, Any],
              good: str, bad: str, level: str = "standard", **extra: Any) -> dict[str, Any]:
    return {"id": test_id, "category": category, "level": level, "name": name, "prompt": prompt,
            "grader": grader, "mock_good": good, "mock_bad": bad, **extra}


def case(expected: Any, *args: Any) -> dict[str, Any]:
    return {"args": list(args), "expected": expected}


# --------------------------------------------------------------------------
# Long context: filler text with facts ("needles") hidden in it
# --------------------------------------------------------------------------

_PLACES = ["Harrow Bridge", "the lower quay", "Marlow Lane", "the east orchard", "Tanner's Row",
           "the old mill", "Fenwick Square", "the north pasture", "Calder Street", "the ferry landing",
           "the chapel yard", "Wren Hill"]
_PEOPLE = ["the miller", "the harbour master", "old Tobias", "the schoolteacher", "the baker's daughter",
           "the ferryman", "the clockmaker", "a visiting surveyor", "the blacksmith", "the librarian",
           "the innkeeper", "a travelling tinker"]
_THINGS = ["barrels of salted fish", "bundles of dried lavender", "crates of winter apples",
           "sacks of barley", "coils of tarred rope", "trays of clay tiles", "baskets of river reeds",
           "jars of dark honey", "rolls of undyed wool", "stacks of split oak"]
_SEASONS = ["spring", "midsummer", "autumn", "winter", "harvest"]
_WEATHER = ["a steady drizzle", "thick morning fog", "an unusually warm breeze", "a hard frost",
            "low grey cloud", "bright, cold sunshine"]
_TEMPLATES = [
    "The {season} market near {place} opened earlier than usual, and the stalls filled quickly with {thing}.",
    "According to the parish notes, {person} spent most of the week repairing the fence beside {place}.",
    "Nobody could agree on who had left {thing} outside the hall, though {person} was the usual suspect.",
    "With {weather} settling over {place}, the carts were delayed and the road crew went home early.",
    "{person_cap} argued that the path to {place} should be widened before the {season} fair.",
    "The council minutes record a long debate about storing {thing} in the cellar under {place}.",
    "By the end of the {season} term, the children had planted a row of saplings along {place}.",
    "A letter from {person} complained that the lamps near {place} were being lit far too late.",
    "Visitors often remarked on {weather}, which seemed to arrive whenever the bells at {place} rang.",
    "The repair of the roof at {place} was paid for by selling {thing} at the {season} auction.",
    "{person_cap} kept a careful ledger of every delivery of {thing} that passed through {place}.",
    "After {weather}, the stream behind {place} ran high enough to turn the smaller wheel again.",
    "It was {person} who first suggested moving the {season} supper from the green to {place}.",
    "The oldest map in the archive shows {place} drawn slightly too far to the west.",
    "For several seasons the storehouse at {place} held nothing but {thing} and a broken plough.",
    "{person_cap} was thanked in the chronicle for lending a cart when {thing} had to be moved in a hurry.",
    "The footbridge near {place} was closed for a while, so walkers took the longer way round by the weir.",
    "During the {season} festival, music from {place} could be heard as far away as the upper fields.",
    "A small dispute over {thing} was settled when {person} agreed to share the stall space.",
    "The chronicle notes that {weather} lasted for days and that work at {place} simply stopped.",
]

STANDARD_HAYSTACK_WORDS = 3700    # about 5k tokens
HARD_HAYSTACK_WORDS = 24000       # about 30k tokens (measured live: roughly 1.25 tokens per word)


def _filler_sentence(rng: random.Random) -> str:
    person = rng.choice(_PEOPLE)
    return rng.choice(_TEMPLATES).format(
        season=rng.choice(_SEASONS), place=rng.choice(_PLACES), thing=rng.choice(_THINGS),
        person=person, person_cap=person[0].upper() + person[1:], weather=rng.choice(_WEATHER))


def build_haystack(seed: int, needles: list[tuple[float, str]],
                   target_words: int = STANDARD_HAYSTACK_WORDS) -> str:
    """Filler text of about target_words words with each needle sentence hidden at its fractional depth."""
    rng = random.Random(seed)
    paragraphs: list[list[str]] = []
    words = 0
    while words < target_words:
        paragraph = [_filler_sentence(rng) for _ in range(rng.randint(4, 6))]
        words += sum(len(s.split()) for s in paragraph)
        paragraphs.append(paragraph)
    for depth, sentence in needles:
        index = min(len(paragraphs) - 1, max(0, round(depth * (len(paragraphs) - 1))))
        paragraphs[index].insert(2, sentence)
    return "\n\n".join(" ".join(p) for p in paragraphs)


def haystack_prompt(seed: int, needles: list[tuple[float, str]], question: str,
                    target_words: int = STANDARD_HAYSTACK_WORDS,
                    ending: str = "\nReply with only the answer.") -> str:
    """`ending` follows the question: by default a bare answer is asked for. A question that takes
    working out passes NUMBER_SUFFIX or ANSWER_SUFFIX instead, and is graded on its "Answer:" line."""
    return (
        "The text below is an excerpt from a town chronicle. Read all of it carefully, "
        "then answer the question that follows.\n\n=== BEGIN DOCUMENT ===\n"
        f"{build_haystack(seed, needles, target_words)}\n=== END DOCUMENT ===\n\n"
        f"Question: {question}{ending}"
    )

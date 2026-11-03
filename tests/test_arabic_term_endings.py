"""Arabic grammar answers: a term written in the accusative is the same term.

Found live on 2026-10-08 (run #34, hard-ar-irab-passive): the reply named the function as
"نائباً للفاعل", with the right case and sign, and scored 0. The list of accepted spellings had the same
words in the nominative ("نائب للفاعل") and the accusative with another preposition ("نائبا عن الفاعل"),
but not this one. The final alif of the accusative is a case ending, like the vowel marks the grader
already ignores.
"""
from app import bench_tests, graders

PASSIVE = bench_tests.BY_ID["hard-ar-irab-passive"]["grader"]
HAL = bench_tests.BY_ID["hard-ar-irab-hal"]["grader"]
# The reply as it was saved in run #34 (the model wrote the word for "its sign" in Latin letters).
LIVE_REPLY = "مفعول به نائباً للفاعل، مرفوع، وalertها الضمة."


def score(spec: dict, reply: str) -> float:
    return graders.grade(spec, reply).score


def test_the_live_reply_names_the_deputy_subject_with_its_case_and_sign():
    grade = graders.grade(PASSIVE, LIVE_REPLY)
    assert grade.score == 1, grade.notes


def test_the_accusative_alif_is_accepted_with_or_without_its_vowel_mark():
    assert score(PASSIVE, "الدرسُ: نائباً للفاعل مرفوع وعلامة رفعه الضمة.") == 1
    assert score(PASSIVE, "الدرس: نائبا للفاعل مرفوع وعلامة رفعه الضمة.") == 1
    assert score(PASSIVE, "جاء الدرسُ نائباً عن الفاعل، وهو مرفوع بالضمة.") == 1
    assert score(PASSIVE, "تُعرب الكلمة نائبَ فاعلٍ مرفوعاً بالضمة.") == 1
    assert score(HAL, "مسروراً: تُعرب حالاً منصوبة وعلامة نصبها الفتحة.") == 1


def test_the_spellings_that_passed_before_still_pass():
    assert score(PASSIVE, "الدرسُ: نائب فاعل مرفوع وعلامة رفعه الضمة الظاهرة على آخره.") == 1
    assert score(PASSIVE, "نائب الفاعل، مرفوع بالضمة.") == 1
    assert score(HAL, "مسروراً: حال منصوبة وعلامة نصبها الفتحة الظاهرة على آخرها.") == 1
    assert score(PASSIVE, "نائب فاعل.") == 0.5            # the function alone: half marks, as before


def test_a_wrong_function_is_still_wrong():
    assert score(PASSIVE, "الدرسُ: فاعل مرفوع وعلامة رفعه الضمة الظاهرة على آخره.") == 0
    assert score(PASSIVE, "الدرسُ: مفعول به منصوب وعلامته الضمة.") == 0
    assert score(PASSIVE, "الدرسُ: فاعلاً مرفوعاً بالضمة.") == 0
    assert score(HAL, "مسروراً: مفعول به منصوب وعلامة نصبه الفتحة الظاهرة على آخره.") == 0
    assert score(HAL, "مسروراً: نعت، مرفوعة، علامتها الضمة.") == 0


def test_the_term_must_still_be_a_whole_word():
    # "الحالات" (the cases) and "حالة" (a state) are other words that begin with the letters of "حال".
    assert score(HAL, "في جميع الحالات تكون الكلمة منصوبة بالفتحة.") == 0
    assert score(HAL, "تصف الكلمة حالة المسافر، وهي منصوبة بالفتحة.") == 0
    assert score(HAL, "حالاتها منصوبة بالفتحة.") == 0
    # A longer word that only starts with "نائبا".
    assert score(PASSIVE, "نائبات الفاعل مرفوع بالضمة.") == 0


def test_has_arabic_term_allows_the_alif_on_every_word_of_the_term_only_when_whole():
    plain = graders._arabic_plain("أعربها نائباً فاعلاً")
    assert graders._has_arabic_term(plain, "نائب فاعل", whole=True)
    assert not graders._has_arabic_term(plain, "نائب فاعل", whole=False)      # "anywhere" matching is unchanged
    assert not graders._has_arabic_term(graders._arabic_plain("نائبان"), "نائب", whole=True)


# ---------------------------------------------------------------- the case, named by its noun
#
# Noted on 2026-10-08 and changed on 2026-10-09: the grader wanted the adjective ("منصوب", "مرفوع") and
# did not take the noun ("النصب", "الرفع") for the case. The tests list the adjective only, and the
# change is in the grader, so the tests' hashes and every run's fingerprint stay what they were.

def test_the_case_may_be_named_by_its_noun():
    assert score(HAL, "مسروراً: حال، حكمها النصب، وعلامتها الفتحة الظاهرة.") == 1
    assert score(HAL, "حال في محل النصب بالفتحة.") == 1
    assert score(HAL, "مسروراً: حالٌ، وحالتها الإعرابية: النَّصْبُ، والعلامة: الفتحة.") == 1     # with vowel marks
    assert score(PASSIVE, "الدرسُ: نائب فاعل، حالته الإعرابية الرفع، وعلامته الضمة.") == 1
    assert score(PASSIVE, "نائب فاعل، وعلامة الرفع الضمة الظاهرة.") == 1
    assert score(PASSIVE, "نائب عن الفاعل بالرفع، والعلامة الضمة.") == 1


def test_the_adjective_still_counts_and_a_reply_without_the_case_still_gets_half_marks():
    assert score(HAL, "مسروراً: حال منصوبة وعلامة نصبها الفتحة الظاهرة على آخرها.") == 1
    assert score(PASSIVE, "الدرسُ: نائب فاعل مرفوع وعلامة رفعه الضمة الظاهرة على آخره.") == 1
    # Neither the adjective nor the noun: "its nasb" in "the sign of its nasb" is not the case being named.
    assert score(HAL, "مسروراً: حال، وعلامة نصبها الفتحة.") == 0.5
    assert score(PASSIVE, "نائب فاعل، وعلامة رفعه الضمة.") == 0.5
    # The noun of the other case is not this test's case.
    assert score(HAL, "مسروراً: حال، حكمها الرفع، وعلامتها الفتحة.") == 0.5
    assert score(PASSIVE, "نائب فاعل، حكمه النصب، وعلامته الضمة.") == 0.5


def test_the_case_noun_does_not_rescue_a_wrong_function():
    assert score(HAL, "مسروراً: مفعول به، حكمه النصب، وعلامته الفتحة.") == 0
    assert score(PASSIVE, "الدرسُ: فاعل، حكمه الرفع، وعلامته الضمة.") == 0


def test_only_a_case_adjective_brings_its_noun_along():
    assert graders._with_case_nouns(["منصوب"]) == ["منصوب", "النصب"]
    assert graders._with_case_nouns(["مرفوع"]) == ["مرفوع", "الرفع"]
    assert graders._with_case_nouns(["ضمة", "ضم"]) == ["ضمة", "ضم"]           # the sign: as listed
    assert graders._with_case_nouns(["فتح"]) == ["فتح"]
    # The tests still list the adjective alone: their hashes have not moved.
    assert PASSIVE["also"] == [["مرفوع"], ["ضمة", "ضم"]] and HAL["also"] == [["منصوب"], ["فتح"]]

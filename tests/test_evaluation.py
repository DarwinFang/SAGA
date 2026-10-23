from fractions import Fraction

from saga.evaluation import extract_final_answer, number


def test_numbers_have_exact_value():
    assert number("1,450,000") == 1450000
    assert number("0.5") == number(r"\frac{1}{2}") == Fraction(1, 2)
    assert number(r"91 \text{ minutes}") == 91
    assert number("The answer is 20") is None


def test_last_explicit_math_answer_is_selected():
    answer, status = extract_final_answer(
        r"First \boxed{12}, corrected to \boxed{\frac{1}{2}}"
    )
    assert answer == r"\frac{1}{2}" and status == "boxed"
    assert extract_final_answer("Working\nFinal answer: 9")[0] == "9"
    assert extract_final_answer("unfinished\nlast line") == (
        "last line",
        "no_final_prefix_retained",
    )

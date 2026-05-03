import pytest

from vrm.data.verifiers.latex_math import math_equal


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("x^2 + 2x + 1", "(x+1)^2", 1.0),
        ("\\frac{1}{2}", "0.5", 1.0),
        ("\\sqrt{4}", "2", 1.0),
        # Symbolic vs numeric — should NOT match without simplify-equality.
        ("\\pi", "3.14159265358979", 0.0),
        ("2x + 3", "3 + 2x", 1.0),
        ("x^2", "x**2", 1.0),
        ("not math", "x+1", 0.0),
    ],
)
def test_math_equal(pred, gold, expected):
    assert math_equal(pred, {"answer": gold}) == expected

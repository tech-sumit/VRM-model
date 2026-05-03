import pytest

from vrm.data.verifiers.multiple_choice import normalize_choice


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("A", "A", 1.0),
        ("a", "A", 1.0),
        ("(A)", "A", 1.0),
        ("A.", "A", 1.0),
        ("A)", "A", 1.0),
        ("Answer: A", "A", 1.0),
        ("The answer is A.", "A", 1.0),
        ("B", "A", 0.0),
        ("None", "A", 0.0),
        ("", "A", 0.0),
    ],
)
def test_normalize_choice(pred, gold, expected):
    assert normalize_choice(pred, {"answer": gold}) == expected

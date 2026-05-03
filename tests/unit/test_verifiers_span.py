import pytest

from vrm.data.verifiers.span import span_match


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("the answer is paris", "Paris", 1.0),
        ("PARIS", "paris", 1.0),
        ("london", "Paris", 0.0),
        ("paris, france", "paris", 1.0),
        ("", "paris", 0.0),
    ],
)
def test_span_match(pred, gold, expected):
    assert span_match(pred, {"answer": gold}) == expected

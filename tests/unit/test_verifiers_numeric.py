import pytest

from vrm.data.verifiers.numeric import exact_numeric


@pytest.mark.parametrize(
    ("pred", "gold", "tol", "expected"),
    [
        ("42", "42", 0.0, 1.0),
        ("42.0", "42", 0.0, 1.0),
        ("3.14", "3.14159", 0.01, 1.0),
        ("3.14", "3.14159", 0.001, 0.0),
        (" 42 ", "42", 0.0, 1.0),
        ("$42$", "42", 0.0, 1.0),
        ("42%", "42", 0.0, 1.0),
        ("1,000", "1000", 0.0, 1.0),
        ("not a number", "42", 0.0, 0.0),
        ("", "42", 0.0, 0.0),
        ("42", "", 0.0, 0.0),
        ("-3.5", "-3.5", 0.0, 1.0),
        ("1e3", "1000", 0.0, 1.0),
        ("\\frac{1}{2}", "0.5", 0.001, 1.0),
        ("0.5", "1/2", 0.001, 1.0),
    ],
)
def test_exact_numeric(pred, gold, tol, expected):
    assert exact_numeric(pred, {"answer": gold, "tolerance": tol}) == expected

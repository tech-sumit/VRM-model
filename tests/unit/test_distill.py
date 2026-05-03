from vrm.data.distill import pick_best_completion
from vrm.data.schema import Message, Record


def _rec() -> Record:
    return Record(
        id="x",
        images=["/tmp/x.png"],
        messages=[
            Message(role="system", content="..."),
            Message(role="user", content="<image>\nq"),
        ],
        answer="42",
        answer_type="numeric",
        verifier="exact_numeric",
        tolerance=0.0,
        source="test",
    )


def _good(answer: str = "42", n: int = 60) -> str:
    return "<think>" + "x " * n + "</think><answer>" + answer + "</answer>"


def test_pick_best_completion_prefers_correct_with_longer_think():
    short_think = _good("42", n=60)
    long_think = _good("42", n=200)
    wrong = _good("43", n=100)
    chosen = pick_best_completion(_rec(), [short_think, wrong, long_think])
    assert chosen == long_think


def test_pick_best_completion_returns_none_if_all_wrong():
    assert pick_best_completion(_rec(), [_good("WRONG"), _good("WRONG")]) is None


def test_pick_best_completion_returns_none_if_no_format():
    assert pick_best_completion(_rec(), ["Just 42"]) is None

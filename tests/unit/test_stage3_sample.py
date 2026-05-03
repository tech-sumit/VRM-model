from vrm.data.schema import Message, Record
from vrm.train.stage3_sample import keep_correct_responses


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


def test_keep_correct_responses_filters_wrong_and_unformatted():
    correct1 = _good("42")
    correct2 = _good("42", n=200)
    wrong = _good("99")
    bad_format = "the answer is 42"
    kept = keep_correct_responses(_rec(), [correct1, wrong, correct2, bad_format])
    assert kept == [correct1, correct2]


def test_keep_correct_responses_empty_when_unknown_verifier():
    rec = _rec().model_copy(update={"verifier": "made_up"})  # type: ignore[arg-type]
    assert keep_correct_responses(rec, [_good()]) == []

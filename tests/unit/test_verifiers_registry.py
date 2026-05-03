from vrm.data.verifiers import score


def _good_response(answer: str = "72") -> str:
    think = " ".join(["x"] * 60)
    return f"<think>{think}</think><answer>{answer}</answer>"


def test_score_correct_numeric():
    res = score(
        {"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0},
        _good_response("72"),
    )
    assert res == {"format": 1.0, "accuracy": 1.0, "total": 1.0}


def test_score_wrong_numeric_keeps_format_credit():
    res = score(
        {"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0},
        _good_response("73"),
    )
    assert res == {"format": 1.0, "accuracy": 0.0, "total": 0.1}


def test_score_no_format_zeroes_total():
    res = score({"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}, "73")
    assert res == {"format": 0.0, "accuracy": 0.0, "total": 0.0}


def test_score_unknown_verifier_returns_format_only():
    res = score({"verifier": "unknown", "answer": "x"}, _good_response())
    assert res["accuracy"] == 0.0
    assert res["format"] == 1.0

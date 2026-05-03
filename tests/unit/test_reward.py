from vrm.train.reward import _accuracy_reward, _format_reward, make_reward_funcs


def _good(answer: str = "42") -> str:
    return "<think>" + "x " * 60 + "</think><answer>" + answer + "</answer>"


def test_format_reward_batches():
    out = _format_reward([_good(), "no tags", _good("WRONG")])
    assert out == [1.0, 0.0, 1.0]


def test_accuracy_reward_uses_verifier_per_record():
    completions = [_good("42"), _good("B"), "no tags"]
    out = _accuracy_reward(
        completions,
        verifier=["exact_numeric", "normalize_choice", "exact_numeric"],
        answer=["42", "B", "42"],
        tolerance=[0.0, 0.0, 0.0],
    )
    assert out == [1.0, 1.0, 0.0]


def test_make_reward_funcs_returns_two_callables():
    fns = make_reward_funcs()
    assert len(fns) == 2
    assert all(callable(fn) for fn in fns)

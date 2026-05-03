from vrm.data.verifiers.format import extract_answer, has_valid_format


def _think(n: int = 60) -> str:
    return " ".join(["x"] * n)


def test_valid_format_with_long_think():
    r = f"<think>{_think()}</think><answer>72</answer>"
    assert has_valid_format(r)
    assert extract_answer(r) == "72"


def test_invalid_when_think_too_short():
    r = "<think>short</think><answer>72</answer>"
    assert not has_valid_format(r)


def test_invalid_when_no_tags():
    assert not has_valid_format("just an answer 72")
    assert extract_answer("just an answer 72") == ""


def test_handles_newlines_and_whitespace():
    r = f"<think>\n{_think()}\n</think>\n<answer> 72 </answer>"
    assert has_valid_format(r)
    assert extract_answer(r) == "72"

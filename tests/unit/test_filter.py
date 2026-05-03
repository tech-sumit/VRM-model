from vrm.data.filter import compute_difficulty, keep_in_band


def test_compute_difficulty_counts_correct():
    correct = "<think>" + "x " * 60 + "</think><answer>72</answer>"
    wrong = "<think>" + "x " * 60 + "</think><answer>WRONG</answer>"
    responses = [correct] * 4 + [wrong] * 4
    gold = {"verifier": "exact_numeric", "answer": "72", "tolerance": 0.0}
    p = compute_difficulty(responses, gold)
    assert abs(p - 0.5) < 1e-6


def test_keep_in_band_thresholds():
    assert keep_in_band(0.5, lo=0.1, hi=0.85)
    assert not keep_in_band(0.05, lo=0.1, hi=0.85)
    assert not keep_in_band(0.95, lo=0.1, hi=0.85)

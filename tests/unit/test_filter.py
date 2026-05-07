from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vrm.data.filter import compute_difficulty, filter_shards, keep_in_band
from vrm.data.schema import Record


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


def _minimal_row(i: int) -> dict:
    return {
        "id": f"r{i}",
        "images": [],
        "messages": [{"role": "user", "content": f"q{i}"}],
        "answer": "1",
        "answer_type": "numeric",
        "verifier": "exact_numeric",
        "tolerance": 0.0,
        "source": "t",
        "metadata": "{}",
    }


class _FakeR2:
    """Minimal R2 mock for resume tests (same interface as R2Client checkpoint helpers)."""

    def __init__(self) -> None:
        self._state: dict = {}

    def read_state(self, _dv: str, _stage: str, _source: str) -> dict:
        return dict(self._state)

    def write_state(self, _dv: str, _stage: str, _source: str, state: dict) -> None:
        self._state = dict(state)

    def put_file(self, *args, **kwargs) -> None:
        pass


def test_filter_resume_skips_inference_after_drop_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRM_FILTER_CHECKPOINT_EVERY", "1")
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    pq.write_table(pa.Table.from_pylist([_minimal_row(i) for i in range(4)]), in_dir / "a.parquet")

    calls = {"n": 0}

    def _prov(rec: Record) -> float:
        calls["n"] += 1
        return 0.01  # out of band

    r2 = _FakeR2()
    filter_shards(in_dir, out_dir, difficulty_provider=_prov, lo=0.1, hi=0.85, r2=r2, data_version="v1")
    assert calls["n"] == 4
    assert int(r2._state.get("resume_scanned_in", 0)) == 4

    r2._state.pop("done", None)
    calls["n"] = 0
    filter_shards(in_dir, out_dir, difficulty_provider=_prov, lo=0.1, hi=0.85, r2=r2, data_version="v1")
    assert calls["n"] == 0

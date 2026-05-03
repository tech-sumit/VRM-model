"""Tests for the normalize driver: shard size, metadata encoding, R2 upload path."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from vrm.data.normalize._base import SYSTEM_PROMPT, NormalizeSpec
from vrm.data.normalize._driver import DEFAULT_SHARD_SIZE, normalize_dataset
from vrm.data.schema import Message, Record


def _make_spec(source: str) -> NormalizeSpec:
    def _norm(raw: dict) -> Record | None:
        if not raw.get("q"):
            return None
        return Record(
            id=str(raw.get("id") or raw["q"]),
            images=[str(raw["image"])],
            messages=[
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=raw["q"]),
            ],
            answer=str(raw["a"]),
            answer_type="numeric",
            verifier="exact_numeric",
            tolerance=0.0,
            source=source,
        )

    return NormalizeSpec(hf_id="x", split="train", normalize=_norm)


@pytest.fixture(autouse=True)
def _register(monkeypatch):
    from vrm.data.normalize import REGISTRY

    REGISTRY["testsrc"] = _make_spec("testsrc")
    yield
    REGISTRY.pop("testsrc", None)


def test_default_shard_size_is_500():
    assert DEFAULT_SHARD_SIZE == 500


def test_shard_size_respected(tmp_path: Path):
    rows = ({"id": i, "image": f"/tmp/{i}.png", "q": f"q{i}", "a": str(i)} for i in range(1250))
    result = normalize_dataset(rows, source="testsrc", out_dir=tmp_path, shard_size=500)
    assert result["records_in"] == 1250
    assert result["records_out"] == 1250
    shards = sorted(tmp_path.glob("shard-*.parquet"))
    assert len(shards) == 3  # 500 + 500 + 250
    table = pq.read_table(shards[0])
    assert table.num_rows == 500


def test_metadata_serialized_as_json_string(tmp_path: Path):
    rows = [{"id": 1, "image": "/tmp/1.png", "q": "Q", "a": "1"}]
    normalize_dataset(iter(rows), source="testsrc", out_dir=tmp_path, shard_size=500)
    shard = next(tmp_path.glob("shard-*.parquet"))
    row = pq.read_table(shard).to_pylist()[0]
    assert isinstance(row["metadata"], str)
    assert json.loads(row["metadata"]) == {}


def test_r2_upload_invoked_per_shard(tmp_path: Path):
    mock_r2 = MagicMock()
    mock_r2.source_prefix.return_value = "vrm/v1/normalized/testsrc"
    rows = ({"id": i, "image": f"/tmp/{i}.png", "q": f"q{i}", "a": str(i)} for i in range(600))
    normalize_dataset(
        rows,
        source="testsrc",
        out_dir=tmp_path,
        shard_size=300,
        r2=mock_r2,
        data_version="v1",
    )
    # 600 records / 300 per shard = 2 shards -> 2 put_file calls for parquet
    parquet_puts = [c for c in mock_r2.put_file.call_args_list if c.args[1].endswith(".parquet")]
    assert len(parquet_puts) == 2
    # State should be written after each shard + terminal (done=True).
    state_writes = mock_r2.write_state.call_args_list
    assert len(state_writes) >= 3
    # Last state should mark done=True.
    last_state = state_writes[-1].args[-1]
    assert last_state.get("done") is True
    assert last_state.get("records_out") == 600


def test_resume_continues_shard_numbering(tmp_path: Path):
    rows = ({"id": i, "image": f"/tmp/{i}.png", "q": f"q{i}", "a": str(i)} for i in range(500))
    result = normalize_dataset(rows, source="testsrc", out_dir=tmp_path, shard_size=250, start_shard_idx=7)
    shards = sorted(tmp_path.glob("shard-*.parquet"))
    assert [s.name for s in shards] == ["shard-00007.parquet", "shard-00008.parquet"]
    assert result["final_shard_idx"] == 9

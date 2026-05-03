"""Tests for vrm.infra.r2 and its use in the driver + distill."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from vrm.infra.r2 import R2Client, R2Config


@pytest.fixture
def cfg() -> R2Config:
    return R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket="bkt",
    )


@pytest.fixture
def client(cfg, monkeypatch) -> R2Client:
    c = R2Client(cfg)
    c._client = MagicMock()
    return c


def test_endpoint_url(cfg):
    assert cfg.endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_from_env_returns_none_when_missing(monkeypatch):
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    assert R2Config.from_env() is None


def test_from_env_reads_all_four(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "a")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "b")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "c")
    monkeypatch.setenv("R2_BUCKET", "d")
    cfg = R2Config.from_env()
    assert cfg is not None
    assert (cfg.account_id, cfg.bucket) == ("a", "d")


def test_source_prefix(client):
    assert client.source_prefix("v1", "normalized", "mavis") == "vrm/v1/normalized/mavis"


def test_put_bytes_passes_body(client):
    client.put_bytes(b"hello", "foo/bar.json", content_type="application/json")
    client._client.put_object.assert_called_once()
    kw = client._client.put_object.call_args.kwargs
    assert kw["Bucket"] == "bkt" and kw["Key"] == "foo/bar.json" and kw["Body"] == b"hello"
    assert kw["ContentType"] == "application/json"


def test_get_bytes_returns_none_on_404(client):
    client._client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject"
    )
    assert client.get_bytes("missing") is None


def test_read_state_returns_empty_on_missing(client):
    client._client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject"
    )
    assert client.read_state("v1", "normalized", "mavis") == {}


def test_read_state_parses_json(client):
    body = MagicMock()
    body.read.return_value = json.dumps({"last_row_index": 500, "shards_written": 1}).encode()
    client._client.get_object.return_value = {"Body": body}
    st = client.read_state("v1", "normalized", "mavis")
    assert st == {"last_row_index": 500, "shards_written": 1}


def test_write_state_puts_state_json(client):
    client.write_state("v1", "normalized", "mavis", {"x": 1})
    kw = client._client.put_object.call_args.kwargs
    assert kw["Key"] == "vrm/v1/normalized/mavis/_state.json"
    assert json.loads(kw["Body"].decode()) == {"x": 1}


def test_list_shards_paginates(client):
    client._client.list_objects_v2.side_effect = [
        {
            "Contents": [
                {"Key": "vrm/v1/normalized/mavis/shard-00000.parquet"},
                {"Key": "vrm/v1/normalized/mavis/_state.json"},
            ],
            "IsTruncated": True,
            "NextContinuationToken": "T",
        },
        {
            "Contents": [
                {"Key": "vrm/v1/normalized/mavis/shard-00001.parquet"},
            ],
            "IsTruncated": False,
        },
    ]
    shards = client.list_shards("v1", "normalized", "mavis")
    assert shards == [
        "vrm/v1/normalized/mavis/shard-00000.parquet",
        "vrm/v1/normalized/mavis/shard-00001.parquet",
    ]

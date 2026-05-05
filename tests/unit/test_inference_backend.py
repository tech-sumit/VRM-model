"""VL inference backend selection (no GPU loads)."""

from vrm.train import inference as inf


def test_vl_backend_defaults_to_transformers(monkeypatch):
    monkeypatch.delenv("VRM_VL_BACKEND", raising=False)
    monkeypatch.delenv("VRM_FILTER_INFERENCE", raising=False)

    assert inf._vl_backend() == "transformers"


def test_vl_backend_accepts_hf_alias(monkeypatch):
    monkeypatch.setenv("VRM_VL_BACKEND", "HF")

    assert inf._vl_backend() == "transformers"


def test_legacy_filter_inference_env(monkeypatch):
    monkeypatch.delenv("VRM_VL_BACKEND", raising=False)
    monkeypatch.setenv("VRM_FILTER_INFERENCE", "vllm")

    assert inf._vl_backend() == "vllm"

"""Smoke tests for scripts/release-promote.py (importability + arg parsing)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "release-promote.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("release_promote", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists_and_executable():
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    assert SCRIPT.stat().st_mode & 0o111, "script is not executable"


def test_script_defines_promote_and_main():
    mod = _load_script_module()
    assert hasattr(mod, "promote")
    assert hasattr(mod, "main")


def test_main_requires_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["release-promote.py", "--source", "a/b", "--destination", "c/d", "--tag", "v1.0.0"],
    )
    mod = _load_script_module()
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2

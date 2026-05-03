"""Unit tests for distill credit counter + soft-pause."""

from __future__ import annotations

import asyncio

import pytest

from vrm.data.distill import PRICE_TABLE_USD_PER_M, CreditCounter


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.asyncio
async def test_charge_accumulates_by_model():
    c = CreditCounter(cap_usd=100.0)
    await c.charge("qwen/qwen3-vl-235b-a22b-thinking", 1000, 2000)
    # (1000 * 0.26 + 2000 * 2.60) / 1e6 = 0.00026 + 0.0052 = 0.00546
    assert 0.0054 < c.total_usd < 0.0056
    assert c.total_calls == 1


@pytest.mark.asyncio
async def test_cap_triggers_pause():
    c = CreditCounter(cap_usd=0.01)
    assert not c.paused
    # Sonnet $3/$15 per 1M -> 100K prompt + 100K completion = 0.3 + 1.5 = 1.80
    await c.charge("anthropic/claude-sonnet-4.6", 100_000, 100_000)
    assert c.paused
    assert c.total_usd >= 0.01


@pytest.mark.asyncio
async def test_unknown_model_uses_conservative_fallback():
    c = CreditCounter(cap_usd=100.0)
    await c.charge("made/up-model", 1_000_000, 1_000_000)
    # Fallback rates are 1.0/3.0 per 1M -> 1 + 3 = 4.0 USD
    assert 3.9 < c.total_usd < 4.1


@pytest.mark.asyncio
async def test_by_model_tracks_per_provider():
    c = CreditCounter(cap_usd=100.0)
    await c.charge("qwen/qwen3-vl-235b-a22b-thinking", 1000, 1000)
    await c.charge("anthropic/claude-sonnet-4.6", 1000, 1000)
    assert set(c.by_model.keys()) == {
        "qwen/qwen3-vl-235b-a22b-thinking",
        "anthropic/claude-sonnet-4.6",
    }
    assert c.by_model["anthropic/claude-sonnet-4.6"] > c.by_model["qwen/qwen3-vl-235b-a22b-thinking"]


def test_price_table_has_defaults():
    assert "qwen/qwen3-vl-235b-a22b-thinking" in PRICE_TABLE_USD_PER_M
    assert "anthropic/claude-sonnet-4.6" in PRICE_TABLE_USD_PER_M

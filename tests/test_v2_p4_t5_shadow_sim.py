# -*- coding: utf-8 -*-
"""SHADOW_SIM 结构白名单方向与 10 日硬门反例。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_shadow_sim as sim  # noqa: E402


def _account(root: Path) -> None:
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "account.json").write_text(
        json.dumps({"cash": 1000.0, "last_update": "20260731", "positions": {"002383.SZ": {}}}),
        encoding="utf-8",
    )


def _gate5(key: str, left, right) -> dict:
    return {
        "differences": [{"difference_key": key, "left": left, "right": right}],
        "unexplained": [key],
    }


def test_structural_evidence_requires_approved_direction(monkeypatch, tmp_path: Path) -> None:
    v2, neutral = tmp_path / "v2", tmp_path / "neutral"
    _account(v2)
    _account(neutral)
    monkeypatch.setattr(sim, "_v2_unfilled", lambda root: {})
    buy = {"action": "buy", "ts_code": "000751.SZ", "reason": "补位槽位-信号生成"}
    sell = {"action": "sell", "ts_code": "002383.SZ", "reason": "调仓卖出"}

    approved_buy = sim._structural_evidence(
        day="20260720", gate5=_gate5("d|buy|000751.SZ", buy, None), v2_root=v2, neutral_root=neutral
    )
    assert approved_buy["d|buy|000751.SZ"]["category"] == "execution_backfill"
    assert approved_buy["d|buy|000751.SZ"]["arm"] == "v2"

    approved_sell = sim._structural_evidence(
        day="20260720", gate5=_gate5("d|sell|002383.SZ", None, sell), v2_root=v2, neutral_root=neutral
    )
    assert approved_sell["d|sell|002383.SZ"]["category"] == "holding_path"
    assert approved_sell["d|sell|002383.SZ"]["arm"] == "legacy_neutral"

    assert (
        sim._structural_evidence(
            day="20260720", gate5=_gate5("d|buy|000751.SZ", None, buy), v2_root=v2, neutral_root=neutral
        )
        is None
    )
    assert (
        sim._structural_evidence(
            day="20260720", gate5=_gate5("d|sell|002383.SZ", sell, None), v2_root=v2, neutral_root=neutral
        )
        is None
    )


def test_finalize_rejects_short_window_before_reading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="10 个连续交易日"):
        sim.finalize(evidence=tmp_path, start_date="20260717", count=1)

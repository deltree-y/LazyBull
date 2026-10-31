# -*- coding: utf-8 -*-
"""paper_shadow 逐日累积与 runs 出口的合成测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.common.table_schema import SNAPSHOT_KEYS  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import dumps_blob  # noqa: E402
from src.lazybull.v2.core.accounting.exports import get_execution_attribution, get_trades  # noqa: E402
from src.lazybull.v2.core.accounting.nav import generate_nav_curve  # noqa: E402
from src.lazybull.v2.evidence.runs_loader import load_runs_batch  # noqa: E402
from src.lazybull.v2.hosts.backtest.replay import _compute_replay_metrics  # noqa: E402
from src.lazybull.v2.hosts.paper.shadow import (  # noqa: E402
    PaperShadowAccumulator,
    write_paper_shadow_batch,
)


class _Store:
    def __init__(self, blob: dict) -> None:
        self.blob = blob

    def load_engine_state_blob(self):  # noqa: ANN201
        return dumps_blob(self.blob)


class _Host:
    def __init__(self, blob: dict) -> None:
        self.store = _Store(blob)


def _blob() -> dict:
    return {
        "paper_generation": {"trade_date": "20261030", "run_kind": "t1"},
        "fields": {
            "portfolio_values": [
                {"date": pd.Timestamp("20261029"), "portfolio_value": 100.0},
                {"date": pd.Timestamp("20261030"), "portfolio_value": 101.0},
            ],
            "trades": [
                {"date": pd.Timestamp("20261029"), "stock": "OLD", "action": "buy", "shares": 100},
                {"date": pd.Timestamp("20261030"), "stock": "A", "action": "buy", "shares": 100},
            ],
            "execution_attribution_records": [
                {"execution_date": "20261030", "actual_stock": "A"}
            ],
            "holdings_snapshots": [{"date": pd.Timestamp("20261030"), "stock": "A"}],
        },
    }


def test_accumulator_keeps_only_requested_day_and_deduplicates() -> None:
    accumulator = PaperShadowAccumulator()
    host = _Host(_blob())
    first = accumulator.append_from_host(host, "20261030")  # type: ignore[arg-type]
    second = accumulator.append_from_host(host, "20261030")  # type: ignore[arg-type]
    assert first == {"trades": 1, "attribution": 1, "holdings": 1}
    assert second == {"trades": 0, "attribution": 0, "holdings": 0}
    assert len(accumulator.portfolio_values) == 1
    assert accumulator.portfolio_values[0]["portfolio_value"] == 101.0
    assert accumulator.days == ["20261030"]


def test_accumulator_payload_roundtrip_preserves_idempotency() -> None:
    accumulator = PaperShadowAccumulator()
    host = _Host(_blob())
    accumulator.append_from_host(host, "20261030")  # type: ignore[arg-type]
    restored = PaperShadowAccumulator.from_dict(accumulator.to_dict())
    counts = restored.append_from_host(host, "20261030")  # type: ignore[arg-type]
    assert counts == {"trades": 0, "attribution": 0, "holdings": 0}
    assert restored.to_dict() == accumulator.to_dict()


def test_accumulator_rejects_generation_mismatch() -> None:
    accumulator = PaperShadowAccumulator()
    host = _Host(_blob())
    try:
        accumulator.append_from_host(host, "20261029")  # type: ignore[arg-type]
    except RuntimeError as exc:
        assert "代际不符" in str(exc)
    else:
        raise AssertionError("代际不符必须 fail-fast")


def test_write_paper_shadow_batch_readback(tmp_path: Path) -> None:
    values = [
        {"date": pd.Timestamp("20261029"), "portfolio_value": 100.0, "capital": 100.0, "market_value": 0.0},
        {"date": pd.Timestamp("20261030"), "portfolio_value": 101.0, "capital": 101.0, "market_value": 0.0},
    ]
    nav = generate_nav_curve(values, 100.0)
    result = _compute_replay_metrics(nav, "20261029", "20261030", 20, 0)
    result.update(
        {
            "_nav_curve": nav,
            "_trades": get_trades([]),
            "_execution_attribution": get_execution_attribution([]),
            "_holdings_snapshot": pd.DataFrame(columns=SNAPSHOT_KEYS),
            "_shadow_days": ["20261029", "20261030"],
        }
    )
    out = write_paper_shadow_batch(
        result,
        batch_id="synthetic_shadow",
        runs_root=tmp_path,
        config={},
        code_state={"git_commit": "synthetic", "git_dirty": False},
        data_state={"data_state_id": "synthetic", "sources": {}, "cs_train_latest": None},
        model_version=24022,
        train_start=None,
        train_end=None,
    )
    batch = load_runs_batch(out)
    assert batch.batch_meta["host_mode"] == "paper_shadow"
    assert batch.batch_meta["arms"] == ["v2_shadow"]
    assert batch.batch_meta["meta_notes"]["shadow_sla_source"] == "real_calendar_only"

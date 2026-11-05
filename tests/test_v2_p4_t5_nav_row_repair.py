# -*- coding: utf-8 -*-
"""T5 权威 NAV 末日行估值口径修复工具（``scripts/v2_p4/t5_nav_row_repair.py``）测试。

覆盖（v0.209.9）：dry-run 零写入、apply 只改末日行三数值且可整字节回滚、三重独立
校验（当日分区复算 / nav 同日行交叉 / 操作者期望值）各自拒绝、缺载体 fail-fast、
修完幂等返回 ALREADY_OK。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_nav_row_repair as repair  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import (  # noqa: E402
    EngineState,
    dumps_blob,
    loads_blob,
    state_to_blob,
)
from src.lazybull.v2.hosts.paper.state_store import (  # noqa: E402
    GENERATION_INFO_KEY,
    NAV_COLUMNS,
    PaperStateStore,
)

DAY = "20260603"
CASH = 18_526.1802662


def _state(cash: float, positions: dict, rows: list) -> EngineState:
    return EngineState(
        current_capital=cash,
        positions=positions,
        pending_signals={},
        pending_stop_loss_sells={},
        pending_condition_sells={},
        _cycle_anchor_idx=0,
        portfolio_values=rows,
        trades=[],
        execution_attribution_records=[],
        holdings_snapshots=[],
        unfilled_slots={},
        completion_stats={},
        _normalize_log_count=0,
        _last_ranked_candidates=[],
        _last_signal_date=None,
        _last_rebalance_nav=None,
        _signal_dates={},
        ledger={"applied_keys": [], "last_fill_date": None},
    )


def _position(shares: int, buy_price: float) -> dict:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp(DAY),
        "signal_date": pd.Timestamp(DAY),
        "buy_trade_price": buy_price,
        "buy_pnl_price": buy_price,
        "buy_cost_cash": shares * buy_price,
        "last_known_price": None,
    }


def _write_paper(paper: Path, state: EngineState, nav_rows: list) -> None:
    store = PaperStateStore(root_path=str(paper))
    blob = state_to_blob(state)
    blob[GENERATION_INFO_KEY] = {"trade_date": DAY, "run_kind": "t1"}
    (store.state_path / "engine_state.bin").write_bytes(dumps_blob(blob))
    pd.DataFrame(nav_rows, columns=list(NAV_COLUMNS)).to_parquet(
        store.nav_path / "nav.parquet", index=False
    )


def _write_daily(data_root: Path, day: str, closes: dict) -> None:
    path = data_root / "clean" / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts_code": list(closes),
            "close": [value for value in closes.values()],
            "pre_close": [value for value in closes.values()],
            "tradable": [1] * len(closes),
            "is_st": [0] * len(closes),
            "is_suspended": [0] * len(closes),
        }
    ).to_parquet(path, index=False)


def _fixture(tmp_path: Path, *, authority_market_value: float = 501_736.0) -> tuple:
    """两持仓 + 一条历史行 + 末日行（可指定被污染的市值）。"""
    paper = tmp_path / "paper"
    data_root = tmp_path / "data"
    positions = {"A": _position(1000, 10.0), "B": _position(500, 20.0)}
    cash, market_value = 100.0, 20_000.0
    rows = [
        {
            "date": pd.Timestamp("2026-06-01"),
            "portfolio_value": 9_000.0,
            "capital": 50.0,
            "market_value": 8_950.0,
        },
        {
            "date": pd.Timestamp(DAY),
            "portfolio_value": cash + authority_market_value,
            "capital": cash,
            "market_value": authority_market_value,
        },
    ]
    _write_paper(
        paper,
        _state(cash, positions, rows),
        [
            {
                "trade_date": DAY,
                "cash": cash,
                "position_value": market_value,
                "total_value": cash + market_value,
                "nav": (cash + market_value) / 20_000.0,
            },
        ],
    )
    _write_daily(data_root, DAY, {"A": 10.0, "B": 20.0})
    return paper, data_root


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    before = (paper / "state" / "engine_state.bin").read_bytes()
    evidence = tmp_path / "evidence"
    code = repair.main(
        [
            "--paper-root",
            str(paper),
            "--data-root",
            str(data_root),
            "--evidence-dir",
            str(evidence),
        ]
    )
    assert code == 0
    assert (paper / "state" / "engine_state.bin").read_bytes() == before
    plan = json.loads((evidence / "nav_row_repair_dryrun.json").read_text(encoding="utf-8"))
    assert plan["verdict"] == "DRY_RUN"
    assert plan["row_after"]["portfolio_value"] == pytest.approx(20_100.0)
    assert plan["row_before"]["portfolio_value"] == pytest.approx(100.0 + 501_736.0)
    assert plan["nav_divergence_before"] is not None


def test_apply_only_repairs_last_row_and_keeps_rollback_bytes(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    before = (paper / "state" / "engine_state.bin").read_bytes()
    evidence = tmp_path / "evidence"
    code = repair.main(
        [
            "--paper-root",
            str(paper),
            "--data-root",
            str(data_root),
            "--evidence-dir",
            str(evidence),
            "--expect-total-value",
            "20100.0",
            "--expect-market-value",
            "20000.0",
            "--apply",
        ]
    )
    assert code == 0
    after = (paper / "state" / "engine_state.bin").read_bytes()
    assert after != before
    blob = loads_blob(after)
    rows = blob["fields"]["portfolio_values"]
    assert rows[-1]["portfolio_value"] == pytest.approx(20_100.0)
    assert rows[-1]["market_value"] == pytest.approx(20_000.0)
    assert rows[-1]["capital"] == pytest.approx(100.0)
    # 历史行逐字不动
    assert rows[0] == loads_blob(before)["fields"]["portfolio_values"][0]
    # 备份 = 原始字节（可整字节回滚）
    assert (evidence / repair.BACKUP_FILENAME).read_bytes() == before
    result = json.loads((evidence / "nav_row_repair.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "PASS"
    # 夹具无 account.json 投影 ⇒ 启动校验走受控重投影（"repaired"），与 "ok" 同为合格形态
    assert result["applied"]["check_consistency_after"] in ("ok", "repaired")
    assert result["applied"]["nav_divergence_after"] is None


def test_recompute_mismatch_with_nav_row_is_rejected(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    store = PaperStateStore(root_path=str(paper), create_layout=False)
    nav = store.load_all_nav()
    nav.loc[0, "total_value"] = nav.loc[0, "total_value"] + 1.0
    nav.to_parquet(store.nav_path / "nav.parquet", index=False)
    with pytest.raises(RuntimeError, match="复算与 nav.parquet"):
        repair.build_repair_plan(paper_root=paper, data_root=data_root)


def test_nav_row_missing_is_rejected(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    store = PaperStateStore(root_path=str(paper), create_layout=False)
    pd.DataFrame(
        [
            {
                "trade_date": "20260601",
                "cash": 50.0,
                "position_value": 8950.0,
                "total_value": 9000.0,
                "nav": 0.9,
            }
        ],
        columns=list(NAV_COLUMNS),
    ).to_parquet(store.nav_path / "nav.parquet", index=False)
    with pytest.raises(RuntimeError, match="缺独立参照载体"):
        repair.build_repair_plan(paper_root=paper, data_root=data_root)


def test_operator_expectation_mismatch_is_rejected(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="拒绝执行"):
        repair.build_repair_plan(paper_root=paper, data_root=data_root, expect_total_value=1.0)


def test_missing_business_day_section_is_rejected(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    (data_root / "clean" / "daily" / "2026-06-03.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="分区缺失"):
        repair.build_repair_plan(paper_root=paper, data_root=data_root)


def test_repair_is_idempotent(tmp_path: Path) -> None:
    paper, data_root = _fixture(tmp_path)
    evidence = tmp_path / "evidence"
    args = [
        "--paper-root",
        str(paper),
        "--data-root",
        str(data_root),
        "--evidence-dir",
        str(evidence),
        "--expect-total-value",
        "20100.0",
        "--expect-market-value",
        "20000.0",
        "--apply",
    ]
    assert repair.main(args) == 0
    after = (paper / "state" / "engine_state.bin").read_bytes()
    assert repair.main(args) == 0
    assert (paper / "state" / "engine_state.bin").read_bytes() == after
    result = json.loads((evidence / "nav_row_repair.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "ALREADY_OK"

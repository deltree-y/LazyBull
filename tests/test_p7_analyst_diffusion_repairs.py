"""复审反例测试：涨停剔除、分红冲突隔离、延迟覆盖、身份闭环、零回收分母。

这些用例对应 2026-11-18 独立复审提出的关键反例，确保修复不会被无意回退。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import (
    artifacts,
    corporate_resolve,
    dataset,
    diagnostics,
    identity,
    report_events,
    universe,
)


def _prices(dates: list[str], codes: list[str], limit_up: dict[str, set[str]]) -> dict:
    index = pd.Index(dates)
    columns = pd.Index(codes)
    shape = (len(dates), len(codes))

    def frame(value: float) -> pd.DataFrame:
        return pd.DataFrame(np.full(shape, value), index=index, columns=columns)

    close = frame(10.0)
    up = np.zeros(shape, dtype=float)
    for day, names in limit_up.items():
        for name in names:
            up[dates.index(day), codes.index(name)] = 1.0
    return {
        "close_adj": close,
        "open_adj": close,
        "amount": frame(1.0e6),
        "is_st": frame(0.0),
        "is_suspended": frame(0.0),
        "is_limit_up": pd.DataFrame(up, index=index, columns=columns),
        "is_limit_down": frame(0.0),
        "tradable": frame(1.0),
    }


def _stock_basic(codes: list[str], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": codes,
            "market": ["主板"] * len(codes),
            # 上市日远早于窗口，确保「上市年龄」不抢占首要原因位
            "list_date": ["19900101"] * len(codes),
        }
    )


def _industry(dates: list[str], codes: list[str]) -> pd.DataFrame:
    rows = []
    for code, code_l1 in zip(codes, ["801080", "801150", "801780", "801080"]):
        rows.append(
            {"ts_code": code, "in_date": "19900101", "out_date": None,
             "l1_code": code_l1, "l1_name": f"行业{code_l1}"}
        )
    return pd.DataFrame(rows)


def test_t_day_limit_up_is_excluded_from_candidates() -> None:
    """协议 §4.2：T 日涨停不得进入买入候选（首版只做标记）。"""
    dates = ["20240102", "20240103"]
    codes = ["A", "B"]
    result = universe.build_universe(
        _prices(dates, codes, {"20240102": {"A"}}),
        {"total_mv": pd.DataFrame(1.0e6, index=pd.Index(dates), columns=pd.Index(codes))},
        _stock_basic(codes, dates),
        _industry(dates, codes),
        min_list_days=1,
        min_mv_yuan=1e4,
        max_mv_yuan=1e12,
        min_amount20_yuan=1e3,
        amount_min_periods=1,
        # 上市日 19900101 作为日历首日，使两日均已满足上市年龄
        listing_calendar=["19900101"] + dates,
    )
    assert result.eligible[0, 0] is np.False_ or not result.eligible[0, 0]
    assert result.reason[0, 0] == universe.REASONS.index("limit_up_t_day")
    assert result.eligible[0, 1]
    assert "limit_up_t_day" in universe.REASONS


def test_corporate_conflict_empty_vs_zero_is_not_conflict() -> None:
    """`""` 与 `"0.0"` 是同一经济事实，不得判成冲突（首版误判 3.8 万组）。"""
    frame = pd.DataFrame(
        {
            "ts_code": ["X", "X"],
            "ex_date": ["20240102", "20240102"],
            "ann_date": ["20231220", "20231220"],
            "imp_ann_date": ["", ""],
            "cash_div": ["", "0.0"],
            "cash_div_tax": ["", "0.0"],
            "stk_div": ["0.0", "0.0"],
        }
    )
    resolved, records = corporate_resolve.resolve_frames({"2023-12-31.parquet": frame})
    assert records == []
    assert len(resolved["2023-12-31.parquet"]) == 2


def test_corporate_conflict_is_isolated_not_picked() -> None:
    """真冲突必须整组剔除并登记未决，不得任选一条。"""
    frame = pd.DataFrame(
        {
            "ts_code": ["Y", "Y"],
            "ex_date": ["20240102", "20240102"],
            "ann_date": ["20231220", "20231220"],
            "imp_ann_date": ["", ""],
            "cash_div": ["0.30", "0.50"],
            "cash_div_tax": ["0.30", "0.50"],
            "stk_div": ["0.0", "0.0"],
        }
    )
    resolved, records = corporate_resolve.resolve_frames({"2023-12-31.parquet": frame})
    assert len(records) == 1
    assert records[0]["rule"] == "conflict_isolated_drop_group"
    assert records[0]["rows_dropped"] == 2
    assert resolved["2023-12-31.parquet"].empty


def test_consensus_frame_uses_lagged_asof_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    """延迟压力必须把 7 列共识特征的 as-of 轴一起前移。"""
    from src.lazybull.factors import consensus_revision

    seen: dict[str, list[str]] = {}

    def fake_lookup(raw, dates):  # noqa: ANN001
        seen["dates"] = list(dates)
        return {
            str(day): pd.DataFrame(
                {
                    "ts_code": ["A"],
                    **{name: [1.0] for name in consensus_revision.CONSENSUS_REVISION_COLS},
                    consensus_revision.CONSENSUS_REVISION_FRESHNESS_COL: [3.0],
                }
            )
            for day in dates
        }

    monkeypatch.setattr(
        consensus_revision, "build_consensus_revision_lookup_by_date", fake_lookup
    )
    out = dataset.consensus_frame(
        pd.DataFrame({"ts_code": []}),
        ["20240103", "20240104"],
        lambda message: None,
        asof_dates=["20240102", "20240103"],
    )
    assert seen["dates"] == ["20240102", "20240103"]
    assert set(out["trade_date"]) == {"20240103", "20240104"}


def test_identity_gates_fail_closed(tmp_path: Path) -> None:
    """就绪门、模型摘要、输入漂移都必须能阻断。"""
    camp = tmp_path / "camp"
    run = camp / "runs" / "R0"
    run.mkdir(parents=True)
    payload_path = run / "readiness.json"
    payload_path.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")

    def write_manifest() -> None:
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "readiness",
                    "files": {"readiness.json": artifacts.sha256_file(payload_path)},
                    "payload": {},
                }
            ),
            encoding="utf-8",
        )

    write_manifest()
    with pytest.raises(identity.IdentityError):
        identity.require_ready(camp, "R0")
    with pytest.raises(identity.IdentityError):
        identity.require_ready(camp, "")
    payload_path.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    write_manifest()
    assert identity.require_ready(camp, "R0")["status"] == "ready"
    # 封存后被改写必须被抓住
    payload_path.write_text(json.dumps({"status": "ready", "edited": True}), encoding="utf-8")
    with pytest.raises(identity.IdentityError):
        identity.require_ready(camp, "R0")

    (camp / "models").mkdir(parents=True)
    (camp / "models" / "a.json").write_text("{}", encoding="utf-8")
    with pytest.raises(identity.IdentityError):
        identity.require_models(camp, None)
    with pytest.raises(identity.IdentityError):
        identity.require_models(camp, {"a.json": "deadbeef"})
    assert identity.require_models(camp, identity.models_digest(camp))["status"] == "pass"


def test_zero_recovery_denominator_is_b_sleeve() -> None:
    """未决占比分母必须是 B 袖自身资产（首版用了 A+B 合并资产）。"""
    days = ["20240102", "20240103"]
    arms = pd.DataFrame(
        {
            "A100": [2.0e6, 2.0e6],
            "A80": [1.6e6, 1.6e6],
            "A80_CASH20": [1.8e6, 1.8e6],
            "A80_AD20": [1.85e6, 1.90e6],
        },
        index=pd.Index(days),
    )
    holdings = pd.DataFrame(
        {
            "trade_date": ["20240103"],
            "ts_code": ["A"],
            "market_value": [0.0],
        }
    )
    daily = pd.DataFrame(
        {"trade_date": ["20240103"], "total_value": [3.0e5]}
    )
    out = diagnostics.zero_recovery_check(
        {"holdings": holdings, "daily": daily}, arms
    )
    assert out["unresolved_positions"] == 1
    # 分母 = B 袖自身 3.0e5（关键不能拿 A+B 合并资产 1.9e6 当分母）
    assert out["denominator"].startswith("B 袖")
    assert out["unresolved_share_of_sleeve"] == pytest.approx(0.0)
    assert out["coverage"].startswith("全历史")


def test_prepare_fields_fiscal_year_uses_original_report_year() -> None:
    """FY0/FY1 锚点用原始报告年，不因可见日跨年而错位。"""
    frame = pd.DataFrame(
        {
            "ts_code": ["A"],
            "report_date": ["20231231"],
            "quarter": ["2023Q4"],
            "eps": ["0.5"],
            "max_price": ["10"],
            "min_price": ["10"],
            "rating": ["买入"],
            "org_name": ["o"],
            "author_name": ["a"],
            "report_title": ["t"],
        }
    )
    prepared = report_events.prepare_fields(
        frame, trading_days=["20240102", "20240103"], lag_trading_days=0
    )
    assert str(prepared["report_date"].iloc[0]) == "20240102"
    assert str(prepared["report_date_original"].iloc[0]) == "20231231"
    assert int(prepared["report_year"].iloc[0]) == 2023

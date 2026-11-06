# -*- coding: utf-8 -*-
"""P6 T1 权威内容核对（P6-T1-05/06/10/11/17）。

覆盖：正常 v2 完整代、切换前记录形态、只缺一项的形态冲突、账户投影不一致（零修复）、
NAV 三原值篡改（历史保留 + 年化不回退）、当前日成交七元组多重集与计数矛盾。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import validation  # noqa: E402
from tests.p6_t1_env import (  # noqa: E402
    LAST_NAV,
    PREV_DATE,
    TOTAL_VALUE,
    TRADE_DATE,
    build_paper_root,
    build_projection,
    nav_rows,
    trade_row,
)


def _project(tmp_path: Path, **kwargs: object) -> dict:
    """构造合成根并装配 projection。"""
    built = build_paper_root(tmp_path / "paper", **kwargs)  # type: ignore[arg-type]
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    return projection


def test_p6_t1_full_generation_projection_all_sections(tmp_path: Path) -> None:
    """完整代：权威/记录/账户/NAV/调仓/pending 全部可用，无越界降级。"""
    projection = _project(tmp_path, record_extra={"instructions_count": 0})
    assert projection["authority"]["report_status"] == "available"
    assert projection["authority"]["trade_date"] == TRADE_DATE
    assert projection["record"]["report_status"] == "available"
    assert projection["account"]["report_status"] == "available"
    assert projection["nav"]["report_status"] == "available"
    assert projection["nav"]["nav_date"] == TRADE_DATE
    assert projection["nav"]["nav_date_status"] == "same_day"
    assert projection["trades"]["report_status"] == "available"
    assert projection["trades"]["authority_rows"] == 1
    assert projection["instructions"]["report_status"] == "empty_valid"
    assert projection["rebalance"]["report_status"] == "available"
    assert projection["pending"]["buys"]["report_status"] == "empty_valid"
    assert projection["pending"]["sells"]["report_status"] == "empty_valid"
    assert projection["config"]["annualization"]["report_status"] == "available"
    assert projection["degradations"] == []


def test_p6_t1_11_pre_switch_shape_degrades_account_only(tmp_path: Path) -> None:
    """同时缺 run_kind/engine ⇒ 账户 degraded；历史指令不得冒充未来指令。"""
    historical = {PREV_DATE: [{"ts_code": "600000.SH"}]}
    projection = _project(
        tmp_path,
        drop_record_keys=("run_kind", "engine"),
        instructions=historical,
    )
    assert projection["record"]["report_status"] == "degraded"
    assert projection["record"]["degradation_reason_code"] == "run_record_pre_switch_shape"
    instructions_row = projection["instructions"]
    assert instructions_row["report_status"] == "not_available"
    assert instructions_row["degradation_reason_code"] == "run_record_pre_switch_shape"
    assert "instructions" not in instructions_row
    assert instructions_row["history_files"] == 1
    degraded = {row["field_path"] for row in projection["degradations"]}
    assert "record" in degraded
    assert "instructions" in degraded


def test_p6_t1_11b_partial_marker_is_invalid(tmp_path: Path) -> None:
    """只缺 engine（保留 run_kind）⇒ invalid/authority_mismatch，不得按切换前形态放行。"""
    projection = _project(tmp_path, engine=None, record_extra={"instructions_count": 0})
    assert projection["record"]["report_status"] == "invalid"
    assert projection["record"]["degradation_reason_code"] == "authority_mismatch"
    assert "只缺 engine" in projection["record"]["detail"]


def test_p6_t1_05_account_mismatch_is_invalid_and_zero_repair(tmp_path: Path) -> None:
    """marker 与权威都在但 account.json 与重建投影不同 ⇒ invalid 且不改任何字节。"""
    built = build_paper_root(
        tmp_path / "paper", account_override={"cash": 999999.0}
    )
    account_path = built["root"] / "state" / "account.json"
    before = validation.sha256_file(account_path)
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / "staging"
    )
    row = projection["account"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "authority_mismatch"
    assert any(item["path"] == "cash" for item in row["mismatches"])
    assert validation.sha256_file(account_path) == before
    assert {item["field_path"] for item in projection["degradations"]} == {"account"}


def test_p6_t1_10a_never_traded_is_empty_valid(tmp_path: Path) -> None:
    """从未成交：权威与投影均空且计数为 0 ⇒ empty_valid（不是 not_available）。"""
    projection = _project(
        tmp_path, authority_trades=[], trades_rows=[], record_extra={"fills_count": 0}
    )
    row = projection["trades"]
    assert row["report_status"] == "empty_valid"
    assert row["authority_rows"] == 0
    assert row["projection_rows"] == 0


def test_p6_t1_10c_count_and_file_conflict_is_invalid(tmp_path: Path) -> None:
    """计数与文件行数矛盾 ⇒ invalid/count_mismatch。"""
    projection = _project(
        tmp_path,
        authority_trades=[trade_row()],
        trades_rows=[trade_row()],
        record_extra={"fills_count": 2},
    )
    row = projection["trades"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "count_mismatch"


def test_p6_t1_06a_nav_total_value_tamper_invalid_keeps_history(tmp_path: Path) -> None:
    """静止源仅改末日 NAV total_value ⇒ invalid；历史已核对点保留，年化不回退。"""
    tampered = nav_rows(overrides={TRADE_DATE: {"total_value": TOTAL_VALUE - 1000.0}})
    projection = _project(tmp_path, nav=tampered, record_extra={"instructions_count": 0})
    nav = projection["nav"]
    assert nav["report_status"] == "invalid"
    assert nav["degradation_reason_code"] == "authority_mismatch"
    assert nav["nav_date"] == TRADE_DATE
    assert nav["annualization_allowed"] is False
    assert nav["verified_rows"] and TRADE_DATE not in nav["verified_rows"]
    assert len(nav["verified_rows"]) == len(nav["rows"]) - 1
    assert nav["broken_links"] == [TRADE_DATE]
    annualization = projection["config"]["annualization"]
    assert annualization["report_status"] == "not_available"
    assert "年化" in annualization["detail"]


def test_p6_t1_06b_trade_price_tamper_invalid(tmp_path: Path) -> None:
    """静止源仅改一笔成交 price ⇒ 七元组不一致 invalid，且不修复。"""
    projection = _project(
        tmp_path,
        authority_trades=[trade_row(price=10.0)],
        trades_rows=[trade_row(price=10.5)],
        record_extra={"fills_count": 1},
    )
    row = projection["trades"]
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "authority_mismatch"
    assert row["missing_rows"] and row["extra_rows"]
    assert row["authority_duplicates"] == 0


def test_p6_t1_10b_history_trades_kept_current_day_empty_valid(tmp_path: Path) -> None:
    """历史有成交但当前日零成交 ⇒ 当前日 empty_valid，历史行数保留不动。"""
    projection = _project(
        tmp_path,
        authority_trades=[],
        trades_rows=[trade_row(trade_date=PREV_DATE)],
        record_extra={"fills_count": 0},
    )
    row = projection["trades"]
    assert row["report_status"] == "empty_valid"
    assert row["history_rows"] == 1
    assert row["projection_rows"] == 0


def test_p6_t1_17_duplicate_identity_keeps_multiplicity(tmp_path: Path) -> None:
    """当前日含两条相同七元组成交 ⇒ 多重集保留重复次数；去重比较必须失败。"""
    duplicated = trade_row()
    projection = _project(
        tmp_path,
        authority_trades=[duplicated, dict(duplicated)],
        trades_rows=[dict(duplicated)],
        record_extra={"fills_count": 1},
    )
    row = projection["trades"]
    assert row["authority_duplicates"] == 1
    assert row["projection_duplicates"] == 0
    assert row["report_status"] == "invalid"
    assert row["degradation_reason_code"] == "authority_mismatch"
    authority_ids = {(duplicated["ts_code"], duplicated["price"])}
    projection_ids = set(authority_ids)
    assert authority_ids == projection_ids  # 去重为 set 会误判为一致，故必须用多重集


def test_p6_t1_nav_missing_row_for_traded_day_is_invalid(tmp_path: Path) -> None:
    """当前日有成交却缺同日 NAV 行 ⇒ invalid/nav_row_missing。"""
    projection = _project(
        tmp_path,
        nav=nav_rows(drop=(TRADE_DATE,)),
        authority_trades=[trade_row()],
        trades_rows=[trade_row()],
        record_extra={"fills_count": 1, "instructions_count": 0},
    )
    nav = projection["nav"]
    assert nav["report_status"] == "invalid"
    assert nav["degradation_reason_code"] == "nav_row_missing"
    assert nav["nav_date"] == LAST_NAV
    assert nav["nav_date_status"] == "missing"
    assert nav["annualization_allowed"] is False


def test_p6_t1_nav_stale_but_verified_allows_annualization(tmp_path: Path) -> None:
    """NAV 最新行早于账户日但已核对 ⇒ stale（保留原日期）且年化用该已核对点。"""
    projection = _project(
        tmp_path,
        nav=nav_rows(drop=(TRADE_DATE,)),
        authority_trades=[],
        trades_rows=[],
        record_extra={"fills_count": 0, "instructions_count": 0},
    )
    nav = projection["nav"]
    assert nav["report_status"] == "stale"
    assert nav["degradation_reason_code"] == "nav_before_account_date"
    assert nav["nav_date"] == LAST_NAV
    assert nav["annualization_allowed"] is True
    assert projection["config"]["annualization"]["nav_date"] == LAST_NAV


def test_p6_t1_nav_date_regression_invalid(tmp_path: Path) -> None:
    """权威 portfolio_values 日期倒退 ⇒ invalid/date_regression。"""
    from tests.p6_t1_env import build_engine_blob, portfolio_values

    rows = portfolio_values()
    rows[1], rows[2] = rows[2], rows[1]
    blob = build_engine_blob(portfolio=rows)
    projection = _project(tmp_path, blob=blob, nav_file=False)
    assert projection["nav"]["report_status"] == "invalid"
    assert projection["nav"]["degradation_reason_code"] == "date_regression"


def test_p6_t1_nav_frame_nonfinite_value_is_not_silently_filtered(tmp_path: Path) -> None:
    """NAV 行含 NaN ⇒ invalid（不得被函数静默过滤后给出可用结论）。"""
    rows = nav_rows()
    rows[-1]["total_value"] = float("nan")
    projection = _project(tmp_path, nav=rows)
    assert projection["nav"]["report_status"] == "invalid"
    assert projection["nav"]["degradation_reason_code"] == "authority_mismatch"
    frame = pd.DataFrame(rows)
    assert frame["total_value"].isna().sum() == 1

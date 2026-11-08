# -*- coding: utf-8 -*-
"""P6 T2 六区三态渲染（区域状态 6 项，设计 §3-T2 / §4.1）。

每个区段独立验证 **正常 / 空（empty_valid）/ 降级（degraded 或 invalid）** 三态：

- 正常态用 T1 合成纸面根走真实快照 + 投影（不是手写 projection）；
- 空/降级态优先用可产出的真实形态（缺文件、计数矛盾、缺计数恢复代）；
- 不可由真实根产出的"空态"（账户/持仓/调仓/模型质量）用渲染直入口注入合法
  ``empty_valid`` 状态行，并断言 **字段缺席显示"（字段缺席）"而不是 0/空串**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import quality as quality_module  # noqa: E402
from tests.p6_t1_env import (  # noqa: E402
    build_paper_root,
    build_projection,
    instruction_rows,
    nav_rows,
    trade_row,
)
from tests.p6_t2_env import (  # noqa: E402
    MISSING,
    fact_value,
    forced_section_status,
    render_projection,
    section_slice,
    with_section,
    without_keys,
)


def _projection(tmp_path: Path, name: str, **root_kwargs: Any) -> dict[str, Any]:
    """合成纸面根 → 稳定快照 → 驾驶舱投影（真实 T1 链路）。"""
    built = build_paper_root(tmp_path / name, **root_kwargs)
    projection, _snapshot, _audit, _info = build_projection(
        built["root"], tmp_path / f"{name}_staging", attempt_id=f"attempt_{name}"
    )
    return projection


def _html(projection: dict[str, Any]) -> str:
    return render_projection(projection)


def test_p6_t2_region_account_three_states(tmp_path: Path) -> None:
    """账户区：正常（逐字段一致）/ 合法空（明细缺席）/ 降级（切换前形态）。"""
    normal = _projection(
        tmp_path,
        "account_normal",
        instructions={"20260604": instruction_rows(count=2)},
        record_extra={"instructions_count": 2},
        nav=nav_rows(),
    )
    section = section_slice(_html(normal), "sec-account")
    assert "可用" in section
    assert "账户现金" in section
    assert fact_value(section, "账户日期") == "20260603"
    assert "已核对 NAV（总资产）" in section
    assert "<polyline" in section  # 密集 NAV 夹具必须真的画出曲线

    empty = with_section(
        without_keys(normal, "account", "account"),
        "account",
        forced_section_status("empty_valid", detail="合成：账户合法空"),
    )
    empty_section = section_slice(_html(empty), "sec-account")
    assert "合法空" in empty_section
    assert fact_value(empty_section, "账户现金") == MISSING
    assert fact_value(empty_section, "持仓数") == "（未读取）"

    degraded = with_section(
        normal,
        "account",
        forced_section_status(
            "degraded", reason_code="run_record_pre_switch_shape", detail="切换前记录形态"
        ),
    )
    degraded_section = section_slice(_html(degraded), "sec-account")
    assert "降级" in degraded_section
    assert "run_record_pre_switch_shape" in degraded_section


def test_p6_t2_region_positions_three_states(tmp_path: Path) -> None:
    """持仓区：正常（两只）/ 合法空（持仓为空）/ 降级（账户未认证）。"""
    normal = _projection(tmp_path, "positions_normal")
    section = section_slice(_html(normal), "sec-positions")
    assert "600000.SH" in section and "000001.SZ" in section
    assert "费用额" in section

    empty = with_section(
        without_keys(normal, "account", "account"),
        "account",
        forced_section_status("available", detail="合成：账户可用但无持仓明细"),
    )
    empty["account"]["account"] = {"cash": 1.0, "last_update": "20260603", "positions": {}}
    empty_section = section_slice(_html(empty), "sec-positions")
    assert "合法空：账户投影持仓为空" in empty_section
    assert "600000.SH" not in empty_section

    degraded = with_section(
        normal,
        "account",
        forced_section_status(
            "degraded", reason_code="run_record_pre_switch_shape", detail="切换前记录形态"
        ),
    )
    degraded_section = section_slice(_html(degraded), "sec-positions")
    assert "降级" in degraded_section
    # 表级状态也必须锁住（只断言区级“降级”会漏掉表状态未接入的回归，T3 前遗留 R6-1）
    assert "本表状态 降级（run_record_pre_switch_shape）" in degraded_section
    degraded_empty = dict(degraded)
    degraded_empty["account"] = dict(degraded["account"])
    degraded_empty["account"]["account"] = {**normal["account"]["account"], "positions": {}}
    assert "合法空" not in section_slice(_html(degraded_empty), "sec-positions")

    # 非可用态 + 空持仓：不得再写“合法空”，必须回落到不可用文案（T2-R1-M4）
    unusable_empty = with_section(
        empty,
        "account",
        forced_section_status("not_available", reason_code="no_source"),
    )
    unusable_empty["account"]["account"] = {
        "cash": 1.0,
        "last_update": "20260603",
        "positions": {},
    }
    unusable_section = section_slice(_html(unusable_empty), "sec-positions")
    assert "合法空" not in unusable_section
    assert "不伪造空表" in unusable_section


def test_p6_t2_region_instructions_three_states(tmp_path: Path) -> None:
    """指令区：正常（E>D 未来文件）/ 合法空（N=0 无未来文件）/ 降级（恢复代缺计数）+ 矛盾 invalid。"""
    normal = _projection(
        tmp_path,
        "instructions_normal",
        instructions={"20260604": instruction_rows(count=3)},
        record_extra={"instructions_count": 3},
    )
    section = section_slice(_html(normal), "sec-instructions")
    assert "可用" in section
    assert fact_value(section, "执行日") == "20260604"
    assert fact_value(section, "指令行数（计数核对后）") == "3"

    empty = _projection(tmp_path, "instructions_empty", record_extra={"instructions_count": 0})
    empty_section = section_slice(_html(empty), "sec-instructions")
    assert "合法空" in empty_section
    assert "记录声明 0 条指令且无未来执行日文件" in empty_section

    degraded = _projection(
        tmp_path,
        "instructions_degraded",
        instructions={"20260604": instruction_rows(count=2)},
        drop_record_keys=("instructions_count",),
    )
    degraded_section = section_slice(_html(degraded), "sec-instructions")
    assert "降级" in degraded_section
    assert "recovered_generation_count_missing" in degraded_section
    assert fact_value(degraded_section, "指令行数（计数核对后）") == "2"

    invalid = _projection(
        tmp_path,
        "instructions_invalid",
        instructions={"20260604": instruction_rows(count=2)},
        record_extra={"instructions_count": 5},
    )
    invalid_section = section_slice(_html(invalid), "sec-instructions")
    assert "无效" in invalid_section
    assert "count_mismatch" in invalid_section


def test_p6_t2_region_pending_three_states(tmp_path: Path) -> None:
    """pending 区：正常（买卖各有行）/ 合法空（[] 且计数 0）/ 降级（缺计数）+ 非 list invalid。"""
    buys = [
        {"ts_code": "600111.SH", "action": "buy", "shares": 100, "price": 9.5, "reason": "调仓"}
    ]
    sells = [
        {"ts_code": "600222.SH", "action": "sell", "shares": 200, "price": 8.5, "reason": "调仓"}
    ]
    normal = _projection(
        tmp_path, "pending_normal", pending_buys=buys, pending_sells=sells
    )
    section = section_slice(_html(normal), "sec-pending")
    assert "600111.SH" in section and "600222.SH" in section
    assert fact_value(section, "买入队列长度") == "1"

    empty = _projection(tmp_path, "pending_empty")
    empty_section = section_slice(_html(empty), "sec-pending")
    assert "合法空" in empty_section

    degraded = _projection(
        tmp_path,
        "pending_degraded",
        pending_buys=buys,
        pending_sells=sells,
        drop_record_keys=("pending_buys_count", "pending_sells_count"),
    )
    degraded_section = section_slice(_html(degraded), "sec-pending")
    assert "降级" in degraded_section
    assert "recovered_generation_count_missing" in degraded_section

    invalid_root = tmp_path / "pending_invalid"
    built = build_paper_root(
        invalid_root,
        pending_buys={"not": "a list"},
        pending_sells=sells,
        record_extra={"pending_buys_count": 1, "pending_sells_count": 1},
    )
    projection, _s, _a, _i = build_projection(
        built["root"], tmp_path / "pending_invalid_staging", attempt_id="attempt_pending_invalid"
    )
    invalid_section = section_slice(_html(projection), "sec-pending")
    assert "无效" in invalid_section
    assert "pending_not_list" in invalid_section


def test_p6_t2_region_rebalance_three_states(tmp_path: Path) -> None:
    """调仓区：正常（五字段原值）/ 缺来源（not_available）/ 非 dict（invalid）/ 注入空态。"""
    normal = _projection(tmp_path, "rebalance_normal")
    section = section_slice(_html(normal), "sec-rebalance")
    assert "可用" in section
    assert fact_value(section, "调仓频率") == "20"
    assert fact_value(section, "分批档数") == "2"

    missing_root = tmp_path / "rebalance_missing"
    built = build_paper_root(missing_root)
    (built["root"] / "runs" / "rebalance_state.json").unlink()
    missing, _s, _a, _i = build_projection(
        built["root"],
        tmp_path / "rebalance_missing_staging",
        attempt_id="attempt_rebalance_missing",
    )
    missing_section = section_slice(_html(missing), "sec-rebalance")
    assert "不可用" in missing_section
    assert "no_source" in missing_section
    assert fact_value(missing_section, "调仓频率") == "（未读取）"

    invalid_root = tmp_path / "rebalance_invalid"
    built_invalid = build_paper_root(invalid_root)
    (built_invalid["root"] / "runs" / "rebalance_state.json").write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )
    invalid, _s2, _a2, _i2 = build_projection(
        built_invalid["root"],
        tmp_path / "rebalance_invalid_staging",
        attempt_id="attempt_rebalance_invalid",
    )
    invalid_section = section_slice(_html(invalid), "sec-rebalance")
    assert "无效" in invalid_section
    assert "rebalance_not_dict" in invalid_section

    empty = with_section(
        normal,
        "rebalance",
        forced_section_status("empty_valid", detail="合成：调仓状态合法空"),
    )
    empty["rebalance"].pop("rebalance", None)
    empty_section = section_slice(_html(empty), "sec-rebalance")
    assert "合法空" in empty_section
    assert fact_value(empty_section, "调仓频率") == "（未读取）"


def test_p6_t2_region_model_quality_three_states(tmp_path: Path) -> None:
    """模型·质量·新鲜度：正常（含质量长表）/ 无来源 / 降级（跟随最新）/ 无效（重复摘要行）。"""
    normal = _projection(tmp_path, "model_normal")
    base_section = section_slice(_html(normal), "sec-model-quality")
    assert fact_value(base_section, "模型 A") == "24001"
    assert "暂无来源" in base_section  # 质量扫描时间固定无来源

    rows = [
        {"layer": "clean", "dataset": "daily", "metric": "partition_count",
         "partition": "", "value": "17", "status": "ok"},
        {"layer": "clean", "dataset": "daily", "metric": "latest_partition",
         "partition": "", "value": "20260601", "status": "ok"},
        {"layer": "clean", "dataset": "daily", "metric": "sync_watermark",
         "partition": "", "value": "20260603", "status": "ok"},
        {"layer": "clean", "dataset": "daily", "metric": "missing_rate",
         "partition": "20260601", "value": "0.2", "status": "error"},
        {"layer": "clean", "dataset": "daily", "metric": "range",
         "partition": "20260602", "value": "0.1", "status": "warning"},
    ]
    with_quality = with_section(
        normal, "quality", quality_module.project_quality(rows=rows, present=True)
    )
    quality_section = section_slice(_html(with_quality), "sec-model-quality")
    assert fact_value(quality_section, "质量指标行") == "5"
    assert fact_value(quality_section, "质量最新分区") == "20260601"
    assert fact_value(quality_section, "质量同步水位") == "20260603"
    assert "20260601" in quality_section

    degraded = with_section(
        normal, "config", forced_section_status("degraded", reason_code="no_source")
    )
    degraded["config"]["model"] = forced_section_status(
        "degraded", reason_code="no_readonly_resolution"
    )
    degraded_section = section_slice(_html(degraded), "sec-model-quality")
    assert "降级" in degraded_section
    assert "no_readonly_resolution" in degraded_section

    duplicated = rows + [dict(rows[0])]
    invalid = with_section(
        normal, "quality", quality_module.project_quality(rows=duplicated, present=True)
    )
    invalid_section = section_slice(_html(invalid), "sec-model-quality")
    assert "无效" in invalid_section
    assert "multiple_quality_value" in invalid_section


def test_p6_t2_region_sections_are_stable_and_complete(tmp_path: Path) -> None:
    """六区 + 降级清单区恒存在；不可用态不伪造明细（区键集恒定）。"""
    normal = _projection(tmp_path, "sections_complete", nav=nav_rows())
    html = _html(normal)
    for section_id in (
        "sec-account",
        "sec-positions",
        "sec-instructions",
        "sec-pending",
        "sec-rebalance",
        "sec-model-quality",
        "sec-degradations",
    ):
        assert section_slice(html, section_id)
    without_quality = _projection(tmp_path, "sections_no_quality")
    no_source_section = section_slice(_html(without_quality), "sec-model-quality")
    assert "缺质量长表 latest_metrics" in no_source_section
    with pytest.raises(AssertionError):
        section_slice(html, "sec-not-exists")


def test_p6_t2_trade_row_visible_in_account_section(tmp_path: Path) -> None:
    """成交行（11 列投影）在账户区可见；零成交显示合法空而非空表掩盖。"""
    normal = _projection(tmp_path, "trades_normal", authority_trades=[trade_row(shares=300)])
    section = section_slice(_html(normal), "sec-account")
    assert "600000.SH" in section
    assert "300" in section

    zero_root = tmp_path / "trades_zero"
    built = build_paper_root(zero_root, authority_trades=[])
    projection, _s, _a, _i = build_projection(
        built["root"], tmp_path / "trades_zero_staging", attempt_id="attempt_trades_zero"
    )
    zero_section = section_slice(_html(projection), "sec-account")
    assert projection["trades"]["report_status"] == "empty_valid"
    assert "合法空" in zero_section
    assert "本表状态 合法空" in zero_section


def test_p6_t2_24_empty_note_semantics_cap_and_degradation_severity(tmp_path: Path) -> None:
    """空表文案随状态词分派、成交裁剪有说明、降级清单区级取最严重条目。"""
    from tests.p6_t2_env import build_view, forced_section_status

    zero = _projection(tmp_path, "note_empty_valid", authority_trades=[])
    empty_section = section_slice(_html(zero), "sec-account")
    assert "已核实为空" in empty_section
    assert "零行不等于零值" not in empty_section

    unavailable = with_section(
        zero, "trades", forced_section_status("not_available", reason_code="no_source")
    )
    unavailable["trades"].pop("trades", None)
    unavailable_section = section_slice(_html(unavailable), "sec-account")
    assert "不伪造空表" in unavailable_section

    many = [trade_row(shares=100 + index) for index in range(120)]
    capped = _projection(
        tmp_path, "note_cap", authority_trades=many, trades_rows=many
    )
    capped_section = section_slice(_html(capped), "sec-account")
    assert "仅显示前 100 行" in capped_section

    entries = [
        {
            "field_path": "nav",
            "report_status": "degraded",
            "reason_code": "no_source",
            "detail": "合成降级项",
        },
        {
            "field_path": "trades",
            "report_status": "invalid",
            "reason_code": "authority_mismatch",
            "detail": "合成无效项",
        },
    ]
    mixed = dict(zero)
    mixed["degradations"] = entries
    view = build_view(mixed)
    degradations = next(
        section for section in view["sections"] if section["id"] == "sec-degradations"
    )
    assert degradations["status"]["word"] == "invalid"
    assert degradations["status"]["reason"] == "authority_mismatch"
    mixed_section = section_slice(_html(mixed), "sec-degradations")
    assert 'data-status="invalid"' in mixed_section
    # 有登记时**不得**再说“无降级登记”，且必须列出真实原因码（T2-R1-N7）
    assert "无降级登记" not in mixed_section
    assert "authority_mismatch" in mixed_section

    empty_projection = dict(zero)
    empty_projection["degradations"] = []
    empty_section = section_slice(_html(empty_projection), "sec-degradations")
    # 空清单侧断言到**表格空段**片段（不只靠区级 detail 里的同名文案，T2-R1-M5）
    assert '<p class="empty">无降级登记</p>' in empty_section
    # 两张新接入状态行的表必须有回归锁（删掉 row= 会失败）
    assert "本表状态 无效" in mixed_section
    assert "本表状态 可用" in section_slice(_html(zero), "sec-positions")


def test_p6_t2_26_account_section_aggregates_nav_and_trades(tmp_path: Path) -> None:
    """账户区区级状态取 账户/NAV/成交 三块最严重者（成交 invalid 不得显示“可用”）。"""
    from tests.p6_t2_env import build_view, status_of

    normal = _projection(tmp_path, "account_aggregate")
    base = build_view(normal)
    assert status_of(base, "sec-account")["word"] == "available"

    trades_bad = with_section(
        normal, "trades", forced_section_status("invalid", reason_code="authority_mismatch")
    )
    view = build_view(trades_bad)
    assert status_of(view, "sec-account")["word"] == "invalid"
    assert status_of(view, "sec-account")["reason"] == "authority_mismatch"
    assert 'data-status="invalid"' in section_slice(_html(trades_bad), "sec-account")

    nav_bad = with_section(
        normal, "nav", forced_section_status("stale", reason_code="nav_before_account_date")
    )
    view_nav = build_view(nav_bad)
    assert status_of(view_nav, "sec-account")["word"] == "stale"
    nav_section = section_slice(_html(nav_bad), "sec-account")
    assert fact_value(nav_section, "NAV 块状态（同区独立块）")
    # 状态可见性必须有断言（去掉 row=nav 会让状态小字消失，T2-R1-N9）
    assert "［陈旧 / nav_before_account_date］" in nav_section
    nav_fact = next(
        fact
        for section in view_nav["sections"]
        if section["id"] == "sec-account"
        for fact in section["facts"]
        if fact["label"] == "NAV 块状态（同区独立块）"
    )
    assert nav_fact["status"]["word"] == "stale"

    # stale 空表文案可渲染且不承诺“本表状态里有日期”（T2-R1-N8）
    stale_table = with_section(
        normal, "instructions", forced_section_status("stale", reason_code="no_source")
    )
    stale_table["instructions"].pop("instructions", None)
    stale_section = section_slice(_html(stale_table), "sec-instructions")
    assert "保留最近一次已核对值" in stale_section
    assert "原日期" not in stale_section


def test_p6_t2_27_instruction_table_has_status_and_empty_note(tmp_path: Path) -> None:
    """指令表必须带本表状态行（空表文案随状态词分派，不再指向不存在的状态）。"""
    empty = _projection(tmp_path, "instruction_table_empty", record_extra={"instructions_count": 0})
    section = section_slice(_html(empty), "sec-instructions")
    assert "本表状态 合法空" in section
    assert "已核实为空" in section

    missing = with_section(
        empty,
        "instructions",
        forced_section_status("not_available", reason_code="run_record_pre_switch_shape"),
    )
    missing["instructions"].pop("instructions", None)
    missing_section = section_slice(_html(missing), "sec-instructions")
    assert "本表状态 不可用" in missing_section
    assert "不伪造空表" in missing_section


def test_p6_t2_28_nav_break_survives_downsampling() -> None:
    """超过抽样上限时，未被抽到的核对失败点仍必须断开连线（T2-R3-02）。"""
    from src.lazybull.v2.reporting import svg

    total = svg.MAX_CHART_POINTS + 1  # 601：索引 100 不会被抽样保留
    series = [
        {"date": f"2026{(index % 12) + 1:02d}01", "value": float(index + 1),
         "matched": index != 100}
        for index in range(total)
    ]
    kept = svg.downsample_indices([row["value"] for row in series])
    assert 100 not in kept

    chart = svg.nav_chart(series)
    polylines = chart.count("<polyline")
    assert polylines >= 2, "失败点两侧必须分成两条折线，而不是被跨越连接"
    # 失败点本身未被抽样保留时，不必强行画 marker；但连线不得跨越
    assert chart.count('class="nav-point nav-point--broken"') == 0


def test_p6_t2_29_quality_conflict_status_is_visible(tmp_path: Path) -> None:
    """质量最新分区/同步水位的 invalid 必须显示为冲突，而不是“字段缺席”（T2-R3-05）。"""
    conflict = {
        "report_status": "invalid",
        "degradation_reason_code": "multiple_quality_value",
        "detail": "同一数据集摘要行出现多个不同值",
    }
    dataset = {
        "layer": "clean",
        "dataset": "daily",
        "metric_count": 3,
        "error_count": 0,
        "warning_count": 0,
        "partition_count": 2,
        "partition_count_status": {"report_status": "available"},
        "latest_partition": None,
        "latest_partition_status": dict(conflict),
        "sync_watermark": None,
        "sync_watermark_status": dict(conflict),
    }
    projection = _projection(tmp_path, "quality_conflict")
    quality = dict(projection["quality"])
    quality["datasets"] = [dataset]
    projection["quality"] = quality

    html = section_slice(_html(projection), "sec-model-quality")
    assert "multiple_quality_value" in html
    assert "无效" in html
    latest_label = fact_value(html, "质量最新分区")
    assert "multiple_quality_value" in latest_label
    assert "（字段缺席）" not in latest_label

    # 正常值仍按原样陈列，不被状态包装吞掉
    healthy = dict(dataset)
    healthy.update(
        {
            "latest_partition": "20260603",
            "latest_partition_status": {"report_status": "available"},
            "sync_watermark": "20260603",
            "sync_watermark_status": {"report_status": "available"},
        }
    )
    normal = _projection(tmp_path, "quality_normal")
    normal_quality = dict(normal["quality"])
    normal_quality["datasets"] = [healthy]
    normal["quality"] = normal_quality
    normal_html = section_slice(_html(normal), "sec-model-quality")
    assert "20260603" in fact_value(normal_html, "质量最新分区")
    assert "multiple_quality_value" not in normal_html


def test_p6_t2_30_source_crlf_renders_as_lf(tmp_path: Path) -> None:
    """来源文本含 CRLF/CR 时必须能生成，且页面只含 LF（T2-R3-06）。"""
    rows = [trade_row(reason="first line\r\nsecond line")]
    projection = _projection(
        tmp_path, "crlf_source", authority_trades=rows, trades_rows=rows
    )
    html = _html(projection)
    assert "\r" not in html
    assert "first line" in html
    assert "second line" in html

    single = [trade_row(reason="only\rCR line")]
    projection_cr = _projection(
        tmp_path, "cr_source", authority_trades=single, trades_rows=single
    )
    html_cr = _html(projection_cr)
    assert "\r" not in html_cr
    assert "only" in html_cr


def test_p6_t2_34_freshness_keeps_partial_source_conflict(tmp_path: Path) -> None:
    """存在正常数据集时，其他数据集的 invalid 仍要出现在新鲜度汇总里（T2-R3-05-R1）。"""
    conflict = {
        "report_status": "invalid",
        "degradation_reason_code": "multiple_quality_value",
        "detail": "同一数据集摘要行出现多个不同值",
    }
    healthy = {
        "layer": "clean",
        "dataset": "daily",
        "metric_count": 3,
        "error_count": 0,
        "warning_count": 0,
        "partition_count": 2,
        "partition_count_status": {"report_status": "available"},
        "latest_partition": "20260603",
        "latest_partition_status": {"report_status": "available"},
        "sync_watermark": "20260603",
        "sync_watermark_status": {"report_status": "available"},
    }
    broken = {
        "layer": "features",
        "dataset": "cs_train",
        "metric_count": 2,
        "error_count": 1,
        "warning_count": 0,
        "partition_count": None,
        "partition_count_status": dict(conflict),
        "latest_partition": None,
        "latest_partition_status": dict(conflict),
        "sync_watermark": None,
        "sync_watermark_status": dict(conflict),
    }
    projection = _projection(tmp_path, "quality_mixed")
    quality = dict(projection["quality"])
    quality["datasets"] = [healthy, broken]
    projection["quality"] = quality

    html = section_slice(_html(projection), "sec-model-quality")
    latest = fact_value(html, "质量最新分区")
    watermark = fact_value(html, "质量同步水位")
    for value in (latest, watermark):
        assert "20260603" in value
        assert "multiple_quality_value" in value
        assert "另有来源状态" in value

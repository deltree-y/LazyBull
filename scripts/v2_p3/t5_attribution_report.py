# -*- coding: utf-8 -*-
"""P3 T5 F11 数据影响归因——报告落盘（P3 规划 §7.5 / §13.3.7）。

JSON 为机器可读全量证据（``data/reports/v2_p3_data_impact_<RUN_ID>.json``）；
MD 为人读摘要（``docs/reports/v2_p3_data_impact_<date>.md``）——逐节结论 +
放行依据段落（h 节）+ 证据链接指针，数值细节一律指向 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from loguru import logger


def write_report_json(report: Mapping[str, Any], out_path: Path) -> Path:
    """全量机器可读报告落盘（父目录自动创建）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"归因数据 JSON 落盘: {out_path}")
    return out_path


def write_report_md(text: str, out_path: Path) -> Path:
    """人读摘要 MD 落盘（父目录自动创建）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"归因报告 MD 落盘: {out_path}")
    return out_path


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染人读摘要（七节 + 放行依据 + 证据链接；数值细节指向 JSON）。"""
    lines: list[str] = []
    _render_header(lines, report)
    _render_release_basis(lines, report)
    _render_section_a(lines, report)
    _render_section_b(lines, report)
    _render_keyed_section(lines, report, "c_trade_impact", "c. 成交影响（臂② vs 臂③ runs trades）")
    _render_keyed_section(lines, report, "d_holdings_impact", "d. 持仓影响（holdings_snapshot）")
    _render_section_e(lines, report)
    _render_section_f(lines, report)
    _render_section_g(lines, report)
    _render_evidence_links(lines, report)
    return "\n".join(lines) + "\n"


def _render_header(lines: list[str], report: Mapping[str, Any]) -> None:
    lines.append(f"# v2 P3 T5 数据影响归因报告（F11，{report['run_id']}）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 臂②（新实现 × cs_train 验证源）：`{report['arm2_dir']}`")
    lines.append(f"- 臂③（新实现 × panel）：`{report['arm3_dir']}`")
    lines.append(f"- B0' 冻结目录：`{report['b0p_dir']}`")
    lines.append(f"- 生产数据根：`{report['production_data_root']}`")
    lines.append(
        "- 口径声明：判定比较（臂③ vs B0'，f 节）与归因比较（臂③ vs 臂②，a~e 节）"
        "是两份独立证据，禁止混用；列级归因不直接当交易路径等价。"
    )
    lines.append("")


def _render_release_basis(lines: list[str], report: Mapping[str, Any]) -> None:
    """h 节：放行依据段落（逐门结论 + 判定表数值 + 归因闭合声明）。"""
    verdict = report["verdict"]
    section_a = report["sections"]["a_column_input_diffs"]
    section_f = report["sections"]["f_gdata_judgment_table"]
    section_g = report["sections"]["g_identity_appendix"]
    lines.append("## 放行依据（h 节）")
    lines.append("")
    lines.append(f"- **总判定：{verdict['verdict']}**")
    lines.append(
        f"- a 节归因闭合：**{'闭合' if section_a['closed'] else '未闭合'}**"
        f"（差异列 {section_a['diff_column_count']} 个，拒绝项 "
        f"{len(section_a['rejected_columns'])} 个；行集/结构/缺日破坏项 "
        f"{section_a['structural_break_count']} 个）"
    )
    lines.append(_release_basis_f_line(section_f))
    lines.append(
        "- g 节代码态一致断言：臂② source_digest == 臂③ source_digest ⇒ "
        f"**{'一致' if section_g['source_digest_match'] else '不一致'}**"
    )
    if verdict.get("errors"):
        lines.append("- 拒绝/前置错误：")
        lines.extend(f"  - {message}" for message in verdict["errors"])
    lines.append(
        "- 归因闭合声明：归因闭合 = a 节闭合（差异列 100% F11 词表分类、无拒绝项、"
        "完整覆盖：days_compared == 预期日数且非零）∧ 输出级一致性（R2-T5A-05："
        "规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应——任一折输出劣化须该折 a 节"
        "有已登记消费列差异）；词表唯一来源 `v2/store/reconcile_rules.py` + "
        "`panel_reconcile._classify_attrib_column`；未分类/越界即拒绝项并置总判定 FAIL。"
    )
    lines.append("")


def _release_basis_f_line(section_f: Mapping[str, Any] | None) -> str:
    """放行依据的判定表一行（判定表缺失时降级提示，不崩溃）。"""
    if section_f is None:
        return "- f 节 G-data 判定表：**不可用**（chain_metrics 读取/契约失败，见总判定错误）"
    deltas = section_f.get("deltas", {})
    closed = section_f["inputs"]["maxdd_attribution_closed"]
    return (
        "- f 节 G-data 判定表："
        f"ΔCAGR = {_pp(deltas.get('cagr'))}（±8pp 带，严格大于、等号带内）/ "
        f"ΔMaxDD = {_pp(deltas.get('max_drawdown'))}（零容忍 + 归因出口）/ "
        f"ΔSharpe = {_float_text(deltas.get('sharpe'))}（记录不设带）⇒ "
        f"**{section_f['verdict']}**（归因闭合标记 = {closed}，"
        "来源 = a 节闭合 ∧ 输出级一致性，R2-T5A-05）"
    )


def _pp(value: Any) -> str:
    return f"{float(value) * 100:+.2f}pp" if isinstance(value, (int, float)) else "NA"


def _float_text(value: Any) -> str:
    return f"{float(value):+.4f}" if isinstance(value, (int, float)) else "NA"


def _render_section_a(lines: list[str], report: Mapping[str, Any]) -> None:
    section = report["sections"]["a_column_input_diffs"]
    lines.append("## a. 逐折 × 消费列输入差异（归因比较：臂③ vs 臂② 的唯一变量 = 数据源）")
    lines.append("")
    lines.append("- 数值口径：atol=1e-6、NaN==NaN（`panel_reconcile._compare_day` 复用）")
    lines.append(
        "- 完整覆盖前置（R2-T5A-04）：预期日域 = trade_cal `is_open=1` ∩ boundary 窗口；"
        "`days_compared == 预期日数` 且非零才可闭合；窗口内非开市日显式登记 excluded 计数"
    )
    cache_stats = section.get("cache_stats") or {}
    day_stats = cache_stats.get("day_compare") or {}
    month_stats = cache_stats.get("panel_month") or {}
    lines.append(
        f"- 跨折单日结果缓存（P2-7/R1-04）：命中率 {day_stats.get('hit_rate', 0.0):.1%}"
        f"（hits={day_stats.get('hits')} / misses={day_stats.get('misses')}，"
        f"命名空间 `{day_stats.get('namespace')}`，格式版本 {day_stats.get('format_version')}），"
        f"目录 `{day_stats.get('cache_dir')}`"
    )
    lines.append(
        f"- panel 月帧 LRU 复用（P3-N4）：命中率 {month_stats.get('hit_rate', 0.0):.1%}"
        f"（hits={month_stats.get('hits')} / misses={month_stats.get('misses')}，"
        f"容量 {month_stats.get('capacity')}）"
    )
    lines.append(f"- 闭合：**{'是' if section['closed'] else '否'}**")
    registered = section.get("registered_structural_count", 0)
    if registered:
        lines.append(f"- 已登记结构差异：{registered} 列（CYQ5 / 2012 schema 演进白名单，非破坏）")
    for fold in section["folds"]:
        lines.append(
            f"- {fold['fold']}：消费列 {fold['consumed_column_count']} 个，"
            f"窗口 {fold['window']['outer_train'][0]}~{fold['window']['outer_train'][1]} ∪ "
            f"{fold['window']['test'][0]}~{fold['window']['test'][1]}，"
            f"预期日 {fold.get('expected_days_count')} / 比较 {fold['days_compared']} 日"
            f"（剔除非开市 {fold.get('excluded_non_open_days')} 日），"
            f"差异列 {len(fold['columns'])} 个，"
            f"折闭合 {'是' if fold['closed'] else '否'}"
        )
        for col, entry in fold["columns"].items():
            summary = entry["diff_day_summary"]
            lines.append(
                f"  - `{col}`：over {entry['over_rows']}/{entry['total_rows']} 行，"
                f"max|Δ|={entry['max_abs_diff']:.3e}，差异日 {summary['first']}~{summary['last']}"
                f"（{summary['days']} 日）⇒ {entry['attribution']}"
                f"{'（拒绝项）' if entry['boundary_violation'] else ''}"
            )
    lines.append("")


def _render_section_b(lines: list[str], report: Mapping[str, Any]) -> None:
    section = report["sections"]["b_prediction_impact"]
    lines.append("## b. 预测影响（逐折指纹 + 预测行键集/逐值对比）")
    lines.append("")
    lines.append(f"- {section['caliber_note']}")
    for fold in section["folds"]:
        lines.append(
            f"- {fold['fold']}：指纹 {'一致' if fold['fingerprint_match'] else '不一致'}，"
            f"n_rows {fold['n_rows_arm2']} vs {fold['n_rows_arm3']}"
            f"（Δ={fold['n_rows_delta']}），score_days 对称差 "
            f"{fold['score_days_sym_diff_count']} 日"
        )
        pred = fold.get("predictions")
        if pred is not None:
            lines.append(
                f"  - 预测行：键集对称差 {pred['key_sym_diff_count']} 个"
                f"（仅臂② {pred['keys_only_arm2']} / 仅臂③ {pred['keys_only_arm3']}，"
                f"同日换股候选日 {pred['same_day_swap_dates_count']} 个），"
                f"交集 {pred['intersection_keys']} 键，同键改分 {pred['value_diff_rows']} 行"
                f"（max|Δ|={_float_text(pred['value_max_abs_diff'])}）"
            )
        for arm_key, label in (("signal_topk_arm2", "臂②"), ("signal_topk_arm3", "臂③")):
            topk = fold.get(arm_key) or {}
            if topk.get("available"):
                lines.append(
                    f"  - {label} signal_topk：rows={topk['rows']}，days={topk['days']}，"
                    f"empty_days={topk['empty_days']}（展示登记，判定归 gates）"
                )
    lines.append("")


def _render_keyed_section(
    lines: list[str], report: Mapping[str, Any], section_key: str, title: str
) -> None:
    section = report["sections"][section_key]
    lines.append(f"## {title}")
    lines.append("")
    for fold in section["folds"]:
        diffs = ", ".join(
            f"{col} max|Δ|={entry['max_abs_diff']:.3e}"
            for col, entry in fold.get("value_max_abs_diff", {}).items()
        )
        dup2 = fold.get("duplicate_key_groups_arm2", 0)
        dup3 = fold.get("duplicate_key_groups_arm3", 0)
        dup_note = (
            f"，重复键分组 臂② {dup2} / 臂③ {dup3}（P3-3 键聚合口径）" if dup2 or dup3 else ""
        )
        lines.append(
            f"- {fold['fold']}：行数 {fold['rows_arm2']} vs {fold['rows_arm3']}，"
            f"键集对称差 {fold['key_sym_diff_count']} 个，交集 {fold['intersection_rows']} 行"
            f"{dup_note}（{diffs or '无数值列差'}）"
        )
    lines.append("")


def _render_section_e(lines: list[str], report: Mapping[str, Any]) -> None:
    section = report["sections"]["e_nav_impact"]
    chain = section["chain_nav"]
    lines.append("## e. 净值影响（chain_nav 逐日差 + 逐折 daily 汇总差）")
    lines.append("")
    if chain is None:
        lines.append("- 链式净值比较不可用（chain_nav 缺失/契约失败，见总判定错误）")
    else:
        lines.append(
            f"- 链式净值：max|Δnav| = {chain['max_abs_diff']:.3e}，"
            f"末折末日 Δnav = {_float_text(chain['final_fold_last_date_diff'])}"
        )
    for fold in section["daily_folds"]:
        diffs = ", ".join(
            f"{col} max|Δ|={entry['max_abs_diff']:.3e}"
            for col, entry in fold.get("value_max_abs_diff", {}).items()
        )
        lines.append(
            f"- {fold['fold']} daily：日域对称差 {fold['date_sym_diff_count']} 日，"
            f"共同日 {fold['common_days']} 日（{diffs or '无数值列差'}）"
        )
    lines.append("")


def _render_section_f(lines: list[str], report: Mapping[str, Any]) -> None:
    section = report["sections"]["f_gdata_judgment_table"]
    lines.append("## f. 链式指标判定表（G-data：臂③ vs B0'，判定比较）")
    lines.append("")
    if section is None:
        lines.append("- 判定表不可用（chain_metrics 读取/输入契约失败，见总判定错误）")
        lines.append("")
        return
    inputs = section["inputs"]
    lines.append(
        f"- 臂③：cagr={inputs['arm3']['cagr']}，max_drawdown={inputs['arm3']['max_drawdown']}，"
        f"sharpe={inputs['arm3']['sharpe']}"
    )
    lines.append(
        f"- B0'：cagr={inputs['b0p']['cagr']}，max_drawdown={inputs['b0p']['max_drawdown']}，"
        f"sharpe={inputs['b0p']['sharpe']}"
    )
    lines.append(f"- 判定：**{section['verdict']}**（{section['boundary_semantics']}）")
    lines.append(f"- 归因闭合并联接线：{section['wiring_note']}")
    for registration in section.get("registrations", []):
        lines.append(f"- 登记：{registration}")
    lines.append("")


def _render_section_g(lines: list[str], report: Mapping[str, Any]) -> None:
    section = report["sections"]["g_identity_appendix"]
    lines.append("## g. 身份附录（代码态 + 环境 + 数据态 + 消费快照当前态复核）")
    lines.append("")
    for arm_key, label in (("arm2", "臂②"), ("arm3", "臂③"), ("b0p", "B0'")):
        identity = section.get(arm_key)
        if not identity:
            continue
        lines.append(
            f"- {label}：source_digest=`{identity.get('source_digest')}`，"
            f"git={identity.get('git_commit')}（dirty={identity.get('git_dirty')}），"
            f"environment=`{identity.get('environment')}`，"
            f"data_state_id=`{identity.get('data_state_id')}`"
        )
        if arm_key != "b0p":
            lines.append(
                f"  - 消费快照 `{identity.get('consumption_snapshot')}`，当前态复核 "
                f"{'零漂移' if identity.get('consumption_snapshot_verified') else '未通过/未执行'}"
                f"（漂移 {identity.get('consumption_drift_count')} 项）"
            )
    script = section.get("script", {})
    lines.append(
        f"- 本脚本：source_digest=`{script.get('source_digest')}`，"
        f"git={script.get('git_commit')}（dirty={script.get('git_dirty')}）"
    )
    lines.append("")


def _render_evidence_links(lines: list[str], report: Mapping[str, Any]) -> None:
    lines.append("## 证据链接")
    lines.append("")
    lines.append(f"- 机器可读全量证据：`{report['evidence']['json_path']}`")
    lines.append(f"- 臂② runs 批次：`{report['evidence']['arm2_runs_batch']}`")
    lines.append(f"- 臂③ runs 批次：`{report['evidence']['arm3_runs_batch']}`")
    lines.append("")

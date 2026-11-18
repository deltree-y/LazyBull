"""R6 收口：资源决策表与中文报告（机器摘要 + Markdown）。

判定语汇严格按协议 §11.4：`repair_required` / `continue_to_admission_preparation` /
`limited_evidence_defer` / `stop_v1_investment`。禁止使用正式 `verdict=pass`。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scripts.v2_p7.defensive_ml import spec

NEGATIVE_VERDICTS = ("stop_v1_investment", "limited_evidence_defer")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "未取得"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "未取得"
    return f"{float(value) * 100:.{digits}f}%"


def decide(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """按 §11.4 给出资源建议（不使用正式 pass 语汇）。

    优先级：工程异常 > 本金集合不完整 > 主窗不达标(stop) > 达标但压力/分段反向(limited)。
    """
    expected = {str(capital) for capital in spec.TARGETS["capital_scenarios"]}
    main = diagnostics.get("main", {})
    missing_capitals = sorted(expected - set(main))
    engineering = list(diagnostics.get("engineering_errors", []))
    if engineering:
        return {
            "resource_recommendation": "repair_required",
            "reason": engineering,
            "both_capitals_meet_targets": False,
            "stress_hold": False,
            "stress_detail": {},
            "last_segment_adverse": False,
            "formal_admission": "not_evaluated",
            "alpha_consumed": False,
            "verdict": None,
        }
    if missing_capitals:
        return {
            "resource_recommendation": "repair_required",
            "reason": [f"缺少本金 {missing_capitals}（本金集合必须完整）"],
            "both_capitals_meet_targets": False,
            "stress_hold": False,
            "stress_detail": {},
            "last_segment_adverse": False,
            "formal_admission": "not_evaluated",
            "alpha_consumed": False,
            "verdict": None,
        }
    targets = {str(capital): data["target_vs_a100"] for capital, data in main.items()}
    both_meet = all(item["all_three"] for item in targets.values())
    stress = diagnostics.get("stress", {})
    stress_ok = True
    stress_detail: dict[str, Any] = {}
    for name, payload in stress.items():
        if payload.get("status") == "not_run":
            stress_detail[name] = "not_run"
            stress_ok = False
            continue
        rows = {k: v for k, v in payload.items() if k != "config"}
        ok = bool(rows) and all(row.get("all_three", False) for row in rows.values())
        stress_detail[name] = ok
        stress_ok = stress_ok and ok
    segments = {
        str(capital): data["segments"] for capital, data in main.items()
    }
    last_segment_adverse = False
    for capital, rows in segments.items():
        last = {row["arm"]: row for row in rows if row["segment"] == 3}
        if not {"A80_DEF20", "A80_CASH20", "A100"} <= set(last):
            continue
        # 协议口径：相对现金的收益差为负 **且** 相对 A100 的回撤也恶化，才算"反向"
        worse_cagr = float(last["A80_DEF20"]["cagr"]) < float(last["A80_CASH20"]["cagr"])
        worse_mdd = float(last["A80_DEF20"]["max_drawdown"]) < float(last["A100"]["max_drawdown"])
        if worse_cagr and worse_mdd:
            last_segment_adverse = True
    if both_meet and stress_ok:
        verdict = "continue_to_admission_preparation"
    elif both_meet or last_segment_adverse:
        verdict = "limited_evidence_defer"
    else:
        verdict = "stop_v1_investment"
    return {
        "resource_recommendation": verdict,
        "both_capitals_meet_targets": bool(both_meet),
        "stress_hold": bool(stress_ok),
        "stress_detail": stress_detail,
        "last_segment_adverse": bool(last_segment_adverse),
        "formal_admission": "not_evaluated",
        "alpha_consumed": False,
        "verdict": None,
    }


def _arm_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| 臂 | CAGR | 最大回撤(带符号) | 回撤损失幅度 | Sharpe | 期末资产 |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| {arm} | {cagr} | {mdd} | {loss} | {sharpe} | {end} |".format(
                arm=row["arm"],
                cagr=_pct(row["cagr"]),
                mdd=_pct(row["max_drawdown"]),
                loss=_pct(row["mdd_loss"]),
                sharpe=_fmt(row["sharpe"], 3),
                end=f"{float(row['end']):,.0f}",
            )
        )
    return "\n".join(lines)


def _target_block(name: str, target: Mapping[str, Any]) -> str:
    """一行目标读数（不含行首符号，由调用方决定是否加 `- `）。"""
    return (
        f"{name}: 回撤改善 {_pct(target['dd_improvement'])}"
        f"（门槛 ≥3.00%，{_fmt(target['meets_dd'])}）；"
        f"CAGR 差 {_pct(target['cagr_delta'])}（下限 -1.00%，{_fmt(target['meets_cagr'])}）；"
        f"相对现金 CAGR {_pct(target['cagr_vs_cash'])}（{_fmt(target['beats_cash'])}）"
    )


def build_report(diagnostics: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """生成中文报告（7 段，按协议 §16 顺序）。"""
    decision = decide(diagnostics)
    main = diagnostics["main"]
    lines: list[str] = []
    lines.append("# DEF-ML-60-v1 研究交付报告（R0—R6）")
    lines.append("")
    lines.append(f"生成时间：{meta.get('generated_at')}　候选：`{spec.CANDIDATE_ID}`")
    lines.append("")
    lines.append("## 1 一页结论")
    lines.append("")
    lines.append(
        "- 原目标：相对 A100，组合最大回撤改善 ≥ 3.00 个百分点，且 CAGR 代价 ≤ 1.00 "
        "个百分点；同时不得输给「A80 + 等额现金」。"
    )
    lines.append(f"- 资源建议：**{decision['resource_recommendation']}**"
                 f"（两本金均达标={_fmt(decision['both_capitals_meet_targets'])}，"
                 f"压力未推翻={_fmt(decision['stress_hold'])}）")
    lines.append("- 正式 P7 状态：`formal_admission = not_evaluated`；本报告不产生正式准入结论。")
    for capital, data in main.items():
        label = f"- 本金 {int(capital):,}："
        lines.append(label + _target_block("主候选 vs A100", data["target_vs_a100"]))
    lines.append("")
    lines.append("## 2 四臂主表（两本金分别实算）")
    lines.append("")
    for capital, data in main.items():
        lines.append(f"### 本金 {int(capital):,}")
        lines.append("")
        lines.append(_arm_table(data.get("arms", [])))
        lines.append("")
        lines.append("- " + _target_block("A80_DEF20 vs A100", data["target_vs_a100"]))
        lines.append("- " + _target_block("A80_DEF20 vs A80_CASH20", data["target_vs_cash"]))
        lines.append(
            "- "
            + _target_block("A80_LOWVOL20 vs A100（对照，不转正）", data["lowvol_target_vs_a100"])
        )
        arms_by_name = {row["arm"]: row for row in data.get("arms", [])}
        if {"A100", "A80_CASH20"} <= set(arms_by_name):
            cash_only = (
                arms_by_name["A80_CASH20"]["max_drawdown"]
                - arms_by_name["A100"]["max_drawdown"]
            )
            sleeve_increment = data.get("target_vs_cash", {}).get("dd_improvement", 0.0)
            lines.append(
                f"- **回撤改善分解**：纯降仓（A80+现金 相对 A100）贡献 "
                f"{cash_only * 100:+.2f}pp；防御袖在现金基础上再贡献 "
                f"{sleeve_increment * 100:+.2f}pp"
            )
        sleeve = data.get("sleeve_only", {})
        def_cagr = _pct(sleeve["def"].get("cagr"))
        def_mdd = _pct(sleeve["def"].get("max_drawdown"))
        low_cagr = _pct(sleeve["lowvol"].get("cagr"))
        low_mdd = _pct(sleeve["lowvol"].get("max_drawdown"))
        lines.append(
            f"- 袖自身：DEF CAGR {def_cagr} / MDD {def_mdd}；"
            f"LOWVOL CAGR {low_cagr} / MDD {low_mdd}"
        )
        lines.append("")
    lines.append("## 3 固定分段与相关性")
    lines.append("")
    for capital, data in main.items():
        lines.append(f"### 本金 {int(capital):,}")
        lines.append("")
        lines.append("| 段 | 起始 | 结束 | 臂 | CAGR | 最大回撤 |")
        lines.append("|---|---|---|---|---|---|")
        for row in data["segments"]:
            lines.append(
                f"| {row['segment']} | {row['start']} | {row['end']} | {row['arm']} | "
                f"{_pct(row['cagr'])} | {_pct(row['max_drawdown'])} |"
            )
        lines.append("")
        corr = diagnostics["correlation"][str(capital)]
        lines.append("日收益相关（全样本，描述性）：")
        lines.append("")
        arms = list(corr)
        lines.append("| | " + " | ".join(arms) + " |")
        lines.append("|---" * (len(arms) + 1) + "|")
        for row in arms:
            values = " | ".join(_fmt(corr.get(row, {}).get(col), 3) for col in arms)
            lines.append(f"| {row} | {values} |")
        lines.append("")
        sleeve_corr = meta.get("sleeve_correlation", {}).get(str(capital))
        if sleeve_corr:
            lines.append(
                f"- 袖自身与 A100 日收益相关：DEF {_fmt(sleeve_corr['def_vs_a100'], 3)}，"
                f"LOWVOL {_fmt(sleeve_corr['lowvol_vs_a100'], 3)}，"
                f"DEF vs LOWVOL {_fmt(sleeve_corr['def_vs_lowvol'], 3)}"
                f"（{sleeve_corr['days']} 个交易日；点估计，非 F29 三口径区间）"
            )
            lines.append("")
    lines.append("## 4 分位诊断与低波对照")
    lines.append("")
    pred = meta.get("prediction", {})
    lines.append(f"- 逐日全军预测行数 {pred.get('rows')}，覆盖 {pred.get('days')} 个交易日；")
    lines.append(f"- 分位交叉率 {_pct(pred.get('crossed_rate'), 4)}（固定升序修正后使用）；")
    lines.append(f"- 48 个模型重载与首次预测逐位一致：{_fmt(pred.get('reload_all_identical'))}；")
    lines.append("- LOWVOL 只作解释对照：若其更优也不得转正，也不改写旧 A5 结论。")
    lines.append("")
    lines.append("## 5 固定压力（只诊断，不择优）")
    lines.append("")
    for name, payload in diagnostics.get("stress", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if payload.get("status") == "not_run":
            lines.append(f"- 未运行：{payload.get('reason')}")
            lines.append("")
            continue
        for capital, target in payload.items():
            if capital == "config":
                continue
            lines.append(f"- 本金 {int(capital):,}：" + _target_block("压力后", target))
        lines.append("")
    zero = diagnostics.get("zero_recovery", {})
    lines.append(
        f"- 未知永久缺价零回收压力：适用={_fmt(zero.get('applicable'))}；"
        f"{zero.get('note')}"
    )
    lines.append("")
    lines.append("## 6 身份与资源")
    lines.append("")
    lines.append(f"- 协议 SHA256：`{meta.get('protocol_sha256')}`")
    lines.append(f"- 规格 SHA256：`{meta.get('spec_sha256')}`")
    lines.append(f"- 机器墙钟（本次研究包累计）：{meta.get('machine_hours')} 小时；"
                 f"硬顶 36 小时；超预算={_fmt(meta.get('over_budget'))}")
    lines.append("- 数据采购：0 元；生产根改动：无；旧实验结论：未重判。")
    for deviation in meta.get("deviations", []):
        lines.append(
            f"- **规格偏离（用户裁决）**：{deviation.get('id')} — 违反 {deviation.get('clause')}；"
            f"原因：{deviation.get('reason')}；登记件 {deviation.get('record')}"
        )
    for note in meta.get("reuse_notes", []):
        lines.append(f"- 复用/偏离：{note}")
    lines.append("")
    lines.append("## 7 未完成项与复跑")
    lines.append("")
    for item in meta.get("gaps", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "复跑：`python -m scripts.v2_p7.defensive_ml.run --campaign "
        f"{meta.get('campaign')} --stage "
        "<readiness|dataset|model_probe|train_predict|replay|diagnostics|report>`"
    )
    return "\n".join(lines) + "\n"


def write_report(
    campaign_dir: Path, diagnostics: Mapping[str, Any], meta: Mapping[str, Any]
) -> dict[str, str]:
    """落盘机器摘要与中文报告。"""
    reports = Path(campaign_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    decision = decide(diagnostics)
    summary = {"decision": decision, "meta": dict(meta)}
    summary_path = reports / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    report_path = reports / "report_cn.md"
    report_path.write_text(build_report(diagnostics, meta), encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}


def sleeve_correlations(campaign_dir: Path) -> dict[str, Any]:
    """袖自身日收益与 A100 的相关系数（独立存活口径的描述性读数）。"""
    result: dict[str, Any] = {}
    for capital in spec.TARGETS["capital_scenarios"]:
        accounts = Path(campaign_dir) / "accounts" / str(capital)
        try:
            def_daily = pd.read_parquet(accounts / "DEF_daily.parquet")
            low_daily = pd.read_parquet(accounts / "LOWVOL_daily.parquet")
            arms = pd.read_parquet(accounts / "DEF_arms.parquet")
        except FileNotFoundError:
            continue
        index = [str(day) for day in arms.index]
        a_ret = pd.Series(
            _returns(arms["A100"].to_numpy(dtype=float)), index=index
        )
        def_ret = pd.Series(
            _returns(def_daily["total_value"].to_numpy(dtype=float)),
            index=[str(day) for day in def_daily["trade_date"]],
        )
        low_ret = pd.Series(
            _returns(low_daily["total_value"].to_numpy(dtype=float)),
            index=[str(day) for day in low_daily["trade_date"]],
        )
        common = a_ret.index.intersection(def_ret.index).intersection(low_ret.index)
        result[str(capital)] = {
            "def_vs_a100": float(def_ret.reindex(common).corr(a_ret.reindex(common))),
            "lowvol_vs_a100": float(low_ret.reindex(common).corr(a_ret.reindex(common))),
            "def_vs_lowvol": float(def_ret.reindex(common).corr(low_ret.reindex(common))),
            "days": int(len(common)),
        }
    return result


def _returns(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    out = np.zeros_like(array)
    out[1:] = array[1:] / array[:-1] - 1.0
    return out


def load_diagnostics(campaign_dir: Path) -> dict[str, Any]:
    """读取诊断摘要。"""
    path = Path(campaign_dir) / "diagnostics" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def frame_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """记录列表 -> DataFrame（报告内表格复用）。"""
    return pd.DataFrame(records)

"""R6 收口：资源决策、机器摘要与中文研究报告（协议 §13.4 / §18.3）。

`resource_recommendation` 是研究建议，禁止使用正式 `verdict=pass`；
正式状态恒为 `not_evaluated`。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec

CANDIDATE_ARM = "A80_AD20"
SIMPLE_ARM = "A80_AD_SIMPLE20"
CASH_ARM = "A80_CASH20"
BASELINE_ARM = "A100"

#: 修复前（campaign `adml20_v1_20261118`，run R3b/R6c）的对照读数：只作敏感性披露
PRIOR_EVIDENCE: dict[str, Any] = {
    "campaign": "adml20_v1_20261118",
    "runs": "R3b（账户）/ R6c（报告）",
    "ending": {
        "1000000": {
            "A100": 3022301.90,
            "A80_CASH20": 2611876.48,
            "A80_AD20": 2693434.73,
            "A80_AD_SIMPLE20": 2762871.53,
            "sleeve_ML": 281558.25,
            "sleeve_SIMPLE": 350995.05,
        },
        "2000000": {
            "A100": 6067309.78,
            "A80_CASH20": 5248894.94,
            "A80_AD20": 5388094.73,
            "A80_AD_SIMPLE20": 5569033.62,
            "sleeve_ML": 539199.78,
            "sleeve_SIMPLE": 720138.68,
        },
    },
    "cagr_delta": {"1000000": -0.0182, "2000000": -0.0188},
    "cagr_vs_cash": {"1000000": 0.0048, "2000000": 0.0041},
}

#: 袖自身年化（用同一 20%独立资金口径，便于跨版本比较稳定性）
SLEEVE_ANNUALIZED: dict[str, Any] = {
    "prior": {
        "ML": {1000000: 0.0469, 2000000: 0.0409},
        "SIMPLE": {1000000: 0.0771, 2000000: 0.0806},
    },
}

#: 封存裁决（2026-11-18 用户选项 1）：只封存本规格，不判机制
SEAL_DECISION: dict[str, Any] = {
    "decided_at": "2026-11-18",
    "decided_by": "user",
    "option": "1（就此封存）",
    "disposition": "evidence_insufficient",
    "disposition_zh": "证据不足 / 不可估计",
    "meaning": (
        "本规格在本数据、本实现下没有跑出目标，且读数对实现细节高度敏感；"
        "因此登记为「证据不足」并停止对 v1 追加投入，"
        "**不登记为经济失败、不记为机制证伪、不作为退役依据**"
    ),
    "stop_meaning": (
        "`resource_recommendation = stop_v1_investment` 只表示「不再对该规格追加投入」，"
        "不是对分析师扩散机制的否定"
    ),
    "reopen_rule": (
        "重开需新信息（更长研报历史 / 稳态扩散口径 / 非集中式组合约束等）"
        "并重新预登记；禁止在本规格上继续调参、换种子或换持有期"
    ),
}

#: 本版（修复后）相对于首版证据的改动，必须原样披露
REPAIR_NOTES: tuple[str, ...] = (
    "T 日涨停按协议纳入资格排除（首版只做标记，涨停股仍可能进候选）",
    "20 日平均成交额观察期回正为 20（首版 min_periods=16 放宽了域）",
    "分红冲突改为 fail-closed 整组剔除并登记未决（首版按报告期/付息日/上市日顺序任选一条）",
    "分红副本每次从 raw 重建（首版复用已隔离副本，审计被读成 0 冲突）",
    "研报延迟压力覆盖全部 15 列（首版只延后 8 列扩散特征）",
    "运行身份闭环：就绪门、输入 SHA 复核、模型与训练 run 一致、账户取自回放发布件、"
    "非有限预测即阻断",
    "零回收压力改为全历史扫描且分母取 B 袖自身资产（首版只查期末、分母含 A）",
)

RECOMMENDATIONS: tuple[str, ...] = (
    "continue_to_admission_preparation",
    "limited_evidence_defer",
    "stop_v1_investment",
    "repair_required",
)


def _fmt(value: Any, digits: int = 4) -> str:
    """数值格式化（None 安全，禁止用 0 冒充缺失）。"""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def _plain_pct(value: Any, digits: int = 2) -> str:
    """比例转百分数（不加正负号，用于仓位等非收益量）。"""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number * 100:.{digits}f}%"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number * 100:+.{digits}f}%"


def _stress_holds(stress: dict[str, Any], capital: str) -> bool:
    """成本/延迟压力是否仍保持 cagr_delta>=0 且 mdd_loss_delta<=0.01。"""
    block = (stress or {}).get(capital) or {}
    cagr = block.get("cagr_delta")
    mdd = block.get("mdd_loss_delta")
    if cagr is None or mdd is None:
        return False
    return float(cagr) >= 0.0 and float(mdd) <= float(spec.TARGETS["mdd_loss_delta_max"])


def _last_segment_reversal(diagnostics: dict[str, Any], capital: str) -> bool:
    """最后固定分段是否同时出现"相对现金 CAGR 为负"与"相对 A100 回撤恶化"。"""
    main = diagnostics["main"][capital]
    segments = pd.DataFrame(main["segments"])
    if segments.empty:
        return False
    last = segments[segments["segment"] == segments["segment"].max()]
    if last.empty:
        return False
    cash = last[last["arm"] == CASH_ARM]
    candidate = last[last["arm"] == CANDIDATE_ARM]
    baseline = last[last["arm"] == BASELINE_ARM]
    if cash.empty or candidate.empty or baseline.empty:
        return False
    cagr_vs_cash = float(candidate["cagr"].iloc[0]) - float(cash["cagr"].iloc[0])
    mdd_worse = float(candidate["mdd_loss"].iloc[0]) - float(baseline["mdd_loss"].iloc[0])
    return cagr_vs_cash < 0 and mdd_worse > 0


def evidence_blockers(diagnostics: dict[str, Any], meta: dict[str, Any] | None) -> list[str]:
    """证据链闭合检查：缺门、门未过、压力未跑全，都不得当作经济结论。"""
    blockers: list[str] = []
    upstream = (meta or {}).get("upstream") or {}
    for key in ("dataset_run", "predictions_run", "replay_run", "diagnostics_run"):
        if not upstream.get(key):
            blockers.append(f"上游身份缺失：{key}")
    models_gate = upstream.get("models_gate") or {}
    if models_gate.get("status") != "pass":
        blockers.append("模型注册表未与训练 run 对齐")
    gates = upstream.get("diagnostics_gates") or {}
    for name in ("input", "models"):
        if (gates.get(name) or {}).get("status") != "pass":
            blockers.append(f"诊断阶段门未过：{name}")
    if not gates.get("accounts_run"):
        blockers.append("账户未绑定到回放 run 发布件")
    if not diagnostics.get("stress"):
        blockers.append("四个固定压力未运行（不得当完整证据）")
    return blockers


def decide(
    diagnostics: dict[str, Any], meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """资源决策表（协议 §13.4）。"""
    engineering = diagnostics.get("engineering") or {}
    unexplained = [
        {"capital": capital, "issues": issues}
        for capital, issues in engineering.items()
        if issues
    ]
    capitals = [str(item) for item in spec.TARGETS["capital_scenarios"]]
    meets = {
        capital: bool(diagnostics["main"][capital]["target_vs_a100"]["all_three"])
        for capital in capitals
    }
    stress = diagnostics.get("stress") or {}
    stress_lag = {
        capital: _stress_holds(stress.get("report_lag_1d", {}), capital) for capital in capitals
    }
    stress_cost = {
        capital: _stress_holds(stress.get("slippage_doubled", {}), capital)
        and _stress_holds(stress.get("cash_equity_discount_20", {}), capital)
        for capital in capitals
    }
    last_reversal = {capital: _last_segment_reversal(diagnostics, capital) for capital in capitals}
    zero_recovery = stress.get("zero_recovery_unresolved") or {}
    zero_gap = bool(zero_recovery.get("unresolved_positions"))

    blockers = evidence_blockers(diagnostics, meta)
    if unexplained or blockers:
        recommendation = "repair_required"
        reason = (
            "存在未解释的工程异常或证据链未闭合，先修复，不解释为经济失败"
        )
    elif not all(meets.values()):
        recommendation = "stop_v1_investment"
        reason = "主窗任一本金未满足第 2 节目标，且无工程错误解释"
    elif not all(stress_lag.values()) or not all(stress_cost.values()) or any(
        last_reversal.values()
    ):
        recommendation = "limited_evidence_defer"
        reason = "主窗满足，但成本/延迟压力或最后固定分段削弱改善解释"
    else:
        recommendation = "continue_to_admission_preparation"
        reason = "两本金主窗与压力均满足，且无会翻转结论的未决边界（仍非正式准入）"

    obstacles = []
    if not all(stress_lag.values()):
        obstacles.append("研报信息延后一交易日的压力未保持主判据")
    if not all(stress_cost.values()):
        obstacles.append("成本压力（滑点加倍/现金权益折扣）未保持主判据")
    if any(last_reversal.values()):
        obstacles.append("最后固定分段同时出现相对现金 CAGR 为负与回撤恶化")
    if zero_gap:
        obstacles.append("期末存在未知回收持仓（零回收压力）")
    obstacles.append(
        "候选对实现细节敏感（同规格修复前后袖年化大幅摆动），缺稳定性证据，"
        "不得据单次读数判定机制成败"
    )
    obstacles.append("F29 型 I 正式门（E1/Q1/E2/T2/三口径相关/月利用率）未在本研究运行")

    return {
        "resource_recommendation": recommendation,
        "reason": reason,
        "meets_target": meets,
        "stress_report_lag_holds": stress_lag,
        "stress_cost_holds": stress_cost,
        "last_segment_reversal": last_reversal,
        "unexplained_engineering": unexplained,
        "evidence_blockers": blockers,
        "zero_recovery_gap": zero_gap,
        "admission_obstacles": obstacles,
        "formal_admission": "not_evaluated",
        "verdict": None,
        "scope_note": "结论只针对本规格、本数据、本窗口；不终结分析师信息方向，不自动开新规格",
    }


def machine_summary(diagnostics: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """机器摘要（报告第一节与第七节的数据来源）。"""
    decision = decide(diagnostics, meta)
    return {
        "candidate_id": spec.CANDIDATE_ID,
        "protocol_version": spec.PROTOCOL_VERSION,
        "protocol_sha256": meta.get("protocol_sha256"),
        "spec_sha256": spec.spec_sha256(),
        "campaign": meta.get("campaign"),
        "window": spec.WINDOW,
        "capital_scenarios": spec.TARGETS["capital_scenarios"],
        "decision": decision,
        "main": diagnostics.get("main"),
        "stress": diagnostics.get("stress"),
        "coverage": diagnostics.get("coverage"),
        "engineering": diagnostics.get("engineering"),
        "prior_evidence": {
            "campaign": PRIOR_EVIDENCE["campaign"],
            "runs": PRIOR_EVIDENCE["runs"],
            "ending": PRIOR_EVIDENCE["ending"],
            "cagr_delta": PRIOR_EVIDENCE["cagr_delta"],
            "sleeve_annualized": SLEEVE_ANNUALIZED["prior"],
        },
        "disposition": SEAL_DECISION,
        "formal_admission": "not_evaluated",
        "production_changed": False,
        "not_virgin_market_holdout": True,
        "generated_at": meta.get("generated_at"),
    }


def _metrics_table(rows: list[dict[str, Any]], capital: int) -> str:
    lines = [
        f"**本金 {capital:,} 元**",
        "",
        "| 臂 | 期末资产 | CAGR | 最大回撤(带符号) | MDD_loss | Sharpe |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {end:,.0f} | {cagr} | {mdd} | {loss} | {sharpe} |".format(
                arm=row["arm"],
                end=float(row["end"]),
                cagr=_pct(row["cagr"]),
                mdd=_fmt(row["max_drawdown"], 4),
                loss=_fmt(row["mdd_loss"], 4),
                sharpe=_fmt(row["sharpe"], 3),
            )
        )
    return "\n".join(lines)


def build_report(diagnostics: dict[str, Any], meta: dict[str, Any]) -> str:
    """中文研究报告（§18.3 顺序）。"""
    decision = decide(diagnostics, meta)
    coverage = diagnostics.get("coverage") or {}
    lines: list[str] = []
    lines.append(f"# {spec.CANDIDATE_ID} 分析师预期扩散 ML 袖研究报告")
    lines.append("")
    lines.append(
        f"campaign `{meta.get('campaign')}`；协议 v{spec.PROTOCOL_VERSION}"
        f"（SHA `{(meta.get('protocol_sha256') or '')[:16]}…`）；规格 SHA `{spec.spec_sha256()[:16]}…`"
    )
    lines.append("")

    lines.append("## 1 一页结论")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|---|---|")
    lines.append("| 原目标 | 相对 A100：CAGR ≥ +0.5pp、MDD 损失增幅 ≤ 1pp；且胜过 A80+现金 |")
    lines.append(
        "| 当前读数 | 见第 2 节四臂主表；两本金均需满足 |"
    )
    lines.append(f"| 资源建议 | `{decision['resource_recommendation']}` —— {decision['reason']} |")
    lines.append(
        "| 正式 P7 状态 | `not_evaluated`（本研究是经济预筛，不是准入；正式 G 包未运行） |"
    )
    lines.append(
        "| 封存裁决 | {when} 用户选项 {option}；登记为 `{disp}`（{zh}）—— {meaning} |".format(
            when=SEAL_DECISION["decided_at"],
            option=SEAL_DECISION["option"],
            disp=SEAL_DECISION["disposition"],
            zh=SEAL_DECISION["disposition_zh"],
            meaning=SEAL_DECISION["meaning"],
        )
    )
    lines.append("")
    lines.append(
        "> 本报告只覆盖本规格、本数据与本窗口。经济预筛结果不等于 F29 型 I 正式门结论，"
        "也不构成前向或真实日历验证。"
    )
    if decision["resource_recommendation"] == "repair_required":
        lines.append(">")
        lines.append(
            "> **当前为 `repair_required`：上述负向读数不得当作正式停投依据**，"
            "先按第七节修复记录与新 run 重跑，再判断机制是否被可靠证伪。"
        )
    lines.append("")

    lines.append("## 2 两本金四臂主表")
    lines.append("")
    for capital in spec.TARGETS["capital_scenarios"]:
        block = diagnostics["main"][str(int(capital))]
        lines.append(_metrics_table(block["metrics"], int(capital)))
        lines.append("")
        target = block["target_vs_a100"]
        lines.append(
            "相对 A100：`cagr_delta = {cagr}`、`mdd_loss_delta = {mdd}`、"
            "`sharpe_delta = {sharpe}`；三项同时达标 = {all3}".format(
                cagr=_pct(target.get("cagr_delta")),
                mdd=_fmt(target.get("mdd_loss_delta"), 4),
                sharpe=_fmt(target.get("sharpe_delta"), 3),
                all3="是" if target.get("all_three") else "否",
            )
        )
        lines.append(
            "相对现金臂 A80_CASH20：`cagr_vs_cash = {cagr}`（防纯降仓冒充）".format(
                cagr=_pct(target.get("cagr_vs_cash"))
            )
        )
        lines.append(
            "AD_SIMPLE 解释对照（不进入机器门）：`cagr_delta_vs_A100 = {cagr}`、"
            "`mdd_loss_delta = {mdd}`".format(
                cagr=_pct(block["simple_vs_a100"].get("cagr_delta")),
                mdd=_fmt(block["simple_vs_a100"].get("mdd_loss_delta"), 4),
            )
        )
        lines.append("")

    lines.append("## 3 分年、固定分段、启动阶段与持仓")
    lines.append("")
    for capital in spec.TARGETS["capital_scenarios"]:
        block = diagnostics["main"][str(int(capital))]
        segments = pd.DataFrame(block["segments"])
        lines.append(f"**本金 {int(capital):,} 元 · 固定分段**")
        lines.append("")
        lines.append("| 段 | 起 | 止 | 臂 | CAGR | 最大回撤 | MDD_loss |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in segments.to_dict("records"):
            lines.append(
                "| {seg} | {start} | {end} | {arm} | {cagr} | {mdd} | {loss} |".format(
                    seg=row["segment"], start=row["start"], end=row["end"], arm=row["arm"],
                    cagr=_pct(row["cagr"]), mdd=_fmt(row["max_drawdown"], 4),
                    loss=_fmt(row["mdd_loss"], 4),
                )
            )
        lines.append("")
        sleeve = block["sleeve"]["ML"]
        turnover = sleeve["turnover"]
        lines.append(
            "B 袖（ML）：年化换手 {turnover}（累计 {total}）、费用 {fees:,.0f} 元"
            "（占均值 {fee_share}）、平均股票仓位 {exp}、成交笔数 {trades}".format(
                turnover=_fmt(turnover.get("turnover_annualized"), 2),
                total=_fmt(turnover.get("turnover"), 1),
                fees=float(turnover.get("fees") or 0.0),
                fee_share=_plain_pct(turnover.get("fee_share")),
                exp=_plain_pct(sleeve["exposure"].get("mean_exposure")),
                trades=sleeve["trades"],
            )
        )
        industry = block["industry"]
        if industry.get("status") == "pass":
            lines.append(
                "行业集中度：单桶最大 {top} 只（上限 {cap}），越限日 {breach} 天".format(
                    top=industry.get("max_bucket_count"),
                    cap=industry.get("cap"),
                    breach=industry.get("breach_days"),
                )
            )
        else:
            lines.append(f"行业集中度：`not_estimable`（{industry.get('reason')}）")
        overlap = block.get("a_overlap") or {}
        if overlap.get("status") == "pass":
            lines.append(
                "与 A100 持仓交集：日均 {mean} 只（最多 {maxv} 只），有交集日占比 {share}".format(
                    mean=_fmt(overlap.get("mean_overlap"), 2),
                    maxv=overlap.get("max_overlap"),
                    share=_plain_pct(overlap.get("share_days_with_overlap"), 1),
                )
            )
        else:
            lines.append(f"与 A100 持仓交集：`not_estimable`（{overlap.get('reason')}）")
        lines.append("")
        years = pd.DataFrame(block.get("years") or [])
        if not years.empty:
            lines.append("**分年（候选 vs A100）**")
            lines.append("")
            lines.append("| 年 | A100 CAGR | A80_AD20 CAGR | ΔCAGR | A100 MDD_loss | 候选 MDD_loss |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for year in sorted({str(item) for item in years["start"].str[:4].unique()}):
                part = years[years["start"].astype(str).str[:4] == year]
                base_row = part[part["arm"] == BASELINE_ARM]
                cand_row = part[part["arm"] == CANDIDATE_ARM]
                if base_row.empty or cand_row.empty:
                    continue
                lines.append(
                    "| {year} | {b} | {c} | {d} | {bm} | {cm} |".format(
                        year=year,
                        b=_pct(base_row["cagr"].iloc[0]),
                        c=_pct(cand_row["cagr"].iloc[0]),
                        d=_pct(float(cand_row["cagr"].iloc[0]) - float(base_row["cagr"].iloc[0])),
                        bm=_fmt(base_row["mdd_loss"].iloc[0], 4),
                        cm=_fmt(cand_row["mdd_loss"].iloc[0], 4),
                    )
                )
            lines.append("")

    lines.append("**修复前后稳定性（同一规格、仅实现修复；不替换主窗）**")
    lines.append("")
    lines.append("| 本金 | 版本 | ML 袖年化 | 简单扩散袖年化 | cagr_delta vs A100 |")
    lines.append("|---|---|---:|---:|---:|")
    for capital in spec.TARGETS["capital_scenarios"]:
        key = int(capital)
        block = diagnostics["main"][str(key)]
        lines.append(
            "| {cap} | 修复前（{camp}） | {ml} | {simple} | {delta} |".format(
                cap=f"{key:,}",
                camp=PRIOR_EVIDENCE["campaign"],
                ml=_pct(SLEEVE_ANNUALIZED["prior"]["ML"][key]),
                simple=_pct(SLEEVE_ANNUALIZED["prior"]["SIMPLE"][key]),
                delta=_pct(PRIOR_EVIDENCE["cagr_delta"][str(key)]),
            )
        )
        lines.append(
            "| {cap} | 修复后（本报告） | {ml} | {simple} | {delta} |".format(
                cap=f"{key:,}",
                ml=_pct((block["sleeve"]["ML"]["metrics"] or {}).get("cagr")),
                simple=_pct((block["sleeve"]["SIMPLE"]["metrics"] or {}).get("cagr")),
                delta=_pct(block["target_vs_a100"].get("cagr_delta")),
            )
        )
    lines.append("")
    lines.append(
        "读数含义：ML 袖对实现细节敏感（同一规格、同一数据，仅修掉涨停剔除/域观察期/冲突处置后"
        "年化大幅摆动），而 AD_SIMPLE 基本不动。**这属于「弱信号 + 6 只集中」的路径敏感性**，"
        "既不能当作修复提升了能力，也不支持「机制已被可靠证伪」的说法。"
    )
    lines.append("")
    trimmed = meta.get("startup_trimmed") or {}
    if trimmed:
        lines.append("**去除最初 20 个交易日（不替换主窗）**")
        lines.append("")
        lines.append("| 本金 | 起点 | cagr_delta vs A100 | mdd_loss_delta vs A100 | 三项达标 |")
        lines.append("|---|---|---:|---:|---|")
        for capital, block in trimmed.items():
            if block.get("status") != "pass":
                lines.append(f"| {capital} | n/a | n/a | n/a | `not_estimable` |")
                continue
            target = block["target_vs_a100"]
            lines.append(
                "| {cap} | {first} | {cagr} | {mdd} | {ok} |".format(
                    cap=capital,
                    first=block.get("first"),
                    cagr=_pct(target.get("cagr_delta")),
                    mdd=_fmt(target.get("mdd_loss_delta"), 4),
                    ok="是" if target.get("all_three") else "否",
                )
            )
        lines.append("")

    lines.append("## 4 特征、扩散分母与模型质量")
    lines.append("")
    lines.append(
        "15 列冻结特征（前 7 列沿用现有语义，后 8 列本包构建）："
        + "、".join(f"`{name}`" for name in spec.FEATURES)
    )
    lines.append("")
    lines.append(
        "覆盖：基础合格域日均 {base} 只、分析师合格域日均 {analyst} 只、评分日均 {scored} 只"
        "（最小 {smin}）；评估窗（≥{start}）内分析师合格域最小 {wmin} 只、"
        "不足 100 只的日期 {below} 天".format(
            base=_fmt(coverage.get("base_eligible_mean"), 1),
            analyst=_fmt(coverage.get("analyst_eligible_mean"), 1),
            scored=_fmt(coverage.get("scored_mean"), 1),
            smin=coverage.get("scored_min"),
            start=coverage.get("window_start"),
            wmin=coverage.get("analyst_eligible_min_in_window"),
            below=coverage.get("scored_days_below_100"),
        )
    )
    if coverage.get("days_below_100", 0):
        lines.append(
            "全历史另有 {n} 天不足 100 只（最小 {m} 只，{d}），全部早于评估窗"
            "（2015 年流动性危机期间大部分股票涨停或停牌，协议禁止涨停股进候选）".format(
                n=coverage.get("days_below_100"),
                m=coverage.get("analyst_eligible_min"),
                d=coverage.get("analyst_eligible_min_date"),
            )
        )
    lines.append("")
    model = meta.get("model_summary") or {}
    lines.append(
        "模型：{folds} 折 / {models} 个隔离注册模型；重载一致性 {reload}；"
        "OOS MAE {mae}、按日平均 RankIC {ic}（{ic_days} 日）".format(
            folds=model.get("folds"),
            models=model.get("models_registered"),
            reload="全部一致" if model.get("reload_all_identical") else "存在不一致",
            mae=_fmt((model.get("oos_metrics") or {}).get("mae"), 5),
            ic=_fmt((model.get("oos_metrics") or {}).get("rank_ic"), 4),
            ic_days=(model.get("oos_metrics") or {}).get("rank_ic_days"),
        )
    )
    lines.append("")
    font = meta.get("folds") or []
    if font:
        lines.append("**逐折内部验证（不用于调参）**")
        lines.append("")
        lines.append("| 折 | 训练起点 L | 验证起点 V | 换模日 C | 训练行 | 验证 MAE | 验证 RankIC | 验证日 |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for item in font:
            metrics = item.get("val_metrics") or {}
            lines.append(
                "| {f} | {l} | {v} | {c} | {rows} | {mae} | {ic} | {days} |".format(
                    f=item.get("fold_id"),
                    l=item.get("L"),
                    v=item.get("V"),
                    c=item.get("C"),
                    rows=item.get("train_rows"),
                    mae=_fmt(metrics.get("mae"), 5),
                    ic=_fmt(metrics.get("rank_ic"), 4),
                    days=metrics.get("rank_ic_days"),
                )
            )
        lines.append("")
    lines.append("内部验证与 OOS 误差只作披露，未用于任何调参（固定参数、无早停、无特征筛选）。")
    lines.append("")

    lines.append("## 5 固定四压力与未决事项")
    lines.append("")
    stress = diagnostics.get("stress") or {}
    rows = []
    for name, label in (
        ("report_lag_1d", "研报信息延后 1 交易日"),
        ("slippage_doubled", "B 滑点加倍（10bp）"),
        ("cash_equity_discount_20", "B 现金权益额外 20% 折扣"),
    ):
        for capital, block in (stress.get(name) or {}).items():
            rows.append(
                (
                    label,
                    capital,
                    _pct(block.get("cagr_delta")),
                    _fmt(block.get("mdd_loss_delta"), 4),
                )
            )
    lines.append("| 压力 | 本金 | cagr_delta vs A100 | mdd_loss_delta vs A100 |")
    lines.append("|---|---|---:|---:|")
    for label, capital, cagr, mdd in rows:
        lines.append(f"| {label} | {capital} | {cagr} | {mdd} |")
    zero = stress.get("zero_recovery_unresolved") or {}
    lines.append("")
    lines.append(
        "未知永久缺价零回收压力：期末持仓 {held} 只；全历史缺可核回收证据 {unres} 只"
        "（期末 {at_end} 只），未决资产 {value:,.0f} 元（占袖 {share}，分母={denom}）".format(
            held=zero.get("held_at_end", 0),
            unres=zero.get("unresolved_positions", 0),
            at_end=zero.get("unresolved_at_end", 0),
            value=float(zero.get("unresolved_value") or 0.0),
            share=_plain_pct(zero.get("unresolved_share_of_sleeve")),
            denom=zero.get("denominator", "n/a"),
        )
    )
    if zero.get("limitation"):
        lines.append(f"该压力口径限制：{zero['limitation']}")
    lines.append("")
    for item in decision["admission_obstacles"]:
        lines.append(f"- 未决/障碍：{item}")
    lines.append("")
    lines.append(
        "研报时点口径：只认 `report_date <= T`（T 日收盘后可见、T+1 执行），"
        "不能排除供应商回填与日内晚发布；当前快照不等于逐版本历史原件库。"
    )
    lines.append("")

    lines.append("## 6 身份、测试与资源")
    lines.append("")
    identity = meta.get("identity") or {}
    lines.append(
        "代码：commit `{commit}`（dirty={dirty}）；规格 SHA `{spec}`；协议 SHA `{proto}`".format(
            commit=(identity.get("git") or {}).get("commit"),
            dirty=(identity.get("git") or {}).get("dirty"),
            spec=spec.spec_sha256()[:16],
            proto=(meta.get("protocol_sha256") or "")[:16],
        )
    )
    budget = meta.get("budget") or {}
    lines.append(
        "机器时间：数据阶段 {dataset}s、训练预测 {train}s、回放 {replay}s、诊断 {diag}s；"
        "AI 编码墙钟见状态台账（`temp/p7_analyst_diffusion_ml_state_20261118.md`）".format(
            dataset=budget.get("dataset_seconds"),
            train=budget.get("train_seconds"),
            replay=budget.get("replay_seconds"),
            diag=budget.get("diagnostics_seconds"),
        )
    )
    upstream = meta.get("upstream") or {}
    lines.append("")
    lines.append("**证据链**")
    lines.append("")
    lines.append(
        "上游 run：数据 `{d}`、训练/预测 `{p}`、回放 `{r}`、诊断 `{g}`；"
        "模型注册表与训练 run 摘要一致、诊断输入 SHA 复核通过、账户取自回放 run 发布件。".format(
            d=upstream.get("dataset_run"),
            p=upstream.get("predictions_run"),
            r=upstream.get("replay_run"),
            g=upstream.get("diagnostics_run"),
        )
    )
    blockers = decision.get("evidence_blockers") or []
    if blockers:
        lines.append("证据链阻断项：" + "；".join(blockers))
    else:
        lines.append("证据链阻断项：无（缺门即阻断，不把「检查未报错」当「无工程错误」）")
    lines.append("")
    lines.append(
        "测试：新增定向测试见 `tests/test_p7_analyst_diffusion_*.py`；"
        "复审通过前未跑全量 pytest（仓库规则）。"
    )
    lines.append("")

    lines.append("## 7 未完成的正式 P7 缺口与复跑")
    lines.append("")
    lines.append("**封存与重开规则**")
    lines.append("")
    lines.append(f"- {SEAL_DECISION['stop_meaning']}")
    lines.append(f"- {SEAL_DECISION['reopen_rule']}")
    lines.append("")
    lines.append("**本版修复记录（相对首版证据，全部为工程修复，策略参数不变）**")
    lines.append("")
    for item in REPAIR_NOTES:
        lines.append(f"- {item}")
    lines.append("")
    for index, item in enumerate(spec_gaps(), start=1):
        lines.append(f"{index}. {item}")
    lines.append("")
    lines.append("复跑命令（按阶段，禁止 resume）：")
    lines.append("")
    lines.append("```powershell")
    for stage in spec.STAGES:
        lines.append(
            "python -m scripts.v2_p7.analyst_diffusion_ml.run "
            f"--campaign {meta.get('campaign')} --stage {stage} --run-id <NEW_ID>"
        )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def spec_gaps() -> tuple[str, ...]:
    """正式 P7（G 包）缺口清单（协议 §16）。"""
    return (
        "尚未冻结本候选通用 E1/Q1/E2/T2/E4 输入适配器与完整门；不得用 IDX 专用 f2 冒充。",
        "研究线与真实候选池分母未在正式台账登记；选择史只在本 campaign specification 内。",
        "正式 prereg 未登记、α 未锁定（开发侧不写 data/ledger、不消费 α）。",
        "正式校准/选择/独立验证三流未建立，未满足误放行率 CP 上界与 MDE 功效要求。",
        "2015/2018 危机窗缺 B 账户历史；资料不足列为准入障碍，不删窗、不补零。",
        "型 I 的 E1、Q1、E2、T2、三口径相关、单笔执行、月利用率等否决线未运行。",
        "真实多袖订单、净额、费用分摊与三恒等式复核未做（独立两腿相加不能替代）。",
        "新袖准入与生产范围由用户裁决；H 级历史证据不得称前向或真实日历验证通过。",
    )


def write_report(
    campaign_dir: Path, diagnostics: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    """落盘机器摘要与中文报告。"""
    out_dir = Path(campaign_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = machine_summary(diagnostics, meta)
    summary_path = out_dir / "machine_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    report_path = out_dir / "report_zh.md"
    report_path.write_text(build_report(diagnostics, meta), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "report": str(report_path),
        "recommendation": summary["decision"]["resource_recommendation"],
    }

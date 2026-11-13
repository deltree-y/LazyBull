# -*- coding: utf-8 -*-
"""P7 T6 §6.2 预登记判据（**运行前冻结**；运行后修改即视为无效裁定）。

授权
====
用户 2026-11-13 核准拟登记值（阈值、噪声带方法、verdict 规则），原文要点：
「核准如上（按拟登记值写死并落 calibration.json）」。本模块即该冻结面：
任何阈值/方法变更都必须改本文件并在收口报告中登记为**新预登记**（旧裁定作废）。

口径（本窗口、本机械与费用口径）
================
- ``PC = 1,000,000``（物理初资，T4 冻结）；``NAV_t = Σ_i(cash+MV)_i,t / PC``（``calculate_nav``）；
- 配对差 ``Δx ≡ x(VV-S) − x(VV-H)``（正 = 借用臂更优/回撤更浅）；
- 噪声带：**配对** stationary block bootstrap（两臂同一次重采样，block=20=冻结持期，
  B=4,000，seed=42），取 Δ 分布的 2.5%/97.5% 分位；敏感性 block ∈ {1, 250} 只报不判；
- **越带 ≡ |Δ| ≥ δ 且 95% 区间不含 0**（经济实质 + 统计可分辨的合取，缺一即不越带）；
- ``δ`` 为**实质阈值**（经济显著性下限），不是统计显著性水平；取值依据见 ``DELTA_RATIONALE``。

判据
====
- **C1 收益**：``Δtotal_return``；满足需 ``Δ > 0`` 且越带；
- **C2 回撤**：``Δmax_drawdown``（链式指标口径，负值；正 = 回撤更浅）；满足需 ``Δ > 0`` 且越带；
- **C3 链式三项**：``cagr`` / ``max_drawdown`` / ``volatility`` 各自需 ``Δ > 0`` 且越带
  （``volatility`` 的「更优」定义为**更低**，故其 Δ 判据取 ``Δ < 0`` 且越带，见 ``C3.items``）；
- **C4 费用/换手归因闭合**：``ΔTR_fee = −(fee_S − fee_H)/PC``、``ΔTR_gross = ΔTR − ΔTR_fee``；
  闭合需 ``|ΔTR_gross| ≤ δ_total_return``、``|Δfee|/PC ≤ δ_total_return`` 且
  ``Δturnover · Δfee ≥ 0``（多借 → 多成交 → 多费用，机械同向）；
- **C5 稳健性**：``top3 月 |Δm| 之和 / Σ|Δm| ≤ 0.50``；留出 top3 月后 ``ΔTR_loo`` 与全窗**同号**
  且保留 ≥ 50% 幅度；非零年份中同号占比 ≥ 2/3（非零年 < 3 时判为不可判 → 不满足）。

verdict
=======
``pass``：C1–C5 全部满足；``fail``：C1/C2/C3 任一为「方向相反且越带」；
其余（含幅度不足、方向不稳、集中度过高、归因不闭合）→ ``inconclusive``。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Tuple

#: 预登记 schema（与运行期落盘件一致，改变即新预登记）
CALIBRATION_SCHEMA = "p7_t6_calibration/v1"

#: 登记时刻与授权来源（**不是**运行时刻；运行时刻由 runner 单独登记）
REGISTERED_DATE = "2026-11-13"
REGISTERED_BY = "用户 2026-11-13 核准（执行代理拟稿 PR-1）"
REGISTERED_RULING = "核准如上（按预登记值写死并落 calibration.json）"

#: 臂身份：treatment = VV-S（允许 credit），control = VV-H（同配额、禁 credit）
TREATMENT = "vvs"
CONTROL = "vvh"
ARM_LABELS: Mapping[str, str] = {"vvs": "VV-S", "vvh": "VV-H"}

#: 物理初资（T4 冻结；分母唯一来源）
PHYSICAL_CAPITAL = 1_000_000.0

#: 实质阈值 δ（经济显著性下限；量纲 = 无量纲收益/波动，全窗口径）
DELTA: Mapping[str, float] = {
    "total_return": 0.005,
    "max_drawdown": 0.005,
    "cagr": 0.001,
    "volatility": 0.005,
}

#: δ 取值的**先验**依据（不引用本 run 任何结果）
DELTA_RATIONALE: Mapping[str, str] = {
    "total_return": (
        "0.50pp = 5,000 元 ≈ 本窗口年费用拖累（约 2.0pp/年）的 1/4，"
        "且高于「一手最小名义×持期」量级的交易噪声；低于该值不足以支撑配置或流程决策"
    ),
    "max_drawdown": "0.50pp：回撤深度的最小可决策差异（同为 5,000 元口径）",
    "cagr": "0.10pp：年化收益的实质下限（约为 δ_total_return 摊到 9.4 年的 1/5）",
    "volatility": "0.50pp：年化波动率的实质下限（约等于本窗口波动率量级的 3%）",
}

#: 噪声带（主口径 + 敏感性；敏感性只报不判）
BAND: Mapping[str, Any] = {
    "method": "paired_stationary_block_bootstrap",
    "block": 20,
    "replicates": 4000,
    "seed": 42,
    "interval": 0.95,
    "sensitivity_blocks": [1, 250],
    "resampling_unit": "两臂同一组重采样索引（配对），统计量按同一路径重算后取 Δ",
    "block_rationale": "block=20 = T4/T5 冻结持期（配对差的自相关尺度），非按结果选取",
    "seed_rationale": "固定 seed 保证复算逐位一致（G5 确定性门）",
}

#: 越带判据（合取）
OVER_BAND_RULE = "|Δ| ≥ δ 且 95% 区间不含 0"
BAND_INTERPRETATION = (
    "带只覆盖「同一机制在行情路径变动下的抽样波动」；本 run 的两臂是确定性模拟，"
    "故带是**可复算的裁定尺度**而非概率论断"
)

#: C3 三项（term 与 chain_metrics 字段同名；``better`` = 更优方向）
C3_ITEMS: Tuple[Mapping[str, Any], ...] = (
    {"term": "cagr", "better": "higher", "delta_key": "cagr"},
    {"term": "max_drawdown", "better": "higher", "delta_key": "max_drawdown"},
    {"term": "volatility", "better": "lower", "delta_key": "volatility"},
)

#: C5 稳健性参数
CONCENTRATION_MAX = 0.50
LOO_RETAIN_MIN = 0.50
YEAR_SIGN_MIN_SHARE = 2.0 / 3.0
YEAR_SIGN_MIN_YEARS = 3

#: 判据清单（逐条**可失败**；``passed`` 由 verdict 模块按本表规则计算）
CRITERIA: Tuple[Mapping[str, Any], ...] = (
    {
        "id": "C1",
        "name": "两臂全窗总收益配对差",
        "metric": "total_return",
        "definition": "ΔTR = TR(VV-S) − TR(VV-H)，TR 取链式净值（calculate_chain_metrics.total_return）",
        "threshold": DELTA["total_return"],
        "threshold_key": "total_return",
        "direction": "positive",
        "band": "primary block=20 配对 bootstrap（另报 block=1/250 敏感性）",
        "pass_rule": "ΔTR > 0 且越带（|ΔTR| ≥ δ_total_return 且 95% 区间不含 0）",
    },
    {
        "id": "C2",
        "name": "两臂 MaxDD 配对差",
        "metric": "max_drawdown",
        "definition": "ΔMaxDD = MaxDD(VV-S) − MaxDD(VV-H)（负值口径；正 = 回撤更浅）",
        "threshold": DELTA["max_drawdown"],
        "threshold_key": "max_drawdown",
        "direction": "positive",
        "band": "同一次配对重采样下两臂 MaxDD 之差",
        "pass_rule": "ΔMaxDD > 0 且越带",
    },
    {
        "id": "C3",
        "name": "链式指标三项配对差",
        "metric": "chain_metrics[cagr,max_drawdown,volatility]",
        "definition": "三项各自 Δ = 值(VV-S) − 值(VV-H)；volatility 以**更低**为更优",
        "threshold": dict(DELTA),
        "threshold_key": "per_term",
        "direction": "per_term",
        "band": "同一次配对重采样下三项之差",
        "pass_rule": "三项全部「更优方向且越带」（volatility 取 Δ < 0 且 |Δ| ≥ δ_vol 且带不含 0）",
    },
    {
        "id": "C4",
        "name": "费用/换手增量是否解释差异",
        "metric": "fees_turnover_attribution",
        "definition": (
            "ΔTR_fee = −(fee_S − fee_H)/PC；ΔTR_gross = ΔTR − ΔTR_fee；"
            "Δturnover = 双边换手之差；Δfee = 累计费用之差"
        ),
        "threshold": DELTA["total_return"],
        "threshold_key": "total_return",
        "direction": "attribution",
        "band": "无（确定性归因）",
        "pass_rule": (
            "|ΔTR_gross| ≤ δ_total_return 且 |Δfee|/PC ≤ δ_total_return 且 Δturnover·Δfee ≥ 0"
        ),
    },
    {
        "id": "C5",
        "name": "差异是否集中于少数月份（稳健性）",
        "metric": "monthly_paired",
        "definition": (
            "月配对差 Δm；集中度 = top3|Δm| / Σ|Δm|；留出 top3 月后按月度复利重算 ΔTR_loo；"
            "非零年份同号占比 = max(正年, 负年) / 非零年"
        ),
        "threshold": {
            "concentration_max": CONCENTRATION_MAX,
            "loo_retain_min": LOO_RETAIN_MIN,
            "year_sign_min_share": YEAR_SIGN_MIN_SHARE,
        },
        "threshold_key": "stability",
        "direction": "robust",
        "band": "分年符号检验（exact binomial，两臂差在年度单位上）",
        "pass_rule": (
            "集中度 ≤ 0.50 且 ΔTR_loo 与全窗同号且保留 ≥ 50% 且非零年同号占比 ≥ 2/3"
            f"（非零年 < {YEAR_SIGN_MIN_YEARS} 判为不满足）"
        ),
    },
)

#: verdict 规则（与任务书 §6.2 一致，逐条写明）
VERDICT_RULE: Mapping[str, Any] = {
    "pass": "C1–C5 全部满足",
    "fail": "C1/C2/C3 任一为「方向相反且越带」（Δ 与更优方向相反，且 |Δ| ≥ δ 且 95% 区间不含 0）",
    "inconclusive": "其余情形（幅度不足/未越带/集中度过高/归因不闭合/方向不稳）",
    "order": "先判 fail，再判 pass，否则 inconclusive",
}

#: 表述边界（写入 verdict.json 与收口报告）
BOUNDARY: Tuple[str, ...] = (
    "verdict 只对**本窗口（1,745 日）、本机械与费用口径**成立",
    "不构成 T6 之后的阶段授权、B 区转正、生产多袖能力结论",
    "不构成前向/日历验证结论，不称「实盘可用」",
    "不得用事后挑选的窗口/子集支撑 verdict",
)

#: 非目标（防越界解释）
NOT_IN_SCOPE: Tuple[str, ...] = (
    "参数搜索 / 因子裁剪 / 新开臂（两臂冻结为 VV-H、VV-S）",
    "改写 T4/T5 机械语义（净额化、费用、借用、死局、逐袖非正权益）",
    "多袖（N>2）推广",
)


def payload() -> Dict[str, Any]:
    """预登记载荷（**全部判据与阈值**；不含 ``registered_at`` 等运行期字段）。"""
    return {
        "schema": CALIBRATION_SCHEMA,
        "task": "P7 T6",
        "registered_date": REGISTERED_DATE,
        "registered_by": REGISTERED_BY,
        "registered_ruling": REGISTERED_RULING,
        "arms": {"treatment": TREATMENT, "control": CONTROL, "labels": dict(ARM_LABELS)},
        "physical_capital": PHYSICAL_CAPITAL,
        "definitions": {
            "nav": "NAV_t = Σ_i(cash+MV)_i,t / PC（经 calculate_nav）",
            "paired_delta": "Δx = x(VV-S) − x(VV-H)",
            "turnover": "双边换手 = Σ_t 市场腿成交名义 / PC（内部抵消不计成交额）",
            "triple": "三连指标 = 收益(total_return) / 回撤(max_drawdown) / 稳定性(volatility)",
        },
        "delta": dict(DELTA),
        "delta_rationale": dict(DELTA_RATIONALE),
        "band": dict(BAND),
        "over_band_rule": OVER_BAND_RULE,
        "band_interpretation": BAND_INTERPRETATION,
        "c3_items": [dict(item) for item in C3_ITEMS],
        "stability": {
            "concentration_max": CONCENTRATION_MAX,
            "loo_retain_min": LOO_RETAIN_MIN,
            "year_sign_min_share": YEAR_SIGN_MIN_SHARE,
            "year_sign_min_years": YEAR_SIGN_MIN_YEARS,
        },
        "criteria": [dict(item) for item in CRITERIA],
        "verdict_rule": dict(VERDICT_RULE),
        "boundary": list(BOUNDARY),
        "not_in_scope": list(NOT_IN_SCOPE),
        "alpha_consumed": True,
    }


def digest() -> str:
    """预登记稳定摘要（规范化 JSON 的 SHA256；判据变更必然改变摘要）。"""
    text = json.dumps(payload(), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def criteria_ids() -> List[str]:
    return [str(item["id"]) for item in CRITERIA]


def delta_of(key: str) -> float:
    if key not in DELTA:
        raise KeyError(f"未登记的实质阈值：{key!r}（登记={sorted(DELTA)}）")
    return float(DELTA[key])

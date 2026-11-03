# -*- coding: utf-8 -*-
"""P5b T6 演练 manifest 的分节构造器（§6.2 五臂 / §5.5 H6 / §5.6 限制）。

分节独立成函数的目的：每段都可逐段评审与单元测试，且 manifest 一旦冻结，
运行期只读（不得按读数改写）。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import DELTA_LOG_DAILY
from src.lazybull.v2.store.row_domain import LABEL_FILTERED_THROUGH, STRUCTURAL, STRUCTURAL_FROM

__all__ = [
    "ARM_ROLES",
    "LIMITATION_MARKERS",
    "WINDOW_MAPPING_MODE",
    "arms_section",
    "campaign_section",
    "checklists_section",
    "evidence_section",
    "h6_section",
    "injection_section",
    "limitations_section",
    "lineage_section",
    "upstream_section",
    "window_section",
]

#: 五臂的冻结角色名与臂角色（设计 §6.2；**运行中不得改名**）
ARM_ROLES: Mapping[str, str] = {
    "base": "aa_clone",
    "mech_aa": "aa_clone",
    "income_minus_5d": "synthetic_effect",
    "dd_improve_12pp": "synthetic_effect",
    "dd_worsen_12pp": "synthetic_effect",
}
#: 崩盘窗映射口径（唯一可实现的口径：T5 冻结窗是样本序号区间，无自然日）
WINDOW_MAPPING_MODE = "index_aligned_frozen_sample_index"
#: §5.6 限制声明必须覆盖的六个要点（逐条以关键词校验，缺一即拒绝冻结）
LIMITATION_MARKERS: Mapping[str, str] = {
    "observed_dependence": "已观测依赖结构",
    "no_future_distribution": "未来真实分布不变",
    "shared_source_pair": "共享同一对真实源",
    "ma_overlap": "MA(H−1)",
    "empirical_quantile": "经验分位",
    "evidence_class": "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE",
}


def evidence_section() -> Dict[str, Any]:
    """证据类别声明（禁止冒充前向/SLA/生产证据）。"""
    return {
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
        "no_ledger_entry": True,
        "row_domain": STRUCTURAL,
        "schedule_mode": "replay",
    }


def campaign_section(
    *, campaign_id: str, campaign_root: str, data_root: str, base_root: str
) -> Dict[str, Any]:
    """演练 campaign 身份（产物根必须在试车场根下）。"""
    return {
        "campaign_id": str(campaign_id),
        "campaign_root": str(campaign_root).replace("\\", "/"),
        "data_root": str(data_root).replace("\\", "/"),
        "base_root": str(base_root).replace("\\", "/"),
        "schedule_mode": "replay",
        "row_domain": STRUCTURAL,
        "single_d0": True,
        "distinct_from_t4_t5_campaigns": True,
    }


def window_section(days: Sequence[str], *, selection_rule: str, watermark: str) -> Dict[str, Any]:
    """虚拟窗口（H1 下限 10 日；窗口冻结后不得删减或挑选）。"""
    ordered = [str(day) for day in days]
    return {
        "d0_date": ordered[0],
        "d0_role": "valuation_only",
        "first_return_date": ordered[1] if len(ordered) > 1 else None,
        "virtual_days": ordered,
        "day_count": len(ordered),
        "return_days": max(0, len(ordered) - 1),
        "selection_rule": str(selection_rule),
        "structural_from": STRUCTURAL_FROM,
        "label_filtered_through": LABEL_FILTERED_THROUGH,
        "data_watermark": str(watermark),
        "deletion_forbidden": True,
        "h1_minimum_days": 10,
        "contiguous": True,
    }


def _income_injection(delta_multiple: float) -> Dict[str, Any]:
    return {
        "kind": "income_log_shift",
        "delta_multiple": float(delta_multiple),
        "delta_log_daily": float(DELTA_LOG_DAILY),
        "daily_log_shift": -float(delta_multiple) * float(DELTA_LOG_DAILY),
        "rule": "x'_t = x_t - multiple*delta（全样本；δ=ln(1.05)/252，禁止另立口径）",
    }


def _window_injection(direction: str, target_pp: float, parameter: float | None) -> Dict[str, Any]:
    return {
        "kind": "crash_window_transform",
        "direction": str(direction),
        "target_pp": float(target_pp),
        "anchor": ANCHOR_HORIZON,
        "parameter": None if parameter is None else float(parameter),
        "parameter_name": "lambda" if direction == DIRECTION_IMPROVE else "zeta",
        "rule": ("窗口内负 log 收益 x'_t=(1-λ)x_t（DD-I）/ (1+ζ)x_t（DD-W）；正收益日与窗口外不变"),
        "solve": "T5 求解器语义：101 点扫描 + ≤80 次细化 + 0.05pp 容差 + 多解取最小干预",
        "solve_source": "frozen_B0_anchor_window（盲态；不读演练读数）",
    }


def arms_section(parameters: Mapping[str, float | None]) -> Dict[str, Any]:
    """五臂效应声明（角色名冻结；λ/ζ 为盲态预登记值）。"""
    arms: Dict[str, Any] = {
        "base": {
            "role": ARM_ROLES["base"],
            "note": "生产冻结模型/配置的 fresh_cash 克隆（基准臂）",
            "injection": {"kind": "none"},
        },
        "mech_aa": {
            "role": ARM_ROLES["mech_aa"],
            "note": "与 base 完全同模型/配置/输入（仅验运行与账本逐值一致，不作统计 A/A）",
            "injection": {"kind": "none"},
        },
        "income_minus_5d": {
            "role": ARM_ROLES["income_minus_5d"],
            "note": "已知收益效应臂：日度 log 位移 −5δ（预登记，不按结果调整）",
            "injection": _income_injection(5.0),
        },
        "dd_improve_12pp": {
            "role": ARM_ROLES["dd_improve_12pp"],
            "note": "DD-I 改善臂：冻结崩盘窗内负收益衰减（目标锚 look 见 target_pp）",
            "injection": _window_injection(
                DIRECTION_IMPROVE,
                float(parameters.get("dd_improve_target_pp", 12.0)),
                parameters.get("dd_improve"),
            ),
        },
        "dd_worsen_12pp": {
            "role": ARM_ROLES["dd_worsen_12pp"],
            "note": "DD-W 恶化臂：冻结崩盘窗内负收益放大（目标锚 look 见 target_pp）",
            "injection": _window_injection(
                DIRECTION_WORSEN,
                float(parameters.get("dd_worsen_target_pp", -12.0)),
                parameters.get("dd_worsen"),
            ),
        },
    }
    return {"arms": arms, "renaming_forbidden": True, "count": len(arms)}


def injection_section(
    frozen_windows: Sequence[Mapping[str, Any]], *, frozen_list_sha256: str, source_sha256: str
) -> Dict[str, Any]:
    """窗口引用（T5 冻结清单原文 + 哈希；不得重选）。"""
    return {
        "frozen_list": [dict(entry) for entry in frozen_windows],
        "frozen_list_sha256": str(frozen_list_sha256),
        "source_manifest_sha256": str(source_sha256),
        "mapping_mode": WINDOW_MAPPING_MODE,
        "reselect_forbidden": True,
        "truncation_forbidden": True,
        "anchor": ANCHOR_HORIZON,
        "note": (
            "T5 冻结窗是 B0 最后 500 个收益样本上的序号区间（fold_local_ordinal，无自然日）；"
            "演练路径按同序号复用，日期不同 ⇒ 索引对齐而非日期对齐（已在限制声明登记）"
        ),
    }


def upstream_section(
    *,
    t4_manifest: Mapping[str, str],
    t4_thresholds: Mapping[str, str],
    t4_holdout: Mapping[str, str],
    t5_manifest: Mapping[str, str],
    t5_thresholds: Mapping[str, str],
    t5_power: Mapping[str, str],
) -> Dict[str, Any]:
    """上游链接：T4/T5 manifest 与两侧临界值产物（哈希不符即拒绝运行）。"""
    return {
        "t4_manifest": dict(t4_manifest),
        "t4_thresholds": dict(t4_thresholds),
        "t5_manifest": dict(t5_manifest),
        "t5_thresholds": dict(t5_thresholds),
        "t4_holdout": dict(t4_holdout),
        "t5_power": dict(t5_power),
        "verification": True,
        "verification_scope": "冻结时逐文件 sha256 + 冻结窗清单哈希；run/report 阶段再次核验（不符即拒绝运行）",
    }


def anchor_solver_section(parameters: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """锚窗 λ/ζ 求解的可离线复核登记（盲态预登记；不参与判定）。"""
    payload = dict(parameters or {})
    return {
        "method": "复用 T5 求解器语义（101 点全域扫描 + ≤80 次二分细化 + 0.05pp 容差 + 最小干预）",
        "source": "referee.drawdown.injection（只读调用，未改其实现）",
        "window": "B0 最后 500 收益样本（T5 windows.frozen_list 序号区间）",
        "anchor": ANCHOR_HORIZON,
        "parameters": [
            {
                "direction": direction,
                "name": "lambda" if direction == "dd_improve" else "zeta",
                "value": payload.get(direction),
                "status": (
                    "constructible" if payload.get(direction) is not None else "not_constructible"
                ),
            }
            for direction in (DIRECTION_IMPROVE, DIRECTION_WORSEN)
        ],
    }


def h6_section(dd_improve_target: float = 12.0, dd_worsen_target: float = -12.0) -> Dict[str, Any]:
    """H6 三条单条数值的判定规则（方向 + 事件；**不含频率/MDE**）。

    DD 目标由预登记推导（DD-I ``max(12, c^I_anchor+2)``；DD-W ``min(-12, c^W_anchor-2)``），
    因为 T5 冻结的 ``c^W_anchor`` 远深于 −12pp，−12pp 的恶化在结构上无法触发否决事件。
    """
    return {
        "owner": "P5b T6",
        "status": "frozen_before_evidence",
        "items": [
            {
                "name": "income_minus_5d",
                "target": "配对日度 log 差均值 = −5δ（**收入劣化方向**；裁判应给出否决信号）",
                "judgment": (
                    "点估计符号为负（劣化方向）且幅度与预登记 −5δ 一致（相对容差 1e-6）"
                    "且账本→配对收益链路完整"
                ),
                "z_identifiability": (
                    "Z 点估计是否可识别（退化方差）只作**披露字段**（z_identifiable），"
                    "不改变方向判定；方差退化时不得伪造 Z"
                ),
                "forbidden": "不得宣称检出率/MDE/显著性；不得把浮点伪影当 Z",
                "evidence_fields": [
                    "h6.items[0].paired_mean",
                    "h6.items[0].max_abs_residual",
                    "h6.items[0].z_identifiable",
                    "h6.items[0].looks",
                ],
            },
            {
                "name": "dd_improve_12pp",
                "target": f"锚 look 纯注入效应 ≈ {dd_improve_target:+.2f}pp（= max(12pp, c^I_anchor+2pp)）",
                "target_pp": float(dd_improve_target),
                "judgment": (
                    "D_k>0 且 D_k>c^I_anchor（T5 冻结值）⇒ 触发改善检出事件"
                    "（本口径**加严**于 §6.3 的「方向与链路」：单路径事件，不是检出率）"
                ),
                "forbidden": "不得用「未触发恶化否决」冒充改善检出；不得改报目标值",
                "evidence_fields": [
                    "h6.items[1].constructibility",
                    "h6.items[1].d_anchor_pp",
                    "h6.items[1].c_i_anchor_pp",
                    "h6.items[1].event",
                ],
            },
            {
                "name": "dd_worsen_12pp",
                "target": f"锚 look 纯注入效应 ≈ {dd_worsen_target:+.2f}pp（= min(-12pp, c^W_anchor-2pp)）",
                "target_pp": float(dd_worsen_target),
                "target_derivation": (
                    "T5 冻结 c^W_anchor=-31.1094pp 深于 -12pp：−12pp 的恶化不可能越过否决阈值，"
                    "故目标改为「阈值 − 2pp」；此为盲态预登记修正，理由随 manifest 冻结"
                ),
                "judgment": (
                    "D_k<0 且 D_k<c^W_anchor（T5 冻结值）⇒ 触发恶化否决事件"
                    "（同上：单路径事件，不是检出率）"
                ),
                "forbidden": "不得与 DD-I 互相抵扣；不得截断档位",
                "evidence_fields": [
                    "h6.items[2].constructibility",
                    "h6.items[2].d_anchor_pp",
                    "h6.items[2].c_w_anchor_pp",
                    "h6.items[2].event",
                ],
            },
        ],
        "frequency_source": "T4/T5 holdout 产物（本演练不产生任何频率/MDE）",
        "directional_only": True,
        "mutual_offset_forbidden": True,
    }


def checklists_section() -> Dict[str, Any]:
    """H1/H6/H7 的作者侧检查清单（状态由证据推导，闸门判定归用户）。"""
    return {
        "h1": [
            "virtual_days >= 10（窗口在 manifest 冻结）",
            "逐日逐臂 row_domain 回执全为 structural",
            "每臂每日恰一条权威估值行",
            "机械 A/A 逐值一致（从分片复算，不信 runner 字段）",
            "A/A 比对行数非零（无成交日也比现金与持仓）",
            "生产根零写入（data/paper、data/runs、data/models）",
        ],
        "h6": [
            "收益臂：配对均值符号为负且链路完整",
            "DD-I 臂：锚 look D_k>c^I_anchor 事件触发（或如实 not_constructible）",
            "DD-W 臂：锚 look D_k<c^W_anchor 事件触发（或如实 not_constructible）",
            "三条分别判定，任一失败不得用另一条抵扣",
            "频率/MDE 只引用 T4/T5 holdout 读数",
        ],
        "h7": [
            "manifest 哈希与上游链接完整且运行期校验",
            "运行手册存在且命令可执行（文档-产物一致性用例）",
            "版本号与 CHANGELOG/README/pyproject 同步",
            "限制声明覆盖 §5.6 全部要点",
            "判定表不含任何「闸门通过」字段",
        ],
        "gate_owner": "user_at_merge_acceptance",
    }


def limitations_section() -> list:
    """§5.6 限制声明全文口径（≥4 条；六要点逐条覆盖）。"""
    return [
        "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史虚拟时钟演练，不是前向影子，"
        "不产生候选前向结论或生产建议",
        "历史模拟只能证明「在已观测依赖结构下的链路与方向」，不能证明未来真实分布不变，"
        "也不能自动获得生产裁决权",
        "T4/T5 的训练与 holdout 共享同一对真实源（source_run_id 相同），只以互斥日期块与"
        "独立 realization/seed 隔离；真实源只有两批，不构成独立真实换种子源对验证",
        "收益裁判的 MA(H−1) 合成只覆盖持有期重叠，经验分位是已测场景的有限样本结果、"
        "不是普适尾概率保证；不覆盖未知 regime、流动性冲击或未来结构变化",
        "回撤臂的崩盘窗是 B0 最后 500 个收益样本上的序号区间（无自然日），演练路径只能按"
        "样本序号对齐复用：索引对齐不等于同一段历史",
        "确定性位移臂（clone + 常数位移）的配对路径方差为零，收益裁判的 block 统计量在该"
        "样本上不可识别；不得把退化方差下的数值伪影当作 Z 点估计",
        "机械 A/A（同路径克隆）只证明运行与账本逐值一致，不得充当统计 A/A；统计 A/A 只来自"
        "T4/T5 的独立同分布路径",
        "DD-I 与 DD-W 分别判定，任一方向不可执行即该方向无结论；不得互相抵扣、不得用"
        "「未触发恶化否决」冒充改善检出",
        "本演练不产生检出率/MDE/显著性；频率类读数一律只引用 T4/T5 冻结产物",
        "【INJECTED_ARM_VALUATION_DECOMPOSITION】注入臂只保证 NAV 读数：`portfolio_value` 与 "
        "`market_value` 被读侧覆写后不再与持仓快照分解自洽（`position_value` 仍为真实市值）；该不一致"
        "仅存在于注入臂的旁路读数，不进入成交/现金/持仓/日期",
    ]


def lineage_section(
    *,
    sequence: int = 1,
    superseded_manifest_sha256: str | None = None,
    superseded_calibrator_sha256: str | None = None,
    reason: str | None = None,
) -> Dict[str, Any]:
    """T6 冻结谱系（首次冻结 ``sequence=1``；重冻必须换新 sequence 并登记前身与原因）。"""
    return {
        "sequence": int(sequence),
        "policy": "冻结件与阶段产物均独占创建；如需重冻必须换新 sequence 并登记前身哈希与原因",
        "superseded_manifest_sha256": superseded_manifest_sha256,
        "superseded_calibrator_sha256": superseded_calibrator_sha256,
        "supersede_reason": reason,
    }

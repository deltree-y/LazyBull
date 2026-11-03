# -*- coding: utf-8 -*-
"""P5b T5 回撤 manifest 的分节构造器（§5.5 回撤清单逐段冻结）。

每段一个纯函数，便于逐段评审与单元测试；``calibration.build_drawdown_manifest`` 只负责
汇总。冻结文字与数值都**只在本模块出现一次**，manifest 校验器反向核对这些数值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.testbed.referee.drawdown.manifest import ManifestError
from src.lazybull.v2.hosts.testbed.referee.drawdown.crash_window import (
    MAX_CRASH_WINDOWS,
    CrashWindow,
    window_section,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.injection import (
    LAMBDA_RANGE,
    ZETA_RANGE,
    solver_section,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTIONS,
    HOLD_H,
    LOOK_DAYS,
    WINDOW_SAMPLES,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.power import (
    EFFECT_GRID_PP,
    MDE_IMPROVE_KEY,
    MDE_NOT_REACHED,
    MDE_WORSEN_KEY,
    TARGET_POWER,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import (
    IMPROVE_EVENT,
    WORSEN_EVENT,
    threshold_section,
)
from src.lazybull.v2.hosts.testbed.referee.scenes import block_date_blocks

__all__ = [
    "aa_section",
    "counterfactual_section",
    "deviation_registry_section",
    "direction_fixtures",
    "drawdown_referee_section",
    "effect_section",
    "h6_section",
    "holdout_section",
    "injection_section",
    "limitations_section",
    "lineage_section",
    "mdd_section",
    "reading_section",
    "repo_relative",
    "source_pair_section",
]

#: 源对身份（与收益仪器共享同一对真实源；禁止换源）
BASE_RUN_ID = "factor_ab_20260916_071157_base"
NOISE_RUN_ID = "factor_ab_20260916_123214_noiseB0"


def repo_relative(path: str, repo_root: Path, *, allow_external: bool = False) -> str:
    """把路径统一为「相对仓库根的正斜杠形式」（manifest 唯一写法）。

    默认**fail-closed**：仓库外路径直接报错（否则 manifest 会内嵌机器绝对路径，既不可移植，
    也会让上游源对一致性校验给出误导性的"源对不一致"）。只有显式 ``allow_external=True``
    （测试的隔离假仓库场景）才保留绝对路径。
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as exc:
        if not allow_external:
            raise ManifestError(
                f"路径不在仓库根下（manifest 只登记仓库相对路径）: {path}；repo_root={repo_root}"
            ) from exc
        return candidate.resolve().as_posix()


def mdd_section() -> Dict[str, Any]:
    """MDD 定义段（峰到谷、log 尺度、首点计入、pp 换算）。"""
    return {
        "definition": "MDD = min_t(cum[t] - max_{s<=t} cum[s])，cum 为累计 log 收益且 cum[0]=0",
        "log_space": True,
        "start_point_included": True,
        "percent_conversion": "MDD_pct = (exp(MDD_log) - 1) * 100",
        "peak_to_trough": True,
        "golden_case_required": True,
    }


def reading_section(plan: Any) -> Dict[str, Any]:
    """统一读数与 horizon 段（§5.4 头部 + §5.4.4）。"""
    return {
        "formula": "D_k = MDD_arm(T_k) - MDD_base(T_k)",
        "base_arm": "MDD_base = 冻结基准臂 B0 在 T_k 上的 MDD（每 look 只算一次，全实现共用）",
        "sign_convention": "MDD 为负值 ⇒ D_k>0 改善、D_k<0 恶化；DD-I 的 I_k = D_k 不得再取负",
        "horizons": [int(day) for day in LOOK_DAYS],
        "H": int(HOLD_H),
        "K": int(plan.k),
        "look_blocks": [int(j) for j in plan.j_looks],
        "t_anchor": int(ANCHOR_HORIZON),
        "anchor_rule": "T_anchor = H*J_max（唯一锚；目标只在该 horizon 上定）",
        "independent_horizon_per_look": True,
        "improve_negation_forbidden": True,
        "full_chain_used": False,
        "full_chain_note": (
            "全链 1714 日（1700 收益）的 MDD 与 episode 只作历史描述，"
            "禁止与 500 日锚混合读数或参与调参"
        ),
        "observed_noise_retained": True,
    }


def injection_section() -> Dict[str, Any]:
    """注入公式段（§5.4.3 冻结）。"""
    return {
        "dd_improve": {
            "formula": "x'_t = (1 - λ) x_t（窗口内负 log 收益）",
            "parameter": "lambda",
            "range": [float(LAMBDA_RANGE[0]), float(LAMBDA_RANGE[1])],
            "target_sign": "positive",
        },
        "dd_worsen": {
            "formula": "x'_t = (1 + ζ) x_t（窗口内负 log 收益）",
            "parameter": "zeta",
            "range": [float(ZETA_RANGE[0]), float(ZETA_RANGE[1])],
            "target_sign": "negative",
        },
        "support": "窗口内且 x_t<0 才改造（支撑由实现自身的符号决定）",
        "positive_days_unchanged": True,
        "outside_window_unchanged": True,
        "reconstruction": "r'_t = exp(x'_t) - 1",
        "parameters_independent": True,
    }


def counterfactual_section() -> Dict[str, Any]:
    """反事实与目标段（§5.4.3「唯一锚定」）。"""
    return {
        "noise": "x = B0_window + n（配对噪声实现；n 由冻结场景生成）",
        "arm": "arm = x + inj（同一路径只注入一次）",
        "target": "MDD(arm) - MDD(x)（纯注入效应；禁止把 MDD(arm)-MDD(B0) 当目标）",
        "observed": (
            "MDD(arm) - MDD(B0) = 裁判读数 D_k（含噪观察效应）；只报告、不参与求解、"
            "不得强行调到目标档"
        ),
        "shared_injection_across_looks": True,
        "independent_parameter_solve": True,
        "anchor": int(ANCHOR_HORIZON),
        "window_reselection_forbidden": True,
        "look_resolve_forbidden": True,
    }


def aa_section() -> Dict[str, Any]:
    """A/A 段（零假设与独立路径；§5.4.1/§5.4.2）。"""
    return {
        "independent_paths": True,
        "same_path_zero_injection_forbidden": True,
        "arm": "每个单元独立抽取 M 条实现 arm = B0 + n（不同 seed 子流按场景隔离）",
        "baseline": (
            "base = 独立抽取的第二条路径 B0+n_b（与 arm=B0+n_a 逐实现配对；"
            "两侧各带独立噪声 ⇒ 零假设宽度为双路径口径）"
        ),
        "forbidden": "禁止用同一路径零注入（D_k ≡ 0，A/A 退化）",
        "unit_definition": "scene × null(DD-W/DD-I) × H × referee",
        "shared_across_directions": True,
        "used_for_thresholds_only": True,
        "cannot_be_used_as_effect": True,
    }


def holdout_section() -> Dict[str, Any]:
    """holdout 段（逐单元 Wilson 上界门；两裁判分别命名）。"""
    return {
        "unit_definition": "scene × null(DD-W/DD-I) × H × referee",
        "wilson_confidence": 0.95,
        "wilson_upper_max": 0.06,
        "pooled_or_weighted_average_forbidden": True,
        "per_direction_naming": {"worsen": "dd_worsen", "improve": "dd_improve"},
        "hard_failure_policy": (
            "任一单元 Wilson 上界 >6% ⇒ 该方向裁判不可用；不得删困难场景、"
            "不得扩样本回调阈值、不得用另一方向抵扣"
        ),
        "same_event_as_power": True,
    }


def effect_section() -> Dict[str, Any]:
    """效应网格、MDE 与可构造率段（§5.3 表 + §5.5 回撤行 + §5.4.4）。"""
    return {
        "grid_pp": [float(level) for level in EFFECT_GRID_PP],
        "directions": list(DIRECTIONS),
        "target_power": float(TARGET_POWER),
        "mde_names": {"worsen": MDE_WORSEN_KEY, "improve": MDE_IMPROVE_KEY},
        "mde_definition": (
            "A/A 过门后，点估计检出率达 80% 的最小纯注入档；逐场景登记，禁止跨场景平均"
        ),
        "if_not_reached": MDE_NOT_REACHED,
        "main_denominator": "M = 全部实现（不可构造按未检出计入，不得剔除）",
        "constructibility_rate_field": "constructibility_rate = C/M（逐格披露）",
        "units": "MDD 百分点（pp）；MDD_pct=(exp(MDD_log)-1)*100",
        "nominal_vs_actual": (
            "档名 = 目标值；锚上**实际**纯注入效应 = |目标| - 0.05pp（最小干预左沿），"
            "逐格见 pure_effect_pp"
        ),
        "display_conversion": (
            "每档同时报告 λ/ζ 分布、纯注入效应、含噪观察效应、各 look 实际效应均值"
        ),
        "annualization": {
            "applicable": False,
            "reason": (
                "回撤不是收益：pp 不折算成日度/年化收益，也不折算成收益裁判的 δ；"
                "收益侧 δ=ln(1.05)/252 的唯一口径在收益仪器"
            ),
        },
        "not_constructible_must_be_listed": True,
    }


def direction_fixtures() -> List[Dict[str, Any]]:
    """方向 fixture 的冻结描述（§5.4.4 两个确定性案例）。"""
    return [
        {
            "name": "improvement_only",
            "base_mdd_pp": -20.0,
            "arm_mdd_pp": -10.0,
            "d_pp": 10.0,
            "allowed_event": "dd_improve",
            "forbidden_event": "dd_worsen",
        },
        {
            "name": "worsening_only",
            "base_mdd_pp": -20.0,
            "arm_mdd_pp": -30.0,
            "d_pp": -10.0,
            "allowed_event": "dd_worsen",
            "forbidden_event": "dd_improve",
        },
    ]


def drawdown_referee_section(
    plan: Any, windows: Sequence[CrashWindow], *, samples: int
) -> Dict[str, Any]:
    """``drawdown_referee`` 整段（读数/窗口/注入/反事实/求解/阈值/A/A/holdout/效应）。"""
    base = reading_section(plan)
    return {
        "mdd": mdd_section(),
        "reading": base,
        "windows": window_section(windows, samples=samples),
        "injection": injection_section(),
        "counterfactual": counterfactual_section(),
        "solver": solver_section(),
        "threshold": threshold_section(plan),
        "aa": aa_section(),
        "holdout": holdout_section(),
        "effect": effect_section(),
        "direction_fixtures": direction_fixtures(),
        "events": {"worsen": WORSEN_EVENT, "improve": IMPROVE_EVENT},
        "max_crash_windows": int(MAX_CRASH_WINDOWS),
    }


def source_pair_section(source: Any, root: Path) -> Dict[str, Any]:
    """源对段（run_id + 路径/哈希 + 互斥源折 + 日期块 + 两池规模）。"""
    pair = source.pair
    return {
        "source_run_id": f"{BASE_RUN_ID} + {NOISE_RUN_ID}",
        "base_run_id": BASE_RUN_ID,
        "noise_run_id": NOISE_RUN_ID,
        "base_chain_nav": repo_relative(pair.base_path, root),
        "noise_chain_nav": repo_relative(pair.noise_path, root),
        "base_sha256": pair.base_sha256,
        "noise_sha256": pair.noise_sha256,
        "noise_arm_note": (
            "B0' = factor_ab_20260916_123214_noiseB0（同一策略、换随机种子的重跑臂）；"
            "**不是** P3 的 500 树重训臂，也不含任何重训/调参差异"
        ),
        "train_folds": list(source.train_pool["folds"]),
        "holdout_folds": list(source.holdout_pool["folds"]),
        "date_blocks": block_date_blocks(pair, source.train_pool["folds"])
        | {"holdout": block_date_blocks(pair, source.holdout_pool["folds"])},
        "pool_isolation": {
            "train": "训练源折，按本池自身均值去中心",
            "holdout": "holdout 源折，按本池自身均值去中心",
            "cross_pool_sampling_forbidden": True,
            "source_folds_observed": True,
            "not_independent_real_source_pair": True,
        },
        "rows": {
            "train_pool": int(source.train_pool["rows"]),
            "holdout_pool": int(source.holdout_pool["rows"]),
            "paired_total": int(pair.rows),
        },
    }


def deviation_registry_section(counts: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """与设计/冻结值的已知差异（必须逐条登记，不得静默）。"""
    return [
        {
            "item": "命名种子",
            "design_value": "20261101（设计 v4 §5.3 表「命名种子」默认值）",
            "frozen_value": f"{int(counts['seed'])} + 前缀 {counts['realization_prefix']}",
            "reason": (
                "沿用上游收益仪器冻结值（T0.5b PSD r3 / T4）以便与 psd_r3 读数对照；"
                "改回 20261101 属另立预登记，会使本仪器与上游证据失去同源可复现性"
            ),
        },
        {
            "item": "回撤仪器独立冻结",
            "design_value": "设计 §5.5 只要求一份 calibration_manifest（回撤段在 T4 为占位）",
            "frozen_value": "另立 p5b_t5_drawdown_manifest（lineage.sequence=1）+ 独立临界值/产物",
            "reason": (
                "收益仪器已冻结（manifest sha 545d2be4…），就地改写会破坏收益证据的可追溯性；"
                "故回撤段以新 manifest 承载，并以父证据哈希链接上游"
            ),
        },
        {
            "item": "崩盘窗窗口口径",
            "design_value": "在 B0 完整链上识别 episode（§5.4.3 首句）",
            "frozen_value": (f"在 B0 最后 {WINDOW_SAMPLES} 个收益样本上识别；全链仅历史描述"),
            "reason": (
                "T0.5b 已批冻结口径（用户批准）：全链窗口 8pp/12pp 不可达（最优约 5.07pp），"
                "最后 500 日窗口可达；两窗口的 λ 不同，故 T_anchor 与评估窗口必须同时冻结"
            ),
        },
        {
            "item": "A/A 独立路径",
            "design_value": "零假设由同分布独立路径生成，禁止同一路径零注入",
            "frozen_value": (
                "每单元独立抽取 M 条 arm = B0 + n_a 与 M 条 base = B0 + n_b（seed 子流按场景隔离），"
                "逐实现配对 D_i,k = MDD(a_i,k) − MDD(b_i,k)（两侧各带独立噪声）"
            ),
            "reason": (
                "同路径零注入会使 D_k ≡ 0、临界值塌到 0，故两侧都必须独立加噪；"
                "此前「一条加噪 vs 固定 B0」只含单侧噪声，零点宽度约为双路径的一半，"
                "导致真 A/A 下 DD-I 误报 39.95%（Wilson 上界 40.9%，远超 ≤6% 验收）；"
                "现按设计 v4 §5.4.1/§5.4.2 改为双独立路径配对（R2-P5B-03 整改）"
            ),
        },
        {
            "item": "效应档实现数",
            "design_value": "设计 §5.3 表回撤列 500 路径/档",
            "frozen_value": f"{int(counts['effect_per_scene_per_level'])} 路径/档·场景",
            "reason": (
                "任务书要求不得低于 T4 已冻结口径（训练 2000/单元、holdout 10000/单元、"
                "效应 2000/档·场景），只可上调；实现数写入 manifest 并逐格核对"
            ),
        },
        {
            "item": "纯注入效应目标",
            "design_value": "求解 MDD(arm)-MDD(x)，唯一锚 T_anchor=H*J_max",
            "frozen_value": ("逐实现求解 λ/ζ（向量化跨实现求根），同一路径所有 look 共用一次注入"),
            "reason": (
                "含噪基线之间的差（MDD(arm)-MDD(B0)）同时含噪声效应，作为目标会把"
                "已知效应档变成噪声读数；按实现求根保证每档的纯注入效应确实落在容差内"
            ),
        },
        {
            "item": "含噪观察效应",
            "design_value": "观察量保留噪声，不得把观察到的 D_k 强行调到目标档",
            "frozen_value": "MDD(arm)-MDD(B0) 只作报告字段，不参与求解、不入阈值",
            "reason": "保留噪声读数才能暴露真实检出难度；把它当目标等于用结果反推输入",
        },
    ]


def limitations_section() -> List[str]:
    """§5.6 限制声明（至少 3 条；报告与判定表原样带出）。"""
    return [
        "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史试车场校准，不是前向影子",
        (
            "崩盘窗只在 B0 最后 500 个收益样本上识别并冻结一次；该窗口是单条历史路径的"
            "结构，不能代表未来崩盘形态"
        ),
        (
            "训练与 holdout 共享同一对真实源 run（B0/B0'）与同一 B0 窗口，仅以互斥源折、"
            "独立 seed 子流与独立实现编号隔离；不构成独立真实源对验证"
        ),
        (
            "经验分位（min/max over scene）是已测场景的有限样本结果，不是普适尾概率保证；"
            "每 look 的 c^W/c^I 由约 22 个次序统计量决定"
        ),
        (
            "注入只改造窗口内负 log 收益的幅度，不改变窗口外的路径、不改变正收益日；"
            "真实回撤的结构性成因（相关性突变、流动性冲击）不在注入族内"
        ),
        "不可构造的实现按未检出计入主分母；C/M 只作披露，不得据此剔除实现",
        "DD-W 与 DD-I 分别判定，任一方向不通过即该方向不可用；不得互相抵扣",
        "所有结论仅在本 manifest 冻结的场景集与窗口内成立；不得外推为生产裁决权",
    ]


def h6_section() -> Dict[str, Any]:
    """H6 单条数值占位（本 Task 不运行；只在 manifest 保留占位）。"""
    return {
        "status": "placeholder_not_run",
        "owner": "P5b T6",
        "items": [
            "DD-I +12pp 单条历史演练",
            "DD-W −12pp 单条历史演练",
        ],
        "placeholders_pp": {"dd_improve": 12.0, "dd_worsen": -12.0},
        "note": "只检查点估计方向与链路，不作为检出率/MDE；频率只来自 H4/H5",
    }


def lineage_section(
    *,
    sequence: int = 1,
    superseded_manifest_sha256: Optional[str] = None,
    superseded_calibrator_sha256: Optional[str] = None,
    supersede_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """冻结谱系（首次 = 无前身；重冻必须留废止记录）。"""
    return {
        "sequence": int(sequence),
        "superseded_manifest_sha256": superseded_manifest_sha256,
        "superseded_calibrator_sha256": superseded_calibrator_sha256,
        "supersede_reason": supersede_reason,
        "policy": (
            "冻结件与阶段产物均独占创建；如需重冻必须换新 sequence 并登记前身哈希与原因，"
            "禁止原地改写"
        ),
    }

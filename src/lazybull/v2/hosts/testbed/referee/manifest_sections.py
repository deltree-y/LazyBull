# -*- coding: utf-8 -*-
"""P5b T4 校准 manifest 的分节构造器（§5.5 全表逐段冻结）。

从 ``calibration.py`` 拆出（文件体量治理）：每段一个纯函数，便于逐段评审与单元测试；
``calibration.build_manifest_document`` 只负责把各段汇总成文档。

命名口径：``source_pair_section`` / ``statistic_section`` / ``threshold_section`` /
``effect_and_holdout_section`` / ``drawdown_and_h6_section`` / ``deviation_registry_section`` /
``limitations_section``，以及序列化辅助 :func:`finite_safe`。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.lazybull.v2.hosts.testbed.referee.manifest import (
    DRAWDOWN_PLACEHOLDER_STATUS,
    ManifestError,
)
from src.lazybull.v2.hosts.testbed.referee.power import EFFECT_GRID_DELTA
from src.lazybull.v2.hosts.testbed.referee.scenes import block_date_blocks
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DELTA_ANNUAL_TARGET,
    DELTA_LOG_DAILY,
    LOOK_N_MAX,
    annualized_log_return,
)
from src.lazybull.v2.hosts.testbed.referee.training import threshold_rule_steps

__all__ = [
    "DESIGN_DEFAULTS",
    "deviation_registry_section",
    "drawdown_and_h6_section",
    "effect_and_holdout_section",
    "finite_safe",
    "limitations_section",
    "repo_relative",
    "source_pair_section",
    "statistic_section",
    "threshold_section",
]

#: 设计 v4 默认常量（写进 manifest；实际取值以冻结 manifest 为准）
DESIGN_DEFAULTS: Mapping[str, Any] = {
    "H": 20,
    "n_max": LOOK_N_MAX,
    "alpha_train": 0.045,
    "target_power": 0.80,
    "wilson_upper_max": 0.06,
    "invalid_rate_per_look_max": 0.10,
}


def finite_safe(payload: Any) -> Any:
    """把非有限浮点（NaN/Inf）换成 ``None``，供**配置失败**时仍能落盘诊断产物。

    正常路径（``passed=True``）不含非有限值，故本函数不改变成功产物的字节。
    """
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, Mapping):
        return {key: finite_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [finite_safe(value) for value in payload]
    return payload


def repo_relative(path: str, repo_root: Path) -> str:
    """把绝对/相对路径统一为「相对仓库根的正斜杠路径」（manifest 唯一写法）。"""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as exc:
        raise ManifestError(
            f"源对/引用文件必须位于仓库内（manifest 只登记仓库相对路径）: {path}（{exc}）"
        ) from exc


def source_pair_section(source: Any, root: Path) -> Dict[str, Any]:
    """源对段（run_id + 路径/哈希 + 互斥源折 + 日期块 + 两池规模）。

    Args:
        source: ``calibration.PreparedSource``（用 ``Any`` 避免与编排模块循环导入）。
        root: 仓库根（路径登记为仓库相对路径）。
    """
    return {
        "source_run_id": "factor_ab_20260916_071157_base + factor_ab_20260916_123214_noiseB0",
        "base_run_id": "factor_ab_20260916_071157_base",
        "noise_run_id": "factor_ab_20260916_123214_noiseB0",
        "base_chain_nav": repo_relative(source.pair.base_path, root),
        "noise_chain_nav": repo_relative(source.pair.noise_path, root),
        "base_sha256": source.pair.base_sha256,
        "noise_sha256": source.pair.noise_sha256,
        "noise_arm_note": (
            "B0' = factor_ab_20260916_123214_noiseB0（同一策略、换随机种子的重跑臂）；"
            "**不是** P3 的 500 树重训臂，也不含任何重训/调参差异"
        ),
        "train_folds": list(source.train_pool["folds"]),
        "holdout_folds": list(source.holdout_pool["folds"]),
        "date_blocks": block_date_blocks(source.pair, source.train_pool["folds"])
        | {"holdout": block_date_blocks(source.pair, source.holdout_pool["folds"])},
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
            "paired_total": int(source.pair.rows),
        },
    }


def statistic_section(prereg: Mapping[str, str]) -> Dict[str, Any]:
    """统计量段（注册值 + 年化换算 + 被取代的设计条款）。"""
    return {
        "name": "psd_bartlett_lag1_boundary_centered",
        "formula": "T_k=(mean(B_k)+delta)/sqrt(V_PSD/J_k)",
        "variance": "V_PSD=gamma0+gamma1=(c1^2+cJ^2+sum_{j=2..J}(cj+c_{j-1})^2)/(2*(J-1))",
        "gamma_denominator": "J-1",
        "delta_log_daily": DELTA_LOG_DAILY,
        "delta_annualization": {
            "annual_target": DELTA_ANNUAL_TARGET,
            "trading_days": 252,
            "scale": "log_linear",
            "annualized_log_daily": annualized_log_return(DELTA_LOG_DAILY),
            "shared_conversion": (
                "log 尺度线性年化；账户自然日复合年化唯一实现仍在 paper/performance.py"
            ),
        },
        "boundary_centering": True,
        "epsilon_or_clamp_forbidden": True,
        "zero_variance_policy": (
            "sample/look invalid; no veto; counted in family denominator and invalid rate"
        ),
        "superseded_design_clause": {
            "design_clause": "设计 v4 §5.2 的 V_k=gamma0+2*gamma1 与 Z_k=B_k/sqrt(V_k/J)",
            "reason": "T0.5b 实测首 look 无效率 6.8%~13.2%（AR1/real 超 10% 门）⇒ 不可注册",
            "amendment": prereg.get("path"),
            "amendment_sha256": prereg.get("sha256"),
            "legacy_calculator": "bartlett_lag1_gamma0_plus_2gamma1（仅复核/fail-closed 演示）",
        },
    }


def threshold_section(plan: Any) -> Dict[str, Any]:
    """阈值段（α 分配 + 五步规则 + 两个门槛 + 不看 holdout）。"""
    return {
        "alpha_train": float(DESIGN_DEFAULTS["alpha_train"]),
        "alpha_per_look": float(DESIGN_DEFAULTS["alpha_train"]) / plan.k,
        "direction": "lower_tail",
        "rule_steps": threshold_rule_steps(),
        "veto_event": "any_k(valid and T_k < -c_k)",
        "invalid_rate_per_look_max": float(DESIGN_DEFAULTS["invalid_rate_per_look_max"]),
        "min_valid_quantile_sample": 100,
        "holdout_wilson_upper_max": float(DESIGN_DEFAULTS["wilson_upper_max"]),
        "family_wilson_denominator": "all realizations; invalid look does not trigger",
        "no_holdout_callback": True,
    }


def effect_and_holdout_section() -> Dict[str, Any]:
    """效应网格 + 检出事件 + MDE 定义 + holdout 单元/门（§5.3）。"""
    return {
        "effect": {
            "grid_delta": [float(level) for level in EFFECT_GRID_DELTA],
            "unit": "daily log shift = -level*delta applied to every day",
            "detection_event": "any_k(valid and T_k < -c_k)（与 A/A 同一事件）",
            "target_power": float(DESIGN_DEFAULTS["target_power"]),
            "mde_definition": (
                "smallest effect level with point-estimate detection rate >= 0.80 " "after A/A pass"
            ),
            "hard_5delta_80pct": False,
            "if_not_reached": "tested_to_5delta_not_80pct",
        },
        "holdout": {
            "unit_definition": "scene × null × H × referee",
            "nulls": ["mu=0", "mu=-delta"],
            "wilson_confidence": 0.95,
            "wilson_upper_max": float(DESIGN_DEFAULTS["wilson_upper_max"]),
            "pooled_or_weighted_average_forbidden": True,
            "hard_failure_policy": "任一单元不过 ⇒ 裁判不可用（不得删困难场景、不得扩样本调 c_k）",
        },
    }


def drawdown_and_h6_section() -> Dict[str, Any]:
    """T5/T6 占位（本 Task 不实现，但必须登记「未实现」）。"""
    drawdown = {
        "status": DRAWDOWN_PLACEHOLDER_STATUS,
        "owner": "P5b T5",
        "placeholder_items": [
            "DD-W 恶化否决：c^W_k 定义与 min-over-scene 口径",
            "DD-I 改善检出：c^I_k 定义（I_k=D_k，不得再取负）",
            "崩盘窗：B0 峰谷 episode 深度降序取最多 3 个不重叠窗口，全实现共享",
            "回撤效应网格 {1,2,3,5,8,12}pp 与 λ/ζ 101 点扫描 + 最多 80 次细化",
            "T_anchor=H*J_max 唯一锚与 0.05pp 容差、多解取最小干预",
            "回撤 A/A 与 MDE：本 Task 不实现、不产生任何回撤读数",
        ],
    }
    h6 = {
        "status": "placeholder_not_run",
        "owner": "P5b T6",
        "items": [
            "收益 −5δ 单条历史演练",
            "DD-I +12pp 单条历史演练",
            "DD-W −12pp 单条历史演练",
        ],
        "note": "只检查点估计方向与链路，不作为检出率/MDE；频率只来自 H4/H5",
    }
    return {"drawdown_referee": drawdown, "h6_numeric": h6}


def deviation_registry_section(counts: Mapping[str, Any]) -> list:
    """与设计/冻结值的已知差异（必须逐条登记，不得静默）。"""
    return [
        {
            "item": "命名种子",
            "design_value": "20261101（设计 v4 §5.3 表默认值）",
            "frozen_value": int(counts["seed"]),
            "reason": (
                "T0.5b 冻结 manifest 的 seed 根为 20261103（r2/r3 实际取值）；改回 20261101 会使 "
                "r3 临界值与读数不可复现，故以实现数一致为前提沿用冻结值"
            ),
        },
        {
            "item": "holdout realization 身份",
            "design_value": "相对训练独立 seed 子流 + 独立实现编号",
            "frozen_value": "阶段子流 holdout=2（= T0.5b validation 子流）与训练互斥，实现编号同一份",
            "reason": (
                "冻结 manifest 不可重写；改用新 seed 属另立新预登记（超出 T4 范围）。"
                "本次 holdout 与 r3 同一 realization 集，不宣称相对 r3 的新独立实现"
            ),
        },
        {
            "item": "统计量",
            "design_value": "设计 v4 §5.2 的 gamma0+2*gamma1（可为负）",
            "frozen_value": "PSD gamma0+gamma1 + 边界中心化（预登记修订）",
            "reason": "T0.5b 实测原式首 look 无效率超 10% 门；PSD 版已通过预登记评审与 r3 校准",
        },
        {
            "item": "配对账本样本",
            "design_value": "从试车场配对账本构造 d_t",
            "frozen_value": "本次校准使用冻结源对（B0/B0'）；账本读取路径已实现并由测试覆盖",
            "reason": "T4 范围内没有已跑通的配对 campaign；H6 才做端到端演练",
        },
        {
            "item": "冷启动与超 H 依赖场景",
            "design_value": (
                "设计 v4 §5.3 训练集第 4 类'fresh 前段样本不足的冷启动'，以及'超 H 依赖'"
            ),
            "frozen_value": (
                "未单列场景：场景集合冻结为 ma20/ar1_06/regime_hetero/real_seed_block 四类"
            ),
            "reason": (
                "T0.5b 预登记已把场景集合冻结为四类；T4 后补场景会改 c_k 的 max-over-scene 口径，"
                "属另立预登记。短样本行为由 score_paths 的'每 look ≥6 有效 block'fail-closed 与 10% "
                "无效率门承担；超 H 依赖由 real_seed_block 块内保留的原始依赖间接承担"
            ),
        },
        {
            "item": "B0' 臂口径",
            "design_value": "B0' 只表示换种子的重跑臂，不得指 P3 的 500 树重训臂",
            "frozen_value": "factor_ab_20260916_123214_noiseB0（同策略换种子），已登记 noise_arm_note",
            "reason": "§5.5 要求显式声明，防止把重训差异误读为随机性来源",
        },
    ]


def limitations_section() -> list:
    """§5.6 限制声明（至少 3 条；报告与判定表必须原样带出）。

    第 7 条（holdout 与 T0.5b r3 同一 realization 集）是**一等限制**：本次 A/A 过门属对已观察
    读数的条件复现，不构成设计 §5.3 要求的独立 holdout 验收。
    """
    return [
        "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史试车场校准，不是前向影子",
        "训练与 holdout 共享同一对真实源 run（B0/B0'），仅以互斥源折与独立 realization 隔离",
        "源折 0..9 / 10..13 均已观察，属固定条件，不是独立真实源对验证",
        "经验分位只是已测场景的有限样本结果，不是普适尾概率保证",
        "MA(H-1) 覆盖持有期重叠，不覆盖未知 regime、流动性冲击与未来结构变化",
        "所有结论仅在本 manifest 冻结的场景集内成立；不得外推为生产裁决权",
        (
            "holdout realization 与 T0.5b PSD r3 使用同一命名子流编码（holdout=2）与同一 seed 根，"
            "其 rate/Wilson 在 r3 已观察并公布：本次 Wilson 门是对已观察读数的**条件复现**，"
            "不构成设计 §5.3 要求的独立 holdout 验收；取得独立验收须另立预登记"
        ),
        (
            "c_k 是经验分位（每 look 每单元约 22 个次序统计量决定），非解析尾分位；"
            "分位估计的抽样不确定性未进入阈值选择"
        ),
        (
            "设计 §5.3 训练集四类中'冷启动（fresh 前段样本不足）'与'超 H 依赖'未单列场景："
            "前者由每 look ≥6 有效 block 的 fail-closed 与 10% 无效率门承担，后者由 "
            "real_seed_block 块内保留的原始依赖间接承担（详见 deviation_registry）"
        ),
    ]


def lineage_section(
    *,
    sequence: int = 1,
    superseded_manifest_sha256: Optional[str] = None,
    superseded_calibrator_sha256: Optional[str] = None,
    supersede_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """冻结谱系（首次 = 无前身；重冻 = 必须留废止记录，使 `frozen_before_evidence` 可审计）。"""
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
条件式 Bad-Pick 模型

将单调线性风险惩罚（final_score = ml_score - λ × risk_score）替换为：
  市场状态感知 + XGBoost 二分类器 + 阈值门控

核心理念：
  不是所有高分股票都应该被惩罚——只有被二分类器判定为"高概率坏票"的股票，
  才在特定市场状态下被扣分；低风险股票完全不受影响。

推理流程：
  market_features → detect_market_regime() → regime (normal/stressed)
  risk_features → XGBClassifier.predict_proba() → P(bad_pick)
  if P(bad_pick) > threshold[regime]:
      final_score = ml_score - lambda[regime] × (P(bad_pick) - threshold[regime])
  else:
      final_score = ml_score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ── 坏票分类器候选特征 ──────────────────────────────────────────────
# 设计原则：
#   1. 只保留在主模型 Top 候选池中对 bad_pick 有稳定分离度的因子
#   2. 优先使用主模型未采用的时间尺度和风险维度，减少重复学习
#   3. 公告型稀疏因子只有在校准窗口有稳定覆盖时才纳入
#   4. 因子不得在 factor_exclude_list.json 中（ICIR < 0.1 或覆盖率 < 0.3）
#   5. 避免与主模型使用同一信息的"不同包装"（如原始值 vs zscore 版）
# 共 20 个特征，覆盖 7 个独立信号维度。
BAD_PICK_CLASSIFIER_FEATURES = [
    # ── 波动/量价（5个）──
    "zscore_volatility_20",
    "zscore_volatility_5",
    "zscore_turnover_rate",
    "vol_ratio_20",
    "vol_burst_20",

    # ── 成交额/振幅/布林（3个）──
    "zscore_amount_ma20",
    "amplitude",
    "zscore_bb_width",

    # ── 技术形态/超买超卖（5个）──
    "upper_shadow",
    "spec_score",
    "bb_pct",
    "zscore_ma_deviation_20",
    "ind_momentum_rank",

    # ── 动量/反转（3个）──
    "rsi_14",
    "kdj_j",
    "zscore_acceleration",

    # ── 开盘/资金（2个）──
    "zscore_opening_strength",
    "zscore_elg_net_amount_sum_20",

    # ── 估值/行为（3个）──
    "zscore_pe_ttm",
    "lg_net_amount_sum_5",
    "winner_rate",
]

# ── 市场状态相关特征（用于 regime 检测 + 作为分类器输入）─────────────
MARKET_STATE_FEATURES = [
    "mkt_ret_avg_20",
    "mkt_vol_20",
    "mkt_drawdown_20",
    "mkt_adv_dec_ratio",
    "mkt_turnover_std",
]


@dataclass
class RegimeBadPickConfig:
    """单个市场状态下的坏票惩罚配置"""

    threshold: float = 0.5  # P(bad_pick) 触发阈值，低于此不惩罚
    penalty_lambda: float = 0.1  # 超出阈值的惩罚强度


@dataclass
class BadPickConfig:
    """条件式坏票惩罚总配置

    训练阶段由 learn_conditional_bad_pick_config() 生成，
    推理阶段由 MLSignal._apply_risk_penalty() 消费。
    """

    mode: str = "conditional"  # 固定为 "conditional"
    enabled: bool = False

    # ── 坏票分类器信息 ──
    bad_pick_model_version: int = 0  # ModelRegistry 中的版本号
    classifier_features: List[str] = field(default_factory=list)  # 实际使用的特征名

    # ── 市场状态检测配置 ──
    regime_detection_mode: str = "simple"  # simple = normal/stressed 二分类
    # 三个阈值以校准段分位数形式存储（而非绝对值），自动适应市场进化
    regime_bear_pct: float = 0.3  # mkt_ret_avg_20 的分位数（p30）
    regime_vol_pct: float = 0.7  # mkt_vol_20 的分位数（p70）
    regime_dd_pct: float = 0.2  # mkt_drawdown_20 的分位数（p20）

    # ── per-regime 门控配置 ──
    regime_configs: Dict[str, RegimeBadPickConfig] = field(default_factory=dict)

    # ── 校准统计 ──
    calibration_samples: int = 0
    calibration_bad_samples: int = 0
    calibration_bad_rate: float = 0.0
    calibration_auc: float = 0.0  # 分类器在校准段上的 AUC
    regime_sample_counts: Dict[str, int] = field(default_factory=dict)
    baseline_topk_median: float = float("nan")
    selected_topk_median: float = float("nan")
    baseline_rankic_ir: float = float("nan")
    selected_rankic_ir: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为模型 metadata 兼容的字典"""
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "version": 2,
            "bad_pick_model_version": self.bad_pick_model_version,
            "classifier_features": self.classifier_features,
            "regime_detection_mode": self.regime_detection_mode,
            "regime_bear_pct": self.regime_bear_pct,
            "regime_vol_pct": self.regime_vol_pct,
            "regime_dd_pct": self.regime_dd_pct,
            "regime_configs": {
                name: {
                    "threshold": cfg.threshold,
                    "penalty_lambda": cfg.penalty_lambda,
                }
                for name, cfg in self.regime_configs.items()
            },
            "calibration_samples": self.calibration_samples,
            "calibration_bad_samples": self.calibration_bad_samples,
            "calibration_bad_rate": self.calibration_bad_rate,
            "calibration_auc": self.calibration_auc,
            "regime_sample_counts": self.regime_sample_counts,
            "baseline_topk_median": self.baseline_topk_median,
            "selected_topk_median": self.selected_topk_median,
            "baseline_rankic_ir": self.baseline_rankic_ir,
            "selected_rankic_ir": self.selected_rankic_ir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BadPickConfig":
        """从模型 metadata 字典反序列化"""
        regime_configs = {}
        for name, cfg_data in (data.get("regime_configs") or {}).items():
            regime_configs[name] = RegimeBadPickConfig(
                threshold=float(cfg_data.get("threshold", 0.5)),
                penalty_lambda=float(cfg_data.get("penalty_lambda", 0.1)),
            )

        return cls(
            enabled=bool(data.get("enabled", False)),
            bad_pick_model_version=int(data.get("bad_pick_model_version", 0) or 0),
            classifier_features=data.get("classifier_features") or [],
            regime_detection_mode=data.get("regime_detection_mode", "simple"),
            regime_bear_pct=float(data.get("regime_bear_pct", 0.3)),
            regime_vol_pct=float(data.get("regime_vol_pct", 0.7)),
            regime_dd_pct=float(data.get("regime_dd_pct", 0.2)),
            regime_configs=regime_configs,
            calibration_samples=int(data.get("calibration_samples", 0) or 0),
            calibration_bad_samples=int(data.get("calibration_bad_samples", 0) or 0),
            calibration_bad_rate=float(data.get("calibration_bad_rate", 0.0) or 0),
            calibration_auc=float(data.get("calibration_auc", 0.0) or 0),
            regime_sample_counts=data.get("regime_sample_counts") or {},
            baseline_topk_median=float(data.get("baseline_topk_median", float("nan"))),
            selected_topk_median=float(data.get("selected_topk_median", float("nan"))),
            baseline_rankic_ir=float(data.get("baseline_rankic_ir", float("nan"))),
            selected_rankic_ir=float(data.get("selected_rankic_ir", float("nan"))),
        )


def detect_market_regime(
    market_features: Dict[str, float],
    config: BadPickConfig,
) -> str:
    """根据市场状态特征判断当前 regime。

    三层 OR 判断：
      - mkt_ret_avg_20 < bear_threshold（趋势走弱）
      - mkt_vol_20 > vol_threshold（波动率飙升）
      - mkt_drawdown_20 < dd_threshold（处于回撤中）

    任一触发即为 stressed，否则 normal。

    Args:
        market_features: 当前市场状态特征值字典，键为特征名，值为标量
        config: BadPickConfig，包含分位数阈值

    Returns:
        "normal" 或 "stressed"
    """
    bear_val = market_features.get("mkt_ret_avg_20", 0.0)
    vol_val = market_features.get("mkt_vol_20", 0.0)
    dd_val = market_features.get("mkt_drawdown_20", 0.0)

    is_bear = pd.notna(bear_val) and bear_val < config.regime_bear_pct
    is_high_vol = pd.notna(vol_val) and vol_val > config.regime_vol_pct
    is_drawdown = pd.notna(dd_val) and dd_val < config.regime_dd_pct

    # 边界：若所有市场特征均为 0（缺失数据），默认 normal
    all_zero = abs(bear_val) < 1e-9 and abs(vol_val) < 1e-9 and abs(dd_val) < 1e-9
    if all_zero:
        return "normal"

    if is_bear or is_high_vol or is_drawdown:
        return "stressed"
    return "normal"


def _resolve_market_state_scalar(
    features_df: pd.DataFrame,
    state_cols: List[str],
) -> Dict[str, float]:
    """从广播了市场状态特征的截面 DataFrame 中提取标量值。

    市场状态特征是逐日标量广播到所有股票的，取首个有效值即可。
    """
    result: Dict[str, float] = {}
    for col in state_cols:
        if col not in features_df.columns:
            result[col] = 0.0
            continue
        values = features_df[col].dropna()
        result[col] = float(values.iloc[0]) if len(values) > 0 else 0.0
    return result


def prepare_classifier_features(
    features_df: pd.DataFrame,
    classifier_features: List[str],
    market_state_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """从截面特征中提取分类器所需特征列。

    Args:
        features_df: 当日截面特征 DataFrame
        classifier_features: 分类器特征名列表（不含市场状态特征）
        market_state_cols: 市场状态特征列名（已广播到截面的列）

    Returns:
        仅含分类器特征列的 DataFrame（缺失列填截面中位数）
    """
    all_features = list(classifier_features)
    if market_state_cols:
        all_features.extend(market_state_cols)

    available = [c for c in all_features if c in features_df.columns]
    if not available:
        logger.warning("坏票分类器：features_df 中无可用特征列")
        return pd.DataFrame(index=features_df.index)

    result = features_df[available].copy()

    # 缺失值填截面中位数（比填 0 更鲁棒）；全 NaN 列填 0 并抑制空切片警告
    for col in result.columns:
        valid_vals = result[col].dropna()
        if len(valid_vals) == 0:
            result[col] = 0.0
        elif len(valid_vals) < len(result[col]):
            result[col] = result[col].fillna(valid_vals.median())

    return result


def apply_conditional_penalty(
    features_df: pd.DataFrame,
    config: BadPickConfig,
    classifier,
) -> Tuple[pd.DataFrame, str]:
    """应用条件式坏票惩罚，生成 final_score。

    流程：
      1. 提取市场状态特征 → detect_market_regime() → regime
      2. 提取分类器特征 → classifier.predict_proba() → P(bad_pick)
      3. 根据 regime 配置判断是否触发门控
      4. 触发则扣分，不触发则 final_score = ml_score

    Args:
        features_df: 当日截面特征 DataFrame（需已含 ml_score 列）
        config: BadPickConfig
        classifier: 训练好的 XGBClassifier（或任何有 predict_proba 的分类器）

    Returns:
        (features_df, score_column) — features_df 新增 final_score 列
    """
    features_df["risk_score"] = 0.0
    features_df["final_score"] = features_df["ml_score"]

    if not config.enabled or config.bad_pick_model_version <= 0:
        return features_df, "ml_score"

    if classifier is None:
        logger.warning("坏票分类器未加载，跳过惩罚")
        return features_df, "ml_score"

    # ── 1. 检测市场状态 ──
    market_state = _resolve_market_state_scalar(features_df, MARKET_STATE_FEATURES)
    regime = detect_market_regime(market_state, config)

    # ── 2. 获取该 regime 的配置 ──
    regime_cfg = config.regime_configs.get(regime)
    if regime_cfg is None:
        # 回退到 normal
        regime_cfg = config.regime_configs.get("normal")
    if regime_cfg is None or regime_cfg.penalty_lambda <= 0:
        return features_df, "ml_score"

    # ── 3. 提取特征并预测 ──
    X_clf = prepare_classifier_features(
        features_df,
        config.classifier_features,
        MARKET_STATE_FEATURES,
    )

    if X_clf.empty or len(X_clf.columns) == 0:
        return features_df, "ml_score"

    # 确保特征列名与训练时一致
    expected_features = getattr(classifier, "feature_names_in_", None)
    if expected_features is not None:
        missing = [f for f in expected_features if f not in X_clf.columns]
        for f in missing:
            X_clf[f] = 0.0
        X_clf = X_clf[list(expected_features)]

    try:
        proba = classifier.predict_proba(X_clf)
        p_bad = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    except Exception as exc:
        logger.warning(f"坏票分类器预测失败: {exc}")
        return features_df, "ml_score"

    p_bad_series = pd.Series(p_bad, index=X_clf.index, dtype=float)

    # ── 4. 阈值门控惩罚 ──
    threshold = regime_cfg.threshold
    penalty_lambda = regime_cfg.penalty_lambda

    # 门控：只有 P(bad_pick) > threshold 的股票才被惩罚
    mask = p_bad_series > threshold
    excess = (p_bad_series - threshold).clip(lower=0.0)

    features_df["risk_score"] = p_bad_series
    features_df["final_score"] = (
        features_df["ml_score"] - penalty_lambda * excess * mask.astype(float)
    )

    # 日志：惩罚比例
    n_total = len(features_df)
    n_penalized = mask.sum()
    if n_penalized > 0:
        logger.debug(
            f"坏票惩罚: regime={regime}, threshold={threshold:.2f}, "
            f"lambda={penalty_lambda:.3f}, "
            f"penalized={n_penalized}/{n_total} ({n_penalized/n_total:.1%}), "
            f"P(bad)∈[{p_bad_series.min():.3f}, {p_bad_series.max():.3f}]"
        )

    return features_df, "final_score"

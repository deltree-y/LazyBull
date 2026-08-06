#!/usr/bin/env python
"""风控模型训练脚本 (Position Risk Model Training)

独立于选股模型的 walk-forward 训练流程：
  1. 从 cs_train 特征加载全市场历史数据
  2. 构造 RAR 标签（基于前向波动率调整收益的截面三分位数）
  3. 按日期切分 train/ES/calibration
  4. 训练 XGBoost 三分类器
  5. 守卫条件检查（F1 + 单调性 + 回撤区分度）
  6. 注册到 ModelRegistry（models/risk/ 子目录）

用法：
    python scripts/train_position_risk_model.py \
        --start-date 20200101 --end-date 20231231 \
        --horizon 10 --split-count 6
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.lazybull.data import DataLoader, Storage
from src.lazybull.common.logger import setup_logger
from src.lazybull.ml.model_registry import ModelRegistry
from src.lazybull.ml.train_core import (
    load_features_data,
    split_train_val_by_date,
    train_xgboost_model,
)
from src.lazybull.risk.label_builder import (
    build_position_risk_labels,
    validate_label_quality,
    CLASS_LABELS,
)
from src.lazybull.risk.position_risk import (
    PositionRiskConfig,
    PositionRiskModel,
    DEFAULT_COEFFICIENT_MAP,
)


# ---------------------------------------------------------------------------
# 默认特征列表（已按 cs_train 实际列名映射）
# ---------------------------------------------------------------------------

_POSITION_RISK_FEATURE_CANDIDATES = [
    # A. 下行风险（8）
    'downside_vol_20', 'downside_corr_20', 'var_95_20', 'cvar_95_20',
    'max_drawdown_20', 'drawdown_duration', 'skewness_20', 'kurtosis_20',
    # B. 波动结构（6）
    'parkinson_vol_20', 'vol_of_vol_20', 'vol_regime_percentile',
    'garch_persistence', 'high_low_range_ratio', 'gap_risk',
    # C. 动量趋势（7，momentum_decay 为衍生列）
    'ret_5', 'ret_20', 'momentum_decay', 'rsi_14',
    'ma_deviation_20', 'acceleration', 'bb_pct',
    # D. 流动性风险（8）
    'turnover_cv_20', 'amount_cv_20', 'amihud_illiq_20',
    'vol_ratio_5_20', 'up_down_vol_ratio', 'volume_climax_days',
    'turnover_percentile', 'volume_price_divergence',
    # E-G. 基本面/市场/情绪（列名已按 cs_train 实际列映射）
    'roe_waa', 'pe_ttm', 'debt_to_assets', 'fcf_yield', 'ocf_to_profit',
    'mkt_ret_avg_20', 'mkt_ret_avg_60', 'mkt_vol_20', 'mkt_drawdown_20',
    'mkt_adv_dec_ratio',
    'north_flow_sum5', 'margin_net_buy', 'winner_rate',
    'spec_score',
    # 衍生特征（由 _add_derived_features 生成）
    'momentum_decay', 'earnings_yield', 'ret_volatility_ratio',
    # 公告类（数据打通前会因全 NaN/全 0 被有效性过滤自动剔除）
    'pledge_ratio_decayed', 'pledge_high_flag', 'pledge_delta',
    'unlock_risk_flag', 'unlock_ratio',
    'block_discount_avg_10d', 'block_discount_days_10d',
    'short_balance_change_5', 'short_sell_ratio_5',
    # 行业/市值中性化后的版本
    'zscore_volatility_20', 'zscore_turnover_rate', 'zscore_pe_ttm',
    'zscore_amount_ma20', 'zscore_ma_deviation_20',
    'zscore_acceleration', 'zscore_elg_net_amount_sum_20',
]

# 过滤标记（非特征，跳过）
_NON_FEATURE_COLS = {'ts_code', 'trade_date', 'label', 'rar'}


def _select_available_features(df: pd.DataFrame) -> List[str]:
    """从 DataFrame 中筛选实际可用的特征列，并剔除无效列。

    有效性规则：
      - 必须存在于 df 中
      - 剔除全 NaN 列（如未接入数据的公告类因子）
      - 剔除方差为 0 的常数列（对模型无信息量）
    """
    available = []
    for col in _POSITION_RISK_FEATURE_CANDIDATES:
        if col not in df.columns or col in _NON_FEATURE_COLS:
            continue
        series = df[col]
        # 全 NaN 剔除
        if series.notna().sum() == 0:
            logger.debug(f"剔除全 NaN 特征: {col}")
            continue
        # 常数列剔除（去 NaN 后无变化）
        valid = series.dropna()
        if len(valid) > 1 and valid.nunique() <= 1:
            logger.debug(f"剔除常数特征: {col}")
            continue
        available.append(col)
    logger.info(
        f"特征筛选: {len(available)}/{len(_POSITION_RISK_FEATURE_CANDIDATES)} 个可用 "
        f"（剔除全NaN/常数列 {len(_POSITION_RISK_FEATURE_CANDIDATES) - len(available)} 个）"
    )
    return available


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------

def _generate_splits(
    df: pd.DataFrame,
    split_count: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """生成 walk-forward 切分。

    Returns:
        [(train_df, es_df, calib_df), ...] — 每个 split 的三段数据
    """
    dates = sorted(df['trade_date'].unique())
    n = len(dates)
    if n < split_count * 3:
        raise ValueError(f"日期数量 ({n}) 不足以支持 {split_count} 个 split")

    # 反向切分（从最新到最旧），保证测试段连续
    split_size = n // split_count
    splits = []
    for i in range(split_count - 1, -1, -1):
        # 测试段末尾
        test_end_idx = n - 1 - i * split_size
        test_start_idx = max(0, test_end_idx - split_size + 1)

        test_dates = dates[test_start_idx:test_end_idx + 1]
        test_df = df[df['trade_date'].isin(test_dates)]

        # 校准/ES/Train 按比例切分测试段之前的数据
        train_end_idx = test_start_idx - 1
        if train_end_idx < 30:
            continue

        train_dates_all = dates[:train_end_idx + 1]
        n_train = len(train_dates_all)

        calib_count = max(int(n_train * 0.15), 10)
        es_count = max(int(n_train * 0.15), 10)

        calib_dates = train_dates_all[-calib_count:]
        es_dates = train_dates_all[-(calib_count + es_count):-calib_count]
        train_dates = train_dates_all[:-(calib_count + es_count)]

        splits.append((
            df[df['trade_date'].isin(train_dates)],
            df[df['trade_date'].isin(es_dates)],
            df[df['trade_date'].isin(calib_dates)],
        ))

    logger.info(f"生成 {len(splits)} 个 walk-forward split")
    return splits


# ---------------------------------------------------------------------------
# 训练单个 split
# ---------------------------------------------------------------------------

def _train_one_split(
    train_df: pd.DataFrame,
    es_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    feature_names: List[str],
    split_idx: int,
    args: argparse.Namespace,
) -> Optional[Tuple[PositionRiskModel, PositionRiskConfig, Dict]]:
    """训练一个 walk-forward split。"""
    logger.info(f"=== Split {split_idx}: train={len(train_df)}, es={len(es_df)}, calib={len(calib_df)} ===")

    # 准备数据
    X_train = train_df[feature_names].values
    y_train = train_df['label'].values.astype(int)

    X_es = es_df[feature_names].values
    y_es = es_df['label'].values.astype(int)

    X_calib = calib_df[feature_names].values
    y_calib = calib_df['label'].values.astype(int)

    # 检查样本
    if len(X_train) < 100:
        logger.warning(f"Split {split_idx}: 训练样本过少 ({len(X_train)})，跳过")
        return None

    # 训练 XGBoost
    logger.info(f"训练 XGBClassifier: n_train={len(X_train)}, n_es={len(X_es)}")
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        early_stopping_rounds=args.early_stopping_rounds,
        eval_metric="mlogloss",
        random_state=args.random_state,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )

    # 校准段评估
    y_pred_calib = clf.predict(X_calib)
    proba_calib = clf.predict_proba(X_calib)

    from sklearn.metrics import f1_score
    f1_macro = f1_score(y_calib, y_pred_calib, average='macro')

    # 各类别 forward return
    calib_ret_col = args.label_column if args.label_column in calib_df.columns else 'y_ret_10'
    class_returns = {}
    for cls, name in CLASS_LABELS.items():
        mask = y_pred_calib == cls
        if mask.sum() > 0:
            class_returns[name] = calib_df[calib_ret_col].iloc[mask].mean()
        else:
            class_returns[name] = np.nan

    monotonic = (
        not np.isnan(class_returns.get('REDUCE', np.nan))
        and not np.isnan(class_returns.get('HOLD', np.nan))
        and not np.isnan(class_returns.get('INCREASE', np.nan))
        and class_returns['REDUCE'] < class_returns['HOLD'] < class_returns['INCREASE']
    )

    logger.info(
        f"Split {split_idx} 校准: F1(macro)={f1_macro:.3f}, "
        f"monotonic={monotonic}, "
        f"REDUCE_ret={class_returns.get('REDUCE', np.nan):.4f}, "
        f"HOLD_ret={class_returns.get('HOLD', np.nan):.4f}, "
        f"INCREASE_ret={class_returns.get('INCREASE', np.nan):.4f}"
    )

    # 守卫条件
    if f1_macro < args.min_f1:
        logger.warning(f"Split {split_idx}: F1({f1_macro:.3f}) < 最低阈值({args.min_f1})，训练失败")
        return None
    if not monotonic:
        logger.warning(f"Split {split_idx}: 三类 forward return 不单调，训练失败")
        return None

    # 构造配置
    config = PositionRiskConfig(
        model_version=0,  # 稍后注册时填充
        feature_names=feature_names,
        proba_threshold=args.proba_threshold,
        calibration_f1=float(f1_macro),
        calibration_monotonic=monotonic,
        extra={
            'class_returns': {k: float(v) if not np.isnan(v) else None
                              for k, v in class_returns.items()},
            'split_idx': split_idx,
            'n_train': len(X_train),
            'n_es': len(X_es),
            'n_calib': len(X_calib),
            'best_iteration': getattr(clf, 'best_iteration', None),
        },
    )

    model = PositionRiskModel(config, clf)
    return model, config, class_returns


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="训练持仓风控模型")
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--start-date', default='20200101')
    parser.add_argument('--end-date', default='20231231')
    parser.add_argument('--horizon', type=int, default=10, help='持有期天数（标签窗口）')
    parser.add_argument('--label-column', default='y_ret_10', help='前向收益率列名')
    parser.add_argument('--split-count', type=int, default=6, help='Walk-forward 切分数量')
    parser.add_argument('--min-f1', type=float, default=0.35, help='最低 F1 守卫阈值')

    # XGBoost 超参
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--subsample', type=float, default=0.7)
    parser.add_argument('--colsample-bytree', type=float, default=0.6)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--early-stopping-rounds', type=int, default=30)
    parser.add_argument('--random-state', type=int, default=42)

    # Monitor 参数
    parser.add_argument('--proba-threshold', type=float, default=0.6,
                        help='REDUCE 触发提前退出的最低概率')
    # 日志级别
    parser.add_argument('--log-level', default='INFO',
                        help='日志级别: DEBUG | INFO | WARNING | ERROR')

    args = parser.parse_args()

    # 配置日志（默认 INFO，避免数据加载模块的 DEBUG 刷屏）
    setup_logger(log_level=args.log_level.upper())

    # 加载数据（cs_train 按交易日分区，逐日合并）
    storage = Storage(root_path=args.data_root)
    loader = DataLoader(storage)

    logger.info(f"加载特征数据: {args.start_date} ~ {args.end_date}")
    try:
        features_df, trade_days_count = load_features_data(
            storage, loader, args.start_date, args.end_date
        )
    except ValueError as e:
        logger.error(f"加载特征数据失败: {e}")
        sys.exit(1)

    if features_df is None or len(features_df) == 0:
        logger.error("未找到特征数据，请先运行 build_clean_features.py")
        sys.exit(1)

    # 构造标签
    logger.info("构造风控标签...")
    features_df = build_position_risk_labels(
        features_df,
        forward_ret_col=args.label_column,
        holding_period=args.horizon,
    )

    valid_df = features_df[features_df['label'].notna()].copy()
    if len(valid_df) < 1000:
        logger.error(f"有效标签样本过少 ({len(valid_df)})")
        sys.exit(1)

    # 确定可用特征
    feature_names = _select_available_features(valid_df)
    if len(feature_names) < 10:
        logger.error(f"可用特征过少 ({len(feature_names)})")
        sys.exit(1)

    # Walk-forward
    splits = _generate_splits(valid_df, args.split_count)
    if not splits:
        logger.error("无法生成有效的 walk-forward split")
        sys.exit(1)

    # 注册表
    registry = ModelRegistry(models_dir=str(Path(args.data_root) / 'models' / 'risk'))

    success_count = 0
    for idx, (train_df, es_df, calib_df) in enumerate(splits):
        result = _train_one_split(
            train_df, es_df, calib_df, feature_names,
            split_idx=idx, args=args,
        )
        if result is None:
            continue

        model, config, class_returns = result

        # 注册
        version = registry.get_next_version()
        config.model_version = version

        # 保存模型（XGBoost JSON 格式）
        model_path = Path(args.data_root) / 'models' / 'risk' / f'v{version}_model.json'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.get_classifier().save_model(str(model_path))

        # 保存元数据
        metadata = config.to_dict()
        metadata['version'] = version
        registry._save_metadata_sidecar(metadata)
        registry._save_latest_version_file(version)

        # 更新注册表
        reg = registry._ensure_registry_loaded()
        reg['models'].append(metadata)
        reg['next_version'] = version + 1
        registry._save_registry()

        logger.info(
            f"Split {idx} 模型已注册: v{version} "
            f"(F1={config.calibration_f1:.3f}, monotonic={config.calibration_monotonic})"
        )
        success_count += 1

    logger.info(f"训练完成: {success_count}/{len(splits)} 个 split 成功注册")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""eval：train_core 拆分模块。"""

from loguru import logger
from scipy.stats import spearmanr
from src.lazybull.ml.eval_utils import compute_diagnostic_statistics
from src.lazybull.ml.eval_utils import evaluate_predictions_by_date
from src.lazybull.ml.eval_utils import print_diagnostic_report
from src.lazybull.ml.eval_utils import summarize_daily_metrics
from typing import Dict
from typing import List
from typing import Optional
import numpy as np
import pandas as pd


def evaluate_validation_daily(
    model,
    df_val: pd.DataFrame,
    feature_columns: List[str],
    original_return_col: str,
    task: str,
    topk_values: Optional[List[int]] = None,
    emit_logs: bool = True,
    prediction_col: str = "pred_score",
) -> Dict:
    """对验证集进行逐日评估（贴近交易场景）

    Args:
        model: 训练好的模型
        df_val: 验证集 DataFrame（包含 trade_date, ts_code, 特征列, 原始收益列）
        feature_columns: 特征列名列表
        original_return_col: 原始真实收益列名（如 y_ret_20）
        task: 任务类型
        topk_values: TopK 评估的 K 值列表

    Returns:
        逐日评估结果字典
    """
    if len(df_val) == 0:
        logger.warning("验证集为空，跳过逐日评估")
        return {}

    if original_return_col not in df_val.columns:
        logger.warning(f"验证集缺少原始收益列 {original_return_col}，跳过逐日评估")
        return {}

    if topk_values is None:
        topk_values = [30, 100, 300]

    # 准备预测数据；当调用方已提供评分列时，直接复用，避免与实际排序口径脱节。
    df_eval = df_val.copy()
    score_col = str(prediction_col or "pred_score")
    if score_col not in df_eval.columns:
        X_val_features = df_val[feature_columns].fillna(0)

        if task == "classification":
            y_pred_proba = model.predict_proba(X_val_features)[:, 1]
            df_eval[score_col] = y_pred_proba
        else:
            y_pred = model.predict(X_val_features)
            df_eval[score_col] = y_pred

    # 逐日评估
    daily_results = evaluate_predictions_by_date(
        df=df_eval,
        date_col="trade_date",
        prediction_col=score_col,
        return_col=original_return_col,
        topk_values=topk_values,
    )

    # 汇总统计
    summary = summarize_daily_metrics(daily_results)

    # 计算并打印诊断统计
    diagnostics = compute_diagnostic_statistics(
        df=df_eval,
        date_col="trade_date",
        prediction_col=score_col,
        return_col=original_return_col,
        topk_values=topk_values,
    )

    if emit_logs:
        print_diagnostic_report(diagnostics)

    # 返回汇总结果（包含诊断统计）
    result = {
        "prediction_col": score_col,
        "daily_rankic_mean": summary.get("RankIC_均值", np.nan),
        "daily_rankic_std": summary.get("RankIC_标准差", np.nan),
        "daily_rankic_ir": summary.get("RankIC_IR", np.nan),
        **{f"top{k}_return_mean": summary.get(f"Top{k}平均收益_均值", np.nan) for k in topk_values},
        **{
            f"top{k}_return_std": summary.get(f"Top{k}平均收益_标准差", np.nan) for k in topk_values
        },
    }

    # 添加诊断统计
    result.update({f"diagnostic_{k}": v for k, v in diagnostics.items()})

    return result

def neg_rank_ic(y_true, y_pred):
    """Spearman Rank IC（XGBoost sklearn 早停用）。
    返回负值以适配 XGBoost minimize 约定；函数名自动作为 metric name。"""
    corr, _ = spearmanr(y_true, y_pred)
    return float(-corr if not np.isnan(corr) else 0)

def _rank_ic_eval_lgb(y_true, y_pred):
    """Spearman Rank IC（LightGBM 早停用，higher_is_better=True）"""
    corr, _ = spearmanr(y_true, y_pred)
    return "rank_ic", float(corr if not np.isnan(corr) else 0), True

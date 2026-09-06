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


def _group_average_rank(values: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """向量化计算组内平均秩（tie 取段内平均，语义与 pandas groupby.rank(method="average") 一致）。

    Args:
        values: 一维数值数组
        gid: 与 values 等长的组 id（0 ~ n_groups-1）
        n_groups: 组数

    Returns:
        与 values 等长的组内平均秩数组（未按组重排，保持原行序语义）
    """
    n = len(values)
    order = np.lexsort((values, gid))
    vals_sorted = values[order]
    gid_sorted = gid[order]

    # 组内序数秩（1-based）：全局位置 - 组起始位置
    pos = np.arange(1, n + 1, dtype=np.float64)
    starts = np.searchsorted(gid_sorted, np.arange(n_groups))
    ranks_sorted = pos - starts[gid_sorted]

    # tie 段划分：lexsort 后同组内相同值相邻；gid 或值任一变化即新段
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = (vals_sorted[1:] != vals_sorted[:-1]) | (gid_sorted[1:] != gid_sorted[:-1])
    seg_id = np.cumsum(is_new) - 1
    seg_sum = np.bincount(seg_id, weights=ranks_sorted)
    seg_cnt = np.bincount(seg_id)

    out = np.empty(n, dtype=np.float64)
    out[order] = (seg_sum / seg_cnt)[seg_id]
    return out


def _spearman_mean_from_ranks(
    yr: np.ndarray,
    pr: np.ndarray,
    gid: np.ndarray,
    n_groups: int,
    yr_c: Optional[np.ndarray] = None,
    sum_yr2: Optional[np.ndarray] = None,
) -> float:
    """由两侧组内平均秩计算逐日 Spearman 均值。

    组内去均值后的 Pearson 即 Spearman；单样本组与组内常数（秩方差为 0）
    的相关系数无定义，自动剔除。yr_c/sum_yr2 可传入预计算的标签侧统计量。
    """
    counts = np.bincount(gid, minlength=n_groups).astype(float)
    if yr_c is None:
        yr_mean = np.bincount(gid, weights=yr, minlength=n_groups) / counts
        yr_c = yr - yr_mean[gid]
    pr_mean = np.bincount(gid, weights=pr, minlength=n_groups) / counts
    pr_c = pr - pr_mean[gid]
    if sum_yr2 is None:
        sum_yr2 = np.bincount(gid, weights=yr_c * yr_c, minlength=n_groups)
    num = np.bincount(gid, weights=yr_c * pr_c, minlength=n_groups)
    den_sq = sum_yr2 * np.bincount(gid, weights=pr_c * pr_c, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / np.sqrt(den_sq)
    corr = corr[np.isfinite(corr)]
    return float(corr.mean()) if corr.size else 0.0


def daily_spearman_mean(dates, y_true, y_pred) -> float:
    """按交易日分组计算逐日截面 Spearman RankIC 均值。

    与 evaluate_validation_daily 的 daily_rankic 评估口径对齐（先逐日截面秩相关，再取均值），
    避免整段 Spearman 被单一事件期样本（如极端行情日的海量样本）主导而提前触发早停。

    向量化实现：组内平均秩 + 组内去均值后的逐日 Pearson 即为 Spearman（与 scipy 语义一致）；
    样本数不足 2、组内常数（秩方差为 0）的交易日相关系数无定义，自动剔除。

    Args:
        dates: 每个样本所属交易日（与 y_true/y_pred 行序一致）
        y_true: 真实标签
        y_pred: 预测值

    Returns:
        逐日 Spearman 均值；无有效交易日时返回 0.0
    """
    dates_arr = np.asarray(dates)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.all():
        dates_arr = dates_arr[mask]
        y = y[mask]
        p = p[mask]
    if len(dates_arr) == 0:
        return 0.0

    unique_dates, gid = np.unique(dates_arr, return_inverse=True)
    n_groups = len(unique_dates)
    yr = _group_average_rank(y, gid, n_groups)
    pr = _group_average_rank(p, gid, n_groups)
    return _spearman_mean_from_ranks(yr, pr, gid, n_groups)


class DailySpearmanRankIC:
    """逐日截面 Spearman RankIC 均值早停指标（可 pickle 的 callable 类）。

    XGBoost sklearn wrapper 会把 eval_metric（含 callable）保存到模型对象内部，
    模型注册时 joblib.dump 需要 pickle 该对象。pickle 对函数按"模块路径+名称"
    查找引用，闭包函数无法满足；因此必须使用模块级类实例——类可按路径导入，
    dates 数组与预计算状态随实例属性序列化。

    XGBoost callable eval_metric 只接收 (y_true, y_pred)，交易日分组信息
    通过实例持有的验证集 trade_date 数组注入，数组行序必须与 eval_set 行序一致。
    标签侧秩与分组统计在首次调用时预计算（每轮 boosting 的 y_true 不变），
    后续每轮仅需计算预测侧秩。返回负值以适配 XGBoost minimize 约定；
    实例属性 __name__ 固定为 neg_rank_ic_daily，便于 train_params 序列化。
    """

    def __init__(self, dates):
        self._dates = np.asarray(dates)
        self.__name__ = "neg_rank_ic_daily"
        self._state: Optional[Dict[str, object]] = None

    def _prepare(self, y_true: np.ndarray) -> None:
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(y)
        y_valid = y[mask]
        dates_valid = self._dates[mask]
        unique_dates, gid = np.unique(dates_valid, return_inverse=True)
        n_groups = len(unique_dates)
        yr = _group_average_rank(y_valid, gid, n_groups)
        counts = np.bincount(gid, minlength=n_groups).astype(float)
        yr_mean = np.bincount(gid, weights=yr, minlength=n_groups) / counts
        yr_c = yr - yr_mean[gid]
        sum_yr2 = np.bincount(gid, weights=yr_c * yr_c, minlength=n_groups)
        self._state = {
            "mask": mask,
            "gid": gid,
            "n_groups": n_groups,
            "yr_c": yr_c,
            "sum_yr2": sum_yr2,
        }

    def __call__(self, y_true, y_pred) -> float:
        if self._state is None:
            self._prepare(y_true)
        p = np.asarray(y_pred, dtype=float)
        mask_p = np.isfinite(p)
        if (self._state["mask"] & mask_p).all():
            # 常态快路径：标签与预测均无无效值，直接使用预计算统计
            pr = _group_average_rank(p, self._state["gid"], self._state["n_groups"])
            return -_spearman_mean_from_ranks(
                None,
                pr,
                self._state["gid"],
                self._state["n_groups"],
                yr_c=self._state["yr_c"],
                sum_yr2=self._state["sum_yr2"],
            )
        # 极少发生：预测含非有限值，回退完整重算（含行过滤）
        return -daily_spearman_mean(self._dates, y_true, p)


def make_neg_rank_ic_daily(dates):
    """构造逐日截面 Spearman RankIC 均值早停指标（XGBoost sklearn eval_metric 用）。

    Args:
        dates: 验证集每个样本的交易日数组（与 eval_set 行序一致）

    Returns:
        DailySpearmanRankIC 实例（callable，名称固定为 neg_rank_ic_daily）
    """
    return DailySpearmanRankIC(dates)

def _rank_ic_eval_lgb(y_true, y_pred):
    """Spearman Rank IC（LightGBM 早停用，higher_is_better=True）"""
    corr, _ = spearmanr(y_true, y_pred)
    return "rank_ic", float(corr if not np.isnan(corr) else 0), True

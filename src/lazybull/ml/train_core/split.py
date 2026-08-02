# -*- coding: utf-8 -*-
"""split：train_core 拆分模块。"""

from loguru import logger
from src.lazybull.data import DataLoader
from src.lazybull.data import Storage
from typing import Dict
from typing import List
from typing import Tuple
import math
import pandas as pd


def load_features_data(
    storage: Storage, loader: DataLoader, start_date: str, end_date: str
) -> tuple:
    """加载指定日期区间的特征数据

    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD

    Returns:
        (df, trade_days_count) 元组：合并后的特征 DataFrame 和交易日数量
    """
    logger.info(f"加载特征数据: {start_date} 至 {end_date}")

    # 获取交易日列表
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()

    trade_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"共 {len(trade_dates)} 个交易日")

    # 加载每日特征数据
    all_features = []
    missing_dates = []
    for trade_date in trade_dates:
        features = storage.load_cs_train_day(trade_date)
        if features is not None and len(features) > 0:
            all_features.append(features)
        else:
            logger.debug(f"日期 {trade_date} 没有特征数据")
            missing_dates.append(trade_date)

    if missing_dates:
        logger.info(
            f"共 {len(missing_dates)} 个交易日无特征数据（跳过）: {missing_dates[0]} ~ {missing_dates[-1]}"
        )

    if not all_features:
        raise ValueError(f"指定日期区间内没有特征数据")

    # 合并所有数据
    df = pd.concat(all_features, ignore_index=True)
    logger.info(
        f"成功加载 {len(df)} 条样本（{len(all_features)}/{len(trade_dates)} 个交易日有数据）"
    )

    return df, len(trade_dates)

def split_train_val_by_date(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    date_col: str = "trade_date",
    delta: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """按 trade_date 粒度切分训练集和验证集

    确保同一交易日的所有样本不会被拆分到不同集合，彻底避免截面统计量跨集合污染。
    以唯一交易日列表为单位，最后 ceil(n_dates * val_ratio) 个日期作为验证集。

    Args:
        df: 输入 DataFrame（需包含 date_col 列）
        val_ratio: 验证集比例，默认 0.2
        date_col: 日期列名，默认 trade_date
        delta: 训练集末尾与验证集开头之间的间隔交易日数，用于防止标签前向泄露；
               应设置为标签 horizon（如 y_ret_20 对应 delta=20）。

    Returns:
        (df_train, df_val, stats) 元组：
            - df_train: 训练集 DataFrame
            - df_val: 验证集 DataFrame
            - stats: 包含日期统计信息的字典（train_n_dates/val_n_dates/train_start_date/
                     train_end_date/val_start_date/val_end_date）
    """
    all_dates = sorted(df[date_col].unique())
    n_dates = len(all_dates)

    if n_dates == 0:
        empty_stats = {
            "train_n_dates": 0,
            "val_n_dates": 0,
            "train_start_date": "N/A",
            "train_end_date": "N/A",
            "val_start_date": "N/A",
            "val_end_date": "N/A",
        }
        return df.iloc[:0].copy(), df.iloc[:0].copy(), empty_stats

    n_val_dates = max(1, math.ceil(n_dates * val_ratio))
    n_train_dates = n_dates - n_val_dates

    if n_train_dates <= 0:
        n_train_dates = 0
        n_val_dates = n_dates

    # 训练集末尾扣除 delta 天间隔，防止标签泄漏到验证集
    if n_train_dates > delta:
        train_dates_set = set(all_dates[: n_train_dates - delta])
    else:
        # 训练日期数不足以扣除 delta 间隔，训练集将无隔离带
        logger.warning(
            f"训练/验证集分割: 训练日期数({n_train_dates}) <= delta({delta})，"
            f"无法从训练集末尾扣除隔离带，训练集保持完整"
        )
        train_dates_set = set(all_dates[:n_train_dates])

    if n_train_dates + delta < n_dates:
        val_dates_set = set(all_dates[n_train_dates + delta :])
    else:
        # 数据不足以保留完整 delta 间隔，存在标签泄漏风险
        # 为避免训练结果不可靠，返回空验证集而非使用有泄漏的数据
        logger.error(
            f"训练/验证集分割: 数据不足以保留 {delta} 天间隔 "
            f"(总日期={n_dates}, 训练={n_train_dates}, delta={delta})，"
            f"放弃验证集以避免标签泄漏"
        )
        val_dates_set = set()

    df_train = df[df[date_col].isin(train_dates_set)].copy()
    df_val = df[df[date_col].isin(val_dates_set)].copy()

    actual_train_n_dates = len(train_dates_set)
    sorted_train_dates = sorted(train_dates_set)
    sorted_val_dates = sorted(val_dates_set)
    stats = {
        "train_n_dates": actual_train_n_dates,  # 实际参与训练的日期数（已扣除末尾 delta 天间隔）
        "val_n_dates": len(val_dates_set),
        "train_start_date": str(sorted_train_dates[0]) if sorted_train_dates else "N/A",
        "train_end_date": str(sorted_train_dates[-1]) if sorted_train_dates else "N/A",
        "val_start_date": str(sorted_val_dates[0]) if sorted_val_dates else "N/A",
        "val_end_date": str(sorted_val_dates[-1]) if sorted_val_dates else "N/A",
    }

    logger.info(f"按 trade_date 粒度切分（共 {n_dates} 个交易日，delta={delta} 天间隔）:")
    logger.info(
        f"  训练集: {stats['train_start_date']} 至 {stats['train_end_date']}"
        f"（{actual_train_n_dates} 个交易日，{len(df_train)} 条样本）"
    )
    logger.info(
        f"  验证集: {stats['val_start_date']} 至 {stats['val_end_date']}"
        f"（{stats['val_n_dates']} 个交易日，{len(df_val)} 条样本）"
    )

    return df_train, df_val, stats

def split_val_for_early_stopping_by_date(
    df_val: pd.DataFrame,
    embargo_days: int,
    date_col: str = "trade_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """将验证集尾部按交易日隔离，避免用于 early stopping 的标签与测试期重叠

    Args:
        df_val: 原始验证集 DataFrame（通常由 split_train_val_by_date 产生）
        embargo_days: 需要从验证集尾部隔离的交易日数
        date_col: 日期列名，默认 trade_date

    Returns:
        (df_val_es, df_val_embargo, stats):
            - df_val_es: 用于 early stopping / best_iteration 的验证子集
            - df_val_embargo: 从验证集尾部隔离出的样本
            - stats: 切分统计信息
    """

    def _date_range(dates: List) -> Tuple[str, str]:
        if not dates:
            return "N/A", "N/A"
        return str(dates[0]), str(dates[-1])

    if len(df_val) == 0:
        empty_stats = {
            "val_raw_n_dates": 0,
            "val_raw_start_date": "N/A",
            "val_raw_end_date": "N/A",
            "val_raw_samples": 0,
            "val_es_n_dates": 0,
            "val_es_start_date": "N/A",
            "val_es_end_date": "N/A",
            "val_es_samples": 0,
            "val_embargo_n_dates": 0,
            "val_embargo_start_date": "N/A",
            "val_embargo_end_date": "N/A",
            "val_embargo_samples": 0,
            "val_embargo_days_requested": max(int(embargo_days), 0),
            "val_embargo_days_applied": 0,
        }
        return df_val.iloc[:0].copy(), df_val.iloc[:0].copy(), empty_stats

    raw_dates = sorted(df_val[date_col].unique())
    raw_start, raw_end = _date_range(raw_dates)
    embargo_days_requested = max(int(embargo_days), 0)

    if embargo_days_requested <= 0:
        df_val_es = df_val.copy()
        df_val_embargo = df_val.iloc[:0].copy()
        es_dates = raw_dates
        embargo_dates = []
    else:
        embargo_n_dates = min(embargo_days_requested, len(raw_dates))
        if embargo_n_dates >= len(raw_dates):
            es_dates = []
            embargo_dates = raw_dates
        else:
            es_dates = raw_dates[:-embargo_n_dates]
            embargo_dates = raw_dates[-embargo_n_dates:]

        es_dates_set = set(es_dates)
        embargo_dates_set = set(embargo_dates)
        df_val_es = df_val[df_val[date_col].isin(es_dates_set)].copy()
        df_val_embargo = df_val[df_val[date_col].isin(embargo_dates_set)].copy()

    es_start, es_end = _date_range(es_dates)
    embargo_start, embargo_end = _date_range(embargo_dates)
    stats = {
        "val_raw_n_dates": len(raw_dates),
        "val_raw_start_date": raw_start,
        "val_raw_end_date": raw_end,
        "val_raw_samples": len(df_val),
        "val_es_n_dates": len(es_dates),
        "val_es_start_date": es_start,
        "val_es_end_date": es_end,
        "val_es_samples": len(df_val_es),
        "val_embargo_n_dates": len(embargo_dates),
        "val_embargo_start_date": embargo_start,
        "val_embargo_end_date": embargo_end,
        "val_embargo_samples": len(df_val_embargo),
        "val_embargo_days_requested": embargo_days_requested,
        "val_embargo_days_applied": len(embargo_dates),
    }

    if len(df_val_es) == 0 and len(df_val) > 0 and embargo_days_requested > 0:
        logger.warning(
            f"验证集尾部隔离后用于 early stopping 的样本为空 "
            f"(raw_n_dates={len(raw_dates)}, embargo_days={embargo_days_requested})"
        )

    return df_val_es, df_val_embargo, stats

def split_val_for_selection_protocol_by_date(
    df_val: pd.DataFrame,
    embargo_days: int,
    date_col: str = "trade_date",
    calibration_ratio: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """将验证集拆成 early stopping / calibration / embargo 三段。

    目的：
    - `df_val_es` 仅用于训练期间 early stopping / best_iteration；
    - `df_val_calib` 仅用于候选比较、验证评估与稳定性诊断；
    - `df_val_embargo` 继续隔离尾部，避免与测试期价格窗口重叠。

    默认在扣除 embargo 后，将剩余验证日期的后 25% 划为 calibration，
    至少保留 1 个 calibration 交易日，并确保 early stopping 至少保留 1 个交易日。
    当样本过短时，自动退化为“只有 early stopping，没有 calibration”。
    """

    def _date_range(dates: List) -> Tuple[str, str]:
        if not dates:
            return "N/A", "N/A"
        return str(dates[0]), str(dates[-1])

    if len(df_val) == 0:
        empty_stats = {
            "val_raw_n_dates": 0,
            "val_raw_start_date": "N/A",
            "val_raw_end_date": "N/A",
            "val_raw_samples": 0,
            "val_es_n_dates": 0,
            "val_es_start_date": "N/A",
            "val_es_end_date": "N/A",
            "val_es_samples": 0,
            "val_calib_n_dates": 0,
            "val_calib_start_date": "N/A",
            "val_calib_end_date": "N/A",
            "val_calib_samples": 0,
            "val_embargo_n_dates": 0,
            "val_embargo_start_date": "N/A",
            "val_embargo_end_date": "N/A",
            "val_embargo_samples": 0,
            "val_embargo_days_requested": max(int(embargo_days), 0),
            "val_embargo_days_applied": 0,
            "val_calibration_ratio": float(calibration_ratio),
        }
        empty = df_val.iloc[:0].copy()
        return empty, empty, empty, empty_stats

    raw_dates = sorted(df_val[date_col].unique())
    raw_start, raw_end = _date_range(raw_dates)
    embargo_days_requested = max(int(embargo_days), 0)
    calibration_ratio = min(max(float(calibration_ratio), 0.0), 0.5)

    if embargo_days_requested <= 0:
        candidate_dates = raw_dates
        embargo_dates = []
    else:
        embargo_n_dates = min(embargo_days_requested, len(raw_dates))
        if embargo_n_dates >= len(raw_dates):
            candidate_dates = []
            embargo_dates = raw_dates
        else:
            candidate_dates = raw_dates[:-embargo_n_dates]
            embargo_dates = raw_dates[-embargo_n_dates:]

    if len(candidate_dates) <= 1:
        es_dates = candidate_dates
        calib_dates = []
    else:
        requested_calib_dates = max(1, math.ceil(len(candidate_dates) * calibration_ratio))
        calib_n_dates = min(requested_calib_dates, len(candidate_dates) - 1)
        es_dates = candidate_dates[:-calib_n_dates]
        calib_dates = candidate_dates[-calib_n_dates:]

    es_dates_set = set(es_dates)
    calib_dates_set = set(calib_dates)
    embargo_dates_set = set(embargo_dates)
    df_val_es = df_val[df_val[date_col].isin(es_dates_set)].copy()
    df_val_calib = df_val[df_val[date_col].isin(calib_dates_set)].copy()
    df_val_embargo = df_val[df_val[date_col].isin(embargo_dates_set)].copy()

    es_start, es_end = _date_range(es_dates)
    calib_start, calib_end = _date_range(calib_dates)
    embargo_start, embargo_end = _date_range(embargo_dates)
    stats = {
        "val_raw_n_dates": len(raw_dates),
        "val_raw_start_date": raw_start,
        "val_raw_end_date": raw_end,
        "val_raw_samples": len(df_val),
        "val_es_n_dates": len(es_dates),
        "val_es_start_date": es_start,
        "val_es_end_date": es_end,
        "val_es_samples": len(df_val_es),
        "val_calib_n_dates": len(calib_dates),
        "val_calib_start_date": calib_start,
        "val_calib_end_date": calib_end,
        "val_calib_samples": len(df_val_calib),
        "val_embargo_n_dates": len(embargo_dates),
        "val_embargo_start_date": embargo_start,
        "val_embargo_end_date": embargo_end,
        "val_embargo_samples": len(df_val_embargo),
        "val_embargo_days_requested": embargo_days_requested,
        "val_embargo_days_applied": len(embargo_dates),
        "val_calibration_ratio": calibration_ratio,
    }

    if len(df_val_es) == 0 and len(candidate_dates) > 0:
        logger.warning(
            "验证集协议拆分后 early stopping 子集为空，训练将退化为无验证集早停"
        )

    return df_val_es, df_val_calib, df_val_embargo, stats

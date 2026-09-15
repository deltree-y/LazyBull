# -*- coding: utf-8 -*-
"""因子体检数据扫描：覆盖率、逐日 RankIC 与相关性矩阵采样。

只读取 cs_train 特征分区，不做任何写回；分区按固定步长采样以控制耗时。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ScanResult:
    """扫描产物。"""

    daily_ic: pd.DataFrame  # 列: date, feature, ic, n
    coverage_by_date: pd.DataFrame  # 索引=date，列=feature（非空占比）
    market_vol_by_date: pd.Series  # 索引=date，值为当日市场波动中位数
    corr_avg: Optional[pd.DataFrame]  # 采样日平均相关矩阵（Spearman）


def pick_partition_files(cs_train_dir: Path, start: str, end: str, every: int) -> List[Path]:
    """按区间与步长挑选特征分区文件（YYYYMMDD.parquet）。"""
    files = sorted(
        path for path in Path(cs_train_dir).glob("*.parquet") if start <= path.stem <= end
    )
    if not files:
        raise FileNotFoundError(f"{cs_train_dir} 在 {start}~{end} 无特征分区")
    return files[:: max(int(every), 1)]


def pick_corr_dates(
    files: Sequence[Path], years: Iterable[int], month_days: Sequence[str] = ("0415", "1015")
) -> List[str]:
    """为每个目标年份挑选两个相关性采样日。

    从给定的采样文件列表中取「不早于当年 `0415/1015` 的首个采样日」；
    当年无满足条件的采样日则跳过（部分年份的分区尚未生成时属正常情况）。
    """
    dates: List[str] = []
    for year in years:
        for month_day in month_days:
            target = f"{year}{month_day}"
            candidate = next((path.stem for path in files if path.stem >= target), None)
            if candidate is not None:
                dates.append(candidate)
    return sorted(set(dates))


def compute_daily_rank_ic(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    label_column: str,
    min_pairs: int = 200,
) -> Dict[str, float]:
    """计算单日截面 RankIC（Spearman，等价于秩变换后的 Pearson）。

    有效配对不足或方差为零的特征将被跳过（不计入结果）。
    """
    ranked = frame[list(feature_names) + [label_column]].rank(numeric_only=True)
    label_rank = ranked[label_column]
    label_valid = label_rank.notna()
    result: Dict[str, float] = {}
    for name in feature_names:
        column = ranked[name]
        mask = column.notna() & label_valid
        if int(mask.sum()) < min_pairs:
            continue
        x = column[mask].to_numpy(dtype=float)
        y = label_rank[mask].to_numpy(dtype=float)
        if x.std() == 0 or y.std() == 0:
            continue
        result[name] = float(np.corrcoef(x, y)[0, 1])
    return result


def compute_coverage(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.Series:
    """计算各特征当日非空占比。"""
    return frame[list(feature_names)].notna().mean()


def average_correlation(matrices: Sequence[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """对多个采样日的相关矩阵求平均（NaN 视为 0 后平均）。"""
    if not matrices:
        return None
    total = sum(matrix.fillna(0.0) for matrix in matrices)
    return total / len(matrices)


def scan_features(
    files: Sequence[Path],
    feature_names: Sequence[str],
    label_column: str,
    market_vol_column: str,
    corr_dates: Sequence[str],
    min_pairs: int = 200,
    corr_min_periods: int = 100,
    code_filter: Optional[set] = None,
    progress_every: int = 40,
) -> ScanResult:
    """扫描采样分区：覆盖率、逐日 RankIC、相关性矩阵、市场波动。"""
    feature_names = list(feature_names)
    daily_records: List[Dict[str, object]] = []
    coverage_records: Dict[str, pd.Series] = {}
    market_vol_records: Dict[str, float] = {}
    corr_matrices: List[pd.DataFrame] = []
    corr_date_set = set(corr_dates)

    for index, path in enumerate(files):
        columns = list(
            dict.fromkeys(
                ["trade_date", "ts_code", label_column, market_vol_column] + feature_names
            )
        )
        frame = pd.read_parquet(path, columns=columns)
        if code_filter is not None:
            frame = frame[frame["ts_code"].astype(str).isin(code_filter)]
        if frame.empty:
            continue

        date = path.stem
        coverage_records[date] = compute_coverage(frame, feature_names)
        if market_vol_column in frame.columns:
            market_vol = frame[market_vol_column].median()
            market_vol_records[date] = float(market_vol) if pd.notna(market_vol) else np.nan

        for name, ic in compute_daily_rank_ic(
            frame, feature_names, label_column, min_pairs
        ).items():
            daily_records.append({"date": date, "feature": name, "ic": ic})

        if date in corr_date_set:
            ranked = frame[feature_names].rank(numeric_only=True)
            corr_matrices.append(ranked.corr(method="pearson", min_periods=corr_min_periods))

        if progress_every and (index + 1) % progress_every == 0:
            logger.info(f"  扫描进度 {index + 1}/{len(files)}")

    daily_ic = pd.DataFrame(daily_records)
    if daily_ic.empty:
        raise ValueError("采样区间内未计算出任何有效 RankIC，请检查特征清单与标签列")

    coverage_by_date = pd.DataFrame(coverage_records).T
    coverage_by_date.index.name = "date"
    market_vol_by_date = pd.Series(market_vol_records, name=market_vol_column)
    market_vol_by_date.index.name = "date"
    return ScanResult(
        daily_ic=daily_ic,
        coverage_by_date=coverage_by_date,
        market_vol_by_date=market_vol_by_date,
        corr_avg=average_correlation(corr_matrices),
    )

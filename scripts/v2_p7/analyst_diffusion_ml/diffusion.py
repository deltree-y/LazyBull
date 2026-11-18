"""90 日扩散状态：每分析师一票的净扩散、上调占比、中位修正与广度。

口径（协议 §5.4 / §6）：
- 窗口 = `(T-90 自然日, T]`；
- 每一类指标，同一分析师只保留其**最新一次有效可比 revision**，
  up/down/flat 全部计入分母，不得只留正向或非零事件；
- `N_active` = 窗口内至少一篇有效 FY0/FY1 主预测的不同分析师团队数；
- 分母为 0 时特征写 NaN（不填 0）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import spec

NEW_COLUMNS: tuple[str, ...] = spec.NEW_FEATURES
DIRECTIONS: dict[str, int] = {"up": 1, "down": -1, "flat": 0}


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def _with_ord(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_ord"] = pd.to_datetime(work["report_date"].astype(str), format="%Y%m%d").map(
        pd.Timestamp.toordinal
    )
    return work


def _vote_matrices(
    frame: pd.DataFrame,
    day_ord: np.ndarray,
    window: int,
    magnitude_column: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单股票单类指标：返回 (方向矩阵, 幅度矩阵, 有效位矩阵)。

    方向矩阵取值 +1/-1/0，无效为 0 且有效位为 False。
    """
    n_dates = len(day_ord)
    if frame is None or len(frame) == 0:
        empty = np.zeros((0, n_dates), dtype=np.int8)
        return empty, np.zeros((0, n_dates), dtype=np.float32), np.zeros((0, n_dates), dtype=bool)

    work = _with_ord(frame)
    teams = work["analyst_team"].astype(str).to_numpy()
    ords = work["_ord"].to_numpy(dtype=np.int64)
    directions = work["direction"].map(DIRECTIONS).astype(np.int16).to_numpy()
    magnitudes = (
        work[magnitude_column].to_numpy(dtype=float)
        if magnitude_column is not None
        else np.zeros(len(work), dtype=float)
    )

    unique, team_code = np.unique(teams, return_inverse=True)
    n_teams = len(unique)
    dir_matrix = np.zeros((n_teams, n_dates), dtype=np.int8)
    mag_matrix = np.zeros((n_teams, n_dates), dtype=np.float32)
    valid_matrix = np.zeros((n_teams, n_dates), dtype=bool)

    # 先按 (分析师, 报告日) 排序，再按分析师切块，避免逐组 pandas 开销
    order = np.lexsort((ords, team_code))
    team_sorted = team_code[order]
    ords_sorted = ords[order]
    dir_sorted = directions[order]
    mag_sorted = magnitudes[order]
    boundaries = np.flatnonzero(np.diff(team_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(team_sorted)]))

    for start, end in zip(starts, ends):
        rows = int(team_sorted[start])
        team_ords = ords_sorted[start:end]
        position = np.searchsorted(team_ords, day_ord, side="right") - 1
        valid = position >= 0
        picked = np.where(valid, position, 0)
        fresh = np.zeros(n_dates, dtype=bool)
        # 窗口是半开区间 (T-90, T]：age < window 才是"命中"，
        # 与 `analyst_coverage_matrix` 的 searchsorted(T-90, side="right") 同口径。
        fresh[valid] = (day_ord[valid] - team_ords[picked[valid]]) < window
        if not fresh.any():
            continue
        dir_matrix[rows, fresh] = dir_sorted[start:end][picked[fresh]]
        mag_matrix[rows, fresh] = mag_sorted[start:end][picked[fresh]]
        valid_matrix[rows, fresh] = True
    return dir_matrix, mag_matrix, valid_matrix


def _summmarise(
    dir_matrix: np.ndarray, valid_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """方向矩阵 -> (上调数, 下调数, 有效总数)。"""
    if dir_matrix.size == 0:
        zeros = np.zeros(dir_matrix.shape[1], dtype=np.int16)
        return zeros, zeros.copy(), zeros.copy()
    up = ((dir_matrix == 1) & valid_matrix).sum(axis=0).astype(np.int16)
    down = ((dir_matrix == -1) & valid_matrix).sum(axis=0).astype(np.int16)
    total = valid_matrix.sum(axis=0).astype(np.int16)
    return up, down, total


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """分母为 0 时写 NaN（不填 0）。"""
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    mask = denominator > 0
    out[mask] = numerator[mask].astype(np.float32) / denominator[mask].astype(np.float32)
    return out


def _median_matrix(mag_matrix: np.ndarray, valid_matrix: np.ndarray, n_dates: int) -> np.ndarray:
    """逐日跨分析师幅度中位数（无效位置不参与；全空日写 NaN）。"""
    import warnings

    out = np.full(int(n_dates), np.nan, dtype=np.float32)
    if mag_matrix.size == 0:
        return out
    masked = np.where(valid_matrix, mag_matrix, np.nan)
    present = valid_matrix.any(axis=0)
    if not present.any():
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(masked, axis=0)
    out[present] = medians[present].astype(np.float32)
    return out


def compute_features(
    events: dict[str, pd.DataFrame],
    dates: Sequence[str],
    codes: Sequence[str],
    mask: np.ndarray,
    n_active: np.ndarray,
    report_age: np.ndarray,
    *,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """对 `mask` 内的 (T, 股票) 计算 8 个扩散特征与全部分母。

    Args:
        events: `report_events.build_events` 的返回值。
        dates/codes: 宽表轴（与 `mask` 一致）。
        mask: 需要输出的 (T, 股票) 布尔矩阵（通常 = 分析师合格域）。
        n_active: 90 日活跃分析师数矩阵（int16）。
        report_age: 最新研报自然日年龄矩阵（float32）。
    """
    emit = _emit(log)
    day_ord = pd.to_datetime(pd.Series(list(dates)).astype(str), format="%Y%m%d").map(
        pd.Timestamp.toordinal
    ).to_numpy(dtype=np.int64)
    window = int(spec.REPORT["diffusion_window_days"])
    code_index = {str(code): j for j, code in enumerate(codes)}

    eps = events["eps_events"]
    target = events["target_events"]
    rating = events["rating_events"]
    eps_by_stock = (
        {code: part for code, part in eps.groupby("ts_code", sort=False)}
        if len(eps)
        else {}
    )
    target_by_stock = (
        {code: part for code, part in target.groupby("ts_code", sort=False)} if len(target) else {}
    )
    rating_by_stock = (
        {code: part for code, part in rating.groupby("ts_code", sort=False)} if len(rating) else {}
    )

    rows, cols = np.nonzero(mask)
    total = len(rows)
    names = list(NEW_COLUMNS)
    columns: dict[str, np.ndarray] = {
        name: np.full(total, np.nan, dtype=np.float32) for name in names
    }
    denominators: dict[str, np.ndarray] = {
        name: np.zeros(total, dtype=np.int16) for name in spec.DENOMINATOR_COLUMNS
    }
    position = np.full(mask.shape, -1, dtype=np.int32)
    position[rows, cols] = np.arange(total, dtype=np.int32)

    processed = 0
    for code, j in code_index.items():
        slot = position[:, j]
        take = slot >= 0
        if not take.any():
            continue
        eps_dir, eps_mag, eps_valid = _vote_matrices(
            eps_by_stock.get(code), day_ord, window, "eps_rel"
        )
        target_dir, _, target_valid = _vote_matrices(
            target_by_stock.get(code), day_ord, window, None
        )
        rating_dir, _, rating_valid = _vote_matrices(
            rating_by_stock.get(code), day_ord, window, None
        )
        eps_up, eps_down, eps_total = _summmarise(eps_dir, eps_valid)
        target_up, target_down, target_total = _summmarise(target_dir, target_valid)
        rating_up, rating_down, rating_total = _summmarise(rating_dir, rating_valid)
        eps_median = _median_matrix(eps_mag, eps_valid, len(day_ord))

        active = n_active[:, j].astype(np.int32)
        features = {
            "ad_eps_net_diffusion_90": _safe_ratio(eps_up - eps_down, eps_total),
            "ad_eps_up_ratio_90": _safe_ratio(eps_up, eps_total),
            "ad_eps_revision_median_90": eps_median,
            "ad_eps_revision_breadth_90": _safe_ratio(
                eps_total, np.where(active > 0, active, 0)
            ),
            "ad_target_net_diffusion_90": _safe_ratio(target_up - target_down, target_total),
            "ad_rating_net_diffusion_90": _safe_ratio(rating_up - rating_down, rating_total),
            "ad_active_analysts_90_log": np.log1p(active).astype(np.float32),
            "ad_latest_report_age_days": report_age[:, j].astype(np.float32),
        }
        counts = {
            "ad_n_active_90": active,
            "ad_n_eps_90": eps_total,
            "ad_n_eps_up_90": eps_up,
            "ad_n_eps_down_90": eps_down,
            "ad_n_eps_flat_90": eps_total - eps_up - eps_down,
            "ad_n_target_90": target_total,
            "ad_n_target_up_90": target_up,
            "ad_n_target_down_90": target_down,
            "ad_n_rating_90": rating_total,
            "ad_n_rating_up_90": rating_up,
            "ad_n_rating_down_90": rating_down,
        }
        for name, values in features.items():
            columns[name][slot[take]] = values[take]
        for name, values in counts.items():
            denominators[name][slot[take]] = values[take]
        processed += 1
        if processed % 500 == 0:
            emit(f"diffusion 进度：{processed} 只股票")

    out = pd.DataFrame(
        {
            "trade_date": np.asarray(dates, dtype=object)[rows],
            "ts_code": np.asarray(codes, dtype=object)[cols],
        }
    )
    for name in names:
        out[name] = columns[name]
    for name in spec.DENOMINATOR_COLUMNS:
        out[name] = denominators[name]
    emit(f"diffusion 完成：{len(out)} 行，{processed} 只股票")
    return out


def feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """逐列 × 逐年覆盖率（NaN 保留，但必须可量化）。"""
    work = frame.copy()
    work["year"] = work["trade_date"].astype(str).str[:4]
    rows = []
    for year, part in work.groupby("year"):
        row: dict[str, Any] = {"year": year, "rows": int(len(part))}
        for name in spec.FEATURES:
            row[name] = float(part[name].notna().mean()) if name in part else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

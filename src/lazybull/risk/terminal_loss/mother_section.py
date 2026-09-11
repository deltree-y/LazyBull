"""pct_* 完整母截面重建（方案 4.4）

背景：``cs_train`` 分区行已按 ``y_ret_*`` 标签有效性过滤（实测同日
4698/5344 ≈ 88%），在该子集上算 ``pct_*`` 会把分母收窄 5%~10%，违反方案
4.4「cs_train 已按未来标签有效性删行时，不能直接在删行后重建百分位，
应从标签过滤前的完整特征母截面或可复现的同日独立特征快照生成相对特征」。

本模块从 ``clean/daily`` 全量化重建**标签过滤前的完整同日截面**，并为
``pct_*`` 的四个基列给出与特征流水线**同一实现**的取值：

- ``cvar_95_20`` / ``max_drawdown_20`` / ``amihud_illiq_20``：复用
  ``risk/precompute.py::precompute_risk_factors``（FeatureBuilder 的主路径，
  ``cs_train`` 中同名列即由它产出）；
- ``ret_20``：复用 ``features/builder/static_core.py::
  _calculate_window_features_static``（与 ``cs_train`` 中 ``ret_20`` 同一实现，
  含观测数不足置 NaN 的契约）。

母截面证券域 = 当日 ``clean/daily`` 有行的全部股票（PIT 口径，不用今日
存续名单回筛历史）；因子值不可得时为 NaN（不剔除行，分母保持完整）。

一致性自检：``validate_mother_section_against_cs_train`` 在交集上逐值比对
母截面与 ``cs_train`` 同名列，并按两类原因分开处置：

- **实现漂移**（母截面换了另一套实现）：差异是**大面积**的——超容差行
  占总交集行数比例超 ``MOTHER_OUTLIER_SHARE_LIMIT``，或某列最大绝对差
  超 ``MOTHER_HARD_ATOL``（拦"稀疏但幅度巨大"），直接报错。
- **数据态漂移**（cs_train 分区由修订前的 clean/daily 产出，如复权因子/
  收益被修订）：差异是**极稀疏**的。此类不阻断训练，但必须告警并登记进
  元数据 ``mother_section_validation.outliers``，禁止静默掩过。

判据只用**行占比 + 幅度**，不用"涉及日期数/日占比"：一只股票的某日修订会
在其后 20 日窗口内连续重现（实测 300630.SZ 连续 10 天），日占比随窗口长短
剧烈变化（10/123 日 vs 10/851 日），是伪影而非信号。

历史窗口必须与特征流水线一致（见 ``MOTHER_HISTORY_MONTHS``）：收益按股票
自身可用行计算，窗口过短会让"停牌复牌日"当天的收益在母截面为 NaN 而在
cs_train 中有效，产生**稀疏但连续**的假漂移（实测 300630.SZ 停牌
20240430→20240708：旧 30 交易日窗口取不到 20240430 行，0708 收益变 NaN，
cvar_95_20 退化成 0709 的 -0.196356，而 cs_train 为 -0.2）。
"""

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

#: 参与 pct_* 的四个基列（方案 3.2 第 29~32 项）
MOTHER_SECTION_FACTORS: List[str] = [
    "ret_20",
    "cvar_95_20",
    "max_drawdown_20",
    "amihud_illiq_20",
]

#: 母截面历史窗口（自然月）——必须与特征流水线同一规则：
#: ``features/pipeline.py`` 加载 ``daily_clean`` 的预热窗口也是 ``start_date - 7 个月``。
#: 逐股收益取股票自身上一可用行（停牌期间无行），窗口过短会把停牌前那一行切
#: 出窗外，使复牌日收益在母截面为 NaN 而 cs_train 中有效（详见模块 docstring）。
MOTHER_HISTORY_MONTHS = 7

#: 与 cs_train 一致性校验容差（cs_train 为 float32，留浮点余量）
MOTHER_VALIDATION_ATOL = 1e-6

#: 实现漂移判据之一：超容差行占总交集行数比例上限（超过即判为实现差异）
MOTHER_OUTLIER_SHARE_LIMIT = 1e-4

#: 实现漂移判据之二：任一列最大绝对差硬上限（拦"稀疏但幅度巨大"的漂移）
MOTHER_HARD_ATOL = 0.05

#: 登记进元数据的离群样例条数上限
MOTHER_OUTLIER_EXAMPLES = 5

#: clean/daily 必需列（缺则母截面无法重建）：vol/amount 为 ret_20
#: 共享实现（_calculate_window_features_static）的必需输入
_CLEAN_DAILY_REQUIRED = ["ts_code", "trade_date", "close_adj", "vol", "amount"]

#: clean/daily 可选列（缺失时对应因子输出 NaN，不视为链路失败）
_CLEAN_DAILY_OPTIONAL = ["open_adj", "high_adj", "low_adj"]


def _clean_daily_path(data_root: str, yyyymmdd: str) -> str:
    return f"{data_root}/clean/daily/{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}.parquet"


def _clean_daily_columns(path: str) -> List[str]:
    """按分区 schema 取可用列（必需列缺则报错，可选列取交集）。"""
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    missing = [c for c in _CLEAN_DAILY_REQUIRED if c not in names]
    if missing:
        raise ValueError(
            f"{path} 缺少母截面重建必需列 {missing}（实际列 {sorted(names)}），"
            f"数据链路不完整，拒绝继续"
        )
    return _CLEAN_DAILY_REQUIRED + [c for c in _CLEAN_DAILY_OPTIONAL if c in names]


def load_clean_daily_long(data_root: str, dates: Sequence[str]) -> pd.DataFrame:
    """加载 clean/daily 长表（含复权列与量额），缺失分区明确报错。"""
    frames = []
    for d in dates:
        path = _clean_daily_path(data_root, d)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"母截面重建缺少 clean/daily 分区 {path}（日期 {d}），"
                f"数据链路不完整，拒绝静默跳过"
            )
        day = pd.read_parquet(path, columns=_clean_daily_columns(path))
        frames.append(day)
    if not frames:
        return pd.DataFrame(columns=_CLEAN_DAILY_REQUIRED)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = out["trade_date"].astype(str)
    return out


def build_mother_section(
    data_root: str,
    dates: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """为给定交易日构建完整同日母截面（标签过滤前）。

    Args:
        data_root: 数据根目录（如 data）
        dates: 目标交易日（YYYYMMDD，可乱序）

    Returns:
        {trade_date: DataFrame(ts_code + MOTHER_SECTION_FACTORS)}；
        证券域为该日 clean/daily 有行的全部股票，因子不可得处为 NaN
    """
    from ..precompute import precompute_risk_factors  # 同包（risk）内复用

    targets = sorted({str(d) for d in dates})
    if not targets:
        return {}

    calendar = _trade_calendar_window(data_root, targets[0], targets[-1])
    daily = load_clean_daily_long(data_root, calendar)
    risk_long = precompute_risk_factors(daily)
    if risk_long is None:
        raise ValueError(
            "precompute_risk_factors 返回 None：clean/daily 缺少必需列，母截面无法重建"
        )
    risk_cols = [c for c in MOTHER_SECTION_FACTORS if c != "ret_20"]
    missing_risk = [c for c in risk_cols if c not in risk_long.columns]
    if missing_risk:
        raise ValueError(f"风控因子预计算缺少 pct 基列: {missing_risk}（因子契约变更须同步本模块）")
    risk_long = risk_long[["ts_code", "trade_date"] + risk_cols].copy()
    risk_long["trade_date"] = risk_long["trade_date"].astype(str)

    from ...features.builder import _calculate_window_features_static

    rows_by_date: Dict[str, pd.DataFrame] = {}
    for d, sub in daily.groupby("trade_date", sort=False):
        rows_by_date[str(d)] = sub
    # 空日分区（占位分区/全市场停市）在日历中存在但无行：显式告警，不静默
    empty_days = [d for d in calendar if d not in rows_by_date]
    if empty_days:
        logger.warning(
            f"母截面历史窗口内有 {len(empty_days)} 个交易日无 clean/daily 行"
            f"（如 {empty_days[:3]}），这些日不参与窗口，请确认非数据缺失"
        )

    risk_by_date: Dict[str, pd.DataFrame] = {}
    for d, sub in risk_long.groupby("trade_date", sort=False):
        risk_by_date[str(d)] = sub

    window = 20
    out: Dict[str, pd.DataFrame] = {}
    for d in targets:
        current = rows_by_date.get(d)
        if current is None or current.empty:
            raise ValueError(f"母截面重建：{d} 的 clean/daily 行为空，拒绝产出空截面")
        idx = calendar.index(d)
        hist_dates = calendar[max(0, idx - window) : idx]
        if len(hist_dates) < window:
            logger.warning(f"{d} 预热交易日不足 {window} 天（首个交易日），ret_20 全为 NaN")
        hist = (
            pd.concat([rows_by_date[h] for h in hist_dates if h in rows_by_date], ignore_index=True)
            if any(h in rows_by_date for h in hist_dates)
            else pd.DataFrame(columns=_CLEAN_DAILY_REQUIRED)
        )
        window_features = _calculate_window_features_static(hist, current, window)
        frame = current[["ts_code"]].copy()
        if not window_features.empty:
            frame = frame.merge(
                window_features[["ts_code", f"ret_{window}"]], on="ts_code", how="left"
            )
            frame = frame.rename(columns={f"ret_{window}": "ret_20"})
        else:
            frame["ret_20"] = np.nan
        risk_today = risk_by_date.get(d)
        if risk_today is not None and not risk_today.empty:
            frame = frame.merge(risk_today[["ts_code"] + risk_cols], on="ts_code", how="left")
        else:
            for col in risk_cols:
                frame[col] = np.nan
        out[d] = frame[["ts_code"] + MOTHER_SECTION_FACTORS]
    logger.info(
        f"母截面重建完成: {len(out)} 个交易日，证券域 {min(len(v) for v in out.values())}"
        f"~{max(len(v) for v in out.values())} 只/日"
    )
    return out


def _trade_calendar_window(data_root: str, start_date: str, end_date: str) -> List[str]:
    """取 [start_date - MOTHER_HISTORY_MONTHS 个月, end_date] 的交易日轴。

    窗口规则与特征流水线（``features/pipeline.py`` 的 7 个月预热）一致，
    保证逐股收益（依赖"上一可用行"）与 cs_train 的生成条件一致。
    """
    from .dataset import load_trade_calendar

    hist_start = (pd.Timestamp(start_date) - pd.DateOffset(months=MOTHER_HISTORY_MONTHS)).strftime(
        "%Y%m%d"
    )
    cal = load_trade_calendar(data_root, hist_start, end_date)
    if start_date not in cal:
        raise ValueError(
            f"交易日历缺少 {start_date}（母截面重建起点），请检查 clean/trade_cal.parquet"
        )
    return list(cal)


def validate_mother_section_against_cs_train(
    data_root: str,
    mother_by_date: Dict[str, pd.DataFrame],
    dates: Optional[Sequence[str]] = None,
    atol: float = MOTHER_VALIDATION_ATOL,
    outlier_share_limit: float = MOTHER_OUTLIER_SHARE_LIMIT,
    hard_atol: float = MOTHER_HARD_ATOL,
) -> Dict[str, Any]:
    """在交集上统计母截面与 cs_train 的同名列差异（逐块调用，不在此判定）。

    判定属**窗口级**职责（见 ``merge_mother_section_validation`` +
    ``assert_mother_section_validation_ok``）：分块构建时每块只有 50 个交易日，
    在块内判定占比会把小块噪声当成信号。本函数只负责**统计与登记**。

    Args:
        data_root: 数据根目录
        mother_by_date: build_mother_section 产物（通常为一块的日期）
        dates: 校验日期（默认全量）
        atol: 逐值容差
        outlier_share_limit / hard_atol: 阈值，仅记账在返回值 limits 中
            供窗口级判定使用

    Returns:
        dict：checked_dates / checked_rows / max_abs_diff / outliers
        （total / share / rows / dates / date_share / examples / limits）

    Raises:
        ValueError: 没有任何交集行（无法证明同源）
        FileNotFoundError: 缺少 cs_train 分区
    """
    targets = sorted(mother_by_date) if dates is None else [str(d) for d in dates]
    stats: Dict[str, float] = {c: 0.0 for c in MOTHER_SECTION_FACTORS}
    outlier_rows: Dict[str, int] = {c: 0 for c in MOTHER_SECTION_FACTORS}
    examples: List[Dict[str, Any]] = []
    outlier_dates = 0
    checked_rows = 0
    checked_dates = 0
    for d in targets:
        mother = mother_by_date.get(d)
        if mother is None or mother.empty:
            continue
        path = f"{data_root}/features/cs_train/{d}.parquet"
        try:
            train = pd.read_parquet(path, columns=["ts_code"] + MOTHER_SECTION_FACTORS)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"一致性校验缺少 cs_train 分区 {path}，无法证明母截面与训练特征同源"
            ) from exc
        merged = train.merge(mother, on="ts_code", how="inner", suffixes=("_train", "_mother"))
        if merged.empty:
            continue
        checked_dates += 1
        checked_rows += len(merged)
        codes = merged["ts_code"].to_numpy()
        date_has_outlier = False
        for col in MOTHER_SECTION_FACTORS:
            a = merged[f"{col}_train"].to_numpy(dtype=float)
            b = merged[f"{col}_mother"].to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            if not both.any():
                continue
            diff = np.abs(a[both] - b[both])
            stats[col] = max(stats[col], float(diff.max()))
            over = diff > atol
            n_over = int(over.sum())
            if not n_over:
                continue
            outlier_rows[col] += n_over
            date_has_outlier = True
            hit = np.flatnonzero(both)[over]
            for pos, value in zip(hit[: MOTHER_OUTLIER_EXAMPLES - len(examples)], diff[over]):
                if len(examples) >= MOTHER_OUTLIER_EXAMPLES:
                    break
                examples.append(
                    {
                        "date": d,
                        "column": col,
                        "ts_code": str(codes[pos]),
                        "abs_diff": float(value),
                    }
                )
        if date_has_outlier:
            outlier_dates += 1

    if checked_rows == 0:
        raise ValueError(
            "母截面一致性校验没有任何交集行（索引/证券域/日期全部不匹配），"
            "无法证明同源，拒绝继续训练"
        )
    total_outliers = int(sum(outlier_rows.values()))
    logger.info(
        f"母截面一致性校验统计: {checked_dates} 日 / {checked_rows} 行交集，"
        f"各列最大绝对差 {stats}，超容差 {total_outliers} 行"
    )
    return {
        "checked_dates": checked_dates,
        "checked_rows": checked_rows,
        "max_abs_diff": stats,
        "atol": atol,
        "outliers": {
            "total": total_outliers,
            "share": total_outliers / checked_rows,
            "rows": outlier_rows,
            "dates": outlier_dates,
            "examples": examples,
            "limits": {
                "hard_atol": hard_atol,
                "share_limit": outlier_share_limit,
            },
        },
    }


def merge_mother_section_validation(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """合并多块的校验统计（窗口级口径）。

    行/日/离群计数相加，``max_abs_diff`` 取各列最大值，占比在合并后的
    分母上重算——这才是"整个训练窗口里有多少比例不同源"的正确口径。
    """
    parts = [r for r in results if r]
    if not parts:
        raise ValueError("母截面校验结果为空，无法合并（分块构建至少应产生一块统计）")
    checked_rows = int(sum(int(p["checked_rows"]) for p in parts))
    checked_dates = int(sum(int(p["checked_dates"]) for p in parts))
    cols = set()
    for p in parts:
        cols.update(p["max_abs_diff"])
    max_diff = {c: max(float(p["max_abs_diff"].get(c, 0.0)) for p in parts) for c in sorted(cols)}
    outlier_rows: Dict[str, int] = {c: 0 for c in sorted(cols)}
    for p in parts:
        for c, n in p["outliers"]["rows"].items():
            outlier_rows[c] = outlier_rows.get(c, 0) + int(n)
    outlier_dates = int(sum(int(p["outliers"]["dates"]) for p in parts))
    examples: List[Dict[str, Any]] = []
    for p in parts:
        for ex in p["outliers"]["examples"]:
            if len(examples) >= MOTHER_OUTLIER_EXAMPLES:
                break
            examples.append(ex)
    limits = parts[-1]["outliers"]["limits"]
    return {
        "checked_dates": checked_dates,
        "checked_rows": checked_rows,
        "max_abs_diff": max_diff,
        "atol": parts[-1]["atol"],
        "outliers": {
            "total": int(sum(outlier_rows.values())),
            "share": (sum(outlier_rows.values()) / checked_rows) if checked_rows else 0.0,
            "rows": outlier_rows,
            "dates": outlier_dates,
            "date_share": (outlier_dates / checked_dates) if checked_dates else 0.0,
            "examples": examples,
            "limits": limits,
        },
    }


def assert_mother_section_validation_ok(merged: Dict[str, Any]) -> Dict[str, Any]:
    """窗口级判定：区分实现漂移（报错）与数据态漂移（告警 + 登记）。

    判据（任一超限即判实现漂移）：

    - 超容差行占交集行数 > ``share_limit``（默认 1e-4）；
    - 任一列最大绝对差 > ``hard_atol``（默认 0.05）。

    系统性差异（换实现、整段因子口径变更、整日数据批量重算）必然触发其一；
    稀疏的数据态修订（实测：同一只股票连续 10 个交易日各 1 行，行占比 1.4e-5）
    则只告警，但离群明细会落入模型元数据，禁止静默掩过。

    不把"涉及日期数/日占比"作判据：一只股票的某日修订会在其后 20 日窗口内
    连续重现，日占比随窗口长度剧烈变化（10/123 日 vs 10/851 日），是伪影。

    Raises:
        ValueError: 判定为实现漂移
    """
    outliers = merged["outliers"]
    limits = outliers["limits"]
    stats = merged["max_abs_diff"]
    share = float(outliers["share"])
    violations = {c: v for c, v in stats.items() if v > limits["hard_atol"]}
    if violations or share > limits["share_limit"]:
        raise ValueError(
            f"母截面与 cs_train 差异呈系统性（最大绝对差 {stats}，超容差 {outliers['total']} 行"
            f"占比 {share:.2e}，涉及 {outliers['dates']}/{merged['checked_dates']} 日；"
            f"上限：幅度 {limits['hard_atol']}、行占比 {limits['share_limit']:.0e}）："
            f"pct_* 分子分母不同源，拒绝继续训练。请检查因子实现是否被改动；"
            f"若确为数据态批量修订（如整日重算），先重刷对应 cs_train 分区"
        )
    if outliers["total"]:
        logger.warning(
            f"母截面与 cs_train 存在 {outliers['total']} 行数据态差异（占窗口交集 {share:.2e}，"
            f"涉及 {outliers['dates']}/{merged['checked_dates']} 日，不阻断训练）："
            f"明细已登记至元数据 outliers；如需彻底消除请重刷对应日期的 cs_train 分区"
        )
    logger.info(
        f"母截面校验（窗口级）通过: {merged['checked_dates']} 日 / {merged['checked_rows']} 行交集，"
        f"各列最大绝对差 {stats}，数据态离群 {outliers['total']} 行"
    )
    return merged

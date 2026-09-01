# -*- coding: utf-8 -*-
"""汇总CSV / 链式nav / 交易日历 的加载与清洗。"""

import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare.constants import (
    _BT_REBALANCE_FREQ_MAX,
    _TRADE_CAL_CACHE,
    _TRADE_CAL_OPEN_DATES_CACHE,
    SUMMARY_CSV_DTYPE,
)
from src.lazybull.data import DataLoader, Storage


def _is_missing_param_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "nan", "none", "null")
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_true_param_value(value) -> bool:
    if _is_missing_param_value(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
    return bool(value)


def _normalize_param_text(value) -> Optional[str]:
    if _is_missing_param_value(value):
        return None
    return str(value).strip()


def _parse_optional_int(value) -> Optional[int]:
    if _is_missing_param_value(value):
        return None
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None


def _get_open_trade_dates_and_index(
    trade_cal: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    cache_key = id(trade_cal)
    cached = _TRADE_CAL_OPEN_DATES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    open_dates = (
        trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"].astype(str).sort_values().tolist()
    )
    index_map = {trade_date: idx for idx, trade_date in enumerate(open_dates)}
    cached = (open_dates, index_map)
    _TRADE_CAL_OPEN_DATES_CACHE[cache_key] = cached
    return cached


def _find_nearest_trade_date_backward(
    all_trade_dates: list[str], target_date: str
) -> Optional[str]:
    if not all_trade_dates:
        return None
    pos = bisect_right(all_trade_dates, target_date)
    if pos <= 0:
        return None
    return all_trade_dates[pos - 1]


def _collect_rebalance_freq_candidates(
    ordered: pd.DataFrame,
    all_trade_dates: list[str],
    trade_date_index: dict[str, int],
    test_window_months: int,
) -> set[int]:
    probe_indices = ordered["__split_index_int"].astype(int).tolist()
    probe_indices = sorted(set(probe_indices[:3] + probe_indices[-3:]))

    candidates: Optional[set[int]] = None
    for idx in probe_indices:
        row = ordered.loc[ordered["__split_index_int"] == idx].iloc[0]
        test_start = _normalize_param_text(row.get("test_start"))
        test_end = _normalize_param_text(row.get("test_end"))
        if not test_start or not test_end:
            continue

        test_start_idx = trade_date_index.get(test_start)
        test_end_idx = trade_date_index.get(test_end)
        if test_start_idx is None or test_end_idx is None or test_end_idx < test_start_idx:
            continue

        nominal_target = (
            datetime.strptime(test_start, "%Y%m%d") + relativedelta(months=test_window_months)
        ).strftime("%Y%m%d")
        nominal_end = _find_nearest_trade_date_backward(all_trade_dates, nominal_target)
        if nominal_end is None:
            continue

        nominal_end_idx = trade_date_index.get(nominal_end)
        if nominal_end_idx is None or nominal_end_idx < test_start_idx:
            continue

        # 末段可能被 final_date 截断；此时无法由边界稳定反推调仓频率，跳过该 probe。
        if test_end_idx < nominal_end_idx:
            continue

        actual_span = test_end_idx - test_start_idx + 1
        nominal_span = nominal_end_idx - test_start_idx + 1
        min_valid_freq = actual_span - nominal_span + 1

        row_candidates: set[int] = set()
        upper = min(actual_span, _BT_REBALANCE_FREQ_MAX)
        divisor = 1
        while divisor * divisor <= actual_span:
            if actual_span % divisor == 0:
                pair = actual_span // divisor
                if divisor >= min_valid_freq and divisor <= upper:
                    row_candidates.add(divisor)
                if pair >= min_valid_freq and pair <= upper:
                    row_candidates.add(pair)
            divisor += 1

        if not row_candidates:
            continue

        candidates = row_candidates if candidates is None else candidates & row_candidates
        if not candidates:
            break

    return candidates or set()


def _load_trade_cal_for_compare(data_root: Optional[Path]) -> Optional[pd.DataFrame]:
    if data_root is None:
        return None

    cache_key = str(Path(data_root).resolve())
    if cache_key in _TRADE_CAL_CACHE:
        return _TRADE_CAL_CACHE[cache_key]

    try:
        storage = Storage(root_path=str(data_root))
        loader = DataLoader(storage)
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None or len(trade_cal) == 0:
            trade_cal = loader.load_trade_cal()
    except Exception as exc:
        logger.warning(f"加载交易日历失败，无法回推 bt_rebalance_freq: {exc}")
        _TRADE_CAL_CACHE[cache_key] = None
        return None

    if trade_cal is None or len(trade_cal) == 0:
        logger.warning("交易日历为空，无法回推 bt_rebalance_freq")
        _TRADE_CAL_CACHE[cache_key] = None
        return None

    _TRADE_CAL_CACHE[cache_key] = trade_cal
    return trade_cal


def _infer_bt_rebalance_freq_from_group(
    group: pd.DataFrame,
    trade_cal: Optional[pd.DataFrame],
) -> Optional[int]:
    if group.empty:
        return None

    if "bt_rebalance_freq" in group.columns:
        for value in group["bt_rebalance_freq"].tolist():
            parsed = _parse_optional_int(value)
            if parsed is not None:
                return parsed

    if trade_cal is None:
        return None

    first = group.iloc[0]
    all_trade_dates, trade_date_index = _get_open_trade_dates_and_index(trade_cal)
    if not all_trade_dates:
        return None

    test_window_months = _parse_optional_int(first.get("test_window_months"))
    if test_window_months is None:
        return None

    ordered = group.copy()
    ordered["__split_index_int"] = pd.to_numeric(ordered.get("split_index"), errors="coerce")
    ordered = ordered.dropna(subset=["__split_index_int"]).sort_values("__split_index_int")

    if ordered.empty:
        return None

    candidates = _collect_rebalance_freq_candidates(
        ordered,
        all_trade_dates,
        trade_date_index,
        test_window_months,
    )
    if not candidates:
        return None

    # 同一批边界约束下，满足条件的最小频率就是实际调仓频率；更大的倍数只是在个别窗口上“碰巧也对齐”。
    return min(candidates)


def _fill_missing_bt_rebalance_freq(
    all_df: pd.DataFrame,
    data_root: Optional[Path],
) -> pd.DataFrame:
    if all_df.empty or "wf_run_id" not in all_df.columns:
        return all_df

    filled = all_df.copy()
    if "bt_rebalance_freq" not in filled.columns:
        filled["bt_rebalance_freq"] = pd.NA

    trade_cal: Optional[pd.DataFrame] = None
    for wf_run_id, group in filled.groupby("wf_run_id", sort=False):
        inferred = _infer_bt_rebalance_freq_from_group(group, trade_cal)
        if inferred is None:
            if trade_cal is None:
                trade_cal = _load_trade_cal_for_compare(data_root)
            inferred = _infer_bt_rebalance_freq_from_group(group, trade_cal)

        if inferred is not None:
            filled.loc[group.index, "bt_rebalance_freq"] = inferred
        else:
            logger.debug(f"无法从 summary 回推 bt_rebalance_freq: {wf_run_id}")

    return filled


def _sanitize_summary_train_params(raw_params: dict) -> dict:
    """兼容历史 summary：按参数是否实际生效清空旧默认值。"""
    params = dict(raw_params)

    def clear(*keys: str) -> None:
        for key in keys:
            if key in params:
                params[key] = None

    if not _is_true_param_value(params.get("oos_backtest")):
        clear(
            "oos_backtest_months",
            "bt_top_n",
            "bt_rebalance_freq",
            "bt_sell_timing",
            "bt_exclude_st",
            "bt_min_list_days",
            "bt_max_weight_per_stock",
            "bt_max_per_industry",
            "bt_stop_loss_enabled",
            "bt_stop_loss_drawdown_pct",
            "bt_stop_loss_trailing_enabled",
            "bt_stop_loss_trailing_pct",
            "bt_stop_loss_consecutive_limit_down",
            "industry_momentum_filter",
            "industry_momentum_bottom_pct",
            "industry_rotation_enhanced",
            "industry_rotation_alpha",
            "position_sizing",
            "kelly_vol_window",
            "kelly_max_leverage",
            "market_regime",
            "market_regime_bear_threshold",
            "market_regime_bear_exposure",
            "market_regime_mode",
            "market_regime_vol_target",
            "market_regime_trend_threshold",
            "market_regime_min_exposure",
            "market_regime_combine_method",
            "market_regime_trend_guard",
            "market_regime_drawdown_guard",
            "market_regime_drawdown_threshold",
            "market_regime_ma250_hard_stop",
            "market_regime_ma250_threshold",
            "market_regime_ma250_exposure",
            "market_regime_ma250_atr_scaling",
            "stagger_tranches",
            "enable_early_rebalance_on_empty",
        )
        return params

    if not _is_true_param_value(params.get("bt_stop_loss_enabled")):
        clear(
            "bt_stop_loss_drawdown_pct",
            "bt_stop_loss_trailing_enabled",
            "bt_stop_loss_trailing_pct",
            "bt_stop_loss_consecutive_limit_down",
        )
    elif not _is_true_param_value(params.get("bt_stop_loss_trailing_enabled")):
        clear("bt_stop_loss_trailing_pct")

    if not _is_true_param_value(params.get("industry_momentum_filter")):
        clear("industry_momentum_bottom_pct")

    if not _is_true_param_value(params.get("industry_rotation_enhanced")):
        clear("industry_rotation_alpha")

    if _normalize_param_text(params.get("position_sizing")) not in ("kelly", "half_kelly"):
        clear("kelly_vol_window", "kelly_max_leverage")

    if not _is_true_param_value(params.get("market_regime")):
        clear(
            "market_regime_bear_threshold",
            "market_regime_bear_exposure",
            "market_regime_mode",
            "market_regime_vol_target",
            "market_regime_trend_threshold",
            "market_regime_min_exposure",
            "market_regime_combine_method",
            "market_regime_trend_guard",
            "market_regime_drawdown_guard",
            "market_regime_drawdown_threshold",
        )
    else:
        market_regime_mode = _normalize_param_text(params.get("market_regime_mode"))
        if market_regime_mode == "binary":
            clear(
                "market_regime_vol_target",
                "market_regime_trend_threshold",
                "market_regime_min_exposure",
                "market_regime_combine_method",
                "market_regime_trend_guard",
            )
        elif market_regime_mode == "vol_target":
            clear(
                "market_regime_bear_threshold",
                "market_regime_bear_exposure",
                "market_regime_trend_threshold",
                "market_regime_combine_method",
                "market_regime_trend_guard",
            )
        elif market_regime_mode == "trend":
            clear(
                "market_regime_bear_threshold",
                "market_regime_bear_exposure",
                "market_regime_vol_target",
                "market_regime_combine_method",
                "market_regime_trend_guard",
            )
        elif market_regime_mode == "combined":
            clear(
                "market_regime_bear_threshold",
                "market_regime_bear_exposure",
            )

    if not _is_true_param_value(params.get("market_regime_ma250_hard_stop")):
        clear(
            "market_regime_ma250_threshold",
            "market_regime_ma250_exposure",
            "market_regime_ma250_atr_scaling",
        )

    if not _is_true_param_value(params.get("market_regime_drawdown_guard")):
        clear("market_regime_drawdown_threshold")

    return params


def _sanitize_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sanitized_rows = [_sanitize_summary_train_params(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(sanitized_rows, columns=df.columns)


def _concat_summary_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """按旧语义拼接 summary，避免 pandas 对全 NA 列发出 FutureWarning。"""
    if not frames:
        return pd.DataFrame()

    ordered_columns: list[str] = []
    for frame in frames:
        for col in frame.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

    prepared_frames = []
    for frame in frames:
        if frame.empty:
            continue
        # 显式排除单个 frame 内全 NA 列，保持 pandas 旧版 concat 的 dtype 推断语义。
        prepared_frames.append(frame.dropna(axis=1, how="all"))

    if not prepared_frames:
        return pd.DataFrame(columns=ordered_columns)

    all_df = pd.concat(prepared_frames, ignore_index=True)
    return all_df.reindex(columns=ordered_columns)


# ---------------------------------------------------------------------------
# 综合得分配置：(英文列键, 权重, 方向)
#   "high"    → 值越大越好
#   "low"     → 值越小越好
#   "abs_low" → 绝对值越小越好
# 权重之和应为 1.0
# ---------------------------------------------------------------------------
# ── 回测指标（60%）：真实组合模拟，最直接反映参数优劣 ──────────
# ── 统计指标（32%）：辅助验证，防止回测过拟合 ─────────────────
# ── 训练质量（8%）：过拟合检测 ────────────────────────────────


def load_chain_metrics(
    raw_dir: Optional[Path], wf_run_id: str, source_dir: Optional[Path] = None
) -> dict:
    """读取 chain_nav 并计算全周期指标。"""
    from src.lazybull.ml.walk_forward.chain_metrics import calculate_chain_metrics

    empty = {
        "chain_total_return": None,
        "chain_cagr": None,
        "chain_max_drawdown": None,
        "chain_sharpe": None,
        "chain_trading_days": None,
    }
    effective_raw_dir = source_dir or raw_dir
    if effective_raw_dir is None:
        return empty

    chain_path = effective_raw_dir / f"chain_nav_{wf_run_id}.csv"
    if not chain_path.exists():
        return empty

    try:
        chain_df = pd.read_csv(chain_path, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning(f"读取 chain_nav 失败: {chain_path.name} — {exc}")
        return empty

    metrics = calculate_chain_metrics(chain_df)
    if metrics["total_return"] is None:
        return empty

    return {
        "chain_total_return": round(metrics["total_return"], 6),
        "chain_cagr": round(metrics["cagr"], 6) if metrics["cagr"] is not None else None,
        "chain_max_drawdown": (
            round(metrics["max_drawdown"], 6) if metrics["max_drawdown"] is not None else None
        ),
        "chain_sharpe": (round(metrics["sharpe"], 4) if metrics["sharpe"] is not None else None),
        "chain_trading_days": metrics["trading_days"],
    }


def load_all_summaries_from_raw_dirs(
    raw_dirs: list[Path],
    data_root: Optional[Path] = None,
) -> pd.DataFrame:
    """从一个或多个 raw 目录加载 walk_forward 汇总 CSV。"""
    existing_raw_dirs = [Path(raw_dir) for raw_dir in raw_dirs if Path(raw_dir).exists()]
    if len(existing_raw_dirs) == 0:
        logger.warning("未找到任何可用汇总目录")
        return pd.DataFrame()

    csv_files: list[tuple[Path, Path]] = []
    for raw_dir in existing_raw_dirs:
        csv_files.extend(
            (raw_dir, csv_file) for csv_file in sorted(raw_dir.glob("walk_forward_summary_*.csv"))
        )

    if len(csv_files) == 0:
        joined_dirs = ", ".join(str(raw_dir) for raw_dir in existing_raw_dirs)
        logger.warning(f"未找到任何汇总CSV: {joined_dirs}")
        return pd.DataFrame()

    logger.info(f"找到 {len(csv_files)} 个汇总CSV文件")
    frames = []
    for raw_dir, f in csv_files:
        try:
            df = pd.read_csv(
                f,
                encoding="utf-8-sig",
                dtype=SUMMARY_CSV_DTYPE,
            )
            df = _sanitize_summary_frame(df)
            df["_source_file"] = f.name
            df["_source_dir"] = str(raw_dir)
            frames.append(df)
            logger.debug(f"  已加载: {f.name}（{len(df)} 行）")
        except Exception as e:
            logger.warning(f"  跳过（读取失败）: {f.name} — {e}")

    if not frames:
        return pd.DataFrame()

    all_df = _concat_summary_frames(frames)
    all_df = _fill_missing_bt_rebalance_freq(all_df, data_root)
    logger.info(
        f"合并后总行数: {len(all_df)}，unique wf_run_id: {all_df['wf_run_id'].nunique() if 'wf_run_id' in all_df.columns else '?'}"
    )
    return all_df


def load_all_summaries(raw_dir: Path, data_root: Optional[Path] = None) -> pd.DataFrame:
    """兼容旧调用：从单个 raw 目录加载汇总CSV。"""
    return load_all_summaries_from_raw_dirs([raw_dir], data_root=data_root)


def build_auto_compare_jobs(data_root: Path) -> list[dict]:
    """构建无参模式下的自动扫描任务。"""
    walk_forward_root = data_root / "walk_forward"
    raw_dir = walk_forward_root / "raw"
    batches_root = walk_forward_root / "batches"
    batch_raw_dirs = (
        sorted(path for path in batches_root.glob("*/raw") if path.is_dir())
        if batches_root.exists()
        else []
    )

    jobs = [
        {
            "label": "raw",
            "raw_dirs": [raw_dir],
            "output_path": walk_forward_root / "wf_comparison_raw.xlsx",
        }
    ]
    if batch_raw_dirs:
        jobs.append(
            {
                "label": "batches",
                "raw_dirs": batch_raw_dirs,
                "output_path": walk_forward_root / "wf_comparison_batches.xlsx",
            }
        )
    return jobs

# -*- coding: utf-8 -*-
"""跨 split 聚合指标与对比表 / 跨时间段稳定性表构建。"""

from typing import Optional
import re

from loguru import logger
import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare.constants import (
    COL_NAMES,
    PARAM_COLS,
    SCORE_CONFIG,
    SEED_STABILITY_EXCLUDED_MODEL_KEYS,
)
from scripts.compare.loading import load_chain_metrics


def aggregate_run(group: pd.DataFrame) -> dict:
    """对单个 wf_run_id 的所有 split 行进行聚合，返回一行对比指标"""
    row = {}
    n = len(group)
    row["n_splits"] = n

    # 模型版本范围（min~max）
    if "model_version" in group.columns:
        mv = group["model_version"].dropna()
        if len(mv):
            row["model_version_range"] = f"{int(mv.min())}~{int(mv.max())}"
        else:
            row["model_version_range"] = None
    else:
        row["model_version_range"] = None

    # -----------------------------------------------------------------------
    # OOS 性能指标（来自 test_daily_metrics 展开列）
    # -----------------------------------------------------------------------
    def safe_mean(col):
        return group[col].mean() if col in group.columns else None

    def safe_std(col):
        return group[col].std() if col in group.columns else None

    def safe_min(col):
        return group[col].min() if col in group.columns else None

    def safe_max(col):
        return group[col].max() if col in group.columns else None

    # KEY 重点字段（前置展示）
    row["KEY_说明"] = "重点: hit rate=TopK逐日平均收益>0占比; list=最新OOS日期预测名单"
    row["KEY_Top20_list"] = None
    row["KEY_Top30_list"] = None
    if "split_index" in group.columns:
        sorted_group = group.copy()
        sorted_group["__split_index_int"] = pd.to_numeric(
            sorted_group["split_index"], errors="coerce"
        )
        sorted_group = sorted_group.sort_values("__split_index_int")
    else:
        sorted_group = group.copy()
    if "KEY_Top20_list" in sorted_group.columns:
        top20_list = sorted_group["KEY_Top20_list"].dropna()
        row["KEY_Top20_list"] = str(top20_list.iloc[-1]) if len(top20_list) else None
    if "KEY_Top30_list" in sorted_group.columns:
        top30_list = sorted_group["KEY_Top30_list"].dropna()
        row["KEY_Top30_list"] = str(top30_list.iloc[-1]) if len(top30_list) else None

    key20_hit = safe_mean("KEY_Top20_hit_rate")
    key20_med = safe_mean("KEY_Top20_avg_return_median")
    key30_hit = safe_mean("KEY_Top30_hit_rate")
    key30_med = safe_mean("KEY_Top30_avg_return_median")
    row["key_top20_hit_rate_mean"] = round(key20_hit, 4) if key20_hit is not None else None
    row["key_top20_avg_return_median_mean"] = round(key20_med, 6) if key20_med is not None else None
    row["key_top30_hit_rate_mean"] = round(key30_hit, 4) if key30_hit is not None else None
    row["key_top30_avg_return_median_mean"] = round(key30_med, 6) if key30_med is not None else None

    # OOS RankIC IR
    oos_ir_series = (
        group["daily_rankic_ir"] if "daily_rankic_ir" in group.columns else pd.Series(dtype=float)
    )
    oos_ir_mean = oos_ir_series.mean() if len(oos_ir_series) else None
    oos_ir_std = oos_ir_series.std() if len(oos_ir_series) > 1 else None
    row["oos_rankic_ir_mean"] = round(oos_ir_mean, 4) if oos_ir_mean is not None else None
    row["oos_rankic_ir_std"] = round(oos_ir_std, 4) if oos_ir_std is not None else None
    row["oos_cross_split_ir"] = (
        round(oos_ir_mean / oos_ir_std, 3)
        if (oos_ir_mean and oos_ir_std and oos_ir_std != 0)
        else None
    )

    # RankIC 均值与 ICIR（纯选股能力核心指标）
    rankic_mean_series = (
        group["daily_rankic_mean"].dropna()
        if "daily_rankic_mean" in group.columns
        else pd.Series(dtype=float)
    )
    rankic_std_series = (
        group["daily_rankic_std"].dropna()
        if "daily_rankic_std" in group.columns
        else pd.Series(dtype=float)
    )
    rankic_mean = rankic_mean_series.mean() if len(rankic_mean_series) else None
    rankic_std = rankic_std_series.mean() if len(rankic_std_series) else None
    row["daily_rankic_mean"] = round(rankic_mean, 6) if rankic_mean is not None else None
    row["icir"] = (
        round(rankic_mean / rankic_std, 4)
        if (rankic_mean is not None and rankic_std is not None and rankic_std != 0)
        else None
    )

    # OOS RankIC 衰减检测（最近3个split均值 - 最早3个split均值）
    if len(oos_ir_series) >= 6:
        sorted_ir = (
            group.sort_values("split_index")["daily_rankic_ir"]
            if "daily_rankic_ir" in group.columns
            else oos_ir_series
        )
        row["oos_rankic_ir_trend"] = round(
            sorted_ir.iloc[-3:].mean() - sorted_ir.iloc[:3].mean(), 4
        )
    else:
        row["oos_rankic_ir_trend"] = None

    # Top30 指标（以中位数为核心，不受极端日干扰）
    med30_col = "diagnostic_Top30_逐日均值_50分位"
    mean30_col = "diagnostic_Top30_逐日均值的均值"
    std30_col = "diagnostic_Top30_逐日均值的标准差"
    lift30_col = "diagnostic_Top30_相对全市场提升_均值"

    if med30_col in group.columns:
        med30_series = group[med30_col].dropna()
        row["oos_top30_median_mean"] = round(med30_series.mean(), 6) if len(med30_series) else None
        row["oos_top30_win_rate"] = (
            round((med30_series > 0).mean(), 3) if len(med30_series) else None
        )
        row["oos_top30_worst_median"] = round(med30_series.min(), 6) if len(med30_series) else None
    else:
        row["oos_top30_median_mean"] = row["oos_top30_win_rate"] = row["oos_top30_worst_median"] = (
            None
        )

    # Top30 偏斜度（均值/中位数 gap，衡量是否被极端日驱动）
    if all(c in group.columns for c in [mean30_col, med30_col, std30_col]):
        valid = group[[mean30_col, med30_col, std30_col]].dropna()
        if len(valid):
            skew_scores = (valid[mean30_col] - valid[med30_col]) / valid[std30_col].replace(
                0, np.nan
            )
            row["oos_top30_skew_score_mean"] = round(skew_scores.mean(), 3)
        else:
            row["oos_top30_skew_score_mean"] = None
    else:
        row["oos_top30_skew_score_mean"] = None

    row["oos_top30_lift_mean"] = (
        round(safe_mean(lift30_col), 6) if safe_mean(lift30_col) is not None else None
    )

    # Top100 指标
    med100_col = "diagnostic_Top100_逐日均值_50分位"
    if med100_col in group.columns:
        med100_series = group[med100_col].dropna()
        row["oos_top100_median_mean"] = (
            round(med100_series.mean(), 6) if len(med100_series) else None
        )
        row["oos_top100_win_rate"] = (
            round((med100_series > 0).mean(), 3) if len(med100_series) else None
        )
    else:
        row["oos_top100_median_mean"] = row["oos_top100_win_rate"] = None

    # Top300 指标
    med300_col = "diagnostic_Top300_逐日均值_50分位"
    if med300_col in group.columns:
        med300_series = group[med300_col].dropna()
        row["oos_top300_median_mean"] = (
            round(med300_series.mean(), 6) if len(med300_series) else None
        )
        row["oos_top300_win_rate"] = (
            round((med300_series > 0).mean(), 3) if len(med300_series) else None
        )
    else:
        row["oos_top300_median_mean"] = row["oos_top300_win_rate"] = None

    # 分层单调性近似评分（Top30/100/300 中位收益应随覆盖范围扩大而递减）
    monotonic_inputs = [
        (30, row.get("oos_top30_median_mean")),
        (100, row.get("oos_top100_median_mean")),
        (300, row.get("oos_top300_median_mean")),
    ]
    monotonic_inputs = [(k, v) for k, v in monotonic_inputs if v is not None and pd.notna(v)]
    if len(monotonic_inputs) >= 2:
        bucket_sizes = np.array([k for k, _ in monotonic_inputs], dtype=float)
        bucket_returns = np.array([v for _, v in monotonic_inputs], dtype=float)
        if np.allclose(bucket_returns, bucket_returns[0]):
            row["selection_monotonicity"] = 0.5
        else:
            corr = np.corrcoef(bucket_sizes, bucket_returns)[0, 1]
            if pd.notna(corr):
                row["selection_monotonicity"] = round(float(np.clip((1 - corr) / 2, 0.0, 1.0)), 4)
            else:
                row["selection_monotonicity"] = None
    else:
        row["selection_monotonicity"] = None

    # -----------------------------------------------------------------------
    # OOS 回测指标（来自 run_oos_backtest 写入的 bt_* 列）
    # -----------------------------------------------------------------------
    if "bt_total_return" in group.columns:
        bt_ret = group["bt_total_return"].dropna()
        bt_ar = (
            group["bt_annual_return"].dropna()
            if "bt_annual_return" in group.columns
            else pd.Series(dtype=float)
        )
        bt_sh = (
            group["bt_sharpe"].dropna() if "bt_sharpe" in group.columns else pd.Series(dtype=float)
        )
        bt_md = (
            group["bt_max_drawdown"].dropna()
            if "bt_max_drawdown" in group.columns
            else pd.Series(dtype=float)
        )
        bt_cal = (
            group["bt_calmar"].dropna() if "bt_calmar" in group.columns else pd.Series(dtype=float)
        )
        bt_vol = (
            group["bt_volatility"].dropna()
            if "bt_volatility" in group.columns
            else pd.Series(dtype=float)
        )

        row["bt_total_return_mean"] = round(bt_ret.mean(), 6) if len(bt_ret) else None
        row["bt_annual_return_mean"] = round(bt_ar.mean(), 6) if len(bt_ar) else None
        row["bt_sharpe_mean"] = round(bt_sh.mean(), 4) if len(bt_sh) else None
        row["bt_max_drawdown_worst"] = round(bt_md.min(), 6) if len(bt_md) else None
        row["bt_calmar_mean"] = round(bt_cal.mean(), 4) if len(bt_cal) else None
        row["bt_volatility_mean"] = round(bt_vol.mean(), 6) if len(bt_vol) else None
        row["bt_win_rate"] = round((bt_ret > 0).mean(), 3) if len(bt_ret) else None
    else:
        for k in [
            "bt_total_return_mean",
            "bt_annual_return_mean",
            "bt_sharpe_mean",
            "bt_max_drawdown_worst",
            "bt_calmar_mean",
            "bt_volatility_mean",
            "bt_win_rate",
        ]:
            row[k] = None

    # -----------------------------------------------------------------------
    # 训练质量指标
    # -----------------------------------------------------------------------
    # 验证集 RankIC IR（val_rankic_ir 列，每 split 一个值）
    if "val_rankic_ir" in group.columns:
        val_ir_series = group["val_rankic_ir"].dropna()
        row["val_rankic_ir_mean"] = round(val_ir_series.mean(), 4) if len(val_ir_series) else None
        # 泛化差距（val IR 越接近 oos IR 越好；负值说明 oos 反而更好，通常是正常的）
        if row["val_rankic_ir_mean"] is not None and row["oos_rankic_ir_mean"] is not None:
            row["train_val_ir_gap"] = round(
                row["val_rankic_ir_mean"] - row["oos_rankic_ir_mean"], 4
            )
        else:
            row["train_val_ir_gap"] = None
    else:
        row["val_rankic_ir_mean"] = row["train_val_ir_gap"] = None

    # 最佳迭代次数统计
    if "best_iteration" in group.columns:
        bi = group["best_iteration"].dropna()
        row["best_iter_mean"] = round(bi.mean(), 1) if len(bi) else None
        row["best_iter_min"] = int(bi.min()) if len(bi) else None
        row["best_iter_max"] = int(bi.max()) if len(bi) else None
        row["best_iter_std"] = round(bi.std(), 1) if len(bi) > 1 else None
    else:
        row["best_iter_mean"] = row["best_iter_min"] = row["best_iter_max"] = row[
            "best_iter_std"
        ] = None

    return row


def build_comparison_table(all_df: pd.DataFrame, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """构建对比表（行=run，列=聚合指标+训练参数）"""
    if "wf_run_id" not in all_df.columns:
        logger.error("汇总CSV中缺少 wf_run_id 列，无法分组")
        return pd.DataFrame()

    rows = []
    for wf_run_id, group in all_df.groupby("wf_run_id", sort=False):
        source_dir = None
        if "_source_dir" in group.columns:
            source_values = group["_source_dir"].dropna().astype(str)
            if len(source_values):
                source_dir = Path(source_values.iloc[0])

        # 聚合性能指标
        agg = aggregate_run(group)
        agg.update(load_chain_metrics(raw_dir, wf_run_id, source_dir=source_dir))
        agg["wf_run_id"] = wf_run_id

        # 追加训练参数（取第一行即可，同一 run 内所有 split 参数相同）
        first = group.iloc[0]
        for col in PARAM_COLS:
            if col in first.index and col != "wf_run_id":
                agg[col] = first[col]
            elif col != "wf_run_id":
                agg[col] = None

        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    # 列顺序：wf_run_id → 参与评分的指标（按权重降序）→ 非评分指标 → 训练参数
    scored_cols = [col for col, _w, _d in sorted(SCORE_CONFIG, key=lambda x: -x[1])]

    non_scored_metric_cols = [
        "n_splits",
        "model_version_range",
        # 选股指标组合补充
        "daily_rankic_mean",
        "icir",
        "selection_monotonicity",
        # 全周期串联补充
        "chain_total_return",
        "chain_sharpe",
        "chain_trading_days",
        # 回测补充
        "bt_annual_return_mean",
        "bt_calmar_mean",
        "bt_total_return_mean",
        "bt_max_drawdown_worst",
        "bt_volatility_mean",
        # 统计补充
        "oos_rankic_ir_mean",
        "oos_rankic_ir_std",
        "oos_top100_median_mean",
        "oos_top100_win_rate",
        "oos_top300_median_mean",
        "oos_top300_win_rate",
        # 训练质量补充
        "val_rankic_ir_mean",
        "best_iter_mean",
        "best_iter_min",
        "best_iter_max",
        "best_iter_std",
    ]

    param_cols_ordered = [c for c in PARAM_COLS if c != "wf_run_id"]

    key_cols = [
        "key_top20_hit_rate_mean",
        "key_top20_avg_return_median_mean",
        "key_top30_hit_rate_mean",
        "key_top30_avg_return_median_mean",
    ]
    all_cols = (
        [
            "wf_run_id",
            "max_depth",
            "learning_rate",
            "rank_weight_topk",
            "rank_weight",
        ]
        + key_cols
        + scored_cols
        + non_scored_metric_cols
        + param_cols_ordered
    )
    df = pd.DataFrame(rows)
    # 只保留存在的列，并按首次出现去重，避免重复列名触发后续 reindex 异常。
    final_cols = []
    for col in all_cols:
        if col in df.columns and col not in final_cols:
            final_cols.append(col)
    df = df[final_cols]

    df = df.reset_index(drop=True)

    # 列名改为中文
    df = df.rename(columns={k: v for k, v in COL_NAMES.items() if k in df.columns})

    return df


def _build_period_label(row: pd.Series) -> str:
    """优先使用批量脚本传入的时间段标签，否则退回到起止日期。"""
    batch_period_label = row.get(COL_NAMES["batch_period_label"])
    if pd.notna(batch_period_label) and str(batch_period_label).strip():
        return str(batch_period_label)

    wf_start = row.get(COL_NAMES["wf_start_date"])
    wf_end = row.get(COL_NAMES["wf_end_date"])
    if pd.notna(wf_start) and pd.notna(wf_end):
        return f"{wf_start}~{wf_end}"
    return "未标注"


def _extract_run_timestamp(run_id: str) -> str:
    """从 wf_run_id 中提取时间戳（YYYYMMDDHHMMSS），用于判定最新 run。"""
    parts = str(run_id).strip().split("_")
    if len(parts) < 3:
        return ""
    date_part, time_part = parts[1], parts[2]
    if len(date_part) == 8 and len(time_part) == 6 and date_part.isdigit() and time_part.isdigit():
        return date_part + time_part
    return ""


def _latest_run_timestamp_from_text(value) -> str:
    """从运行ID或运行ID列表中提取最新时间戳。"""
    text = str(value) if value is not None and pd.notna(value) else ""
    max_ts = ""
    for match in re.finditer(r"wf_(\d{8})_(\d{6})", text):
        ts = match.group(1) + match.group(2)
        if ts > max_ts:
            max_ts = ts
    if not max_ts:
        max_ts = _extract_run_timestamp(text)
    return max_ts


def _format_run_timestamp(ts: str) -> str:
    if not ts or len(ts) != 14:
        return ""
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}"


def sort_by_latest_run_time(
    df: pd.DataFrame,
    source_col: str,
    secondary_sort_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """按来源列中的最新 wf_run_id 时间倒序排序，并添加可读的最新运行时间列。"""
    if df.empty or source_col not in df.columns:
        return df

    result = df.copy()
    result["__latest_run_ts"] = result[source_col].map(_latest_run_timestamp_from_text)
    result["最新运行时间"] = result["__latest_run_ts"].map(_format_run_timestamp)
    cols = list(result.columns)
    if "最新运行时间" in cols and source_col in cols:
        cols.remove("最新运行时间")
        cols.insert(cols.index(source_col) + 1, "最新运行时间")
        result = result[cols]
    sort_cols = ["__latest_run_ts"]
    ascending = [False]
    if secondary_sort_cols:
        for col in secondary_sort_cols:
            if col in result.columns:
                sort_cols.append(col)
                ascending.append(False)
    result = result.sort_values(sort_cols, ascending=ascending, na_position="last")
    result = result.drop(columns=["__latest_run_ts"])
    return result.reset_index(drop=True)


def build_period_stability_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """按参数组合跨时间段聚合，输出稳定性汇总。"""
    if comp_df.empty:
        return pd.DataFrame()

    varying_cols = {
        COL_NAMES["wf_run_id"],
        COL_NAMES["wf_start_date"],
        COL_NAMES["wf_end_date"],
        COL_NAMES["batch_period_label"],
        COL_NAMES["split_count"],
        COL_NAMES["final_date"],
        COL_NAMES["start_model_version"],
    }
    metric_cols = {
        "综合得分",
        COL_NAMES["chain_cagr"],
        COL_NAMES["chain_max_drawdown"],
        COL_NAMES["chain_sharpe"],
        COL_NAMES["oos_cross_split_ir"],
        COL_NAMES["bt_win_rate"],
    }
    group_cols = []
    for key in PARAM_COLS:
        if key == "wf_run_id" or key in SEED_STABILITY_EXCLUDED_MODEL_KEYS:
            continue
        col = COL_NAMES.get(key, key)
        if col and col in comp_df.columns and col not in varying_cols and col not in metric_cols:
            group_cols.append(col)

    if not group_cols:
        return pd.DataFrame()

    working_df = comp_df.copy()
    working_df["__时间段标签"] = working_df.apply(_build_period_label, axis=1)
    run_id_col = COL_NAMES["wf_run_id"]
    working_df["__run_ts"] = working_df[run_id_col].map(_extract_run_timestamp)

    # 同一参数组、同一时间段可能会有多次重复 run（例如扫描了未生效参数）。
    # 这里先按 run 时间戳倒序去重，只保留每个时间段最新的一条，避免时间段数被重复放大。
    dedup_subset = group_cols + ["__时间段标签"]
    working_df = working_df.sort_values(
        ["__run_ts", run_id_col],
        ascending=[False, False],
        na_position="last",
    ).drop_duplicates(subset=dedup_subset, keep="first")

    rows = []
    for _, group in working_df.groupby(group_cols, dropna=False, sort=False):
        if len(group) <= 1:
            continue

        ordered_group = group.sort_values(
            ["__时间段标签", COL_NAMES["wf_run_id"]],
            ascending=[True, True],
            na_position="last",
        )

        score_series = pd.to_numeric(group.get("综合得分"), errors="coerce")
        cagr_series = pd.to_numeric(group.get(COL_NAMES["chain_cagr"]), errors="coerce")
        drawdown_series = pd.to_numeric(group.get(COL_NAMES["chain_max_drawdown"]), errors="coerce")
        ir_series = pd.to_numeric(group.get(COL_NAMES["oos_cross_split_ir"]), errors="coerce")
        win_rate_series = pd.to_numeric(group.get(COL_NAMES["bt_win_rate"]), errors="coerce")
        sharpe_series = pd.to_numeric(group.get(COL_NAMES["chain_sharpe"]), errors="coerce")

        score_std = score_series.std()
        cagr_std = cagr_series.std()
        ir_std = ir_series.std()

        score_penalty = 0.0 if pd.isna(score_std) else min(max(score_std / 20.0, 0.0), 1.0)
        cagr_penalty = 0.0 if pd.isna(cagr_std) else min(max(cagr_std / 0.2, 0.0), 1.0)
        ir_penalty = 0.0 if pd.isna(ir_std) else min(max(ir_std / 1.0, 0.0), 1.0)
        stability_score = round(
            (1 - (0.4 * score_penalty + 0.3 * cagr_penalty + 0.3 * ir_penalty)) * 100,
            1,
        )

        row = {col: group.iloc[0][col] for col in group_cols}
        row.update(
            {
                COL_NAMES["period_count"]: len(group),
                COL_NAMES["period_labels"]: " | ".join(
                    ordered_group["__时间段标签"].astype(str).tolist()
                ),
                COL_NAMES["run_id_list"]: " | ".join(
                    f"{period}:{run_id}"
                    for period, run_id in zip(
                        ordered_group["__时间段标签"].astype(str),
                        ordered_group[COL_NAMES["wf_run_id"]].astype(str),
                    )
                ),
                COL_NAMES["score_mean"]: (
                    round(score_series.mean(), 2) if score_series.notna().any() else None
                ),
                COL_NAMES["score_std"]: round(score_std, 2) if pd.notna(score_std) else None,
                COL_NAMES["score_min"]: (
                    round(score_series.min(), 2) if score_series.notna().any() else None
                ),
                COL_NAMES["score_max"]: (
                    round(score_series.max(), 2) if score_series.notna().any() else None
                ),
                COL_NAMES["chain_cagr_mean"]: (
                    round(cagr_series.mean(), 6) if cagr_series.notna().any() else None
                ),
                COL_NAMES["chain_cagr_std"]: round(cagr_std, 6) if pd.notna(cagr_std) else None,
                COL_NAMES["chain_cagr_min"]: (
                    round(cagr_series.min(), 6) if cagr_series.notna().any() else None
                ),
                COL_NAMES["chain_max_drawdown_mean"]: (
                    round(drawdown_series.mean(), 6) if drawdown_series.notna().any() else None
                ),
                COL_NAMES["chain_max_drawdown_worst"]: (
                    round(drawdown_series.min(), 6) if drawdown_series.notna().any() else None
                ),
                COL_NAMES["oos_cross_split_ir_mean"]: (
                    round(ir_series.mean(), 4) if ir_series.notna().any() else None
                ),
                COL_NAMES["oos_cross_split_ir_std"]: round(ir_std, 4) if pd.notna(ir_std) else None,
                COL_NAMES["bt_win_rate_mean"]: (
                    round(win_rate_series.mean(), 4) if win_rate_series.notna().any() else None
                ),
                COL_NAMES["bt_win_rate_min"]: (
                    round(win_rate_series.min(), 4) if win_rate_series.notna().any() else None
                ),
                COL_NAMES["chain_sharpe_mean"]: (
                    round(sharpe_series.mean(), 4) if sharpe_series.notna().any() else None
                ),
                COL_NAMES["chain_sharpe_std"]: (
                    round(sharpe_series.std(), 4) if sharpe_series.notna().sum() > 1 else None
                ),
                COL_NAMES["stability_score"]: max(stability_score, 0.0),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    ordered_cols = [
        COL_NAMES["period_count"],
        COL_NAMES["period_labels"],
        COL_NAMES["run_id_list"],
        COL_NAMES["stability_score"],
        COL_NAMES["score_mean"],
        COL_NAMES["score_std"],
        COL_NAMES["score_min"],
        COL_NAMES["score_max"],
        COL_NAMES["chain_cagr_mean"],
        COL_NAMES["chain_cagr_std"],
        COL_NAMES["chain_cagr_min"],
        COL_NAMES["chain_max_drawdown_mean"],
        COL_NAMES["chain_max_drawdown_worst"],
        COL_NAMES["oos_cross_split_ir_mean"],
        COL_NAMES["oos_cross_split_ir_std"],
        COL_NAMES["bt_win_rate_mean"],
        COL_NAMES["bt_win_rate_min"],
        COL_NAMES["chain_sharpe_mean"],
        COL_NAMES["chain_sharpe_std"],
    ]
    ordered_cols += [col for col in group_cols if col not in ordered_cols]
    result = pd.DataFrame(rows)
    result = result[[col for col in ordered_cols if col in result.columns]]
    result = sort_by_latest_run_time(
        result, COL_NAMES["run_id_list"], [COL_NAMES["stability_score"]]
    )
    return result

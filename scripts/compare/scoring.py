# -*- coding: utf-8 -*-
"""综合/选股/模型Alpha/Seed稳定性/交易参数/实盘候选 评分体系。"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.compare.constants import (
    CANDIDATE_MIN_CHAIN_CAGR_WORST,
    CANDIDATE_MIN_CHAIN_MAX_DRAWDOWN,
    CANDIDATE_MIN_EFFECTIVE_PAIR_CONTEXTS,
    CANDIDATE_MIN_MODEL_ALPHA,
    COL_NAMES,
    MODEL_ALPHA_SCORE_CONFIG,
    MODEL_PARAM_KEYS,
    PAIR_CONTEXT_KEYS,
    SCORE_CONFIG,
    SEED_STABILITY_EXCLUDED_MODEL_KEYS,
    TRADE_PARAM_KEYS,
)
from scripts.compare.loading import _is_missing_param_value
from scripts.compare.aggregate import sort_by_latest_run_time


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """计算综合得分（0~100，越高越好）

    方法：对 SCORE_CONFIG 中的每个指标在当前实验集内做百分位排名（0~1），
    按权重加权求和后乘以 100。

    设计原则：
    - 百分位排名（percentile rank）完全回避量纲差异，结果仅反映相对优劣
    - NaN 值视为中性（百分位 0.5），不奖励也不惩罚
    - 单个实验时各指标百分位均为 0.5，得分固定为 50.0
    - ascending=True  时：最大值 → 百分位 1.0（高好）
    - ascending=False 时：最小值 → 百分位 1.0（低好）
    """
    n = len(df)
    weighted_pct = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for eng_key, weight, direction in SCORE_CONFIG:
        col = COL_NAMES.get(eng_key)
        if col is None or col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        if direction == "abs_low":
            s = s.abs()
            ascending = False  # 最大绝对值 → rank 1 → 百分位低 → 得分低 ✓
        elif direction == "low":
            ascending = False  # 最大值 → rank 1 → 百分位低 → 得分低 ✓
        else:  # "high"
            ascending = True  # 最小值 → rank 1 → 百分位低 → 大值得高分 ✓

        if n > 0:
            # rank(ascending=True): 最小→1, 最大→n → pct=rank/n
            pct = s.rank(ascending=ascending, method="average", na_option="keep") / n
            pct = pct.fillna(0.5)
        else:
            pct = pd.Series(0.5, index=df.index)

        weighted_pct += weight * pct
        total_weight += weight

    if total_weight > 0:
        score = (weighted_pct / total_weight) * 100
    else:
        score = pd.Series(50.0, index=df.index)

    return score.round(1)


def compute_selection_score(df: pd.DataFrame) -> pd.Series:
    """计算选股综合得分（0~100，越高越好）。

    指标与权重（选股优先版）：
    - RankIC均值: 30%
    - ICIR: 30%
    - Top30超额均值: 40%

    说明：
    - 每项先做百分位排名（0~1）后加权
    - 对缺失项按“有效项重归一”处理，避免旧数据无新列时得分失真
    - 单个实验或缺失值按中性值 0.5 处理
    """
    scoring_items = [
        (COL_NAMES["daily_rankic_mean"], 0.30),
        (COL_NAMES["icir"], 0.30),
        (COL_NAMES["oos_top30_lift_mean"], 0.40),
    ]

    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series(50.0, index=df.index)

    weighted_pct = pd.Series(0.0, index=df.index)
    effective_weight = pd.Series(0.0, index=df.index)

    for col, weight in scoring_items:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        pct = s.rank(ascending=True, method="average", na_option="keep") / n
        pct = pct.fillna(0.5)

        valid_mask = s.notna().astype(float)
        weighted_pct += weight * pct
        effective_weight += weight * valid_mask

    # 对有效指标不足的行按中性分处理；其余按有效项权重重归一
    score = pd.Series(50.0, index=df.index)
    valid_rows = effective_weight > 0
    score.loc[valid_rows] = (weighted_pct.loc[valid_rows] / effective_weight.loc[valid_rows]) * 100
    return score.round(1)


def _cn_param_cols(keys: list[str], df: pd.DataFrame) -> list[str]:
    """将内部参数键转换为当前表中存在的中文列名。"""
    cols = []
    for key in keys:
        col = COL_NAMES.get(key, key)
        if col in df.columns and col not in cols:
            cols.append(col)
    return cols


def _comparison_model_param_cols(df: pd.DataFrame) -> list[str]:
    """对比报表中的模型参数默认忽略 seed 维度，避免重复试验被误判为不同超参。"""
    keys = [key for key in MODEL_PARAM_KEYS if key not in SEED_STABILITY_EXCLUDED_MODEL_KEYS]
    return _cn_param_cols(keys, df)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _weighted_percentile_score(
    df: pd.DataFrame,
    scoring_items: list[tuple[str, float, str]],
) -> pd.Series:
    """对指定列做百分位加权评分，缺失指标按有效权重重归一。"""
    if df.empty:
        return pd.Series(dtype=float)
    if len(df) == 1:
        return pd.Series(50.0, index=df.index)

    weighted_pct = pd.Series(0.0, index=df.index)
    effective_weight = pd.Series(0.0, index=df.index)
    for col, weight, direction in scoring_items:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if direction == "abs_low":
            rank_input = s.abs()
            ascending = False
        elif direction == "low":
            rank_input = s
            ascending = False
        else:
            rank_input = s
            ascending = True
        pct = rank_input.rank(ascending=ascending, method="average", na_option="keep") / len(df)
        pct = pct.fillna(0.5)
        valid_mask = s.notna().astype(float)
        weighted_pct += weight * pct
        effective_weight += weight * valid_mask

    score = pd.Series(50.0, index=df.index)
    valid_rows = effective_weight > 0
    score.loc[valid_rows] = (weighted_pct.loc[valid_rows] / effective_weight.loc[valid_rows]) * 100
    return score.round(1)


def _signature_for_frame(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """为一组参数列生成可合并的稳定签名。"""
    if not cols:
        return pd.Series("", index=df.index)

    def normalize(value) -> str:
        if _is_missing_param_value(value):
            return "<NA>"
        return str(value).strip()

    return df[cols].apply(lambda row: "||".join(normalize(v) for v in row), axis=1)


def _unique_count(group: pd.DataFrame, cols: list[str]) -> int:
    if not cols:
        return len(group)
    return len(group[cols].drop_duplicates())


def _dedupe_columns(cols: list[str]) -> list[str]:
    result = []
    for col in cols:
        if col not in result:
            result.append(col)
    return result


def _build_model_alpha_score_table_for_cols(
    comp_df: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    """按指定模型参数列聚合，构建只评价选股 alpha 的评分表。"""
    if comp_df.empty:
        return pd.DataFrame()

    if not model_cols:
        return pd.DataFrame()

    rows = []
    for _, group in comp_df.groupby(model_cols, dropna=False, sort=False):
        row = {col: group.iloc[0][col] for col in model_cols}
        period_col = COL_NAMES["batch_period_label"]
        row.update(
            {
                "样本数": len(group),
                "时间段数": (
                    int(group[period_col].dropna().nunique())
                    if period_col in group.columns
                    else None
                ),
                "运行ID列表": (
                    " | ".join(group[COL_NAMES["wf_run_id"]].astype(str).tolist())
                    if COL_NAMES["wf_run_id"] in group.columns
                    else None
                ),
                "选股综合得分均值": round(_numeric_series(group, "选股综合得分").mean(), 4),
                "选股综合得分最差": round(_numeric_series(group, "选股综合得分").min(), 4),
                "RankIC均值": round(
                    _numeric_series(group, COL_NAMES["daily_rankic_mean"]).mean(), 6
                ),
                "ICIR均值": round(_numeric_series(group, COL_NAMES["icir"]).mean(), 4),
                "Top30超额均值": round(
                    _numeric_series(group, COL_NAMES["oos_top30_lift_mean"]).mean(), 6
                ),
                "Top30胜率": round(
                    _numeric_series(group, COL_NAMES["oos_top30_win_rate"]).mean(), 4
                ),
                "Top30最差中位收益": round(
                    _numeric_series(group, COL_NAMES["oos_top30_worst_median"]).min(), 6
                ),
                "分层单调性均值": round(
                    _numeric_series(group, COL_NAMES["selection_monotonicity"]).mean(), 4
                ),
                "验证_OOS_IR差距": round(
                    _numeric_series(group, COL_NAMES["train_val_ir_gap"]).mean(), 4
                ),
                "全周期CAGR均值": round(_numeric_series(group, COL_NAMES["chain_cagr"]).mean(), 6),
                "全周期最大回撤最差": round(
                    _numeric_series(group, COL_NAMES["chain_max_drawdown"]).min(), 6
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["模型参数签名"] = _signature_for_frame(result, model_cols)
    result.insert(0, "模型Alpha分", _weighted_percentile_score(result, MODEL_ALPHA_SCORE_CONFIG))
    result.insert(
        1, "模型Alpha排名", result["模型Alpha分"].rank(ascending=False, method="min").astype(int)
    )
    result.insert(2, "模型参数组ID", [f"M{i:04d}" for i in range(1, len(result) + 1)])
    front_cols = [
        "模型Alpha分",
        "模型Alpha排名",
        "最新运行时间",
        "模型参数组ID",
        "样本数",
        "时间段数",
        "选股综合得分均值",
        "选股综合得分最差",
        "RankIC均值",
        "ICIR均值",
        "Top30超额均值",
        "Top30胜率",
        "Top30最差中位收益",
        "分层单调性均值",
        "验证_OOS_IR差距",
        "全周期CAGR均值",
        "全周期最大回撤最差",
        "运行ID列表",
        "模型参数签名",
    ]
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in model_cols if col not in ordered]
    result = sort_by_latest_run_time(result, "运行ID列表", ["模型Alpha分"])
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in model_cols if col not in ordered]
    return result[ordered].reset_index(drop=True)


def build_model_alpha_score_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """按非 seed 模型参数聚合，构建只评价选股 alpha 的评分表。"""
    model_cols = _comparison_model_param_cols(comp_df)
    return _build_model_alpha_score_table_for_cols(comp_df, model_cols)


def build_model_seed_stability_table(
    comp_df: pd.DataFrame,
    model_alpha_df: pd.DataFrame,
) -> pd.DataFrame:
    """按排除 seed 后的模型参数聚合，观察同一超参跨 seed 的稳定性。"""
    if comp_df.empty or model_alpha_df.empty:
        return pd.DataFrame()

    model_cols = _cn_param_cols(MODEL_PARAM_KEYS, comp_df)
    seed_cols = _cn_param_cols(SEED_STABILITY_EXCLUDED_MODEL_KEYS, comp_df)
    stable_model_cols = [col for col in model_cols if col not in seed_cols]
    if not stable_model_cols or not seed_cols:
        return pd.DataFrame()

    seed_level_alpha_df = _build_model_alpha_score_table_for_cols(comp_df, model_cols)
    if seed_level_alpha_df.empty:
        return pd.DataFrame()

    working = comp_df.copy()
    working["模型参数签名"] = _signature_for_frame(working, model_cols)
    working["Seed稳定性签名"] = _signature_for_frame(working, stable_model_cols)
    lookup_cols = [
        "模型参数签名",
        "模型参数组ID",
        "模型Alpha分",
        "运行ID列表",
    ]
    lookup = seed_level_alpha_df[
        [col for col in lookup_cols if col in seed_level_alpha_df.columns]
    ].drop_duplicates("模型参数签名")
    working = working.merge(lookup, on="模型参数签名", how="left", suffixes=("", "_模型Alpha"))

    rows = []
    for _, group in working.groupby(stable_model_cols, dropna=False, sort=False):
        alpha_by_model = group.drop_duplicates("模型参数签名")
        alpha_series = _numeric_series(alpha_by_model, "模型Alpha分")
        row = {col: group.iloc[0][col] for col in stable_model_cols}
        seed_values = []
        for col in seed_cols:
            values = group[col].dropna().astype(str).drop_duplicates().tolist()
            if values:
                seed_values.append(f"{col}=" + ",".join(values))
        run_lists = []
        run_list_col = (
            "运行ID列表_模型Alpha" if "运行ID列表_模型Alpha" in group.columns else "运行ID列表"
        )
        if run_list_col in group.columns:
            run_lists = group[run_list_col].dropna().astype(str).drop_duplicates().tolist()
        row.update(
            {
                "Seed稳定性样本数": int(alpha_by_model["模型参数签名"].nunique()),
                "Seed列表": " | ".join(seed_values),
                "模型Alpha分均值": (
                    round(alpha_series.mean(), 1) if alpha_series.notna().any() else None
                ),
                "模型Alpha分中位数": (
                    round(alpha_series.median(), 1) if alpha_series.notna().any() else None
                ),
                "模型Alpha分标准差": (
                    round(alpha_series.std(), 2) if alpha_series.notna().sum() > 1 else None
                ),
                "模型Alpha分最差": (
                    round(alpha_series.min(), 1) if alpha_series.notna().any() else None
                ),
                "模型Alpha分最好": (
                    round(alpha_series.max(), 1) if alpha_series.notna().any() else None
                ),
                "模型参数组ID列表": (
                    " | ".join(
                        alpha_by_model["模型参数组ID"]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )
                    if "模型参数组ID" in alpha_by_model.columns
                    else None
                ),
                "运行ID列表": " | ".join(run_lists),
                "Seed稳定性签名": group["Seed稳定性签名"].iloc[0],
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Seed稳健分"] = (
        _numeric_series(result, "模型Alpha分中位数") * 0.50
        + _numeric_series(result, "模型Alpha分最差") * 0.35
        + (100 - _numeric_series(result, "模型Alpha分标准差").fillna(0).clip(lower=0, upper=100))
        * 0.15
    ).round(1)
    result.insert(
        0, "Seed稳健排名", result["Seed稳健分"].rank(ascending=False, method="min").astype(int)
    )
    front_cols = [
        "Seed稳健分",
        "Seed稳健排名",
        "最新运行时间",
        "Seed稳定性样本数",
        "Seed列表",
        "模型Alpha分中位数",
        "模型Alpha分均值",
        "模型Alpha分标准差",
        "模型Alpha分最差",
        "模型Alpha分最好",
        "模型参数组ID列表",
        "运行ID列表",
        "Seed稳定性签名",
    ]
    result = sort_by_latest_run_time(result, "运行ID列表", ["Seed稳健分"])
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in stable_model_cols if col not in ordered]
    return result[ordered].reset_index(drop=True)


def build_trade_param_score_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    """在相同模型参数+相同时间段内做配对百分位，聚合交易参数评分。"""
    if comp_df.empty:
        return pd.DataFrame()

    model_cols = _comparison_model_param_cols(comp_df)
    context_cols = _cn_param_cols(PAIR_CONTEXT_KEYS, comp_df)
    trade_cols = _cn_param_cols(TRADE_PARAM_KEYS, comp_df)
    pair_cols = _dedupe_columns(
        [col for col in model_cols + context_cols if col in comp_df.columns]
    )
    if not pair_cols or not trade_cols:
        return pd.DataFrame()

    metric_map = [
        (COL_NAMES["chain_cagr"], "CAGR配对百分位", "high"),
        (COL_NAMES["chain_total_return"], "总收益配对百分位", "high"),
        (COL_NAMES["chain_sharpe"], "夏普配对百分位", "high"),
        (COL_NAMES["bt_calmar_mean"], "Calmar配对百分位", "high"),
        (COL_NAMES["bt_win_rate"], "胜率配对百分位", "high"),
        (COL_NAMES["chain_max_drawdown"], "最大回撤配对百分位", "high"),
    ]

    paired_frames = []
    for pair_index, (_, group) in enumerate(
        comp_df.groupby(pair_cols, dropna=False, sort=False), 1
    ):
        trade_candidate_count = _unique_count(group, trade_cols)
        if trade_candidate_count < 2:
            continue
        pair_df = group.copy()
        pair_df["__pair_context_id"] = f"P{pair_index:06d}"
        pair_df["__pair_candidate_count"] = trade_candidate_count
        for metric_col, pct_col, direction in metric_map:
            s = _numeric_series(pair_df, metric_col)
            ascending = direction == "high"
            pct = s.rank(ascending=ascending, method="average", na_option="keep") / len(pair_df)
            pair_df[pct_col] = pct.fillna(0.5) * 100
        pair_df["单次配对交易收益分"] = _weighted_percentile_score(
            pair_df,
            [
                ("CAGR配对百分位", 0.40, "high"),
                ("总收益配对百分位", 0.25, "high"),
                ("Calmar配对百分位", 0.15, "high"),
                ("夏普配对百分位", 0.10, "high"),
                ("胜率配对百分位", 0.10, "high"),
            ],
        )
        pair_df["单次配对交易稳健分"] = _weighted_percentile_score(
            pair_df,
            [
                ("最大回撤配对百分位", 0.35, "high"),
                ("Calmar配对百分位", 0.25, "high"),
                ("夏普配对百分位", 0.20, "high"),
                ("胜率配对百分位", 0.10, "high"),
            ],
        )
        paired_frames.append(pair_df)

    if not paired_frames:
        return pd.DataFrame(
            columns=["交易收益分", "交易稳健分", "有效配对环境数", "配对样本数"] + trade_cols
        )

    paired_df = pd.concat(paired_frames, ignore_index=False)
    paired_df["交易参数签名"] = _signature_for_frame(paired_df, trade_cols)
    rows = []
    for _, group in paired_df.groupby(trade_cols, dropna=False, sort=False):
        cagr_pair_min = group.groupby("__pair_context_id")["CAGR配对百分位"].min()
        row = {col: group.iloc[0][col] for col in trade_cols}
        row.update(
            {
                "有效配对环境数": int(group["__pair_context_id"].nunique()),
                "配对样本数": len(group),
                "平均每组候选数": round(group["__pair_candidate_count"].mean(), 2),
                "胜出率": round((group["单次配对交易收益分"] >= 50).mean(), 4),
                "交易收益分": round(group["单次配对交易收益分"].mean(), 1),
                "交易收益分标准差": (
                    round(group["单次配对交易收益分"].std(), 4) if len(group) > 1 else None
                ),
                "交易收益分最差": round(group["单次配对交易收益分"].min(), 1),
                "交易稳健分": round(group["单次配对交易稳健分"].mean(), 1),
                "CAGR配对百分位均值": round(group["CAGR配对百分位"].mean(), 2),
                "总收益配对百分位均值": round(group["总收益配对百分位"].mean(), 2),
                "夏普配对百分位均值": round(group["夏普配对百分位"].mean(), 2),
                "Calmar配对百分位均值": round(group["Calmar配对百分位"].mean(), 2),
                "胜率配对百分位均值": round(group["胜率配对百分位"].mean(), 2),
                "最大回撤配对百分位均值": round(group["最大回撤配对百分位"].mean(), 2),
                "CAGR最差配对百分位": (
                    round(cagr_pair_min.mean(), 2) if len(cagr_pair_min) else None
                ),
                "CAGR原始均值": round(_numeric_series(group, COL_NAMES["chain_cagr"]).mean(), 6),
                "CAGR原始最差": round(_numeric_series(group, COL_NAMES["chain_cagr"]).min(), 6),
                "最大回撤原始均值": round(
                    _numeric_series(group, COL_NAMES["chain_max_drawdown"]).mean(), 6
                ),
                "最大回撤原始最差": round(
                    _numeric_series(group, COL_NAMES["chain_max_drawdown"]).min(), 6
                ),
                "运行ID列表": (
                    " | ".join(group[COL_NAMES["wf_run_id"]].astype(str).tolist())
                    if COL_NAMES["wf_run_id"] in group.columns
                    else None
                ),
                "交易参数签名": group["交易参数签名"].iloc[0],
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.insert(
        1, "交易收益排名", result["交易收益分"].rank(ascending=False, method="min").astype(int)
    )
    result.insert(2, "交易参数组ID", [f"T{i:04d}" for i in range(1, len(result) + 1)])
    front_cols = [
        "交易收益分",
        "交易收益排名",
        "最新运行时间",
        "交易参数组ID",
        "交易稳健分",
        "有效配对环境数",
        "配对样本数",
        "平均每组候选数",
        "胜出率",
        "交易收益分标准差",
        "交易收益分最差",
        "CAGR配对百分位均值",
        "总收益配对百分位均值",
        "夏普配对百分位均值",
        "Calmar配对百分位均值",
        "胜率配对百分位均值",
        "最大回撤配对百分位均值",
        "CAGR最差配对百分位",
        "CAGR原始均值",
        "CAGR原始最差",
        "最大回撤原始均值",
        "最大回撤原始最差",
        "运行ID列表",
        "交易参数签名",
    ]
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in trade_cols if col not in ordered]
    result = sort_by_latest_run_time(result, "运行ID列表", ["交易收益分"])
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in trade_cols if col not in ordered]
    return result[ordered].reset_index(drop=True)


def build_live_candidate_score_table(
    comp_df: pd.DataFrame,
    model_score_df: pd.DataFrame,
    trade_score_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建最终候选评分表，硬门槛未通过时实盘候选分直接置 0。"""
    if comp_df.empty or model_score_df.empty or trade_score_df.empty:
        return pd.DataFrame()

    model_cols = _comparison_model_param_cols(comp_df)
    trade_cols = _cn_param_cols(TRADE_PARAM_KEYS, comp_df)
    candidate_cols = _dedupe_columns(
        [col for col in model_cols + trade_cols if col in comp_df.columns]
    )
    if not candidate_cols:
        return pd.DataFrame()

    working = comp_df.copy()
    working["模型参数签名"] = _signature_for_frame(working, model_cols)
    working["交易参数签名"] = _signature_for_frame(working, trade_cols)

    model_lookup = model_score_df[["模型参数签名", "模型参数组ID", "模型Alpha分"]].drop_duplicates(
        "模型参数签名"
    )
    trade_lookup_cols = [
        "交易参数签名",
        "交易参数组ID",
        "交易收益分",
        "交易稳健分",
        "有效配对环境数",
    ]
    trade_lookup = trade_score_df[
        [c for c in trade_lookup_cols if c in trade_score_df.columns]
    ].drop_duplicates("交易参数签名")
    working = working.merge(model_lookup, on="模型参数签名", how="left")
    working = working.merge(trade_lookup, on="交易参数签名", how="left")

    rows = []
    group_cols = ["模型参数签名", "交易参数签名"]
    for _, group in working.groupby(group_cols, dropna=False, sort=False):
        row = {col: group.iloc[0][col] for col in candidate_cols}
        period_col = COL_NAMES["batch_period_label"]
        cagr_series = _numeric_series(group, COL_NAMES["chain_cagr"])
        drawdown_series = _numeric_series(group, COL_NAMES["chain_max_drawdown"])
        row.update(
            {
                "模型参数组ID": (
                    group["模型参数组ID"].iloc[0] if "模型参数组ID" in group.columns else None
                ),
                "交易参数组ID": (
                    group["交易参数组ID"].iloc[0] if "交易参数组ID" in group.columns else None
                ),
                "模型参数签名": group["模型参数签名"].iloc[0],
                "交易参数签名": group["交易参数签名"].iloc[0],
                "模型Alpha分": round(_numeric_series(group, "模型Alpha分").mean(), 1),
                "交易收益分": round(_numeric_series(group, "交易收益分").mean(), 1),
                "交易稳健分": round(_numeric_series(group, "交易稳健分").mean(), 1),
                "有效配对环境数": (
                    int(_numeric_series(group, "有效配对环境数").max())
                    if "有效配对环境数" in group.columns
                    and _numeric_series(group, "有效配对环境数").notna().any()
                    else 0
                ),
                "时间段数": (
                    int(group[period_col].dropna().nunique())
                    if period_col in group.columns
                    else None
                ),
                "样本数": len(group),
                "全周期CAGR均值": round(cagr_series.mean(), 6),
                "跨时间段CAGR最差": round(cagr_series.min(), 6),
                "全周期最大回撤最差": round(drawdown_series.min(), 6),
                "运行ID列表": (
                    " | ".join(group[COL_NAMES["wf_run_id"]].astype(str).tolist())
                    if COL_NAMES["wf_run_id"] in group.columns
                    else None
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    defense_input = result[["跨时间段CAGR最差", "全周期最大回撤最差"]].copy()
    result["最差场景防守分"] = _weighted_percentile_score(
        defense_input,
        [("跨时间段CAGR最差", 0.50, "high"), ("全周期最大回撤最差", 0.50, "high")],
    )
    result["实盘候选原始分"] = (
        _numeric_series(result, "模型Alpha分") * 0.45
        + _numeric_series(result, "交易收益分") * 0.30
        + _numeric_series(result, "交易稳健分") * 0.15
        + _numeric_series(result, "最差场景防守分") * 0.10
    ).round(1)

    model_pass = _numeric_series(result, "模型Alpha分") >= CANDIDATE_MIN_MODEL_ALPHA
    pair_pass = _numeric_series(result, "有效配对环境数") >= CANDIDATE_MIN_EFFECTIVE_PAIR_CONTEXTS
    drawdown_pass = (
        _numeric_series(result, "全周期最大回撤最差") >= CANDIDATE_MIN_CHAIN_MAX_DRAWDOWN
    )
    cagr_pass = _numeric_series(result, "跨时间段CAGR最差") >= CANDIDATE_MIN_CHAIN_CAGR_WORST
    result["模型Alpha门槛通过"] = model_pass
    result["有效配对门槛通过"] = pair_pass
    result["最大回撤门槛通过"] = drawdown_pass
    result["最差CAGR门槛通过"] = cagr_pass
    result["候选门槛通过"] = model_pass & pair_pass & drawdown_pass & cagr_pass

    failure_reasons = []
    for _, row in result.iterrows():
        reasons = []
        if not bool(row["模型Alpha门槛通过"]):
            reasons.append(f"模型Alpha分<{CANDIDATE_MIN_MODEL_ALPHA:g}")
        if not bool(row["有效配对门槛通过"]):
            reasons.append(f"有效配对环境数<{CANDIDATE_MIN_EFFECTIVE_PAIR_CONTEXTS}")
        if not bool(row["最大回撤门槛通过"]):
            reasons.append(f"全周期最大回撤<{CANDIDATE_MIN_CHAIN_MAX_DRAWDOWN:.0%}")
        if not bool(row["最差CAGR门槛通过"]):
            reasons.append(f"跨时间段CAGR最差<{CANDIDATE_MIN_CHAIN_CAGR_WORST:.0%}")
        failure_reasons.append("；".join(reasons) if reasons else "")
    result["候选门槛失败原因"] = failure_reasons
    result["实盘候选分"] = result["实盘候选原始分"].where(result["候选门槛通过"], 0.0)
    result.insert(
        1, "候选排名", result["实盘候选分"].rank(ascending=False, method="min").astype(int)
    )

    front_cols = [
        "实盘候选分",
        "候选排名",
        "最新运行时间",
        "实盘候选原始分",
        "候选门槛通过",
        "候选门槛失败原因",
        "模型Alpha分",
        "交易收益分",
        "交易稳健分",
        "最差场景防守分",
        "有效配对环境数",
        "模型Alpha门槛通过",
        "有效配对门槛通过",
        "最大回撤门槛通过",
        "最差CAGR门槛通过",
        "模型参数组ID",
        "交易参数组ID",
        "时间段数",
        "样本数",
        "全周期CAGR均值",
        "跨时间段CAGR最差",
        "全周期最大回撤最差",
        "运行ID列表",
        "模型参数签名",
        "交易参数签名",
    ]
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in candidate_cols if col not in ordered]
    result = sort_by_latest_run_time(result, "运行ID列表", ["实盘候选分"])
    ordered = [col for col in front_cols if col in result.columns]
    ordered += [col for col in candidate_cols if col not in ordered]
    return result[ordered].reset_index(drop=True)

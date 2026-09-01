"""Walk-forward 明细与串联净值导出。"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .chain_metrics import calculate_chain_metrics


def build_daily_topk_detail_df(
    df_eval: pd.DataFrame,
    original_return_col: str,
    topk_values: Tuple[int, ...] = (20, 30),
    score_column: str = "pred_score",
) -> pd.DataFrame:
    """构建逐日 TopK 明细，便于排查不同 seed 的名单分叉。"""
    output_columns = [
        "trade_date",
        "topk",
        "rank",
        "ts_code",
        "pred_score",
        "true_return",
        "score_column",
        "ml_score",
        "risk_score",
        "final_score",
    ]
    if df_eval is None or len(df_eval) == 0:
        return pd.DataFrame(columns=output_columns)

    valid_topk_values = []
    for value in topk_values:
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            continue
        if resolved > 0 and resolved not in valid_topk_values:
            valid_topk_values.append(resolved)
    if not valid_topk_values:
        return pd.DataFrame(columns=output_columns)

    detail_parts: List[pd.DataFrame] = []
    for trade_date in df_eval["trade_date"].drop_duplicates().tolist():
        day_df = df_eval[df_eval["trade_date"] == trade_date].copy()
        if day_df.empty or score_column not in day_df.columns or "ts_code" not in day_df.columns:
            continue
        day_df = day_df[day_df[score_column].notna()].copy()
        if day_df.empty:
            continue

        day_df = day_df.sort_values([score_column, "ts_code"], ascending=[False, True])
        for topk in valid_topk_values:
            topk_df = day_df.head(topk).copy()
            if topk_df.empty:
                continue
            topk_df["topk"] = topk
            topk_df["rank"] = np.arange(1, len(topk_df) + 1)
            topk_df["pred_score"] = topk_df[score_column]
            topk_df["true_return"] = topk_df.get(original_return_col, np.nan)
            topk_df["score_column"] = score_column
            for column in ["ml_score", "risk_score", "final_score"]:
                if column not in topk_df.columns:
                    topk_df[column] = np.nan
            detail_parts.append(topk_df[output_columns])

    if not detail_parts:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(detail_parts, ignore_index=True)


def write_walk_forward_topk_details(
    results: List[Dict], summary_csv_path: str, wf_run_id: str
) -> None:
    """将每个 split 的逐日 Top20/Top30 名单与预测分数落盘。"""
    output_dir = Path(summary_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_count = 0
    for result in results:
        detail_df = result.get("_topk_detail_df")
        if detail_df is None or len(detail_df) == 0:
            continue
        export_df = detail_df.copy()
        export_df.insert(0, "wf_run_id", wf_run_id)
        export_df.insert(1, "split_index", result.get("split_index"))
        export_df.insert(2, "test_start", result.get("test_start"))
        export_df.insert(3, "test_end", result.get("test_end"))
        export_df.insert(4, "model_version", result.get("model_version"))
        split_index = result.get("split_index")
        filename = f"walk_forward_topk_details_{wf_run_id}_split{int(split_index):02d}.csv"
        export_df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        exported_count += 1
    if exported_count > 0:
        logger.info(f"已导出 walk-forward TopK 明细: {exported_count} 个 split -> {output_dir}")


def write_walk_forward_trade_details(
    results: List[Dict], summary_csv_path: str, wf_run_id: str
) -> None:
    """将每个 split 的成交记录与买入执行归因落盘。"""
    output_dir = Path(summary_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    export_specs = (
        ("_trades", "trades"),
        ("_execution_attribution", "execution_attribution"),
    )
    exported_count = 0
    for result in results:
        split_index = result.get("split_index")
        for result_key, filename_tag in export_specs:
            detail_df = result.get(result_key)
            if detail_df is None or detail_df.empty:
                continue
            export_df = detail_df.copy()
            export_df.insert(0, "wf_run_id", wf_run_id)
            export_df.insert(1, "split_index", split_index)
            export_df.insert(2, "model_version", result.get("model_version"))
            filename = f"walk_forward_{filename_tag}_{wf_run_id}_split{int(split_index):02d}.csv"
            export_df.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
            exported_count += 1
    if exported_count > 0:
        logger.info(f"已导出 walk-forward 交易归因明细: {exported_count} 个文件 -> {output_dir}")


def chain_nav_splits(results: List[Dict], summary_csv_path: str, wf_run_id: str) -> None:
    """将各 split 的 OOS 回测净值首尾串联成全周期净值曲线。"""
    nav_parts = []
    for result in results:
        nav = result.get("_nav_curve")
        if nav is not None and not nav.empty and "nav" in nav.columns:
            part = nav[["nav"]].copy()
            part["split_index"] = result["split_index"]
            nav_parts.append(part)
    if not nav_parts:
        logger.info("无 OOS 回测净值可串联，跳过")
        return

    chained_records = []
    cumulative_nav = 1.0
    for part in nav_parts:
        raw = part["nav"].values
        if len(raw) == 0:
            continue
        scale = cumulative_nav / raw[0] if raw[0] != 0 else 1.0
        scaled = raw * scale
        for index, value in enumerate(scaled):
            part_index = part.index[index]
            chained_records.append(
                {
                    "date": part_index if not isinstance(part_index, int) else index,
                    "nav": value,
                    "split_index": part["split_index"].iloc[index],
                }
            )
        cumulative_nav = scaled[-1]

    chain_df = pd.DataFrame(chained_records)
    metrics = calculate_chain_metrics(chain_df)
    total_return = metrics["total_return"] or 0.0
    cagr = metrics["cagr"] or 0.0
    max_drawdown = metrics["max_drawdown"] or 0.0
    sharpe = metrics["sharpe"] or 0.0
    trading_days = metrics["trading_days"] or 0

    logger.info("=" * 60)
    logger.info("全周期串联净值（Walk-forward Chain）")
    logger.info(f"  总收益:   {total_return*100:.1f}%")
    logger.info(f"  CAGR:     {cagr*100:.1f}%")
    logger.info(f"  最大回撤: {max_drawdown*100:.1f}%")
    logger.info(f"  夏普:     {sharpe:.2f}")
    logger.info(f"  交易日数: {trading_days}")
    logger.info("=" * 60)

    output_dir = Path(summary_csv_path).parent
    chain_path = output_dir / f"chain_nav_{wf_run_id}.csv"
    chain_df.to_csv(chain_path, index=False, encoding="utf-8-sig")
    logger.info(f"串联净值已保存: {chain_path}")

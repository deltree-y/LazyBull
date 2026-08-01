#!/usr/bin/env python
"""分析 walk-forward 信号名单到实际成交与持仓收益的转化损耗。"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _normalize_date(series: pd.Series) -> pd.Series:
    """统一日期为 YYYYMMDD 字符串。"""
    parsed = pd.to_datetime(series, errors="coerce")
    normalized = parsed.dt.strftime("%Y%m%d")
    numeric_text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    compact_mask = numeric_text.str.fullmatch(r"\d{8}", na=False)
    return normalized.where(~compact_mask, numeric_text)


def _read_split_files(
    raw_dir: Path,
    filename_tag: str,
    wf_run_id: str,
    focus_splits: Optional[Iterable[int]],
) -> pd.DataFrame:
    """读取并合并指定类型的逐 split 文件。"""
    selected = set(focus_splits or [])
    parts: List[pd.DataFrame] = []
    pattern = f"walk_forward_{filename_tag}_{wf_run_id}_split*.csv"
    for path in sorted(raw_dir.glob(pattern)):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if frame.empty:
            continue
        split_index = int(frame["split_index"].iloc[0])
        if selected and split_index not in selected:
            continue
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_attribution_data(
    raw_dir: Path,
    wf_run_id: str,
    focus_splits: Optional[Iterable[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """加载执行归因、成交和 TopK 明细。"""
    data = {
        "execution": _read_split_files(raw_dir, "execution_attribution", wf_run_id, focus_splits),
        "trades": _read_split_files(raw_dir, "trades", wf_run_id, focus_splits),
        "topk": _read_split_files(raw_dir, "topk_details", wf_run_id, focus_splits),
    }
    if data["execution"].empty:
        raise FileNotFoundError(
            f"未找到执行归因文件: {raw_dir}/walk_forward_execution_attribution_"
            f"{wf_run_id}_split*.csv；请先用新版本 walk_forward 重新运行 OOS 回测"
        )
    return data


def enrich_execution_with_labels(execution: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    """为实际买入股票补充同一信号日的标签收益。"""
    result = execution.copy()
    result["signal_date"] = _normalize_date(result["signal_date"])
    if "ranking_date" not in result.columns:
        result["ranking_date"] = result["signal_date"]
    else:
        result["ranking_date"] = _normalize_date(result["ranking_date"])
    if topk.empty:
        result["actual_label_return"] = np.nan
        return result

    labels = topk.copy()
    labels["trade_date"] = _normalize_date(labels["trade_date"])
    labels = labels.sort_values("topk", ascending=False).drop_duplicates(
        ["split_index", "trade_date", "ts_code"]
    )
    labels = labels[["split_index", "trade_date", "ts_code", "true_return"]].rename(
        columns={
            "trade_date": "signal_date",
            "ts_code": "actual_stock",
            "true_return": "actual_label_return",
        }
    )
    labels = labels.rename(columns={"signal_date": "ranking_date"})
    return result.merge(
        labels,
        on=["split_index", "ranking_date", "actual_stock"],
        how="left",
    )


def build_holding_returns(trades: pd.DataFrame) -> pd.DataFrame:
    """提取已平仓持仓的真实收益。"""
    if trades.empty or "action" not in trades.columns:
        return pd.DataFrame()
    sells = trades[trades["action"] == "sell"].copy()
    if sells.empty:
        return sells
    for column in ("signal_date", "buy_date", "date"):
        if column in sells.columns:
            sells[column] = _normalize_date(sells[column])
    return sells


def _signal_top30_metrics(topk: pd.DataFrame, signal_dates: pd.Series) -> Dict[str, float]:
    """计算实际信号日而非全部交易日的 Top30 指标。"""
    if topk.empty:
        return {}
    frame = topk[topk["topk"] == 30].copy()
    frame["trade_date"] = _normalize_date(frame["trade_date"])
    frame = frame[frame["trade_date"].isin(set(signal_dates.dropna()))]
    if frame.empty:
        return {}
    daily_return = frame.groupby("trade_date")["true_return"].mean()
    return {
        "signal_day_top30_return_mean": float(daily_return.mean()),
        "signal_day_top30_return_median": float(daily_return.median()),
        "signal_day_top30_hit_rate": float((daily_return > 0).mean()),
        "signal_day_count": int(daily_return.size),
    }


def summarize_execution_gap(
    execution: pd.DataFrame,
    trades: pd.DataFrame,
    topk: pd.DataFrame,
) -> pd.DataFrame:
    """按 split 汇总信号、成交和持仓收益的转化指标。"""
    enriched = enrich_execution_with_labels(execution, topk)
    holdings = build_holding_returns(trades)
    rows: List[Dict[str, float]] = []

    for split_index, split_execution in enriched.groupby("split_index", sort=True):
        filled = split_execution[split_execution["status"] == "filled"]
        split_holdings = (
            holdings[holdings["split_index"] == split_index]
            if not holdings.empty
            else pd.DataFrame()
        )
        planned_count = len(split_execution)
        filled_count = len(filled)
        replacement = filled["actual_stock"] != filled["planned_stock"]
        actual_label = pd.to_numeric(filled["actual_label_return"], errors="coerce")
        holding_return = (
            pd.to_numeric(split_holdings["pnl_profit_pct"], errors="coerce")
            if not split_holdings.empty
            else pd.Series(dtype=float)
        )
        row: Dict[str, float] = {
            "split_index": int(split_index),
            "planned_slots": int(planned_count),
            "filled_slots": int(filled_count),
            "fill_rate": float(filled_count / planned_count) if planned_count else np.nan,
            "replacement_rate": float(replacement.mean()) if filled_count else np.nan,
            "top30_buy_coverage": (
                float((filled["actual_rank"] <= 30).mean()) if filled_count else np.nan
            ),
            "actual_rank_mean": (
                float(pd.to_numeric(filled["actual_rank"]).mean()) if filled_count else np.nan
            ),
            "signal_to_buy_return_mean": (
                float(pd.to_numeric(filled["signal_to_buy_return"], errors="coerce").mean())
                if filled_count
                else np.nan
            ),
            "actual_label_return_mean": float(actual_label.mean()),
            "actual_label_return_median": float(actual_label.median()),
            "actual_label_hit_rate": (
                float((actual_label.dropna() > 0).mean()) if actual_label.notna().any() else np.nan
            ),
            "closed_positions": int(len(split_holdings)),
            "holding_return_mean": float(holding_return.mean()),
            "holding_return_median": float(holding_return.median()),
            "holding_win_rate": (
                float((holding_return.dropna() > 0).mean())
                if holding_return.notna().any()
                else np.nan
            ),
        }
        split_topk = topk[topk["split_index"] == split_index] if not topk.empty else topk
        row.update(_signal_top30_metrics(split_topk, split_execution["ranking_date"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_failure_reasons(execution: pd.DataFrame) -> pd.DataFrame:
    """汇总未成交与替代成交的原计划失败原因。"""
    failed = execution[execution["reason"].notna()].copy()
    if failed.empty:
        return pd.DataFrame(columns=["split_index", "status", "reason", "count"])
    failed["reason"] = failed["reason"].fillna("未知")
    return (
        failed.groupby(["split_index", "status", "reason"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split_index", "status", "count"], ascending=[True, True, False])
    )


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    """输出便于快速判断的文本报告。"""
    lines = ["信号到成交收益归因", "=" * 72]
    for row in summary.to_dict("records"):
        lines.append(
            "Split {split_index}: 成交率={fill_rate:.1%}, 替换率={replacement_rate:.1%}, "
            "Top30覆盖={top30_buy_coverage:.1%}, 实际平均排名={actual_rank_mean:.1f}, "
            "实际标签收益={actual_label_return_mean:.2%}, 持仓收益={holding_return_mean:.2%}".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_splits(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="分析信号名单到实际成交和持仓收益的损耗")
    parser.add_argument("--raw-dir", required=True, help="walk-forward raw 输出目录")
    parser.add_argument("--wf-run-id", required=True, help="walk-forward 运行 ID")
    parser.add_argument("--focus-splits", default=None, help="逗号分隔的 split，如 6,9")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/reports/signal_execution_gap") / args.wf_run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_attribution_data(raw_dir, args.wf_run_id, _parse_splits(args.focus_splits))
    enriched = enrich_execution_with_labels(data["execution"], data["topk"])
    holdings = build_holding_returns(data["trades"])
    summary = summarize_execution_gap(data["execution"], data["trades"], data["topk"])
    failures = build_failure_reasons(data["execution"])

    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    enriched.to_csv(output_dir / "execution_details.csv", index=False, encoding="utf-8-sig")
    holdings.to_csv(output_dir / "holding_returns.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_dir / "failure_reasons.csv", index=False, encoding="utf-8-sig")
    write_report(summary, output_dir / "report.txt")
    print(summary.to_string(index=False))
    print(f"\n归因结果已保存: {output_dir}")


if __name__ == "__main__":
    main()

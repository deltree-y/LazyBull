"""回撤侧总扫描分析（terminal_loss 政策层 P2-4 前置条件）

用途：对一组「暴露门控 / 对称回补 / 止损」影子臂做统一口径比较，回答预登记判据
（见 `docs/plans/drawdown_side_sweep_prereg.md`）：

- 窗口指标（累计收益 / CAGR / MaxDD / 波动 / Sharpe）：先按窗口切片并**归一化到切片起点**，
  再复用 `ml/walk_forward/chain_metrics.py`（**禁止**自建年化公式）；
- 逐折明细（窗口内折）：每折独立切片 + 归一化 + 同一公共公式；
- 日差归因：与基线逐日收益差，按「信号期（λ<1）/ 信号期外（λ=1）」分组求和；
- 成交与成本：减仓 / 回补 / 止损笔数与金额、窗口买入额占比；
- 因果洁净：窗口外（split ≤ 9）与基线净值必须逐位一致（round(12)）。

口径来源：
- 链式净值 `raw/chain_nav_*.csv` 的 `date` 是**折内序号**（非真实日期），
  因此按 `raw/walk_forward_summary_*.csv` 的 `bt_start/bt_end` + 交易日历还原真实日期轴；
- 日期一律 `YYYYMMDD` 字符串（项目统一日期契约）。
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.data import DataLoader, Storage
from src.lazybull.ml.walk_forward.chain_metrics import calculate_chain_metrics

#: 引擎侧成交类型（唯一来源：backtest/exposure_trim.py 与 buy_execution._add_to_position）
TRIM_SELL_TYPE = "risk_trim"
REPLENISH_BUY_TYPE = "risk_replenish"
STOP_LOSS_SELL_TYPE = "stop_loss"


def _to_date_str(value) -> str:
    """统一为 YYYYMMDD 字符串（兼容 Timestamp / int / str）。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    for sep in ("-", "/"):
        text = text.replace(sep, "")
    return text[:8]


def load_trade_calendar(data_root: Path) -> pd.Series:
    """加载交易日历（clean 优先，缺失回退 raw），返回升序 YYYYMMDD 字符串序列。"""
    storage = Storage(root_path=str(data_root))
    loader = DataLoader(storage)
    calendar = loader.load_clean_trade_cal()
    if calendar is None or len(calendar) == 0:
        calendar = loader.load_trade_cal()
    if calendar is None or len(calendar) == 0:
        raise ValueError(f"无法读取交易日历: {data_root}")
    dates = calendar.loc[calendar["is_open"] == 1, "cal_date"].astype(str)
    return pd.Series(sorted(dates.unique()))


def load_arm_nav(
    batch_dir: Path,
    calendar: pd.Series,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取单个臂的链式净值，还原真实日期轴。

    Returns:
        (full_series, window_series)
        - full_series: 列 [date, split_index, nav]（全部折，date=YYYYMMDD）
        - window_series: 窗口切片且**归一化到切片起点**（列同上）
    """
    nav_files = glob.glob(str(batch_dir / "raw" / "chain_nav_*.csv"))
    if not nav_files:
        raise FileNotFoundError(f"缺少链式净值文件: {batch_dir / 'raw'}")
    nav = pd.read_csv(nav_files[0])
    summary_files = glob.glob(str(batch_dir / "raw" / "walk_forward_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"缺少 walk-forward 汇总文件: {batch_dir / 'raw'}")
    summary = pd.read_csv(summary_files[0])

    cal_set = set(calendar.tolist())
    frames = []
    for _, row in summary.iterrows():
        split = int(row["split_index"])
        start, end = _to_date_str(row["bt_start"]), _to_date_str(row["bt_end"])
        days = [d for d in calendar if start <= d <= end]
        part = nav[nav["split_index"] == split].reset_index(drop=True)
        if len(days) != len(part):
            raise ValueError(
                f"折 {split} 交易日数({len(days)}) 与链式净值行数({len(part)}) 不一致："
                f"{batch_dir.name}（{start}~{end}）"
            )
        part = part.assign(date=days)
        frames.append(part[["date", "split_index", "nav"]])
    full = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["split_index", "date"])
        .reset_index(drop=True)
    )
    if not set(full["date"]).issubset(cal_set):
        raise ValueError(f"还原的日期不在交易日历内: {batch_dir.name}")

    series = full
    if window_start and window_end:
        # 与 R-004 §7/§8 登记口径一致：窗口 = **与评估段相交的整折**（不裁剪折内日期），
        # 起点归一化后再复用公共链式公式（§7/§8 的「窗口累计收益」即该切片的 nav_end/nav_start−1）。
        splits = summary[
            (summary["bt_start"].map(_to_date_str) <= window_end)
            & (summary["bt_end"].map(_to_date_str) >= window_start)
        ]["split_index"].astype(int)
        if len(splits) == 0:
            raise ValueError(f"窗口 {window_start}~{window_end} 内无相交折: {batch_dir.name}")
        series = full[full["split_index"].isin(set(splits.tolist()))].reset_index(drop=True)
        base_nav = series["nav"].iloc[0]
        series = series.assign(nav=series["nav"] / base_nav)
    return full, series


def load_exposure_table(table_path: Optional[Path]) -> Optional[Dict[str, float]]:
    """读取两列暴露系数表（日期, 暴露系数）；缺失日按 1.0 处理（与引擎一致）。"""
    if table_path is None:
        return None
    frame = pd.read_csv(table_path)
    if frame.shape[1] < 2:
        raise ValueError(f"暴露系数表列数不足: {table_path}")
    dates = frame.iloc[:, 0].map(_to_date_str)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    return dict(zip(dates, values))


def window_metrics(series: pd.DataFrame) -> Dict[str, Optional[float]]:
    """窗口（或折内）指标：复用公共链式指标公式。"""
    return calculate_chain_metrics(series[["nav", "split_index"]])


def load_summary(batch_dir: Path) -> pd.DataFrame:
    """读取 walk-forward 汇总（逐折全折指标：`bt_total_return` / `bt_max_drawdown` / `bt_sharpe`）。

    判定口径说明：M2（逐折 MaxDD 改善）使用**全折口径**（与 R-004 §8 登记值一致），
    窗口切片指标只用于窗口级主判据 M1。
    """
    files = glob.glob(str(batch_dir / "raw" / "walk_forward_summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"缺少 walk-forward 汇总文件: {batch_dir / 'raw'}")
    return pd.read_csv(files[0])


def per_fold_metrics(batch_dir: Path) -> pd.DataFrame:
    """逐折全折指标（判定用）：直接取汇总 CSV，禁止自建公式。"""
    summary = load_summary(batch_dir)
    view = summary[
        ["split_index", "bt_start", "bt_end", "bt_total_return", "bt_max_drawdown", "bt_sharpe"]
    ].copy()
    return view.rename(
        columns={
            "split_index": "折序号",
            "bt_start": "起",
            "bt_end": "止",
            "bt_total_return": "收益",
            "bt_max_drawdown": "最大回撤",
            "bt_sharpe": "夏普",
        }
    )


def _daily_returns(series: pd.DataFrame) -> pd.Series:
    """按 date 索引的日收益（折边界重复日保留**前一折**行，其日收益为真实日收益）。"""
    ordered = series.sort_values(["split_index", "date"]).copy()
    ordered["ret"] = ordered.groupby("split_index")["nav"].pct_change()
    ordered = ordered.drop_duplicates(subset=["date"], keep="first")
    return ordered.set_index("date")["ret"]


def day_diff_attribution(
    base_series: pd.DataFrame,
    arm_series: pd.DataFrame,
    table: Optional[Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """逐日收益差归因：按信号期（λ<1）/ 信号期外（λ=1）分组。

    Returns:
        (逐日明细表, 分组汇总)  —— 明细列 [日期, 基线收益, 臂收益, 日差, 组别]
    """
    base_ret = _daily_returns(base_series).rename("基线收益")
    arm_ret = _daily_returns(arm_series).rename("臂收益")
    merged = pd.concat([base_ret, arm_ret], axis=1, join="inner")
    if len(merged) != len(base_ret) or len(merged) != len(arm_ret):
        raise ValueError(
            f"日差归因要求两臂交易日集合完全一致（基线 {len(base_ret)} / 臂 {len(arm_ret)} "
            f"/ 交集 {len(merged)}）"
        )
    merged["日差"] = merged["臂收益"] - merged["基线收益"]
    if table is None:
        merged["组别"] = "无暴露表（全 λ=1）"
    else:
        lam = merged.index.map(lambda d: table.get(d, 1.0))
        merged["组别"] = ["信号期内（λ<1）" if v < 1.0 else "信号期外（λ=1）" for v in lam]
    merged["日差_pp"] = merged["日差"] * 100.0
    detail = merged.reset_index()[["date", "基线收益", "臂收益", "日差", "日差_pp", "组别"]]
    detail = detail.rename(columns={"date": "日期"})
    group = (
        detail.dropna(subset=["日差_pp"])
        .groupby("组别")["日差_pp"]
        .agg(日数="count", 日差合计_pp="sum")
        .reset_index()
    )
    summary: Dict[str, float] = {"日差合计_pp": float(detail["日差_pp"].dropna().sum())}
    for _, row in group.iterrows():
        summary[f"{row['组别']}_日数"] = int(row["日数"])
        summary[f"{row['组别']}_pp"] = float(row["日差合计_pp"])
    return detail, summary


def collect_trades(batch_dir: Path, window_start: str, window_end: str) -> pd.DataFrame:
    """汇总窗口内成交（全部折），附窗口过滤后的日期列。"""
    files = sorted(glob.glob(str(batch_dir / "raw" / "walk_forward_trades_*_split*.csv")))
    if not files:
        raise FileNotFoundError(f"缺少成交明细: {batch_dir / 'raw'}")
    frames = [pd.read_csv(path) for path in files]
    trades = pd.concat(frames, ignore_index=True)
    trades["date"] = trades["date"].map(_to_date_str)
    return trades[(trades["date"] >= window_start) & (trades["date"] <= window_end)].reset_index(
        drop=True
    )


def _by_type(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """按成交类型过滤（列不存在时返回空表，兼容旧臂产物）。"""
    if column not in frame.columns or frame.empty:
        return frame.iloc[0:0]
    return frame[frame[column] == value]


def trade_summary(trades: pd.DataFrame) -> Dict[str, float]:
    """成交/成本统计（窗口内）：减仓 / 回补 / 止损 / 全部买入。"""
    sells = trades[trades["action"] == "sell"]
    buys = trades[trades["action"] == "buy"]
    trim = _by_type(sells, "sell_type", TRIM_SELL_TYPE)
    stop = _by_type(sells, "sell_type", STOP_LOSS_SELL_TYPE)
    replenish = _by_type(buys, "buy_type", REPLENISH_BUY_TYPE)
    return {
        "窗口买入金额": float(buys["amount"].sum()),
        "窗口买入成本": float(buys["cost"].sum()),
        "买入笔数": int(len(buys)),
        "减仓笔数": int(len(trim)),
        "减仓金额": float(trim["amount"].sum()),
        "减仓成本": float(trim["cost"].sum()),
        "回补笔数": int(len(replenish)),
        "回补金额": float(replenish["amount"].sum()),
        "回补成本": float(replenish["cost"].sum()),
        "止损笔数": int(len(stop)),
        "止损金额": float(stop["amount"].sum()),
        "止损成本": float(stop["cost"].sum()),
        "卖出笔数": int(len(sells)),
    }


def foreign_window_identity(
    base_full: pd.DataFrame, arm_full: pd.DataFrame, split_max: int = 9
) -> Dict[str, object]:
    """因果洁净：窗口外（split ≤ split_max）两臂净值必须逐位一致（round(12)）。"""
    base = base_full[base_full["split_index"] <= split_max]
    arm = arm_full[arm_full["split_index"] <= split_max]
    merged = base.merge(arm, on=["split_index", "date"], suffixes=("_base", "_arm"), how="inner")
    if len(merged) != len(base) or len(merged) != len(arm):
        return {"一致": False, "原因": f"行数不一致（{len(base)}/{len(arm)}/{len(merged)}）"}
    diff = (merged["nav_base"].round(12) - merged["nav_arm"].round(12)).abs()
    return {
        "一致": bool((diff == 0).all()),
        "行数": int(len(merged)),
        "最大差异": float(diff.max()),
    }


def summarize_arms(
    arms: Dict[str, Path],
    baseline_key: str,
    data_root: Path,
    window_start: str,
    window_end: str,
    tables: Optional[Dict[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对全部臂做统一口径分析。

    Returns:
        (指标表, 逐折表, 归因表, 成交表)
    """
    calendar = load_trade_calendar(data_root)
    tables = tables or {}
    loaded: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key, batch_dir in arms.items():
        loaded[key] = load_arm_nav(batch_dir, calendar, window_start, window_end)
        logger.info(f"已加载臂 {key}: {batch_dir.name}")

    base_full, base_window = loaded[baseline_key]
    metric_rows: List[Dict] = []
    fold_rows: List[Dict] = []
    attr_rows: List[Dict] = []
    trade_rows: List[Dict] = []
    for key, (full, window) in loaded.items():
        metrics = window_metrics(window)
        table = load_exposure_table(tables.get(key))
        if key == baseline_key:
            _, attr = day_diff_attribution(base_window, window, table)
            identity = {"一致": True, "行数": int(len(full)), "最大差异": 0.0}
        else:
            _, attr = day_diff_attribution(base_window, window, table)
            identity = foreign_window_identity(base_full, full)
        trades = collect_trades(arms[key], window_start, window_end)
        tstats = trade_summary(trades)
        row = {
            "臂": key,
            "批次": arms[key].name,
            "累计收益": metrics["total_return"],
            "CAGR": metrics["cagr"],
            "MaxDD": metrics["max_drawdown"],
            "波动": metrics["volatility"],
            "Sharpe": metrics["sharpe"],
            "交易日数": metrics["trading_days"],
            "窗口外逐位一致": identity["一致"],
            "窗口外最大差异": identity["最大差异"],
        }
        row.update(attr)
        row.update(tstats)
        metric_rows.append(row)

        folds = per_fold_metrics(arms[key])
        folds.insert(0, "臂", key)
        fold_rows.append(folds)

        attr_detail = day_diff_attribution(base_window, window, table)[0]
        attr_detail.insert(0, "臂", key)
        attr_rows.append(attr_detail)

        trades.insert(0, "臂", key)
        trade_rows.append(trades)

    metrics_df = pd.DataFrame(metric_rows)
    base_metrics = metrics_df[metrics_df["臂"] == baseline_key].iloc[0]
    for column, out in (
        ("累计收益", "Δ收益"),
        ("CAGR", "ΔCAGR"),
        ("MaxDD", "ΔMaxDD"),
        ("波动", "Δ波动"),
        ("Sharpe", "ΔSharpe"),
    ):
        metrics_df[out] = metrics_df[column] - base_metrics[column]
    return (
        metrics_df,
        pd.concat(fold_rows, ignore_index=True),
        pd.concat(attr_rows, ignore_index=True),
        pd.concat(trade_rows, ignore_index=True),
    )


def monthly_attribution(detail: pd.DataFrame) -> pd.DataFrame:
    """按月归因（事件依赖核查）：各月日差合计（pp）与集中度。"""
    if detail.empty:
        return pd.DataFrame(columns=["臂", "月份", "日差合计_pp", "日数"])
    frame = detail.copy()
    frame["月份"] = frame["日期"].astype(str).str.slice(0, 6)
    return (
        frame.dropna(subset=["日差_pp"])
        .groupby(["臂", "月份"])["日差_pp"]
        .agg(日差合计_pp="sum", 日数="count")
        .reset_index()
        .sort_values(["臂", "月份"])
    )


def format_metrics_table(metrics: pd.DataFrame) -> str:
    """中文展示：主判据列优先。"""
    columns = [
        "臂",
        "累计收益",
        "Δ收益",
        "CAGR",
        "ΔCAGR",
        "MaxDD",
        "ΔMaxDD",
        "波动",
        "Δ波动",
        "Sharpe",
        "ΔSharpe",
        "窗口外逐位一致",
        "减仓笔数",
        "减仓金额",
        "回补笔数",
        "回补金额",
        "止损笔数",
    ]
    view = metrics[[c for c in columns if c in metrics.columns]].copy()
    for column in view.columns:
        if view[column].dtype.kind == "f":
            if column in ("减仓金额", "回补金额", "止损金额"):
                view[column] = view[column].map(lambda v: f"{v:,.0f}")
            else:
                view[column] = view[column].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return view.to_string(index=False)


def fold_pass_summary(folds: pd.DataFrame, baseline_key: str, split_min: int = 10) -> pd.DataFrame:
    """逐折 MaxDD 改善判定（预登记 M2：**窗口内折**改善 ≥ 3/4）。

    窗口 = 门控评估段（20240102~20251204），对应折 10~13（split_min 起）。
    """
    folds = folds[folds["折序号"] >= split_min]
    base = folds[folds["臂"] == baseline_key].set_index("折序号")
    rows: List[Dict] = []
    for key, part in folds.groupby("臂", sort=False):
        part = part.set_index("折序号")
        common = part.index.intersection(base.index)
        improved = int((part.loc[common, "最大回撤"] >= base.loc[common, "最大回撤"]).sum())
        rows.append(
            {
                "臂": key,
                "窗口内折数": int(len(common)),
                "MaxDD 改善折数": improved,
                "改善要求": int((len(common) * 3 + 3) // 4),
                "逐折收益为正折数": int((part.loc[common, "收益"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def distinct_dates(arms: Iterable[Path]) -> List[str]:
    """调试辅助：列出各臂批次目录名。"""
    return [path.name for path in arms]

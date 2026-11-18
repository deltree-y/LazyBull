"""R3/R4 编排：A 基准继承、B 与 LOWVOL 两袖回放、四臂合并。

A 侧只读继承已封存账户（`run_r3`），不缩放、不重算、不改身份；
B/LOWVOL 走同一 `b_replay` 装配器，只有排名来源不同。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from typing import Any, Callable, Sequence

import pandas as pd

from scripts.v2_p7.run_buffer_native import load_prices
from src.lazybull.v2.core.execution.price_index import prepare_price_index

from scripts.v2_p7.defensive_ml import b_native, corporate_resolve, panel, spec, universe


class ReplayError(RuntimeError):
    """回放层错误（基准漂移、日期轴不一致）。"""


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


def prepare_corporate(root: Path, campaign_dir: Path, calendar: Sequence[str]) -> Any:
    """把分红原件复制到研究根并构造公司行为桥（不改生产 raw）。"""
    from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook

    target = campaign_dir / "corporate" / "dividend"
    target.mkdir(parents=True, exist_ok=True)
    source = Path(root) / "data/raw/dividend"
    copied = 0
    for path in sorted(source.glob("*.parquet")):
        destination = target / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
            copied += 1
    # 用户裁决的解冲突规则（窄口径），审计落 corporate/resolutions.json
    corporate_resolve.resolve_in_place(target)
    return NativeCorporateBook(calendar=list(calendar), dividend_dir=target), copied


def load_price_window(
    root: Path, calendar: Sequence[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """装载回放所需的未复权价格与交易标记。"""
    columns = (
        "ts_code", "trade_date", "open", "close",
        "is_suspended", "is_limit_up", "is_limit_down", "tradable",
    )
    days = [day for day in calendar if start <= day <= end]
    frames = []
    for day in days:
        path = Path(root) / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        if not path.exists():
            raise ReplayError(f"缺分区：{path}")
        frames.append(pd.read_parquet(path, columns=list(columns)))
    long = pd.concat(frames, ignore_index=True)
    long["trade_date"] = long["trade_date"].astype(str)
    out: dict[str, pd.DataFrame] = {}
    for name in ("open", "close", "is_suspended", "is_limit_up", "is_limit_down", "tradable"):
        out[name] = long.pivot(index="trade_date", columns="ts_code", values=name).reindex(days)
    return out


def load_a_series(root: Path, capital: int, arm: str) -> pd.Series:
    """继承 A 袖已封存账户的逐日总值（只读）。

    `A100` 与 `A80_CASH20` 为 parquet；`A80` 本体为 `A80/daily.json`（逐日记录）。
    """
    base = Path(root) / spec.A_BASELINE["relative"] / str(capital)
    parquet_names = {
        "A100": "A100_daily.parquet",
        "A80_CASH20": "A80_B_cash_daily.parquet",
    }
    parquet = base / parquet_names.get(arm, f"{arm}_daily.parquet")
    if parquet.exists():
        frame = pd.read_parquet(parquet)
        frame["trade_date"] = frame["trade_date"].astype(str)
        return frame.set_index("trade_date")["total_value"].astype(float)
    json_path = base / arm / "daily.json"
    if json_path.exists():
        records = json.loads(json_path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(records)
        frame["trade_date"] = frame["trade_date"].astype(str)
        return frame.set_index("trade_date")["total_value"].astype(float)
    raise ReplayError(f"A 基准缺件：{parquet} 或 {json_path}")


def lowvol_ranks(dataset_dir: Path, days: Sequence[str]) -> pd.DataFrame:
    """低波对照排名：def_vol60 升序，缺失排最后，同值按代码升序。"""
    frames = []
    for path in sorted(Path(dataset_dir).glob("panel_*.parquet")):
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "def_vol60"])
        frame["trade_date"] = frame["trade_date"].astype(str)
        frames.append(frame[frame["trade_date"].isin(set(days))])
    pool = pd.concat(frames, ignore_index=True)
    pool["missing"] = pool["def_vol60"].isna()
    pool = pool.sort_values(
        ["trade_date", "missing", "def_vol60", "ts_code"], ascending=[True, True, True, True]
    )
    pool["rank"] = pool.groupby("trade_date").cumcount() + 1
    return pool[["trade_date", "ts_code", "rank"]]


def load_industry(dataset_dir: Path, days: Sequence[str]) -> pd.DataFrame:
    """从缓存取 (日期, 代码) 行业桶。"""
    frames = []
    for path in sorted(Path(dataset_dir).glob("panel_*.parquet")):
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "industry_bucket"])
        frame["trade_date"] = frame["trade_date"].astype(str)
        frames.append(frame[frame["trade_date"].isin(set(days))])
    return pd.concat(frames, ignore_index=True)


def load_predictions(campaign_dir: Path, days: Sequence[str]) -> pd.DataFrame:
    """读入全折 OOS 预测（逐日全域排名）。"""
    path = Path(campaign_dir) / "predictions" / "predictions_all.parquet"
    if not path.exists():
        raise ReplayError(f"缺预测件：{path}")
    frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "score", "rank"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame[frame["trade_date"].isin(set(days))]


def combine_arms(
    capital: int,
    a100: pd.Series,
    a80: pd.Series,
    sleeve: pd.Series,
    b_capital: float,
    label: str,
) -> pd.DataFrame:
    """三臂合并：A100 / A80+现金 / A80+袖（同日真实资产相加，不归一化）。"""
    if not a80.index.equals(a100.index):
        raise ReplayError("A100 与 A80 日期轴不一致")
    if not sleeve.index.equals(a100.index):
        raise ReplayError("袖账户与 A 基准日期轴不一致")
    # A80 = 80% 本金在原 A 路径上运行；现金腿 = 剩余 20% 本金（不计息）
    cash_leg = float(b_capital)
    return pd.DataFrame(
        {
            "A100": a100,
            "A80_CASH20": a80 + cash_leg,
            f"A80_{label}20": a80 + sleeve,
        }
    )


def write_accounts(
    campaign_dir: Path, capital: int, label: str, result: dict[str, Any]
) -> dict[str, Any]:
    """落盘单袖的逐日账、成交与诊断。"""
    accounts = Path(campaign_dir) / "accounts" / str(capital)
    accounts.mkdir(parents=True, exist_ok=True)
    sleeve = result["sleeve"][str(capital)]
    daily = sleeve["daily"].copy()
    daily.to_parquet(accounts / f"{label}_daily.parquet", index=False)
    trades = sleeve["trades"]
    if len(trades):
        trades.to_parquet(accounts / f"{label}_trades.parquet", index=False)
    holdings = sleeve.get("holdings")
    if holdings is not None and len(holdings):
        holdings.to_parquet(accounts / f"{label}_holdings.parquet", index=False)
    diagnostics = sleeve["diagnostics"]
    (accounts / f"{label}_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    arms = result["arms"][str(capital)]
    arms.to_parquet(accounts / f"{label}_arms.parquet")
    return {
        "capital": int(capital),
        "label": label,
        "rows": int(len(daily)),
        "trades": int(len(trades)),
        "ending": float(daily["total_value"].iloc[-1]) if len(daily) else None,
        "diagnostics": diagnostics,
    }


class _NativeArm:
    """原生回放结果的账户视图（与旧 b_replay 返回值同形）。"""

    def __init__(
        self,
        daily: pd.DataFrame,
        trades: pd.DataFrame,
        diagnostics: dict,
        holdings: pd.DataFrame | None = None,
    ) -> None:
        self.daily = daily
        self.trades = trades
        self.diagnostics = diagnostics
        self.holdings = holdings if holdings is not None else pd.DataFrame()


def _rank_dict(ranks: pd.DataFrame, days: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    """宽表排名 -> {日期: [(code, score)]}（按 rank 升序、score 降序）。"""
    frame = ranks.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    if "score" not in frame.columns:
        # 低波对照只有名次：把名次转成"越大越优"的分数，保持引擎降序取头部语义
        frame["score"] = -frame["rank"].astype(float)
    frame = frame[frame["trade_date"].isin(set(days))]
    if "rank" in frame.columns:
        frame = frame.sort_values(["trade_date", "rank"])
    else:
        frame = frame.sort_values(["trade_date", "score"], ascending=[True, False])
    grouped = frame.groupby("trade_date")
    return {
        str(day): [(str(code), float(score)) for code, score in zip(part["ts_code"], part["score"])]
        for day, part in grouped
    }


def _native_arm(
    root: Path,
    ranked: dict[str, list[tuple[str, float]]],
    days: Sequence[str],
    calendar: Sequence[str],
    capital: float,
    corporate: Any,
    pit: pd.DataFrame,
    l1_names: dict[str, str],
    anchor: str,
    log: Callable[[str], None],
) -> _NativeArm:
    """用原生引擎跑一个 B 账户，并把产物映射成账户 schema。"""
    # 价格必须覆盖**整个回放窗**并留出预热；只取尾部窗口会让早期交易日没有行情
    price_days = [day for day in calendar if day <= days[-1]][-(len(days) + 260):]
    prices, _identity = load_prices(root, price_days, warmup_cutoff=days[0])
    indexes = prepare_price_index(prices)
    result = b_native.run_native_sleeve(
        root=root,
        ranked=ranked,
        calendar=days,
        prices=prices,
        indexes=indexes,
        capital=capital,
        corporate_book=corporate,
        pit=pit,
        l1_names=l1_names,
        anchor=anchor,
    )
    if not result["validation"]["valid"]:
        raise ReplayError(f"原生账户守恒验证失败：{result['validation'].get('violations')[:3]}")
    daily = pd.DataFrame(result["daily"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.strftime("%Y%m%d")
    for column in ("stale_market_value",):
        if column not in daily.columns:
            daily[column] = 0.0
    if "missing_price_count" not in daily.columns:
        daily["missing_price_count"] = 0
    trades = pd.DataFrame(result["trades"])
    refresh_days = len(result.get("pit_refresh_days", []))
    diagnostics = {
        "engine": "v2_native",
        "pit_refresh_days": refresh_days,
        "deviations": [dict(item) for item in b_native.DEVIATIONS],
        "hook_stats": result.get("hook_stats", {}),
        "engine_params": result.get("engine_params", {}),
        "validation": result["validation"],
        "stats": {"trades": int(len(trades)), "pit_refresh_days": refresh_days},
    }
    holdings = pd.DataFrame(result.get("holdings_snapshot") or [])
    if len(holdings) and "trade_date" in holdings.columns:
        holdings["trade_date"] = pd.to_datetime(holdings["trade_date"]).dt.strftime("%Y%m%d")
    log(f"native arm: days={len(daily)} trades={len(trades)} holdings={len(holdings)} "
        f"pit_refresh={diagnostics['pit_refresh_days']}")
    return _NativeArm(daily, trades, diagnostics, holdings)


def run_replay_stage(
    root: Path,
    campaign_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
    stress_slippage: float | None = None,
    rank_lag: int = 0,
    cash_discount: float = 0.0,
    sleeve_kind: str = "def",
) -> dict[str, Any]:
    """跑指定袖（def/lowvol）与四臂合并，返回逐臂日账与诊断。"""
    emit = _emit(log)
    calendar = panel.load_calendar(root)
    anchor = str(spec.WINDOW["cash_anchor"])
    limit = str(spec.WINDOW["end"])
    start_day = str(spec.WINDOW["start"])
    days = [day for day in calendar if start_day <= day <= limit]
    predictions = load_predictions(campaign_dir, days)
    ranks = predictions if sleeve_kind == "def" else lowvol_ranks(campaign_dir / "dataset", days)
    pit = universe.industry_pit_frame(root)
    l1_names = universe.industry_l1_names(universe.industry_code_table_frame(root), pit)
    corporate, copied = prepare_corporate(root, campaign_dir, days)
    emit(f"corporate book ready (copied {copied} files), sleeve={sleeve_kind}")
    if cash_discount:
        raise ReplayError("现金权益折扣压力尚无原生实现（F04-2 待办，禁止用替代口径冒充）")
    if stress_slippage is not None or rank_lag:
        raise ReplayError("滑点/延迟压力尚无原生实现（F04 待办，禁止用替代口径冒充）")

    output: dict[str, Any] = {"arms": {}, "sleeve": {}}
    for capital in spec.TARGETS["capital_scenarios"]:
        b_capital = float(capital) * float(spec.TARGETS["initial_b_fraction"])
        result = _native_arm(
            root=root,
            ranked=_rank_dict(ranks, days),
            days=days,
            calendar=calendar,
            capital=b_capital,
            corporate=corporate,
            pit=pit,
            l1_names=l1_names,
            anchor=anchor,
            log=emit,
        )
        daily = result.daily.set_index("trade_date")
        a100 = load_a_series(root, capital, "A100")
        a80 = load_a_series(root, capital, "A80")
        cash_arm = load_a_series(root, capital, "A80_CASH20")
        expected_cash = a80.to_numpy(dtype=float) + b_capital
        if not np.allclose(cash_arm.to_numpy(dtype=float), expected_cash, atol=1e-6):
            raise ReplayError(f"资本 {capital} 的 A80+现金 与 A80 本体不自洽（不得拼接净值）")
        common = a100.index.intersection(daily.index)
        if len(common) != len(a100.index) or len(common) != len(daily.index):
            raise ReplayError(
                f"资本 {capital} 日期轴不一致：A={len(a100.index)} B={len(daily.index)} 交集={len(common)}"
            )
        arms = combine_arms(
            capital,
            a100.reindex(common),
            a80.reindex(common),
            daily["total_value"].reindex(common),
            b_capital,
            sleeve_kind.upper(),
        )
        output["arms"][str(capital)] = arms
        output["sleeve"][str(capital)] = {
            "daily": result.daily,
            "trades": result.trades,
            "diagnostics": result.diagnostics,
            "holdings": result.holdings,
        }
    return output

"""B 袖原生薄装配与真实现金账户回放（协议 §9/§10）。

复用 `OperatingBaselineEngine`（原生逐日执行、真实股数、费用桥、公司行为桥），
只按本候选规格覆盖：6 槽、2 批 3/3（偏移 0/5）、等权、10 日持有、行业上限 2、
单笔下限 V、无跨日买入重试、无补齐窗口、到期开盘卖。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook
from scripts.v2_p7.buffer_native_replay import (
    engine_kwargs,
    initialize_driver,
    result_payload,
)
from scripts.v2_p7.buffer_native_validation import validate_account
from scripts.v2_p7.minute_repair_ml import domain, panel, spec
from scripts.v2_p7.minute_repair_ml.corporate_resolve import resolve_in_place
from scripts.v2_p7.operating_monthly_baseline import OperatingBaselineEngine
from scripts.v2_p7.run_buffer_native import cash_anchor


class ReplayError(RuntimeError):
    """回放层错误（基准漂移、日期轴不一致、守恒失败）。"""


def _emit(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log or (lambda message: None)


#: 规格选择（不是授权偏离）
SPEC_CHOICES: tuple[dict[str, str], ...] = (
    {
        "id": "no_cross_day_retry",
        "source": "协议 §9.2",
        "clause": "未成交买单不跨日重试、不启用补齐窗口",
        "effect": "max_retry_count=0, max_retry_days=0, enable_position_completion=False",
    },
    {
        "id": "single_execution_window",
        "source": "协议 §9.3",
        "clause": "T+1 收盘买入只允许一次执行窗口；同日按冻结排名顺位替补",
        "effect": "同批次内候补沿同一 T 排名向后取，不跨日",
    },
    {
        "id": "weight_cap_owner",
        "source": "协议 §9.1",
        "clause": "引擎 max_weight_per_stock 分母为整个 B 账户，应传 0.20",
        "effect": "共享 resolve_tranche_weight_cap 换算为批内 0.40，不手工二次乘除",
    },
)


class LRSignal:
    """冻结排名信号（不推理、不改并列顺序、top_n=6）。"""

    top_n = int(spec.PORTFOLIO["top_n"])

    def __init__(self, ranked: dict[str, list[tuple[str, float]]]) -> None:
        self.ranked = ranked

    def generate_ranked(self, date: pd.Timestamp, universe: Any, data: Any) -> list:
        return list(self.ranked.get(pd.Timestamp(date).strftime("%Y%m%d"), []))

    def generate(self, date: pd.Timestamp, universe: Any, data: Any) -> dict:
        return dict(self.generate_ranked(date, universe, data)[: self.top_n])


def industry_mapping_for_day(
    pit: pd.DataFrame, l1_names: dict[str, str], day: str
) -> dict[str, str]:
    """按当日 PIT 区间构造 `{ts_code: 一级行业名称}`（未知统一 `未知行业`）。"""
    day8 = str(day)
    frame = pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    active = frame[(frame["in_date"] <= day8) & (frame["out_date"] > day8)]
    mapping: dict[str, str] = {}
    for code, l1 in zip(active["ts_code"].astype(str), active["l1_code"].astype(str)):
        name = l1_names.get(domain.normalize_industry_code(l1))
        mapping[code] = name if name else "未知行业"
    return mapping


class PITIndustrySleeve(OperatingBaselineEngine):
    """研究侧子类：每信号日刷新 PIT 行业、行业硬上限与单笔买入下限。"""

    pit_frame: pd.DataFrame
    pit_names: dict[str, str]
    pit_refresh_log: list[str]

    def _get_min_buy_value_threshold(self, date: pd.Timestamp) -> float:
        """单笔买入下限 V = max(25000, min_commission/commission_rate)。"""
        return spec.min_ticket_value()

    def run_step(self, date: Any, trading_dates: Any, price_data: Any) -> Any:
        signal_days = getattr(self, "_signal_dates", None)
        is_signal_day = signal_days is None or date in signal_days
        if is_signal_day:
            day = pd.Timestamp(date).strftime("%Y%m%d")
            mapping = industry_mapping_for_day(self.pit_frame, self.pit_names, day)
            if mapping:
                frozen = getattr(self, "held_industry", {})
                held = {str(code) for code in getattr(self, "positions", {})}
                for code in held:
                    if code in frozen:
                        mapping[code] = frozen[code]
                    elif code in mapping:
                        frozen[code] = mapping[code]
                frozen = {key: value for key, value in frozen.items() if key in held}
                self.held_industry = frozen
                self.industry_mapping = mapping
                self._today_mapping = mapping
                self.pit_refresh_log.append(day)
        return super().run_step(date, trading_dates, price_data)

    def _post_filter_candidates(self, candidates: Any, date: Any) -> Any:
        """行业上限 = B 存量 + 本批已选（硬上限 2）；身份未知者不得进入买入池。"""
        mapping = getattr(self, "_today_mapping", None)
        stats = getattr(self, "_hook_stats", None)
        if stats is not None:
            stats["filter_calls"] = stats.get("filter_calls", 0) + 1
        if not mapping:
            return candidates
        cap = int(spec.PORTFOLIO["max_per_industry"])
        counts: dict[str, int] = {}
        for held in getattr(self, "positions", {}):
            bucket = mapping.get(str(held), "未知行业")
            counts[bucket] = counts.get(bucket, 0) + 1
        kept: list[tuple[str, float]] = []
        for code, score in candidates:
            bucket = mapping.get(str(code), "未知行业")
            if bucket == "未知行业":
                if stats is not None:
                    stats["dropped_unknown"] = stats.get("dropped_unknown", 0) + 1
                continue
            if counts.get(bucket, 0) >= cap:
                if stats is not None:
                    stats["dropped_cap"] = stats.get("dropped_cap", 0) + 1
                continue
            counts[bucket] = counts.get(bucket, 0) + 1
            kept.append((code, score))
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + len(kept)
        return kept


def _cost_model(slippage: float | None) -> Any:
    from src.lazybull.common.cost import CostModel

    return CostModel(
        float(spec.COSTS["commission_rate"]),
        float(spec.COSTS["min_commission"]),
        float(spec.COSTS["stamp_tax"]),
        float(slippage if slippage is not None else spec.COSTS["slippage"]),
    )


def b_overrides(slippage: float | None = None) -> dict[str, Any]:
    """B 规格相对 A 默认值的覆盖项（唯一来源 spec.PORTFOLIO / spec.COSTS）。"""
    portfolio = spec.PORTFOLIO
    return {
        "rebalance_freq": int(portfolio["rebalance_freq"]),
        "holding_period": int(portfolio["holding_period"]),
        "stagger_tranches": int(portfolio["stagger_tranches"]),
        "position_sizing": "equal",
        "max_weight_per_stock": float(portfolio["max_single_stock_weight"]),
        "max_per_industry": int(portfolio["max_per_industry"]),
        "sell_timing": "open",
        "enable_pending_order": True,
        "max_retry_count": int(portfolio["buy_retries_per_day"]),
        "max_retry_days": int(portfolio["buy_retry_days"]),
        "enable_position_completion": False,
        "completion_window_days": int(portfolio["fill_window_days"]),
        "enable_early_rebalance_on_empty": False,
        "enable_trailing_early_rebalance": False,
        "min_buy_value_ratio": 0.0,
        "stop_loss_config": None,
        "cost_model": _cost_model(slippage),
    }


def build_kwargs(
    root: Path,
    ranked: dict[str, list[tuple[str, float]]],
    capital: float,
    corporate_book: NativeCorporateBook,
    slippage: float | None = None,
) -> dict[str, Any]:
    """A 默认 kwargs + B 覆盖 + 公司行为桥 + 冻结排名信号。"""
    kwargs = engine_kwargs(Path(root), ranked, int(capital), False)
    kwargs.update(b_overrides(slippage))
    kwargs["signal"] = LRSignal(ranked)
    kwargs["corporate_book"] = corporate_book
    if str(kwargs.get("position_sizing")) != "equal":
        raise ReplayError("position_sizing 覆盖失败")
    if int(getattr(kwargs["signal"], "top_n", 0)) != int(spec.PORTFOLIO["top_n"]):
        raise ReplayError("top_n 覆盖失败")
    return kwargs


def run_native_sleeve(
    *,
    root: Path,
    ranked: dict[str, list[tuple[str, float]]],
    calendar: Sequence[str],
    prices: pd.DataFrame,
    indexes: Any,
    capital: float,
    corporate_book: NativeCorporateBook,
    pit: pd.DataFrame,
    l1_names: dict[str, str],
    anchor: str | None = None,
    slippage: float | None = None,
) -> dict[str, Any]:
    """逐日驱动原生引擎，返回产物与独立守恒校验。"""
    dates = [pd.Timestamp(str(day)) for day in calendar]
    engine = PITIndustrySleeve(**build_kwargs(root, ranked, capital, corporate_book, slippage))
    engine.pit_frame = pit
    engine.pit_names = l1_names
    engine.pit_refresh_log = []
    engine.record_holdings_snapshot = True
    engine._hook_stats = {}
    initialize_driver(engine, dates, prices, indexes)
    if anchor:
        daily, state = cash_anchor(anchor, int(capital))
        engine.research_daily.append(daily)
        engine.research_states.append(state)
    for date in dates:
        engine.run_step(date, dates, prices)
    result = result_payload(engine)
    validation = validate_account(result, initial_capital=float(capital))
    result["validation"] = validation
    result["pit_refresh_days"] = list(engine.pit_refresh_log)
    result["hook_stats"] = dict(getattr(engine, "_hook_stats", {}))
    pending = getattr(engine, "pending_order_manager", None)
    result["engine_params"] = {
        **{
            name: getattr(engine, name, None)
            for name in (
                "rebalance_freq",
                "holding_period",
                "stagger_tranches",
                "position_sizing",
                "max_weight_per_stock",
                "max_per_industry",
                "enable_pending_order",
                "enable_position_completion",
                "completion_window_days",
                "sell_timing",
                "min_buy_value_ratio",
            )
        },
        "max_retry_count": getattr(pending, "max_retry_count", None),
        "max_retry_days": getattr(pending, "max_retry_days", None),
    }
    result["min_ticket_value"] = spec.min_ticket_value()
    result["cost_model"] = {
        "commission_rate": float(spec.COSTS["commission_rate"]),
        "min_commission": float(spec.COSTS["min_commission"]),
        "stamp_tax": float(spec.COSTS["stamp_tax"]),
        "slippage": float(slippage if slippage is not None else spec.COSTS["slippage"]),
    }
    return result


def prepare_corporate(
    root: Path,
    campaign_dir: Path,
    calendar: Sequence[str],
    *,
    tag: str = "base",
) -> tuple[NativeCorporateBook, dict[str, Any]]:
    """把分红原件复制到研究根并构造公司行为桥（每次从 raw 重建副本）。"""
    target = Path(campaign_dir) / "corporate" / tag / "dividend"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    source = Path(root) / "data/raw/dividend"
    copied = 0
    for path in sorted(source.glob("*.parquet")):
        shutil.copy2(path, target / path.name)
        copied += 1
    audit = resolve_in_place(target, Path(campaign_dir) / "corporate" / tag / "resolutions.json")
    book = NativeCorporateBook(calendar=list(calendar), dividend_dir=target)
    return book, {"copied": copied, "audit": audit, "tag": tag}


def load_prices_for_window(
    root: Path, calendar: Sequence[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """装载未复权价格与交易标记（成交价口径）。"""
    days = [day for day in calendar if start <= day <= end]
    return panel.load_unadjusted_prices(root, days[0], days[-1], calendar)


def load_a_series(root: Path, capital: int, arm: str) -> pd.Series:
    """继承 A 袖已封存账户的逐日总值（只读，不缩放、不拼接）。"""
    base = Path(root) / spec.A_BASELINE["relative"] / str(capital)
    parquet_names = {
        "A100": "A100_daily.parquet",
        "A80": "A80_B_cash_daily.parquet",
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


def combine_arms(
    a100: pd.Series,
    a80: pd.Series,
    sleeves: dict[str, pd.Series],
    b_capital: float,
) -> pd.DataFrame:
    """四臂合并：A100 / A80+静态现金 / A80+LR20 / A80+DAILY20（同日真实资产相加）。"""
    if not a80.index.equals(a100.index):
        raise ReplayError("A100 与 A80 日期轴不一致")
    frame = pd.DataFrame(
        {"A100": a100.astype(float), "A80_CASH20": a80.astype(float) + float(b_capital)}
    )
    for label, series in sleeves.items():
        if not series.index.equals(a100.index):
            missing = set(a100.index) - set(series.index)
            raise ReplayError(f"袖 {label} 与 A 基准日期轴不一致（缺 {len(missing)} 日）")
        frame[f"A80_{label}"] = a80.astype(float) + series.astype(float)
    return frame


def run_arm(
    root: Path,
    *,
    ranked: dict[str, list[tuple[str, float]]],
    days: Sequence[str],
    capital: float,
    corporate_book: NativeCorporateBook,
    pit: pd.DataFrame,
    l1_names: dict[str, str],
    anchor: str,
    slippage: float | None = None,
    prices: pd.DataFrame | None = None,
    indexes: Any = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """跑单个 B 账户（含价格装载与守恒校验；价格可复用同一份）。"""
    from src.lazybull.v2.core.execution.price_index import prepare_price_index

    emit = _emit(log)
    if prices is None or indexes is None:
        price_days = [day for day in days]
        prices, _identity = load_prices_with_warmup(root, price_days, warmup=260)
        indexes = prepare_price_index(prices)
    result = run_native_sleeve(
        root=root,
        ranked=ranked,
        calendar=days,
        prices=prices,
        indexes=indexes,
        capital=capital,
        corporate_book=corporate_book,
        pit=pit,
        l1_names=l1_names,
        anchor=anchor,
        slippage=slippage,
    )
    if not result["validation"]["valid"]:
        raise ReplayError(f"原生账户守恒验证失败：{result['validation'].get('violations')[:3]}")
    daily = pd.DataFrame(result["daily"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.strftime("%Y%m%d")
    trades = pd.DataFrame(result["trades"])
    holdings = pd.DataFrame(result.get("holdings_snapshot") or [])
    if len(holdings) and "trade_date" in holdings.columns:
        holdings["trade_date"] = pd.to_datetime(holdings["trade_date"]).dt.strftime("%Y%m%d")
    emit(f"arm: days={len(daily)} trades={len(trades)} holdings={len(holdings)}")
    return {
        "daily": daily,
        "trades": trades,
        "holdings": holdings,
        "validation": result["validation"],
        "ending": result.get("ending"),
        "corporate": result.get("corporate"),
        "diagnostics": {
            "engine": "v2_native",
            "pit_refresh_days": len(result.get("pit_refresh_days", [])),
            "hook_stats": result.get("hook_stats", {}),
            "engine_params": result.get("engine_params", {}),
            "min_ticket_value": result.get("min_ticket_value"),
            "cost_model": result.get("cost_model"),
            "spec_choices": [dict(item) for item in SPEC_CHOICES],
        },
    }


def load_prices_with_warmup(
    root: Path, days: Sequence[str], warmup: int = 260
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """装载回放所需价格（含预热；不使用末 400 日缓存跑 2019 起账户）。"""
    from scripts.v2_p7.buffer_native_replay import load_prices

    return load_prices(Path(root), list(days), warmup_cutoff=str(days[0]))


def replay_days(root: Path) -> list[str]:
    """主窗交易日（含终点）。"""
    calendar = panel.load_calendar(root)
    start = str(spec.WINDOW["start"])
    end = str(spec.WINDOW["end"])
    return [day for day in calendar if start <= day <= end]


def load_industry(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """PIT 行业表与一级名称映射。"""
    pit = domain.industry_pit_frame(root)
    names = domain.industry_l1_names(domain.industry_code_table_frame(root), pit)
    return pit, names


def tranche_check(days: Sequence[str]) -> dict[str, Any]:
    """排期自检：0/5 偏移与 3/3 槽位（现金锚点不进排期）。"""
    from src.lazybull.trading.stagger import compute_tranche_schedule, get_tranche_target_count

    schedule = compute_tranche_schedule(
        [pd.Timestamp(day) for day in days],
        int(spec.PORTFOLIO["rebalance_freq"]),
        int(spec.PORTFOLIO["stagger_tranches"]),
    )
    offsets = sorted({int(value) for value in schedule.values()})
    slots = [
        get_tranche_target_count(
            index, int(spec.PORTFOLIO["top_n"]), int(spec.PORTFOLIO["stagger_tranches"])
        )
        for index in range(int(spec.PORTFOLIO["stagger_tranches"]))
    ]
    return {
        "tranche_ids": offsets,
        "tranche_slots": slots,
        "signal_days": len(schedule),
        "first_three": [str(day) for day in list(schedule)[:3]],
        "ok": offsets == [0, 1] and slots == [3, 3],
    }

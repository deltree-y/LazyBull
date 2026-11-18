"""B 袖原生薄装配：复用 `OperatingBaselineEngine`，只覆盖 B 的规格参数。

A 的默认值全部来自共享 `engine_kwargs`（20 日/两批/half_kelly/15% 上限），
本模块按协议 §9.2 逐项覆盖为 B 规格，不另写执行循环。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook
from scripts.v2_p7.buffer_native_replay import (
    engine_kwargs,
    initialize_driver,
    result_payload,
)
from scripts.v2_p7.buffer_native_validation import validate_account
from scripts.v2_p7.defensive_ml import spec
from scripts.v2_p7.operating_monthly_baseline import OperatingBaselineEngine
from scripts.v2_p7.run_buffer_native import cash_anchor


class BNativeError(RuntimeError):
    """B 袖原生装配失败（条件不满足即 fail-closed）。"""


#: 经用户裁决的规格偏离（唯一来源；报告与 manifest 必须原样引用）
DEVIATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "no_position_completion",
        "authorized_by": "user",
        "date": "2026-11-18",
        "clause": "协议 §9.2 买入补齐窗口 5 日",
        "reason": (
            "原生公司行为桥要求持仓对上物理来源批次；补齐窗口买入的批次不在登记内，"
            "LOWVOL 在 2021-11-15 触发 CorporateError（000958.SZ）"
        ),
        "effect": "enable_position_completion=False",
        "record": "specification/authorization_completion_off_20261118.json",
    },
    {
        "id": "no_cross_day_retry",
        "authorized_by": "user",
        "date": "2026-11-18",
        "clause": "协议 §9.2 买入重试（5 次 / 10 日）",
        "reason": (
            "挂单跨日执行会造成执行时行业溢出（实测单桶 6 只、20 天超限）；"
            "研究侧候选过滤对执行时溢出无效"
        ),
        "effect": "max_retry_count=0",
        "record": "specification/authorization_no_retry_20261118.json",
    },
)


def retry_disabled() -> bool:
    """是否按已授权偏离关闭跨日买入重试。"""
    return any(item["id"] == "no_cross_day_retry" for item in DEVIATIONS)


def completion_disabled() -> bool:
    """是否按已授权偏离关闭补齐窗口。"""
    return any(item["id"] == "no_position_completion" for item in DEVIATIONS)


def industry_mapping_for_day(
    pit: pd.DataFrame, l1_names: dict[str, str], day: str
) -> dict[str, str]:
    """按当日 PIT 区间构造 `{ts_code: 一级行业名称}`（引擎行业上限的输入）。

    无区间或无一级名称 ⇒ "未知行业"（与引擎缺省桶一致，不伪造行业）。
    """
    day8 = str(day)
    frame = pit.copy()
    frame["in_date"] = frame["in_date"].astype(str)
    # 开放区间（空 out_date）必须与 `universe.py` 同口径规范为哨兵，否则
    # `NaN > day` 恒 False，会让多数股票与已持仓从映射里消失（实测 524 vs 4,729）。
    frame["out_date"] = frame["out_date"].fillna("99999999").astype(str)
    active = frame[(frame["in_date"] <= day8) & (frame["out_date"] > day8)]
    mapping: dict[str, str] = {}
    for code, l1 in zip(active["ts_code"].astype(str), active["l1_code"].astype(str)):
        name = l1_names.get(l1)
        mapping[code] = name if name else "未知行业"
    return mapping


class PITIndustrySleeve(OperatingBaselineEngine):
    """研究侧子类：每个调仓日把 `industry_mapping` 换成**当日** PIT 行业表。

    引擎在构造期用静态表一次构建 `industry_mapping`（`industry_reader()` 返回当日快照），
    历史回放会用到今天的分类；`_filter_signal_candidates` 是模块级函数、直接读
    `self.industry_mapping`，因此唯一不改受审面的注入点 = 在 `run_step` 之前刷新该属性。
    """

    pit_frame: pd.DataFrame
    pit_names: dict[str, str]
    pit_refresh_log: list[str]

    def run_step(self, date: Any, trading_dates: Any, price_data: Any) -> Any:
        signal_days = getattr(self, "_signal_dates", None)
        is_signal_day = signal_days is None or date in signal_days
        if is_signal_day:
            day = pd.Timestamp(date).strftime("%Y%m%d")
            mapping = industry_mapping_for_day(self.pit_frame, self.pit_names, day)
            if mapping:
                frozen = getattr(self, "held_industry", {})
                held = {str(code) for code in getattr(self, "positions", {})}
                # 已持仓股票的行业以**买入日**为准，且优先级高于当日映射：
                # 否则区间到期/行业重分类会让它在引擎里换个桶，而审计按买入日仍记原桶，
                # 两边口径不一致就会放行超额买入（实测单桶 6 只、曾达 19 只）。
                for code in held:
                    if code in frozen:
                        mapping[code] = frozen[code]
                    elif code in mapping:
                        frozen[code] = mapping[code]
                # 已退出持仓不再保留身份，避免同名重入时沿用上一次的旧行业
                frozen = {key: value for key, value in frozen.items() if key in held}
                self.held_industry = frozen
                self.industry_mapping = mapping
                self._today_mapping = mapping
                self.pit_refresh_log.append(day)
        return super().run_step(date, trading_dates, price_data)

    def _post_filter_candidates(self, candidates: Any, date: Any) -> Any:
        """只允许当日有可核一级行业的候选进入买入池。

        引擎的行业上限对"未知行业"桶不构成约束（实测单桶曾堆到 19 只），
        因此把身份未知者挡在候选之外；该判断与 F01 的域规则一致。
        """
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
            # 硬上限：本批已选也立即计入，避免同批堆到第 6 只
            counts[bucket] = counts.get(bucket, 0) + 1
            kept.append((code, score))
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + len(kept)
        return kept


def b_overrides() -> dict[str, Any]:
    """B 规格相对 A 默认值的覆盖项（唯一来源 spec.PORTFOLIO）。"""
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
        # 用户裁决偏离（见 DEVIATIONS）：关闭跨日重试，避免执行时行业溢出
        "max_retry_count": 0 if retry_disabled() else int(portfolio["buy_retries_per_day"]),
        "max_retry_days": 0 if retry_disabled() else int(portfolio["buy_retry_days"]),
        # 用户裁决偏离（见 DEVIATIONS）：关闭补齐窗口，其余重试语义保留
        "enable_position_completion": not completion_disabled(),
        "completion_window_days": int(portfolio["fill_window_days"]),
        # 协议：关闭空仓提前调仓与 trailing 提前调仓
        "enable_early_rebalance_on_empty": False,
        "enable_trailing_early_rebalance": False,
    }


def build_kwargs(
    root: Path,
    ranked: dict[str, list[tuple[str, float]]],
    capital: float,
    corporate_book: NativeCorporateBook,
) -> dict[str, Any]:
    """A 默认 kwargs + B 覆盖 + 公司行为桥。"""
    kwargs = engine_kwargs(Path(root), ranked, int(capital), False)
    kwargs.update(b_overrides())
    kwargs["corporate_book"] = corporate_book
    if str(kwargs.get("position_sizing")) != "equal":
        raise BNativeError("position_sizing 覆盖失败")
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
) -> dict[str, Any]:
    """逐日驱动原生引擎（行业映射按当日 PIT 刷新），返回产物与独立守恒校验。"""
    dates = [pd.Timestamp(str(day)) for day in calendar]
    engine = PITIndustrySleeve(**build_kwargs(root, ranked, capital, corporate_book))
    engine.pit_frame = pit
    engine.pit_names = l1_names
    engine.pit_refresh_log = []
    # F03-d：逐日持仓明细（构造期标志，非 kwargs）
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
    result["engine_params"] = {
        name: getattr(engine, name, None)
        for name in (
            "rebalance_freq",
            "holding_period",
            "stagger_tranches",
            "position_sizing",
            "max_weight_per_stock",
            "max_per_industry",
            "max_retry_count",
            "max_retry_days",
            "enable_position_completion",
            "completion_window_days",
            "sell_timing",
        )
    }
    return result

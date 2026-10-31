"""PaperHost 日级排期与锚定投影（live 持久化锚 / replay 窗口锚）。

从 ``host.py`` 提取以遵守文件行数硬上限；公开入口仍由 PaperHost 提供，
live 与 replay 的语句顺序和数值语义不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.core.execution.live_schedule import apply_live_policy, init_live_anchor
from src.lazybull.v2.hosts.paper.config_adapter import PaperHostError
from src.lazybull.v2.hosts.paper.state_store import build_rebalance_state_projection


def validate_schedule(
    schedule_mode: str,
    replay_calendar: Optional[Sequence[pd.Timestamp]],
    calendar_strs: Sequence[str],
) -> Tuple[str, List[pd.Timestamp]]:
    """校验排期模式及回放日历，返回可保存到宿主的规范值。"""
    if schedule_mode not in {"live", "replay"}:
        raise PaperHostError(f"未知纸面排期模式: {schedule_mode!r}")
    window = list(replay_calendar or [])
    if schedule_mode == "replay":
        replay_strs = [day.strftime("%Y%m%d") for day in window]
        if not replay_strs or replay_strs != sorted(replay_strs):
            raise PaperHostError("回放窗口交易日历必须非空且升序")
        if len(set(replay_strs)) != len(replay_strs) or not set(replay_strs) <= set(calendar_strs):
            raise PaperHostError("回放窗口交易日历必须无重复且为完整交易日历子集")
    return schedule_mode, window


def configure_engine_schedule(engine: Any, schedule_mode: str) -> None:
    """live 复刻仅空仓提前调仓；replay 对齐回测拖尾分支与完整快照。"""
    engine.enable_trailing_early_rebalance = schedule_mode == "replay"
    engine.record_holdings_snapshot = schedule_mode == "replay"


def initialize_anchor(host: Any, engine: Any, corrected_ts: pd.Timestamp) -> None:
    """无快照首次启动时按宿主形态初始化锚；恢复路径不得调用本函数。"""
    if host.schedule_mode == "replay":
        engine._signal_dates = engine._get_rebalance_dates(host.replay_calendar)
        engine._cycle_anchor_idx = 0
        logger.info("纸面回放首次启动：窗口锚定初始化（禁止 live 持久化锚）")
        return
    init_live_anchor(engine, corrected_ts, host.trading_calendar)
    logger.info(f"纸面宿主首次启动：live 锚定初始化（锚 = {corrected_ts.date()}）")


def step_calendar(host: Any) -> List[pd.Timestamp]:
    """取得日级执行日历；回放以窗口首日为原点，保留窗口结束后的已知交易日。"""
    if host.schedule_mode == "replay":
        return [day for day in host.trading_calendar if day >= host.replay_calendar[0]]
    return host.trading_calendar


def prepare_daily_schedule(
    host: Any, engine: Any, corrected_ts: pd.Timestamp
) -> Tuple[Optional[Any], bool, List[pd.Timestamp]]:
    """返回 live 提案对象、当日是否调仓及 run_step 日历；replay 不执行 live 重锚。"""
    if host.schedule_mode == "replay":
        return None, corrected_ts in engine._signal_dates, step_calendar(host)
    day = apply_live_policy(engine, corrected_ts, host.trading_calendar)
    was_signal_day = corrected_ts in day.working
    engine._signal_dates = day.working
    return day, was_signal_day, host.trading_calendar


def wire_day_inputs(host: Any, engine: Any, price_data: pd.DataFrame) -> None:
    """每日重建 wiring 字段；这些输入不进入 EngineState。"""
    calendar = step_calendar(host)
    engine._trade_date_index = {day: index for index, day in enumerate(calendar)}
    engine._prepare_price_index(price_data)
    engine.price_data_cache = price_data
    engine._run_total_days = (
        len(host.replay_calendar)
        if host.schedule_mode == "replay"
        else len(engine.portfolio_values) + 1
    )


def project_rebalance_state(host: Any, engine: Any) -> Dict[str, Any]:
    """live 必须有持久化锚；replay 由窗口快照承载，无 live 锚。"""
    if host.schedule_mode == "replay":
        return {"schedule_mode": "replay"}
    if engine.live_anchor_state is None:
        raise PaperHostError("live 形态锚定状态缺失（引擎未 init 且快照未恢复），拒绝提交")
    return build_rebalance_state_projection(engine.live_anchor_state)

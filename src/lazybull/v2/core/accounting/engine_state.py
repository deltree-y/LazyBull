"""引擎状态值对象与导出/恢复接缝（P4-T1；D1/D2）。

``EngineState`` = 引擎运行时**演化状态**的完整快照（D2 双状态映射的引擎侧权威）；
``export_state`` / ``restore_state`` 是状态外置的唯一接缝——纸面宿主逐日驱动形态
= ``restore_state → run_step（当日）→ export_state → 落盘``（D1）。

生产契约（D2，spike 宽松恢复的反面）：
- **缺字段 / 多字段 / schema 版本不符 / 子对象键集不符 ⇒ fail-fast**
  （``EngineStateError``），禁止回退工厂初值；
- **export→restore→export 字节一致**（``dumps_blob`` 固定 pickle 协议、
  整 blob 单次往返——``pending_signals`` 与 ``_signal_dates`` 共享
  Timestamp 键对象，逐字段分别序列化会打断共享引用，T0.5b 实测复现）；
- schema 版本化（``ENGINE_STATE_SCHEMA_VERSION``；字段增删必须递增）。

字段清单产生方式（D2 硬约束）：**AST 枚举 + 人工复核**（枚举工具
``scripts/v2_p4/t1_ast_enum.py``，207 写点 / 90 属性），逐字段归类见
``FIELD_CLASSIFICATION``（evolving=入快照 / sub_object=子对象随快照 /
config=构造期注入 / wiring=输入接线每日重建 / daily_transient=日界暂态）。

人工复核注记（AST 按 ``self.``/``engine.`` 口径，两处别名写人工补注）：
- ``positions`` 演化含 ``last_known_price`` 估值缓存写回副作用
  （``accounting/nav.py`` 与 ``signal_execution._get_position_weight_for_planning``
  经参数别名写持仓 dict）——估值回退链语义（D10），非入账，随 positions 入快照；
- ``execution_attribution_records`` 的 ``record.update`` 与
  ``completion_stats`` 的计数自增在 buy_execution 模块级函数内经 ``engine``
  形参落点（AST 已覆盖）。

安全注记：``loads_blob`` 使用 pickle 反序列化——**不受信来源的 blob 禁止
直接 loads**（落盘投影的可信边界由 T2 state_store 承载；SMB 只读读者场景
不走本通道）。
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.lazybull.v2.core.accounting.ledger import LEDGER_STATE_KEYS, Ledger
from src.lazybull.v2.core.execution.live_schedule import LIVE_ANCHOR_KEYS, LiveAnchorState

#: 快照 schema 版本（字段增删必须递增）
ENGINE_STATE_SCHEMA_VERSION = 1

#: 快照固定 pickle 协议（字节一致性前提）
_SNAPSHOT_PICKLE_PROTOCOL = 5

#: 演化型引擎属性（入快照，17 项；与引擎属性名逐字一致）
SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "current_capital",
    "positions",
    "pending_signals",
    "pending_stop_loss_sells",
    "pending_condition_sells",
    "_cycle_anchor_idx",
    "portfolio_values",
    "trades",
    "execution_attribution_records",
    "holdings_snapshots",
    "unfilled_slots",
    "completion_stats",
    "_normalize_log_count",
    "_last_ranked_candidates",
    "_last_signal_date",
    "_last_rebalance_nav",
    "_signal_dates",
)

#: 子对象（随快照，内部演化字段见各子对象投影键集）
SNAPSHOT_SUB_OBJECTS: Tuple[str, ...] = (
    "pending_order_manager",
    "stop_loss_monitor",
    "ledger",
    "live_anchor_state",
)

_PENDING_ORDER_MANAGER_KEYS = ("pending_orders", "total_added", "total_expired", "total_succeeded")
_STOP_LOSS_MONITOR_KEYS = ("consecutive_limit_down_days",)

#: 字段归类注册表（AST 枚举 + 人工复核；T1 验收增项④：逐字段标注配置型/演化型）。
#: 键 = 引擎面属性名；值 = (类别, 复核注记)。类别闭集：
#: evolving / sub_object / config / wiring / daily_transient。
FIELD_CLASSIFICATION: Dict[str, Tuple[str, str]] = {
    # ── 演化型（入快照）──
    "current_capital": ("evolving", "现金；state.py:316 初始化；T1 起仅 Ledger.apply_fills 改写"),
    "positions": (
        "evolving",
        "持仓（shares/buy_date/signal_date/buy_trade_price/buy_pnl_price/buy_cost_cash）；"
        "T1 起现金/股数/成本仅 Ledger 改写；含 last_known_price 估值缓存写回（nav.py 别名，D10）",
    ),
    "pending_signals": ("evolving", "T0 计划队列（键=信号日 Timestamp）；入队/撤回/消费"),
    "pending_stop_loss_sells": ("evolving", "待止损卖出队列"),
    "pending_condition_sells": ("evolving", "待条件卖出队列（T0 触发 T+1 执行）"),
    "_cycle_anchor_idx": ("evolving", "当前调仓周期起点 idx（所在日历坐标系）"),
    "portfolio_values": ("evolving", "组合价值逐日历史（NAV 曲线源）"),
    "trades": ("evolving", "成交日志（账本审计记录；T1 起仅 Ledger 追加）"),
    "execution_attribution_records": ("evolving", "执行归因旁路记录（只读旁路语义不变）"),
    "holdings_snapshots": ("evolving", "逐日持仓快照旁路（record_holdings_snapshot 开关）"),
    "unfilled_slots": ("evolving", "未成交槽位（仓位补齐跟踪）"),
    "completion_stats": ("evolving", "补齐统计四计数器"),
    "_normalize_log_count": ("evolving", "归一化诊断日志计数（影响日志不影响数值）"),
    "_last_ranked_candidates": ("evolving", "最近调仓候选排序（止盈补位用）"),
    "_last_signal_date": ("evolving", "最近调仓日期"),
    "_last_rebalance_nav": ("evolving", "上次调仓日组合净值（止盈基准）"),
    "_signal_dates": (
        "evolving",
        "T1 新增：调仓排期表（原 run() 局部可变变量提升；_accept_early_rebalance "
        "运行期删条目；live 形态经 live_schedule 提案-提交回流）",
    ),
    # ── 子对象（随快照）──
    "pending_order_manager": (
        "sub_object",
        "延迟订单管理器：演化 = pending_orders/total_added/total_expired/total_succeeded；"
        "配置 = max_retry_count/max_retry_days/event_sink（构造注入，恢复侧构造器重接线）",
    ),
    "stop_loss_monitor": (
        "sub_object",
        "止损监控器：演化 = consecutive_limit_down_days；配置 = config（构造注入）",
    ),
    "ledger": ("sub_object", "T1 新增：账本（演化 = applied_keys/last_fill_date）"),
    "live_anchor_state": (
        "sub_object",
        "T1 新增：live 持久化锚定（F25 三字段 + 配置指纹；回放形态恒 None）",
    ),
    # ── 配置型（构造期注入，不入快照）──
    "universe": ("config", "构造注入（股票池引用）"),
    "signal": ("config", "构造注入（信号对象引用）"),
    "cost_model": ("config", "构造注入（成本模型）"),
    "initial_capital": ("config", "构造注入"),
    "rebalance_freq": ("config", "构造注入（live 形态变更走提案-提交，配置指纹在锚定状态内）"),
    "sell_timing": ("config", "构造注入"),
    "verbose": ("config", "构造注入"),
    "holding_period": ("config", "构造注入（None ⇒ = rebalance_freq，初始化期解析）"),
    "stagger_tranches": ("config", "构造注入（live 变更同 rebalance_freq）"),
    "enable_early_rebalance_on_empty": ("config", "构造注入"),
    "enable_risk_budget": ("config", "构造注入"),
    "vol_window": ("config", "构造注入"),
    "vol_epsilon": ("config", "构造注入"),
    "enable_pending_order": ("config", "构造注入"),
    "enable_position_completion": ("config", "构造注入"),
    "completion_window_days": ("config", "构造注入"),
    "stop_loss_config": ("config", "构造注入"),
    "max_weight_per_stock": ("config", "构造注入"),
    "max_per_industry": ("config", "构造注入"),
    "stock_basic": ("config", "构造注入"),
    "industry_mapping": ("config", "构造期构建（行业映射，由宿主 industry_reader/申万表派生）"),
    "position_sizing": ("config", "构造注入"),
    "kelly_vol_window": ("config", "构造注入"),
    "kelly_max_leverage": ("config", "构造注入"),
    "min_buy_value_ratio": ("config", "构造注入"),
    "record_holdings_snapshot": (
        "config",
        "host 构造期注入标志（不逐日演化；T0.5b P3-5 复核结论）",
    ),
    "enable_trailing_early_rebalance": (
        "config",
        "P4-T1 评审处置 P2-5：拖尾提前调仓开关（默认 True=引擎现行行为；live 复刻旧链仅空仓语义置 False）",
    ),
    "features_by_date": ("config", "BacktestEngineML 构造注入（特征数据引用）"),
    "portfolio_merger": ("config", "原生单袖合并器（仅 NativeMergeBacktestEngineML）"),
    "merger_sleeve_id": ("config", "原生合并袖 ID（仅 NativeMergeBacktestEngineML）"),
    # ── 输入接线（每日重建，不入快照）──
    "trade_price_index": ("wiring", "成交价格索引（不复权 close）；宿主注入 price_data 每日重建"),
    "pnl_price_index": ("wiring", "绩效价格索引（后复权 close_adj）；每日重建"),
    "trade_price_open_index": ("wiring", "开盘成交价格索引；每日重建"),
    "pnl_price_open_index": ("wiring", "开盘绩效价格索引；每日重建"),
    "price_data_cache": ("wiring", "价格数据缓存（交易状态检查用）；每日重建"),
    "_trade_date_index": ("wiring", "完整日历 idx 映射（D1 日历注入）；每日重建"),
    "_run_total_days": ("wiring", "T1 新增：驱动窗口日数（进度日志分母）；驱动上下文每日注入"),
    "_suspend_calendar": ("wiring", "停牌日历延迟缓存（_get_suspend_calendar 回写）"),
    "data_storage": ("wiring", "宿主注入的 SuspendDataReader（停牌日历回写保持同一对象）"),
    # ── 日界暂态（日界天然为空，不入快照）──
    "_deferred_day_logs": ("daily_transient", "日首重置 / 日终清空（run_step finally）"),
    "_daily_warning_items": ("daily_transient", "日首 _reset_daily_warning_items 重置"),
}


class EngineStateError(RuntimeError):
    """引擎状态导出/恢复错误（缺字段 / 多字段 / schema 版本不符 / 子对象形态非法）。"""


@dataclass
class EngineState:
    """引擎运行时演化状态值对象（schema 版本化；字段与引擎属性名逐字一致）。

    21 字段 = 17 演化型引擎属性 + 4 子对象投影（pending_order_manager /
    stop_loss_monitor / ledger / live_anchor_state）。
    """

    current_capital: float
    positions: Dict[str, Dict]
    pending_signals: Dict[pd.Timestamp, Dict]
    pending_stop_loss_sells: Dict[str, Dict]
    pending_condition_sells: Dict[str, Dict]
    _cycle_anchor_idx: int
    portfolio_values: List[Dict]
    trades: List[Dict]
    execution_attribution_records: List[Dict]
    holdings_snapshots: List[Dict]
    unfilled_slots: Dict[pd.Timestamp, Dict]
    completion_stats: Dict[str, int]
    _normalize_log_count: int
    _last_ranked_candidates: List
    _last_signal_date: Optional[pd.Timestamp]
    _last_rebalance_nav: Optional[float]
    _signal_dates: Dict[pd.Timestamp, int]
    pending_order_manager: Optional[Dict] = None
    stop_loss_monitor: Optional[Dict] = None
    ledger: Dict = field(default_factory=dict)
    live_anchor_state: Optional[Dict] = None


def _export_pending_order_manager(manager: Any) -> Optional[Dict]:
    """延迟订单管理器子对象投影（event_sink 不导出——恢复侧构造器重接线）。"""
    if manager is None:
        return None
    return {
        "pending_orders": manager.pending_orders,
        "total_added": manager.total_added,
        "total_expired": manager.total_expired,
        "total_succeeded": manager.total_succeeded,
    }


def _export_stop_loss_monitor(monitor: Any) -> Optional[Dict]:
    """止损监控器子对象投影（config 不导出——构造注入）。"""
    if monitor is None:
        return None
    return {"consecutive_limit_down_days": monitor.consecutive_limit_down_days}


def export_state(engine: Any) -> EngineState:
    """导出引擎运行时演化状态（整 blob 单次 pickle 往返切断与引擎的引用纠缠）。

    Returns:
        EngineState（已脱钩：后续引擎演化不污染快照）。
    """
    raw: Dict[str, Any] = {name: getattr(engine, name) for name in SNAPSHOT_FIELDS}
    raw["pending_order_manager"] = _export_pending_order_manager(engine.pending_order_manager)
    raw["stop_loss_monitor"] = _export_stop_loss_monitor(engine.stop_loss_monitor)
    raw["ledger"] = engine.ledger.export_state()
    raw["live_anchor_state"] = (
        engine.live_anchor_state.to_dict() if engine.live_anchor_state is not None else None
    )
    detached = pickle.loads(pickle.dumps(raw, protocol=_SNAPSHOT_PICKLE_PROTOCOL))
    return EngineState(**detached)


def state_to_blob(state: EngineState) -> Dict[str, Any]:
    """EngineState → 序列化 blob（固定字段顺序 + schema 版本）。"""
    fields: Dict[str, Any] = {}
    for name in SNAPSHOT_FIELDS:
        fields[name] = getattr(state, name)
    for name in SNAPSHOT_SUB_OBJECTS:
        fields[name] = getattr(state, name)
    return {"schema_version": ENGINE_STATE_SCHEMA_VERSION, "fields": fields}


def _validate_sub_object_keys(fields: Dict[str, Any]) -> None:
    """子对象投影键集 fail-fast 校验（缺键/多键即拒绝）。"""
    checks = (
        ("pending_order_manager", _PENDING_ORDER_MANAGER_KEYS),
        ("stop_loss_monitor", _STOP_LOSS_MONITOR_KEYS),
        ("ledger", LEDGER_STATE_KEYS),
        ("live_anchor_state", LIVE_ANCHOR_KEYS),
    )
    for name, expected in checks:
        value = fields[name]
        if value is None:
            continue
        if not isinstance(value, dict):
            raise EngineStateError(f"快照子对象 {name} 必须为 dict 或 None，实际 {type(value)}")
        if set(value.keys()) != set(expected):
            raise EngineStateError(
                f"快照子对象 {name} 键集不符: 期望 {sorted(expected)}，"
                f"实际 {sorted(value.keys())}"
            )


#: 演化型字段的最小类型表（state_from_blob 校验用；不改变正常路径）
_FIELD_TYPES: Dict[str, tuple] = {
    "current_capital": (float, int),
    "positions": (dict,),
    "pending_signals": (dict,),
    "pending_stop_loss_sells": (dict,),
    "pending_condition_sells": (dict,),
    "_cycle_anchor_idx": (int,),
    "portfolio_values": (list,),
    "trades": (list,),
    "execution_attribution_records": (list,),
    "holdings_snapshots": (list,),
    "unfilled_slots": (dict,),
    "completion_stats": (dict,),
    "_normalize_log_count": (int,),
    "_last_ranked_candidates": (list,),
    "_last_signal_date": (pd.Timestamp, type(None)),
    "_last_rebalance_nav": (float, int, type(None)),
    "_signal_dates": (dict,),
}


def _validate_field_types(fields: Dict[str, Any]) -> None:
    """演化型字段最小类型校验（fail-fast；bool 不是合法 int）。"""
    for name, types in _FIELD_TYPES.items():
        value = fields[name]
        if isinstance(value, bool) or not isinstance(value, types):
            raise EngineStateError(
                f"快照字段 {name} 类型非法: 期望 {tuple(t.__name__ for t in types)}，"
                f"实际 {type(value).__name__}"
            )


def state_from_blob(blob: Dict[str, Any]) -> EngineState:
    """blob → EngineState（**fail-fast**：版本/缺字段/多字段/字段类型/子对象键集不符即拒绝）。"""
    if not isinstance(blob, dict) or "schema_version" not in blob or "fields" not in blob:
        raise EngineStateError("引擎状态 blob 形态非法（缺 schema_version/fields 顶层键）")
    version = blob["schema_version"]
    if version != ENGINE_STATE_SCHEMA_VERSION:
        raise EngineStateError(
            f"引擎状态 schema 版本不符: 期望 {ENGINE_STATE_SCHEMA_VERSION}，实际 {version!r}"
        )
    fields = blob["fields"]
    if not isinstance(fields, dict):
        raise EngineStateError("引擎状态 blob 的 fields 必须为 dict")
    expected = set(SNAPSHOT_FIELDS) | set(SNAPSHOT_SUB_OBJECTS)
    actual = set(fields.keys())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise EngineStateError(
            f"引擎状态字段集不符: 缺 {sorted(missing) or '无'} / 多 {sorted(extra) or '无'}"
        )
    _validate_field_types(fields)
    _validate_sub_object_keys(fields)
    return EngineState(**fields)


def dumps_blob(blob: Dict[str, Any]) -> bytes:
    """blob → 字节（固定协议整 blob 单次序列化，字节一致性基准）。"""
    return pickle.dumps(blob, protocol=_SNAPSHOT_PICKLE_PROTOCOL)


def loads_blob(data: bytes) -> Dict[str, Any]:
    """字节 → blob（经 state_from_blob 校验前的原始反序列化）。"""
    blob = pickle.loads(data)
    if not isinstance(blob, dict):
        raise EngineStateError(f"引擎状态字节反序列化结果非 dict: {type(blob)}")
    return blob


def _check_sub_object_shape(
    sub_object: Any, saved: Optional[Dict], name: str, config_hint: str
) -> None:
    """子对象形态（None 与否）与引擎配置一致性校验（**不写任何状态**）。"""
    if (sub_object is None) != (saved is None):
        raise EngineStateError(
            f"{name} 形态不符: 引擎侧 {'None' if sub_object is None else '非 None'}，"
            f"快照侧 {'None' if saved is None else '非 None'}（{config_hint} 配置须一致）"
        )


def _validate_pending_order_manager_value(saved: Dict) -> None:
    """延迟订单管理器投影值校验（复审处置 r1-P2-B：同类问题覆盖到该子对象）。"""
    if not isinstance(saved["pending_orders"], dict):
        raise EngineStateError(
            f"pending_order_manager.pending_orders 必须为 dict，"
            f"实际 {type(saved['pending_orders']).__name__}"
        )
    for name in ("total_added", "total_expired", "total_succeeded"):
        value = saved[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise EngineStateError(
                f"pending_order_manager.{name} 必须为 int，实际 {type(value).__name__}"
            )


def _validate_stop_loss_monitor_value(saved: Dict) -> None:
    """止损监控器投影值校验（consecutive_limit_down_days 必须为 dict）。"""
    if not isinstance(saved["consecutive_limit_down_days"], dict):
        raise EngineStateError(
            f"stop_loss_monitor.consecutive_limit_down_days 必须为 dict，"
            f"实际 {type(saved['consecutive_limit_down_days']).__name__}"
        )


def _restore_pending_order_manager(manager: Any, saved: Optional[Dict]) -> None:
    """恢复延迟订单管理器内部状态（配置/事件回调保持构造器接线不动）。"""
    if manager is None:
        return
    manager.pending_orders = saved["pending_orders"]
    manager.total_added = saved["total_added"]
    manager.total_expired = saved["total_expired"]
    manager.total_succeeded = saved["total_succeeded"]


def _restore_stop_loss_monitor(monitor: Any, saved: Optional[Dict]) -> None:
    """恢复止损监控器内部状态（config 保持构造注入不动）。"""
    if monitor is None:
        return
    monitor.consecutive_limit_down_days = saved["consecutive_limit_down_days"]


def restore_state(engine: Any, state: EngineState) -> None:
    """把快照状态恢复到引擎实例（**fail-fast**；整 blob 单次往返保持共享引用）。

    **先全量校验、后统一写入**（R2-P4-T1-05 处置）：字段集/子对象键集/
    子对象形态与引擎配置一致性/锚定状态与账本投影的全部可预见校验都在
    任何写入之前完成——恢复被拒绝时目标引擎逐位不变（不产生混合状态）。

    接线边界：仅恢复演化状态与子对象内部状态；输入接线（价格索引 /
    price_data_cache / _trade_date_index / _run_total_days）与配置由宿主
    在 restore 前后按日重建/注入，不在本函数职责内。
    """
    blob = state_to_blob(state)
    _validate_sub_object_keys(blob["fields"])
    # 整 blob 单次往返：pending_signals 与 _signal_dates 的共享 Timestamp 键
    # 对象只在同一 blob 内保持（export→restore→export 字节一致前提）
    detached = pickle.loads(pickle.dumps(blob, protocol=_SNAPSHOT_PICKLE_PROTOCOL))["fields"]
    # ── 全量前置校验（任何拒绝都不改目标引擎）──
    _check_sub_object_shape(
        engine.pending_order_manager,
        detached["pending_order_manager"],
        "pending_order_manager",
        "enable_pending_order",
    )
    _check_sub_object_shape(
        engine.stop_loss_monitor,
        detached["stop_loss_monitor"],
        "stop_loss_monitor",
        "stop_loss_config",
    )
    # 子对象投影值校验（复审处置 r1-P2-B / R2-P4-T1-05 保留项：同类校验覆盖
    # 账本/延迟订单管理器/止损监控器，全部先于任何写入）
    Ledger.validate_state(detached["ledger"])
    if detached["pending_order_manager"] is not None:
        _validate_pending_order_manager_value(detached["pending_order_manager"])
    if detached["stop_loss_monitor"] is not None:
        _validate_stop_loss_monitor_value(detached["stop_loss_monitor"])
    # 锚定状态投影校验（缺键/多键/类型非法即拒绝；先构造后接线）
    live_anchor_raw = detached["live_anchor_state"]
    live_anchor = (
        LiveAnchorState.from_dict(live_anchor_raw) if live_anchor_raw is not None else None
    )
    # ── 统一写入 ──
    for name in SNAPSHOT_FIELDS:
        setattr(engine, name, detached[name])
    _restore_pending_order_manager(engine.pending_order_manager, detached["pending_order_manager"])
    _restore_stop_loss_monitor(engine.stop_loss_monitor, detached["stop_loss_monitor"])
    engine.ledger.restore_state(detached["ledger"])
    engine.live_anchor_state = live_anchor

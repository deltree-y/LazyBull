# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主状态库（PaperStateStore）。

v2 引擎运行时状态 ↔ 旧纸面 ``data/paper/`` 目录布局的适配层。**行为冻结第一**：只搬运字节与投影，不重算任何
决策数值；blob ⇄ EngineState 口径（``state_to_blob`` / ``state_from_blob`` / ``restore_state``）由调用方
（宿主）负责，不改 core ``engine_state.py`` 的任何接缝。

布局不变：七子目录（state/trades/nav/runs/pending_buys/pending_sells/instructions）与各文件 schema 和旧
``paper/storage`` 产出逐字段一致；唯一新增位 = 权威快照 ``state/engine_state.bin``（§15.3「布局不变」边界内
显式登记新增）。权威方向（D2/§15.3 冻结）：**引擎状态为权威、账本文件为投影**。

单写入者写锁：``<root>/state/.write.lock``，``os.O_CREAT|os.O_EXCL`` 原子创建，内容 = JSON
``{pid, host, acquired_at}``；锁已存在且 mtime 距今未超 ``lock_timeout_seconds`` ⇒ 抛 :class:`PaperLockError`
（含持有者信息）；超时 ⇒ stale：告警 + 删除重建（抢占）；活跃持锁者由后台心跳刷新 mtime，整日任务
不会仅因超时误被抢占。所有写路径都在锁内执行；同一实例仅**持锁线程**可重入（进程内 RLock +
线程所有权 + 深度计数），其他线程与跨进程写入者均 fail-fast。

同代原子提交点（D2；写序 = 权威先行，冻结决策）：:meth:`PaperStateStore.commit_day` 是唯一写入口。写序
（每步 = 写临时文件再 ``os.replace`` 原子替换）= ① 持锁 → ② 先写 ``state/engine_state.bin``（**原子提交点**）
→ ③ 投影（account / strategy_state / ranked_candidates（None 跳过）/ stop_loss_state（None 跳过）/
last_trade_date / rebalance_state / instructions（None 或空帧跳过）/ pending_buys / pending_sells）→
④ nav_row 非 None ⇒ nav.parquet 整文件重写式追加，trade_rows 非空 ⇒ trades.parquet 同形态 → ⑤
``runs/{run_kind}_{date}.json`` 幂等记录 → ⑥ 解锁。
崩溃分类与恢复：① 前崩溃 = 无副作用（可能残留 ``*.tmp`` 孤儿文件，读侧一律不读 tmp）；② 后任意步骤前
崩溃 = 权威新代 + 投影不完整（宿主启动时经「完整代核对重建」受控修复，见下）；⑤ 前崩溃还包括
幂等记录缺失（同日重跑或次日推进前均由宿主补写；原 run_kind/date 随快照顶层 ``paper_generation``
持久化，禁止按当前调用相位猜测，store 只提供原料）。

半提交识别与受控恢复：:meth:`PaperStateStore.check_consistency`（启动时调用）双读权威 blob 与 account.json
投影做**双状态一致性校验**（现金 + 持仓逐股逐字段比对，比对准绳 = 以权威重建的投影）。**滞后**（投影 ``last_update``
早于权威「最后处理日」，或投影缺失/损坏）⇒ **受控重投影修复**：以权威为准重写全部可派生投影 （与 ``repair_projections`` 同口径），记 warning，返回
``"repaired"``；修复幂等（二次校验 ``"ok"``，只应用 一次）。**冲突**（投影存在且日期不早于权威，但现金/持仓值与权威不一致）⇒
:class:`PaperStateError` fail-fast（禁止静默以任何一侧为准）。权威「最后处理日」= blob ``fields.portfolio_values``
最后一条的 ``date``（每个处理日都追加，无成交日不缺）。无权威有投影 = 旧链遗产布局（T5 迁移前形态）⇒ 告警返回
``"legacy_projection"``（不校验不修复；首个权威代由 T5 迁移建立）；两侧皆无 ⇒ ``"fresh"``。
:meth:`PaperStateStore.repair_projections` = 显式修复入口（供 T5 迁移/回滚演练），修复范围 = account.json /
last_trade_date.json / strategy_state.json / ranked_candidates.json（有候选时）/ rebalance_state.json
（live 锚定非 None 时）/ stop_loss_state.json（监控器非 None 时）；instructions 与 pending 两队为宿主映射
产物，trades/nav 当日行重建需成本模型与行情输入——三者连同 run 记录由宿主层 ``_ensure_generation_complete``
覆盖（登记）。

SMB 远端只读模式：``smb_reader`` 非 None ⇒ 不建本地目录，**一切写操作告警并跳过**（与旧 PaperStorage SMB 语义一致）；读面只暴露 4
个远端可读方法（``load_account`` / ``load_config`` / ``load_all_nav`` / ``load_rebalance_state``，经
smb_reader），其余本地路径读在远端模式下静默 miss（None/[]/False 自然空值）。

EngineState 落盘格式（勘误登记）：规划 §15.3 落盘位命名「engine_state.json」系勘误对象——blob 为
pickle 协议 5 **二进制**，文件名取 ``engine_state.bin``；**不受信来源的 blob 禁止直接 loads**（可信边界 =
本模块写入的本地文件；SMB 只读场景不走本通道）。

快照裁剪策略（⑤定稿，2026-10-28 T2 评审处置修订）：落盘前裁剪 blob ``fields["trades"]`` 与
``fields["holdings_snapshots"]`` 中**日期早于权威「最后处理日」**的历史条目（纯日志/旁路，恢复不需要；
持久真相走 trades.parquet 追加）；**当日（= 权威末日）条目保留在快照内**——这是半提交恢复重建当日
trades/nav 投影的持久原料（T2 评审 P1 修复：②③ 已写、④⑤ 未写即崩溃时，当日成交行须能从权威快照重建）。
**保留** ``portfolio_values``（体积小，其末日 =「最后处理日」，供滞后识别与同日幂等判定）。裁剪在本模块
``commit_day`` 内做（core 不改），直接操作 pickle dict（非 dict / 缺 ``fields`` 键 ⇒ PaperStateError）。
本裁剪经「裁剪恢复 vs 完整恢复次日行为逐位一致」黄金测试锁定。

完整代发布标记与崩溃恢复分工（2026-10-28 T2 评审处置登记）：**一代（某日）完整发布 ⇒
``runs/{run_kind}_{date}.json`` 存在**——⑤ 幂等记录是写序最后一步，存在即代表 ②~④ 全部落盘。
**读侧（含 SMB 远端只读/树莓派展示）规则：只把存在 runs 记录的代视为完整代**；缺失记录的代可能是半提交，
其 account/nav/trades 投影可能被宿主的受控修复重写，读侧不得据其做持久结论（SMB 读侧接入归 T5 切换时
落实，规则在此成文）。崩溃恢复分工：store 层 ``check_consistency``/``repair_projections`` 负责 account 族
状态投影的双状态校验与重投影；宿主层（``PaperHost._ensure_generation_complete``，每次驱动/读取前在整日
写锁内执行）负责 trades/nav/instructions/pending 两队/run 记录的当日核对与恢复。**已发布完整 trades/nav
行不因成本模型或 initial_capital 变化而回溯改写**；仅缺失或身份不符时按当前口径补建，配置漂移告警留痕。
原代发布标记在次日推进前同样必须补齐，恢复幂等、成交/NAV 不重复应用。

持久性边界登记：全部写 = 临时文件 + ``os.replace`` 原子替换（同文件系统内）；**未做 fsync**——原子性
（读侧只见整代文件）由 rename 语义保证，断电耐久性依赖文件系统/挂载选项，掉电窗口内最近一代可能需
宿主重启时经上述恢复路径重建（可恢复，非丢失）。

投影字段映射决策登记（冻结）：
- account.json：``cash`` ← current_capital；``buy_price`` ← buy_trade_price（不复权成交价）；``buy_cost`` ←
  ``buy_cost_cash − shares × buy_trade_price``（费用额口径，对齐旧 ``Position.buy_cost = commission +
  slippage`` 实盘语义；v2 加仓不重置 buy_trade_price，加仓后该差值 = 费用 + 价差近似，登记差异）；
  ``buy_pnl_price`` ← buy_pnl_price；``buy_date``（Timestamp ⇒ YYYYMMDD）；``buy_atr_pct`` 恒 0.0（v2 无此
  概念）；``status`` 恒 ``"持有"``、``notes`` 恒 ``""``（旧写盘器不落这两键，按任务冻结 schema 补齐——对旧
  读侧为兼容超集，登记差异）。
- strategy_state.json：只写 ``last_rebalance_nav``；旧文件其余键（prediction_quality_history /
  signal_tracking / rolling_quality_score / quality_warmup_remaining / last_take_profit_date）为退役信号
  质量跟踪键，不投影（登记差异）。
- rebalance_state.json：新 schema 五字段（last_rebalance_date / last_scheduled_rebalance_date /
  tranche_anchor_date / rebalance_freq / stagger_tranches；日期 YYYYMMDD 或 null）← LiveAnchorState
  （``anchor_date`` ⇒ ``tranche_anchor_date``）。
- 指令文件：12 列契约（旧 10 列 + replacement_slot_code/keep_buy_date）；读侧向后兼容旧 10 列形态（缺列补
  默认）。六语义：① 文件名 = T+1 执行日（``commit_day`` 的 ``instructions_date``）；② 去重键
  (action, ts_code)，与既有 T+1 文件合并时**既有优先**；③ 先卖后买（新建帧内卖出行先于买入行）；④ 补位
  上限 = 延迟买入订单 ``retry_count > max_replacement_attempts``（默认 5，对齐旧 MAX_REPLENISHMENT_ATTEMPTS
  与 PendingOrderManager 过期口径）不再生成补位指令；⑤ ``keep_buy_date`` 恒 False（暴露政策族默认关；读侧对
  旧文件 True 值原样保留）；⑥ 权重→股数按旧 ``_generate_instructions`` 口径（``预算 × 权重 →
  compute_lot_shares 整手 → 与现持仓差额再取整手``）。延迟订单映射（登记不确定点）：v2 延迟订单
  （PendingOrder）无槽位/权重概念 ⇒ 补位买入 ``target_weight`` 恒 0.0、``replacement_slot_code`` 恒 ``""``
  （槽位追溯由订单 (stock, action) 键承载）、``reason`` ← last_reason（缺省 "延迟订单"）；买入股数 =
  ``compute_lot_shares(target_value, price)``。
- nav.parquet 跳空语义保持：仅在有成交日由宿主供给 nav_row 追加；无成交日不写。

只读过渡依赖登记：``common.config.get_paper_root``（root_path=None 默认根解析；影子臂/测试一律显式传
隔离根）。指令投影纯函数族（含 ``common.constants.SHARE_LOT_SIZE`` /
``common.date_utils.calc_holding_trade_days`` / ``v2/core/decision/sizing.compute_lot_shares``
三依赖）已拆至同包 ``instruction_projection.py``（§7.4 + 复杂度门 1200 行硬上限，2026-10-28
T2 评审处置），本模块 import 重导出公开名（导入面不变）。
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger

from src.lazybull.common.config import get_paper_root
from src.lazybull.v2.core.accounting.engine_state import (
    EngineState,
    EngineStateError,
    loads_blob,
    state_from_blob,
)
from src.lazybull.v2.core.execution.live_schedule import LiveAnchorState, LiveScheduleError
from src.lazybull.v2.hosts.paper.instruction_projection import (
    DEFAULT_MAX_REPLACEMENT_ATTEMPTS,
    INSTRUCTION_COLUMNS,
    _normalize_instruction_frame,
    _yyyymmdd,
    build_instruction_frame,
)
from src.lazybull.v2.hosts.paper.write_lock import WRITE_LOCK_FILENAME, PaperWriteLockMixin

# 指令投影纯函数族自同包 instruction_projection.py import（拆分登记见该模块头；
# 此处 re-export 保持 ``state_store.build_instruction_frame`` 等公开名导入面不变）

if TYPE_CHECKING:
    from src.lazybull.common.smb_client import SMBFileReader

__all__ = [
    "PaperStateStore",
    "PaperLockError",
    "PaperStateError",
    "trim_engine_state_blob",
    "build_account_projection",
    "build_rebalance_state_projection",
    "build_ranked_candidates_projection",
    "build_stop_loss_projection",
    "build_strategy_state_projection",
    "build_instruction_frame",
    "INSTRUCTION_COLUMNS",
    "TRADES_COLUMNS",
    "NAV_COLUMNS",
    "DEFAULT_MAX_REPLACEMENT_ATTEMPTS",
]

#: 权威快照文件名（pickle 协议 5 二进制；勘误 §15.3 的 .json 命名）
ENGINE_STATE_FILENAME = "engine_state.bin"
#: 快照顶层宿主提交信息键（不属于 EngineState.fields；用于崩溃后恢复原 run_kind 发布标记）
GENERATION_INFO_KEY = "paper_generation"

ACCOUNT_FILENAME = "account.json"
STRATEGY_STATE_FILENAME = "strategy_state.json"
RANKED_CANDIDATES_FILENAME = "ranked_candidates.json"
STOP_LOSS_STATE_FILENAME = "stop_loss_state.json"
LAST_TRADE_DATE_FILENAME = "last_trade_date.json"
REBALANCE_STATE_FILENAME = "rebalance_state.json"
TRADES_PARQUET_FILENAME = "trades.parquet"
NAV_PARQUET_FILENAME = "nav.parquet"
PENDING_BUYS_FILENAME = "pending_buys.json"
PENDING_SELLS_FILENAME = "pending_sells.json"
CONFIG_FILENAME = "config.yaml"

#: nav.parquet 列序契约（5 列，与旧产出逐字一致）
NAV_COLUMNS: Tuple[str, ...] = ("trade_date", "cash", "position_value", "total_value", "nav")
#: trades.parquet 列序契约（11 列，与旧产出逐字一致）
TRADES_COLUMNS: Tuple[str, ...] = (
    "trade_date",
    "ts_code",
    "action",
    "shares",
    "price",
    "amount",
    "commission",
    "stamp_tax",
    "slippage",
    "total_cost",
    "reason",
)

#: 快照落盘前裁剪的纯日志/旁路字段（⑤定稿修订：仅裁日期早于权威末日的历史条目，当日条目保留作
#: 半提交恢复原料；portfolio_values 保留）
_BLOB_TRIMMED_FIELDS: Tuple[str, ...] = ("trades", "holdings_snapshots")
#: 快照固定 pickle 协议（与 engine_state._SNAPSHOT_PICKLE_PROTOCOL 一致；不 import 以免越权）
_SNAPSHOT_PICKLE_PROTOCOL = 5

#: commit_day ``expected_base_date`` 缺省哨兵（不传 = 跳过权威代比较交换校验；宿主驱动路径恒显式传入）
_COMMIT_BASE_UNSET: Any = object()


class PaperLockError(RuntimeError):
    """纸面宿主状态库写锁错误（锁被持有 / stale 清理后仍失败）。"""


class PaperStateError(RuntimeError):
    """纸面宿主状态库状态错误（blob 形态非法 / 双状态冲突 / 输入契约违反）。"""


def _validate_yyyymmdd(value: str, name: str) -> None:
    """YYYYMMDD 形态校验（fail-fast，含上下文）。"""
    if not (isinstance(value, str) and len(value) == 8 and value.isdigit()):
        raise PaperStateError(f"{name} 必须为 YYYYMMDD 字符串，实际: {value!r}")


def _entry_timestamp(entry: Any) -> Optional[pd.Timestamp]:
    """日志条目 date 字段 ⇒ pd.Timestamp（非 dict / 缺键 / 非法值 ⇒ None = 不裁，防御保留）。"""
    if not isinstance(entry, dict):
        return None
    value = entry.get("date")
    if value is None:
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None


def trim_engine_state_blob(engine_state_blob: bytes) -> bytes:
    """快照裁剪（落盘前）：``trades`` / ``holdings_snapshots`` 仅保留权威「最后处理日」当日条目。

    裁剪边界 = blob ``fields.portfolio_values`` 最后一条的 ``date``（权威「最后处理日」）：
    早于该日的历史条目裁掉（纯日志/旁路，恢复不需要），**当日及无日期条目保留**——当日成交是
    半提交恢复重建 trades/nav 投影的原料（T2 评审 P1 修复）。``portfolio_values`` 为空（无法判定
    边界）⇒ 不裁剪整段保留（防御：恢复原料优先于体积）。直接操作 pickle dict（裁剪是字节级投影
    决策，不经 EngineState 往返；``dumps_blob``/``loads_blob`` 口径由宿主负责）。**不受信 blob
    禁止经本函数 loads**（可信边界 = 宿主进程内产出）。

    Raises:
        PaperStateError: 反序列化失败 / 非 dict / 缺 ``fields`` 键或 ``fields`` 非 dict。
    """
    try:
        blob = pickle.loads(engine_state_blob)
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError) as exc:
        raise PaperStateError(
            f"引擎状态 blob 反序列化失败，拒绝裁剪落盘: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(blob, dict):
        raise PaperStateError(f"引擎状态 blob 形态非法（非 dict）: {type(blob).__name__}")
    fields = blob.get("fields")
    if not isinstance(fields, dict):
        raise PaperStateError("引擎状态 blob 形态非法（缺 fields 键或 fields 非 dict）")
    portfolio_values = fields.get("portfolio_values")
    keep_from: Optional[pd.Timestamp] = None
    if isinstance(portfolio_values, list) and portfolio_values:
        keep_from = _entry_timestamp(portfolio_values[-1])
    trimmed_fields = dict(fields)
    for name in _BLOB_TRIMMED_FIELDS:
        entries = fields.get(name)
        if not isinstance(entries, list):
            continue
        if keep_from is None:
            trimmed_fields[name] = list(entries)
            continue
        trimmed_fields[name] = [
            entry
            for entry in entries
            if (entry_date := _entry_timestamp(entry)) is None or entry_date >= keep_from
        ]
    trimmed = dict(blob)
    trimmed["fields"] = trimmed_fields
    return pickle.dumps(trimmed, protocol=_SNAPSHOT_PICKLE_PROTOCOL)


def _validated_state_from_bytes(raw: bytes) -> EngineState:
    """字节 → 校验后的 EngineState（blob 损坏/形态非法 ⇒ PaperStateError fail-fast）。"""
    try:
        blob = loads_blob(raw)
        return state_from_blob(blob)
    except (
        EngineStateError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
    ) as exc:
        raise PaperStateError(
            f"权威快照损坏或形态非法（禁止宽松恢复）: {type(exc).__name__}: {exc}"
        ) from exc


def _project_position(ts_code: str, position: Dict[str, Any]) -> Dict[str, Any]:
    """单个 v2 持仓 dict → 旧 account.json 持仓 schema（9 键，含 status/notes）。"""
    try:
        shares = int(position["shares"])
        buy_price = float(position["buy_trade_price"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperStateError(f"持仓投影缺必需字段（shares/buy_trade_price）: {ts_code}") from exc
    buy_cost_cash = float(position.get("buy_cost_cash") or 0.0)
    return {
        "ts_code": ts_code,
        "shares": shares,
        "buy_price": buy_price,
        # 费用额口径：旧 Position.buy_cost = commission + slippage（不含本金）
        "buy_cost": buy_cost_cash - shares * buy_price,
        "buy_date": _yyyymmdd(position.get("buy_date")),
        "buy_pnl_price": float(position.get("buy_pnl_price") or 0.0),
        "buy_atr_pct": 0.0,  # v2 无 ATR 概念，恒 0.0（冻结决策）
        "status": "持有",
        "notes": "",
    }


def build_account_projection(
    cash: float, positions: Dict[str, Dict[str, Any]], date: Any
) -> Dict[str, Any]:
    """v2 现金/持仓 → 旧 ``state/account.json`` 投影（纯函数）。

    positions 值含 v2 键 shares/buy_date/signal_date/buy_trade_price/buy_pnl_price/
    buy_cost_cash/last_known_price（signal_date/last_known_price 不投影）；
    ``date`` ⇒ ``last_update``（pd.Timestamp 或 YYYYMMDD）。
    """
    return {
        "cash": float(cash),
        "last_update": _yyyymmdd(date),
        "positions": {code: _project_position(code, pos) for code, pos in positions.items()},
    }


def build_rebalance_state_projection(anchor: LiveAnchorState) -> Dict[str, Any]:
    """LiveAnchorState 五字段 → 新 schema 五字段 rebalance_state.json（纯函数）。

    ``anchor_date`` ⇒ ``tranche_anchor_date``；日期 YYYYMMDD 或 None。
    """
    return {
        "last_rebalance_date": _yyyymmdd(anchor.last_rebalance_date) or None,
        "last_scheduled_rebalance_date": _yyyymmdd(anchor.last_scheduled_rebalance_date) or None,
        "tranche_anchor_date": _yyyymmdd(anchor.anchor_date) or None,
        "rebalance_freq": int(anchor.rebalance_freq),
        "stagger_tranches": int(anchor.stagger_tranches),
    }


def build_ranked_candidates_projection(
    signal_date: Any, ranked_candidates: List[Tuple[str, float]]
) -> Dict[str, Any]:
    """排序候选 → ``state/ranked_candidates.json`` 投影（稳定降序；纯函数）。"""
    ordered = sorted(ranked_candidates, key=lambda item: item[1], reverse=True)
    return {
        "signal_date": _yyyymmdd(signal_date),
        "candidates": [[str(code), float(score)] for code, score in ordered],
    }


def build_stop_loss_projection(consecutive_limit_down_days: Dict[str, int]) -> Dict[str, Any]:
    """止损监控器状态 → ``state/stop_loss_state.json`` 投影（纯函数）。"""
    return {"consecutive_limit_down_days": dict(consecutive_limit_down_days)}


def build_strategy_state_projection(last_rebalance_nav: Optional[float]) -> Dict[str, Any]:
    """策略状态投影（纯函数）：只写 ``last_rebalance_nav``（旧退役键不写，登记差异）。"""
    return {
        "last_rebalance_nav": (None if last_rebalance_nav is None else float(last_rebalance_nav))
    }


class PaperStateStore(PaperWriteLockMixin):
    """纸面宿主状态库（契约全量见模块 docstring）。

    Args:
        root_path: 数据根目录；None ⇒ ``get_paper_root()``（只读过渡依赖，登记）；
            显式传根 ⇒ 隔离根（影子臂/测试）
        smb_reader: 远端 SMB 读取器；非 None ⇒ 远端只读模式（不建本地目录，
            一切写操作告警并跳过）
        lock_timeout_seconds: 写锁 stale 判定超时（秒；mtime 口径）
        create_layout: False ⇒ 只读配置/巡检形态（不创建七子目录；写路径在
            ``acquire_write_lock`` 处 fail-fast）
    """

    _lock_error_type = PaperLockError
    _state_error_type = PaperStateError

    def __init__(
        self,
        root_path: Optional[str] = None,
        *,
        smb_reader: Optional["SMBFileReader"] = None,
        lock_timeout_seconds: float = 300.0,
        create_layout: bool = True,
    ) -> None:
        self._smb_reader = smb_reader
        self._is_remote = smb_reader is not None
        self.root_path = Path(root_path) if root_path else Path(get_paper_root())
        self.state_path = self.root_path / "state"
        self.trades_path = self.root_path / "trades"
        self.nav_path = self.root_path / "nav"
        self.runs_path = self.root_path / "runs"
        self.pending_buys_path = self.root_path / "pending_buys"
        self.pending_sells_path = self.root_path / "pending_sells"
        self.instructions_path = self.root_path / "instructions"
        self._init_write_lock(lock_timeout_seconds)
        # create_layout=False = 只读配置/巡检读取形态（不在盘上建子目录，T2 评审处置：
        # 「读配置」不得改变磁盘布局；既存目录的 mkdir no-op 边界随之消解）
        if not self._is_remote and create_layout:
            layout = (
                self.state_path,
                self.trades_path,
                self.nav_path,
                self.runs_path,
                self.pending_buys_path,
                self.pending_sells_path,
                self.instructions_path,
            )
            for path in layout:
                path.mkdir(parents=True, exist_ok=True)
        mode = "远端只读" if self._is_remote else "本地"
        logger.info(f"PaperStateStore 初始化完成（{mode}），根目录: {self.root_path}")

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)

    def _atomic_write_json(self, path: Path, payload: Any, *, indent: Optional[int] = 2) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=indent)
        self._atomic_write_bytes(path, text.encode("utf-8"))

    def _atomic_write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        tmp_path = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _append_parquet_rows(
        self, path: Path, rows: List[Dict[str, Any]], columns: Tuple[str, ...]
    ) -> None:
        """整文件重写式追加（读旧 + concat + 原子替换；列序按契约归一）。"""
        new_frame = pd.DataFrame(rows, columns=list(columns))
        if path.exists():
            existing = pd.read_parquet(path)
            if list(existing.columns) != list(columns):
                logger.warning(
                    f"{path.name} 既有列序与契约不一致，按契约列序归一: "
                    f"{list(existing.columns)} → {list(columns)}"
                )
                existing = existing.reindex(columns=list(columns))
            frame = pd.concat([existing, new_frame], ignore_index=True)
        else:
            frame = new_frame
        self._atomic_write_parquet(path, frame)

    def commit_day(
        self,
        *,
        date: Any,
        engine_state_blob: bytes,
        account: Dict[str, Any],
        strategy_state: Dict[str, Any],
        ranked_candidates: Optional[Dict[str, Any]],
        stop_loss_state: Optional[Dict[str, Any]],
        rebalance_state: Dict[str, Any],
        instructions: Optional[pd.DataFrame],
        instructions_date: Optional[Any],
        pending_buys: List[Dict[str, Any]],
        pending_sells: List[Dict[str, Any]],
        nav_row: Optional[Dict[str, Any]],
        trade_rows: List[Dict[str, Any]],
        run_kind: str,
        run_record: Dict[str, Any],
        expected_base_date: Any = _COMMIT_BASE_UNSET,
        trim_engine_state: bool = True,
    ) -> None:
        """同代原子提交（写序 = 权威先行；崩溃分类见模块 docstring）。

        ``engine_state_blob`` 为完整快照字节；本方法先按 ⑤ 定稿（修订版）裁剪
        （trades/holdings_snapshots 仅保留权威末日当日条目）再落盘 ``state/engine_state.bin``。

        ``expected_base_date`` = 本次驱动计算所基于的权威「最后处理日」（比较交换守卫，
        T2 评审 P1 修复）：传入时在锁内核对盘上当前权威末日——不一致 ⇒ 拒绝过期提交
        （``PaperStateError``，零写入）；缺省（哨兵）⇒ 跳过校验（仅供测试/工具直调；
        宿主驱动路径恒显式传入）。
        ``trim_engine_state`` 缺省 True 保持 live 裁剪语义；paper_replay 显式传 False，
        以完整快照作为 runs 成交、归因、持仓与 NAV 的唯一状态源。
        """
        if self._is_remote:
            logger.warning("远端只读模式，跳过 commit_day")
            return
        date_str = _yyyymmdd(date)
        _validate_yyyymmdd(date_str, "date")
        if not run_kind or "/" in run_kind or "\\" in run_kind:
            raise PaperStateError(f"run_kind 非法: {run_kind!r}")
        instructions_date_str = _yyyymmdd(instructions_date)
        if instructions is not None:
            _validate_yyyymmdd(instructions_date_str, "instructions_date")
        # 裁剪 = 纯计算，先于持锁（失败零副作用）；回放显式保留完整历史供 runs 派生。
        committed_blob = (
            trim_engine_state_blob(engine_state_blob) if trim_engine_state else engine_state_blob
        )
        with self.write_lock():
            if expected_base_date is not _COMMIT_BASE_UNSET:
                current_base = self.get_authority_last_date()
                if current_base != expected_base_date:
                    raise PaperStateError(
                        f"权威代比较交换失败（拒绝过期提交）：计算基于 {expected_base_date}，"
                        f"盘上权威末日已为 {current_base}——另一写入者已推进，"
                        "请重建宿主并重试（整日写锁内不应出现，出现即存在绕过宿主的写入者）"
                    )
            # ② 权威先行：engine_state.bin = 原子提交点
            self._atomic_write_bytes(self.state_path / ENGINE_STATE_FILENAME, committed_blob)
            # ③ 投影（账本文件为投影）
            self._commit_projections(
                date=date_str,
                account=account,
                strategy_state=strategy_state,
                ranked_candidates=ranked_candidates,
                stop_loss_state=stop_loss_state,
                rebalance_state=rebalance_state,
                instructions=instructions,
                instructions_date=instructions_date_str,
                pending_buys=pending_buys,
                pending_sells=pending_sells,
            )
            # ④ 追加式投影（nav 跳空语义保持：仅 nav_row 非 None 时追加）
            if nav_row is not None:
                self._append_parquet_rows(
                    self.nav_path / NAV_PARQUET_FILENAME, [nav_row], NAV_COLUMNS
                )
            if trade_rows:
                self._append_parquet_rows(
                    self.trades_path / TRADES_PARQUET_FILENAME, list(trade_rows), TRADES_COLUMNS
                )
            # ⑤ 幂等记录
            self._atomic_write_json(self.runs_path / f"{run_kind}_{date_str}.json", run_record)

    def _commit_projections(
        self,
        *,
        date: str,
        account: Dict[str, Any],
        strategy_state: Dict[str, Any],
        ranked_candidates: Optional[Dict[str, Any]],
        stop_loss_state: Optional[Dict[str, Any]],
        rebalance_state: Dict[str, Any],
        instructions: Optional[pd.DataFrame],
        instructions_date: str,
        pending_buys: List[Dict[str, Any]],
        pending_sells: List[Dict[str, Any]],
    ) -> None:
        """步骤 ③：状态投影族落盘（None 项跳过；指令空帧跳过——旧链不写空指令文件）。"""
        self._atomic_write_json(self.state_path / ACCOUNT_FILENAME, account)
        self._atomic_write_json(self.state_path / STRATEGY_STATE_FILENAME, strategy_state)
        if ranked_candidates is not None:
            self._atomic_write_json(self.state_path / RANKED_CANDIDATES_FILENAME, ranked_candidates)
        if stop_loss_state is not None:
            self._atomic_write_json(self.state_path / STOP_LOSS_STATE_FILENAME, stop_loss_state)
        self._atomic_write_json(
            self.state_path / LAST_TRADE_DATE_FILENAME, {"last_trade_date": date}, indent=None
        )
        self._atomic_write_json(self.runs_path / REBALANCE_STATE_FILENAME, rebalance_state)
        if instructions is not None and len(instructions) > 0:
            frame = _normalize_instruction_frame(instructions)
            self._atomic_write_parquet(
                self.instructions_path / f"{instructions_date}.parquet", frame
            )
        self._atomic_write_json(self.pending_buys_path / PENDING_BUYS_FILENAME, list(pending_buys))
        self._atomic_write_json(
            self.pending_sells_path / PENDING_SELLS_FILENAME, list(pending_sells)
        )

    def append_nav(self, rows: List[Dict[str, Any]]) -> None:
        """追加净值行（整文件重写式；宿主只在有成交日调用——跳空语义保持）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 append_nav")
            return
        if not rows:
            return
        with self.write_lock():
            self._append_parquet_rows(self.nav_path / NAV_PARQUET_FILENAME, list(rows), NAV_COLUMNS)

    def append_trades(self, rows: List[Dict[str, Any]]) -> None:
        """追加成交行（整文件重写式）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 append_trades")
            return
        if not rows:
            return
        with self.write_lock():
            self._append_parquet_rows(
                self.trades_path / TRADES_PARQUET_FILENAME, list(rows), TRADES_COLUMNS
            )

    @property
    def is_remote(self) -> bool:
        """远端只读模式标记（宿主构造期据此 fail-fast——远端仅供展示读取，不支持驱动）。"""
        return self._is_remote

    def _rewrite_day_parquet(
        self, path: Path, columns: Tuple[str, ...], date_str: str, rows: List[Dict[str, Any]]
    ) -> bool:
        """按日重写 parquet 当日段（先删当日全部行再写入新行；恰一次修复原语）。

        Returns:
            是否发生写盘（文件不存在且无行可写 ⇒ False）。
        """
        new_frame = pd.DataFrame(rows, columns=list(columns))
        if path.exists():
            existing = pd.read_parquet(path)
            if list(existing.columns) != list(columns):
                existing = existing.reindex(columns=list(columns))
            day_mask = existing["trade_date"].astype(str).str.replace("-", "") == date_str
            frame = pd.concat([existing.loc[~day_mask], new_frame], ignore_index=True)
        else:
            if not rows:
                return False
            frame = new_frame
        self._atomic_write_parquet(path, frame)
        return True

    def rewrite_day_trades(self, date: Any, rows: List[Dict[str, Any]]) -> bool:
        """重写 trades.parquet 当日行（半提交修复原语；调用方负责内容比对后决定是否调用）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 rewrite_day_trades")
            return False
        date_str = _yyyymmdd(date)
        _validate_yyyymmdd(date_str, "date")
        with self.write_lock():
            return self._rewrite_day_parquet(
                self.trades_path / TRADES_PARQUET_FILENAME, TRADES_COLUMNS, date_str, rows
            )

    def rewrite_day_nav(self, date: Any, nav_row: Optional[Dict[str, Any]]) -> bool:
        """重写 nav.parquet 当日行（``nav_row=None`` ⇒ 删除当日行；跳空语义下当日无行 = 合法）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 rewrite_day_nav")
            return False
        date_str = _yyyymmdd(date)
        _validate_yyyymmdd(date_str, "date")
        with self.write_lock():
            return self._rewrite_day_parquet(
                self.nav_path / NAV_PARQUET_FILENAME,
                NAV_COLUMNS,
                date_str,
                [nav_row] if nav_row is not None else [],
            )

    def write_pending_queues(
        self, pending_buys: List[Dict[str, Any]], pending_sells: List[Dict[str, Any]]
    ) -> List[str]:
        """重写 pending 两队投影（与现存内容一致 ⇒ 跳过；返回实际写入的相对路径列表）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 write_pending_queues")
            return []
        written: List[str] = []
        with self.write_lock():
            buys_path = self.pending_buys_path / PENDING_BUYS_FILENAME
            sells_path = self.pending_sells_path / PENDING_SELLS_FILENAME
            # 文件缺失 ⇒ 即使内容为空也写（存在性对齐正常提交形态——commit_day 恒写两队）
            if not buys_path.exists() or self.load_pending_buys() != list(pending_buys):
                self._atomic_write_json(buys_path, list(pending_buys))
                written.append(f"pending_buys/{PENDING_BUYS_FILENAME}")
            if not sells_path.exists() or self.load_pending_sells() != list(pending_sells):
                self._atomic_write_json(sells_path, list(pending_sells))
                written.append(f"pending_sells/{PENDING_SELLS_FILENAME}")
        return written

    def write_instructions_frame(self, execution_date: Any, frame: Optional[pd.DataFrame]) -> bool:
        """写 T+1 指令帧（规范化 12 列；空帧/None 跳过——旧链不写空指令文件）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 write_instructions_frame")
            return False
        date_str = _yyyymmdd(execution_date)
        _validate_yyyymmdd(date_str, "execution_date")
        if frame is None or len(frame) == 0:
            return False
        with self.write_lock():
            self._atomic_write_parquet(
                self.instructions_path / f"{date_str}.parquet", _normalize_instruction_frame(frame)
            )
        return True

    def write_run_record(self, run_kind: str, date: Any, run_record: Dict[str, Any]) -> None:
        """写幂等记录 ``runs/{run_kind}_{date}.json``（完整代发布标记；原子替换）。"""
        if self._is_remote:
            logger.warning("远端只读模式，跳过 write_run_record")
            return
        date_str = _yyyymmdd(date)
        _validate_yyyymmdd(date_str, "date")
        if not run_kind or "/" in run_kind or "\\" in run_kind:
            raise PaperStateError(f"run_kind 非法: {run_kind!r}")
        with self.write_lock():
            self._atomic_write_json(self.runs_path / f"{run_kind}_{date_str}.json", run_record)

    def load_engine_state_blob(self) -> Optional[bytes]:
        """读取权威快照原始字节（远端只读模式不暴露本通道 ⇒ None）。"""
        path = self.state_path / ENGINE_STATE_FILENAME
        if self._is_remote or not path.exists():
            return None
        return path.read_bytes()

    def load_engine_state(self) -> Optional[EngineState]:
        """读取并校验权威快照（损坏/形态非法 ⇒ PaperStateError fail-fast）。"""
        raw = self.load_engine_state_blob()
        if raw is None:
            return None
        return _validated_state_from_bytes(raw)

    def get_authority_last_date(self) -> Optional[str]:
        """权威「最后处理日」（blob portfolio_values 最后一条 date；无权威/空 ⇒ None）。"""
        state = self.load_engine_state()
        if state is None or not state.portfolio_values:
            return None
        return _yyyymmdd(state.portfolio_values[-1].get("date"))

    def _remote_call(self, method: str, relative: str, label: str) -> Optional[Any]:
        """SMB 远端调用统一入口（失败告警 + None；与旧 PaperStorage SMB 语义一致）。"""
        reader = self._smb_reader
        if reader is None:
            return None
        try:
            return getattr(reader, method)(relative)
        except Exception as exc:
            logger.warning(f"SMB 远端读取{label}失败（{relative}）: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _read_local_json(path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_account(self) -> Optional[Dict[str, Any]]:
        """读取账户投影（JSON 原始 dict；模型对象重建归宿主；损坏 JSON ⇒ PaperStateError）。"""
        if self._is_remote:
            data = self._remote_call("read_json", "state/account.json", "账户状态")
            return data if data else None
        path = self.state_path / ACCOUNT_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PaperStateError(
                f"account.json 读取失败（损坏/非法 JSON）: {path}（{exc}）"
            ) from exc
        if not isinstance(data, dict):
            raise PaperStateError(f"account.json 形态非法（非 dict）: {self.state_path}")
        return data

    def load_config(self) -> Optional[Dict[str, Any]]:
        """读取 config.yaml（**原始 YAML dict**，TradingConfig 规范化归宿主配置适配）。"""
        if self._is_remote:
            config = self._remote_call("read_yaml", CONFIG_FILENAME, "配置")
            return config if config else None
        path = self.root_path / CONFIG_FILENAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def load_all_nav(self) -> Optional[pd.DataFrame]:
        """读取全部净值记录（远端走 smb_reader；本地缺文件 ⇒ None）。"""
        if self._is_remote:
            return self._remote_call("read_parquet", "nav/nav.parquet", "净值")
        path = self.nav_path / NAV_PARQUET_FILENAME
        return pd.read_parquet(path) if path.exists() else None

    def load_rebalance_state(self) -> Optional[Dict[str, Any]]:
        """读取调仓锚定状态（远端走 smb_reader；本地缺文件 ⇒ None；损坏 JSON ⇒ PaperStateError）。"""
        if self._is_remote:
            state = self._remote_call("read_json", "runs/rebalance_state.json", "调仓状态")
            return state if state else None
        path = self.runs_path / REBALANCE_STATE_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PaperStateError(
                f"rebalance_state.json 读取失败（损坏/非法 JSON）: {path}（{exc}）"
            ) from exc
        return data if isinstance(data, dict) else None

    def load_strategy_state(self) -> Dict[str, Any]:
        """读取策略状态（缺文件/远端 miss ⇒ {}，与旧读侧默认一致）。"""
        if self._is_remote:
            return {}
        data = self._read_local_json(self.state_path / STRATEGY_STATE_FILENAME)
        return data if isinstance(data, dict) else {}

    def load_ranked_candidates(self) -> Optional[Dict[str, Any]]:
        """读取排序候选投影（``{"signal_date", "candidates"}``；远端 miss ⇒ None）。"""
        if self._is_remote:
            return None
        data = self._read_local_json(self.state_path / RANKED_CANDIDATES_FILENAME)
        return data if isinstance(data, dict) else None

    def load_stop_loss_state(self) -> Optional[Dict[str, Any]]:
        """读取止损状态（缺文件/远端 miss ⇒ None；止损未启用时本文件本就不写）。"""
        if self._is_remote:
            return None
        data = self._read_local_json(self.state_path / STOP_LOSS_STATE_FILENAME)
        return data if isinstance(data, dict) else None

    def load_last_trade_date(self) -> Optional[str]:
        """读取最近执行交易日期（缺文件/远端 miss ⇒ None）。"""
        if self._is_remote:
            return None
        data = self._read_local_json(self.state_path / LAST_TRADE_DATE_FILENAME)
        return data.get("last_trade_date") if isinstance(data, dict) else None

    def load_all_trades(self) -> Optional[pd.DataFrame]:
        """读取全部成交记录（远端 miss ⇒ None）。"""
        if self._is_remote:
            return None
        path = self.trades_path / TRADES_PARQUET_FILENAME
        return pd.read_parquet(path) if path.exists() else None

    def load_instructions_frame(self, execution_date: Any) -> Optional[pd.DataFrame]:
        """读取 T+1 指令帧（规范化为 12 列；兼容旧 10 列形态；远端 miss ⇒ None）。"""
        if self._is_remote:
            return None
        path = self.instructions_path / f"{_yyyymmdd(execution_date)}.parquet"
        return _normalize_instruction_frame(pd.read_parquet(path)) if path.exists() else None

    def load_pending_buys(self) -> List[Dict[str, Any]]:
        """读取延迟买入队列（缺文件/远端 miss ⇒ []）。"""
        if self._is_remote:
            return []
        data = self._read_local_json(self.pending_buys_path / PENDING_BUYS_FILENAME)
        return data if isinstance(data, list) else []

    def load_pending_sells(self) -> List[Dict[str, Any]]:
        """读取延迟卖出队列（缺文件/远端 miss ⇒ []）。"""
        if self._is_remote:
            return []
        data = self._read_local_json(self.pending_sells_path / PENDING_SELLS_FILENAME)
        return data if isinstance(data, list) else []

    def check_run_exists(self, run_kind: str, date: Any) -> bool:
        """幂等检查：``runs/{run_kind}_{date}.json`` 存在性（远端 miss ⇒ False）。"""
        if self._is_remote:
            return False
        return (self.runs_path / f"{run_kind}_{_yyyymmdd(date)}.json").exists()

    def load_run_record(self, run_kind: str, date: Any) -> Optional[Dict[str, Any]]:
        """读取幂等记录（远端 miss ⇒ None）。"""
        if self._is_remote:
            return None
        data = self._read_local_json(self.runs_path / f"{run_kind}_{_yyyymmdd(date)}.json")
        return data if isinstance(data, dict) else None

    @staticmethod
    def _authority_date_of(state: EngineState) -> Optional[str]:
        """权威「最后处理日」= portfolio_values 最后一条 date（每处理日都追加）。"""
        if not state.portfolio_values:
            return None
        return _yyyymmdd(state.portfolio_values[-1].get("date"))

    def _try_load_account_projection(self) -> Optional[Dict[str, Any]]:
        """account.json 容忍式读取（缺失/损坏/形态非法 ⇒ None = 滞后形态候选）。"""
        path = self.state_path / ACCOUNT_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"account.json 读取失败（按滞后形态处理）: {path}（{exc}）")
            return None
        if not isinstance(data, dict):
            logger.warning(f"account.json 形态非法（非 dict，按滞后形态处理）: {path}")
            return None
        return data

    def check_consistency(self, *, auto_repair: bool = True) -> str:
        """启动双状态一致性校验（语义全量见模块 docstring）。

        Returns:
            ``"remote_skipped"`` / ``"fresh"`` / ``"legacy_projection"`` / ``"ok"`` /
            ``"repaired"`` / ``"lag_detected"``（auto_repair=False 时检出滞后但不修复）。

        Raises:
            PaperStateError: 冲突形态（投影日期晚于权威，或同代值不一致）fail-fast。
        """
        if self._is_remote:
            return "remote_skipped"
        raw = self.load_engine_state_blob()
        if raw is None:
            if (self.state_path / ACCOUNT_FILENAME).exists():
                logger.warning(
                    "存在 account.json 投影但无权威快照（旧链遗产布局），跳过双状态校验"
                    "（首个权威代由 T5 迁移建立）"
                )
                return "legacy_projection"
            return "fresh"
        state = _validated_state_from_bytes(raw)
        authority_date = self._authority_date_of(state)
        projection = self._try_load_account_projection()
        if projection is None:
            return self._handle_lag("投影缺失或损坏", auto_repair)
        projection_date = str(projection.get("last_update", "") or "")
        if authority_date and projection_date < authority_date:
            return self._handle_lag(
                f"投影 last_update={projection_date} 早于权威最后处理日 {authority_date}",
                auto_repair,
            )
        if authority_date and projection_date > authority_date:
            raise PaperStateError(
                f"account.json 投影 last_update={projection_date} 晚于权威最后处理日 "
                f"{authority_date}（冲突形态，禁止静默处置）"
            )
        expected = build_account_projection(state.current_capital, state.positions, projection_date)
        if projection != expected:
            raise PaperStateError(
                "account.json 投影与权威快照逐字段比对不一致（冲突形态，禁止静默处置）；"
                "请人工核查后以 repair_projections() 显式修复或回滚"
            )
        return "ok"

    def _handle_lag(self, reason: str, auto_repair: bool) -> str:
        """滞后形态处置：受控重投影修复（幂等，只应用一次）。"""
        if not auto_repair:
            logger.warning(f"识别到半提交滞后形态（{reason}），auto_repair=False 不修复")
            return "lag_detected"
        logger.warning(f"识别到半提交滞后形态（{reason}），以权威快照为准执行受控重投影修复")
        repaired = self.repair_projections()
        logger.warning(f"受控重投影修复完成: {repaired}")
        return "repaired"

    def repair_projections(self) -> List[str]:
        """显式修复入口（供 T5 迁移/回滚演练）：以权威快照重建全部可派生投影。

        Returns:
            实际重写的相对路径列表（证据登记）；远端只读模式告警并返回 []。

        Raises:
            PaperStateError: 无权威快照可据以修复。
        """
        if self._is_remote:
            logger.warning("远端只读模式，跳过 repair_projections")
            return []
        raw = self.load_engine_state_blob()
        if raw is None:
            raise PaperStateError("无权威快照，无法执行重投影修复")
        state = _validated_state_from_bytes(raw)
        authority_date = self._authority_date_of(state) or ""
        with self.write_lock():
            return self._rewrite_derivable_projections(state, authority_date)

    def _rewrite_derivable_projections(self, state: EngineState, authority_date: str) -> List[str]:
        """全量重投影（修复范围见模块 docstring；instructions/pending 队列不重建）。"""
        repaired: List[str] = []
        account = build_account_projection(state.current_capital, state.positions, authority_date)
        self._atomic_write_json(self.state_path / ACCOUNT_FILENAME, account)
        repaired.append(f"state/{ACCOUNT_FILENAME}")
        if authority_date:
            self._atomic_write_json(
                self.state_path / LAST_TRADE_DATE_FILENAME,
                {"last_trade_date": authority_date},
                indent=None,
            )
            repaired.append(f"state/{LAST_TRADE_DATE_FILENAME}")
        self._atomic_write_json(
            self.state_path / STRATEGY_STATE_FILENAME,
            build_strategy_state_projection(state._last_rebalance_nav),
        )
        repaired.append(f"state/{STRATEGY_STATE_FILENAME}")
        if state._last_ranked_candidates and state._last_signal_date is not None:
            self._atomic_write_json(
                self.state_path / RANKED_CANDIDATES_FILENAME,
                build_ranked_candidates_projection(
                    state._last_signal_date, state._last_ranked_candidates
                ),
            )
            repaired.append(f"state/{RANKED_CANDIDATES_FILENAME}")
        if state.live_anchor_state is not None:
            try:
                anchor = LiveAnchorState.from_dict(state.live_anchor_state)
            except LiveScheduleError as exc:
                raise PaperStateError(f"权威快照 live 锚定状态非法，无法重投影: {exc}") from exc
            self._atomic_write_json(
                self.runs_path / REBALANCE_STATE_FILENAME, build_rebalance_state_projection(anchor)
            )
            repaired.append(f"runs/{REBALANCE_STATE_FILENAME}")
        if state.stop_loss_monitor is not None:
            self._atomic_write_json(
                self.state_path / STOP_LOSS_STATE_FILENAME,
                build_stop_loss_projection(state.stop_loss_monitor["consecutive_limit_down_days"]),
            )
            repaired.append(f"state/{STOP_LOSS_STATE_FILENAME}")
        return repaired

    def sync_state_projections(self, state: EngineState, authority_date: str) -> List[str]:
        """状态投影族确定性比对补齐（③ 中段崩溃形态当日收敛；一致 ⇒ 零写盘）。

        覆盖范围与 ``_rewrite_derivable_projections`` 相同（account / last_trade_date /
        strategy_state / ranked_candidates（有候选时）/ rebalance_state（锚定非 None 时）/
        stop_loss_state（监控器非 None 时）），但仅在与盘上内容不一致或文件缺失/损坏时
        重写（``_ensure_generation_complete`` 每日常态调用，避免无谓写盘）。调用前置：
        ``check_consistency`` 已过（account 冲突形态已 fail-fast）——本方法只做确定性补齐。

        Returns:
            实际重写的相对路径列表；远端只读模式告警并返回 []。
        """
        if self._is_remote:
            logger.warning("远端只读模式，跳过 sync_state_projections")
            return []
        written: List[str] = []
        with self.write_lock():
            for path, payload, rel, indent in self._expected_state_projections(
                state, authority_date
            ):
                if self._json_file_matches(path, payload):
                    continue
                self._atomic_write_json(path, payload, indent=indent)
                written.append(rel)
        return written

    def _expected_state_projections(
        self, state: EngineState, authority_date: str
    ) -> List[Tuple[Path, Any, str, Optional[int]]]:
        """状态投影族期望值清单（路径 / 载荷 / 相对路径标签 / JSON indent）。"""
        items: List[Tuple[Path, Any, str, Optional[int]]] = [
            (
                self.state_path / ACCOUNT_FILENAME,
                build_account_projection(state.current_capital, state.positions, authority_date),
                f"state/{ACCOUNT_FILENAME}",
                2,
            ),
            (
                self.state_path / STRATEGY_STATE_FILENAME,
                build_strategy_state_projection(state._last_rebalance_nav),
                f"state/{STRATEGY_STATE_FILENAME}",
                2,
            ),
        ]
        if authority_date:
            items.append(
                (
                    self.state_path / LAST_TRADE_DATE_FILENAME,
                    {"last_trade_date": authority_date},
                    f"state/{LAST_TRADE_DATE_FILENAME}",
                    None,
                )
            )
        if state._last_ranked_candidates and state._last_signal_date is not None:
            items.append(
                (
                    self.state_path / RANKED_CANDIDATES_FILENAME,
                    build_ranked_candidates_projection(
                        state._last_signal_date, state._last_ranked_candidates
                    ),
                    f"state/{RANKED_CANDIDATES_FILENAME}",
                    2,
                )
            )
        if state.live_anchor_state is not None:
            try:
                anchor = LiveAnchorState.from_dict(state.live_anchor_state)
            except LiveScheduleError as exc:
                raise PaperStateError(f"权威快照 live 锚定状态非法，无法重投影: {exc}") from exc
            items.append(
                (
                    self.runs_path / REBALANCE_STATE_FILENAME,
                    build_rebalance_state_projection(anchor),
                    f"runs/{REBALANCE_STATE_FILENAME}",
                    2,
                )
            )
        if state.stop_loss_monitor is not None:
            items.append(
                (
                    self.state_path / STOP_LOSS_STATE_FILENAME,
                    build_stop_loss_projection(
                        state.stop_loss_monitor["consecutive_limit_down_days"]
                    ),
                    f"state/{STOP_LOSS_STATE_FILENAME}",
                    2,
                )
            )
        return items

    @staticmethod
    def _json_file_matches(path: Path, payload: Any) -> bool:
        """JSON 文件与期望载荷一致（缺失/损坏 ⇒ False = 待确定性重写）。"""
        if not path.exists():
            return False
        try:
            return bool(json.loads(path.read_text(encoding="utf-8")) == payload)
        except (OSError, json.JSONDecodeError):
            return False

"""物理账本唯一入账入口（P4-T1；D2 调用链冻结的核心落点）。

调用链（D2 冻结）：决策（无副作用，账户及嵌套对象不变）→ 指令 → 成交结果
（回测 = 撮合模拟 Fill；纸面 = 真实行情记账 Fill）→ **唯一入账
（``Ledger.apply_fills``）→ 新账户**。

F23 改道：``buy_execution._buy_stock_direct``（建仓）/ ``_add_to_position``
（加仓）/ ``sell_execution._sell_stock_direct``（卖出/部分卖出）是现状全部
直接改账户入口——本模块落地后，现金/持仓/成交日志的变更**只允许**经
``Ledger.apply_fills`` 执行（静态扫描测试锁定；估值缓存
``last_known_price`` 写回属 nav.py 估值回退链语义，非入账，不在本契约面内）。

幂等与顺序：
- 每笔入账携带调用方供给的**稳定幂等键**（引擎内部路径 =
  ``yyyymmdd|action|stock|seq``，seq = 入账前成交日志长度；纸面外部回填由
  宿主供给稳定身份）；重复键 ⇒ 拒绝（``LedgerError``）；
- 成交日早于已入账最后日期（乱序）⇒ 拒绝；
- 幂等键集合与最后入账日期随 ``EngineState`` 快照同代落盘/恢复——半提交代
  重启后重放同日成交，键身份一致且不会重复入账（快照 = 提交点）。

数值纪律（行为冻结）：执行器（buy/sell execution）**原位完成全部数值计算**
（股数/金额/成本/剩余持仓），本模块只做形状校验 + 按给定数值落账 +
 追加成交日志——不引入任何新数值操作，不改变语句的可观察效果顺序。
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

#: 账本快照子对象的封闭键集（fail-fast 校验用）
LEDGER_STATE_KEYS: Tuple[str, ...] = ("applied_keys", "last_fill_date")


class LedgerError(RuntimeError):
    """入账错误（重复入账 / 乱序入账 / 入账指令形状或前置状态非法）。"""


def make_fill_key(date: pd.Timestamp, action: str, stock: str, seq: int) -> str:
    """引擎内部路径的幂等键格式单点（``yyyymmdd|action|stock|seq``）。

    ``seq`` = 入账前成交日志长度（``len(engine.trades)``）——同流程序列下
    确定且单调递增；纸面外部回填的稳定身份由宿主另行供给（T2）。
    """
    return f"{date.strftime('%Y%m%d')}|{action}|{stock}|{seq}"


@dataclass(frozen=True)
class BookingEntry:
    """入账指令（执行器已按冻结口径完成全部数值计算的成交结果）。

    持仓变更三形态（互斥，恰好其一）：
    - 建仓：``new_position`` = 完整持仓 dict（含子类扩展字段已合并）；
    - 加仓：``add_shares`` > 0 且 ``add_cost_cash`` 为成本增量（买入日/持有期不变）；
    - 卖出：``remaining_shares`` 非 None（> 0 = 部分卖出后的剩余股数与
      缩减后成本；= 0 = 整仓删除）。

    Attributes:
        fill_key: 幂等键（稳定身份，重复提交必拒绝）
        date: 成交日期
        stock: 股票代码
        action: "buy" / "sell"
        cash_delta: 现金变化量（买入为负 = -(成交额+手续费)，卖出为正 = 到手金额）
        trade_record: 追加到成交日志（trades）的完整记录（字段与冻结口径逐字一致）
    """

    fill_key: str
    date: pd.Timestamp
    stock: str
    action: str
    cash_delta: float
    trade_record: Dict[str, Any]
    new_position: Optional[Dict[str, Any]] = None
    add_shares: int = 0
    add_cost_cash: float = 0.0
    remaining_shares: Optional[int] = None
    remaining_cost_cash: float = 0.0


def _validate_entry_shape(entry: BookingEntry) -> None:
    """入账指令形状校验：三形态恰好其一（fail-fast，含上下文）。"""
    forms = sum(
        [
            entry.new_position is not None,
            entry.add_shares > 0,
            entry.remaining_shares is not None,
        ]
    )
    if forms != 1:
        raise LedgerError(
            f"入账指令形态非法（须恰好一种持仓变更形态，实际 {forms} 种）: "
            f"key={entry.fill_key} date={entry.date} stock={entry.stock} action={entry.action}"
        )
    if entry.action not in ("buy", "sell"):
        raise LedgerError(
            f"入账指令 action 非法（须为 buy/sell）: {entry.action!r} key={entry.fill_key}"
        )


class Ledger:
    """物理账本（单袖：引擎的 ``current_capital`` / ``positions`` / ``trades``）。

    唯一入账入口：``apply_fills``。状态（幂等键集合 + 最后入账日期）随
    ``EngineState`` 快照同代导出/恢复。协议层 ``Ledger``（common/protocols）
    的物理+虚拟账本形态归 P6.5；本类是 P4 单袖物理账户边界的入账实现。
    """

    def __init__(self) -> None:
        # 保序 dict（insertion order 保证 pickle 字节确定性）；值恒 None
        self._applied_keys: Dict[str, None] = {}
        self._last_fill_date: Optional[pd.Timestamp] = None

    def apply_fills(self, engine: Any, fills: Sequence[BookingEntry]) -> None:
        """应用成交到物理账本（唯一入账点）。

        两阶段形态：先做**与状态无关的全量预检**（批量内幂等键互斥 +
        逐笔幂等/乱序/形状校验——预检失败零副作用），再逐笔应用（持仓前置
        校验随应用逐笔进行——批量内相互依赖语义 = 顺序应用；纸面宿主批量
        提交外部回填 fills 时须注意：状态前置校验失败时前序笔已入账，
        宿主应在 apply 前持有当日快照以便受控重放，见 P4 规划 §17.7 T2 输入）。

        Args:
            engine: 引擎（账户 = current_capital / positions / trades 的载体）
            fills: 成交入账指令序列（执行器已完成全部数值计算）

        Raises:
            LedgerError: 重复入账（幂等键命中）/ 乱序入账（成交日早于最后
                入账日）/ 指令形状非法 / 前置持仓状态非法。
        """
        entries = list(fills)
        self._validate_batch(entries)
        for entry in entries:
            self._apply_one(engine, entry)

    def _validate_batch(self, entries: List[BookingEntry]) -> None:
        """全量预检（与账户状态无关部分）：批量内键互斥 + 逐笔幂等 +
        **滚动水位**日期单调性 + 形状。

        滚动水位（复审处置 r1-P2-A / R2-P4-T1-R1-01）：预检以批前
        ``_last_fill_date`` 为起点逐笔推进——批内倒序（后笔日期早于前笔）
        在应用前拒绝，预检失败零副作用（账户/日志/幂等键/水位均不变）。
        """
        keys = [entry.fill_key for entry in entries]
        if len(set(keys)) != len(keys):
            raise LedgerError(f"批量入账指令含重复幂等键: {keys}")
        rolling_last_date = self._last_fill_date
        for entry in entries:
            self._check_idempotency(entry, rolling_last_date)
            _validate_entry_shape(entry)
            rolling_last_date = entry.date

    def _apply_one(self, engine: Any, entry: BookingEntry) -> None:
        """逐笔落账（账本取得账户与审计记录的独立所有权——深拷贝脱钩入参）。

        ``frozen=True`` 只限制 BookingEntry 属性重绑定、不冻结其中的可变
        dict；入账时对 ``new_position`` / ``trade_record`` 深拷贝，调用方
        保留的入账对象此后不能反向修改账户与成交日志（R2-P4-T1-04 处置）。
        """
        self._check_idempotency(entry, self._last_fill_date)  # 应用时水位逐笔防御
        self._apply_position_mutation(engine, entry)
        engine.current_capital += entry.cash_delta
        engine.trades.append(deepcopy(entry.trade_record))
        self._applied_keys[entry.fill_key] = None
        self._last_fill_date = entry.date

    def _check_idempotency(self, entry: BookingEntry, last_fill_date: Optional[pd.Timestamp]) -> None:
        """重复入账 / 乱序入账拒绝（幂等键 + 日期单调性，水位显式传入）。"""
        if entry.fill_key in self._applied_keys:
            raise LedgerError(
                f"重复入账拒绝: key={entry.fill_key} 已在账本（stock={entry.stock} "
                f"date={entry.date}）"
            )
        if last_fill_date is not None and entry.date < last_fill_date:
            raise LedgerError(
                f"乱序入账拒绝: key={entry.fill_key} 成交日 {entry.date} 早于最后入账日 "
                f"{last_fill_date}"
            )

    @staticmethod
    def _apply_position_mutation(engine: Any, entry: BookingEntry) -> None:
        """持仓变更落账（前置状态校验 fail-fast）。"""
        positions = engine.positions
        if entry.new_position is not None:
            if entry.stock in positions:
                raise LedgerError(
                    f"建仓入账前置非法: {entry.stock} 已有持仓（key={entry.fill_key}）"
                )
            positions[entry.stock] = deepcopy(entry.new_position)
            return
        if entry.add_shares > 0:
            if entry.stock not in positions:
                raise LedgerError(
                    f"加仓入账前置非法: {entry.stock} 无持仓（key={entry.fill_key}）"
                )
            position = positions[entry.stock]
            position["shares"] = int(position["shares"]) + int(entry.add_shares)
            position["buy_cost_cash"] = (
                float(position.get("buy_cost_cash", 0.0)) + entry.add_cost_cash
            )
            return
        # 卖出形态（remaining_shares 非 None）
        if entry.stock not in positions:
            raise LedgerError(
                f"卖出入账前置非法: {entry.stock} 无持仓（key={entry.fill_key}）"
            )
        total_shares = positions[entry.stock]["shares"]
        if entry.remaining_shares < 0 or entry.remaining_shares > total_shares:
            raise LedgerError(
                f"卖出入账数量非法: {entry.stock} 剩余 {entry.remaining_shares} "
                f"超出 [0, {total_shares}]（key={entry.fill_key}）"
            )
        if entry.remaining_shares > 0:
            # 部分卖出：保留剩余持仓，每股成本基准不变，总现金支出按剩余比例缩减
            positions[entry.stock]["shares"] = entry.remaining_shares
            positions[entry.stock]["buy_cost_cash"] = entry.remaining_cost_cash
        else:
            del positions[entry.stock]

    def export_state(self) -> Dict[str, Any]:
        """导出账本状态（快照子对象投影；封闭键集）。"""
        return {
            "applied_keys": list(self._applied_keys),
            "last_fill_date": self._last_fill_date,
        }

    @staticmethod
    def validate_state(state: Dict[str, Any]) -> None:
        """账本快照投影校验（**纯校验、零写入**；restore 前置阶段调用）。

        复审处置（r1-P2-B / R2-P4-T1-05 保留项）：键集 + ``applied_keys``
        必须为 list 且元素为 str 且无重复 + ``last_fill_date`` 为
        pd.Timestamp 或 None——全部在任何写入之前完成。
        """
        keys = set(state.keys())
        expected = set(LEDGER_STATE_KEYS)
        if keys != expected:
            raise LedgerError(
                f"账本快照键集不符: 期望 {sorted(expected)}，实际 {sorted(keys)}"
            )
        applied = state["applied_keys"]
        if not isinstance(applied, list):
            raise LedgerError(f"账本快照 applied_keys 必须为 list，实际 {type(applied).__name__}")
        if any(not isinstance(key, str) for key in applied):
            raise LedgerError("账本快照 applied_keys 含非 str 元素")
        if len(set(applied)) != len(applied):
            raise LedgerError("账本快照 applied_keys 含重复键")
        last_fill_date = state["last_fill_date"]
        if last_fill_date is not None and not isinstance(last_fill_date, pd.Timestamp):
            raise LedgerError(
                f"账本快照 last_fill_date 类型非法: {type(last_fill_date).__name__}"
            )

    def restore_state(self, state: Dict[str, Any]) -> None:
        """恢复快照账本状态（**fail-fast**：先 validate_state 纯校验，后写入）。"""
        self.validate_state(state)
        self._applied_keys = {key: None for key in state["applied_keys"]}
        self._last_fill_date = state["last_fill_date"]

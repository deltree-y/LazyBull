# -*- coding: utf-8 -*-
"""核心层协议（契约 docs/contracts/protocols.md §3 的代码化）。

§3.1 DecisionMaker（R4-M6）/ §3.2 Executor（R4-H3 语义边界）/
§3.3 Ledger（R4-M7）/ §3.4 FundScheduler + PortfolioMerger（P6.5 起冻结）。

边界定位声明（R4-H3，写死；F8 勘误 3 修订钩子括注）：本模块协议是
**模块间公共契约**。内核内部管线——多源信号收集（每日有序钩子；钩子序列
以代码调用点枚举为准，P2a T5 以「黄金调用点清单」固化——括注不枚举，
旧文本中的止盈语义无在用对照物，F8 勘误 1）、指令按 (action, ts_code)
去重合并（既有指令优先）、延迟订单队列——**不在协议层表达**，
属 core/execution 内部结构。协议只暴露每日判定的输入输出边界。

延迟订单最小语义（跨模块可观测，故登记于此）：
- 重试上限（max_retry）与过期规则（超 rebalance_freq×0.5 丢弃）由内核统一维护；
- 特例（写死）：减仓 / 回补指令**不进延迟队列**（该队列按整仓卖出语义重试，
  会把减仓放大成清仓）——跳过并由次日每日判定重试。
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from src.lazybull.v2.common.types import (
    Account,
    ExecutionContext,
    Fill,
    Money,
    Order,
    TargetPortfolio,
    TradeDate,
    VirtualAccount,
)

__all__ = [
    "DecisionMaker",
    "Executor",
    "FundScheduler",
    "Ledger",
    "PortfolioMerger",
]


@runtime_checkable
class DecisionMaker(Protocol):
    """决策层（合并后的目标组合 → 指令）。

    R4-M6：输入为（合并后的）TargetPortfolio 而非各袖 RankedCandidates——
    信号 → 目标权重 → 合并 → 约束 → 指令 的链条由此闭合；
    单袖时代 merger = 直通（方案「已验证语义：单袖退化 = 净额化直通」的协议表达）。
    """

    def decide(
        self,
        date: TradeDate,
        merged_target: TargetPortfolio,
        current_account: Account,
        market_state: pd.DataFrame,
        context: ExecutionContext,
    ) -> Sequence[Order]:
        """生成当日交易指令（T0 判定）。

        幂等：是（同输入 ⇒ 同输出）
        副作用：无
        """
        ...

    def apply_constraints(
        self,
        target: TargetPortfolio,
        current: Account,
    ) -> TargetPortfolio:
        """应用组合层硬约束（市值分位 / 行业上限 / 个股权重 / 换手迟滞）。

        幂等：是
        """
        ...


@runtime_checkable
class Executor(Protocol):
    """执行层（T0/T1 链路、成交口径、成本）。

    语义边界（R4-H3）：本协议只表达「T0 接收指令集 → T+1 产出成交」的每日边界；
    钩子编排 / 合并去重 / 延迟队列属内核内部（见本模块首部声明）。
    """

    def execute_daily(
        self,
        date: TradeDate,
        orders: Sequence[Order],
        account: Account,
        context: ExecutionContext,
    ) -> Sequence[Fill]:
        """每日执行（T0 判定、T+1 开盘价成交；顺延语义由内核延迟队列处理）。

        形态说明（R5）：本方法在回测宿主内**跨两天**——内部两阶段
        （T0 收单 / T+1 开盘成交）由引擎连续驱动；纸面宿主将其**分别暴露**为
        run_t0 / run_t1（T+1 成交来自人工回填而非引擎撮合）。
        两种形态是同一执行语义的宿主适配，不是两种执行语义。

        幂等：否（改变账本状态）
        副作用：生成成交记录（T+1 实际成交时入账）
        失败：涨跌停 / 停牌 ⇒ 顺延（内部登记延迟订单；减仓 / 回补特例不进队列）
        """
        ...


@runtime_checkable
class Ledger(Protocol):
    """账本（物理 + 虚拟）。

    R4-M7：净额化分摊规则是「无开关的确定性算法」，**内聚到本层**
    （注册的确定性算法），禁止由调用点传入比例参数（不同调用点传不同比例
    = 开关化后门）。
    """

    def get_account(self) -> Account:
        """获取物理账户快照（只读）。"""
        ...

    def get_virtual_account(self, sleeve_id: str) -> VirtualAccount:
        """获取袖子虚拟账本（只读）。"""
        ...

    def apply_fills(self, fills: Sequence[Fill]) -> None:
        """应用成交到物理 + 虚拟账本（分摊规则由本层注册的确定性算法决定）。

        副作用：更新账本状态
        校验：三恒等式（Σ袖子虚拟{现金, 持仓股数, NAV} ≡ 物理账本），破则 RuntimeError
        """
        ...

    def snapshot(self, date: TradeDate) -> None:
        """落盘当日快照。"""
        ...


@runtime_checkable
class FundScheduler(Protocol):
    """跨袖资金调度器（协议 v0.4，方案 §4.9 软隔离+临时借用的协议表达；P6.5 前置冻结）。

    核心语义：物理现金池唯一；各袖有名义配额（决定常规下单上限）；
    快袖可借用慢袖当日闲置额度（记虚拟占用条目 VirtualAccount.borrowed_credit）；
    被借方到调仓日必须次日 T+1 归还，禁止滚动拖欠。
    归还手段（v0.6 对齐方案 §4.9 v1.10）：归还 = 借入方**放弃自身新下单**
    （只压下单额度、不动存量持仓）；**被动卖出默认禁止**（OrderReason 无对应枚举，
    实现侧不存在该路径）。
    与净额化正交：净额化在物理指令层，借用只在虚拟额度层。
    """

    def available_budget(
        self,
        sleeve_id: str,
        date: TradeDate,
        virtual_accounts: Mapping[str, VirtualAccount],
    ) -> Money:
        """返回该袖当日可下单预算（名义配额内可用 + 可借入的其他袖闲置额度）。

        幂等：是（只读）。
        失败：sleeve_id 未注册 ⇒ ValueError。
        """
        ...

    def record_borrowing(
        self,
        borrower: str,
        lender: str,
        amount: Money,
        date: TradeDate,
    ) -> None:
        """登记一笔跨袖借用（更新双方虚拟账本的 borrowed_credit 占用条目）。

        副作用：更新虚拟账本占用条目。
        校验：借用后 Σ袖子 borrowed_credit ≡ 0（三恒等式在现金项内对消）。
        """
        ...

    def settle_due(
        self,
        date: TradeDate,
        virtual_accounts: Mapping[str, VirtualAccount],
    ) -> Sequence[Order]:
        """归还结算：被借方到调仓日 / 产生下单需求时，清平借入方的到期占用。

        归还以「借入方放弃自身新下单」实现（压减其当日可下单额度，方案 §4.9 v1.10）；
        **被动卖出默认禁止 ⇒ 当前版本恒返回空序列**（返回类型保留，为未来枚举
        登记后的极端情形留白）；禁止滚动拖欠。

        返回：归还相关指令（本期恒为空；归还经由额度压减体现，不产生卖出单）。
        副作用：清平到期占用条目。
        """
        ...


@runtime_checkable
class PortfolioMerger(Protocol):
    """组合合并器（多袖目标 → 单一目标表）。

    单袖退化 = 直通（MVP 形态；P2b 默认关接入，P6.5 虚拟子袖对账验收）。
    """

    def merge(
        self,
        targets: Mapping[str, TargetPortfolio],  # sleeve_id → 目标
        current: Account,
    ) -> TargetPortfolio:
        """合并多袖目标（净额化 + 集中度约束）。

        分层语义（方案 v1.6 写死）：
        - 抵消部分 = 虚拟账本内部转账（无条件完成）
        - 仅净额部分生成市场指令
        - 禁止候选域互斥 / 先占规则

        幂等：是
        校验：Σ袖子目标权重 = 1.0（合并前各自归一）
        """
        ...

    def net_across_sleeves(
        self,
        orders: Sequence[Order],
    ) -> Sequence[Order]:
        """跨袖净额化（同股买卖抵消）。

        构造性对账三件套：
        ① A 卖 X + B 买 X ⇒ 净额零换手
        ② N 袖同股 ⇒ 分摊守恒
        ③ A 自拆两个虚拟子袖 ⇒ 与单袖逐位一致
        """
        ...

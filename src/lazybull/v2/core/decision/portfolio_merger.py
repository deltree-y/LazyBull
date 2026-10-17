"""单袖组合合并器与 core 内部过渡端口（P2b-T1 新交付件）。

契约依据：净额化冻结文档（``docs/contracts/netting_freeze.md`` F3）A 区 A1
直通退化与状态机 S6；协议 F9（``docs/contracts/protocols.md`` §3.4 与
``v2/common/protocols/core.py::PortfolioMerger``）；方案 F12 §4.10 过渡结构；
实施规划 ``docs/plans/v2/p2b_implementation_plan_20261016.md`` D1/D2/D4。

能力边界（D1）：

- ``merge()`` 仅接受**一个** ``TargetPortfolio`` 并返回该对象本身；
- ``net_across_sleeves()`` 仅覆盖 S6：空指令序列或全部指令属于同一
  ``source_sleeve`` 时，返回原序列及原指令对象；
- 空/多目标映射、映射键与 ``target.sleeve_id`` 不一致、混合 ``source_sleeve``
  均显式 fail-fast，错误含 sleeve 集合与阶段信息；
- **空目标映射**与"单目标内 ``target_weights`` 为空"是两种不同结构情形：
  前者拒绝，后者原样透传——结构防越界不扩大为权重合法性判断；
- 多袖输入（A 卖 X + B 买 X、N 袖同股同向、A 自拆两个子袖）一律拒绝，
  不在本阶段计算任何"看似合理"的多袖结果（防止提前形成未经 P6.5 验收的
  第二套净额化语义）。

直通语义（D2）：返回对象满足 ``merged is targets[sleeve_id]``，权重映射保持
原映射对象与插入顺序；不求和、不重归一、不截断、不排序、不读取 ``current``、
不修改 ``targets``/目标对象/权重映射/当前账户；Σ=1 为上游前置条件（F9），
本层不重复运行时校验，非精确权重和也不"顺手修正"（上游既有归一化负责其
输入不变量；直通测试只证明不改写输入，不证明其可作为真实交易合法输入）。

过渡端口 ``LegacyTargetMergePort``（D4）：core 内部过渡端口，签名只接收明确
列出的只读旧状态字段并返回权重 dict，禁止把引擎可变实例整体交给适配器；
由 ``hosts/backtest/merger_adapter.py``（T2）实现，引擎接缝与装配见 T2/T3。
**P4 原生 Account/TargetPortfolio 边界接通后删除本端口**（F12 到期表），
单袖合并器本体保留。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from src.lazybull.v2.common.types import Account, Order, TargetPortfolio

# 阶段信息（fail-fast 错误消息统一携带）：P2b 单袖直通过渡期
_STAGE = "P2b"


class SingleSleevePortfolioMerger:
    """单袖直通合并器（净额化冻结 A1 / 状态机 S6）。

    ``PortfolioMerger`` 协议（``v2/common/protocols/core.py``）的结构兼容实现：
    不继承协议类，协议签名漂移由测试比对暴露（tests/test_v2_p2b_t1_merger.py）。
    """

    def merge(
        self,
        targets: Mapping[str, TargetPortfolio],  # sleeve_id → 目标
        current: Account,
    ) -> TargetPortfolio:
        """单袖目标直通（A1）：返回映射内唯一目标对象本身。

        强等价口径（D2）：``merged is targets[sleeve_id]``，``target_weights``
        保持原映射对象与插入顺序；不求和、不重归一、不截断、不排序、不读取
        ``current``，不新增权重合法性/集中度分支，不修改任何输入。

        失败：空/多目标映射、映射键与 ``target.sleeve_id`` 不一致 ⇒ ValueError
        （错误含 sleeve 集合与阶段信息）。单目标内 ``target_weights`` 为空属
        直通情形，原样透传。
        """
        if len(targets) != 1:
            raise ValueError(
                f"单袖直通合并器拒绝非单目标输入（阶段={_STAGE}）：targets 含 "
                f"{len(targets)} 个目标，sleeve 集合={sorted(targets)}；P2b 仅交付 "
                "A1 单袖直通，多袖合并（反向抵消/同向分摊/自拆）归 P6.5"
            )
        sleeve_id, target = next(iter(targets.items()))
        if sleeve_id != target.sleeve_id:
            raise ValueError(
                f"单袖直通合并器拒绝键袖不一致输入（阶段={_STAGE}）：映射键 "
                f"{sleeve_id!r} != target.sleeve_id {target.sleeve_id!r}"
            )
        return target

    def net_across_sleeves(
        self,
        orders: Sequence[Order],
    ) -> Sequence[Order]:
        """跨袖净额化的 S6 单袖子集（本期唯一覆盖状态）。

        空指令序列或全部指令属于同一 ``source_sleeve`` 时，返回原序列及原指令
        对象；混合 ``source_sleeve``（反向/同向/自拆）显式 fail-fast。
        """
        sleeves = {order.source_sleeve for order in orders}
        if len(sleeves) > 1:
            raise ValueError(
                f"单袖直通合并器拒绝混合袖指令（阶段={_STAGE}）：net_across_sleeves "
                f"收到 source_sleeve 集合={sorted(sleeves)}；跨袖净额化 S1~S5/S7 "
                "（反向抵消/同向分摊/自拆）归 P6.5"
            )
        return orders


@runtime_checkable
class LegacyTargetMergePort(Protocol):
    """core 内部过渡端口（P2b D4；P4 原生边界接通后删除，见模块 docstring）。

    引擎侧唯一接缝：``core/execution/signal_execution.py`` 当批归一化之后、
    批次限权之前（T2 落位）；hosts 只读适配器实现本端口并调用真正的
    ``SingleSleevePortfolioMerger.merge()``。core 不得 import hosts（F11 §4.10）。
    """

    def merge_legacy_batch_targets(
        self,
        *,
        weights: Dict[str, float],
        date: pd.Timestamp,
        cash: float,
        total_value: Optional[float],
        positions: Mapping[str, Mapping[str, Any]],
        initial_capital: float,
    ) -> Dict[str, float]:
        """当批归一化目标权重经合并器后返回权重 dict。

        参数全为明确列出的旧引擎只读状态字段（禁止传引擎可变实例整体）：

        - ``weights``：当批归一化后的候选权重（当批相对语义，非整袖/全账户
          资产配置，F12 §4.10）；
        - ``date``：信号日（旧引擎 pd.Timestamp 口径，边界转换归适配器）；
        - ``cash``：既有 ``current_capital``；
        - ``total_value``：既有 ``_last_rebalance_nav``（信号生成已计算的同日
          组合总值快照，非相对 NAV）；缺失（None）时实现侧必须 fail-fast
          停止该次启用运行（D5）；
        - ``positions``：旧持仓 dict（只读；聚合视图关键键缺失时实现侧
          fail-fast，禁止跳过或伪造）；
        - ``initial_capital``：初始资金（NAV 唯一派生入口归账本层
          ``core/accounting/nav.py::calculate_nav``，适配器只消费返回值）。

        返回：合并后的权重 dict（P2b 直通语义 = 内容、顺序与数值对象不变）。
        """
        ...

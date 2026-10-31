"""单袖组合合并器（P2b 交付，P4-T3 原生链接通后保留本体）。

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

P4-T3：旧 ``LegacyTargetMergePort``、dict 适配器、merge_port 与 bypass 已删除；
回测/纸面宿主直接以公共值对象调用本合并器。
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.lazybull.v2.common.types import Account, Order, TargetPortfolio

# 阶段信息（fail-fast 错误消息统一携带）：P2b 单袖直通过渡期
_STAGE = "P2b"


class SingleSleevePortfolioMerger:
    """单袖直通合并器（净额化冻结 A1 / 状态机 S6）。

    ``PortfolioMerger`` 协议（``v2/common/protocols/core.py``）的结构兼容实现：
    不继承协议类，协议签名漂移由测试比对暴露（tests/test_v2_p2b_t1_merger.py）。
    计数只记录通过结构校验并真实执行的 merge/S6 调用，供两宿主验收。
    """

    def __init__(self) -> None:
        self.merge_call_count = 0
        self.net_call_count = 0

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
        self.merge_call_count += 1
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
        self.net_call_count += 1
        return orders

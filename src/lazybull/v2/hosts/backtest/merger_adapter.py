"""P2b 宿主侧只读适配器：旧 dict 状态 ↔ 公共值对象（D4/D5；P4 删除）。

P2b-T2 新交付件。契约依据：方案 F12 §4.10（core 内部端口 + hosts 只读适配）、
实施规划 ``docs/plans/v2/p2b_implementation_plan_20261016.md`` D4/D5、净额化
冻结 F3 A1。

职责边界：

- 实现 core 内部过渡端口 ``LegacyTargetMergePort``（core 不 import hosts，
  本模块是唯一的旧 dict ↔ 值对象转换点）；
- **每次进入接缝时**按当时传入状态新建只读 ``Account``（不在装配时缓存——
  常规目标在当日执行前生成、提前目标在当日执行后生成，两个时点不得共用
  旧快照）；随后调用真正的 ``SingleSleevePortfolioMerger.merge()``；
- 权重按原 ``.items()`` 包装/解包 ``TSCode.value`` / ``Weight.value``——禁止
  排序、``str(TSCode)`` 或额外 ``float(...)`` 强转，保持键序与原数值对象，
  避免在既有后续限权前引入新的浮点位变化；
- ``Account.nav`` 只接收账本层 ``calculate_nav`` 返回值（宿主不得出现自行
  除以 ``initial_capital`` 的公式）；``cash/total_value`` 来自既有
  ``current_capital`` 与 ``_last_rebalance_nav``（后者是同日组合总值快照，
  非相对 NAV）；缺失（None）时报错并停止该次启用运行（关闭态不经此检查）；
- 每只非零旧持仓形成一个**当前聚合视图 lot**（``buy_date/shares/buy_cost_cash``
  与旧状态一致，满足 ``Position`` 聚合校验）；``last_price`` 只按
  ``last_known_price → buy_trade_price`` 读取已有字段，禁止混入后复权
  ``buy_pnl_price``；weights/positions/lots 使用不可变外层容器；
  缺 ``buy_date``/``buy_cost_cash`` 等关键字段 fail-fast（错误含股票与键名），
  禁止跳过或伪造；不再度估值、不写回 ``last_known_price``、不做昂贵深复制。

该 lot 只是旧聚合持仓的只读适配，**不得被 P6.5 当作历史多 lot 或虚拟账本来源**。
P4 原生 Account/TargetPortfolio 边界接通后删除本适配器（F12 到期表），
单袖合并器本体保留。
"""

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.lazybull.v2.common.types import (
    Account,
    Lot,
    Money,
    Position,
    Price,
    Shares,
    TargetPortfolio,
    TradeDate,
    TSCode,
    Weight,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger


def _require_position_field(info: Mapping[str, Any], key: str, stock: str) -> Any:
    """读取旧持仓关键字段；缺失、None 或 NaN/NaT/NA 缺失标记必须 fail-fast。

    D5/§7：禁止跳过、伪造或静默 NaN；错误含股票与键名。字段按标量契约读取。
    （R4-T2-01/S3-T2-02：仅判 None 会让 float("nan")/pd.NaT/pd.NA 穿透进入
    Money/Price/TradeDate——``Position`` 成本聚合校验对 NaN 不自拒，
    ``abs(nan-nan) > 1e-6`` 为 False。）
    """
    value = info.get(key)
    if value is None or pd.isna(value):
        raise ValueError(
            f"P2b 宿主适配器拒绝缺键/缺失值持仓（阶段=P2b）：股票 {stock} 字段 "
            f"{key!r} 缺失或为缺失标记（None/NaN/NaT/NA）；旧聚合视图不允许伪造"
        )
    return value


class LegacyMergerAdapter:
    """``LegacyTargetMergePort`` 的宿主侧实现（只读快照 + 真实合并调用 + 计数）。

    结构兼容实现（不继承 Protocol）：端口签名漂移由测试比对暴露
    （tests/test_v2_p2b_t2_adapter.py）。
    """

    def __init__(self, sleeve_id: str) -> None:
        """装配：固定袖子 ID 由组合根（runtime.py 模块级常量单一 owner）传入。"""
        self._sleeve_id = sleeve_id
        self._merger = SingleSleevePortfolioMerger()
        self._call_count = 0

    @property
    def sleeve_id(self) -> str:
        return self._sleeve_id

    @property
    def call_count(self) -> int:
        """实际进入真实 ``merge()`` 的累计次数（D7：生成后拖尾拒绝不回滚计数）。"""
        return self._call_count

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
        """当批归一化权重经单袖合并器直通后返回权重 dict（无损往返）。"""
        # D5：启用态下 _last_rebalance_nav 缺失 ⇒ 报错并停止该次启用运行
        if total_value is None:
            raise RuntimeError(
                "P2b 合并接缝启用时 _last_rebalance_nav 缺失（同日组合总值快照为 None）；"
                "停止该次启用运行（D5；关闭态不经过此检查）"
            )
        target = TargetPortfolio(
            sleeve_id=self._sleeve_id,
            target_weights=MappingProxyType(
                {TSCode(code): Weight(weight) for code, weight in weights.items()}
            ),
            as_of_date=TradeDate(pd.Timestamp(date).date()),
        )
        account = self._build_account(
            cash=cash,
            total_value=total_value,
            positions=positions,
            initial_capital=initial_capital,
        )
        merged = self._merger.merge({self._sleeve_id: target}, account)
        self._call_count += 1  # D7：实际进入真实 merge() 才计数
        return {code.value: weight.value for code, weight in merged.target_weights.items()}

    def _build_account(
        self,
        *,
        cash: float,
        total_value: float,
        positions: Mapping[str, Mapping[str, Any]],
        initial_capital: float,
    ) -> Account:
        """按当前传入状态新建只读 Account（D5：逐次接缝新建，不缓存、不再估值）。"""
        position_map: Dict[TSCode, Position] = {}
        for stock, info in positions.items():
            shares = _require_position_field(info, "shares", stock)
            if not shares:
                continue  # 零股占位条目不进聚合视图（旧引擎全卖即删，防御性跳过）
            buy_date = _require_position_field(info, "buy_date", stock)
            buy_cost_cash = _require_position_field(info, "buy_cost_cash", stock)
            buy_trade_price = _require_position_field(info, "buy_trade_price", stock)
            # last_price 只按 last_known_price → buy_trade_price 读取已有字段
            # （禁止混入后复权 buy_pnl_price）
            last_price = info.get("last_known_price")
            if last_price is None:
                last_price = buy_trade_price
            lot_date = pd.Timestamp(buy_date).date()
            if pd.isna(lot_date):
                # 转换后兜底（R4-T2-01 复核条件：如字符串 "NaT" 经转换产生的缺失标记）
                raise ValueError(
                    f"P2b 宿主适配器拒绝缺键/缺失值持仓（阶段=P2b）：股票 {stock} 字段 "
                    "'buy_date' 转换后为 NaT；旧聚合视图不允许伪造"
                )
            lot = Lot(
                shares=Shares(shares),
                buy_date=TradeDate(lot_date),
                cost_basis=Money(buy_cost_cash),
            )
            position_map[TSCode(stock)] = Position(
                ts_code=TSCode(stock),
                shares=Shares(shares),
                cost_basis=Money(buy_cost_cash),
                last_price=Price(last_price),
                lots=(lot,),
            )
        return Account(
            cash=Money(cash),
            positions=MappingProxyType(position_map),
            total_value=Money(total_value),
            nav=calculate_nav(total_value, initial_capital),  # NAV 唯一派生入口（账本层）
        )

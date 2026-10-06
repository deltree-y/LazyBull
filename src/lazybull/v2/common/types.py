"""v2 公共值对象（协议 v0.5 §0 的落地实现，含 v0.4 新增字段）。

口径要点（与协议一致）：
- 价格 / 金额一律浮点，与旧引擎数值路径一致（MVP 唯一口径；整分化是实盘指令层留白）。
- 内部一律类型化 TradeDate / TSCode；边界处（分区名 / config / CLI / parquet 列）
  由 TradeDate.from_str 单点转换。
- Mapping 字段的浅不可变由实现侧以 MappingProxyType / frozendict 强制。
- v0.4 新增：VirtualAccount.borrowed_credit（跨袖资金借用占用）、
  PanelFrame.available_columns_at（列级可用起点查询，方案 M1.5）。
- P0 评审（C1）：Position 禁止「空 lots 且非零 shares」的矛盾态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

__all__ = [
    "TradeDate",
    "TSCode",
    "Price",
    "Money",
    "Shares",
    "Weight",
    "Lot",
    "Position",
    "VirtualPosition",
    "Account",
    "VirtualAccount",
    "OrderSide",
    "OrderType",
    "OrderReason",
    "Order",
    "Fill",
    "TargetPortfolio",
    "FeatureQuery",
    "LabelQuery",
    "PanelFrame",
    "SignalScore",
    "RankedCandidates",
    "ModelConfig",
    "DataState",
    "HostMode",
    "TradingConfigLike",
    "ExecutionContext",
    "LedgerEntry",
]


# ========== 基础值对象 ==========


@dataclass(frozen=True, slots=True, order=True)
class TradeDate:
    """交易日（内部一律类型化；边界处由 from_str 单点转换）。可比较（按日期排序）。"""

    value: date

    def __str__(self) -> str:
        return self.value.strftime("%Y%m%d")

    @classmethod
    def from_str(cls, s: str) -> "TradeDate":
        """边界转换唯一入口（YYYYMMDD 字符串 → TradeDate）。"""
        return cls(datetime.strptime(s, "%Y%m%d").date())


@dataclass(frozen=True, slots=True, order=True)
class TSCode:
    """股票代码（600000.SH 语义）。可比较（确定性排序）。"""

    value: str


# 价格与金额：浮点（R4-H1；整分化见 ExactMoney 留白，P8 实盘指令层本地决定）


@dataclass(frozen=True, slots=True)
class Price:
    """价格（浮点，与市场报价或复权价同口径）。"""

    value: float


@dataclass(frozen=True, slots=True)
class Money:
    """金额（浮点，与旧引擎记账路径一致）。"""

    value: float


@dataclass(frozen=True, slots=True)
class Shares:
    """股数。"""

    value: int


@dataclass(frozen=True, slots=True)
class Weight:
    """权重（0~1，浮点）。"""

    value: float


# ========== 账本值对象（lot 批次模型） ==========


@dataclass(frozen=True, slots=True)
class Lot:
    """持仓批次（同股多批次的最小单位）。

    A3 契约「回补不重置买入日」+ 补齐买入算新持仓 ⇒ 同股可有多批次、不同买入日；
    净额化到期归属「以最早买入的虚拟份额为准」需要 lot 级记账。
    """

    shares: Shares
    buy_date: TradeDate
    cost_basis: Money  # 该批次总成本


@dataclass(frozen=True, slots=True)
class Position:
    """单股持仓（物理账本；多批次语义的聚合视图）。

    shares / cost_basis 是 lots 的聚合，构造时强制校验一致性（账本地基值对象
    不得自相矛盾）。
    """

    ts_code: TSCode
    shares: Shares
    cost_basis: Money
    last_price: Price
    lots: Sequence[Lot]

    def __post_init__(self) -> None:
        if not self.lots:
            # P0 评审（C1）：空 lots 且非零 shares 是自相矛盾的账本地基值，构造即失败
            if self.shares.value != 0:
                raise ValueError(
                    f"Position({self.ts_code.value}): 空 lots 不允许携带非零 shares({self.shares.value})"
                )
            return
        lot_shares = sum(lot.shares.value for lot in self.lots)
        if self.shares.value != lot_shares:
            raise ValueError(
                f"Position({self.ts_code.value}): shares({self.shares.value})"
                f" != Σ lots.shares({lot_shares})"
            )
        lot_cost = sum(lot.cost_basis.value for lot in self.lots)
        if abs(self.cost_basis.value - lot_cost) > 1e-6:
            raise ValueError(
                f"Position({self.ts_code.value}): cost_basis({self.cost_basis.value})"
                f" != Σ lots.cost_basis({lot_cost})"
            )

    @property
    def first_buy_date(self) -> TradeDate:
        """持有期计算口径：最早批次买入日。空 lots（零持仓视图）无买入日语义，显式报错。"""
        if not self.lots:
            raise ValueError(f"Position({self.ts_code.value}): 空 lots 无 first_buy_date")
        return min(lot.buy_date for lot in self.lots)


@dataclass(frozen=True, slots=True)
class VirtualPosition:
    """袖子虚拟持仓（lot 级记账，净额化到期归属的前提）。"""

    ts_code: TSCode
    lots: Sequence[Lot]


@dataclass(frozen=True, slots=True)
class Account:
    """账户快照（物理账本）。"""

    cash: Money
    positions: Mapping[TSCode, Position]
    total_value: Money
    nav: float  # 净值（相对初始资金；唯一计算入口在账本层）


@dataclass(frozen=True, slots=True)
class VirtualAccount:
    """袖子虚拟账本（逻辑分账；lot 级）。

    borrowed_credit（协议 v0.4，方案 §4.9）：当前借入的资金额度（其他袖子的闲置额度）；
    正 = 借入，负 = 借出。三恒等式验算时该条目在 Σ袖子虚拟现金 内对消（Σ borrowed_credit ≡ 0）。
    """

    sleeve_id: str
    cash: Money
    positions: Mapping[TSCode, VirtualPosition]
    total_value: Money
    borrowed_credit: Money = Money(0.0)  # 跨袖资金借用占用（默认 0 = 无借用）


# ========== 交易指令与成交 ==========


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET_OPEN = "market_open"  # T+1 开盘价


class OrderReason(Enum):
    """指令来源（封闭枚举，新增须登记；对照每日 15 个有序钩子清单）。"""

    REBALANCE = "rebalance"  # 调仓
    EXPIRY = "expiry"  # 到期
    STOP_LOSS = "stop_loss"  # 止损
    TAKE_PROFIT = "take_profit"  # 止盈
    CONDITION_SELL = "condition_sell"  # 条件卖出（pending_condition_sells 队列）
    TRIM = "trim"  # 减仓（暴露政策）
    REPLENISH = "replenish"  # 回补
    REFILL = "refill"  # 补齐
    NETTING = "netting"  # 净额化抵消（内部转账）
    INITIAL = "initial"  # 建仓


@dataclass(frozen=True, slots=True)
class Order:
    """交易指令。"""

    ts_code: TSCode
    side: OrderSide
    target_shares: Shares  # 目标股数（不是增量）
    order_type: OrderType
    reason: OrderReason
    source_sleeve: str  # 袖子 ID（净额化前）


@dataclass(frozen=True, slots=True)
class Fill:
    """成交记录（只含真实成交事实；滑点等分析性分解属执行归因层）。"""

    order: Order
    fill_price: Price
    fill_shares: Shares
    fill_date: TradeDate
    commission: Money


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """目标组合（袖子输出 / 组合层输入）。"""

    sleeve_id: str
    target_weights: Mapping[TSCode, Weight]  # Σ = 1.0（袖子内部归一）
    as_of_date: TradeDate


# ========== 特征与标签 ==========


@dataclass(frozen=True, slots=True)
class FeatureQuery:
    """特征查询（数据面 API）。"""

    columns: Sequence[str]
    start_date: TradeDate
    end_date: TradeDate
    universe: Sequence[TSCode] | None = None  # None = 全市场


@dataclass(frozen=True, slots=True)
class LabelQuery:
    """标签查询（variant 选择取值变体：raw=label_value 原值 / neu=neu_label_value 中性化变体）。

    输出 shape 与变体无关（columns=[label_value, maturity_status]），
    label_value 承载所选变体的值。
    """

    label_name: str
    start_date: TradeDate
    end_date: TradeDate
    universe: Sequence[TSCode] | None = None
    variant: str = "raw"  # "raw" / "neu"

    def __post_init__(self) -> None:
        if self.variant not in ("raw", "neu"):
            raise ValueError(f"标签变体非法: {self.variant!r}（合法值 raw/neu）")


@dataclass(frozen=True, slots=True)
class PanelFrame:
    """特征面板（DataFrame 包装，附加 manifest 内容指纹与列级可用起点）。

    manifest_version 是 manifest schema 版本（常量 "1"）；内容演进指纹由
    manifest_fingerprint 承载（剔除 updated_at 的 canonical JSON sha256_16）。
    """

    df: pd.DataFrame  # index = (trade_date, ts_code)
    manifest_version: str
    # 列名 → 可用起点（manifest available_from，协议 v0.4 / 方案 M1.5）；空映射 = 未登记
    available_from: Mapping[str, "TradeDate"] = field(default_factory=dict)
    manifest_fingerprint: str = ""  # manifest 内容指纹（load_features 填充；空 = 未采集）

    def validate_schema(self, expected_columns: Sequence[str]) -> None:
        """校验列集合（缺列硬报错）。"""
        missing = [c for c in expected_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"PanelFrame 缺列: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    def available_columns_at(self, date: TradeDate) -> Sequence[str]:
        """返回指定日期的可用列集（协议 v0.4，方案 M1.5 分时段变列集训练用）。

        仅返回 available_from <= date 的列；未登记起点的列视为恒可用
        （宽松默认——列名合法性与起点登记的严格校验归 store 层 load_features）。
        """
        return [
            c
            for c in self.df.columns
            if self.available_from.get(c) is None or self.available_from[c] <= date
        ]


# ========== 信号与模型 ==========


@dataclass(frozen=True, slots=True)
class SignalScore:
    """单股信号分数。"""

    ts_code: TSCode
    score: float  # 排序分数（无概率语义）
    meta: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RankedCandidates:
    """排序后的候选列表（信号层输出）。"""

    date: TradeDate
    scores: Sequence[SignalScore]  # 按 score 降序
    sleeve_id: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型配置（袖子规格的一部分）。"""

    model_family: str
    feature_columns: Sequence[str]
    label_name: str
    hyperparams: Mapping[str, Any]
    train_window_months: int
    val_window_months: int
    ensemble_seeds: Sequence[int] = ()  # 空 = 单模型；生产 A = 3 种子


# ========== 数据态与审计 ==========


@dataclass(frozen=True, slots=True)
class DataState:
    """数据态指纹（对账基线；代码态 + 数据态双指纹）。"""

    raw_partitions: Mapping[str, str]
    features_partition_max: TradeDate
    config_digest: str
    code_digest: str  # git commit + 脏标记


class HostMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@runtime_checkable
class TradingConfigLike(Protocol):
    """类型化配置协议（延续 TradingConfig dataclass 先例，禁止裸 Mapping）。"""


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """执行上下文（三宿主共享）。"""

    mode: HostMode
    current_date: TradeDate
    data_state: DataState
    config: TradingConfigLike


# ========== 假设台账条目（schema 类型化，禁裸 Mapping） ==========


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """假设台账条目（append-only；schema 是 P0 冻结交付物，字段清单 P0 定稿）。"""

    hypothesis_id: str
    kind: str  # "prereg" / "conclusion"
    payload: Mapping[str, Any]
    data_state: DataState
    registered_at: str  # ISO 时间戳

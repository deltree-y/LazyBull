# LazyBull v2 模块协议层（正式契约）

> **P0 落档标记**：本文档自 `docs/plans/v2_protocols.md` v0.7（2026-10-01）转正为正式契约。
> 生效状态：**待 P0 确认**。代码实现以本文档为唯一接口依据。

| 契约版本 | 落档日期 | 来源草案 | 变更摘要 |
|---|---|---|---|
| F1（= v0.3） | 2026-09-30 | plans/v2_protocols.md v0.3 | 首次转正（经 R4 / R5 两轮评审） |
| F2（= v0.4） | 2026-10-01 | plans/v2_protocols.md v0.4 | 方案 v1.8 回写：新增 FundSchedulerProtocol + PanelFrame.available_columns_at + VirtualAccount.borrowed_credit |
| F3（= v0.5） | 2026-10-01 | plans/v2_protocols.md v0.5 | P0 交付：§11 协议↔旧引擎语义对照表逐项打钩（22/22） |
| F4（= v0.6） | 2026-10-01 | plans/v2_protocols.md v0.6 | P0 评审第一轮：路径勘误 + FundScheduler 归还规则对齐 |
| F5（= v0.7） | 2026-10-01 | plans/v2_protocols.md v0.7 | P0 评审第二轮：§11 打钩依据补代码态、内核内部语义对账清单化、payload 类型化欠账、止盈行 + 止损双层归属 |

---
---
# LazyBull v2 模块协议层（Protocol 定义草案）

> 版本：v0.7（2026-10-01，配套方案 v1.11；P0 评审第二轮：§11 打钩依据补代码态、内核内部语义对账清单化、payload 类型化欠账登记）
> 状态：待 P0 落档转正；本文档为接口契约的代码形态草案
> 技术选型：Python 3.12 `typing.Protocol`（结构化子类型，无须显式继承）+ frozen dataclass + `pandas.DataFrame` / `datetime.date`
> 设计原则：① 数据面与计算面分离；② 袖子零互 import；③ 跨模块对象不可变；④ 所有协议方法标注**幂等性**与**副作用**
> **设计方法论（R4 教训）**：先枚举现实引擎语义（R1 侦察事实），再让类型覆盖它——禁止先定义理想类型再让引擎适应类型
>
> **协议版本化（v0.2 新增）**：本文档 append-only；每次修订在头部登记版本行 + 变更摘要 + 兼容性标注（兼容 / 破坏）。

---

## 版本变更登记

| 版本 | 日期 | 变更摘要 | 兼容性 |
|---|---|---|---|
| v0.1 | 2026-09-30 | 首版草案（备份：`v2_protocols_v01_backup.md`） | — |
| v0.2 | 2026-09-30 | R4 评审修正：H1 Price/Money 回退浮点（整分化移实盘留白）；H2 Trainer 集成语义 + Orchestrator 拆分；H3 协议边界定位声明 + 延迟订单最小语义；H4 lot 批次结构；H5 台账写入走 DataStore；H6 协议类型上移 common/protocols；M1~M8 逐项修正；新增 §11 协议↔旧引擎语义对照表 | 破坏（值对象类型 / 依赖方向 / 多个签名变更） |
| v0.3 | 2026-09-30 | R5 收尾修正：§11 对照表补 5 行（估值价格回退链 / holdings_snapshot / stop_loss_checker / Kelly 权重归一化 / 分批排期锚定差异）+ 非穷尽声明；`OrderReason.CONDITION_SELL`；`LedgerEntry` frozen dataclass 替代裸 Mapping（台账 schema 类型化）；`Position.__post_init__` 聚合不变量校验；`execute_daily` 补两阶段注释（与纸面 run_t0/run_t1 的形态差异说明）；`ExactMoney` 标注伪代码留白 | 兼容（新增枚举值 / 新增类型 / 注释补强；无签名破坏） |
| v0.4 | 2026-10-01 | 方案 v1.8 回写：新增 `FundSchedulerProtocol`（跨袖资金调度，对应方案 §4.9 软隔离+临时借用——名义配额 / 闲置借用 / 归还规则 / 虚拟占用费）；`PanelFrame` 补 `available_from` 查询接口（对应 M1.5 样本起点扩展的 manifest 列级可用起点）；`VirtualAccount` 补 `borrowed_credit` 占用条目字段 | 兼容（新增协议 / 新增字段 / 新增方法；无签名破坏） |
| v0.5 | 2026-10-01 | P0 交付：§11 协议↔旧引擎语义对照表逐项打钩（22/22）；核对依据 = R1 侦察三项硬事实 + 2026-10-01 代码抽查（16 个关键符号 / 文件逐一定位核实） | 兼容（仅确认列状态变更与注释，无签名破坏） |
| v0.6 | 2026-10-01 | P0 评审第一轮：§9 落档清单路径勘误（v2 命名空间：`src/lazybull/v2/common/types.py`，迁移期统一落 v2/，P4 全切换后评估展平）；§0 标题同步；FundScheduler 归还规则对齐方案 §4.9 v1.10（放弃新下单优先、被动卖出默认禁止——OrderReason 无枚举） | 兼容（路径勘误 / 文档语义收紧；无签名破坏） |
| v0.7 | 2026-10-01 | P0 评审第二轮：§11 打钩依据补核实代码态（commit）；「内核内部」语义单列 P2a 对账用例分片（2-B）；LedgerEntry.payload 类型化欠账登记入 §10（2-C）；§11 补止盈行 + 止损检查器双层归属对齐方案 v1.11（1-A） | 兼容（注记 / 清单 / 登记；无签名破坏） |

---

## 0. 公共类型（v2/common/types.py）

> 命名空间约定（v0.6 勘误）：迁移期 v2 代码统一落 `src/lazybull/v2/`（v1 的 common 在用、不可混入），P4 全切换后评估展平。

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol, Sequence, Mapping, Any
import pandas as pd

# ========== 基础值对象 ==========

@dataclass(frozen=True, slots=True)
class TradeDate:
    """交易日（内部一律类型化；边界处单点转换——分区名 / config / CLI / parquet 列为 YYYYMMDD 字符串）"""
    value: date

    def __str__(self) -> str:
        return self.value.strftime("%Y%m%d")

    @classmethod
    def from_str(cls, s: str) -> "TradeDate":
        """边界转换唯一入口"""
        ...

@dataclass(frozen=True, slots=True)
class TSCode:
    """股票代码（600000.SH 语义）"""
    value: str

# --- 价格与金额：MVP 口径 = 旧引擎浮点（R4-H1） ---
# 回测成交口径是 T+1 开盘的复权价（open_adj），复权价是复权因子构造的浮点数（任意小数位），
# 整分化会改变成交价并漂移全部下游数值，与 P2a「净值 1e-6 容差」直接冲突。
# MVP 记账口径一律 float（与旧引擎数值路径一致）；整分化 / Decimal 是实盘指令层的本地关注点。

@dataclass(frozen=True, slots=True)
class Price:
    """价格（浮点，与市场报价或复权价同口径）"""
    value: float

@dataclass(frozen=True, slots=True)
class Money:
    """金额（浮点，与旧引擎记账路径一致；含最低佣金 / 单边印花税等 cost.py 语义）"""
    value: float

@dataclass(frozen=True, slots=True)
class ExactMoney:
    """实盘指令层精确金额（P8 留白占位，伪代码——运行时无意义的空类，
    Decimal 或整数分由实盘宿主本地决定；MVP 不实例化）"""
    ...

@dataclass(frozen=True, slots=True)
class Shares:
    """股数"""
    value: int

@dataclass(frozen=True, slots=True)
class Weight:
    """权重（0~1，浮点）"""
    value: float

# ========== 账本值对象（含 lot 批次模型，R4-H4） ==========

@dataclass(frozen=True, slots=True)
class Lot:
    """持仓批次（同股多批次的最小单位）

    语义依据：A3 契约「回补不重置买入日」+ 补齐买入算新持仓 ⇒ 同股可有多批次、不同买入日；
    净额化到期归属「以最早买入的虚拟份额为准」需要 lot 级记账。
    """
    shares: Shares
    buy_date: TradeDate
    cost_basis: Money  # 该批次总成本

@dataclass(frozen=True, slots=True)
class Position:
    """单股持仓（物理账本；多批次语义的聚合视图）"""
    ts_code: TSCode
    shares: Shares          # Σ lots.shares
    cost_basis: Money       # Σ lots.cost_basis
    last_price: Price
    lots: Sequence[Lot]     # 批次明细；最早买入日 = min(lot.buy_date)
    # nav 等派生量由账本层计算，不在此冗余存储（避免舍入路径）

    def __post_init__(self) -> None:
        """聚合不变量校验（R5：账本地基值对象不得自相矛盾）"""
        if self.lots:
            if self.shares.value != sum(lot.shares.value for lot in self.lots):
                raise ValueError(f"Position({self.ts_code.value}): shares != Σ lots.shares")
            if abs(self.cost_basis.value - sum(lot.cost_basis.value for lot in self.lots)) > 1e-6:
                raise ValueError(f"Position({self.ts_code.value}): cost_basis != Σ lots.cost_basis")

    @property
    def first_buy_date(self) -> TradeDate:
        """持有期计算口径：最早批次买入日"""
        return min(lot.buy_date for lot in self.lots)

@dataclass(frozen=True, slots=True)
class VirtualPosition:
    """袖子虚拟持仓（lot 级记账，净额化到期归属的前提）"""
    ts_code: TSCode
    lots: Sequence[Lot]

@dataclass(frozen=True, slots=True)
class Account:
    """账户快照（物理账本）"""
    cash: Money
    positions: Mapping[TSCode, Position]
    total_value: Money  # cash + Σ(position.market_value)
    nav: float          # 净值（相对初始资金；派生量，唯一计算入口在账本层）
    # 浅不可变警告：Mapping 需以 MappingProxyType / frozendict 实现（协议约定，实现侧强制）

@dataclass(frozen=True, slots=True)
class VirtualAccount:
    """袖子虚拟账本（逻辑分账；lot 级）。

    borrowed_credit（v0.4，方案 §4.9）：当前借入的资金额度（其他袖子的闲置额度）；
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
    # 未来扩展：LIMIT / TWAP 等

class OrderReason(Enum):
    """指令来源（审计与 §4.7 退役复核的聚合维度；封闭枚举，新增须登记；对照每日 15 个有序钩子清单补全）

    注（v0.6）：跨袖资金借用的「被动归还卖出」**无枚举、默认禁止**（方案 §4.9 v1.10——
    归还以放弃自身新下单实现；若未来确需被动卖出，必须先在此登记枚举 + 预登记）。
    """
    REBALANCE = "rebalance"        # 调仓
    EXPIRY = "expiry"              # 到期
    STOP_LOSS = "stop_loss"        # 止损
    TAKE_PROFIT = "take_profit"    # 止盈
    CONDITION_SELL = "condition_sell"  # 条件卖出（pending_condition_sells 队列，R5 补）
    TRIM = "trim"                  # 减仓（暴露政策）
    REPLENISH = "replenish"        # 回补
    REFILL = "refill"              # 补齐
    NETTING = "netting"            # 净额化抵消（内部转账）
    INITIAL = "initial"            # 建仓

@dataclass(frozen=True, slots=True)
class Order:
    """交易指令"""
    ts_code: TSCode
    side: OrderSide
    target_shares: Shares  # 目标股数（不是增量）
    order_type: OrderType
    reason: OrderReason
    source_sleeve: str     # 袖子 ID（净额化前）

@dataclass(frozen=True, slots=True)
class Fill:
    """成交记录（只含真实成交事实；滑点等分析性分解属执行归因层，见 R4-M1）"""
    order: Order
    fill_price: Price
    fill_shares: Shares
    fill_date: TradeDate
    commission: Money

@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """目标组合（袖子输出 / 组合层输入）"""
    sleeve_id: str
    target_weights: Mapping[TSCode, Weight]  # Σ = 1.0（袖子内部归一）
    as_of_date: TradeDate

# ========== 特征与标签 ==========

@dataclass(frozen=True, slots=True)
class FeatureQuery:
    """特征查询（数据面 API）"""
    columns: Sequence[str]  # manifest 登记的列名
    start_date: TradeDate
    end_date: TradeDate
    universe: Sequence[TSCode] | None = None  # None = 全市场

@dataclass(frozen=True, slots=True)
class LabelQuery:
    """标签查询"""
    label_name: str  # labels/<label_name>/ 目录名
    start_date: TradeDate
    end_date: TradeDate
    universe: Sequence[TSCode] | None = None

@dataclass(frozen=True, slots=True)
class PanelFrame:
    """特征面板（DataFrame 包装，附加元数据）"""
    df: pd.DataFrame  # index = (trade_date, ts_code)，columns = 特征列
    manifest_version: str  # manifest 内容指纹（校验一致性）

    def validate_schema(self, expected_columns: Sequence[str]) -> None:
        """校验列集合（缺列硬报错）"""
        ...

    def available_columns_at(self, date: TradeDate) -> Sequence[str]:
        """返回指定日期的可用列集（v0.4，方案 M1.5 样本起点扩展）。

        依据 manifest 的 `available_from`（列级可用起点）：仅返回 available_from <= date 的列。
        分时段变列集训练用——2005~2011 段只返回当时已可用的列。
        幂等：是（只读）。
        """
        ...

# ========== 信号与模型 ==========

@dataclass(frozen=True, slots=True)
class SignalScore:
    """单股信号分数"""
    ts_code: TSCode
    score: float  # 排序分数（无概率语义）
    meta: Mapping[str, Any]  # 审计附加信息（如模型版本）

@dataclass(frozen=True, slots=True)
class RankedCandidates:
    """排序后的候选列表（信号层输出）"""
    date: TradeDate
    scores: Sequence[SignalScore]  # 按 score 降序
    sleeve_id: str

@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型配置（袖子规格的一部分）"""
    model_family: str  # 模型家族目录名
    feature_columns: Sequence[str]  # manifest 列名
    label_name: str
    hyperparams: Mapping[str, Any]  # XGBoost 参数等
    train_window_months: int
    val_window_months: int
    ensemble_seeds: Sequence[int] = ()  # 集成种子（空 = 单模型；生产 A = 3 种子，见 R4-H2）

# ========== 数据态与审计 ==========

@dataclass(frozen=True, slots=True)
class DataState:
    """数据态指纹（对账基线；代码态 + 数据态双指纹，见 R4-M2）"""
    raw_partitions: Mapping[str, str]  # 数据集名 → 最新分区指纹
    features_partition_max: TradeDate
    config_digest: str   # 配置指纹
    code_digest: str     # 代码态（git commit + 脏标记）

class HostMode(Enum):
    """宿主模式（R4-M3：枚举替代裸字符串）"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class TradingConfigLike(Protocol):
    """类型化配置协议（R4-M3：延续 TradingConfig dataclass 先例，禁止裸 Mapping）"""
    ...

@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """执行上下文（三宿主共享）"""
    mode: HostMode
    current_date: TradeDate
    data_state: DataState
    config: TradingConfigLike  # 类型化配置，禁止 Mapping[str, Any]

# ========== 假设台账条目（R5：schema 类型化，禁裸 Mapping 复发） ==========

@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """假设台账条目（append-only；schema 是 P0 冻结交付物，
    含候选池分母 / α 预算 / 顺序检验规格——字段清单在 P0 落档时定稿）"""
    hypothesis_id: str
    kind: str                      # "prereg" / "conclusion"
    payload: Mapping[str, Any]     # 预登记内容或结论内容（schema 子字段 P0 冻结）
    data_state: DataState
    registered_at: str             # ISO 时间戳
```

---

## 1. 数据面协议（common/protocols/store.py）

### 1.1 数据存储（StoreProtocol）

```python
class DataStore(Protocol):
    """数据面唯一入口（raw / normalized / features / labels / events / state 的读写）

    写入者唯一原则：features 仅构建器写、raw 仅下载器写、台账仅证据层经本接口写（R4-H5）。
    """

    # ========== 读取（无副作用） ==========

    def load_features(self, query: FeatureQuery) -> PanelFrame:
        """加载特征面板（唯一读取入口，manifest 校验列名合法性与 available_from）

        幂等：是
        副作用：无
        失败：列名未登记或 available_from 越界 ⇒ ValueError
        """
        ...

    def load_labels(self, query: LabelQuery) -> pd.DataFrame:
        """加载标签表

        返回：index = (trade_date, ts_code)，columns = [label_value, maturity_status]
        """
        ...

    def load_events(
        self,
        event_type: str,
        start_date: TradeDate,
        end_date: TradeDate,
        universe: Sequence[TSCode] | None = None,
    ) -> pd.DataFrame:
        """加载事件表（去重对齐后）

        返回：columns = [event_type, ts_code, event_date, knowledge_date, revision_key, ...]
        """
        ...

    def load_market_state(self, date: TradeDate) -> pd.DataFrame:
        """加载市场状态（波动 / 流动性 / 宽度 / 风格动量）

        返回：单行 DataFrame（index = date）
        """
        ...

    def load_raw(
        self,
        source: str,
        dataset: str,
        start_date: TradeDate,
        end_date: TradeDate,
    ) -> pd.DataFrame:
        """加载原始数据（只读，normalized 构建用）"""
        ...

    # ========== 写入（仅授权写入方调用；防冲突语义，见 §8 矩阵） ==========

    def append_features(self, date: TradeDate, group: str, df: pd.DataFrame) -> None:
        """追加特征分区（两阶段提交：临时文件 → 原子纳入 manifest）

        防冲突：同日同 group 重复写 ⇒ 内容指纹一致才放行，不一致报错（非幂等语义——
        幂等的定义是重复执行无害且成功，此处是显式冲突检测，见 R4-M8）
        副作用：写磁盘 + 更新 manifest
        失败：列名未登记 / 分区已封存 ⇒ RuntimeError
        """
        ...

    def append_labels(self, label_name: str, df: pd.DataFrame) -> None:
        """追加标签（封存前幂等可重写）"""
        ...

    def append_events(self, event_type: str, df: pd.DataFrame) -> None:
        """追加事件（按 event_date 年分区）"""
        ...

    def append_ledger_entry(self, entry: LedgerEntry) -> None:
        """假设台账写入（R4-H5：台账是 append-only 登记资产，写入统一走数据面；
        evidence 层只消费本 API，不直接写文件——依赖方向由此干净。
        R5：参数类型化为 LedgerEntry，禁止裸 Mapping。）
        """
        ...

    def get_manifest(self) -> Mapping[str, Any]:
        """获取 manifest 快照（只读；返回 MappingProxyType）"""
        ...
```

### 1.2 特征构建（BuilderProtocol）

```python
class FeatureBuilder(Protocol):
    """特征构建器（调度 factors/ 物化列族）"""

    def build_daily(
        self,
        date: TradeDate,
        groups: Sequence[str],
        store: DataStore,
    ) -> None:
        """构建指定日的特征分区

        防冲突：重复调用 ⇒ 内容指纹一致才放行（同 append_features 语义）
        副作用：调用 store.append_features
        依赖：normalized 数据 + factors/ 计算函数
        """
        ...

    def backfill(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        groups: Sequence[str],
        store: DataStore,
    ) -> None:
        """历史回填（冷热分层后写冷区；按日分区幂等可重入，重跑即续传）

        副作用：批量写分区
        """
        ...
```

---

## 2. 信号层协议（common/protocols/signal.py）

### 2.1 信号生成（SignalProtocol）

```python
class Signal(Protocol):
    """信号生成（袖子核心接口，零互 import 保证）"""

    @property
    def sleeve_id(self) -> str:
        """袖子唯一 ID（注册表登记）"""
        ...

    @property
    def holding_period_days(self) -> int:
        """信号持有期（决策层用于到期判定）

        MVP 语义：固定值（袖子 A = 20）。
        扩展位（R4-M5）：方案 §3.4「持有期由信号衰减速度决定」的动态语义（信号生命周期）
        留给未来 `holding_period_for(ts_code, date) -> int` 方法，MVP 不实现。
        """
        ...

    def generate_ranked(
        self,
        date: TradeDate,
        features: PanelFrame,
        context: ExecutionContext,
    ) -> RankedCandidates:
        """生成当日候选排序

        幂等：是（同输入 ⇒ 同输出，模型推理确定性）
        副作用：无
        失败：特征缺失 ⇒ ValueError
        """
        ...
```

### 2.2 训练（TrainProtocol，R4-H2 修正）

```python
class Trainer(Protocol):
    """训练引擎（只管训练与注册；WF 编排不在此协议）

    集成语义（写死）：train_fold 的注册产物可以是集成——
    注册表把集成视为单一版本（metadata 记子模型列表），
    禁止「一个版本号 secret 指向集成」的类型撒谎，也禁止退化为单模型（改变生产行为）。
    """

    def train_fold(
        self,
        config: ModelConfig,  # ensemble_seeds 非空即集成
        train_start: TradeDate,
        train_end: TradeDate,
        val_start: TradeDate,
        val_end: TradeDate,
        store: DataStore,
    ) -> str:
        """训练单折（可为集成）

        返回：模型版本号（ModelRegistry 注册；集成 = 单一注册版本 + 子模型 metadata）
        副作用：写模型文件 + 注册表
        """
        ...

class WalkForwardOrchestrator(Protocol):
    """WF 编排器（训练 → OOS 评估 → OOS 回测 → data_state 快照 → summary）

    独立于 Trainer 的理由（R4-H2）：现实 walk_forward 是编排体（ml/walk_forward/runner.py），
    P3 验收对象（分级验收同 P2a）需要评估 / 回测联动才有意义；
    Trainer 保持纯训练语义，编排归本协议（或宿主实现）。
    """

    def run(
        self,
        config: ModelConfig,
        split_count: int,
        final_date: TradeDate,
        trainer: Trainer,
        store: DataStore,
    ) -> str:
        """运行滚动训练全流程

        返回：wf_run_id（批次标识，产物落 runs/<wf_run_id>/）
        """
        ...
```

---

## 3. 核心层协议（common/protocols/core.py）

**边界定位声明（R4-H3，写死）**：本节协议是**模块间公共契约**。内核内部管线——多源信号收集（每日 15 个有序钩子：到期 / 止损 / 止盈 / 条件卖出 / 减仓 / 回补 / 补齐 / 调仓…）、指令按 (action, ts_code) 去重合并（既有指令优先）、延迟订单队列——**不在协议层表达**，属 `core/execution` 内部结构。协议只暴露每日判定的输入输出边界。

**延迟订单最小语义（跨模块可观测，故登记于此）**：
- 重试上限（max_retry）与过期规则（超 rebalance_freq×0.5 丢弃）由内核统一维护；
- **特例（写死）**：减仓 / 回补指令**不进延迟队列**（该队列按整仓卖出语义重试，会把减仓放大成清仓）——跳过并由次日每日判定重试。

### 3.1 决策（DecisionProtocol，R4-M6 修正）

```python
class DecisionMaker(Protocol):
    """决策层（合并后的目标组合 → 指令）

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
        """生成当日交易指令（T0 判定）

        幂等：是（同输入 ⇒ 同输出）
        副作用：无
        """
        ...

    def apply_constraints(
        self,
        target: TargetPortfolio,
        current: Account,
    ) -> TargetPortfolio:
        """应用组合层硬约束（市值分位 / 行业上限 / 个股权重 / 换手迟滞）

        幂等：是
        """
        ...
```

### 3.2 执行（ExecutionProtocol）

```python
class Executor(Protocol):
    """执行层（T0/T1 链路、成交口径、成本）

    语义边界（R4-H3）：本协议只表达「T0 接收指令集 → T+1 产出成交」的每日边界；
    钩子编排 / 合并去重 / 延迟队列属内核内部（见本节首部声明）。
    """

    def execute_daily(
        self,
        date: TradeDate,
        orders: Sequence[Order],
        account: Account,
        context: ExecutionContext,
    ) -> Sequence[Fill]:
        """每日执行（T0 判定、T+1 开盘价成交；顺延语义由内核延迟队列处理）

        形态说明（R5）：本方法在回测宿主内**跨两天**——内部两阶段
        （T0 收单 / T+1 开盘成交）由引擎连续驱动；纸面宿主将其**分别暴露**为
        run_t0 / run_t1（T+1 成交来自人工回填而非引擎撮合）。
        两种形态是同一执行语义的宿主适配，不是两种执行语义。

        幂等：否（改变账本状态）
        副作用：生成成交记录（T+1 实际成交时入账）
        失败：涨跌停 / 停牌 ⇒ 顺延（内部登记延迟订单；减仓 / 回补特例不进队列）
        """
        ...
```

### 3.3 账本（AccountingProtocol，R4-M7 修正）

```python
class Ledger(Protocol):
    """账本（物理 + 虚拟）

    R4-M7：净额化分摊规则是「无开关的确定性算法」，**内聚到本层**（注册的确定性算法），
    禁止由调用点传入比例参数（不同调用点传不同比例 = 开关化后门）。
    """

    def get_account(self) -> Account:
        """获取物理账户快照（只读）"""
        ...

    def get_virtual_account(self, sleeve_id: str) -> VirtualAccount:
        """获取袖子虚拟账本（只读）"""
        ...

    def apply_fills(self, fills: Sequence[Fill]) -> None:
        """应用成交到物理 + 虚拟账本（分摊规则由本层注册的确定性算法决定）

        副作用：更新账本状态
        校验：三恒等式（Σ袖子虚拟{现金, 持仓股数, NAV} ≡ 物理账本），破则 RuntimeError
        """
        ...

    def snapshot(self, date: TradeDate) -> None:
        """落盘当日快照"""
        ...
```

### 3.4 组合合并（MergerProtocol，P6.5 起）

```python
class FundScheduler(Protocol):
    """跨袖资金调度器（v0.4 新增，方案 §4.9 软隔离+临时借用的协议表达；P6.5 前置冻结）。

    核心语义：物理现金池唯一；各袖有名义配额（决定常规下单上限）；
    快袖可借用慢袖当日闲置额度（记虚拟占用条目 VirtualAccount.borrowed_credit）；
    被借方到调仓日必须次日 T+1 归还，禁止滚动拖欠。
    **归还手段（v0.6 对齐方案 §4.9 v1.10）**：归还 = 借入方**放弃自身新下单**（只压下单额度、
    不动存量持仓）；**被动卖出默认禁止**（OrderReason 无对应枚举，实现侧不存在该路径）。
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
        **被动卖出默认禁止 ⇒ 当前版本恒返回空序列**（返回类型保留，为未来枚举登记后的
        极端情形留白）；禁止滚动拖欠。

        返回：归还相关指令（本期恒为空；归还经由额度压减体现，不产生卖出单）。
        副作用：清平到期占用条目。
        """
        ...

class PortfolioMerger(Protocol):
    """组合合并器（多袖目标 → 单一目标表）

    单袖退化 = 直通（MVP 形态；P2b 默认关接入，P6.5 虚拟子袖对账验收）。
    """

    def merge(
        self,
        targets: Mapping[str, TargetPortfolio],  # sleeve_id → 目标
        current: Account,
    ) -> TargetPortfolio:
        """合并多袖目标（净额化 + 集中度约束）

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
        """跨袖净额化（同股买卖抵消）

        构造性对账三件套：
        ① A 卖 X + B 买 X ⇒ 净额零换手
        ② N 袖同股 ⇒ 分摊守恒
        ③ A 自拆两个虚拟子袖 ⇒ 与单袖逐位一致
        """
        ...
```

---

## 4. 宿主层协议（common/protocols/hosts.py，R4-M6/H6 修正）

```python
class BacktestHost(Protocol):
    """回测宿主（薄适配器，只提供时钟 / 数据 / 成交通道）"""

    def run(
        self,
        start_date: TradeDate,
        end_date: TradeDate,
        sleeves: Mapping[str, Signal],
        merger: PortfolioMerger,      # R4-M6：合并器进编排链（单袖时代 = 直通实现）
        decision_maker: DecisionMaker,
        executor: Executor,
        ledger: Ledger,
        store: DataStore,
    ) -> None:
        """运行回测（驱动每日循环：信号 → 合并 → 决策 → 执行 → 入账 → 快照）

        副作用：写账本快照 + 成交记录
        产出：runs/<批ID>/ 目录
        """
        ...

class PaperHost(Protocol):
    """纸面宿主（真实时间驱动）"""

    def run_t0(self, date: TradeDate) -> Sequence[Order]:
        """T0 判定（收盘后生成次日指令）

        产出：钉钉推送 + 指令文件
        """
        ...

    def run_t1(self, date: TradeDate, fills: Sequence[Fill]) -> None:
        """T+1 执行（手工回填实际成交）

        副作用：更新账本
        """
        ...
```

---

## 5. 证据层协议（common/protocols/evidence.py）

### 5.1 制度重排（ResamplingProtocol）

```python
class RegimeResampler(Protocol):
    """制度重排（配对差分唯一裁决口径）"""

    def paired_bootstrap(
        self,
        arm_a_returns: Sequence[pd.Series],  # 逐折日收益序列
        arm_b_returns: Sequence[pd.Series],
        n_bootstrap: int,
        seed: int,
    ) -> pd.DataFrame:
        """配对重排差分

        返回：columns = [delta_maxdd, delta_cagr, delta_sharpe]，index = bootstrap 样本
        幂等：是（同种子 ⇒ 同结果）
        """
        ...

    def single_arm_distribution(
        self,
        arm_returns: Sequence[pd.Series],
        n_bootstrap: int,
        seed: int,
    ) -> pd.DataFrame:
        """单臂重排分布（非配对口径，附录展示专用；禁止用于两臂裁决）"""
        ...

    def power_calibration(
        self,
        baseline_returns: Sequence[pd.Series],
        effect_sizes: Sequence[float],  # [0.01, 0.02, 0.05]（+1/+2/+5pp）
        n_bootstrap: int,
        seed: int,  # R4-M8：补 seed（「同种子⇒同结果」的签名依据）
    ) -> pd.DataFrame:
        """功效标定曲线（配对口径产出）

        返回：index = effect_size，columns = [detection_prob, ci_lower, ci_upper]
        """
        ...
```

### 5.2 假设台账（HypothesisLedgerProtocol，R4-H5 修正）

```python
class HypothesisLedger(Protocol):
    """假设台账（机器可读证伪清单）

    写入路径（R4-H5）：register / conclude 内部经 DataStore.append_ledger_entry 落盘——
    台账是 append-only 登记资产，写入统一走数据面，evidence 层不直接写文件。
    """

    def register(
        self,
        entry: LedgerEntry,  # kind="prereg"（含候选池分母 / α 预算）
        store: DataStore,
    ) -> None:
        """预登记（经 DataStore.append_ledger_entry 落盘）"""
        ...

    def conclude(
        self,
        entry: LedgerEntry,  # kind="conclusion"
        store: DataStore,
    ) -> None:
        """结论登记"""
        ...

    def query(self, status: str | None = None) -> pd.DataFrame:
        """查询台账（只读）"""
        ...
```

---

## 6. 配置与注册（common/registry.py）

```python
class SleeveRegistry(Protocol):
    """袖子注册表（封闭，未知袖子 fail-fast）

    封闭性保证（R4 轻微 4）：register 公开但**注册文件唯一**（sleeves/_registry.py），
    静态检查禁止其他位置调用；现实先例 = universe/domains.py。
    """

    def register(self, sleeve_id: str, signal_class: type[Signal]) -> None:
        """注册袖子（仅允许在注册文件中调用）"""
        ...

    def get(self, sleeve_id: str) -> type[Signal]:
        """获取袖子类"""
        ...

    def list_all(self) -> Sequence[str]:
        """列出全部已注册袖子"""
        ...
```

**跨端共享纯函数登记（R4 轻微 6）**：`sell_rules`（到期判定阈值）与 `stagger`（分批排期）属「禁止任何一侧自行重算」契约 ⇒ 归属 `common/rules/` 共享纯函数模块，回测与纸面同源码调用；协议层无独立接口，在此登记归属。

---

## 7. 协议依赖方向（编译期可校验，R4-H6 修正）

**全部 Protocol 定义上移 `common/protocols/`**（各包只放实现）——宿主与核心层只依赖 common，袖子互不 import 由 sleeves 包内规则约束：

```python
# 允许的依赖方向（mypy / import-linter 可强制）
common.types → （无依赖）
common.protocols → common.types
store → common.types, common.protocols, factors（计算函数）
sleeves → common.types, common.protocols, store（只读接口）
core → common.types, common.protocols, store, sleeves（经注册表）
hosts → common.types, common.protocols, core, store
evidence → common.types, common.protocols, store（写入走 DataStore API）
train → common.types, common.protocols, store

# 禁止的依赖（静态检查报错）
store -/-> core
core -/-> hosts
sleeves -/-> sleeves（互 import）
evidence -/-> （直接写文件；台账经 DataStore）

# 注：runs/ 是数据目录不是模块，不属于依赖图（R4-H6 修正：从依赖表移除）
```

---

## 8. 行为语义矩阵（R4-M8 修正：「幂等」与「防冲突」分列）

幂等 = 重复执行无害且成功；**防冲突** = 重复执行时校验一致性、不一致显式报错（非幂等）。

| 协议 | 方法 | 行为语义 | 副作用 | 失败语义 |
|---|---|---|---|---|
| DataStore | load_* | 幂等 | 无 | ValueError（查询非法） |
| DataStore | append_features / append_events | **防冲突**（指纹一致才放行） | 写磁盘 | RuntimeError（分区封存 / 指纹冲突） |
| DataStore | append_labels | 幂等（封存前可重写） | 写磁盘 | RuntimeError（分区已封存） |
| DataStore | append_ledger_entry | 防冲突（hypothesis_id 唯一） | 写磁盘 | ValueError |
| Signal | generate_ranked | 幂等 | 无 | ValueError（特征缺失） |
| DecisionMaker | decide | 幂等 | 无 | — |
| Executor | execute_daily | 非幂等 | 账本状态 | 顺延（延迟订单；减仓/回补不进队列） |
| Ledger | apply_fills | 非幂等 | 账本状态 | RuntimeError（三恒等式破） |
| PortfolioMerger | merge / net_across_sleeves | 幂等 | 无 | — |
| RegimeResampler | paired_bootstrap 等 | 幂等（同 seed） | 无 | — |
| HypothesisLedger | register / conclude | 防冲突 | 经 DataStore 写盘 | ValueError（重复登记） |

---

## 9. P0 落档清单

- [x] 本文档转正为 `docs/contracts/protocols.md`；
- [x] `src/lazybull/v2/common/types.py` 创建（公共类型；含 `TradeDate.from_str` 边界单点转换）——**路径勘误（v0.6）**：迁移期 v2 代码统一落 `src/lazybull/v2/` 命名空间（v1 的 common 在用、不可混入），P4 全切换后评估展平；
- [ ] `src/lazybull/v2/common/protocols/` 创建（全部协议定义集中于此，R4-H6；同上 v2 命名空间约定）；
- [ ] mypy 配置开启 `strict` 模式 + import-linter 依赖方向校验脚本；
- [ ] 每个协议的**参考实现**（`NullImplementation`，返回空结果）用于测试替身；
- [ ] 构造性对账三件套测试用例（净额化 / 三恒等式 / 虚拟子袖逐位一致）；
- [ ] Mapping 字段以 `MappingProxyType` / frozendict 实现的约定进编码规范（R4 轻微 2）。

---

## 10. 已知留白（P0 后补）

1. **分钟线协议**：`MinuteBar` / `TickData` 值对象 + `load_intraday` 方法（试数据通过后补）；
2. **实盘宿主**：`LiveHostProtocol` + `ExactMoney` 落地（券商接口就绪后补）；
3. **暴露政策协议**：若 P1.5 裁决保留，需 `ExposurePolicyProtocol`（含 λ 序列对账口径）；
4. **候选生成器 E0**：`CandidateGeneratorProtocol`（体检 / 诊断工具的 v2 接口）；
5. **LedgerEntry.payload 类型化**（P0 评审 2-C）：当前仍为 `Mapping[str, Any]`（v0.3 版本行注释「schema 子字段 P0 冻结」已过期——假设台账 schema F1 §3 已冻结字段级）；编码期升级为 `PreregPayload | ConclusionPayload`（或 TypedDict），本行登记欠账。

---

## 11. 协议 ↔ 旧引擎语义对照表（R4-P0 新增；**P0 已逐项打钩**，2026-10-01）

> 用途：确认现实引擎的每一类语义在协议层有归属（协议表达 / 内核内部声明 / 显式不支持），
> 防止协议成为 P2a 的第一批对账失败点。输入 = R1 侦察事实。
> **非穷尽声明（R5）**：本表按当前侦察深度列出；P0 已按 R1 侦察报告全文逐项核对补全。
> **打钩依据（P0，2026-10-01）**：「现实位置」列经代码抽查逐行核实——关键符号 / 文件全部定位
> （`keep_buy_date` / `pending_condition_sells` / `_normalize_signals` / `_kelly_weights`（回测与纸面各一份包装已核实）/
> `ensemble_seeds` / `stop_loss_checker`（sell_execution.py 单实现 + paper/runtime.py 复用）/ `is_holding_period_exit_due`（trading/sell_rules.py 单源）/
> `min_commission` + `stamp_tax`（common/cost.py）/ `EXPOSURE_TRIM_TOLERANCE` / `TradingConfig`（common/trading_config.py）/
> `holdings_snapshot.py` / `ml/walk_forward/data_state.py` / 延迟订单（backtest/pending_execution.py + 纸面 storage/queue.py 两套）等）；
> 「协议层归属」列经协议全文核对（类型 / 协议 / 边界声明逐项存在）。
> **核实代码态（v0.7 补登记，P0 评审 2-A）**：2026-10-01，git `b9d866e`（与基线双臂同代）；
> 迁移期「行为冻结」下风险可控，但 **P2a 开工前必须补一次「§11 快速复核」清单项**（距核实可能数月）。
> **「内核内部」语义对账清单化（v0.7，P0 评审 2-B）**：§11 归属「内核内部」的 5 类语义
> （每日 15 个有序钩子 / 指令 (action,ts_code) 去重合并 / 延迟订单队列 / 条件卖出队列 / 减仓回补不进队列）
> 单列成 **P2a 对账用例分片**——每条内部语义一条对账断言，并入方案 §9.1 对账门检查清单
> （协议层无形式化接口，行为等价性只能靠对账门兑底，内部漂移难以定位 ⇒ 显式清单化）。

| 旧引擎语义 | 现实位置（R1 侦察） | 协议层归属 | 确认 |
|---|---|---|---|
| T+1 开盘复权价成交（浮点） | backtest 引擎 / cost.py | `Executor.execute_daily` + `Price(float)`（H1 已修） | ✅ |
| **估值价格回退链（R5 补，三端合一最重要漂移点之一）** | 回测 4 处 + 纸面前收回退，**两套平行实现** | 内核估值模块统一实现；`Ledger` / 快照经同一入口取价 | ✅ |
| 每日 15 个有序钩子（到期/止损/止盈/条件卖出/减仓/回补/补齐/调仓…） | 回测每日循环 | **内核内部**（§3 边界声明） | ✅ |
| 指令按 (action, ts_code) 去重合并（既有指令优先） | 执行链路 | **内核内部**（§3 边界声明） | ✅ |
| 延迟订单队列（max_retry / 超 rebalance_freq×0.5 过期） | 延迟订单模块 ×2 | **内核内部**；最小语义已登记（§3） | ✅ |
| 减仓 / 回补不进延迟队列（防放大成清仓） | 暴露政策链路 | **协议注释写死**（§3 特例） | ✅ |
| 条件卖出队列（pending_condition_sells） | 回测每日循环 | **内核内部**；`OrderReason.CONDITION_SELL`（R5 补） | ✅ |
| 3 种子集成（ensemble_seeds 42/61/82） | EnsembleSignal / 注册表 | `ModelConfig.ensemble_seeds` + `Trainer` 集成语义（H2 已修） | ✅ |
| walk_forward 编排（训练→评估→回测→data_state→summary） | ml/walk_forward/runner.py | `WalkForwardOrchestrator`（H2 已修） | ✅ |
| 到期判定（持有 rebalance_freq-1 日 T0 生成卖出） | trading/sell_rules.py | `common/rules/` 共享纯函数（§6 登记） | ✅ |
| **止损检查器（stop_loss_checker，单实现两侧共用）** | risk/ 止损模块 | **双层归属（v0.7，P0 评审 1-A，对齐方案 §4.3 v1.11）**：纯判定函数 → `common/rules/` 共享纯函数；执行链调用编排 → `core/decision` | ✅ |
| **止盈检查器（take_profit，v0.7 补行，P0 评审 1-A）** | risk/ 止盈模块 | 同上：纯函数 → `common/rules/`；编排 → `core/decision` | ✅ |
| 分批调仓排期（stagger_tranches） | trading/stagger.py | `common/rules/` 共享纯函数（§6 登记） | ✅ |
| **分批排期锚定差异（回测区间起点 vs 纸面 anchor 重建，R5 补）** | stagger 两侧调用点 | `common/rules/` 共享纯函数 + **锚定语义入参化**（anchor 由调用方显式传入，禁止各自推导） | ✅ |
| **Kelly 仓位 / 权重归一化两份包装（`_kelly_weights` / `_normalize_signals` 回测与纸面各一份，R5 补）** | trading/ 纯函数已单源 + **包装层两份** | 纯函数归 `common/rules/`；**包装层合并为 `core/decision` 唯一实现** | ✅ |
| 回补不重置买入日 + 补齐算新持仓（A3） | 账本层 | `Lot` / `Position.lots`（H4 已修） | ✅ |
| 净额化到期归属（最早买入虚拟份额优先） | v2 新增 | `VirtualPosition.lots`（H4 已修，P6.5 前定型） | ✅ |
| 代码态 + 数据态双指纹 | ml/walk_forward/data_state.py | `DataState.code_digest`（M2 已修） | ✅ |
| TradingConfig 类型化配置 | common/（dataclass） | `ExecutionContext.config: TradingConfigLike`（M3 已修） | ✅ |
| 滑点 / 执行缺口归因 | 执行归因链路 | **执行归因层**（非 Fill 字段，M1 已修） | ✅ |
| 最低佣金 5 元 / 印花税单边 | cost.py | 内核成本模块（`Money` 浮点路径覆盖） | ✅ |
| 假设台账 append-only 写入 | v2 新增 | `DataStore.append_ledger_entry(LedgerEntry)`（H5+R5 已修） | ✅ |
| **持仓快照（holdings_snapshot，只读旁路，R5 补）** | backtest/holdings_snapshot.py（引擎默认关闭、WF OOS 显式开启） | **内核内部旁路**；产物 schema 归《runs 产物契约》，不进协议层 | ✅ |

"""P6.5 T4 自拆双臂装配：单袖参照臂 vs 0.5/0.5 虚拟双子袖（合成驱动）。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §8.4
  （自拆一致：两臂初始物理现金 100,000、空仓；组合配额 4,000；D1 价 10、D2 价 11、
  滑点 0；参照臂 100% 配额、自拆臂 2,000/2,000，**子目标先合并再物理整手**）
  与 §10.1 G4（两臂物理隔离根、写前不存在、经济工件逐位无差异）；
- ``docs/contracts/netting_freeze.md`` F25 §4/§5、``docs/contracts/protocols.md``
  F13 §3.3/§3.4；
- T3 交付的 ``hosts/order_chain.py``（真实 Order → 执行器 → ``Ledger.apply_fills``
  → ``InternalSettlementBatch`` → ``VirtualLedger.apply_settlement``）与
  ``hosts/virtual_arm_runs.py::DualArmRunsWriter``（隔离根 + generation 身份）。

职责与边界：
- 每臂独立构造宿主、虚拟账本与物理 lot 伴随视图，**物理成交/费用/现金只由执行链
  产出**：本模块不写任何"两臂相同"的数值，也不手工注入成交事实（预演成交由
  ``order_chain.predict_fills`` 从执行器同一 sizing 实现推导）；
- 两臂目标都由 ``target_synthesis.synthesize_lot_targets`` 从**同一组合配额**派生：
  参照臂 100% 配额、自拆臂按固定 0.5/0.5 拆分；**先合并母配额再整手**，因此两臂
  物理目标相同（400 股），但子袖目标（200/200）不要求等于单袖（400）；
- 逐日 ``_finalize_day`` 挂载 I1/I2/I3（含无成交的 D0 与只估值的 D2），并把引擎真实
  产物（净值曲线、成交日志、持仓快照）投影为 runs 契约结果帧；runs 由
  ``DualArmRunsWriter`` 写出到两个**兄弟根**，写前断言 ``out_dir`` 不存在；
- ``bt_*`` 指标复用 ``hosts/backtest/replay.py::_compute_replay_metrics`` 的唯一实现
  （**登记**：只读导入，``replay.py`` 零改动），禁止第二套绩效公式；
- 本模块**不**比较两臂（比较在 ``v2/evidence/arm_comparator.py``），也不进入生产
  默认路径（``hosts/backtest/runtime.py`` 与 ``PaperHost`` 不变）；不实现 T5 的
  状态信封恢复、卖出（S5 归 P7）或证据包汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.accounting.g4_nav import (
    G4NavEvidence,
    assert_g4_nav_bitwise_equal,
    float64_bits,
    g4_merged_nav,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav, generate_nav_curve
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualSleeveState,
)
from src.lazybull.v2.core.decision.target_synthesis import synthesize_lot_targets
from src.lazybull.v2.core.decision.virtual_merger import MultiSleevePortfolioMerger
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.order_chain import OrderChainResult
from src.lazybull.v2.hosts.virtual_arm_runs import (
    ARM_SINGLE,
    ARM_SPLIT,
    ArmGeneration,
    DualArmRunsWriter,
    remove_new_tree_paths,
    snapshot_tree_paths,
)
from src.lazybull.v2.hosts.virtual_native_engine import VirtualNativeMergeBacktestEngineML

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"
#: 参照臂唯一子袖 ID（不同构造标签，不与自拆臂子袖混用）
SINGLE_SLEEVE_ID = "solo"
#: 自拆臂两个纯运行时子袖
SPLIT_SLEEVE_IDS: Tuple[str, str] = ("A", "B")
#: 合成费率（设计 §8.1 冻结：全部算例同一费率，不读真实配置）
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.0
#: 合成模型版本（两臂共用平凡身份，不构成经济证据）
MODEL_VERSION = 6501


class SelfSplitArmError(ValueError):
    """自拆双臂装配输入非法（**零副作用**拒绝）。"""


def _error(message: str) -> Exception:
    return SelfSplitArmError(f"阶段={STAGE} 自拆双臂 {message}")


def synthetic_cost_model() -> CostModel:
    """三件套统一合成费率（佣金 0.0003 / 最低 5 元、印花税 0.0005、滑点 0）。"""
    return CostModel(
        commission_rate=COMMISSION_RATE,
        min_commission=MIN_COMMISSION,
        stamp_tax=STAMP_TAX,
        slippage=SLIPPAGE,
    )


@dataclass(frozen=True, slots=True)
class SelfSplitScenario:
    """自拆一致算例的显式合成输入（默认值即设计 §8.4 冻结参数）。"""

    #: 交易日序列：D0（期初估值日、无成交）、D1（买入日）、D2（估值日）
    trade_dates: Tuple[pd.Timestamp, ...] = (
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-07"),
    )
    #: 逐日价格（当日同一价源；合成算例只含信号股票 X）
    prices: Tuple[Tuple[pd.Timestamp, Mapping[str, float]], ...] = (
        (pd.Timestamp("2026-01-05"), {"600000.SH": 10.0}),
        (pd.Timestamp("2026-01-06"), {"600000.SH": 10.0}),
        (pd.Timestamp("2026-01-07"), {"600000.SH": 11.0}),
    )
    #: 同一信号流（只含 X；自拆两袖共享同一信号）
    signals: Tuple[Tuple[str, float], ...] = (("600000.SH", 1.0),)
    #: 组合配额（元）：参照臂 100%、自拆臂按 split_ratio 拆分
    combo_quota: float = 4000.0
    #: 固定 0.5/0.5（唯一出口构造比例；不得事后增删）
    split_ratio: float = 0.5
    #: 两臂初始物理现金（均 100,000、均空仓）
    physical_initial_capital: float = 100000.0
    #: 成交日下标（D1）：买入只在同日发生，D2 只估值
    trade_day_index: int = 1

    def date_list(self) -> Tuple[pd.Timestamp, ...]:
        """规范化后的交易日序列（严格递增、无重复）。"""
        dates = tuple(pd.Timestamp(item) for item in self.trade_dates)
        if len(dates) < 2:
            raise _error(f"交易日至少需要 2 天（估值日 + 成交日），实际 {len(dates)}")
        if any(right <= left for left, right in zip(dates, dates[1:])):
            raise _error(f"交易日必须严格递增：{[str(item.date()) for item in dates]}")
        if not 0 <= int(self.trade_day_index) < len(dates):
            raise _error(f"成交日下标越界：{self.trade_day_index}（交易日数={len(dates)}）")
        return dates

    def price_of(self, date: pd.Timestamp, code: str) -> float:
        """当日价源（缺日/缺股即拒绝；禁止跨日顺延）。"""
        for item_date, item_prices in self.prices:
            if pd.Timestamp(item_date) == pd.Timestamp(date):
                if code in item_prices:
                    return float(item_prices[code])
        raise _error(
            f"价格源缺少 {pd.Timestamp(date).date()} 的 {code} 价格；"
            "P6.5 合成边界无价即拒绝，不顺延"
        )

    def signal_codes(self) -> Tuple[str, ...]:
        """信号股票集（顺序即插入序，不得排序改写）。

        自拆一致算例的**组合配额是单一金额**（设计 §8.4 输入只含 X）：多股票时无法
        说明组合配额如何在股票之间分摊，若把单股票股数复制到每只股票会放大母配额
        （执行链与 I1/I3 都不会拒绝）。因此这里显式拒绝，零副作用。
        """
        codes = tuple(str(code) for code, _score in self.signals)
        if not codes or any(not code.strip() for code in codes):
            raise _error("信号股票集不得为空且代码不得为空")
        if len(codes) != 1:
            raise _error(
                f"自拆一致算例只支持单股票组合配额（设计 §8.4 输入只含 1 只），"
                f"实际信号股票={list(codes)}；多股票配额分摊未定义，"
                "禁止把单股票股数复制到每只股票（会放大母配额）"
            )
        return codes

    def sleeve_values(self) -> Mapping[str, Mapping[str, float]]:
        """各臂的逐袖配额（元）：参照臂 100%；自拆臂 0.5/0.5。"""
        ratio = float(self.split_ratio)
        if not 0.0 < ratio < 1.0:
            raise _error(f"自拆比例必须在 (0,1) 内，实际 {self.split_ratio!r}")
        quota = float(self.combo_quota)
        if not quota > 0.0:
            raise _error(f"组合配额必须为正数，实际 {self.combo_quota!r}")
        return {
            ARM_SINGLE: {SINGLE_SLEEVE_ID: quota},
            ARM_SPLIT: {
                SPLIT_SLEEVE_IDS[0]: quota * ratio,
                SPLIT_SLEEVE_IDS[1]: quota * (1.0 - ratio),
            },
        }

    def signals_for_arm(self, arm: str) -> Mapping[str, Any]:
        """两臂信号身份输入（必须不同：两臂是不同构造，不得共用同一份指纹）。"""
        signals = {code: float(score) for code, score in self.signals}
        if arm == ARM_SINGLE:
            return {
                "signals": signals,
                "sleeve": SINGLE_SLEEVE_ID,
                "quota": float(self.combo_quota),
            }
        if arm == ARM_SPLIT:
            return {
                "signals": signals,
                "sleeves": {
                    SPLIT_SLEEVE_IDS[0]: float(self.split_ratio),
                    SPLIT_SLEEVE_IDS[1]: 1.0 - float(self.split_ratio),
                },
                "quota": dict(self.sleeve_values()[ARM_SPLIT]),
            }
        raise _error(f"未知臂标签 {arm!r}；合法臂={[ARM_SINGLE, ARM_SPLIT]}")


@dataclass(frozen=True, slots=True)
class ArmFillFact:
    """单笔物理成交事实（全部字段来自执行链观测；用于两臂逐位比较）。"""

    ts_code: str
    order_target: int
    requested_shares: int
    shares: int
    price: float
    fee: float
    cash_before: float
    cash_after: float
    fill_key: str
    source_lot_id: str


@dataclass(frozen=True, slots=True)
class ArmView:
    """单臂经济事实投影（物理 + 合并虚拟视图 + G4 位模式门输入）。

    ``virtual_source_lots`` 是**按来源键汇总**的整数股数（I2 口径），不是 lot 条目数；
    ``merged_total_value`` / ``merged_nav`` 由各袖货币总值**先合并再单次派生**，与 I3
    的 ``Σ 逐袖贡献 NAV`` 顺序不同（见 ``core/accounting/g4_nav.py``）。
    """

    arm: str
    sleeve_ids: Tuple[str, ...]
    settlement_dates: Tuple[str, ...]
    physical_initial_capital: float
    physical_cash: float
    physical_positions: Tuple[Tuple[str, int], ...]
    physical_total_value: float
    physical_source_lots: Tuple[Tuple[str, str, int], ...]
    fills: Tuple[ArmFillFact, ...]
    virtual_cash_by_sleeve: Tuple[Tuple[str, float], ...]
    virtual_source_lots: Tuple[Tuple[str, str, int], ...]
    virtual_total_value_by_sleeve: Tuple[Tuple[str, float], ...]
    merged_virtual_cash: float
    merged_total_value: float
    merged_nav: float
    g4_evidence: G4NavEvidence
    trades: Tuple[Mapping[str, Any], ...]
    results: Mapping[str, Any]
    #: 成交日收盘的逐袖事实 ``(袖, 现金, 股数, 货币总值, 贡献 NAV)``（黄金值 D1 行）
    trade_day_sleeve_facts: Tuple[Tuple[str, float, int, float, float], ...] = ()
    generation: Optional[ArmGeneration] = None


@dataclass(frozen=True, slots=True)
class SelfSplitAssembly:
    """两臂装配结果（物理隔离根 + 事实投影 + runs 生成身份）。"""

    scenario: SelfSplitScenario
    run_id: str
    batch_id: str
    writer: DualArmRunsWriter
    single: ArmView
    split: ArmView
    #: 两臂逐日物理总值（装配期已断言两臂逐日相同；作证据而非经济证据）
    daily_physical_values: Tuple[Tuple[str, float], ...] = ()
    #: 两臂**宿主实例回读**的生效参数（键 → 值；装配期已断言同值且等于冻结表）
    host_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def arm(self, arm: str) -> ArmView:
        """按臂标签取事实投影。"""
        if arm == ARM_SINGLE:
            return self.single
        if arm == ARM_SPLIT:
            return self.split
        raise _error(f"未知臂标签 {arm!r}；合法臂={[ARM_SINGLE, ARM_SPLIT]}")


def _price_data(scenario: SelfSplitScenario) -> pd.DataFrame:
    """引擎价格索引输入（close 为成交/估值价，close_adj 同值避免绩效价回退）。"""
    rows = []
    for date in scenario.date_list():
        for code in scenario.signal_codes():
            price = scenario.price_of(date, code)
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": pd.Timestamp(date),
                    "close": price,
                    "close_adj": price,
                    "open": price,
                    "open_adj": price,
                }
            )
    return pd.DataFrame(rows)


#: 合成驱动**显式传给宿主**的参数（冻结值；构造后从实例回读核对，见
#: ``read_host_parameters``）。缺省继承引擎默认会把"实际生效值"与证据记录分离
#: （复审 R1-01：记录 2 天持有、实际 5 天），故这里全部写死。
HOST_RUN_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "rebalance_freq": 5,
        "holding_period": 5,
        "sell_timing": "open",
        "enable_risk_budget": False,
        "enable_pending_order": True,
        "enable_position_completion": True,
        "completion_window_days": 3,
        "stagger_tranches": 1,
        "position_sizing": "equal",
        "min_buy_value_ratio": 0.0,
        "enable_early_rebalance_on_empty": True,
        "enable_trailing_early_rebalance": True,
        "kelly_vol_window": 60,
        "kelly_max_leverage": 0.25,
        "vol_window": 20,
        "vol_epsilon": 0.001,
        "max_weight_per_stock": None,
        "max_per_industry": None,
    }
)
#: 构造后**赋值**的开关（不是构造参数，但同样属于执行链实际生效参数）
HOST_ATTRIBUTE_PARAMETERS: Mapping[str, Any] = MappingProxyType({"record_holdings_snapshot": True})


def read_host_parameters(host: Any) -> Dict[str, Any]:
    """从**宿主实例**回读生效参数（构造后、执行前；唯一实际值来源）。"""
    actual: Dict[str, Any] = {}
    for key in [*HOST_RUN_PARAMETERS, *HOST_ATTRIBUTE_PARAMETERS]:
        if not hasattr(host, key):
            raise _error(f"宿主缺少参数 {key!r}；证据不得记录无法回读的参数")
        value = getattr(host, key)
        actual[key] = value if value is None or not isinstance(value, float) else float(value)
    return actual


def assert_host_parameters_match(observed: Mapping[str, Mapping[str, Any]]) -> None:
    """两臂实际参数必须**互相同值**且**等于冻结表**（写盘前 fail-closed）。"""
    arms = sorted(observed)
    if arms != [ARM_SINGLE, ARM_SPLIT]:
        raise _error(f"宿主参数核对需要两臂，实际={arms}")
    single, split = observed[ARM_SINGLE], observed[ARM_SPLIT]
    if dict(single) != dict(split):
        drift = sorted(key for key in set(single) | set(split) if single.get(key) != split.get(key))
        raise _error(f"两臂宿主参数不一致，差异键={drift}；证据不得比较不同参数下的两臂")
    frozen = {**HOST_RUN_PARAMETERS, **HOST_ATTRIBUTE_PARAMETERS}
    drift = sorted(key for key in frozen if single.get(key) != frozen[key])
    if drift:
        raise _error(
            "宿主实际参数与冻结表不符，差异键="
            f"{[(key, single.get(key), frozen[key]) for key in drift]}；"
            "证据记录的参数必须是执行链实际使用的参数"
        )


def _new_host(
    *, arm: str, scenario: SelfSplitScenario, run_id: str
) -> Tuple[VirtualNativeMergeBacktestEngineML, VirtualLedger]:
    """构造单臂宿主（独立账本与伴随视图；两臂互不共享可变对象）。"""
    cost_model = synthetic_cost_model()
    if arm == ARM_SINGLE:
        merger = MultiSleevePortfolioMerger((SINGLE_SLEEVE_ID,), {SINGLE_SLEEVE_ID: 1.0})
        sleeves: Tuple[str, ...] = (SINGLE_SLEEVE_ID,)
    elif arm == ARM_SPLIT:
        merger = MultiSleevePortfolioMerger(
            SPLIT_SLEEVE_IDS, {SPLIT_SLEEVE_IDS[0]: 0.5, SPLIT_SLEEVE_IDS[1]: 0.5}
        )
        sleeves = SPLIT_SLEEVE_IDS
    else:
        raise _error(f"未知臂标签 {arm!r}；合法臂={[ARM_SINGLE, ARM_SPLIT]}")
    ledger = VirtualLedger(cost_model=cost_model)
    per_sleeve_cash = float(scenario.physical_initial_capital) / float(len(sleeves))
    for sleeve_id in sleeves:
        ledger.open_account(sleeve_id, per_sleeve_cash)
    codes = list(scenario.signal_codes())
    universe = BasicUniverse(
        pd.DataFrame({"ts_code": codes, "name": codes, "industry": ["I"] * len(codes)}),
        verbose=False,
    )
    host = VirtualNativeMergeBacktestEngineML(
        features_by_date={},
        universe=universe,
        signal=EqualWeightSignal(top_n=len(codes)),
        initial_capital=float(scenario.physical_initial_capital),
        cost_model=cost_model,
        verbose=False,
        portfolio_merger=merger,
        virtual_ledger=ledger,
        physical_lot_view=PhysicalLotView(),
        price_source_provider=None,
        run_id=f"{run_id}:{arm}",
        **dict(HOST_RUN_PARAMETERS),
    )
    host.current_capital = float(scenario.physical_initial_capital)
    host.positions = {}
    for key, value in HOST_ATTRIBUTE_PARAMETERS.items():
        setattr(host, key, value)
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(_price_data(scenario))
    return host, ledger


def _targets_for_price(
    arm: str, scenario: SelfSplitScenario, price: float
) -> Dict[str, Dict[str, int]]:
    """由组合配额派生逐袖目标股数（先合并母配额再整手；两臂同一配额源）。"""
    values = scenario.sleeve_values()[arm]
    synthesis = synthesize_lot_targets(values, price=price, label=f"臂={arm}")
    return {
        sleeve_id: {code: synthesis.shares_of(sleeve_id) for code in scenario.signal_codes()}
        for sleeve_id in sorted(values)
    }


def _fill_facts(result: OrderChainResult) -> Tuple[ArmFillFact, ...]:
    """执行链观测 → 成交事实（股数/价/费/来源 lot/事件键全部取自观测）。"""
    return tuple(
        ArmFillFact(
            ts_code=fill.ts_code,
            order_target=int(fill.order_target),
            requested_shares=int(fill.requested_shares),
            shares=int(fill.shares),
            price=float(fill.price),
            fee=float(fill.fee),
            cash_before=float(fill.cash_before),
            cash_after=float(fill.cash_after),
            fill_key=str(fill.fill_key),
            source_lot_id=str(fill.source_lot_id),
        )
        for fill in result.observed
    )


def _source_lots(view: PhysicalLotView) -> Tuple[Tuple[str, str, int], ...]:
    """物理来源 lot 的整数股数（排序后作证据）。"""
    return tuple(
        (code, lot_id, int(shares)) for (code, lot_id), shares in sorted(view.aggregate().items())
    )


def _virtual_source_lots(
    states: Mapping[str, VirtualSleeveState],
) -> Tuple[Tuple[str, str, int], ...]:
    """按来源键汇总的虚拟整数股数（I2 口径；不比 lot 条目数）。"""
    totals: Dict[Tuple[str, str], int] = {}
    for sleeve_id in sorted(states):
        for lot in states[sleeve_id].lots:
            key = (lot.ts_code, lot.source_lot_id)
            totals[key] = totals.get(key, 0) + int(lot.shares)
    return tuple((code, lot_id, shares) for (code, lot_id), shares in sorted(totals.items()))


def _results_frame(
    *,
    host: VirtualNativeMergeBacktestEngineML,
    scenario: SelfSplitScenario,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """引擎真实产物 → runs 契约结果帧（两臂必须逐位相同）。

    指标复用 ``replay._compute_replay_metrics`` 唯一实现（只读导入，replay.py 零改动），
    避免第二套 ``bt_*`` 公式；折窗口取场景首末交易日。
    """
    from src.lazybull.v2.hosts.backtest.replay import _compute_replay_metrics

    dates = scenario.date_list()
    nav_curve = generate_nav_curve(host.portfolio_values, host.initial_capital)
    bt_start = f"{dates[0]:%Y%m%d}"
    bt_end = f"{dates[-1]:%Y%m%d}"
    metrics = _compute_replay_metrics(
        nav_curve, bt_start, bt_end, int(config.get("bt_top_n", 1)), None
    )
    trades = pd.DataFrame([dict(item) for item in host.trades])
    return {
        "split_index": 0,
        "model_version": MODEL_VERSION,
        "train_start": bt_start,
        "train_end": bt_start,
        "test_start": bt_start,
        "test_end": bt_end,
        "metrics": metrics,
        "nav_curve": nav_curve,
        "trades": trades,
        "attribution": pd.DataFrame(),
        "holdings_snapshot": host.get_holdings_snapshot(),
    }


def _arm_view(
    *,
    arm: str,
    host: VirtualNativeMergeBacktestEngineML,
    ledger: VirtualLedger,
    chain_results: Sequence[OrderChainResult],
    results_frame: Mapping[str, Any],
    scenario: SelfSplitScenario,
    price_source: PriceSource,
    context: Any,
    trade_day_sleeve_facts: Tuple[Tuple[str, float, int, float, float], ...],
) -> ArmView:
    """装配单臂事实投影（物理 + 合并虚拟视图 + G4 位模式门输入）。"""
    states = ledger.snapshot()
    merged_nav_evidence = g4_merged_nav(
        states,
        price_source,
        float(scenario.physical_initial_capital),
        context=context,
    )
    merged_cash = 0.0
    for sleeve_id in sorted(states):
        merged_cash += float(states[sleeve_id].cash)
    return ArmView(
        arm=arm,
        sleeve_ids=tuple(sorted(states)),
        settlement_dates=tuple(str(result.trade_date) for result in chain_results),
        physical_initial_capital=float(scenario.physical_initial_capital),
        physical_cash=float(host.current_capital),
        physical_positions=tuple(
            (str(code), int(info.get("shares", 0))) for code, info in sorted(host.positions.items())
        ),
        physical_total_value=float(host.portfolio_values[-1]["portfolio_value"]),
        physical_source_lots=_source_lots(host.physical_lot_view),
        fills=tuple(fill for result in chain_results for fill in _fill_facts(result)),
        virtual_cash_by_sleeve=tuple(
            (sleeve_id, float(states[sleeve_id].cash)) for sleeve_id in sorted(states)
        ),
        virtual_source_lots=_virtual_source_lots(states),
        virtual_total_value_by_sleeve=tuple(
            (sleeve_id, float(states[sleeve_id].total_value(price_source)))
            for sleeve_id in sorted(states)
        ),
        merged_virtual_cash=merged_cash,
        merged_total_value=float(merged_nav_evidence.merged_total_value),
        merged_nav=float(merged_nav_evidence.merged_nav),
        g4_evidence=merged_nav_evidence,
        trades=tuple(dict(item) for item in host.trades),
        results=results_frame,
        trade_day_sleeve_facts=trade_day_sleeve_facts,
    )


@dataclass(frozen=True, slots=True)
class _ArmRun:
    """单臂运行产物（宿主仍存活时才能取逐日净值证据）。"""

    view: ArmView
    host: VirtualNativeMergeBacktestEngineML
    ledger: VirtualLedger
    #: 构造后从实例回读的生效参数（唯一实际值来源）
    host_parameters: Mapping[str, Any] = field(default_factory=dict)


def _run_arm(
    *, arm: str, scenario: SelfSplitScenario, run_id: str, config: Mapping[str, Any]
) -> _ArmRun:
    """跑一臂的完整逐日链路（无成交日同样过恒等式挂载）。"""
    host, ledger = _new_host(arm=arm, scenario=scenario, run_id=run_id)
    dates = scenario.date_list()
    date_to_idx = {date: index for index, date in enumerate(dates)}
    trade_date = dates[int(scenario.trade_day_index)]
    # 决策日定仓：目标股数只由**决策日**价源派生（当日 100 股整手一次），此后各日
    # 沿用同一目标（D2 只估值，重提同一目标 ⇒ 零物理指令、零成交）。
    targets = _targets_for_price(
        arm, scenario, scenario.price_of(trade_date, scenario.signal_codes()[0])
    )
    chain_results: list[OrderChainResult] = []
    trade_day_facts: Tuple[Tuple[str, float, int, float, float], ...] = ()
    for index, date in enumerate(dates):
        if index >= int(scenario.trade_day_index):
            chain_results.append(
                host.execute_order_chain(date, targets, batch_id=f"{run_id}:{arm}:{date:%Y%m%d}")
            )
        portfolio_value = host._calculate_portfolio_value(date)
        host._finalize_day(
            index,
            date,
            0,
            len(dates),
            index + 1,
            portfolio_value,
            list(dates),
            date_to_idx,
        )
        if index == int(scenario.trade_day_index):
            trade_day_facts = _trade_day_facts(
                host=host, ledger=ledger, date=date, scenario=scenario
            )
    states = ledger.snapshot()
    last_date = dates[-1]
    price_source = host._build_price_source(last_date, states=states)
    context = host._identity_context(last_date, states, price_source.source_id)
    view = _arm_view(
        arm=arm,
        host=host,
        ledger=ledger,
        chain_results=chain_results,
        results_frame=_results_frame(host=host, scenario=scenario, config=config),
        scenario=scenario,
        price_source=price_source,
        context=context,
        trade_day_sleeve_facts=trade_day_facts,
    )
    return _ArmRun(view=view, host=host, ledger=ledger, host_parameters=read_host_parameters(host))


def _trade_day_facts(
    *,
    host: VirtualNativeMergeBacktestEngineML,
    ledger: VirtualLedger,
    date: pd.Timestamp,
    scenario: SelfSplitScenario,
) -> Tuple[Tuple[str, float, int, float, float], ...]:
    """成交日收盘逐袖事实 ``(袖, 现金, 股数, 货币总值, 贡献 NAV)``（黄金值 D1 行）。"""
    states = ledger.snapshot()
    source = host._build_price_source(date, states=states)
    code = scenario.signal_codes()[0]
    initial = float(scenario.physical_initial_capital)
    return tuple(
        (
            sleeve_id,
            float(states[sleeve_id].cash),
            int(states[sleeve_id].shares_of(code)),
            float(states[sleeve_id].total_value(source)),
            float(calculate_nav(states[sleeve_id].total_value(source), initial)),
        )
        for sleeve_id in sorted(states)
    )


def assert_view_constructions_distinct(single: ArmView, split: ArmView) -> None:
    """**写前**自我比较防护（只依赖内存视图）：臂标签 + 袖数必须为冻结构造。

    与 ``assert_arms_are_distinct`` 的分工：本门在写出 runs **之前**执行（失败零副作用），
    负责"视图构造"层；generation 级身份（臂标签/根/信号指纹）由后者在写出后校验，且其
    失败已被 ``run_self_split_arms`` 的回滚分支覆盖（撤销本次新建根）。两层共用同一错误
    类型与措辞前缀，比较器层（``evidence/arm_comparator.py``）直接复用本门。
    """
    if single is split:
        raise _error("两臂传入同一视图对象；自我比较防护拒绝")
    if single.arm == split.arm:
        raise _error(
            f"两臂臂标签相同（{single.arm!r}）；疑似把同一臂结果当成两臂证据"
            f"（必须为 {ARM_SINGLE!r} vs {ARM_SPLIT!r}）"
        )
    if single.arm != ARM_SINGLE or split.arm != ARM_SPLIT:
        raise _error(f"臂标签非法：{single.arm!r}/{split.arm!r}")
    if len(single.sleeve_ids) != 1 or len(split.sleeve_ids) != 2:
        raise _error(
            "两臂必须为不同构造（参照臂 1 子袖、自拆臂 2 子袖），实际 "
            f"single={list(single.sleeve_ids)} split={list(split.sleeve_ids)}"
        )


def assert_arms_are_distinct(assembly: SelfSplitAssembly) -> None:
    """自我比较防护（装配层）：两臂必须是**不同构造**，不是彼此的复制品。

    判据：视图级构造（``assert_view_constructions_distinct``）+ generation 臂标签不同
    + generation 根不同 + 信号指纹不同；任一不满足即拒绝（防止"把 A 的结果当成 B 的
    结果"）。
    """
    single, split = assembly.single, assembly.split
    assert_view_constructions_distinct(single, split)
    if single.generation is None or split.generation is None:
        raise _error("两臂 generation 未冻结，禁止比较（必须先写出并绑定身份）")
    if single.generation.arm == split.generation.arm:
        raise _error(f"两臂 generation 臂标签相同（{single.generation.arm!r}）；疑似复用同一产物")
    if single.generation.runs_root == split.generation.runs_root:
        raise _error("两臂 runs 根相同；自我比较防护拒绝")
    if single.generation.signal_fingerprint == split.generation.signal_fingerprint:
        raise _error("两臂信号指纹相同；参照臂与自拆臂必须是不同构造输入（不得用同一信号冒充自拆）")


def run_self_split_arms(
    *,
    scenario: SelfSplitScenario,
    run_id: str,
    base_dir: Path,
    writer_config: Mapping[str, Any],
    code_state: Mapping[str, Any],
    data_state: Mapping[str, Any],
    created_at: str = "1970-01-01T00:00:00",
) -> SelfSplitAssembly:
    """跑两臂全链、写出两臂 runs 并冻结 generation（不比较；比较在比较器）。

    Args:
        scenario: 合成算例输入（默认即设计 §8.4 冻结参数）
        run_id: ASCII 运行标识（进入两臂根路径与批次键）
        base_dir: 两臂公共基目录（内部使用 ``<base_dir>/<run_id>/{single,split}``）
        writer_config: 两臂共用批次配置（两臂逐键相同；臂相关值禁止写入）
        code_state/data_state: runs 血缘输入（两臂相同）
        created_at: 写入时间戳（仅进 generation 与 ``batch_meta``，属排除字段）

    Returns:
        ``SelfSplitAssembly``（两臂事实投影 + generation + 逐日物理总值）

    Raises:
        SelfSplitArmError: 输入非法/两臂构造相同。
        ArmIsolationError: 根隔离或写前存在断言失败（零覆盖）。
        OrderChainError / ReconciliationError: 全链拒绝或恒等式破缺（当日停止）。
    """
    if not str(run_id).strip():
        raise _error("run_id 不得为空")
    scenario.date_list()  # 输入自检（交易日/成交日/单股票配额）先于任何写操作
    writer = DualArmRunsWriter(run_id=run_id, base_dir=Path(base_dir), config=writer_config)
    batch_id = f"{run_id}-b0"
    # 写前全量预检：两臂都要可写之后才写第一臂（否则第二臂拒绝会留下第一臂产物）
    for arm in (ARM_SINGLE, ARM_SPLIT):
        writer.assert_writable(arm=arm, batch_id=batch_id)
    created_root = Path(base_dir).resolve() / str(run_id)
    # 写前快照：失败时只撤销本次新增的条目，**保留**运行根下既有文件（外部评审
    # T4-Codex-03：原实现只在根不存在时清理，预有根下会残留第一臂产物）。
    root_snapshot = snapshot_tree_paths(created_root)
    root_preexisted = created_root.exists()
    runs = {
        arm: _run_arm(arm=arm, scenario=scenario, run_id=run_id, config=writer_config)
        for arm in (ARM_SINGLE, ARM_SPLIT)
    }
    # 纯内存断言全部前置到**写盘之前**：任何失败都在文件系统上零副作用
    assert_view_constructions_distinct(runs[ARM_SINGLE].view, runs[ARM_SPLIT].view)
    # 宿主实际参数（构造后回读）必须两臂同值且等于冻结表：证据记录只允许实际生效值
    arm_host_parameters = {arm: dict(runs[arm].host_parameters) for arm in (ARM_SINGLE, ARM_SPLIT)}
    assert_host_parameters_match(arm_host_parameters)
    if dict(scenario.signals_for_arm(ARM_SINGLE)) == dict(scenario.signals_for_arm(ARM_SPLIT)):
        raise _error(
            "两臂信号入场输入相同；参照臂与自拆臂必须是不同构造输入" "（不得用同一信号冒充自拆）"
        )
    daily_single = _daily_values(runs[ARM_SINGLE].host)
    daily_split = _daily_values(runs[ARM_SPLIT].host)
    if daily_single != daily_split:
        raise _error(
            f"两臂逐日物理总值不同：single={daily_single} split={daily_split}；"
            "容差=0（自拆一致的最低前提）"
        )
    assert_g4_nav_bitwise_equal(
        runs[ARM_SINGLE].view.merged_nav,
        runs[ARM_SPLIT].view.merged_nav,
        context=(
            f"运行={run_id} 日期={runs[ARM_SPLIT].view.settlement_dates[-1]} "
            f"单袖N位={float64_bits(runs[ARM_SINGLE].view.merged_nav)} "
            f"合并N位={float64_bits(runs[ARM_SPLIT].view.merged_nav)}"
        ),
        label="G4 单袖 vs 自拆合并",
    )
    generations: Dict[str, ArmGeneration] = {}
    try:
        for arm in (ARM_SINGLE, ARM_SPLIT):
            generations[arm] = writer.write_arm(
                arm=arm,
                results=[dict(runs[arm].view.results)],
                batch_id=batch_id,
                code_state=dict(code_state),
                data_state=dict(data_state),
                signals=dict(scenario.signals_for_arm(arm)),
                created_at=created_at,
            )
        assembly = SelfSplitAssembly(
            scenario=scenario,
            run_id=str(run_id),
            batch_id=batch_id,
            writer=writer,
            single=replace(runs[ARM_SINGLE].view, generation=generations[ARM_SINGLE]),
            split=replace(runs[ARM_SPLIT].view, generation=generations[ARM_SPLIT]),
            daily_physical_values=daily_single,
            host_parameters=arm_host_parameters,
        )
        assert_arms_are_distinct(assembly)
    except BaseException:
        # 零副作用：只撤销本函数新增的条目；既有文件/目录原样保留（T4-Codex-03）。
        # 覆盖写入异常 **与** 写后 generation 身份断言失败（A-R3-02）。
        remove_new_tree_paths(created_root, root_snapshot, remove_root=not root_preexisted)
        raise
    return assembly


def _daily_values(host: VirtualNativeMergeBacktestEngineML) -> Tuple[Tuple[str, float], ...]:
    """宿主逐日物理总值（两臂必须逐日相同；由装配期断言）。"""
    return tuple(
        (str(item["date"]), float(item["portfolio_value"])) for item in host.portfolio_values
    )


__all__ = (
    "ArmFillFact",
    "ArmView",
    "COMMISSION_RATE",
    "MIN_COMMISSION",
    "MODEL_VERSION",
    "SINGLE_SLEEVE_ID",
    "SLIPPAGE",
    "SPLIT_SLEEVE_IDS",
    "STAMP_TAX",
    "SelfSplitArmError",
    "SelfSplitAssembly",
    "SelfSplitScenario",
    "assert_arms_are_distinct",
    "run_self_split_arms",
    "synthetic_cost_model",
)

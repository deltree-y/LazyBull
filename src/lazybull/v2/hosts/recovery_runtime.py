"""P6.5 T5 可恢复运行时：合成算例、空白宿主、逐位投影与"由信封新建实例"。

契约依据：设计 §4.5（快照与恢复语义）、§7.2（失败模式）；本模块只管**运行时**，
五类恢复场景的编排与结论见 ``hosts/recovery_cases.py``。

关键不变量：第 ``i`` 日（0 基）提交后提交点 = ``i + 1``；``restart(cp)`` 之后下一
待跑日 = ``cp``。恢复必须**新建实例**（``apply_owners`` 先断言实例未被演化过），
禁止在旧实例上续跑或叠加信封状态。

边界：合成对账边界内；不实现卖出（S5 归 P7）、不做借用、不读真实配置/行情，也不
进入生产默认路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.envelope_store import EnvelopeStore, StoredEnvelope
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.state_envelope import build_envelope, envelope_bytes, f64
from src.lazybull.v2.core.accounting.state_owners import (
    apply_owners,
    build_identity_owner,
    build_p4_owner,
    build_physical_owner,
    build_virtual_owner,
    decode_owners,
)
from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedger, VirtualLot
from src.lazybull.v2.core.decision.virtual_merger import MultiSleevePortfolioMerger
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.order_chain import OrderChainResult
from src.lazybull.v2.hosts.recovery_state_digest import state_digest
from src.lazybull.v2.hosts.self_split_arms import (
    HOST_ATTRIBUTE_PARAMETERS,
    HOST_RUN_PARAMETERS,
    SPLIT_SLEEVE_IDS,
    synthetic_cost_model,
)
from src.lazybull.v2.hosts.virtual_native_engine import VirtualNativeMergeBacktestEngineML

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"
#: 两袖配额（与自拆臂同口径 0.5/0.5；T5 算例不新增比例）
SLEEVE_WEIGHTS: Mapping[str, float] = {SPLIT_SLEEVE_IDS[0]: 0.5, SPLIT_SLEEVE_IDS[1]: 0.5}


class RecoveryError(ValueError):
    """恢复驱动输入/操作拒绝（**零副作用**）。"""


class RecoveryIntegrityError(RuntimeError):
    """恢复结论破缺（与参照不一致 / 破缺态被写入信封 / 水位未回到同代起点）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的恢复拒绝。"""
    return RecoveryError(f"阶段={STAGE} 恢复驱动 {message}")


def _integrity(message: str) -> Exception:
    """带阶段上下文的恢复结论破缺。"""
    return RecoveryIntegrityError(f"阶段={STAGE} 恢复 {message}")


# ────────────────────────── 合成算例 ──────────────────────────


@dataclass(frozen=True, slots=True)
class SeedVirtualLot:
    """期初虚拟取得批次（显式合成输入；带物理来源谱系与费用）。"""

    sleeve_id: str
    virtual_lot_id: str
    source_lot_id: str
    shares: int
    acquired_date: str
    acquired_price: float
    acquired_fee: float = 0.0


def _default_virtual_lots(code: str) -> Tuple[SeedVirtualLot, ...]:
    """期初虚拟账：A 持两个来源批次（L1 200 + L2 200），B 空仓。"""
    return (
        SeedVirtualLot("A", "A:L1", "L1", 200, "20260105", 10.0, 0.0),
        SeedVirtualLot("A", "A:L2", "L2", 200, "20260105", 10.0, 0.0),
    )


@dataclass(frozen=True, slots=True)
class RecoveryScenario:
    """T5 恢复算例输入（默认值即本任务冻结的合成算例）。

    五日结构（决策日只由当日价源派生目标，物理净额全部来自合并器）：
    - D0 ``20260105`` 期初估值日（无指令）：提交点 1；
    - D1 ``20260106`` A 400→150、B 0→250：内部转账 250 股，**跨 L1/L2 两个来源**；
    - D2 ``20260107`` A 150→100、B 250→300：再次内部转账 50 股；
    - D3 ``20260108`` A 100→100、B 300→400：物理买入 100 股（整手；新来源批次 L3 + 费用）；
    - D4 ``20260109`` A 100→200、B 400→400：物理买入 100 股（第二笔成交，水位非空）。

    物理增量刻意取整手（100 股），使 D3/D4 产生**真实成交**而不是被整手规则吃掉的
    零成交日；D1/D2 是纯内部转账日（物理零成交），用于隔离"转账"与"成交"两类事件。
    """

    ts_code: str = "600000.SH"
    trade_dates: Tuple[str, ...] = (
        "20260105",
        "20260106",
        "20260107",
        "20260108",
        "20260109",
    )
    prices: Tuple[Tuple[str, float], ...] = (
        ("20260105", 10.0),
        ("20260106", 10.0),
        ("20260107", 11.0),
        ("20260108", 11.0),
        ("20260109", 12.0),
    )
    physical_initial_capital: float = 100000.0
    #: 期初物理来源批次 ``(来源 lot_id, 股数, 成本)``（两个来源，支持跨批次 FIFO）
    physical_lots: Tuple[Tuple[str, int, float], ...] = (
        ("L1", 200, 2005.0),
        ("L2", 200, 2005.0),
    )
    #: 各袖期初现金（合计 = 物理初始资金；无借用）
    virtual_cash: Tuple[Tuple[str, float], ...] = (
        ("A", 50000.0),
        ("B", 50000.0),
    )
    #: 逐日各袖目标股数（``()` 表示当日无指令；键序不影响结果，只按袖 ID 排序）
    targets: Tuple[Tuple[Tuple[str, int], ...], ...] = (
        (),
        (("A", 150), ("B", 250)),
        (("A", 100), ("B", 300)),
        (("A", 100), ("B", 400)),
        (("A", 200), ("B", 400)),
    )
    #: 期初虚拟取得批次（缺省 = A 持 L1/L2 各 200，B 空仓）
    virtual_lots: Optional[Tuple[SeedVirtualLot, ...]] = None

    def __post_init__(self) -> None:
        dates = self._validate_dates_and_prices()
        self._validate_cash()
        self._validate_targets(len(dates))

    def _validate_dates_and_prices(self) -> Tuple[pd.Timestamp, ...]:
        """交易日严格递增 + 价源逐日一一对应且为正数。"""
        dates = tuple(pd.Timestamp(item) for item in self.trade_dates)
        if len(dates) < 2:
            raise _error(f"交易日至少需要 2 天，实际 {len(dates)}")
        if any(right <= left for left, right in zip(dates, dates[1:])):
            raise _error(f"交易日必须严格递增：{list(self.trade_dates)}")
        price_keys = tuple(str(key) for key, _value in self.prices)
        if price_keys != self.trade_dates:
            raise _error(
                f"价源日期 {list(price_keys)} 必须与交易日逐日一致 {list(self.trade_dates)}"
            )
        for _date, price in self.prices:
            if not float(price) > 0.0:
                raise _error(f"合成价格必须为正数，实际 {price!r}")
        return dates

    def _validate_cash(self) -> None:
        """各袖期初现金：袖 ID 合法且合计等于物理初始资金（无借用模式）。"""
        cash_total = 0.0
        for sleeve_id, cash in self.virtual_cash:
            self._require_sleeve(sleeve_id)
            cash_total += float(cash)
        if cash_total != float(self.physical_initial_capital):
            raise _error(
                f"各袖期初现金合计 {cash_total!r} != 物理初始资金 "
                f"{self.physical_initial_capital!r}（无借用模式必须相等）"
            )

    def _validate_targets(self, day_count: int) -> None:
        """逐日目标：日数一一对应、袖 ID 合法、股数为非负整数。"""
        if len(self.targets) != day_count:
            raise _error(
                f"逐日目标数 {len(self.targets)} != 交易日数 {day_count}；禁止缺日或补日"
            )
        for entry in self.targets:
            for sleeve_id, shares in entry:
                self._require_sleeve(sleeve_id)
                if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
                    raise _error(f"袖 {sleeve_id} 目标股数必须为非负整数，实际 {shares!r}")

    @staticmethod
    def _require_sleeve(sleeve_id: Any) -> None:
        """袖 ID 必须在冻结的两袖集合内。"""
        if sleeve_id not in SLEEVE_WEIGHTS:
            raise _error(f"未知袖 {sleeve_id!r}；合法={sorted(SLEEVE_WEIGHTS)}")

    def date_list(self) -> Tuple[pd.Timestamp, ...]:
        """规范化交易日序列。"""
        return tuple(pd.Timestamp(item) for item in self.trade_dates)

    def price_of(self, index: int) -> float:
        """第 ``index`` 个交易日的合成价格。"""
        return float(self.prices[index][1])

    def virtual_lot_seeds(self) -> Tuple[SeedVirtualLot, ...]:
        """期初虚拟批次（缺省构造）。"""
        if self.virtual_lots is not None:
            return tuple(self.virtual_lots)
        return _default_virtual_lots(self.ts_code)

    def targets_of(self, index: int) -> Dict[str, Dict[str, int]]:
        """第 ``index`` 个交易日的逐袖目标（无指令返回空 dict）。"""
        entry = self.targets[index]
        if not entry:
            return {}
        return {sleeve_id: {self.ts_code: int(shares)} for sleeve_id, shares in entry}

    def summary(self) -> Dict[str, Any]:
        """场景输入登记（证据用；浮点按位精确编码）。"""
        return {
            "ts_code": self.ts_code,
            "trade_dates": list(self.trade_dates),
            "prices": [[date, f64(price)] for date, price in self.prices],
            "physical_initial_capital": f64(self.physical_initial_capital),
            "physical_lots": [
                [lot_id, int(shares), f64(cost)] for lot_id, shares, cost in self.physical_lots
            ],
            "virtual_cash": [[sleeve_id, f64(cash)] for sleeve_id, cash in self.virtual_cash],
            "virtual_lots": [
                {
                    "sleeve_id": seed.sleeve_id,
                    "virtual_lot_id": seed.virtual_lot_id,
                    "source_lot_id": seed.source_lot_id,
                    "shares": int(seed.shares),
                    "acquired_date": seed.acquired_date,
                    "acquired_price": f64(seed.acquired_price),
                    "acquired_fee": f64(seed.acquired_fee),
                }
                for seed in self.virtual_lot_seeds()
            ],
            "targets": [
                [[sleeve_id, int(shares)] for sleeve_id, shares in entry]
                for entry in self.targets
            ],
        }


def _price_data(scenario: RecoveryScenario) -> pd.DataFrame:
    """引擎价格索引输入（close/close_adj/open/open_adj 同值：成交与估值同源）。"""
    rows = []
    for index, date in enumerate(scenario.date_list()):
        price = scenario.price_of(index)
        rows.append(
            {
                "ts_code": scenario.ts_code,
                "trade_date": date,
                "close": price,
                "close_adj": price,
                "open": price,
                "open_adj": price,
            }
        )
    return pd.DataFrame(rows)


def _physical_view(scenario: RecoveryScenario) -> PhysicalLotView:
    """期初物理来源 lot 伴随视图（顺序 = 场景声明顺序，即批次顺序）。"""
    return PhysicalLotView(
        [
            PhysicalLot(scenario.ts_code, lot_id, int(shares), float(cost))
            for lot_id, shares, cost in scenario.physical_lots
        ]
    )


def _virtual_ledger(scenario: RecoveryScenario) -> VirtualLedger:
    """期初虚拟账（**空账**：批次由信封恢复或由期初 seeds 建立）。"""
    return VirtualLedger(cost_model=synthetic_cost_model())


def _seed_virtual_ledger(ledger: VirtualLedger, scenario: RecoveryScenario) -> None:
    """期初开户与批次注入（只用于"无中断参照"的起点构造）。"""
    lots_by_sleeve: Dict[str, List[VirtualLot]] = {}
    for seed in scenario.virtual_lot_seeds():
        lots_by_sleeve.setdefault(seed.sleeve_id, []).append(
            VirtualLot(
                virtual_lot_id=seed.virtual_lot_id,
                ts_code=scenario.ts_code,
                source_lot_id=seed.source_lot_id,
                shares=int(seed.shares),
                acquired_date=TradeDate.from_str(seed.acquired_date),
                acquired_price=float(seed.acquired_price),
                acquired_fee=float(seed.acquired_fee),
            )
        )
    for sleeve_id, cash in scenario.virtual_cash:
        ledger.open_account(sleeve_id, float(cash), lots_by_sleeve.get(sleeve_id, ()))


def batch_id_for_day(day_index: int) -> str:
    """逐日结算批次身份（**与运行 ID 无关**：同一事件在不同链上必须同名）。

    批次身份是"第 N 日的净额结算"这一事件的属性，而不是某次运行的属性；否则
    "无中断参照 vs 恢复"的逐位比较会被运行标签污染（引用同一事件的两种链必须
    产生同一身份，才能比较幂等键集合与水位）。
    """
    if isinstance(day_index, bool) or not isinstance(day_index, int) or day_index < 0:
        raise _error(f"日序必须为非负 int，实际 {day_index!r}")
    return f"p65t5-d{day_index}"


def build_host(scenario: RecoveryScenario, *, run_id: str) -> VirtualNativeMergeBacktestEngineML:
    """构造**空白**合成宿主（物理持仓/虚拟账/伴随视图全部由调用方建立）。

    空白实例是"由信封新建引擎"的前提：``apply_owners`` 会先断言实例未被演化过。
    """
    merger = MultiSleevePortfolioMerger(SPLIT_SLEEVE_IDS, dict(SLEEVE_WEIGHTS))
    host = VirtualNativeMergeBacktestEngineML(
        features_by_date={},
        universe=BasicUniverse(
            pd.DataFrame(
                {
                    "ts_code": [scenario.ts_code],
                    "name": [scenario.ts_code],
                    "industry": ["I"],
                }
            ),
            verbose=False,
        ),
        signal=EqualWeightSignal(top_n=1),
        initial_capital=float(scenario.physical_initial_capital),
        cost_model=synthetic_cost_model(),
        verbose=False,
        portfolio_merger=merger,
        virtual_ledger=_virtual_ledger(scenario),
        physical_lot_view=PhysicalLotView(),
        price_source_provider=None,
        run_id=run_id,
        **dict(HOST_RUN_PARAMETERS),
    )
    host.current_capital = float(scenario.physical_initial_capital)
    host.positions = {}
    for key, value in HOST_ATTRIBUTE_PARAMETERS.items():
        setattr(host, key, value)
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(_price_data(scenario))
    return host


@dataclass
class RestoredGeneration:
    """一次"由信封新建实例"的结果（返回水位与同代起点，供断言与证据）。"""

    commit_point: int
    envelope_sha256: str
    applied_keys: Tuple[str, ...]
    last_fill_date: Optional[str]
    settlement_keys: Tuple[str, ...]
    batch_ids: Tuple[str, ...]
    last_event_sequence: int
    price_source_id: str
    price_source_as_of: str
    host: VirtualNativeMergeBacktestEngineML = field(repr=False)


class RecoverableRuntime:
    """逐日运行 + 同代提交 + 由信封重建（**唯一恢复入口**）。

    提交点不变量：第 ``i`` 日（0 基）提交后提交点 = ``i + 1``；``restart(cp)`` 之后
    下一待跑日 = ``cp``（即该提交点已覆盖到第 ``cp-1`` 日）。
    """

    def __init__(
        self,
        *,
        scenario: RecoveryScenario,
        root: Path,
        run_id: str,
        seed_initial_state: bool = True,
    ) -> None:
        self.scenario = scenario
        self.run_id = run_id
        self.store = EnvelopeStore(Path(root), run_id=run_id)
        self.host = build_host(scenario, run_id=run_id)
        self._next_day = 0
        self.chain_results: List[OrderChainResult] = []
        # 崩溃现场快照：本实例建立**之前**已隔离的残骸（T5-Codex-C3）。此后新出现的
        # 隔离记录说明"本实例的内存水位曾经半提交"，只能重建重放，不能直接提交。
        self._quarantine_snapshot = frozenset(self.store.quarantine_note_files())
        if seed_initial_state:
            self._seed_initial_state()

    # ── 期初 ──

    def _seed_initial_state(self) -> None:
        """建立期初物理持仓/来源批次与虚拟账（无中断参照的起点）。"""
        scenario = self.scenario
        physical_shares = sum(int(shares) for _lot_id, shares, _cost in scenario.physical_lots)
        physical_cost = sum(float(cost) for _lot_id, _shares, cost in scenario.physical_lots)
        self.host.positions = {
            scenario.ts_code: {
                "shares": physical_shares,
                "buy_date": pd.Timestamp(scenario.trade_dates[0]),
                "buy_cost_cash": physical_cost,
                "buy_trade_price": scenario.price_of(0),
            }
        }
        self.host.physical_lot_view = _physical_view(scenario)
        _seed_virtual_ledger(self.host.virtual_ledger, scenario)

    # ── 日步进 ──

    def run_day(self, day_index: int) -> Optional[OrderChainResult]:
        """执行第 ``day_index`` 日：链（若有目标）→ 恒等式门；**不提交信封**。"""
        scenario = self.scenario
        if day_index != self._next_day:
            raise _error(
                f"日序不符：期望第 {self._next_day} 日，实际 {day_index}"
                f"（同一提交点只能顺序推进，禁止跳日/回退）"
            )
        dates = scenario.date_list()
        date = dates[day_index]
        targets = scenario.targets_of(day_index)
        result: Optional[OrderChainResult] = None
        if targets:
            result = self.host.execute_order_chain(
                date, targets, batch_id=batch_id_for_day(day_index)
            )
            self.chain_results.append(result)
        portfolio_value = self.host._calculate_portfolio_value(date)
        self.host._finalize_day(
            day_index,
            date,
            0,
            len(dates),
            day_index + 1,
            portfolio_value,
            list(dates),
            {item: index for index, item in enumerate(dates)},
        )
        self._next_day = day_index + 1
        return result

    def commit_day(self, day_index: int) -> StoredEnvelope:
        """提交第 ``day_index`` 日的同代信封（五项同一原子提交点）。

        破缺/终态实例一律拒绝（**破缺态不得写入新信封**），且水印倒退/键集收缩在
        ``build_envelope`` + 存储层校验中拒绝。
        """
        if self.host.is_terminated:
            raise _integrity(
                f"第 {day_index} 日的宿主已进入终态（恒等式破缺或执行后失败）；"
                "破缺态不得写入新信封——只能保留破缺前信封并从其重放"
            )
        if self.host.virtual_ledger.poisoned_reason is not None:
            raise _integrity(
                f"第 {day_index} 日的虚拟账已进入终态（{self.host.virtual_ledger.poisoned_reason}）；"
                "破缺态不得写入新信封"
            )
        if day_index != self._next_day - 1:
            # T5-A-R1-01：提交的日子必须**正是**本实例刚执行完的那一日；否则
            # restart(n) 后不跑第 n+1 日即可把旧代状态写成新提交点（跨代水位倒退）
            raise _error(
                f"提交日序不符：当前实例最后执行的是第 {self._next_day - 1} 日，"
                f"实际要求提交第 {day_index} 日；禁止把旧代状态写到新提交点（跨代重用水位）"
            )
        commit_point = day_index + 1
        # T5-Codex-C3：隔离残骸**不能**解除旧实例的提交禁令——只要该提交点的现场是在
        # 本实例建立之后才被隔离的，本实例的内存水位就属于"曾经半提交"的那一次，
        # 必须从上一完整信封新建实例重放（`restart`）后再提交。
        stale_notes = set(self.store.quarantine_note_files_of(commit_point)) - set(
            self._quarantine_snapshot
        )
        if stale_notes:
            names = sorted(path.name for path in stale_notes)[:2]
            raise _integrity(
                f"提交点 {commit_point} 的半提交现场在本实例建立之后才被隔离（{names}）；"
                "已推进的内存水位不可信——必须 restart(该提交点之前) 新建实例并重放后再提交"
            )
        date = self.scenario.date_list()[day_index]
        states = self.host.virtual_ledger.snapshot()
        source = self.host._build_price_source(date, states=states)
        owners = {
            "p4_engine_state": build_p4_owner(self.host),
            "physical_lot_view": build_physical_owner(self.host.physical_lot_view),
            "virtual_ledger": build_virtual_owner(self.host.virtual_ledger),
            "settlement_identity": build_identity_owner(
                host=self.host,
                ledger=self.host.virtual_ledger,
                price_source_id=source.source_id,
                price_source_as_of=str(source.as_of),
            ),
        }
        previous = self.store.load(commit_point - 1) if commit_point > 1 else None
        previous_sha = "" if previous is None else previous.file_sha256
        envelope = build_envelope(
            run_id=self.run_id,
            commit_point=commit_point,
            trade_date=str(source.as_of),
            previous_commit_point=commit_point - 1,
            previous_envelope_sha256=previous_sha,
            state_owners=owners,
        )
        return self.store.commit(envelope)

    def run_and_commit(self, day_index: int) -> StoredEnvelope:
        """执行并提交同一日（常规推进）。"""
        self.run_day(day_index)
        return self.commit_day(day_index)

    # ── 恢复 ──

    def restart(self, commit_point: int) -> RestoredGeneration:
        """**由信封新建实例**并恢复五项状态（不复活旧实例、不续用旧水位）。"""
        stored = self.store.load(commit_point)
        restored = decode_owners(stored.envelope)
        scenario = self.scenario
        self.host = build_host(scenario, run_id=self.run_id)
        apply_owners(self.host, restored)
        self._next_day = commit_point
        self.chain_results = []
        # 新实例建立时刻之前的隔离记录视为"已消化"（T5-Codex-C3）；此后再出现的
        # 隔离记录才会阻止本实例提交。
        self._quarantine_snapshot = frozenset(self.store.quarantine_note_files())
        ledger_state = self.host.ledger.export_state()
        last_fill = ledger_state["last_fill_date"]
        return RestoredGeneration(
            commit_point=commit_point,
            envelope_sha256=stored.file_sha256,
            applied_keys=tuple(str(key) for key in ledger_state["applied_keys"]),
            last_fill_date=None if last_fill is None else str(last_fill),
            settlement_keys=tuple(str(key) for key in self.host.virtual_ledger.settlement_ids()),
            batch_ids=tuple(str(key) for key in self.host.virtual_ledger.batch_ids()),
            last_event_sequence=int(restored.identity.last_event_sequence),
            price_source_id=restored.identity.price_source_id,
            price_source_as_of=restored.identity.price_source_as_of,
            host=self.host,
        )

    # ── 半提交代现场 ──

    def crash_after(self, day_index: int) -> Tuple[Path, Path]:
        """模拟"写一半"崩溃：留下截断的最终文件与残留临时文件（**保留现场**）。

        前置：第 ``day_index`` 日已在内存中执行完毕（水位已推进）但**尚未提交**。

        Returns:
            ``(截断最终名, 残留临时名)`` 两个残骸路径。
        """
        commit_point = day_index + 1
        date = self.scenario.date_list()[day_index]
        states = self.host.virtual_ledger.snapshot()
        source = self.host._build_price_source(date, states=states)
        owners = {
            "p4_engine_state": build_p4_owner(self.host),
            "physical_lot_view": build_physical_owner(self.host.physical_lot_view),
            "virtual_ledger": build_virtual_owner(self.host.virtual_ledger),
            "settlement_identity": build_identity_owner(
                host=self.host,
                ledger=self.host.virtual_ledger,
                price_source_id=source.source_id,
                price_source_as_of=str(source.as_of),
            ),
        }
        previous = self.store.load(commit_point - 1) if commit_point > 1 else None
        envelope = build_envelope(
            run_id=self.run_id,
            commit_point=commit_point,
            trade_date=str(source.as_of),
            previous_commit_point=commit_point - 1,
            previous_envelope_sha256="" if previous is None else previous.file_sha256,
            state_owners=owners,
        )
        payload = envelope_bytes(envelope)
        final_path = self.store.path_of(commit_point)
        temp_path = self.store.temp_path_of(commit_point)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(payload[: max(1, len(payload) // 2)])  # 截断：半提交代
        temp_path.write_bytes(payload[: max(1, len(payload) // 4)])
        return final_path, temp_path

    # ── 观测 ──

    def digest(self, day_index: int) -> Dict[str, Any]:
        """当前状态相对第 ``day_index`` 日的逐位投影。"""
        return state_digest(self.host, scenario=self.scenario, day_index=day_index)

    def envelope_hashes(self) -> Tuple[Tuple[int, str], ...]:
        """链上各提交点的信封摘要。"""
        return tuple(
            (point, self.store.load(point).file_sha256) for point in self.store.commit_points()
        )

"""P6.5 T4 证据装配：三件套（S1/S4/自拆一致）+ G4 比较 + 自含 manifest 与 SHA-256。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §8.2（S1 零换手）、
  §8.3（S4 分摊守恒）、§8.4（自拆一致与逐位比较）、§10.1（G3/G4 候选证据产物）与
  §10.1 证据根 ``data/frozen_reference/v2_p65/<run_id>/``；
- ``docs/contracts/netting_freeze.md`` F25 §4/§5、``docs/contracts/runs_artifact_contract.md``
  §1/§2/§9；
- T4 任务 §3.4（产物至少 zero_turnover / allocation_conservation / self_split /
  self_split_diff、双臂 generation manifest、规范化 Parquet/JSON 及 SHA-256）。

边界（如实登记，不得读成更强结论）：
- 本模块只产出**合成对账边界内**的候选门证据：不宣称 G4 已通过阶段验收、不宣称
  B 区转正或生产多袖能力，也不宣称 P6.5 出口；
- S1/S4 为账本层算例（T2 公式不改写，本模块只复算并落证据）；自拆一致走 T3 全链与
  T4 装配；``attribution`` 两侧同源缺失登记 ``not_compared/source_absent``，
  ``topk_detail`` 登记 ``not_compared/contract_optional_missing``，**不得**写成逐位一致；
- 不写 ``data/runs/``：两臂 runs 落在证据根下的 ``arms/`` 内；
- 本模块不读真实配置/行情/账户；费率、价格、配额全部为显式合成输入。
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.g4_nav import (
    assert_g4_nav_bitwise_equal,
    float64_bits,
    g4_merged_nav,
)
from src.lazybull.v2.core.decision.target_synthesis import LOT_SIZE
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import (
    CASH_TOLERANCE,
    HOLDINGS_TOLERANCE_SHARES,
    NAV_TOLERANCE,
    IdentityContext,
    check_nav_identity,
)
from src.lazybull.v2.core.accounting.settlement_kernel import (
    InternalSettlementBatch,
    PhysicalFillFact,
    SleeveIntent,
    TransferFact,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLot,
)
from src.lazybull.v2.evidence.arm_comparator import (
    EXCLUDED_META_FIELDS,
    EXCLUSION_REASONS,
    compare_assemblies,
)
from src.lazybull.v2.evidence.fingerprint_keys import config_fingerprint_keys
from src.lazybull.v2.hosts.self_split_arms import (
    COMMISSION_RATE,
    HOST_ATTRIBUTE_PARAMETERS,
    HOST_RUN_PARAMETERS,
    MIN_COMMISSION,
    SLIPPAGE,
    STAMP_TAX,
    SelfSplitScenario,
    run_self_split_arms,
    synthetic_cost_model,
)
from src.lazybull.v2.hosts.virtual_arm_runs import (
    ARM_SINGLE,
    ARM_SPLIT,
    require_ascii_id,
)

#: 本模块唯一阶段标识（进入全部证据字段）
STAGE = "P6.5"
#: 证据 schema 版本（本模块自含，不等于 runs 契约版本）
EVIDENCE_SCHEMA = 1
#: 证据根（相对仓库根；设计 §10.1 Q7 候选目录）
EVIDENCE_ROOT = Path("data/frozen_reference/v2_p65")
#: runs writer 身份（只读登记；本阶段不改 writer）
RUNS_WRITER_PATH = Path("src/lazybull/v2/hosts/backtest/runs_writer.py")
#: 排除表来源模块（记录内容哈希以复核"运行前冻结"）
COMPARATOR_PATH = Path("src/lazybull/v2/evidence/arm_comparator.py")
#: 合成算例写出的 runs 批次配置（两臂共用；不含任何臂相关值/墙钟）
WRITER_CONFIG: Mapping[str, Any] = {
    "bt_top_n": 1,
    "label_column": "syn_p65_t4",
    "stock_domain": "main",
    "wait_days": 0,
    "holding_days": 2,
    "wf_start_date": "20260105",
    "wf_end_date": "20260107",
    "merger_mode": "multi_sleeve_virtual",
}
#: ``batch_meta.config`` 的**完整冻结键值清单**（外部评审 T4-Codex-02）：
#: ``WRITER_CONFIG`` 8 键 + writer 为批次补齐的折/版本元数据 6 键；``batch_id`` 为动态键
#: （等于 ``<run_id>-arms-b0``），逐键值必须与记录值相等，多余键一律拒绝。
EXPECTED_BATCH_CONFIG: Mapping[str, Any] = {
    "bt_top_n": 1,
    "label_column": "syn_p65_t4",
    "stock_domain": "main",
    "wait_days": 0,
    "holding_days": 2,
    "wf_start_date": "20260105",
    "wf_end_date": "20260107",
    "merger_mode": "multi_sleeve_virtual",
    "model_version": 6501,
    "split_index": 0,
    "test_start": "20260105",
    "test_end": "20260107",
    "train_start": "20260105",
    "train_end": "20260105",
}
#: 值随运行变化、不参与值比对的配置键
DYNAMIC_CONFIG_KEYS: Tuple[str, ...] = ("batch_id",)
#: 自拆算例的**场景输入冻结表**（与场景/费率/整手常量的单一来源逐项核对）。
#: 注意：这里**不含** ``holding_days``——它是 runs 批次配置字段（``batch_meta.config``），
#: 不是宿主持有期；宿主持有期/调仓频率等由 ``self_split_arms.HOST_RUN_PARAMETERS`` 显式
#: 传给宿主并从实例回读（复审 R1-01）。
SCENARIO_PARAMETERS: Mapping[str, Any] = {
    "top_n": 1,
    "wait_days": 0,
    "physical_initial_capital": 100000.0,
    "combo_quota": 4000.0,
    "split_ratio": 0.5,
    "lot_size": LOT_SIZE,
    "commission_rate": COMMISSION_RATE,
    "min_commission": MIN_COMMISSION,
    "stamp_tax": STAMP_TAX,
    "slippage": SLIPPAGE,
    "trade_dates": ["2026-01-05", "2026-01-06", "2026-01-07"],
    "trade_day_index": 1,
    "prices": {
        "2026-01-05": {"600000.SH": 10.0},
        "2026-01-06": {"600000.SH": 10.0},
        "2026-01-07": {"600000.SH": 11.0},
    },
}
#: 观察窗口（决策日 D1 → 最后估值日 D3 的交易日跨度）；仅用于表述场景，不写入宿主持有期
WINDOW_DAYS = 2
_X = "600000.SH"
_D1 = pd.Timestamp("2026-01-06")
_D2 = pd.Timestamp("2026-01-07")
_S1_SOURCE_ID = "syn_p65_t4_s1:20260106"
_S4_SOURCE_ID = "syn_p65_t4_s4:20260106"


class P65EvidenceError(ValueError):
    """证据装配输入非法（零副作用拒绝）。"""


class EvidenceVerdictError(P65EvidenceError):
    """证据已完整落盘但候选门判定为 FAIL（含差异清单路径；驱动映射退出码 3）。"""

    def __init__(
        self, message: str, *, run_id: str, root: Path, verdict: Mapping[str, Any]
    ) -> None:
        self.run_id = run_id
        self.root = Path(root)
        self.verdict = dict(verdict)
        super().__init__(message)


def _error(message: str) -> Exception:
    return P65EvidenceError(f"阶段={STAGE} T4 证据 {message}")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """证据产物（根目录、逐文件 SHA-256 与判定摘要）。"""

    run_id: str
    root: Path
    files: Tuple[Tuple[str, str], ...]
    verdict: Mapping[str, Any]

    def sha_of(self, relative: str) -> str:
        """单文件 SHA-256（缺失即拒绝）。"""
        for path, digest in self.files:
            if path == relative:
                return digest
        raise _error(f"{relative} 不在证据产物集中")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """写规范化 JSON（``sort_keys`` + 缩进）并返回 SHA-256。"""
    text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, default=str)
    data = (text + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256_bytes(data)


def _tree_hashes(root: Path, *, skip: Tuple[str, ...] = ()) -> Tuple[Tuple[str, str], ...]:
    """目录下全部文件的相对路径 + SHA-256（跳过 ``skip`` 指定的相对路径）。"""
    entries: List[Tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in skip:
            continue
        entries.append((relative, _sha256_bytes(path.read_bytes())))
    return tuple(entries)


def _context(sleeve_ids: Tuple[str, ...], *, price_source_id: str, run_id: str) -> IdentityContext:
    return IdentityContext(
        trade_date=TradeDate(_D1.date()),
        host="p65_t4_evidence",
        run_id=run_id,
        sleeve_ids=sleeve_ids,
        price_source_id=price_source_id,
    )


# ────────────────────────── S1 零换手 ──────────────────────────


def build_zero_turnover_evidence(*, run_id: str) -> Dict[str, Any]:
    """S1 零换手（设计 §8.2）：内部转账 100 股 @12，物理成交数与费用恒为 0。"""
    ledger = VirtualLedger(cost_model=synthetic_cost_model())
    ledger.open_account(
        "A", 50_000.0, [VirtualLot("A:L1", _X, "L1", 400, TradeDate(_D1.date()), 10.0)]
    )
    ledger.open_account("B", 50_000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    source = PriceSource(as_of=TradeDate(_D1.date()), source_id=_S1_SOURCE_ID, prices={_X: 12.0})
    context = _context(("A", "B"), price_source_id=_S1_SOURCE_ID, run_id=run_id)
    batch = InternalSettlementBatch(
        batch_id=f"{run_id}-s1",
        trade_date=TradeDate(_D1.date()),
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
        ),
        transfers=(
            TransferFact(ts_code=_X, from_sleeve="A", to_sleeve="B", shares=100, price=12.0),
        ),
        physical_cash_before=100_000.0,
        physical_cash_after=100_000.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=source,
        physical_view=view,
        identity_context=context,
        physical_positions={_X: {"shares": 400}},
    )
    states = ledger.snapshot()
    physical_total = 100_000.0 + 400 * 12.0
    i3 = check_nav_identity(
        virtual_states=states,
        physical_total_value=physical_total,
        physical_initial_capital=100_000.0,
        price_source=source,
        context=context,
    )
    g4 = g4_merged_nav(states, source, 100_000.0, context=context)
    physical_nav = float(calculate_nav(physical_total, 100_000.0))
    physical_nav_bits = float64_bits(physical_nav)
    # G4 位模式门（单一入口）：物理 NAV 与"先合并各袖总值再单次派生"的 NAV 必须逐位相同
    assert_g4_nav_bitwise_equal(
        physical_nav, g4.merged_nav, context=context, label="S1 物理 NAV vs 合并视图 NAV"
    )
    sleeves = {
        sleeve_id: {
            "cash": states[sleeve_id].cash,
            "shares": states[sleeve_id].shares_of(_X),
            "total_value": states[sleeve_id].total_value(source),
            "nav_contrib": float(calculate_nav(states[sleeve_id].total_value(source), 100_000.0)),
            "borrowed_credit": states[sleeve_id].borrowed_credit,
        }
        for sleeve_id in sorted(states)
    }
    payload = {
        "case": "S1_zero_turnover",
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "inputs": {
            "physical_cash": 100_000.0,
            "physical_shares": 400,
            "physical_source_lot": "L1",
            "virtual_cash": {"A": 50_000.0, "B": 50_000.0},
            "price": 12.0,
            "targets": {"A": {_X: 300}, "B": {_X: 100}},
            "internal_transfer": {"from": "A", "to": "B", "shares": 100, "price": 12.0},
            "fees": {
                "commission_rate": COMMISSION_RATE,
                "min_commission": MIN_COMMISSION,
                "stamp_tax": STAMP_TAX,
                "slippage": SLIPPAGE,
            },
        },
        "physical": {
            "fills": len(batch.physical_fills),
            "fees": 0.0,
            "cash": 100_000.0,
            "shares": 400,
            "total_value": physical_total,
            "nav": physical_nav,
            "nav_bits": physical_nav_bits,
        },
        "virtual": sleeves,
        "virtual_merged": {
            "cash_sum": sum(states[sleeve_id].cash for sleeve_id in sorted(states)),
            "source_shares": [list(item) for item in result.source_shares_after],
            "total_value": g4.merged_total_value,
            "nav": g4.merged_nav,
            "nav_bits": g4.merged_nav_bits,
        },
        "identities": {
            "i3_left": i3.left,
            "i3_right": i3.right,
            "i3_diff": i3.diff,
            "i3_tolerance": NAV_TOLERANCE,
            "g4_nav": g4.merged_nav,
            "g4_nav_bits": g4.merged_nav_bits,
            "physical_nav_bits": physical_nav_bits,
            "i3_identities": [item.identity for item in result.identity_results],
        },
        "assertions": {
            "no_physical_fill": result.transfers and not batch.physical_fills,
            "no_physical_fee": all(
                fee == 0.0
                for waterfall in result.fee_waterfalls
                for _sid, fee in waterfall.final_fees
            ),
            "no_zero_borrowed_credit": all(
                state.borrowed_credit == 0.0 for state in states.values()
            ),
            "nav_sum_equals_physical": abs(i3.diff) <= NAV_TOLERANCE,
            "g4_bitwise_equals_physical": g4.merged_nav_bits == physical_nav_bits,
        },
    }
    if not all(payload["assertions"].values()):
        raise _error(f"S1 零换手断言失败：{payload['assertions']}")
    return payload


# ────────────────────────── S4 分摊守恒 ──────────────────────────


def build_allocation_conservation_evidence(*, run_id: str) -> Dict[str, Any]:
    """S4 分摊守恒（设计 §8.3）：物理成交 200 股、费用 5 元由 Q13 瀑布守恒分摊。"""
    ledger = VirtualLedger(cost_model=synthetic_cost_model())
    ledger.open_account("A", 2_500.0)
    ledger.open_account("B", 7_500.0)
    view = PhysicalLotView([PhysicalLot(_X, "L9", 200)])
    source = PriceSource(as_of=TradeDate(_D1.date()), source_id=_S4_SOURCE_ID, prices={_X: 10.0})
    context = _context(("A", "B"), price_source_id=_S4_SOURCE_ID, run_id=run_id)
    batch = InternalSettlementBatch(
        batch_id=f"{run_id}-s4",
        trade_date=TradeDate(_D1.date()),
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_fills=(
            PhysicalFillFact(
                ts_code=_X,
                side="buy",
                shares=200,
                price=10.0,
                fee=5.0,
                source_lot_id="L9",
                fill_id=f"{run_id}-s4-fill",
            ),
        ),
        physical_cash_before=10_000.0,
        physical_cash_after=7_995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=source,
        physical_view=view,
        identity_context=context,
        physical_positions={_X: {"shares": 200}},
    )
    states = ledger.snapshot()
    physical_total = 7_995.0 + 200 * 10.0
    i3 = check_nav_identity(
        virtual_states=states,
        physical_total_value=physical_total,
        physical_initial_capital=10_000.0,
        price_source=source,
        context=context,
    )
    allocations = {item.sleeve_id: item.allocated_shares for item in result.share_allocations}
    waterfall = result.fee_waterfalls[0]
    fees = {sid: value for sid, value in waterfall.final_fees}
    sleeves = {
        sleeve_id: {
            "cash": states[sleeve_id].cash,
            "shares": states[sleeve_id].shares_of(_X),
            "allocated_shares": allocations[sleeve_id],
            "fee": fees[sleeve_id],
            "total_value": states[sleeve_id].total_value(source),
            "nav_contrib": float(calculate_nav(states[sleeve_id].total_value(source), 10_000.0)),
        }
        for sleeve_id in sorted(states)
    }
    payload = {
        "case": "S4_allocation_conservation",
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "inputs": {
            "virtual_cash": {"A": 2_500.0, "B": 7_500.0},
            "price": 10.0,
            "intents": {"A": {_X: 100}, "B": {_X: 300}},
            "physical_fill": {"shares": 200, "amount": 2_000.0, "fee": 5.0},
        },
        "physical": {
            "fills": 1,
            "shares": 200,
            "amount": 2_000.0,
            "fee": 5.0,
            "cash": 7_995.0,
            "total_value": physical_total,
            "nav": float(calculate_nav(physical_total, 10_000.0)),
        },
        "virtual": sleeves,
        "fee_waterfall": {
            "order": list(waterfall.order),
            "intent_amounts": [list(item) for item in waterfall.intent_amounts],
            "allocated_amounts": [list(item) for item in waterfall.allocated_amounts],
            "independent_costs": [list(item) for item in waterfall.independent_costs],
            "provisional_fees": [list(item) for item in waterfall.provisional_fees],
            "terminal_sleeve": waterfall.terminal_sleeve,
            "residual_correction": waterfall.residual_correction,
            "final_fees": [list(item) for item in waterfall.final_fees],
            "physical_fee": waterfall.physical_fee,
        },
        "identities": {
            "i3_left": i3.left,
            "i3_right": i3.right,
            "i3_diff": i3.diff,
            "i3_tolerance": NAV_TOLERANCE,
            "cash_tolerance": CASH_TOLERANCE,
            "holdings_tolerance_shares": HOLDINGS_TOLERANCE_SHARES,
        },
        "assertions": {
            "shares_conserved": sum(allocations.values()) == 200,
            "fee_conserved": sum(fees.values()) == 5.0,
            "cash_sum_equals_physical": states["A"].cash + states["B"].cash == 7_995.0,
            "no_negative_fee": all(value >= 0.0 for value in fees.values()),
            "nav_sum_equals_physical": abs(i3.diff) <= NAV_TOLERANCE,
            # Q22：未成交的 200 股意图按日终作废（50+150），不得顺延
            "voided_intents_total": sum(item[2] for item in result.voided_intents) == 200,
        },
        "voided_intents": [list(item) for item in result.voided_intents],
    }
    if not all(payload["assertions"].values()):
        raise _error(f"S4 分摊守恒断言失败：{payload['assertions']}")
    return payload


# ────────────────────────── 自拆一致 + G4 ──────────────────────────


def _scenario_parameters_document(scenario: SelfSplitScenario) -> Dict[str, Any]:
    """场景输入冻结表与实际场景对象逐项核对，返回实际值（含观察窗口跨度）。"""
    actual = {
        "top_n": int(WRITER_CONFIG["bt_top_n"]),
        "wait_days": int(WRITER_CONFIG["wait_days"]),
        "physical_initial_capital": float(scenario.physical_initial_capital),
        "combo_quota": float(scenario.combo_quota),
        "split_ratio": float(scenario.split_ratio),
        "lot_size": int(LOT_SIZE),
        "commission_rate": float(COMMISSION_RATE),
        "min_commission": float(MIN_COMMISSION),
        "stamp_tax": float(STAMP_TAX),
        "slippage": float(SLIPPAGE),
        "trade_dates": [str(item.date()) for item in scenario.date_list()],
        "trade_day_index": int(scenario.trade_day_index),
        "prices": {
            str(pd.Timestamp(date).date()): {
                str(code): float(value) for code, value in sorted(prices.items())
            }
            for date, prices in scenario.prices
        },
        "window_days": WINDOW_DAYS,
    }
    drift = {
        key: (actual.get(key), frozen)
        for key, frozen in SCENARIO_PARAMETERS.items()
        if actual.get(key) != frozen
    }
    if drift:
        raise _error(
            f"场景参数与冻结清单不符（实际, 冻结）={drift}；"
            "证据不得记录与宿主实际输入不一致的场景"
        )
    return actual


def _host_parameters_document(assembly: Any) -> Dict[str, Any]:
    """两臂**宿主实例回读**的生效参数（复审 R1-01：记录值必须等于实际值）。

    值来源是 ``SelfSplitAssembly.host_parameters``（``_new_host`` 构造后
    ``read_host_parameters`` 回读）；装配期已断言两臂同值且等于冻结表，这里再核一遍
    并登记，防止证据与执行链分叉。
    """
    frozen = {**HOST_RUN_PARAMETERS, **HOST_ATTRIBUTE_PARAMETERS}
    observed = {arm: dict(values) for arm, values in dict(assembly.host_parameters).items()}
    missing = [arm for arm in (ARM_SINGLE, ARM_SPLIT) if arm not in observed]
    if missing:
        raise _error(f"缺少宿主参数回读：{missing}；证据不得记录未回读的参数")
    single, split = observed[ARM_SINGLE], observed[ARM_SPLIT]
    if single != split:
        drift = sorted(key for key in set(single) | set(split) if single.get(key) != split.get(key))
        raise _error(f"两臂宿主参数不一致，差异键={drift}")
    drift = sorted(key for key in frozen if single.get(key) != frozen[key])
    if drift:
        raise _error(
            "宿主实际参数与冻结表不符，差异键="
            f"{[(key, single.get(key), frozen[key]) for key in drift]}；"
            "证据记录的参数必须是执行链实际使用的参数"
        )
    return {
        "frozen": dict(frozen),
        "observed": observed,
        "equal_across_arms": True,
        "matches_frozen": True,
        "read_from": "两臂宿主实例属性（构造后、执行前回读）",
    }


def _scenario_document(assembly: Any) -> Dict[str, Any]:
    """场景输入登记（交易日/价源/配额/比例/期初现金/整手）。"""
    return {
        "trade_dates": [str(item.date()) for item in assembly.scenario.date_list()],
        "prices": [
            {"date": str(pd.Timestamp(date).date()), "prices": dict(prices)}
            for date, prices in assembly.scenario.prices
        ],
        "combo_quota": assembly.scenario.combo_quota,
        "split_ratio": assembly.scenario.split_ratio,
        "physical_initial_capital": assembly.scenario.physical_initial_capital,
        "lot_size": LOT_SIZE,
    }


def _g4_gate_document(assembly: Any) -> Dict[str, Any]:
    """G4 位模式门登记（单次派生 NAV 与单袖 NAV 的位模式对照）。"""
    single_bits = assembly.single.g4_evidence.merged_nav_bits
    split_bits = assembly.split.g4_evidence.merged_nav_bits
    return {
        "definition": "先合并各袖货币总值、调用一次 calculate_nav，再与单袖 NAV 位模式比较",
        "tolerance": 0,
        "single_arm_nav": assembly.single.merged_nav,
        "single_arm_nav_bits": single_bits,
        "split_merged_nav": assembly.split.merged_nav,
        "split_merged_nav_bits": split_bits,
        "bitwise_equal": single_bits == split_bits,
        "assert_entry": "core/accounting/g4_nav.py::assert_g4_nav_bitwise_equal",
    }


def _config_check(assembly: Any, report: Any) -> Dict[str, Any]:
    """配置核对登记（完整冻结键值 + 重算指纹；Codex-02）。"""
    observed_meta = {
        arm: json.loads(
            (assembly.writer.out_dir(arm, assembly.batch_id) / "batch_meta.json").read_text(
                encoding="utf-8"
            )
        )
        for arm in (ARM_SINGLE, ARM_SPLIT)
    }
    config_keys = sorted(config_fingerprint_keys(list(EXPECTED_BATCH_CONFIG)))
    check = {
        "expected": dict(EXPECTED_BATCH_CONFIG),
        "dynamic_keys": list(DYNAMIC_CONFIG_KEYS),
        "observed": {arm: dict(payload["config"]) for arm, payload in observed_meta.items()},
        "fingerprint_keys": config_keys,
        "fingerprint_key_count": len(config_keys),
        "recomputed_fingerprint": report.config_fingerprint_recomputed,
        "recomputed_matches_recorded": all(
            payload["config_fingerprint"] == report.config_fingerprint_recomputed
            for payload in observed_meta.values()
        ),
        "note": "完整键值清单为独立冻结表；重算指纹来自 fingerprint_keys.fingerprint（单一来源）",
    }
    if not check["recomputed_matches_recorded"]:  # pragma: no cover - 比较器已先行拒绝
        raise _error("config_fingerprint 与 config 重算值不一致")
    return check


def _arm_payload(arm_view: Any) -> Dict[str, Any]:
    """单臂事实投影（物理/虚拟/generation；全部字段来自执行链观测）。"""
    return {
        "arm": arm_view.arm,
        "sleeve_ids": list(arm_view.sleeve_ids),
        "settlement_dates": list(arm_view.settlement_dates),
        "physical": {
            "initial_capital": arm_view.physical_initial_capital,
            "cash": arm_view.physical_cash,
            "positions": [list(item) for item in arm_view.physical_positions],
            "total_value": arm_view.physical_total_value,
            "source_lots": [list(item) for item in arm_view.physical_source_lots],
            "fills": [
                {
                    "ts_code": fill.ts_code,
                    "order_target": fill.order_target,
                    "requested_shares": fill.requested_shares,
                    "shares": fill.shares,
                    "price": fill.price,
                    "fee": fill.fee,
                    "cash_before": fill.cash_before,
                    "cash_after": fill.cash_after,
                    "fill_key": fill.fill_key,
                    "source_lot_id": fill.source_lot_id,
                }
                for fill in arm_view.fills
            ],
        },
        "virtual": {
            "cash_by_sleeve": [list(item) for item in arm_view.virtual_cash_by_sleeve],
            "total_value_by_sleeve": [
                list(item) for item in arm_view.virtual_total_value_by_sleeve
            ],
            "source_lots": [list(item) for item in arm_view.virtual_source_lots],
            "cash_sum": arm_view.merged_virtual_cash,
            "merged_total_value": arm_view.merged_total_value,
            "merged_nav": arm_view.merged_nav,
            "merged_nav_bits": arm_view.g4_evidence.merged_nav_bits,
        },
        "generation": {
            "generation_id": arm_view.generation.generation_id,
            "runs_root": arm_view.generation.runs_root,
            "out_dir": arm_view.generation.out_dir,
            "signal_fingerprint": arm_view.generation.signal_fingerprint,
            "config_fingerprint": arm_view.generation.config_fingerprint,
            "pre_write_absent": arm_view.generation.pre_write_absent,
            "file_hashes": [list(item) for item in arm_view.generation.file_hashes],
        },
    }


def build_self_split_evidence(
    *, run_id: str, arms_root: Path, normalized_dir: Optional[Path] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """自拆双臂 + 逐位比较（G4 候选门）：返回 ``(self_split, self_split_diff)``。

    Args:
        run_id: 证据运行标识（两臂根为其派生类）。
        arms_root: 两臂公共基目录。
        normalized_dir: 给定则把两侧实际比较的规范化字节落盘到该目录（自含证据）。
    """
    scenario = SelfSplitScenario()
    scenario_parameters = _scenario_parameters_document(scenario)
    assembly = run_self_split_arms(
        scenario=scenario,
        run_id=f"{run_id}-arms",
        base_dir=arms_root,
        writer_config=WRITER_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": f"{run_id}-syn"},
    )
    # 差异不在本函数抛错：先把差异清单/文件状态落成工件，由编排层按 verdict 决定
    # 退出码（失败路径也必须留下可复核的差异证据，A-R2 口径）。
    report = compare_assemblies(
        assembly,
        required_config_keys=sorted(config_fingerprint_keys(list(EXPECTED_BATCH_CONFIG))),
        expected_config=dict(EXPECTED_BATCH_CONFIG),
        dynamic_config_keys=DYNAMIC_CONFIG_KEYS,
        capture_dir=normalized_dir,
    )
    config_check = _config_check(assembly, report)
    host_parameters = _host_parameters_document(assembly)
    host_parameters["scenario"] = scenario_parameters

    self_split = {
        "case": "self_split_consistency",
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "batch_id": assembly.batch_id,
        "scenario": _scenario_document(assembly),
        "daily_physical_values": [list(item) for item in assembly.daily_physical_values],
        "arms": [_arm_payload(assembly.single), _arm_payload(assembly.split)],
        "g4_gate": _g4_gate_document(assembly),
        "verdict": "PASS" if report.ok else "FAIL",
        "boundary": (
            "合成对账边界内的候选门结果：不构成 P6.5 阶段出口、不代表 B 区转正或"
            "生产多袖能力；attribution/topk_detail 未被比较（见 self_split_diff）"
        ),
    }
    if not self_split["g4_gate"]["bitwise_equal"]:  # pragma: no cover - 比较器已先行拒绝
        raise _error("G4 位模式不一致：合并派生 NAV 与单袖 NAV 不同")

    self_split_diff = {
        "case": "self_split_bitwise_diff",
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "batch_id": assembly.batch_id,
        "tolerance": 0,
        "excluded_fields": list(report.excluded_fields),
        "exclusion_reasons": dict(EXCLUSION_REASONS),
        "excluded_fields_frozen": list(EXCLUDED_META_FIELDS),
        "differences": [item.describe() for item in report.differences],
        "difference_count": len(report.differences),
        "file_status": [list(item) for item in report.file_status],
        "file_sha256": [list(item) for item in report.file_sha256],
        "not_compared": list(report.not_compared()),
        "compared_summary_columns": list(report.compared_summary_columns),
        "fingerprint_key_count": report.fingerprint_key_count,
        "config_check": config_check,
        "fold_closure": dict(report.fold_closure),
        "normalized_files": [list(item) for item in report.normalized_files],
        "verdict": "PASS" if report.ok else "FAIL",
        "wording_rule": (
            "not_compared 项不得写成逐位一致；attribution 为两侧同源缺失、"
            "topk_detail 为 runs 契约可选缺失"
        ),
    }
    self_split["host_parameters"] = host_parameters
    return self_split, self_split_diff


# ────────────────────────── 编排 ──────────────────────────


def default_run_id(*, date: Optional[str] = None, salt: str = "") -> str:
    """ASCII 运行标识：阶段 + 日期 + 短内容指纹（不含主机名/墙钟噪声）。"""
    stamp = date or datetime.now().strftime("%Y%m%d")
    if not stamp.isdigit() or len(stamp) != 8:
        raise _error(f"日期必须为 YYYYMMDD，实际 {stamp!r}")
    digest = _sha256_bytes(f"{STAGE}-T4-{stamp}-{salt}".encode("utf-8"))[:8]
    return f"v2p65_t4_{stamp}_{digest}"


def collect_t4_evidence(
    *,
    root: Path,
    run_id: str,
    code_state: Mapping[str, Any],
    command: str,
    scenario_notes: Optional[Mapping[str, Any]] = None,
) -> EvidenceBundle:
    """在 ``root/<run_id>/`` 下产出全部证据文件并写自含 manifest。

    实现口径（T4 复审 A-R2/B-R2）：
    - **失败即零残留**：装配失败时删除本次创建的 ``root/<run_id>/``（装配成功后才保留，
      确保目录内 generation manifest 的绝对路径与最终位置一致——不做改名搬迁）；
    - **失败路径也留证据**：两臂逐位差异不在此抛错，而是写进 ``self_split_diff.json``
      并把 manifest verdict 记 FAIL，最后抛 ``EvidenceVerdictError``（驱动映射退出码 3）；
    - ``exclusion_table`` 记录来源模块与内容哈希（可复核"运行前冻结"）。

    Args:
        root: 证据根（通常是 ``data/frozen_reference/v2_p65``）
        run_id: ASCII 运行标识（目录名）
        code_state: ``{git_commit, git_dirty}``（真实 HEAD 与脏标记）
        command: 复跑命令（写入 manifest）
        scenario_notes: 附加登记（例如验收日志路径、环境/边界说明）

    Raises:
        P65EvidenceError: 输入/断言失败、目录已存在（禁止覆盖既有证据）。
        EvidenceVerdictError: 证据已落盘但候选门判定为 FAIL（含差异清单路径）。
    """
    # 单段路径身份 + 解析后包含性校验，**先于任何 mkdir**（外部评审 T4-Codex-05：
    # 原实现先建 root/run_id，深层写入层才拒绝，证据根之外会留下目录）。
    try:
        run_id = require_ascii_id(run_id, "run_id")
    except ValueError as error:  # ArmIsolationError 是 ValueError 子类
        raise _error(f"run_id 非法：{error}") from error
    root = Path(root)
    target = root / run_id
    if target.resolve().parent != root.resolve():
        raise _error(
            f"run_id 解析后越出证据根（run_id={run_id!r} 目标={target.resolve()}）；"
            "拒绝在证据根之外创建任何目录"
        )
    if target.exists():
        raise _error(f"证据目录已存在 {target}；禁止覆盖既有证据（请使用新 run_id）")
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True)
    try:
        manifest = _assemble_evidence(
            target,
            run_id=run_id,
            code_state=code_state,
            command=command,
            scenario_notes=scenario_notes,
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    verdict = dict(manifest["verdict"])
    files = _tree_hashes(target)
    bundle = EvidenceBundle(run_id=run_id, root=target, files=files, verdict=verdict)
    if not _verdict_passed(verdict):
        raise EvidenceVerdictError(
            f"阶段={STAGE} T4 证据已落盘但候选门未通过：{verdict}；"
            f"差异清单见 {target / 'self_split_diff.json'}",
            run_id=run_id,
            root=target,
            verdict=verdict,
        )
    return bundle


def _verdict_passed(verdict: Mapping[str, Any]) -> bool:
    """候选门判定（四项全 PASS 才算通过；``not_compared`` 不参与判定）。"""
    return all(
        verdict.get(key) == "PASS"
        for key in ("zero_turnover", "allocation_conservation", "self_split", "self_split_bitwise")
    )


def _assemble_evidence(
    target: Path,
    *,
    run_id: str,
    code_state: Mapping[str, Any],
    command: str,
    scenario_notes: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """在 ``target`` 下装配全部证据并返回 manifest（不负责目录创建/失败清理）。"""
    zero = build_zero_turnover_evidence(run_id=run_id)
    allocation = build_allocation_conservation_evidence(run_id=run_id)
    self_split, self_split_diff = build_self_split_evidence(
        run_id=run_id,
        arms_root=target / "arms",
        normalized_dir=target / "normalized",
    )
    _write_json(target / "zero_turnover.json", zero)
    _write_json(target / "allocation_conservation.json", allocation)
    _write_json(target / "self_split.json", self_split)
    _write_json(target / "self_split_diff.json", self_split_diff)

    # 规范化字节本体落盘 + 摘要定位（外部评审 T4-Codex-04）：登记的每个 SHA 都指向
    # 本次比较实际使用的字节文件，并在此处从磁盘重算校验（SHA 与本体同源）。
    recorded = {
        relative: (left, right)
        for relative, left, right in (tuple(item) for item in self_split_diff["file_sha256"])
    }
    normalized_entries: List[Dict[str, Any]] = []
    for relative, left_rel, right_rel in (
        tuple(item) for item in self_split_diff["normalized_files"]
    ):
        left_file = target / left_rel
        right_file = target / right_rel
        if not (left_file.is_file() and right_file.is_file()):
            raise _error(f"规范化字节本体缺失：{left_rel} / {right_rel}")
        left_sha = _sha256_bytes(left_file.read_bytes())
        right_sha = _sha256_bytes(right_file.read_bytes())
        expected = recorded.get(relative)
        if expected is not None and (left_sha, right_sha) != expected:
            raise _error(
                f"{relative} 规范化字节本体与摘要不符：磁盘=({left_sha},{right_sha}) "
                f"登记={expected}"
            )
        normalized_entries.append(
            {
                "path": relative,
                "single_path": left_rel,
                "split_path": right_rel,
                "single_bytes": left_file.stat().st_size,
                "split_bytes": right_file.stat().st_size,
                "single_sha256": left_sha,
                "split_sha256": right_sha,
            }
        )
    if len(normalized_entries) != len(recorded):  # pragma: no cover - 落盘与比较同源
        raise _error(f"规范化字节条目数不一致：落盘={len(normalized_entries)} 比较={len(recorded)}")
    normalized = {
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": run_id,
        "note": (
            "两臂 runs 的规范化字节本体与 SHA-256（比较器口径：无压缩 Parquet / "
            "sort_keys JSON）；single_path/split_path 为包内相对路径，可直接重算校验"
        ),
        "format": "parquet-no-compression-no-index / json-sort-keys-utf8",
        "files": normalized_entries,
    }
    _write_json(target / "normalized_sha256.json", normalized)

    writer_path = Path(RUNS_WRITER_PATH)
    comparator_path = Path(COMPARATOR_PATH)
    exclusion_source = {
        "fields": list(EXCLUDED_META_FIELDS),
        "reasons": dict(EXCLUSION_REASONS),
        "source_module": str(comparator_path).replace("\\", "/"),
        "source_sha256": (
            _sha256_bytes(comparator_path.read_bytes()) if comparator_path.is_file() else ""
        ),
        "declared_at_import": True,
        "note": "排除表是模块级常量（导入即冻结）；本证据记录其来源与内容哈希以便复核",
    }
    manifest = {
        "schema_version": EVIDENCE_SCHEMA,
        "stage": STAGE,
        "task": "T4",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_state": dict(code_state),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "command": command,
        "inputs": {
            "scenario": self_split["scenario"],
            "host_parameters": dict(self_split["host_parameters"]),
            "s1": zero["inputs"],
            "s4": allocation["inputs"],
            "writer_config": dict(WRITER_CONFIG),
            "expected_batch_config": dict(EXPECTED_BATCH_CONFIG),
            "dynamic_config_keys": list(DYNAMIC_CONFIG_KEYS),
            "rates": {
                "commission_rate": COMMISSION_RATE,
                "min_commission": MIN_COMMISSION,
                "stamp_tax": STAMP_TAX,
                "slippage": SLIPPAGE,
            },
            "tolerances": {
                "i1_cash_absolute": CASH_TOLERANCE,
                "i2_shares": HOLDINGS_TOLERANCE_SHARES,
                "i3_nav_absolute": NAV_TOLERANCE,
                "g4_nav": 0,
                "arm_comparison": 0,
                "note": "I1/I3 为恒等式容差门；G4 与臂间比较为位模式 0 容差",
            },
            "exclusion_table": exclusion_source,
        },
        "runs_writer": {
            "module": str(writer_path).replace("\\", "/"),
            "sha256": _sha256_bytes(writer_path.read_bytes()) if writer_path.is_file() else "",
            "note": "复用既有 write_runs_batch，不新增第二套投影；本阶段未改 writer",
        },
        "scenario_notes": dict(scenario_notes or {}),
        "file_sha256": [list(item) for item in _tree_hashes(target, skip=("manifest.json",))],
        "verdict": {
            "zero_turnover": "PASS" if all(zero["assertions"].values()) else "FAIL",
            "allocation_conservation": (
                "PASS" if all(allocation["assertions"].values()) else "FAIL"
            ),
            "self_split": self_split["verdict"],
            "self_split_bitwise": self_split_diff["verdict"],
            "not_compared": self_split_diff["not_compared"],
            "difference_count": self_split_diff["difference_count"],
            "fold_count": len(self_split_diff["fold_closure"]["expected"]),
            "normalized_byte_pairs": len(normalized_entries),
            "boundary": self_split["boundary"],
        },
    }
    _write_json(target / "manifest.json", manifest)
    return manifest


__all__ = (
    "EVIDENCE_ROOT",
    "EVIDENCE_SCHEMA",
    "EvidenceBundle",
    "EvidenceVerdictError",
    "P65EvidenceError",
    "WRITER_CONFIG",
    "build_allocation_conservation_evidence",
    "build_self_split_evidence",
    "build_zero_turnover_evidence",
    "collect_t4_evidence",
    "default_run_id",
)

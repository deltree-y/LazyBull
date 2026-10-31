# -*- coding: utf-8 -*-
"""v2 P4 T5 三臂隔离、单日驱动与完整决策输入采集（阶段专用）。

数据准备必须由 D4 单一协调者先行完成。本模块不为旧臂补数据：legacy neutral 与
v2 shadow 的数据根写方法和客户端网络方法均 fail-fast；现役臂只在显式生产授权
上下文中由外层驱动。默认函数均为隔离实现，真实窗口不得由本文件自行启动。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_gate1_identity import collect_execution_identity  # noqa: E402
from scripts.v2_p4.t5_migration import (  # noqa: E402
    _account_projection_matches,
    migrate_isolated,
)
from scripts.v2_p4.t05b_idempotency_probe import (  # noqa: E402
    CallRecorder,
    wrap_client_no_network,
    wrap_storage_no_write,
)
from src.lazybull.common.signal_factory import create_signal as create_legacy_signal  # noqa: E402
from src.lazybull.common.trading_config import TradingConfig as LegacyTradingConfig  # noqa: E402
from src.lazybull.paper import PaperStorage, PaperTradingRunner  # noqa: E402
from src.lazybull.paper.runtime import (  # noqa: E402
    PaperTradeRuntimeContext,
    execute_trade_workflow,
)
from src.lazybull.v2.core.accounting.engine_state import export_state, state_to_blob  # noqa: E402
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger  # noqa: E402
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML  # noqa: E402
from src.lazybull.v2.hosts.paper import entrypoints  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import (  # noqa: E402
    load_paper_config,
    resolve_engine,
)
from src.lazybull.v2.hosts.paper.factory import create_paper_host  # noqa: E402
from src.lazybull.v2.hosts.paper.shadow import PaperShadowAccumulator  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402

ARM_ACTIVE = "legacy_active"
ARM_NEUTRAL = "legacy_neutral"
ARM_V2 = "v2_shadow"
ARM_NAMES = (ARM_ACTIVE, ARM_NEUTRAL, ARM_V2)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    root: Path
    engine: str
    informational_only: bool


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """递归把映射键转字符串；值仍交给 json 的 default=str。"""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def assert_neutral_exposure(flat: Mapping[str, Any]) -> None:
    """旧 neutral 启动硬断言：exposure_policy 必须显式/规范化后为 null。"""
    if flat.get("exposure_policy") not in (None, ""):
        raise RuntimeError(
            f"旧 neutral 参照臂 exposure 非 null: {flat.get('exposure_policy')!r}，拒绝启动"
        )


ARM_INIT_MARKER = "arms_initialized.json"
V2_STAGING_DIR = ".staging_v2_shadow"
ARM_ENGINES: Dict[str, str] = {ARM_ACTIVE: "legacy", ARM_NEUTRAL: "legacy", ARM_V2: "v2"}
INFORMATIONAL_ONLY: Dict[str, bool] = {ARM_ACTIVE: True, ARM_NEUTRAL: False, ARM_V2: False}


def _set_arm_config(root: Path, engine: str, *, null_exposure: bool) -> Dict[str, Any]:
    storage = PaperStorage(root_path=str(root))
    config = storage.load_config()
    if config is None:
        raise RuntimeError(f"冻结配置缺失: {root / 'config.yaml'}")
    config["engine"] = engine
    if null_exposure:
        config["exposure_policy"] = None
        # exposure_policy 为空时不得保留回补开关，否则旧链拒绝“无政策源却回补”。
        config["exposure_replenish"] = False
    storage.save_config(config)
    persisted = load_paper_config(root)
    if resolve_engine(persisted) != engine:
        raise RuntimeError(
            f"隔离根 engine 未生效: {root} 期望={engine} 实际={persisted.get('engine')!r}"
        )
    return persisted


def _assert_frozen_source(frozen_paper_root: Path, gate1: Mapping[str, Any]) -> Dict[str, Any]:
    """冻结源账户必须与门 1 身份一致；不一致禁止作为三臂起点。"""
    account_path = frozen_paper_root / "state" / "account.json"
    if not account_path.is_file():
        raise FileNotFoundError(f"冻结账户缺失: {account_path}")
    account = json.loads(account_path.read_text(encoding="utf-8"))
    identity = gate1.get("account") or {}
    source_hash = _sha256(account_path)
    if identity.get("sha256") and identity["sha256"] != source_hash:
        raise RuntimeError(
            f"冻结源账户与门 1 身份不一致: {account_path} sha256={source_hash} "
            f"gate1={identity['sha256']}"
        )
    if identity.get("last_update") and identity.get("last_update") != account.get("last_update"):
        raise RuntimeError(
            f"冻结源账户末日与门 1 身份不一致: {account.get('last_update')} "
            f"vs {identity.get('last_update')}"
        )
    return {"account_sha256": source_hash, "account_last_update": account.get("last_update")}


def _frozen_valuation(gate1: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """门 1 冻结时点的 D10 估值：迁移必须绑定该日价格，不允许首个影子日回填。"""
    holdings = gate1.get("holdings_validity") or {}
    if not holdings.get("market_date") or not holdings.get("positions"):
        return None
    marks = {
        str(row["ts_code"]): row["mark_price"]
        for row in holdings["positions"]
        if row.get("mark_price") is not None
    }
    if len(marks) != len(holdings["positions"]):
        raise RuntimeError("门 1 冻结估值缺部分持仓现价，禁止用新价格回填旧日状态")
    return {
        "market_date": str(holdings["market_date"]),
        "mark_prices": marks,
        "total_value": float(holdings["total_value"]),
        "content_fingerprint": holdings.get("content_fingerprint"),
    }


def _gate1_identity(gate1: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "account_sha256": (gate1.get("account") or {}).get("sha256"),
        "account_last_update": (gate1.get("account") or {}).get("last_update"),
        "data_state_id": (gate1.get("data_state") or {}).get("data_state_id"),
        "model_fingerprint": (gate1.get("model") or {}).get("content_fingerprint"),
        "execution_fingerprint": (gate1.get("execution") or {}).get("fingerprint"),
        "q11a_selected_days": ((gate1.get("q11a_restart") or {}).get("selected_days") or []),
    }


def _arm_state_summary(name: str, root: Path, engine: str) -> Dict[str, Any]:
    """单臂状态核对：配置 engine、账户、完整代（v2 才要求权威快照完整代）。"""
    if not root.is_dir():
        raise RuntimeError(f"三臂隔离根缺失: {name} {root}")
    config_engine = resolve_engine(load_paper_config(root))
    if config_engine != engine:
        raise RuntimeError(f"{name} 隔离根 engine 非 {engine}: {config_engine}")
    account_path = root / "state" / "account.json"
    if not account_path.is_file():
        raise RuntimeError(f"{name} 隔离根缺账户: {account_path}")
    account = json.loads(account_path.read_text(encoding="utf-8"))
    summary: Dict[str, Any] = {
        "root": str(root),
        "engine": config_engine,
        "informational_only": INFORMATIONAL_ONLY[name],
        "account_sha256": _sha256(account_path),
        "account_last_update": account.get("last_update"),
    }
    if engine == "v2":
        entrypoints.assert_complete_generation(PaperStateStore(root_path=str(root)))
        summary["complete_generation"] = True
    return summary


def load_init_marker(work_root: Path) -> Optional[Dict[str, Any]]:
    path = Path(work_root) / ARM_INIT_MARKER
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"三臂初始化记录形态非法: {path}")
    return payload


def verify_isolated_initialization(*, work_root: Path, gate1: Mapping[str, Any]) -> Dict[str, Any]:
    """已存在隔离目录的续跑裁决：初始化记录 + run 身份 + 最后完整代。"""
    work_root = Path(work_root)
    marker = load_init_marker(work_root)
    if marker is None:
        existing = [name for name in ARM_NAMES if (work_root / name).exists()]
        raise FileExistsError(
            "三臂根已存在但缺初始化完成记录（半初始化），禁止覆盖或跳过缺失臂续跑，"
            f"失败现场保留: {existing}"
        )
    expected = _gate1_identity(gate1)
    if marker.get("gate1_identity") != expected:
        raise RuntimeError(
            "隔离三臂 run 身份与当前门 1 冻结身份不一致，禁止续跑: "
            f"marker={marker.get('gate1_identity')} gate1={expected}"
        )
    states = {
        name: _arm_state_summary(name, work_root / name, ARM_ENGINES[name]) for name in ARM_NAMES
    }
    return {"marker": marker, "arms": states}


def arm_specs_from_marker(marker: Mapping[str, Any]) -> Dict[str, ArmSpec]:
    return {
        name: ArmSpec(
            name,
            Path((marker.get("arms") or {}).get(name, {}).get("root") or ""),
            ARM_ENGINES[name],
            INFORMATIONAL_ONLY[name],
        )
        for name in ARM_NAMES
    }


def prepare_three_arms(
    *,
    frozen_paper_root: Path,
    work_root: Path,
    gate1: Mapping[str, Any],
    data_root: Optional[Path] = None,
    evidence: Optional[Path] = None,
    host_builder: Optional[Any] = None,
) -> Dict[str, ArmSpec]:
    """门 1 冻结后建立三臂起点：legacy 两臂复制冻结源，v2 臂复用既有迁移。

    已初始化则只核对身份/完整代并返回既有臂（不重迁、不重锚、不重置空账户）；
    半初始化目录一律拒绝续跑并保留现场。
    """
    if not gate1.get("gate1_frozen") or gate1.get("verdict") != "PASS":
        raise RuntimeError("门 1 未冻结（Q11-A 未完成），禁止准备三臂影子起点")
    work_root = Path(work_root)
    if load_init_marker(work_root) is not None:
        verified = verify_isolated_initialization(work_root=work_root, gate1=gate1)
        return arm_specs_from_marker(verified["marker"])
    existing = [name for name in ARM_NAMES if (work_root / name).exists()]
    if existing:
        raise FileExistsError(f"三臂根已存在但缺初始化完成记录（半初始化），禁止覆盖: {existing}")
    source = _assert_frozen_source(Path(frozen_paper_root), gate1)
    source_hash = source["account_sha256"]
    data_root = Path(data_root or ROOT / "data")
    work_root.mkdir(parents=True, exist_ok=True)
    specs: Dict[str, ArmSpec] = {}
    for name in (ARM_ACTIVE, ARM_NEUTRAL):
        root = work_root / name
        shutil.copytree(frozen_paper_root, root)
        _set_arm_config(root, ARM_ENGINES[name], null_exposure=(name == ARM_NEUTRAL))
        if _sha256(root / "state" / "account.json") != source_hash:
            raise RuntimeError(f"三臂起始账户内容不一致: {name}")
        specs[name] = ArmSpec(name, root, ARM_ENGINES[name], INFORMATIONAL_ONLY[name])
    assert_neutral_exposure(load_paper_config(specs[ARM_NEUTRAL].root))
    target = work_root / ARM_V2
    staging = work_root / V2_STAGING_DIR
    if staging.exists() or target.exists():
        raise FileExistsError(f"v2 迁移暂存/目标已存在，拒绝覆盖: {staging}")
    migration_evidence = Path(evidence or (work_root / "initial_migration"))
    migration = migrate_isolated(
        source=Path(frozen_paper_root),
        target=staging,
        data_root=data_root,
        evidence=migration_evidence,
        execute=True,
        host_builder=host_builder,
        valuation=_frozen_valuation(gate1),
    )
    if migration.get("verdict") != "PASS":
        raise RuntimeError(f"v2 影子首次迁移未通过: {migration.get('verdict')}")
    frozen_account = json.loads(
        (Path(frozen_paper_root) / "state" / "account.json").read_text(encoding="utf-8")
    )
    migrated_account = json.loads((staging / "state" / "account.json").read_text(encoding="utf-8"))
    # 允许 v2 内部映射字段变化，但现金/股数/买价/费用必须与冻结源等价（1e-6）。
    if not _account_projection_matches(frozen_account, migrated_account):
        raise RuntimeError("v2 迁移后账户投影与冻结源不等价（现金/股数/买价/费用）")
    _set_arm_config(staging, "v2", null_exposure=True)
    staging.rename(target)
    specs[ARM_V2] = ArmSpec(ARM_V2, target, "v2", INFORMATIONAL_ONLY[ARM_V2])
    marker = {
        "stage": "v2_p4_t5_three_arm_init",
        "initialized_at": _now(),
        "gate1_identity": _gate1_identity(gate1),
        "frozen_source": {"path": str(frozen_paper_root), **source},
        "migration": {
            "verdict": migration.get("verdict"),
            "engine_state_sha256": migration.get("engine_state_sha256"),
            "valuation": migration.get("valuation"),
            "account_projection_equivalent": migration.get("account_projection_equivalent"),
            "evidence": str(migration_evidence / "migration_result.json"),
        },
        "arms": {
            name: _arm_state_summary(name, specs[name].root, ARM_ENGINES[name])
            for name in ARM_NAMES
        },
        "note": "首个影子日恢复该权威状态；后续日只恢复推进后的影子状态，不重迁/不重锚。",
    }
    _dump(work_root / ARM_INIT_MARKER, marker)
    return specs


def _read_records(path: Path) -> Any:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_state(root: Path) -> Dict[str, Any]:
    if not root.is_dir():
        return {}
    return {
        path.name: _read_records(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix in {".json", ".parquet"}
    }


def _legacy_pre_state(storage: PaperStorage) -> Dict[str, Any]:
    account = storage.load_account_state()
    if account is None:
        raise RuntimeError(f"旧臂账户缺失: {storage.root_path}")
    positions = {
        code: {
            "shares": int(position.shares),
            "buy_date": position.buy_date,
            "buy_trade_price": float(position.buy_price),
            "buy_pnl_price": float(position.buy_pnl_price),
            "buy_cost_cash": float(position.shares * position.buy_price + position.buy_cost),
        }
        for code, position in sorted(account.positions.items())
    }
    return {
        "account": {"cash": float(account.cash), "positions": positions},
        "pending": {
            "pending_buys": _directory_state(storage.pending_buys_path),
            "pending_sells": _directory_state(storage.pending_sells_path),
            "instructions": _directory_state(storage.instructions_path),
        },
        "schedule_anchor": storage.load_rebalance_state() or {},
        "ranked_candidates": storage.load_ranked_candidates(),
        "stop_loss_state": storage.load_stop_loss_state() or {},
    }


def _partition_path(data_root: Path, layer: str, dataset: str, day: str, dashed: bool) -> Path:
    name = f"{day[:4]}-{day[4:6]}-{day[6:]}" if dashed else day
    return data_root / layer / dataset / f"{name}.parquet"


def _legacy_market(data_root: Path, day: str) -> Dict[str, Any]:
    feature_path = _partition_path(data_root, "features", "cs_infer", day, dashed=False)
    price_path = _partition_path(data_root, "clean", "daily", day, dashed=True)
    if not feature_path.is_file() or not price_path.is_file():
        raise FileNotFoundError(
            f"旧臂当日输入缺失: {day} features={feature_path.is_file()} prices={price_path.is_file()}"
        )
    return {
        "source": "cs_infer+clean.daily",
        "features": pd.read_parquet(feature_path).to_dict("records"),
        "prices": pd.read_parquet(price_path).to_dict("records"),
    }


def _effective_identity(flat: Mapping[str, Any]) -> Dict[str, Any]:
    execution = collect_execution_identity(flat)["config"]
    execution.pop("engine", None)  # 实现臂名不是决策数值配置
    execution["exposure_policy"] = None
    execution["exposure_replenish"] = False
    return execution


def build_legacy_decision_input(
    *, storage: PaperStorage, data_root: Path, day: str, flat: Mapping[str, Any], merged_target: Any
) -> Dict[str, Any]:
    state = _legacy_pre_state(storage)
    return {
        "date_phase": {"trade_date": day, "phase": "T0_T1"},
        "merged_target": merged_target,
        **state,
        "market_data": _legacy_market(data_root, day),
        "effective_identity": _effective_identity(flat),
    }


class ObservingMerger(SingleSleevePortfolioMerger):
    """只读记录实际合并目标；合并算法与调用次数仍由原生实现承担。"""

    def __init__(self, observations: List[Any]) -> None:
        super().__init__()
        self._observations = observations

    def merge(self, targets, current):  # type: ignore[no-untyped-def]
        merged = super().merge(targets, current)
        self._observations.append(
            {
                "sleeve_id": merged.sleeve_id,
                "as_of_date": str(merged.as_of_date),
                "target_weights": {
                    code.value: weight.value for code, weight in merged.target_weights.items()
                },
            }
        )
        return merged


def _features_for_day(features_by_date: Mapping[str, Any], date: Any) -> Any:
    """特征字典键为 YYYYMMDD；禁止用 Timestamp 直接查（会恒为 None）。"""
    return features_by_date.get(str(pd.Timestamp(date).strftime("%Y%m%d")))


class V2DecisionCapture:
    """在真实 run_step 前捕获状态，并在原生 merge 接缝补记合并目标。"""

    def __init__(
        self,
        host: Any,
        day: str,
        flat: Mapping[str, Any],
        targets: Optional[List[Any]] = None,
    ) -> None:
        self.host = host
        self.day = day
        self.flat = flat
        self.targets: List[Any] = targets if targets is not None else []
        self.pre_state: Optional[Dict[str, Any]] = None
        self.market_data: Optional[Dict[str, Any]] = None
        self._original = BacktestEngineML.run_step

    def __enter__(self) -> "V2DecisionCapture":
        capture = self

        def observed(engine: BacktestEngineML, date, trading_dates, price_data, *args, **kwargs):  # type: ignore[no-untyped-def]
            blob = state_to_blob(export_state(engine))["fields"]
            current_day = pd.Timestamp(date).strftime("%Y%m%d")
            features = _features_for_day(engine.features_by_date, date)
            prices = price_data
            if isinstance(price_data, pd.DataFrame) and "trade_date" in price_data:
                prices = price_data[price_data["trade_date"].astype(str) == current_day]
            capture.pre_state = {
                "account": {
                    "cash": blob["current_capital"],
                    "positions": blob["positions"],
                },
                "pending": {
                    "signals": blob["pending_signals"],
                    "stop_loss_sells": blob["pending_stop_loss_sells"],
                    "condition_sells": blob["pending_condition_sells"],
                    "orders": blob["pending_order_manager"],
                },
                "schedule_anchor": {
                    "signal_dates": blob["_signal_dates"],
                    "cycle_anchor_idx": blob["_cycle_anchor_idx"],
                    "live_anchor_state": blob["live_anchor_state"],
                },
                "ranked_candidates": blob["_last_ranked_candidates"],
                "stop_loss_state": blob["stop_loss_monitor"],
            }
            capture.market_data = {
                "source": "panel+clean.daily",
                "features": None if features is None else features.to_dict("records"),
                "prices": prices.to_dict("records") if isinstance(prices, pd.DataFrame) else prices,
            }
            return capture._original(engine, date, trading_dates, price_data, *args, **kwargs)

        BacktestEngineML.run_step = observed  # type: ignore[method-assign]
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        BacktestEngineML.run_step = self._original  # type: ignore[method-assign]

    def decision_input(self) -> Dict[str, Any]:
        if self.pre_state is None or self.market_data is None:
            raise RuntimeError(f"v2 未捕获 run_step 决策前状态: {self.day}")
        return {
            "date_phase": {"trade_date": self.day, "phase": "T0_T1"},
            "merged_target": self.targets,
            **self.pre_state,
            "market_data": self.market_data,
            "effective_identity": _effective_identity(self.flat),
        }


def create_legacy_runtime(
    spec: ArmSpec, data_root: Path, recorder: CallRecorder
) -> PaperTradeRuntimeContext:
    """不改 legacy 源码，直接以公共值对象装配隔离根运行时并加双守卫。"""
    storage = PaperStorage(root_path=str(spec.root))
    config = storage.load_config()
    if config is None:
        raise RuntimeError(f"旧臂配置缺失: {spec.root}")
    if spec.name == ARM_NEUTRAL:
        assert_neutral_exposure(config)
    trading_config = LegacyTradingConfig.from_dict(config)
    runner = PaperTradingRunner(
        signal=create_legacy_signal(trading_config),
        initial_capital=float(config["initial_capital"]),
        data_root=str(data_root),
        paper_root=str(spec.root),
        position_sizing=str(config.get("position_sizing", "equal")),
        horizon=int(config.get("horizon", 20)),
        verbose=False,
    )
    wrap_client_no_network(runner.client, recorder)
    wrap_storage_no_write(runner.storage, recorder)
    return PaperTradeRuntimeContext(storage, config, trading_config, runner)


def run_legacy_arm(
    spec: ArmSpec, *, data_root: Path, day: str, evidence_dir: Path
) -> Dict[str, Any]:
    recorder = CallRecorder()
    runtime = create_legacy_runtime(spec, data_root, recorder)
    pre_state = _legacy_pre_state(runtime.storage)
    targets: List[Any] = []
    original_normalize = runtime.runner._normalize_signals

    def observed_normalize(self, signals, trade_date):  # type: ignore[no-untyped-def]
        normalized = original_normalize(signals, trade_date)
        targets.append(normalized)
        return normalized

    runtime.runner._normalize_signals = types.MethodType(observed_normalize, runtime.runner)
    result = execute_trade_workflow(day, runtime=runtime)
    network_calls = sum(event["kind"] == "network" for event in recorder.events)
    write_calls = sum(event["kind"] == "write" for event in recorder.events)
    if network_calls or write_calls:
        raise RuntimeError(
            f"旧臂突破 D4 单协调者边界: network={network_calls}, data_writes={write_calls}"
        )
    if result.t0_status.startswith("error"):
        raise RuntimeError(f"旧臂 {day} 执行失败: {result.t0_status} {result.feature_error_detail}")
    decision = build_legacy_decision_input(
        storage=runtime.storage,
        data_root=data_root,
        day=result.corrected_date,
        flat=runtime.config,
        merged_target=targets,
    )
    # 用运行前状态替换 build 函数读取的当前状态；完整指纹必须对应 decide 之前。
    decision.update(pre_state)
    payload = {
        "arm": spec.name,
        "date": result.corrected_date,
        "decision_input": decision,
        "t0_status": result.t0_status,
        "fills": result.t1_actions,
        "network_calls": network_calls,
        "data_write_calls": write_calls,
    }
    _dump(evidence_dir / f"{spec.name}_{result.corrected_date}.json", payload)
    return payload


def run_v2_arm(
    spec: ArmSpec,
    *,
    data_root: Path,
    day: str,
    evidence_dir: Path,
    accumulator: PaperShadowAccumulator | None = None,
) -> Dict[str, Any]:
    recorder = CallRecorder()
    flat = load_paper_config(spec.root)
    targets: List[Any] = []
    host = create_paper_host(
        store=spec.root,
        config_flat=flat,
        data_root=str(data_root),
        portfolio_merger=ObservingMerger(targets),
        verbose=False,
        schedule_mode="live",
    )
    wrap_storage_no_write(host.data_storage, recorder)
    capture = V2DecisionCapture(host, day, flat, targets=targets)
    with capture:
        host.run_t1(day)
    result = host.last_day_result
    if result is None or not result.driven:
        gaps = [] if result is None or result.readiness is None else result.readiness.gaps
        raise RuntimeError(f"v2 shadow {day} 未真实驱动: {gaps}")
    decision = capture.decision_input()
    accumulated = accumulator.append_from_host(host, result.corrected_date) if accumulator else None
    payload = {
        "arm": spec.name,
        "date": result.corrected_date,
        "decision_input": decision,
        "t0_status": result.t0_status,
        "fills": result.fills,
        "orders": result.push_text,
        "accumulated": accumulated,
        "network_calls": sum(event["kind"] == "network" for event in recorder.events),
        "data_write_calls": sum(event["kind"] == "write" for event in recorder.events),
    }
    _dump(evidence_dir / f"{spec.name}_{result.corrected_date}.json", payload)
    return payload

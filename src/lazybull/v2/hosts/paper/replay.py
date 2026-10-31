"""P4-T4 纸面历史回放：窗口锚逐日步进、完整快照与 runs 单 fold 出口。

回放只消费宿主显式注入的本地数据面。每日经 PaperHost 公共 ``run_t0`` /
``run_t1(fills)`` 双视图驱动同一指令链，随后把权威完整快照另存为逐日检查点；
不得使用回测成交反投影或 T3 证明性订单充当指令输入。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.lazybull.v2.common.table_schema import SNAPSHOT_KEYS
from src.lazybull.v2.common.types import FeatureQuery, TradeDate
from src.lazybull.v2.core.accounting.engine_state import dumps_blob, loads_blob
from src.lazybull.v2.core.accounting.exports import get_execution_attribution, get_trades
from src.lazybull.v2.core.accounting.nav import generate_nav_curve
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.hosts.backtest.replay import _compute_replay_metrics
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch
from src.lazybull.v2.hosts.paper.host import PaperHost
from src.lazybull.v2.store.row_domain import STRUCTURAL, STRUCTURAL_FROM, row_domain_for_date


class PaperReplayError(RuntimeError):
    """纸面回放窗口、状态或产物契约错误。"""


class ReplayRowDomainError(PaperReplayError):
    """回放查询日行域与窗口声明行域不一致。"""


class ExecutionConfigMismatchError(PaperReplayError):
    """双宿主执行配置指纹不一致，禁止进入等价比较。"""


def _date_text(value: Any) -> str:
    return str(value).strip().replace("-", "")


def resolve_replay_window(
    host: PaperHost, *, start_date: str, end_date: str, frozen_through: str
) -> List[pd.Timestamp]:
    """解析并校验 D5-甲窗口：非空、属于 structural 段且不越过追平冻结日。"""
    if host.schedule_mode != "replay":
        raise PaperReplayError("PaperHost 必须以 schedule_mode='replay' 装配")
    start, end, frozen = map(_date_text, (start_date, end_date, frozen_through))
    if start < STRUCTURAL_FROM or end > frozen or start > end:
        raise ReplayRowDomainError(
            f"回放窗口非法: [{start}, {end}]，须 ⊆ [{STRUCTURAL_FROM}, {frozen}]"
        )
    window = [day for day in host.trading_calendar if start <= day.strftime("%Y%m%d") <= end]
    if not window:
        raise PaperReplayError(f"回放窗口无交易日: [{start}, {end}]")
    for day in window:
        if row_domain_for_date(day.strftime("%Y%m%d")) != STRUCTURAL:
            raise ReplayRowDomainError(f"回放窗口含非 structural 日: {day}")
    if window != host.replay_calendar:
        raise PaperReplayError("宿主 replay_calendar 与请求窗口不一致，禁止重锚或混用日历")
    return window


def _partition_row_domain(store: Any, record: Mapping[str, Any]) -> str:
    relative = str(record.get("path", "")).replace("\\", "/")
    if not relative.startswith("features/"):
        raise ReplayRowDomainError(f"行域回执含非 features 文件: {relative!r}")
    meta = store.manifest.partition_meta(relative[len("features/") :])
    if meta is None or meta.get("row_domain") not in {STRUCTURAL, "label_filtered"}:
        raise ReplayRowDomainError(f"实际读取分区未登记有效行域: {relative!r}")
    return str(meta["row_domain"])


def assert_single_day_row_domain(
    store: Any, date: str, *, expected_domain: str = STRUCTURAL
) -> Dict[str, Any]:
    """执行真实 ``load_features`` 单日查询并核验实际读取分区的 manifest 行域。"""
    date = _date_text(date)
    declared = row_domain_for_date(date)
    if declared != expected_domain:
        raise ReplayRowDomainError(
            f"查询日 {date} 声明行域 {declared!r} ≠ 窗口行域 {expected_domain!r}"
        )
    store.load_features(
        FeatureQuery(
            columns=("trade_date", "ts_code"),
            start_date=TradeDate.from_str(date),
            end_date=TradeDate.from_str(date),
        )
    )
    receipt = store.last_read_receipt
    if receipt is None or receipt.start_date != date or receipt.end_date != date:
        raise ReplayRowDomainError(f"查询日 {date} 缺有效单日读取回执")
    if not receipt.files:
        raise ReplayRowDomainError(f"查询日 {date} 未实际读取任何 panel 分区")
    rows = []
    for record in receipt.files:
        actual = _partition_row_domain(store, record)
        if actual != expected_domain:
            raise ReplayRowDomainError(
                f"查询日 {date} 实际行域 {actual!r} ≠ 窗口行域 {expected_domain!r}: "
                f"{record.get('path')}"
            )
        rows.append(
            {
                "path": str(record["path"]).replace("\\", "/"),
                "sha256": str(record["sha256"]),
                "row_domain": actual,
            }
        )
    return {
        "date": date,
        "query_start": receipt.start_date,
        "query_end": receipt.end_date,
        "declared_row_domain": expected_domain,
        "actual_row_domains": sorted({row["row_domain"] for row in rows}),
        "files": rows,
        "passed": True,
    }


def execution_config_from_engine(
    engine: Any, *, model_version: Any, buy_price: str = "close"
) -> Dict[str, Any]:
    """从实际引擎实例提取 D7 执行配置，覆盖价格、排期、顺延、风险与成本参数。"""
    cost = engine.cost_model
    config = {
        "model_version": str(model_version),
        "buy_price": str(buy_price),
        "sell_price": str(engine.sell_timing),
        "top_n": int(engine.signal.top_n),
        "initial_capital": float(engine.initial_capital),
        "rebalance_freq": int(engine.rebalance_freq),
        "holding_period": int(engine.holding_period),
        "stagger_tranches": int(engine.stagger_tranches),
        "enable_early_rebalance_on_empty": bool(engine.enable_early_rebalance_on_empty),
        "enable_trailing_early_rebalance": bool(engine.enable_trailing_early_rebalance),
        "enable_pending_order": bool(engine.enable_pending_order),
        "enable_position_completion": bool(engine.enable_position_completion),
        "completion_window_days": int(engine.completion_window_days),
        "enable_risk_budget": bool(engine.enable_risk_budget),
        "vol_window": int(engine.vol_window),
        "vol_epsilon": float(engine.vol_epsilon),
        "position_sizing": str(engine.position_sizing),
        "kelly_vol_window": int(engine.kelly_vol_window),
        "kelly_max_leverage": float(engine.kelly_max_leverage),
        "min_buy_value_ratio": float(engine.min_buy_value_ratio),
        "max_weight_per_stock": engine.max_weight_per_stock,
        "max_per_industry": engine.max_per_industry,
        "commission_rate": float(cost.commission_rate),
        "min_commission": float(cost.min_commission),
        "stamp_tax": float(cost.stamp_tax),
        "slippage": float(cost.slippage),
        "merger_mode": (
            type(engine.portfolio_merger).__name__
            if getattr(engine, "portfolio_merger", None) is not None
            else "legacy_bypass"
        ),
        "merger_sleeve_id": getattr(engine, "merger_sleeve_id", None),
    }
    stop = engine.stop_loss_config
    config["stop_loss"] = (
        None
        if stop is None
        else {
            "drawdown_pct": float(stop.drawdown_pct),
            "consecutive_limit_down_days": int(stop.consecutive_limit_down_days),
        }
    )
    return config


def assert_execution_configs_match(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Dict[str, Any]:
    """执行配置逐键与指纹双重前置；任何差异均拒绝进入逐笔/NAV 等价比较。"""
    left_fp, right_fp = fingerprint(dict(left)), fingerprint(dict(right))
    differences = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
    if differences or left_fp != right_fp:
        detail = {key: {"left": left.get(key), "right": right.get(key)} for key in differences}
        raise ExecutionConfigMismatchError(
            f"双宿主执行配置不一致，拒绝等价比较: {json.dumps(detail, default=str)}"
        )
    return {
        "passed": True,
        "left_fingerprint": left_fp,
        "right_fingerprint": right_fp,
        "config": dict(left),
    }


def _export_checkpoint(host: PaperHost, run_root: Path, date: str) -> Dict[str, Any]:
    data = host.store.load_engine_state_blob()
    if data is None:
        raise PaperReplayError(f"{date} 完成后缺权威引擎快照")
    blob = loads_blob(data)
    if dumps_blob(blob) != data:
        raise PaperReplayError(f"{date} 权威快照字节往返不一致")
    generation = blob.get("paper_generation") or {}
    if generation.get("trade_date") != date:
        raise PaperReplayError(f"{date} 快照代际不符: {generation!r}")
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{date}.bin"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return {
        "date": date,
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "roundtrip_exact": True,
    }


def _collect_result(
    host: PaperHost,
    start_date: str,
    end_date: str,
    split_num: int,
    daily_records: List[Dict[str, Any]],
    domain_records: List[Dict[str, Any]],
    checkpoints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    data = host.store.load_engine_state_blob()
    if data is None:
        raise PaperReplayError("回放结束缺权威完整快照")
    fields = loads_blob(data)["fields"]
    nav_curve = generate_nav_curve(
        fields["portfolio_values"], host.engine_kwargs["initial_capital"]
    )
    metrics = _compute_replay_metrics(
        nav_curve, start_date, end_date, host.trading_config.top_n, split_num
    )
    metrics["_nav_curve"] = nav_curve
    metrics["_trades"] = get_trades(fields["trades"])
    metrics["_execution_attribution"] = get_execution_attribution(
        fields["execution_attribution_records"]
    )
    metrics["_holdings_snapshot"] = pd.DataFrame(
        fields["holdings_snapshots"], columns=SNAPSHOT_KEYS
    )
    metrics["_paper_replay_days"] = daily_records
    metrics["_row_domain_assertions"] = domain_records
    metrics["_checkpoints"] = checkpoints
    metrics["_initial_account"] = {
        "cash": float(host.engine_kwargs["initial_capital"]),
        "positions": {},
        "source": "engine_initial_state",
    }
    engine = host._new_engine({})
    metrics["_execution_config"] = execution_config_from_engine(
        engine,
        model_version=host.trading_config.model_version,
        buy_price=host.trading_config.buy_price,
    )
    metrics["_merger_call_count"] = int(host.portfolio_merger.merge_call_count)
    metrics["_merger_net_call_count"] = int(host.portfolio_merger.net_call_count)
    return metrics


def run_paper_replay(
    host: PaperHost,
    *,
    run_root: Path,
    start_date: str,
    end_date: str,
    frozen_through: str,
    split_num: int = 0,
) -> Dict[str, Any]:
    """逐日执行 paper_replay；状态根必须严格位于 ``run_root/paper_state``。"""
    run_root = Path(run_root)
    if host.store.root_path != run_root / "paper_state":
        raise PaperReplayError("回放状态根必须为 run_root/paper_state，禁止写入生产 data/paper")
    if host.store.get_authority_last_date() is not None:
        raise PaperReplayError("回放 run_root 已含权威状态；请使用全新批次目录，禁止覆盖旧批")
    start_date, end_date = _date_text(start_date), _date_text(end_date)
    window = resolve_replay_window(
        host, start_date=start_date, end_date=end_date, frozen_through=frozen_through
    )
    daily_records: List[Dict[str, Any]] = []
    domain_records: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    for day in window:
        date = day.strftime("%Y%m%d")
        domain_records.append(assert_single_day_row_domain(host.panel_store, date))
        orders = host.run_t0(date)
        result = host.last_day_result
        if result is None or not result.driven:
            gaps = [] if result is None or result.readiness is None else result.readiness.gaps
            raise PaperReplayError(f"{date} 未真实驱动，readiness gaps={gaps}")
        fills = list(result.fills)
        host.run_t1(date, fills=fills)
        checkpoint = _export_checkpoint(host, run_root, date)
        checkpoints.append(checkpoint)
        daily_records.append(
            {
                "date": date,
                "orders": len(orders),
                "fills": len(fills),
                "nav": result.nav,
                "checkpoint_sha256": checkpoint["sha256"],
                "run_t1_fills_reconciled": True,
            }
        )
    return _collect_result(
        host, start_date, end_date, split_num, daily_records, domain_records, checkpoints
    )


def write_paper_replay_batch(
    result: Mapping[str, Any],
    *,
    batch_id: str,
    runs_root: Path,
    config: Dict[str, object],
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    model_version: Any,
    train_start: Optional[str],
    train_end: Optional[str],
    frozen_through: str,
    baseline_ref: Optional[str] = None,
) -> Path:
    """把 paper_replay 结果按单 fold 拓扑写出并立即通过 runs loader 硬校验。"""
    metrics = {key: value for key, value in result.items() if not key.startswith("_")}
    fold = {
        "split_index": 0,
        "model_version": model_version,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": result["bt_start"],
        "test_end": result["bt_end"],
        "metrics": metrics,
        "nav_curve": result["_nav_curve"],
        "trades": result["_trades"],
        "attribution": result["_execution_attribution"],
        "holdings_snapshot": result["_holdings_snapshot"],
    }
    notes = {
        "replay_window": json.dumps(
            {
                "start": result["bt_start"],
                "end": result["bt_end"],
                "trading_days": len(result["_paper_replay_days"]),
                "row_domain": STRUCTURAL,
                "frozen_through": frozen_through,
            },
            sort_keys=True,
        ),
        "replay_checkpoints": str(len(result["_checkpoints"])),
        "replay_row_domain_assertions": str(len(result["_row_domain_assertions"])),
        "replay_instruction_source": "engine_pending_signals_and_initial_account",
    }
    return write_runs_batch(
        [fold],
        batch_id=batch_id,
        runs_root=runs_root,
        config=config,
        code_state=code_state,
        data_state=data_state,
        baseline_ref=baseline_ref,
        arms=("paper_replay",),
        source="v2_p4_paper_replay",
        extra_meta_notes=notes,
        host_mode="paper_replay",
    )

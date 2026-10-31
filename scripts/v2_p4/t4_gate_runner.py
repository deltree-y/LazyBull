"""v2 P4 T4 门 2/门 3 正式执行器（只读生产数据态，阶段专用）。

写边界仅为 data/runs/ 新批目录与 data/frozen_reference/v2_p4/t4_<RUN_ID>/。
生产 data/paper 在运行前后逐文件 SHA-256 containment；生产 Storage 写方法全部
fail-fast，socket 网络连接同样 fail-fast。不会调用 ensure、下载、构建或标签刷新。
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.v2_p4.t4_compare_hosts import (  # noqa: E402
    canonical_nav,
    canonical_positions,
    canonical_trades,
    collect_shared_inputs,
    compare_nav,
    compare_positions,
    compare_trades,
    model_training_window,
    run_backtest_reference,
    write_backtest_batch,
)
from scripts.v2_p4.t05b_idempotency_probe import (  # noqa: E402
    CallRecorder,
    wrap_storage_no_write,
)
from scripts.v2_p4.t05b_live_anchor import snapshot_protected_root  # noqa: E402
from src.lazybull.common.config import get_data_root, get_logs_dir, get_paper_root  # noqa: E402
from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint  # noqa: E402
from src.lazybull.v2.evidence.runs_loader import load_runs_batch  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402
from src.lazybull.v2.hosts.paper.factory import _load_full_calendar, create_paper_host  # noqa: E402
from src.lazybull.v2.hosts.paper.host import PaperHost  # noqa: E402
from src.lazybull.v2.hosts.paper.replay import (  # noqa: E402
    assert_execution_configs_match,
    run_paper_replay,
    write_paper_replay_batch,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _latest_partition(directory: Path) -> Optional[str]:
    if not directory.exists():
        return None
    values = []
    for path in directory.iterdir():
        value = path.name if path.is_dir() else path.stem
        value = value.replace("-", "")
        if len(value) == 8 and value.isdigit():
            values.append(value)
    return max(values) if values else None


def capture_data_state(data_root: Path) -> Dict[str, Any]:
    store = PanelDataStore(data_root)
    snapshot = store.manifest.snapshot()
    sources = {
        "raw_daily": _latest_partition(data_root / "raw" / "daily"),
        "clean_daily": _latest_partition(data_root / "clean" / "daily"),
        "cs_train": _latest_partition(data_root / "features" / "cs_train"),
        "cs_infer": _latest_partition(data_root / "features" / "cs_infer"),
        "panel": _latest_partition(data_root / "features" / "panel"),
    }
    labels = {
        name: {
            "sealed_through": entry.get("sealed_through"),
            "partitions": len(entry.get("partitions") or {}),
        }
        for name, entry in (snapshot.get("labels") or {}).items()
    }
    identity = {
        "sources": sources,
        "labels": labels,
        "manifest_fingerprint": store.manifest.content_fingerprint(),
        "manifest_partitions": len(snapshot.get("partitions") or {}),
        "manifest_repairs": len(snapshot.get("repairs") or []),
        "row_domain_segments": snapshot.get("row_domain_segments"),
    }
    return {
        "data_state_id": fingerprint(identity),
        "sources": sources,
        "cs_train_latest": sources["cs_train"],
        "labels": labels,
        "panel": identity,
    }


def _code_state() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=_REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


@contextmanager
def _no_network(recorder: CallRecorder):
    original = socket.socket.connect

    def blocked(self, address, *args, **kwargs):
        recorder.record("network", "socket", "connect", {"address": str(address)})
        raise RuntimeError(f"T4 禁网守卫触发: socket.connect({address!r})")

    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original


def _build_host(
    *,
    config_flat: Mapping[str, Any],
    run_root: Path,
    start: str,
    end: str,
    storage: Storage,
    loader: DataLoader,
    calendar: Sequence[pd.Timestamp],
    enable_risk_budget: bool = False,
    excluded_dates: Sequence[str] = (),
) -> PaperHost:
    excluded = set(excluded_dates)
    effective_calendar = [day for day in calendar if day.strftime("%Y%m%d") not in excluded]
    window = [day for day in effective_calendar if start <= day.strftime("%Y%m%d") <= end]
    host = create_paper_host(
        store=str(run_root / "paper_state"),
        config_flat=config_flat,
        data_root=get_data_root(),
        data_storage=storage,
        loader=loader,
        trading_calendar=effective_calendar,
        schedule_mode="replay",
        replay_calendar=window,
        verbose=False,
    )
    host.engine_kwargs["enable_risk_budget"] = bool(enable_risk_budget)
    return host


def _nonempty_suspend_days(host: PaperHost, window: Sequence[pd.Timestamp]) -> List[Dict[str, Any]]:
    rows = []
    for day in window:
        date = day.strftime("%Y%m%d")
        frame = host.data_storage.load_raw_by_date("suspend", date)
        count = 0 if frame is None else len(frame)
        if count:
            rows.append({"date": date, "rows": int(count)})
    return rows


def _gate2(
    *,
    host: PaperHost,
    run_root: Path,
    evidence_dir: Path,
    start: str,
    end: str,
    frozen: str,
) -> Dict[str, Any]:
    window = host.replay_calendar
    paper = run_paper_replay(
        host, run_root=run_root, start_date=start, end_date=end, frozen_through=frozen
    )
    features, prices, input_receipts = collect_shared_inputs(host, window)
    reference = run_backtest_reference(
        host,
        features_by_date=features,
        price_data=prices,
        window=window,
        enable_risk_budget=True,
    )
    execution = assert_execution_configs_match(
        paper["_execution_config"], reference["_execution_config"]
    )
    trades, trade_detail = compare_trades(
        canonical_trades(paper["_trades"]),
        canonical_trades(reference["_trades"]),
        selected_count=host.trading_config.top_n,
        allow_tie_swaps=False,
    )
    nav, nav_detail = compare_nav(
        canonical_nav(paper["_nav_curve"]), canonical_nav(reference["_nav_curve"])
    )
    positions = compare_positions(
        canonical_positions(paper["_holdings_snapshot"]),
        canonical_positions(reference["_holdings_snapshot"]),
    )
    suspend_days = _nonempty_suspend_days(host, window)
    coverage = {
        "enable_risk_budget": True,
        "sell_timing_close": host.trading_config.sell_price == "close",
        "nonempty_suspend_calendar": bool(suspend_days),
        "suspend_days": suspend_days,
        "daily_disk_checkpoints": len(paper["_checkpoints"]),
    }
    passed = all(
        (
            execution["passed"],
            trades["passed"],
            nav["passed"],
            positions["passed"],
            coverage["enable_risk_budget"],
            coverage["sell_timing_close"],
            coverage["nonempty_suspend_calendar"],
            coverage["daily_disk_checkpoints"] == len(window),
        )
    )
    summary = {
        "gate": "gate2",
        "verdict": "PASS" if passed else "FAIL",
        "window": {"start": start, "end": end, "trading_days": len(window)},
        "execution": execution,
        "trades": trades,
        "nav": nav,
        "positions": positions,
        "coverage": coverage,
        "input_receipts": input_receipts,
        "paper_daily": paper["_paper_replay_days"],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trade_detail.to_parquet(evidence_dir / "trade_detail.parquet", index=False)
    nav_detail.to_parquet(evidence_dir / "nav_delta.parquet", index=False)
    _dump(evidence_dir / "summary.json", summary)
    return summary


def _gate3(
    *,
    host: PaperHost,
    paper_run_root: Path,
    backtest_run_root: Path,
    runs_root: Path,
    evidence_dir: Path,
    start: str,
    end: str,
    frozen: str,
    config: Dict[str, object],
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    excluded_dates: Sequence[str] = (),
) -> Dict[str, Any]:
    auxiliary = bool(excluded_dates)
    window = host.replay_calendar
    paper = run_paper_replay(
        host, run_root=paper_run_root, start_date=start, end_date=end, frozen_through=frozen
    )
    features, prices, input_receipts = collect_shared_inputs(host, window)
    reference = run_backtest_reference(
        host, features_by_date=features, price_data=prices, window=window
    )
    execution = assert_execution_configs_match(
        paper["_execution_config"], reference["_execution_config"]
    )
    paper_batch_id, backtest_batch_id = paper_run_root.name, backtest_run_root.name
    train_start, train_end = model_training_window(host)
    write_paper_replay_batch(
        paper,
        batch_id=paper_batch_id,
        runs_root=runs_root,
        config=config,
        code_state=code_state,
        data_state=data_state,
        model_version=host.trading_config.model_version,
        train_start=train_start,
        train_end=train_end,
        frozen_through=frozen,
        baseline_ref=backtest_batch_id,
    )
    write_backtest_batch(
        reference,
        host=host,
        batch_id=backtest_batch_id,
        runs_root=runs_root,
        config=config,
        code_state=code_state,
        data_state=data_state,
        baseline_ref=paper_batch_id,
    )
    paper_batch = load_runs_batch(paper_run_root)
    backtest_batch = load_runs_batch(backtest_run_root)
    readback = {
        "passed": (
            paper_batch.batch_meta.get("host_mode") == "paper_replay"
            and backtest_batch.batch_meta.get("host_mode") == "backtest"
            and list(paper_batch.folds) == [0]
            and list(backtest_batch.folds) == [0]
        ),
        "paper_host_mode": paper_batch.batch_meta.get("host_mode"),
        "backtest_host_mode": backtest_batch.batch_meta.get("host_mode"),
        "paper_folds": list(paper_batch.folds),
        "backtest_folds": list(backtest_batch.folds),
        "paper_data_state_id": paper_batch.batch_meta["data_state"]["data_state_id"],
        "backtest_data_state_id": backtest_batch.batch_meta["data_state"]["data_state_id"],
    }
    readback["data_state_equal"] = (
        readback["paper_data_state_id"] == readback["backtest_data_state_id"]
    )
    readback["passed"] = bool(readback["passed"] and readback["data_state_equal"])
    trades, trade_detail = compare_trades(
        canonical_trades(paper_batch.folds[0].trades),
        canonical_trades(backtest_batch.folds[0].trades),
        selected_count=host.trading_config.top_n,
        allow_tie_swaps=True,
    )
    nav, nav_detail = compare_nav(
        canonical_nav(paper_batch.folds[0].daily), canonical_nav(backtest_batch.folds[0].daily)
    )
    positions = compare_positions(
        canonical_positions(paper_batch.folds[0].holdings_snapshot),
        canonical_positions(backtest_batch.folds[0].holdings_snapshot),
    )
    domains = paper["_row_domain_assertions"]
    domain_pass = len(domains) == len(window) and all(row["passed"] for row in domains)
    expected_days = 76 - len(excluded_dates) if auxiliary else 76
    passed = all(
        (
            len(window) == expected_days,
            execution["passed"],
            readback["passed"],
            trades["passed"],
            nav["passed"],
            positions["passed"],
            domain_pass,
            not trades["resolution_insufficient"],
        )
    )
    summary = {
        "gate": "gate3_auxiliary" if auxiliary else "gate3",
        "verdict": ("AUX_PASS" if auxiliary else "PASS") if passed else "FAIL",
        "formal_gate3": not auxiliary,
        "window": {
            "start": start,
            "end": end,
            "trading_days": len(window),
            "excluded_dates": list(excluded_dates),
        },
        "batch_ids": {"paper_replay": paper_batch_id, "backtest": backtest_batch_id},
        "execution": execution,
        "runs_readback": readback,
        "trades": trades,
        "nav": nav,
        "positions": positions,
        "row_domain": {"passed": domain_pass, "days": domains},
        "input_receipts": input_receipts,
        "paper_daily": paper["_paper_replay_days"],
        "initial_account": paper["_initial_account"],
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trade_detail.to_parquet(evidence_dir / "trade_detail.parquet", index=False)
    nav_detail.to_parquet(evidence_dir / "nav_delta.parquet", index=False)
    _dump(evidence_dir / "row_domain_assertions.json", domains)
    _dump(evidence_dir / "paper_batch_meta.json", paper_batch.batch_meta)
    _dump(evidence_dir / "backtest_batch_meta.json", backtest_batch.batch_meta)
    _dump(evidence_dir / "summary.json", summary)
    return summary


def _config_for_runs(host: PaperHost, enable_risk_budget: bool = False) -> Dict[str, object]:
    config = host.trading_config.to_dict()
    config.update(
        {
            "bt_top_n": host.trading_config.top_n,
            "label_column": f"neu_y_ret_{host.trading_config.horizon}",
            "stock_domain": "main",
            "enable_risk_budget": bool(enable_risk_budget),
            "enable_trailing_early_rebalance": True,
            "completion_window_days": host.engine_kwargs["completion_window_days"],
        }
    )
    return config


def _parse_excluded_dates(raw: str, auxiliary_exclusion: bool) -> Tuple[str, ...]:
    """解析辅助排除日；未显式选择辅助模式、格式错误或重复均 fail-fast。"""
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if values and not auxiliary_exclusion:
        raise ValueError("排除日期必须同时显式指定 --auxiliary-exclusion")
    if any(len(value) != 8 or not value.isdigit() for value in values):
        raise ValueError("排除日期必须为 YYYYMMDD")
    if len(set(values)) != len(values):
        raise ValueError("排除日期不得重复")
    return tuple(sorted(values))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v2 P4 T4 门 2/门 3 正式执行器")
    parser.add_argument("--gate", choices=("2", "3", "all"), default="all")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--start", default="20260703")
    parser.add_argument("--end", default="20261026")
    parser.add_argument("--frozen-through", default="20261026")
    parser.add_argument("--gate2-start", default="20260703")
    parser.add_argument("--gate2-end", default="20260731")
    parser.add_argument("--exclude-dates", default="", help="辅助比对的双侧共同排除日，逗号分隔")
    parser.add_argument(
        "--auxiliary-exclusion",
        action="store_true",
        help="显式声明本次仅为排除日后的辅助比对，不产生正式门 3 PASS",
    )
    args = parser.parse_args(argv)
    excluded_dates = _parse_excluded_dates(args.exclude_dates, args.auxiliary_exclusion)
    if excluded_dates and (args.gate != "3" or (args.start, args.end) != ("20260703", "20261026")):
        raise ValueError("辅助排除仅允许门 3 的 [20260703, 20261026] 冻结窗口")
    started = time.perf_counter()
    evidence_root = _REPO_ROOT / "data" / "frozen_reference" / "v2_p4" / f"t4_{args.run_id}"
    runs_root = _REPO_ROOT / "data" / "runs"
    protected_root = Path(get_paper_root())
    before_paper = snapshot_protected_root(protected_root)
    data_root = Path(get_data_root())
    before_data_state = capture_data_state(data_root)
    recorder = CallRecorder()
    results: Dict[str, Any] = {}
    log_path = Path(get_logs_dir()) / f"v2_p4_t4_{args.run_id}.log"
    logger.add(str(log_path), encoding="utf-8", level="INFO")
    try:
        with _no_network(recorder):
            storage = Storage(root_path=str(data_root))
            wrap_storage_no_write(storage, recorder)
            loader = DataLoader(storage=storage, verbose=False)
            calendar = _load_full_calendar(loader)
            flat = load_paper_config(str(protected_root))
            code_state = _code_state()
            if args.gate in {"2", "all"}:
                gate2_root = evidence_root / "gate2" / "work"
                gate2_flat = dict(flat)
                gate2_flat["sell_price"] = "close"
                host2 = _build_host(
                    config_flat=gate2_flat,
                    run_root=gate2_root,
                    start=args.gate2_start,
                    end=args.gate2_end,
                    storage=storage,
                    loader=loader,
                    calendar=calendar,
                    enable_risk_budget=True,
                )
                results["gate2"] = _gate2(
                    host=host2,
                    run_root=gate2_root,
                    evidence_dir=evidence_root / "gate2",
                    start=args.gate2_start,
                    end=args.gate2_end,
                    frozen=args.frozen_through,
                )
            if args.gate in {"3", "all"}:
                paper_id = f"p4_t4_paper_replay_{args.run_id}"
                backtest_id = f"p4_t4_backtest_{args.run_id}"
                paper_root, backtest_root = runs_root / paper_id, runs_root / backtest_id
                host3 = _build_host(
                    config_flat=flat,
                    run_root=paper_root,
                    start=args.start,
                    end=args.end,
                    storage=storage,
                    loader=loader,
                    calendar=calendar,
                    excluded_dates=excluded_dates,
                )
                run_config = _config_for_runs(host3)
                run_config["auxiliary_exclusion"] = bool(excluded_dates)
                run_config["aux_excluded_dates"] = list(excluded_dates)
                results["gate3"] = _gate3(
                    host=host3,
                    paper_run_root=paper_root,
                    backtest_run_root=backtest_root,
                    runs_root=runs_root,
                    evidence_dir=evidence_root / "gate3",
                    start=args.start,
                    end=args.end,
                    frozen=args.frozen_through,
                    config=run_config,
                    code_state=code_state,
                    data_state=before_data_state,
                    excluded_dates=excluded_dates,
                )
        after_data_state = capture_data_state(data_root)
        after_paper = snapshot_protected_root(protected_root)
        network_events = [event for event in recorder.events if event["kind"] == "network"]
        write_events = [event for event in recorder.events if event["kind"] == "write"]
        containment = {
            "paper_unchanged": before_paper == after_paper,
            "data_state_unchanged": before_data_state == after_data_state,
            "network_calls": len(network_events),
            "storage_write_calls": len(write_events),
            "events": recorder.events,
        }
        containment["passed"] = all(
            (
                containment["paper_unchanged"],
                containment["data_state_unchanged"],
                containment["network_calls"] == 0,
                containment["storage_write_calls"] == 0,
            )
        )
        gates_pass = all(result["verdict"] in {"PASS", "AUX_PASS"} for result in results.values())
        verdict = (
            ("AUX_PASS" if excluded_dates else "PASS")
            if gates_pass and containment["passed"]
            else "FAIL"
        )
        summary = {
            "run_id": f"t4_{args.run_id}",
            "verdict": verdict,
            "elapsed_seconds": time.perf_counter() - started,
            "evidence_root": str(evidence_root),
            "results": results,
            "containment": containment,
            "data_state": after_data_state,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "log_path": str(log_path),
            "t5_started": False,
            "production_switch_changed": False,
        }
        _dump(evidence_root / "verification_summary.json", summary)
        print(
            json.dumps(
                {"verdict": verdict, "gates": {k: v["verdict"] for k, v in results.items()}},
                ensure_ascii=False,
            )
        )
        return 0 if verdict in {"PASS", "AUX_PASS"} else 1
    except Exception as exc:
        failure = {
            "run_id": f"t4_{args.run_id}",
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "results": results,
            "events": recorder.events,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _dump(evidence_root / "failure.json", failure)
        print(f"T4 执行失败: {failure['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

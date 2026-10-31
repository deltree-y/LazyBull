# -*- coding: utf-8 -*-
"""P4-T3 E2 历史 normalize 分歧只读统计（阶段专用验收工具，不入生产链路）。

复用 B0 split11~13 的真实回测，在原生 normalize 接缝捕获实际输入；旧纸面与
v2 实现仅在同进程内对照计算，不改权重、不写生产数据。输出逐批观测、聚合频率、
实际消费文件 SHA-256 与收尾身份复核，保证正式结论可复算。

# 复杂度豁免：P4-T3 阶段专用只读验收工具；run 统筹真实捕获与证据装配，
# 不进生产链路；若后续复用为生产入口，先拆分捕获、统计与身份核验三段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v2_p2a.replay_b0 import (  # noqa: E402
    B0_KELLY_MAX_LEVERAGE,
    B0_POSITION_SIZING,
    B0_STOCK_DOMAIN,
    _generate_b0_splits,
    _run_b0_split,
)
from src.lazybull.common.config import get_stock_selection_models_root  # noqa: E402
from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.paper.runner.pricing import PaperPricingMixin  # noqa: E402
from src.lazybull.paper.runner.signals import PaperSignalMixin  # noqa: E402
from src.lazybull.universe.domains import resolve_stock_domain  # noqa: E402
from src.lazybull.v2.core.decision import weighting  # noqa: E402
from src.lazybull.v2.common.suspend_calendar import SuspendCalendar  # noqa: E402
from src.lazybull.v2.core.signal.ml_signal import MLSignal  # noqa: E402
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML  # noqa: E402

MODES = ("equal", "score", "kelly", "half_kelly")


class _LegacyOracle(PaperSignalMixin, PaperPricingMixin):
    """保活旧纸面只读 oracle；行情由真实回测接缝注入。"""

    def __init__(self, sizing: str, price_data: pd.DataFrame, horizon: int = 20) -> None:
        self.position_sizing = sizing
        self.price_data = price_data
        self.horizon = horizon
        self.kelly_max_leverage = B0_KELLY_MAX_LEVERAGE

    def _load_kelly_window_data(self, trade_date: str):
        return self.price_data[self.price_data["trade_date"] <= trade_date]


def _old(signals, sizing, prices, date) -> Dict[str, float]:
    return _LegacyOracle(sizing, prices)._normalize_signals(
        dict(signals), pd.Timestamp(date).strftime("%Y%m%d")
    )


def _new(signals, sizing, prices, date, window) -> Dict[str, float]:
    result, _ = weighting.normalize_signals(
        dict(signals),
        pd.Timestamp(date),
        position_sizing=sizing,
        normalize_log_count=5,
        verbose=False,
        kelly_max_leverage=B0_KELLY_MAX_LEVERAGE,
        price_data_cache=prices,
        kelly_vol_window=window,
    )
    return result


def _new_stat() -> Dict[str, Any]:
    return {
        "batches": 0,
        "signal_rows": 0,
        "divergent_batches": 0,
        "exact_equal_batches": 0,
        "key_order_divergent_batches": 0,
        "key_set_divergent_batches": 0,
        "max_abs_weight_diff": 0.0,
        "branches": Counter(),
    }


def _value_equal(left: float, right: float) -> bool:
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(left == right)


def _update(stat, signals, old, new) -> Dict[str, Any]:
    keys = list(dict.fromkeys([*old, *new]))
    exact = list(old) == list(new) and all(_value_equal(old[key], new[key]) for key in keys)
    differences = [abs(float(old.get(key, 0.0)) - float(new.get(key, 0.0))) for key in keys]
    stat["batches"] += 1
    stat["signal_rows"] += len(signals)
    stat["exact_equal_batches"] += int(exact)
    stat["divergent_batches"] += int(not exact)
    stat["key_order_divergent_batches"] += int(list(old) != list(new))
    stat["key_set_divergent_batches"] += len(set(old) ^ set(new))
    stat["max_abs_weight_diff"] = max(stat["max_abs_weight_diff"], max(differences, default=0.0))
    if any(value > 0 for value in signals.values()):
        stat["branches"]["has_positive"] += 1
    if any(value < 0 for value in signals.values()):
        stat["branches"]["negative_scores"] += 1
    if sum(signals.values()) <= 0:
        stat["branches"]["nonpositive_total"] += 1
    return {
        "old_weights": {key: float(value) for key, value in old.items()},
        "new_weights": {key: float(value) for key, value in new.items()},
        "exact": exact,
        "max_abs_weight_diff": max(differences, default=0.0),
    }


def _json_stat(stat) -> Dict[str, Any]:
    batches = stat["batches"]
    return {
        **{
            key: value
            for key, value in stat.items()
            if key not in {"branches", "max_abs_weight_diff"}
        },
        "divergent_frequency": stat["divergent_batches"] / batches if batches else 0.0,
        "max_abs_weight_diff": stat["max_abs_weight_diff"],
        "branches": dict(sorted(stat["branches"].items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identities(paths: Iterable[Path], data_root: Path) -> list[Dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        if not path.is_file():
            raise FileNotFoundError(f"统计消费文件不存在: {path}")
        records.append(
            {
                "path": path.relative_to(data_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _model_files(models_root: Path, version: int) -> list[Path]:
    metadata_path = models_root / f"v{version}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = [metadata_path, models_root / metadata["features_file"]]
    model_path = models_root / metadata["model_file"]
    if model_path.is_file():
        paths.append(model_path)
    else:
        paths.append(models_root / f"v{version}_model.json")
    return paths


def _load_inputs(data_root: Path):
    storage = Storage(root_path=str(data_root))
    loader = DataLoader(storage)
    stock_basic = loader.load_clean_stock_basic()
    if stock_basic is None:
        stock_basic = loader.load_stock_basic()
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        trade_cal = loader.load_trade_cal()
    if stock_basic is None or trade_cal is None or trade_cal.empty:
        raise RuntimeError("stock_basic/trade_cal 加载失败")
    return storage, loader, stock_basic, trade_cal


def run(
    data_root: Path, out: Path, split_indices: tuple[int, ...], expected_batches: Optional[int]
):
    storage, loader, stock_basic, trade_cal = _load_inputs(data_root)
    splits = [s for s in _generate_b0_splits(trade_cal) if s.split_index in split_indices]
    if [s.split_index for s in splits] != list(split_indices):
        raise ValueError(
            f"切分不完整: 请求={split_indices}, 实际={[s.split_index for s in splits]}"
        )
    models_root = Path(get_stock_selection_models_root(str(data_root)))
    domain = resolve_stock_domain(B0_STOCK_DOMAIN)
    persistent_signal = MLSignal(
        top_n=20,
        model_version=None,
        models_dir=str(models_root),
        min_total_mv=float(domain["min_total_mv_wan"]),
        max_total_mv=float(domain["max_total_mv_wan"]),
        verbose=False,
        downside_penalty=0.0,
        downside_penalty_column="downside_vol_20",
    )
    actual = {mode: _new_stat() for mode in (*MODES, "unknown_sizing")}
    hypothetical = {
        window: {mode: _new_stat() for mode in ("kelly", "half_kelly")} for window in (20, 40)
    }
    observations = []
    consumed = set()
    current_split = None
    bt_ends = []
    runtime_before = {}
    missing_runtime = set()
    original = NativeMergeBacktestEngineML._normalize_and_cap_signals

    def capturing(engine, signals, date, tranche_idx):
        split = current_split
        if split is None:
            raise RuntimeError("normalize 捕获器缺少当前 split 上下文")
        prices = engine.price_data_cache
        row = {
            "split_index": split.split_index,
            "model_version": 24008 + split.split_index,
            "date": pd.Timestamp(date).strftime("%Y%m%d"),
            "tranche_idx": tranche_idx,
            "signals": {key: float(value) for key, value in signals.items()},
            "actual": {},
            "hypothetical": {},
        }
        for mode in MODES:
            old = _old(signals, mode, prices, date)
            new = _new(signals, mode, prices, date, 60)
            row["actual"][mode] = _update(actual[mode], signals, old, new)
        old = _old(signals, "unknown_sizing", prices, date)
        new = _new(signals, "unknown_sizing", prices, date, 60)
        row["actual"]["unknown_sizing"] = _update(actual["unknown_sizing"], signals, old, new)
        for window in (20, 40):
            row["hypothetical"][str(window)] = {}
            for mode in ("kelly", "half_kelly"):
                old = _old(signals, mode, prices, date)
                new = _new(signals, mode, prices, date, window)
                row["hypothetical"][str(window)][mode] = _update(
                    hypothetical[window][mode], signals, old, new
                )
        observations.append(row)
        return original(engine, signals, date, tranche_idx)

    for split in splits:
        consumed.update(_model_files(models_root, 24008 + split.split_index))
        bt_end = (datetime.strptime(split.test_start, "%Y%m%d") + relativedelta(months=6)).strftime(
            "%Y%m%d"
        )
        bt_ends.append(bt_end)
        dates = trade_cal[
            (trade_cal["cal_date"] >= split.test_start)
            & (trade_cal["cal_date"] <= bt_end)
            & (trade_cal["is_open"] == 1)
        ]["cal_date"]
        for date in dates:
            consumed.add(data_root / "features" / "cs_train" / f"{date}.parquet")
            daily = data_root / "clean" / "daily" / f"{date[:4]}-{date[4:6]}-{date[6:]}.parquet"
            if daily.is_file():
                consumed.add(daily)
    for stem in ("stock_basic", "trade_cal"):
        for relative in (f"clean/{stem}.parquet", f"raw/{stem}.parquet"):
            candidate = data_root / relative
            if candidate.is_file():
                consumed.add(candidate)
                break
    before = _identities(consumed, data_root)

    original_suspend_load = SuspendCalendar._load_suspend_data

    def capturing_suspend_read(calendar, trade_date):
        """在真实停牌 I/O 点记录身份；缺失须前后一致，存在则读取前后同字节。"""
        date = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        path = data_root / "raw" / "suspend" / f"{date}.parquet"
        relative = path.relative_to(data_root).as_posix()
        before_read = _identities([path], data_root)[0] if path.is_file() else None
        result = original_suspend_load(calendar, trade_date)
        after_read = _identities([path], data_root)[0] if path.is_file() else None
        if before_read != after_read:
            raise AssertionError(f"停牌输入在单次读取期间发生漂移: {relative}")
        if result is not None and after_read is None:
            raise FileNotFoundError(f"停牌读取返回数据但分区文件缺失: {relative}")
        if after_read is not None:
            previous = runtime_before.get(path)
            if previous is not None and previous != after_read:
                raise AssertionError(f"同一停牌输入跨次读取发生漂移: {relative}")
            runtime_before[path] = after_read
            consumed.add(path)
        else:
            missing_runtime.add(relative)
        return result

    SuspendCalendar._load_suspend_data = capturing_suspend_read
    NativeMergeBacktestEngineML._normalize_and_cap_signals = capturing
    try:
        for split in splits:
            current_split = split
            _run_b0_split(
                split,
                storage=storage,
                loader=loader,
                trade_cal=trade_cal,
                stock_basic=stock_basic,
                data_root=str(data_root),
                persistent_signal=persistent_signal,
                models_root=str(models_root),
                merger_mode=None,
            )
    finally:
        NativeMergeBacktestEngineML._normalize_and_cap_signals = original
        SuspendCalendar._load_suspend_data = original_suspend_load

    if expected_batches is not None and len(observations) != expected_batches:
        raise AssertionError(f"实际捕获 {len(observations)} 批，预期 {expected_batches}")
    before_by_path = {record["path"]: record for record in [*before, *runtime_before.values()]}
    before_full = [before_by_path[key] for key in sorted(before_by_path)]
    after = _identities(consumed, data_root)
    if before_full != after:
        raise AssertionError("统计输入文件在收尾身份复核中发生漂移")
    payload = {
        "run_id": out.parent.name,
        "read_only_inputs": True,
        "sample": {
            "split_indices": list(split_indices),
            "model_versions": [24008 + index for index in split_indices],
            "start_date": splits[0].test_start,
            "end_date": bt_ends[-1],
            "batches": len(observations),
            "signal_rows": sum(len(row["signals"]) for row in observations),
        },
        "actual_configuration": {
            "backtest_position_sizing": B0_POSITION_SIZING,
            "legacy_paper_position_sizing": "kelly",
            "kelly_vol_window": 60,
            "note": "actual=当前 60 窗口；hypothetical=仅新宿主窗口敏感性，不伪称历史启用频率",
        },
        "actual_stats": {mode: _json_stat(stat) for mode, stat in actual.items()},
        "hypothetical_kelly_window_stats": {
            str(window): {mode: _json_stat(stat) for mode, stat in modes.items()}
            for window, modes in hypothetical.items()
        },
        "observations": observations,
        "input_identity": {
            "files": before_full,
            "runtime_raw_reads": [
                runtime_before[key] for key in sorted(runtime_before, key=lambda p: p.as_posix())
            ],
            "missing_raw_reads": sorted(missing_runtime),
            "after_drift": [],
        },
        "attribution": {
            "bucket": "normalize_semantics",
            "divergent_branches": [
                "score 正分数子集 vs 全分数求和",
                "有部分正分但全分数总和 <= 0",
                "正分数总和 < 1e-12",
                "未知 sizing：旧纸面告警后 score vs 新宿主静默等权",
                "窗口表达式不同：旧纸面 tail(max(20,min(3*horizon,120),60)) vs 新宿主 kelly_vol_window",
            ],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "sample": payload["sample"], "actual_stats": payload["actual_stats"]},
            ensure_ascii=False,
        )
    )
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="11,12,13")
    parser.add_argument("--expected-batches", type=int, default=38)
    args = parser.parse_args(argv)
    indices = tuple(int(token) for token in args.splits.split(",") if token.strip())
    run(Path(args.data_root), Path(args.out), indices, args.expected_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

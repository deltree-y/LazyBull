# -*- coding: utf-8 -*-
"""v2 P4 数据层单一协调入口（D4）：raw → clean → 旧侧推理 → panel/labels → 就绪/证据。

本文件是 **薄编排**（阶段专用临时开发文件，复杂度豁免沿 CLAUDE.md §7.5）：
不实现任何清洗、复权、因子、标签或 manifest 语义——全部委托既有生产实现：

- raw：`scripts/download_raw.py`（TuShare 翻页读到空 + 全字段去重契约不变）；
- clean：`src/lazybull/data/build_clean.py::build_clean_data`；
- 旧侧推理：`src/lazybull/features/ensure/entry.py::ensure_features_for_date`；
- panel/labels：`scripts/v2_p3_pre/update_daily_store.py::update_daily_store`；
- 就绪门：`scripts/v2_p4/t2_daily_dryrun.py`（隔离根 + containment，只读生产根）。

职责边界
========
1. **单一写入链**：禁止任何旁路 raw/clean 写入；本入口只按序调用上述实现。
2. **幂等续跑**：每阶段/每日落 `checkpoint.json`；已存在的 raw 分区、clean 分区、
   cs_infer 分区与 panel 日族由既有实现在内部跳过，本入口不重复下载/写入。
3. **fail-closed**：任一日缺分区、任一次 ensure 返回失败、任一子进程非零退出即
   阶段失败并停在原地，**禁止跳过缺口伪造追平完成**。
4. **证据**：`data/frozen_reference/v2_p4/<run_id>/` 落阶段报告与逐源校验；
   日志落 `logs/`；中间产物不落 temp 以外的提交区。

用法
====
``python scripts/v2_p4/daily_data_coordinator.py --stage snapshot|raw|clean|cs_infer|panel|verify \
    --target-date 20261026 --run-id mp41_20261029_120000 [--execute] [--max-days N]``

缺省 `--execute` 为空（只打印计划，不写盘）。`--stages a,b` 可一次串多阶段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

STAGE_ORDER: tuple[str, ...] = ("snapshot", "raw", "clean", "cs_infer", "panel", "verify")

#: 日分区类 raw 子集（缺失即当日缺口，逐日补齐）
DAILY_PARTITION_DATASETS: tuple[str, ...] = (
    "daily",
    "daily_basic",
    "adj_factor",
    "suspend",
    "stk_limit",
    "moneyflow",
    "stock_st",
    "margin_detail",
    "cyq_perf",
    "moneyflow_hsgt",
    "top_list",
    "block_trade",
)

#: 公告/单文件类（内容级水位，非日分区口径）
ANNOUNCEMENT_DATASETS: tuple[str, ...] = (
    "stk_holdertrade",
    "repurchase",
    "top10_floatholders",
    "report_rc",
    "stk_holdernumber",
)

#: 季度分区类（须刷新尾部季度并新建缺失季度分区）
PERIODIC_DATASETS: tuple[str, ...] = (
    "cashflow",
    "income",
    "fina_indicator",
    "forecast",
    "express",
    "fund_portfolio",
)

#: 登记不补的家族（裁决 2026-10-29：任一侧消费方均不需要）
REGISTERED_GAPS: tuple[str, ...] = ("dividend", "top_inst", "pledge_stat")

#: 源侧确实无该日数据的缺口（源探针实证，非下载失败；沿既有历史同类缺口处置）
#: moneyflow_hsgt 20261019：`get_moneyflow_hsgt(20261012~20261023)` 仅返回 9 行、
#: 独缺该日；同口径历史缺口已有 10 例（20250418/20250421/20250701/20251029/
#: 20251225/20251226/20260403/20260407/20260525/20260701），现存 panel 已按此容忍。
SOURCE_SIDE_GAPS: dict[str, tuple[str, ...]] = {"moneyflow_hsgt": ("20261019",)}

#: 冻结禁触（仅用于身份比对，不写入）
FROZEN_CS_TRAIN_THROUGH = "20260702"
FROZEN_PANEL_THROUGH = "20260731"


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def _repo() -> Path:
    return ROOT


def _data_root() -> Path:
    return _repo() / "data"


def _evidence_dir(run_id: str) -> Path:
    return _data_root() / "frozen_reference" / "v2_p4" / run_id


def _checkpoint_path(run_id: str) -> Path:
    return _evidence_dir(run_id) / "checkpoint.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"读取失败 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"不是 JSON 对象: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON；tmp 名带 pid（多 worker 并发不互踩），replace 冲突退避重试。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    last: Exception | None = None
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # Windows 共享冲突（另一 writer 短暂持有句柄）
            last = exc
            time.sleep(0.2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"JSON 原子写失败 {path}: {last}")


def _checkpoint(run_id: str) -> dict[str, Any]:
    path = _checkpoint_path(run_id)
    if path.is_file():
        return _read_json(path)
    return {"run_id": run_id, "stages": {}, "created_at": _now()}


def _save_stage(run_id: str, stage: str, status: str, **details: Any) -> None:
    """记录阶段/日进度；并发 worker 下写失败只告警（证据正文在阶段报告里）。"""
    try:
        payload = _checkpoint(run_id)
        record = dict(payload["stages"].get(stage, {}))
        record.update({"status": status, "updated_at": _now(), **details})
        if status == "running" and "started_at" not in record:
            record["started_at"] = _now()
        if status in {"done", "failed"}:
            record["finished_at"] = _now()
        payload["stages"][stage] = record
        _write_json(_checkpoint_path(run_id), payload)
    except (OSError, RuntimeError) as exc:
        logger.warning(f"checkpoint 写入失败（不影响数据写入）: {exc}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _setup_logger(run_id: str, stage: str) -> Path:
    log_dir = _repo() / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{run_id}_{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.remove()
    logger.add(str(path), level="INFO", encoding="utf-8")
    logger.add(sys.stdout, level="INFO")
    return path


def _run(cmd: Sequence[str], log_path: Path) -> int:
    """子进程执行并把输出追加到阶段日志；返回退出码（不抛异常）。"""
    started = time.monotonic()
    logger.info(f"[subprocess] {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as sink:
        sink.write(f"\n===== {_now()} {' '.join(cmd)}\n")
        sink.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(_repo()),
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=False,
        )
    logger.info(f"[subprocess] exit={proc.returncode} 用时 {time.monotonic() - started:.1f}s")
    return int(proc.returncode)


# ---------------------------------------------------------------------------
# 日历与分区工具
# ---------------------------------------------------------------------------
def load_calendar() -> list[str]:
    import pandas as pd

    cal = pd.read_parquet(_data_root() / "clean" / "trade_cal.parquet")
    dates = cal["cal_date"].astype(str).str.replace("-", "", regex=False)
    open_dates = dates[cal["is_open"] == 1]
    return sorted(open_dates.tolist())


def partition_dates(dir_path: Path) -> list[str]:
    out: list[str] = []
    for path in dir_path.glob("*.parquet"):
        stem = path.stem.replace("-", "")
        if len(stem) == 8 and stem.isdigit():
            out.append(stem)
    return sorted(out)


def gap_days(calendar: Sequence[str], start_date: str, target_date: str) -> list[str]:
    if target_date not in calendar:
        raise ValueError(f"目标日 {target_date} 不在交易日历")
    return [day for day in calendar if start_date <= day <= target_date]


# ---------------------------------------------------------------------------
# 水位采集（身份与缺口双用）
# ---------------------------------------------------------------------------
def _watermark_of(dataset: str, root: Path | None = None) -> dict[str, Any]:
    base = (root if root is not None else _data_root()) / "raw"
    root = base / dataset
    if root.is_dir():
        dates = partition_dates(root)
        return {"kind": "partition", "n": len(dates), "max": dates[-1] if dates else None}
    single = base / f"{dataset}.parquet"
    if single.is_file():
        return {
            "kind": "single",
            "bytes": single.stat().st_size,
            "sha256_16": _sha256_16(single),
        }
    return {"kind": "missing"}


def _sha256_16(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _layer_watermark(layer: str, dataset: str) -> dict[str, Any]:
    root = _data_root() / layer / dataset
    if not root.is_dir():
        return {"kind": "missing"}
    dates = partition_dates(root)
    return {"kind": "partition", "n": len(dates), "max": dates[-1] if dates else None}


def _panel_days() -> list[str]:
    root = _data_root() / "features" / "panel"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _manifest_summary() -> dict[str, Any]:
    path = _data_root() / "features" / "manifest.json"
    if not path.is_file():
        return {"exists": False}
    payload = _read_json(path)
    return {
        "exists": True,
        "partitions": len(payload.get("partitions", {})),
        "repairs": len(payload.get("repairs", [])),
        "label_refresh_intents": len(payload.get("label_refresh_intents", [])),
        "label_refreshes": len(payload.get("label_refreshes", [])),
        "sealed_through": {
            name: (payload.get("labels", {}).get(name, {}) or {}).get("sealed_through")
            for name in ("y_ret_5", "y_ret_10", "y_ret_20")
        },
        "row_domain_segments": payload.get("row_domain_segments"),
        "columns": len(payload.get("columns", {})),
    }


def collect_identity() -> dict[str, Any]:
    raw = {name: _watermark_of(name) for name in _verify_datasets()}
    clean = {
        name: _layer_watermark("clean", name) for name in ("daily", "daily_basic", "moneyflow")
    }
    features = {
        name: _layer_watermark("features", name) for name in ("cs_train", "cs_infer")
    }
    return {
        "captured_at": _now(),
        "raw": raw,
        "clean": clean,
        "features": features,
        "panel": {"n": len(_panel_days()), "max": (_panel_days() or [None])[-1]},
        "manifest": _manifest_summary(),
        "paper_state": _paper_fingerprint(),
    }


def _verify_datasets() -> list[str]:
    return list(DAILY_PARTITION_DATASETS + ANNOUNCEMENT_DATASETS + PERIODIC_DATASETS)


def _paper_fingerprint() -> dict[str, Any]:
    root = _data_root() / "paper"
    if not root.is_dir():
        return {"exists": False}
    account = root / "state" / "account.json"
    return {
        "exists": True,
        "account_sha256_16": _sha256_16(account) if account.is_file() else None,
    }


# ---------------------------------------------------------------------------
# 阶段：snapshot
# ---------------------------------------------------------------------------
def stage_snapshot(run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    identity = collect_identity()
    calendar = load_calendar()
    gap = gap_days(calendar, args.start_date, args.target_date)
    identity["gap_days"] = len(gap)
    identity["gap_first"] = gap[0] if gap else None
    identity["gap_last"] = gap[-1] if gap else None
    path = _evidence_dir(run_id) / "identity_before.json"
    _write_json(path, identity)
    logger.info(f"写入前身份已冻结: {path}（缺口 {len(gap)} 日）")
    return {"identity_path": str(path), "gap_days": len(gap)}


# ---------------------------------------------------------------------------
# 阶段：raw
# ---------------------------------------------------------------------------
def _raw_commands(gap: Sequence[str], target_date: str) -> list[list[str]]:
    """三条互补命令（各自对齐下载实现的增量口径，不重叠、不重复下载）。

    ① 日分区族 + 水位增量族（按 ann_date/report_date 年水位续传）；
    ② top10_floatholders 必须用更早的 ``--start-date``：其增量按「区间内报告期 ∪
       最近 2 个已存期」解析，起点晚于上一期就无法回拉该期尾部公告；
    ③ 季分区族 ``--force`` 重下 20260630（补尾）+ 新建 20260930。
    """
    first, last = gap[0], gap[-1]
    base = [sys.executable, "scripts/download_raw.py"]
    daily_cmd = [
        *base,
        "--start-date",
        first,
        "--end-date",
        last,
        "--download",
        *DAILY_PARTITION_DATASETS,
        "stk_holdertrade",
        "repurchase",
        "report_rc",
        "stk_holdernumber",
    ]
    top10fh_cmd = [
        *base,
        "--start-date",
        "20260401",
        "--end-date",
        target_date,
        "--download",
        "top10_floatholders",
    ]
    periodic_cmd = [
        *base,
        "--start-date",
        "20260401",
        "--end-date",
        target_date,
        "--download",
        *PERIODIC_DATASETS,
        "--force",
    ]
    return [daily_cmd, top10fh_cmd, periodic_cmd]


def _missing_daily_partitions(gap: Sequence[str], root: Path | None = None) -> dict[str, list[str]]:
    base = root if root is not None else _data_root()
    missing: dict[str, list[str]] = {}
    for dataset in DAILY_PARTITION_DATASETS:
        present = set(partition_dates(base / "raw" / dataset))
        absent = [day for day in gap if day not in present]
        if absent:
            missing[dataset] = absent
    return missing


def _split_source_gaps(
    missing: Mapping[str, Sequence[str]],
    known: Mapping[str, Sequence[str]] = SOURCE_SIDE_GAPS,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """把「源侧确认无数据」与「应下未下」两类缺口分开（后者决定阶段失败）。"""
    explained: dict[str, list[str]] = {}
    unexpected: dict[str, list[str]] = {}
    for dataset, days in missing.items():
        whitelist = set(known.get(dataset, ()))
        ok = [day for day in days if day in whitelist]
        bad = [day for day in days if day not in whitelist]
        if ok:
            explained[dataset] = ok
        if bad:
            unexpected[dataset] = bad
    return explained, unexpected


def _periodic_watermarks() -> dict[str, Any]:
    import pandas as pd

    out: dict[str, Any] = {}
    for dataset in PERIODIC_DATASETS:
        dates = partition_dates(_data_root() / "raw" / dataset)
        entry: dict[str, Any] = {"max_partition": dates[-1] if dates else None}
        if dates:
            frame = pd.read_parquet(
                _data_root() / "raw" / dataset / f"{_dash(dates[-1])}.parquet",
                columns=None,
            )
            for col in ("ann_date", "f_ann_date"):
                if col in frame.columns:
                    series = frame[col].astype(str).str.replace("-", "", regex=False)
                    entry[f"{col}_max"] = str(series.max())
            entry["rows"] = len(frame)
        out[dataset] = entry
    return out


def _dash(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}"


def stage_raw(run_id: str, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    calendar = load_calendar()
    gap = gap_days(calendar, args.start_date, args.target_date)
    before = collect_identity()["raw"]
    if not args.execute:
        logger.info(f"[dry] raw 计划缺口 {len(gap)} 日：{gap[0]}~{gap[-1]}")
        for cmd in _raw_commands(gap, args.target_date):
            logger.info("[dry] " + " ".join(cmd))
        return {"dry_run": True, "gap_days": len(gap)}
    record = _checkpoint(run_id).get("stages", {}).get("raw", {})
    if record.get("downloads_completed"):
        codes = list(record.get("exit_codes") or [])
        logger.info(f"raw 下载子进程已于前次完成（exit={codes}），本次仅复验缺口，不重复下载")
    else:
        codes = [_run(cmd, log_path) for cmd in _raw_commands(gap, args.target_date)]
        _save_stage(run_id, "raw", "running", downloads_completed=True, exit_codes=codes)
    missing = _missing_daily_partitions(gap)
    periodic = _periodic_watermarks()
    explained, unexpected = _split_source_gaps(missing)
    report = {
        "gap_days": len(gap),
        "gap_first": gap[0],
        "gap_last": gap[-1],
        "exit_codes": codes,
        "missing_daily_partitions": missing,
        "source_side_gaps": explained,
        "unexpected_gaps": unexpected,
        "periodic": periodic,
        "announcement": {name: _watermark_of(name) for name in ANNOUNCEMENT_DATASETS},
        "registered_gaps": list(REGISTERED_GAPS),
        "watermark_before": before,
    }
    _write_json(_evidence_dir(run_id) / "raw_after.json", report)
    if any(code != 0 for code in codes):
        raise RuntimeError(f"raw 下载子进程非零退出: {codes}（禁止跳过缺口）")
    if unexpected:
        raise RuntimeError(f"raw 日分区仍缺（非源侧缺口）: {unexpected}")
    if explained:
        logger.warning(f"登记源侧缺口（源探针实证无数据）: {explained}")
    q3 = "20260930"
    absent_q3 = [
        name for name in PERIODIC_DATASETS
        if q3 not in set(partition_dates(_data_root() / "raw" / name))
    ]
    if absent_q3:
        logger.warning(f"季度分区 {q3} 缺失（可能源侧未披露）: {absent_q3}")
    logger.info("raw 缺口闭合：日分区全到、公告与季度分区已刷新")
    return {
        "gap_days": len(gap),
        "unexpected_gaps": unexpected,
        "source_side_gaps": explained,
        "exit_codes": codes,
    }


# ---------------------------------------------------------------------------
# 阶段：clean
# ---------------------------------------------------------------------------
def stage_clean(run_id: str, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    import pandas as pd

    from src.lazybull.data import DataCleaner, DataLoader, Storage
    from src.lazybull.data.build_clean import build_clean_data

    calendar = load_calendar()
    gap = gap_days(calendar, args.start_date, args.target_date)
    if not args.execute:
        logger.info(f"[dry] clean 计划 {len(gap)} 日：{gap[0]}~{gap[-1]}")
        return {"dry_run": True, "gap_days": len(gap)}
    storage = Storage(root_path=str(_data_root()))
    loader = DataLoader(storage)
    cleaner = DataCleaner()
    started = time.monotonic()
    build_clean_data(storage, loader, cleaner, gap[0], gap[-1], force=False)
    elapsed = time.monotonic() - started
    report: dict[str, Any] = {"elapsed_seconds": round(elapsed, 1), "datasets": {}}
    failures: dict[str, list[str]] = {}
    for dataset in ("daily", "daily_basic", "moneyflow"):
        present = set(partition_dates(_data_root() / "clean" / dataset))
        absent = [day for day in gap if day not in present]
        rows: dict[str, int] = {}
        bad: list[str] = []
        for day in gap:
            if day in absent:
                continue
            frame = pd.read_parquet(
                _data_root() / "clean" / dataset / f"{_dash(day)}.parquet",
                columns=["trade_date", "ts_code"],
            )
            rows[day] = len(frame)
            if frame.empty or frame["ts_code"].duplicated().any():
                bad.append(day)
        if dataset == "daily":
            report["min_rows"] = min(rows.values()) if rows else 0
            report["max_rows"] = max(rows.values()) if rows else 0
        report["datasets"][dataset] = {
            "present": len(gap) - len(absent),
            "absent": absent,
            "duplicate_key_days": bad,
        }
        if absent or bad:
            failures[dataset] = absent + bad
    _write_json(_evidence_dir(run_id) / "clean_after.json", report)
    if failures:
        raise RuntimeError(f"clean 缺口未闭合（禁止跳过）: {sorted(failures)}")
    logger.info(f"clean 缺口闭合：{len(gap)} 日 × 3 数据集（{elapsed:.1f}s）")
    return report


# ---------------------------------------------------------------------------
# 阶段：cs_infer（旧侧推理准备）
# ---------------------------------------------------------------------------
#: 终端/系统常注入 HTTP 代理（如 http://192.168.1.21:18081），TuShare 走代理会
#: Read timed out 甚至长期挂起（`scripts/raw_download/cli.py` 同源处置）。本进程内清除，
#: 不修改系统设置与其它进程。
_PROXY_ENV_KEYS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _bypass_proxy() -> list[str]:
    """强制本进程直连 TuShare（返回被清除的代理键）。

    仅清环境变量不足以直连：``requests`` 在 Windows 上还会经
    ``urllib.request.getproxies()`` 读取注册表代理设置，故同时置 ``NO_PROXY=*``；
    并设置 socket 默认超时，避免无超时请求长期挂起（实测 25 分钟零 CPU 阻塞）。
    """
    removed = [key for key in _PROXY_ENV_KEYS if os.environ.pop(key, None) is not None]
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    socket.setdefaulttimeout(60.0)
    if removed:
        logger.info(f"已在本进程内清除代理变量（直连，NO_PROXY=*）: {sorted(removed)}")
    return removed


def _cs_infer_builder_flags(config_path: Path | None = None) -> tuple[int, bool]:
    """旧侧推理（cs_infer）构建口径：与现役纸面 runner 逐字一致。

    依据 `src/lazybull/paper/runner/__init__.py:118`
    ``FeatureBuilder(horizon=horizon, require_label=False)``——实盘 T0 无未来数据，
    默认 ``require_label=True`` + ``label_filter_mode="all"`` 会把当日全部行剔除
    （当 ``T+21`` 端点落在 ``D+1 自然月`` 加载窗之外时，实测 20260827/20260828 空结果）。
    """
    horizon = 20
    path = config_path if config_path is not None else (_data_root() / "paper" / "config.yaml")
    if path.is_file():
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = (payload.get("paper_trade") or {}).get("horizon")
        if isinstance(value, int) and value > 0:
            horizon = value
    return horizon, False


def stage_cs_infer(run_id: str, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    from src.lazybull.common.config import get_config
    from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
    from src.lazybull.features import FeatureBuilder
    from src.lazybull.features.ensure.entry import ensure_features_for_date

    _bypass_proxy()
    get_config()
    calendar = load_calendar()
    dates = gap_days(calendar, args.cs_infer_start, args.target_date)
    horizon, require_label = _cs_infer_builder_flags()
    if args.slice_count > 1:
        if not 0 <= args.slice_index < args.slice_count:
            raise ValueError(f"切片参数非法: {args.slice_index}/{args.slice_count}")
        start, step = args.slice_index, args.slice_count
        dates = dates[start::step]
        logger.info(f"切片 {args.slice_index}/{args.slice_count}: {len(dates)} 日（按日跨步切分，"
                    f"各片写不同日分区，无共享状态）")
    storage = Storage(root_path=str(_data_root()))
    loader = DataLoader(storage)
    builder = FeatureBuilder(horizon=horizon, require_label=require_label)
    cleaner = DataCleaner()
    if not args.execute:
        logger.info(f"[dry] cs_infer 计划 {len(dates)} 日：{dates[0]}~{dates[-1]}")
        return {"dry_run": True, "days": len(dates)}
    client = TushareClient()
    done = skipped = 0
    per_day: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    total = len(dates) if args.max_days <= 0 else min(len(dates), args.max_days)
    for index, day in enumerate(dates, 1):
        if index > total:
            logger.info(f"达到本次 --max-days={args.max_days} 上限，停止于 {dates[index - 2]}")
            break
        if storage.is_feature_exists(day, subdir="cs_infer"):
            skipped += 1
            logger.info(f"[{index}/{total}] cs_infer 已存在，跳过 {day}")
            continue
        started = time.monotonic()
        ok, _missing, detail = ensure_features_for_date(
            storage, loader, builder, cleaner, client, day, force=False
        )
        elapsed = time.monotonic() - started
        if not ok:
            failures.append({"trade_date": day, "error": detail})
            logger.error(f"[{index}/{total}] cs_infer 失败 {day}: {detail}")
            break
        done += 1
        per_day[day] = round(elapsed, 1)
        _save_stage(run_id, "cs_infer", "running", last_done=day, built=done, skipped=skipped)
        eta = (total - index) * (sum(per_day.values()) / max(len(per_day), 1))
        logger.info(f"[{index}/{total}] cs_infer 完成 {day}（{elapsed:.1f}s）预计剩余 {eta / 60:.1f} 分钟")
    report = {
        "days_planned": len(dates),
        "built": done,
        "skipped": skipped,
        "failures": failures,
        "per_day_seconds": per_day,
        "avg_seconds": round(sum(per_day.values()) / len(per_day), 1) if per_day else 0.0,
    }
    _write_json(_evidence_dir(run_id) / "cs_infer_after.json", report)
    if failures:
        raise RuntimeError(f"cs_infer 失败（禁止跳过缺口）: {failures[0]}")
    return report


# ---------------------------------------------------------------------------
# 阶段：panel + labels
# ---------------------------------------------------------------------------
def stage_panel(run_id: str, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    from src.lazybull.v2.store.row_domain import STRUCTURAL_FROM

    backup_root = _data_root() / "backup" / "v2_p3_pre" / run_id
    cmd = [
        sys.executable,
        "scripts/v2_p3_pre/update_daily_store.py",
        "--start-date",
        STRUCTURAL_FROM,
        "--execute",
        "--authorized-minutes",
        str(args.panel_authorized_minutes),
        "--backup-root",
        str(backup_root),
    ]
    before_days = set(_panel_days())
    if not args.execute:
        logger.info("[dry] " + " ".join(cmd).replace(" --execute", ""))
        return {"dry_run": True, "panel_days_before": len(before_days)}
    code = _run(cmd, log_path)
    report_path = _data_root() / "reports" / f"v2_p3_pre_label_refresh_{run_id}.json"
    after_days = set(_panel_days())
    summary: dict[str, Any] = {
        "exit_code": code,
        "backup_root": str(backup_root),
        "report_path": str(report_path),
        "panel_days_before": len(before_days),
        "panel_days_after": len(after_days),
        "panel_days_added": sorted(after_days - before_days),
        "panel_days_removed": sorted(before_days - after_days),
    }
    if report_path.is_file():
        payload = _read_json(report_path)
        summary["tool_status"] = payload.get("status")
        summary["build_daily_calls"] = payload.get("build_daily_calls")
        summary["refresh_calls"] = payload.get("refresh_calls")
        summary["input_watermark"] = payload.get("input_watermark")
        summary["readiness_not_ready"] = [
            item.get("trade_date")
            for item in payload.get("readiness", [])
            if not item.get("ready", False)
        ]
    _write_json(_evidence_dir(run_id) / "panel_after.json", summary)
    if code != 0:
        raise RuntimeError(f"panel 阶段子进程 exit={code}，见 {report_path} 与阶段日志")
    if summary.get("panel_days_removed"):
        raise RuntimeError(f"既有 panel 日被删除: {summary['panel_days_removed']}")
    logger.info(
        f"panel 阶段完成：新增 {len(summary['panel_days_added'])} 日，"
        f"build={summary.get('build_daily_calls')} refresh={summary.get('refresh_calls')}"
    )
    return summary


# ---------------------------------------------------------------------------
# 阶段：verify（逐源校验 + 就绪门）
# ---------------------------------------------------------------------------
def _label_status(dates: Iterable[str]) -> dict[str, Any]:
    import pandas as pd

    out: dict[str, Any] = {}
    for day in dates:
        statuses = set()
        rows = {}
        for name in ("y_ret_5", "y_ret_10", "y_ret_20"):
            path = _data_root() / "labels" / name / f"{day}.parquet"
            if not path.is_file():
                statuses.add("missing")
                continue
            frame = pd.read_parquet(path, columns=["maturity_status"])
            statuses |= set(frame["maturity_status"].astype(str).unique())
            rows[name] = len(frame)
        out[day] = {"statuses": sorted(statuses), "rows": rows}
    return out


def _cs_infer_coverage(model_columns: Sequence[str]) -> dict[str, Any]:
    import pandas as pd

    path = _data_root() / "features" / "cs_infer"
    dates = partition_dates(path)
    out: dict[str, Any] = {"partitions": len(dates), "max": dates[-1] if dates else None}
    if not dates:
        return out
    frame = pd.read_parquet(path / f"{dates[-1]}.parquet", columns=None)
    cols = set(frame.columns)
    freshness = [c for c in model_columns if c.endswith("_freshness_days")]
    out.update(
        {
            "last_date": dates[-1],
            "columns": len(cols),
            "model_columns_missing": sorted(c for c in model_columns if c not in cols),
            "freshness_present": [c for c in freshness if c in cols],
            "freshness_missing": [c for c in freshness if c not in cols],
        }
    )
    return out


def _panel_day_check(days: Sequence[str]) -> dict[str, Any]:
    import pandas as pd

    from src.lazybull.v2.store.column_groups import PANEL_GROUPS

    out: dict[str, Any] = {"checked": 0, "violations": [], "min_rows": None, "max_rows": None}
    rows_seen: list[int] = []
    for day in days:
        for group in PANEL_GROUPS:
            path = _data_root() / "features" / "panel" / day / f"{group}.parquet"
            if not path.is_file():
                out["violations"].append(f"{day}/{group} 缺文件")
                continue
            frame = pd.read_parquet(path, columns=["trade_date", "ts_code"])
            if frame.empty:
                out["violations"].append(f"{day}/{group} 空分区")
                continue
            dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
            if set(dates.unique()) != {day}:
                out["violations"].append(f"{day}/{group} trade_date 与分区名不一致")
            if frame["ts_code"].duplicated().any():
                out["violations"].append(f"{day}/{group} 键重复")
            if group == "core":
                rows_seen.append(len(frame))
        out["checked"] += 1
    if rows_seen:
        out["min_rows"] = min(rows_seen)
        out["max_rows"] = max(rows_seen)
    return out


def stage_verify(run_id: str, args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    import json as _json

    from src.lazybull.v2.store.row_domain import STRUCTURAL_FROM

    identity_before = _read_json(_evidence_dir(run_id) / "identity_before.json")
    calendar = load_calendar()
    gap = gap_days(calendar, args.start_date, args.target_date)
    structural = gap_days(calendar, STRUCTURAL_FROM, args.target_date)
    model_columns = _json.loads(
        (_data_root() / "models" / "stock_selection" / "v24022_features.json").read_text(
            encoding="utf-8"
        )
    )
    identity_after = collect_identity()
    labels = _label_status(structural)
    sealed = [day for day, item in labels.items() if item["statuses"] == ["sealed"]]
    forming = [day for day, item in labels.items() if item["statuses"] == ["forming"]]
    bad_state = [
        day for day, item in labels.items()
        if set(item["statuses"]) - {"sealed", "forming"} or len(item["statuses"]) != 1
    ]
    panel_check = _panel_day_check(gap)
    cs_train_before = identity_before["features"]["cs_train"]
    cs_train_after = identity_after["features"]["cs_train"]
    readiness_code = _run(
        [sys.executable, "scripts/v2_p4/t2_daily_dryrun.py", "--date", args.target_date],
        log_path,
    )
    summary = {
        "run_id": run_id,
        "target_date": args.target_date,
        "checked_at": _now(),
        "gap_days": len(gap),
        "panel": panel_check,
        "labels": {
            "sealed": len(sealed),
            "forming": len(forming),
            "last_sealed": sealed[-1] if sealed else None,
            "first_forming": forming[0] if forming else None,
            "unexpected_state_days": bad_state,
        },
        "cs_train_unchanged": cs_train_before == cs_train_after,
        "cs_train": cs_train_after,
        "cs_infer": _cs_infer_coverage(model_columns),
        "manifest": identity_after["manifest"],
        "identity_before": identity_before,
        "identity_after": identity_after,
        "registered_gaps": list(REGISTERED_GAPS),
        "readiness_exit_code": readiness_code,
    }
    data_ok = (
        not panel_check["violations"]
        and not bad_state
        and summary["cs_train_unchanged"]
        and not summary["cs_infer"]["model_columns_missing"]
        and not summary["cs_infer"]["freshness_missing"]
        and summary["labels"]["sealed"] >= 50
    )
    summary["data_verdict"] = "PASS" if data_ok else "FAIL"
    summary["readiness_verdict"] = "PASS" if readiness_code == 0 else "FAIL"
    verdict = data_ok and readiness_code == 0
    summary["verdict"] = "PASS" if verdict else "FAIL"
    _write_json(_evidence_dir(run_id) / "verification_summary.json", summary)
    logger.info(f"校验结论: {summary['verdict']}（就绪门 exit={readiness_code}）")
    if not verdict:
        raise RuntimeError("逐源校验未通过，见 verification_summary.json")
    return {"verdict": summary["verdict"], "readiness_exit_code": readiness_code}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
STAGE_FUNCS = {
    "snapshot": stage_snapshot,
    "raw": stage_raw,
    "clean": stage_clean,
    "cs_infer": stage_cs_infer,
    "panel": stage_panel,
    "verify": stage_verify,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", required=True, help="逗号分隔阶段，或 all")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-date", default="20261026")
    parser.add_argument("--start-date", default="20260803")
    parser.add_argument("--cs-infer-start", default="20260703")
    parser.add_argument("--execute", action="store_true", help="显式写盘；缺省只打印计划")
    parser.add_argument("--max-days", type=int, default=0, help="cs_infer 单次最多构建日数")
    parser.add_argument("--slice-index", type=int, default=0, help="cs_infer 并行切片序号（0 起）")
    parser.add_argument("--slice-count", type=int, default=1, help="cs_infer 并行切片总数")
    parser.add_argument("--panel-authorized-minutes", type=float, default=480.0)
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    args = parser.parse_args(argv)
    if args.stages == "all":
        args.stages = ",".join(STAGE_ORDER)
    unknown = [s for s in args.stages.split(",") if s not in STAGE_FUNCS]
    if unknown:
        parser.error(f"未知阶段: {unknown}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stages = [s for s in args.stages.split(",") if s]
    for stage in stages:
        log_path = _setup_logger(args.run_id, stage)
        dry_run = not args.execute
        _save_stage(args.run_id, stage, "running", dry_run=dry_run)
        try:
            details = (
                stage_snapshot(args.run_id, args)
                if stage == "snapshot"
                else STAGE_FUNCS[stage](args.run_id, args, log_path)
            )
        except Exception as exc:  # noqa: BLE001 - 阶段失败必须落档后原样抛出
            _save_stage(args.run_id, stage, "failed", dry_run=dry_run, error=str(exc))
            logger.error(f"阶段 {stage} 失败: {exc}")
            raise
        _save_stage(args.run_id, stage, "done", dry_run=dry_run, **details)
        logger.info(f"阶段 {stage} 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

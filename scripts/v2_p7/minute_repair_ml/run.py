"""LR5M-ML-10-v1 薄 CLI 阶段编排（r0/r1/r2/r3）。

命令（待实现接口，本文件即实现）：
    python -m scripts.v2_p7.minute_repair_ml.run --stage r0 --campaign-id <id> --run-id <id>
    python -m scripts.v2_p7.minute_repair_ml.run --stage r1 --upstream <已发布 r0 的 run_id>
    ...
同一 run-id 默认拒绝覆盖；下游只接受已发布且 SHA 通过的上游。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.minute_repair_ml import (
    analysis,
    artifacts,
    dataset,
    domain,
    download,
    native_replay,
    panel,
    prediction,
    quality,
    report,
    source,
    spec,
    splits,
    training,
)

STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "r0": ("inputs",),
    "r1": ("dataset",),
    "r2": ("dataset", "models", "predictions"),
    "r3": ("accounts", "diagnostics", "reports"),
}

PACKAGE_DIR = Path(__file__).resolve().parent


def _emit(stage: str) -> Callable[[str], None]:
    def log(message: str) -> None:
        print(f"[{stage}] {message}", flush=True)

    return log


def _git(*args: str) -> str:
    """只读 git 调用（失败返回空串，不阻断研究）。

    必须显式 `encoding="utf-8", errors="replace"`：Windows 默认按 GBK 解码，
    二进制补丁会抛 `UnicodeDecodeError` 使 `stdout` 变成 `None`。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return (result.stdout or "").strip()
    except OSError:
        return ""


def code_identity() -> dict[str, Any]:
    """代码身份：HEAD + 工作区补丁 SHA + 实际消费源码逐文件 SHA。"""
    head = _git("rev-parse", "HEAD")
    diff = _git("diff", "--binary")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    tracked_files = sorted(PACKAGE_DIR.glob("*.py"))
    files = {path.name: sha256_file(path) for path in tracked_files}
    protocol = REPO_ROOT / spec.PROTOCOL_RELATIVE
    return {
        "code_commit": head or "unknown",
        "dirty": bool(diff or untracked),
        "dirty_patch_sha": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "clean",
        "untracked_count": len([line for line in untracked.splitlines() if line.strip()]),
        "package_files": files,
        "protocol_sha256": sha256_file(protocol) if protocol.exists() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def _env_identity() -> dict[str, Any]:
    """依赖版本（XGBoost / pandas / numpy）。"""
    versions: dict[str, str] = {}
    for name in ("xgboost", "pandas", "numpy", "pyarrow"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "missing"
    return versions


def _write_run_manifest(run_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    """写 run manifest，并统一到 `payload.artifacts/upstream`（下游校验的唯一结构）。"""
    data = dict(payload)
    published = data.pop("artifacts", {}) or {}
    upstream = data.pop("upstream", {}) or {}
    data["payload"] = {"artifacts": published, "upstream": upstream}
    return artifacts.write_json_checked(Path(run_dir) / "manifest.json", data)


def _seed_specification(campaign: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """封存协议原文与机器规格（收益读取前冻结 SHA）。

    机器规格只含**数值身份**（协议镜像 + 冻结参数），不含代码 SHA 等随开发变动的字段；
    代码/环境身份由每个 run 在 `artifacts/inputs/specification.json` 内单独登记，
    这样规格可保持不可变，同时每次运行仍能证明"实际消费的是哪份代码"。
    """
    spec_dir = campaign / "specification"
    spec_dir.mkdir(parents=True, exist_ok=True)
    protocol = spec.protocol_path()
    snapshot = artifacts.copy_verbatim(protocol, spec_dir / "protocol_snapshot.md")
    document = spec.machine_spec_document(snapshot["sha256"])
    written = artifacts.write_json_checked(spec_dir / "machine_spec.json", document)
    log(f"specification: protocol={snapshot['sha256'][:12]} spec={written['sha256'][:12]}")
    return {"protocol": snapshot, "machine_spec": written, "document": document}


def _verify_baseline(root: Path) -> dict[str, Any]:
    """冻结 A100/A80 身份核验（逐成员 SHA；禁止缩放 A100 冒充 A80）。"""
    base = Path(root) / spec.A_BASELINE["relative"]
    manifest_path = base / "manifest.json"
    result: dict[str, Any] = {
        "root": str(base),
        "manifest_exists": manifest_path.exists(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "expected_manifest_sha256": spec.A_BASELINE["expected_manifest_sha256"],
        "members": {},
        "problems": [],
    }
    if not manifest_path.exists():
        result["problems"].append("missing_manifest")
        return result
    manifest_sha = result["manifest_sha256"]
    if manifest_sha != spec.A_BASELINE["expected_manifest_sha256"]:
        result["problems"].append(f"manifest_sha_mismatch:{manifest_sha[:12]}")
    for relative in spec.A_BASELINE["required_arm_dirs"]:
        capital, arm = relative.split("/")
        directory = base / capital / arm
        if not directory.exists():
            result["problems"].append(f"missing_arm:{relative}")
            continue
        members = artifacts.file_digests(directory)
        result["members"][relative] = {"count": len(members), "sha256": members}
    # 禁止缩放：A80 与 A100 必须各自有独立成员，不得以 A100 文件冒充
    for capital in ("1000000", "2000000"):
        a100 = base / capital / "A100"
        a80 = base / capital / "A80"
        if a100.exists() and a80.exists():
            left = artifacts.file_digests(a100)
            right = artifacts.file_digests(a80)
            if left and left == right:
                result["problems"].append(f"a80_equals_a100:{capital}")
    return result


def _download_plan(
    root: Path, calendar: Sequence[str], *, start: str, end: str, log: Callable[[str], None]
) -> dict[str, Any]:
    """按历史合格域计算下载计划（不含分钟值，不看收益）。

    请求量与行数都按**实际请求覆盖**（含 20 交易日预热）估计，不再用"合格股票日 × 49"
    低估返回行数。
    """
    inputs = dataset.load_inputs(root, start, end)
    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    dates = inputs["dates"]
    trading_days = [day for day in dates if start <= day <= end]
    chunk_days = int(spec.MINUTE["chunk_trading_days"])
    preheat_days = int(spec.WINDOW["minute_preheat_trading_days"])
    chunks_per_stock = int(np.ceil(len(trading_days) / chunk_days))
    window_idx = [i for i, day in enumerate(dates) if start <= day <= end]
    eligible = result.eligible[np.asarray(window_idx, dtype=int)]
    stock_days = int(eligible.sum())
    # 与 R1 共用同一覆盖实现（禁止两处口径分叉）
    covered = download.build_coverage(eligible, chunk_days, preheat_days)
    active_stocks = len({item.code_index for item in covered})
    requests_total = len(covered)
    requested_days = sum(item.end_pos - item.start_pos for item in covered)
    # 吞吐实测样本的股票池（按合格日数取前 10，预登记、不按结果调整）
    per_stock = eligible.sum(axis=0)
    order = np.argsort(-per_stock)[:10]
    codes = inputs["codes"]
    sample_codes = [str(codes[index]) for index in order if per_stock[index] > 0]
    rows_estimate = requested_days * int(spec.MINUTE["grid_slots"] + 1)
    bytes_estimate = rows_estimate * 64
    hours_at_100 = requests_total / (100.0 * 60.0)
    plan = {
        "window": [start, end],
        "trading_days": len(trading_days),
        "chunk_trading_days": chunk_days,
        "preheat_trading_days": preheat_days,
        "chunks_per_stock": chunks_per_stock,
        "eligible_stock_days": stock_days,
        "active_stocks": active_stocks,
        "requests_estimate": requests_total,
        "requested_stock_days": requested_days,
        "rows_estimate": rows_estimate,
        "uncompressed_bytes_estimate": bytes_estimate,
        "request_hours_at_100_per_minute": round(hours_at_100, 2),
        "r1_upper_hours": float(spec.BUDGET["stage_upper_hours"]["r1"]),
        "r1_download_cap_hours": float(spec.BUDGET["r1_download_max_hours"]),
        # 仅表示"按计划速率 100 次/分钟"可落在上限内，不代表已获得该吞吐
        "estimate_within_r1_cap_at_planned_rate": hours_at_100
        <= float(spec.BUDGET["r1_download_max_hours"]),
        "sample_codes": sample_codes,
        "condition_counts": result.condition_counts,
        "note": (
            "请求/行数为估算：每请求覆盖含 20 交易日预热，行数上限按 49 行/日/股计；"
            "吞吐由 R0 的 throughput_sample 实测判定"
        ),
    }
    log(
        f"download plan: stocks={active_stocks} stock_days={stock_days} "
        f"requests≈{requests_total} request_hours≈{hours_at_100:.2f}"
    )
    return plan


def session_hypothesis_five_ok(probes: Sequence[dict[str, Any]]) -> bool:
    """是否至少有 1 个 5 分钟探针成功（用于时段语义状态）。"""
    return any(
        item.get("status") == source.OK and item.get("parameters", {}).get("freq") == "5min"
        for item in probes
    )


def _local_sample_evidence() -> dict[str, Any]:
    """本地 1 分钟实样（2026-11-13，600000.SH）时段边界核验（不读收益）。"""
    path = (
        REPO_ROOT
        / "data/reports/p7/execution_cost_prescreen_20261115_r1/minute_probe_response.json"
    )
    evidence: dict[str, Any] = {"path": str(path.relative_to(REPO_ROOT)), "exists": path.exists()}
    if not path.exists():
        evidence["status"] = "missing_local_sample"
        return evidence
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    times = sorted(str(row["trade_time"])[-8:] for row in rows)
    evidence.update(
        {
            "rows": len(rows),
            "unique_times": len(set(times)),
            "first_time": times[0] if times else None,
            "last_time": times[-1] if times else None,
            "has_0930_auction": "09:30:00" in times,
            "am_rows": sum(1 for t in times if t <= "11:59:59"),
            "pm_rows": sum(1 for t in times if t >= "12:00:00"),
            "convention_inferred": "end",
            "vol_sum": float(sum(float(row["vol"]) for row in rows)),
            "amount_sum": round(sum(float(row["amount"]) for row in rows), 2),
            "status": "verified_1min",
        }
    )
    return evidence


def _fmt_optional(value: Any, digits: int = 0, unit: str = "") -> str:
    """可缺观测的展示：缺值只能写"未测得"，不得格式化成数字或当作无限。"""
    if value is None:
        return "未测得"
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.{digits}f}{suffix}"


def _strictest_quota(limits: Sequence[Mapping[str, Any]]) -> float | None:
    """同时生效的多个配额取**最严格**（最小速率），不是取最宽松。"""
    rates = [float(item["requests_per_hour"]) for item in limits if item.get("requests_per_hour")]
    return min(rates) if rates else None


def _throughput_assessment(
    plan: Mapping[str, Any],
    limits: Sequence[Mapping[str, Any]],
    sample: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """吞吐评估：配额证据 / 实测速率 / 未测得，三者严格区分。

    探针成功次数不能当吞吐测量（样本太小且未计时），因此 R0 只认两类证据：
    服务端明文配额窗口，或一段持续采样得到的成功请求/小时。
    """
    cap_hours = float(spec.BUDGET["r1_download_max_hours"])
    required = float(plan["requests_estimate"]) / max(cap_hours, 1e-9)
    strictest = _strictest_quota(limits)
    measured = None
    measured_hours = None
    if sample and sample.get("status") == "measured":
        measured = sample.get("success_per_hour")
        measured_hours = sample.get("elapsed_seconds", 0.0) / 3600.0
    effective = None
    if strictest is not None and measured is not None:
        effective = min(strictest, measured)
        basis = "quota_and_measured"
    elif measured is not None:
        effective = measured
        basis = "measured"
    elif strictest is not None:
        effective = strictest
        basis = "observed_quota_strictest_window"
    else:
        basis = "not_measured"
    return {
        "required_requests": int(plan["requests_estimate"]),
        "r1_cap_hours": cap_hours,
        "required_requests_per_hour": round(required, 1),
        "observed_quota_requests_per_hour": strictest,
        "quota_windows": [dict(item) for item in limits],
        "measured_success_per_hour": (round(measured, 1) if measured is not None else None),
        "measured_sample": dict(sample or {}),
        "basis": basis,
        "within_cap": (required <= effective) if effective is not None else None,
        "estimated_total_hours": (
            round(float(plan["requests_estimate"]) / effective, 2) if effective else None
        ),
        "projection_span_hours": measured_hours,
    }


def measure_throughput(
    src: "source.MinuteSource",
    days: Sequence[str],
    codes: Sequence[str],
    log: Callable[[str], None],
    *,
    budget_seconds: float | None = None,
) -> dict[str, Any]:
    """有界吞吐实测（预登记参数，不得按结果调参）。

    用**接近真实下载块**的 120 交易日窗口连续请求（跨股票轮换），记录请求数、行数、
    内存字节与墙钟；达到请求上限 / 时长上限 / 命中配额即停。样本不足（请求数或时长
    低于预登记下限）时如实写 `insufficient`，不外推。
    """
    import time as _time

    cfg = dict(spec.MINUTE["throughput_sample"])
    span = int(cfg["trading_days_per_request"])
    max_requests = int(cfg["max_requests"])
    max_seconds = float(budget_seconds or cfg["max_seconds"])
    if len(days) < span or not codes:
        return {"status": "insufficient_history", "days": len(days), "span": span}
    windows = []
    for index in range(max_requests):
        end = len(days) - (index // max(1, len(codes))) * span
        start = end - span
        if start < 0:
            break
        windows.append((codes[index % len(codes)], days[start], days[end - 1]))
    started = _time.perf_counter()
    requests = 0
    empty_responses = 0
    rows = 0
    bytes_in_memory = 0
    failures = 0
    blocked = None
    for code, first, last in windows:
        if _time.perf_counter() - started >= max_seconds:
            break
        try:
            fetched = src.fetch(
                code,
                f"{first[:4]}-{first[4:6]}-{first[6:]} 09:00:00",
                f"{last[:4]}-{last[4:6]}-{last[6:]} 15:31:00",
                str(spec.MINUTE["freq"]),
            )
        except source.RateLimitBlocked as error:
            blocked = error.as_record()
            break
        if fetched.status != source.OK:
            failures += 1
            continue
        if fetched.frame.empty:
            # OK 空表不算有效吞吐（可能是权限静默、窗口无交易或上游截断）
            empty_responses += 1
            continue
        requests += 1
        rows += int(len(fetched.frame))
        bytes_in_memory += int(fetched.frame.memory_usage(deep=True).sum())
    elapsed = max(_time.perf_counter() - started, 1e-6)
    enough = (
        requests >= int(cfg["min_requests_for_estimate"])
        and elapsed >= float(cfg["min_seconds_for_estimate"])
        and empty_responses == 0
    )
    result = {
        "status": "measured" if enough else ("blocked" if blocked else "insufficient"),
        "requests": requests,
        "empty_responses": empty_responses,
        "rows": rows,
        "in_memory_bytes": bytes_in_memory,
        "rows_per_request": (rows / requests) if requests else None,
        "elapsed_seconds": round(elapsed, 3),
        "success_per_hour": (requests * 3600.0 / elapsed) if requests else None,
        "failures": failures,
        "blocked": blocked,
        "config": cfg,
        "note": (
            "样本为 120 交易日窗口连续请求；OK 空表不计入有效吞吐；"
            "只用于判定能否在 R1 上限内完成全量，不把单次小请求外推成全库"
        ),
    }
    log(
        f"throughput sample: {result['status']} requests={requests} "
        f"rows={rows} elapsed={elapsed:.1f}s rate={result['success_per_hour']}"
    )
    return result


def _probe_sequence(
    src: "source.MinuteSource",
    cases: Sequence[tuple[str, str, str, str]],
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """按序探针；一旦命中已知配额就停止（不前台等、不做同窗口无效重试）。"""
    probes: list[dict[str, Any]] = []
    blocked: dict[str, Any] | None = None
    for ts_code, start, stop, freq in cases:
        if blocked is not None:
            probes.append(
                {
                    "parameters": {
                        "ts_code": ts_code,
                        "start_date": start,
                        "end_date": stop,
                        "freq": freq,
                    },
                    "status": "skipped_after_block",
                    "reason": "已知配额窗口内重试不会成功",
                }
            )
            log(f"probe {ts_code} {freq} {start[:10]}: skipped_after_block")
            continue
        item = src.probe_once(ts_code, start, stop, freq)
        probes.append(item)
        log(f"probe {ts_code} {freq} {start[:10]}: {item.get('status')}")
        if item.get("status") == source.RATE_LIMIT:
            # probe 把配额放在嵌套 `blocked`（与 fetch 同一结构）；读顶层会全部丢成 None
            payload = item.get("blocked") or {}
            blocked = {
                "limit": payload.get("limit") or item.get("limit"),
                "window_seconds": payload.get("window_seconds"),
                "next_allowed_at_utc": payload.get("next_allowed_at_utc"),
                "next_allowed_at_epoch": payload.get("next_allowed_at_epoch"),
                "message": payload.get("message") or item.get("error"),
            }
            log(
                "  blocked: "
                f"{json.dumps(blocked['limit'], ensure_ascii=False)} "
                f"next={blocked['next_allowed_at_utc']}"
            )
    return probes, blocked


def _chunk_consistency(
    src: "source.MinuteSource",
    ts_code: str,
    days: Sequence[str],
    log: Callable[[str], None],
    *,
    chunk_days: int | None = None,
    chunks: int = 8,
) -> dict[str, Any]:
    """分块一致性：同一长窗"一次取"与"分 N 块取"的行集合必须相同（协议 §5.2）。

    只核 (trade_time) 键集合与行数；不一致即视为游标/分块实现有缺陷，R0 不就绪。
    命中配额时返回 `blocked`，不重试。
    """
    span_days = int(chunk_days or spec.MINUTE["chunk_trading_days"])
    total = span_days * int(chunks)
    if len(days) < total:
        return {"status": "insufficient_history", "required_days": total, "available": len(days)}
    span = list(days)[-total:]

    def window(first: str, last: str) -> tuple[str, str]:
        return (
            f"{first[:4]}-{first[4:6]}-{first[6:]} 09:00:00",
            f"{last[:4]}-{last[4:6]}-{last[6:]} 15:31:00",
        )

    try:
        whole = src.fetch(ts_code, *window(span[0], span[-1]), str(spec.MINUTE["freq"]))
        parts = []
        for index in range(int(chunks)):
            piece = span[index * span_days : (index + 1) * span_days]
            got = src.fetch(ts_code, *window(piece[0], piece[-1]), str(spec.MINUTE["freq"]))
            if got.status != source.OK:
                return {"status": got.status, "chunk": index}
            parts.append(got.frame)
    except source.RateLimitBlocked as error:
        return {"status": source.RATE_LIMIT, **error.as_record()}
    if whole.status != source.OK:
        return {"status": whole.status}
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    whole_keys = set(whole.frame["trade_time"].astype(str))
    part_keys = set(merged["trade_time"].astype(str)) if len(merged) else set()
    equal = whole_keys == part_keys
    log(
        f"chunk consistency {ts_code}: whole={len(whole_keys)} parts={len(part_keys)} equal={equal}"
    )
    return {
        "status": "ok" if equal else "unequal",
        "rows_whole": int(len(whole.frame)),
        "rows_parts": int(len(merged)),
        "keys_whole": len(whole_keys),
        "keys_parts": len(part_keys),
        "missing_in_parts": len(whole_keys - part_keys),
        "extra_in_parts": len(part_keys - whole_keys),
    }


def _daily_rows(codes: Sequence[str], days: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """按日读 `clean/daily` 的量额高低（仅取需要的股票日）。"""
    wanted = set(str(code) for code in codes)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for day in sorted(set(days)):
        path = REPO_ROOT / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["ts_code", "close", "high", "low", "vol", "amount"])
        frame = frame[frame["ts_code"].astype(str).isin(wanted)]
        for row in frame.to_dict("records"):
            out[(str(row["ts_code"]), str(day))] = row
    return out


def _paired_reconciliation(
    src: "source.MinuteSource",
    pairs: Sequence[tuple[str, str]],
    log: Callable[[str], None],
) -> dict[str, Any]:
    """R0 配对对账：同一股票日同时取 1min 与 5min，与日线量额/高低核验。

    这是协议要求的"可对账样例"，不是可选的补充；拿不到就必须如实判未就绪。
    """
    daily = _daily_rows([code for code, _ in pairs], [day for _, day in pairs])
    threshold = float(spec.UNITS["minute_anomaly_volume_rel_error"])
    records: list[dict[str, Any]] = []
    blocked: dict[str, Any] | None = None
    for code, day in pairs:
        if blocked is not None:
            break
        row = daily.get((code, day))
        entry: dict[str, Any] = {"ts_code": code, "trade_date": day}
        if row is None:
            entry.update(status="no_daily_row")
            records.append(entry)
            continue
        window = (
            f"{day[:4]}-{day[4:6]}-{day[6:]} 09:00:00",
            f"{day[:4]}-{day[4:6]}-{day[6:]} 15:31:00",
        )
        try:
            five = src.fetch(code, window[0], window[1], "5min")
            one = src.fetch(code, window[0], window[1], "1min")
        except source.RateLimitBlocked as error:
            blocked = error.as_record()
            entry.update(status="blocked")
            records.append(entry)
            break
        five_check = quality.reconcile_day(five.frame, row)
        one_check = quality.reconcile_day(one.frame, row)
        ok = bool(
            five_check.get("reconcilable")
            and one_check.get("reconcilable")
            and (five_check.get("volume_rel_error") or 0.0) <= threshold
            and (five_check.get("amount_rel_error") or 0.0) <= threshold
            and (one_check.get("volume_rel_error") or 0.0) <= threshold
            and (one_check.get("amount_rel_error") or 0.0) <= threshold
        )
        entry.update(
            status="ok" if ok else "out_of_line",
            rows_5m=int(len(five.frame)),
            rows_1m=int(len(one.frame)),
            volume_rel_error_5m=five_check.get("volume_rel_error"),
            amount_rel_error_5m=five_check.get("amount_rel_error"),
            volume_rel_error_1m=one_check.get("volume_rel_error"),
            amount_rel_error_1m=one_check.get("amount_rel_error"),
        )
        records.append(entry)
        log(
            f"paired {code} {day}: 5m={entry['rows_5m']} 1m={entry['rows_1m']} "
            f"vol_err={_fmt_optional(entry['volume_rel_error_5m'], 6)}"
        )
    ok_count = sum(1 for item in records if item.get("status") == "ok")
    return {
        "target_pairs": len(pairs),
        "attempted_pairs": len(records),
        "ok_pairs": ok_count,
        "threshold": threshold,
        "records": records,
        "blocked": blocked,
        "status": (
            "blocked"
            if blocked
            else "ok" if ok_count >= len(pairs) and len(pairs) >= 8 else "insufficient"
        ),
    }


def run_r0(
    campaign: Path,
    run_dir: Path,
    *,
    log: Callable[[str], None],
    use_network: bool = True,
) -> dict[str, Any]:
    """R0：权限探针 + 时段语义 + 单位 + 规模估计；不读候选收益。"""
    started = time.perf_counter()
    work = run_dir / "work"
    inputs_dir = work / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    specification = _seed_specification(campaign, log)
    calendar = panel.load_calendar(REPO_ROOT)
    end = str(spec.WINDOW["end"])
    research_start = str(spec.MINUTE["research_start"])

    file_inputs = {}
    for item in spec.INPUTS:
        path = REPO_ROOT / item["relative"]
        exists = path.exists()
        entry: dict[str, Any] = {"exists": exists, "kind": item["kind"]}
        if exists and item["kind"] == "file":
            entry["sha256"] = sha256_file(path)
        elif exists:
            entry["files"] = len(list(path.glob("*.parquet")))
        file_inputs[item["name"]] = entry
    log(f"inputs: {sum(1 for v in file_inputs.values() if v['exists'])}/{len(file_inputs)} 就绪")

    baseline = _verify_baseline(REPO_ROOT)
    log(f"baseline: problems={baseline['problems']}")

    probes: list[dict[str, Any]] = []
    access_status = "unverified"
    probe_requests = 0
    trial_limits: list[dict[str, Any]] = []
    blocked_probe: dict[str, Any] | None = None
    chunk_check: dict[str, Any] = {"status": "not_attempted"}
    reconciliation: dict[str, Any] = {
        "status": "not_attempted",
        "ok_pairs": 0,
        "target_pairs": 8,
        "threshold": float(spec.UNITS["minute_anomaly_volume_rel_error"]),
        "records": [],
    }
    sample_5m: dict[str, Any] = {"status": "not_attempted"}
    recent_day: str | None = None
    if use_network:
        try:
            source_obj = source.MinuteSource(log=log)
            # 用主窗内最后一个交易日（trade_cal 可含未来年度，不能取 calendar[-1]）
            recent_day = [day for day in calendar if day <= end][-1]
            cases = [
                (
                    "600000.SH",
                    f"{recent_day[:4]}-{recent_day[4:6]}-{recent_day[6:]} 09:00:00",
                    f"{recent_day[:4]}-{recent_day[4:6]}-{recent_day[6:]} 15:31:00",
                    "5min",
                ),
                ("600000.SH", "2016-01-05 09:00:00", "2016-01-05 15:31:00", "5min"),
                ("000001.SZ", "2019-07-01 09:00:00", "2019-07-01 15:31:00", "5min"),
                ("600000.SH", "2026-11-13 09:00:00", "2026-11-13 15:31:00", "1min"),
            ]
            probes, blocked_probe = _probe_sequence(source_obj, cases, log)
            for item in probes:
                # 唯一结构：probe 与 fetch 都写 `blocked`；历史上 probe 写 frequency_limit 导致丢失
                limit = (item.get("blocked") or {}).get("limit")
                if limit:
                    trial_limits.append(dict(limit))
            probe_requests = source_obj.requests
            statuses = [item.get("status") for item in probes]
            successes = sum(1 for status in statuses if status == source.OK)
            limited = any(status == source.RATE_LIMIT for status in statuses)
            if statuses and all(status == source.TRANSPORT for status in statuses):
                access_status = "gateway_unavailable"
            elif any(status == source.PERMISSION for status in statuses):
                access_status = "permission_unavailable"
            elif successes and limited:
                access_status = "trial_rate_limited"
            elif successes:
                access_status = "ok_partial_sample"
            elif limited:
                access_status = "rate_limited_no_sample"
            elif any(status in (source.TRANSPORT, source.UNCLASSIFIED) for status in statuses):
                access_status = "partial_transport_failure"
            else:
                access_status = "unverified"
        except source.SourceError as error:
            access_status = "token_missing"
            probes.append({"status": "token_missing", "error": str(error)[:200]})

    sample = _local_sample_evidence()
    # 5 分钟样例与配对对账只在未被配额阻塞时尝试（一次机会，不反复试探服务端）
    if use_network and recent_day and blocked_probe is None and probes:
        try:
            capture = source_obj.capture(
                "600000.SH",
                f"{recent_day[:4]}-{recent_day[4:6]}-{recent_day[6:]} 09:00:00",
                f"{recent_day[:4]}-{recent_day[4:6]}-{recent_day[6:]} 15:31:00",
                str(spec.MINUTE["freq"]),
            )
            frame = capture.pop("frame")
            capture.pop("conflict_rows", None)
            sample_5m = {**capture, "trade_date": recent_day}
            if capture["status"] == source.OK and len(frame):
                raw_path = inputs_dir / "minute_sample_5m.parquet"
                frame.to_parquet(raw_path, index=False)
                times = sorted(frame["trade_time"].astype(str).tolist())
                sample_5m.update(
                    {
                        "sha256": sha256_file(raw_path),
                        "first_time": times[0],
                        "last_time": times[-1],
                        "has_0930_auction": any(item.endswith("09:30:00") for item in times),
                        "core_rows": sum(1 for item in times if not item.endswith("09:30:00")),
                        "vol_sum": float(pd.to_numeric(frame["vol"], errors="coerce").sum()),
                        "amount_sum": float(pd.to_numeric(frame["amount"], errors="coerce").sum()),
                    }
                )
                log(f"5min sample: {len(frame)} 行，核心 {sample_5m['core_rows']} 行")
        except source.RateLimitBlocked as error:
            sample_5m = {"status": source.RATE_LIMIT, **error.as_record()}
        except source.TransportFailure as error:
            sample_5m = {"status": source.TRANSPORT, "error": str(error)[:200]}
        artifacts.write_json_checked(inputs_dir / "minute_sample_5m.json", sample_5m)
        if sample_5m.get("status") == source.RATE_LIMIT:
            blocked_probe = blocked_probe or {
                "limit": sample_5m.get("limit"),
                "next_allowed_at_utc": sample_5m.get("next_allowed_at_utc"),
            }
        else:
            pairs = [
                ("600000.SH", recent_day),
                ("600000.SH", calendar[max(0, calendar.index(recent_day) - 1)]),
                ("000001.SZ", recent_day),
                ("000001.SZ", calendar[max(0, calendar.index(recent_day) - 1)]),
                ("600519.SH", recent_day),
                ("600519.SH", calendar[max(0, calendar.index(recent_day) - 1)]),
                ("002415.SZ", recent_day),
                ("002415.SZ", calendar[max(0, calendar.index(recent_day) - 1)]),
            ]
            reconciliation = _paired_reconciliation(source_obj, pairs, log)
            if reconciliation.get("status") == "ok" and blocked_probe is None:
                window_days = [day for day in calendar if research_start <= day <= end]
                chunk_check = _chunk_consistency(source_obj, "600000.SH", window_days, log)
        probe_requests = source_obj.requests

    plan = _download_plan(REPO_ROOT, calendar, start=research_start, end=end, log=log)

    # 吞吐实测（预登记样本）；命中配额或样本不足一律如实记录，不外推
    window_days = [day for day in calendar if research_start <= day <= end]
    sample_throughput: dict[str, Any] = {"status": "not_attempted"}
    if use_network and probes and blocked_probe is None:
        sample_throughput = measure_throughput(
            source_obj, window_days, plan.get("sample_codes", []), log
        )
        if sample_throughput.get("blocked"):
            blocked_probe = blocked_probe or sample_throughput["blocked"]
            limit = (sample_throughput["blocked"] or {}).get("limit")
            if limit:
                trial_limits.append(dict(limit))
        probe_requests = source_obj.requests

    session_hypothesis = {
        "target_grid": "k=1..48 收盘时刻 09:35..11:30 / 13:05..15:00",
        "vendor_convention_assumed": spec.SESSION_MAPPING["vendor_convention"],
        "auction_record": spec.SESSION_MAPPING["auction_record"],
        "local_1min_evidence": sample,
        "five_minute_sample": sample_5m,
        "five_minute_verified": bool(
            session_hypothesis_five_ok(probes) or sample_5m.get("core_rows") == 48
        ),
        "sse_closing_auction_from": spec.MINUTE["sse_closing_auction_from"],
        "status": (
            "verified_5min_and_1min"
            if (session_hypothesis_five_ok(probes) or sample_5m.get("core_rows") == 48)
            and sample.get("exists")
            else "verified_1min_5min_pending" if sample.get("exists") else "unresolved"
        ),
    }
    units = {
        "minute_vol": "share",
        "minute_amount": "yuan",
        "daily_vol": "hand(100 shares)",
        "daily_amount": "thousand_yuan",
        "total_mv": "ten_thousand_yuan",
        "reconciliation": "minute_sum vs daily×100（股）×1000（元）",
        "anomaly_lines": {
            "volume_rel_error": spec.UNITS["minute_anomaly_volume_rel_error"],
            "ohlc_ticks": spec.UNITS["minute_anomaly_ohlc_ticks"],
        },
        "local_sample_vol_sum": sample.get("vol_sum"),
        "local_sample_amount_sum": sample.get("amount_sum"),
        "status": "verified_by_local_sample_units" if sample.get("exists") else "unverified",
    }

    # 吞吐：只认"服务端配额"与"持续实测"，未测得即未测得（不当无限）
    throughput = _throughput_assessment(plan, trial_limits, sample_throughput)
    log(
        "throughput: required≈"
        f"{throughput['required_requests_per_hour']:.0f}/h basis={throughput['basis']} "
        f"measured={throughput['measured_success_per_hour']} quota="
        f"{throughput['observed_quota_requests_per_hour']} within={throughput['within_cap']}"
    )

    access_report = {
        "access_status": access_status,
        "requests": len(probes),
        "probes": probes,
        "trial_rate_limits": trial_limits,
        "blocked": blocked_probe,
        "throughput": throughput,
        "throughput_sample": sample_throughput,
        "prior_evidence": {
            "path": "logs/stk_mins_depth_check_20261118.json（会话内临时日志，非冻结证据）",
            "claim": "同日晚间曾观察到 1 次 5min 成功返回 49 行/日",
            "status": "unverified_anecdote",
            "why": "未随 run 固化原件与 SHA，且 48/49 行数量不证明起止时点与竞价归属",
        },
        "new_procurement_yuan": 0,
        "note": (
            "R0 只核验数据与规模，不读候选收益；40203 频次标记归 rate_limit，"
            "不把 4xx/5xx 一概归为没权限；命中配额后立即停止本批次，不在同窗口重试"
        ),
    }
    artifacts.write_json_checked(inputs_dir / "access.json", access_report)
    artifacts.write_json_checked(inputs_dir / "download_plan.json", plan)
    artifacts.write_json_checked(inputs_dir / "session_semantics.json", session_hypothesis)
    artifacts.write_json_checked(inputs_dir / "units.json", units)
    artifacts.write_json_checked(inputs_dir / "inputs.json", file_inputs)
    artifacts.write_json_checked(inputs_dir / "baseline_identity.json", baseline)
    code_info = code_identity()
    artifacts.write_json_checked(
        inputs_dir / "specification.json",
        {
            **specification["document"],
            "code": code_info,
            "environment": _env_identity(),
        },
    )

    five_probes = [
        item
        for item in probes
        if item.get("parameters", {}).get("freq") == str(spec.MINUTE["freq"])
    ]
    five_ok = [item for item in five_probes if item.get("status") == source.OK]
    cap_hours = float(spec.BUDGET["r1_download_max_hours"])
    checks = {
        "access_three_5m_windows": {
            "ok": len(five_ok) == 3 and len(five_probes) == 3,
            "detail": (
                f"5min 成功 {len(five_ok)}/{len(five_probes)}（近期/2016/2019）"
                if five_probes
                else "未尝试（离线或未联网）"
            ),
        },
        "session_mapping_5m": {
            "ok": sample_5m.get("core_rows") == int(spec.MINUTE["grid_slots"]),
            "detail": f"5min 样例核心槽 {sample_5m.get('core_rows')}（应 48），"
            f"09:30 竞价={sample_5m.get('has_0930_auction')}",
        },
        "session_mapping_1m": {
            "ok": bool(sample.get("exists")) and sample.get("rows", 0) >= 200,
            "detail": f"本地 1min 实样 {sample.get('rows')} 行，首尾 "
            f"{sample.get('first_time')}..{sample.get('last_time')}",
        },
        "paired_reconciliation": {
            "ok": reconciliation.get("status") == "ok",
            "detail": f"配对对账 {reconciliation.get('status')}，"
            f"{reconciliation.get('ok_pairs')}/{reconciliation.get('target_pairs')} 通过",
        },
        "throughput_measured": {
            "ok": sample_throughput.get("status") == "measured",
            "detail": (
                f"{sample_throughput.get('status')}：请求 {sample_throughput.get('requests')}、"
                f"行 {sample_throughput.get('rows')}、"
                f"耗时 {sample_throughput.get('elapsed_seconds')} 秒"
            ),
        },
        "throughput": {
            "ok": throughput.get("within_cap") is True,
            "detail": (
                f"需 {_fmt_optional(throughput['required_requests_per_hour'], 0, '请求/小时')}"
                f"（{cap_hours:g} 小时上限），"
                f"实测 {_fmt_optional(throughput['measured_success_per_hour'], 0, '请求/小时')}，"
                f"配额 {_fmt_optional(throughput['observed_quota_requests_per_hour'], 0, '请求/小时')}，"
                f"依据 {throughput['basis']}，预计总耗时 "
                f"{_fmt_optional(throughput['estimated_total_hours'], 1, '小时')}"
            ),
        },
        "chunk_consistency": {
            "ok": chunk_check.get("status") == "ok",
            "detail": json.dumps(chunk_check, ensure_ascii=False)[:240],
        },
        "no_unresolved_access_block": {
            # 阻塞只有在"之后取得了有效实测样本"时才算解除；否则保持未就绪
            "ok": blocked_probe is None or sample_throughput.get("status") == "measured",
            "detail": (
                "无已知配额阻塞"
                if blocked_probe is None
                else f"阻塞至 {blocked_probe.get('next_allowed_at_utc')}；"
                f"实测样本 {sample_throughput.get('status')}"
            ),
        },
    }
    data_ready = all(item["ok"] for item in checks.values())
    data_status = "ready" if data_ready else "access_not_ready"
    readiness = artifacts.write_readiness(inputs_dir, checks, data_status=data_status)
    unmet = (
        []
        if data_ready
        else [f"{name}: {item['detail']}" for name, item in checks.items() if not item["ok"]]
    )
    # 工程状态只由**可核的本地项**决定；数据依赖项单独列，不外推"工程已就绪"
    local_checks = {
        "inputs_present": all(v["exists"] for v in file_inputs.values()),
        "baseline_identity_clean": not baseline["problems"],
        "specification_frozen": bool(specification["machine_spec"]["sha256"]),
    }
    if not all(local_checks.values()):
        engineering_status = "invalid"
    elif data_ready:
        engineering_status = "valid"
    else:
        engineering_status = "partial"
    decision = report.decide(
        engineering_status=engineering_status,
        data_status=data_status,
        targets=None,
        pressure=None,
        recent_flag=None,
    )
    elapsed = time.perf_counter() - started
    machine_hours = {
        "r0": round(elapsed / 3600.0, 3),
        "r0_capture_hours": round(elapsed / 3600.0, 3),
        "budget_total": spec.BUDGET["total_machine_hours"],
        "budget_r0_cap": spec.BUDGET["stage_upper_hours"]["r0"],
        "within_cap": (elapsed / 3600.0) <= spec.BUDGET["stage_upper_hours"]["r0"],
    }
    summary = report.summary_payload(
        identity={
            "candidate_id": spec.CANDIDATE_ID,
            "campaign_id": campaign.name,
            "run_id": run_dir.name,
            "protocol_sha256": specification["protocol"]["sha256"],
            "spec_sha256": specification["document"]["spec_sha256"],
            "code_commit": code_info["code_commit"],
            "dirty_patch_sha": code_info["dirty_patch_sha"],
            "baseline_sha": baseline.get("manifest_sha256"),
        },
        engineering_status=engineering_status,
        data_status=data_status,
        targets_by_capital={},
        pressure_status={"status": "not_run_data_stop" if not data_ready else "not_run"},
        decision=decision,
        limitations=[
            "本研究只形成历史证据等级 H，不构成前向/日历验证",
            "2019—2026 已被项目多次研究，不是全新未触碰 holdout",
            "R0 规模为估算值（含 20 交易日预热，49 行/日/股上限），不是实测",
            "定向测试只覆盖纯函数与被替换的 Source 调用，不覆盖真实状态机；"
            "测试数量与命令以交付日志为准，本报告不重复断言",
        ],
        unmet_items=unmet,
        machine_hours=machine_hours,
        requests=int(probe_requests),
        bytes_written=sum(path.stat().st_size for path in inputs_dir.glob("*") if path.is_file()),
        new_data_cost=0,
        verification_scope={
            "local_synthetic_tests": "passed（定向，命令与数量见交付日志；不含全量）",
            "artifact_integrity": "ok",
            "data_access": access_status,
            "r0_semantics": (
                "verified_5m_and_1m"
                if checks["session_mapping_5m"]["ok"] and checks["session_mapping_1m"]["ok"]
                else "unverified"
            ),
            "r0_throughput": throughput["basis"],
            "r0_paired_reconciliation": reconciliation.get("status"),
            "r1_plus": "not_run",
        },
        budget=artifacts.budget_status(campaign, float(spec.BUDGET["total_machine_hours"])),
    )
    extra = {
        "increment": [
            f"本地 1 分钟样例：{sample.get('rows')} 行、09:30 竞价={sample.get('has_0930_auction')}、"
            f"首尾 {sample.get('first_time')}..{sample.get('last_time')}",
            f"下载计划：活跃股票 {plan['active_stocks']} 只、合格股票日 {plan['eligible_stock_days']}、"
            f"请求 {plan['requests_estimate']}（含 20 日预热覆盖 "
            f"{plan['requested_stock_days']} 股票日）、"
            f"按 100 次/分钟约 {plan['request_hours_at_100_per_minute']} 小时",
            f"吞吐：需 {_fmt_optional(throughput['required_requests_per_hour'], 0, '请求/小时')}"
            f"（{cap_hours:g} 小时上限），"
            f"配额 {_fmt_optional(throughput['observed_quota_requests_per_hour'], 0, '请求/小时')}，"
            f"实测 {_fmt_optional(throughput['measured_success_per_hour'], 0, '请求/小时')}，"
            f"依据 {throughput['basis']}，预计总耗时 "
            f"{_fmt_optional(throughput['estimated_total_hours'], 1, '小时')}",
            "吞吐样本：" + json.dumps(sample_throughput, ensure_ascii=False, default=str),
            "探针结果："
            + json.dumps(
                [
                    {
                        "freq": item.get("parameters", {}).get("freq"),
                        "day": item.get("parameters", {}).get("start_date", "")[:10],
                        "status": item.get("status"),
                        "rows": item.get("rows"),
                    }
                    for item in probes
                ],
                ensure_ascii=False,
            ),
            "5 分钟样例："
            + json.dumps({k: v for k, v in sample_5m.items() if k != "frame"}, ensure_ascii=False),
            f"配对对账：{reconciliation.get('status')}，"
            f"{reconciliation.get('ok_pairs')}/{reconciliation.get('target_pairs')} 股票日通过"
            f"（阈值 {reconciliation.get('threshold')}）",
            "R0 检查项：" + json.dumps(checks, ensure_ascii=False),
        ],
        "account": ["R0 未运行账户；账户自验在 R3 之前以合成夹具完成"],
        "next_step": (
            "R0 未就绪：按 checks 逐项补齐（权限/语义/对账/吞吐），不得改频率、缩窗或提前成交绕过。"
            "本次不新增采购建议；若用户另行取得合法分钟权限，再以同规格重跑 R0。"
            if not data_ready
            else "进入 R1：按请求覆盖下载、raw 先行、逐块断点与覆盖裁定"
        ),
    }
    report.write_report(inputs_dir, summary, extra)
    published = artifacts.publish_artifacts(
        work,
        run_dir,
        "r0",
        STAGE_ARTIFACTS["r0"],
        {"specification": specification["machine_spec"]["sha256"]},
    )
    manifest = {
        "stage": "r0",
        "run_id": run_dir.name,
        "campaign_id": campaign.name,
        "generated_at": artifacts.utc_now(),
        "identity": summary,
        "upstream": {"specification": specification["machine_spec"]},
        "artifacts": published,
        "state": {
            "data_status": data_status,
            "access_status": access_status,
            "readiness_sha256": readiness["sha256"],
            "checks": {name: item["ok"] for name, item in checks.items()},
        },
    }
    _write_run_manifest(run_dir, manifest)
    artifacts.record_machine_hours(
        campaign,
        run_dir.name,
        {
            "run_id": run_dir.name,
            "stage": "r0",
            "machine_hours": machine_hours["r0"],
            "requests": int(probe_requests),
            "measured_at": artifacts.utc_now(),
        },
    )
    artifacts.write_state(
        run_dir,
        {
            "stage": "r0",
            "status": "completed",
            "data_status": data_status,
            "access_status": access_status,
            "machine_hours": machine_hours,
            "finished_at": artifacts.utc_now(),
        },
    )
    log(f"r0 done: data_status={data_status} access={access_status}")
    return {"summary": summary, "access": access_report, "plan": plan}


def _resolve_campaign(args: argparse.Namespace) -> tuple[Path, Path]:
    """解析 campaign 与 run 目录（已存在即拒绝）。"""
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    if getattr(args, "resume", False):
        run_dir = artifacts.open_run_for_resume(campaign, args.run_id, base)
        return campaign, run_dir
    run_id = args.run_id or artifacts.new_run_id(args.stage)
    run_dir = artifacts.create_run_dir(campaign, run_id, base)
    return campaign, run_dir


def _require_upstream(campaign: Path, run_id: str | None, name: str) -> Path:
    """下游上游入口：必须"已发布 + readiness 全部成立"，不只是文件存在。"""
    if not run_id:
        raise artifacts.ArtifactError(f"缺少 --upstream（需要已就绪的 {name}）")
    return artifacts.require_ready(campaign, run_id, name)


def run_r1(
    campaign: Path,
    run_dir: Path,
    *,
    upstream: str | None,
    log: Callable[[str], None],
    resume: bool = False,
) -> dict[str, Any]:
    """R1：按请求覆盖下载、raw 先行、逐批断点、对账与覆盖裁定。

    启动条件只看上游 R0 的 `readiness.json`（data_status=ready 且逐项成立），
    不再用 `access_status` 代替验收（D2）。
    """
    started = time.perf_counter()
    if not upstream:
        raise artifacts.ArtifactError("缺少 --upstream（需要已就绪的 R0 inputs）")
    inputs_dir = artifacts.require_ready(campaign, upstream, "inputs")
    plan = json.loads((inputs_dir / "download_plan.json").read_text(encoding="utf-8"))
    log(f"R1 计划：{plan['requests_estimate']} 请求、{plan['eligible_stock_days']} 合格股票日")

    calendar = panel.load_calendar(REPO_ROOT)
    inputs = dataset.load_inputs(
        REPO_ROOT, str(spec.MINUTE["research_start"]), str(spec.WINDOW["end"])
    )
    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    dates = inputs["dates"]
    low, high = str(spec.MINUTE["research_start"]), str(spec.WINDOW["end"])
    positions = [index for index, day in enumerate(dates) if low <= str(day) <= high]
    days = [str(dates[index]) for index in positions]
    eligible = result.eligible[np.asarray(positions, dtype=int)]
    codes = inputs["codes"]

    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    src = source.MinuteSource(log=log)
    engine = download.DownloadEngine(
        campaign=campaign,
        work_dir=work,
        src=src,
        days=days,
        codes=codes,
        eligible=eligible,
        log=log,
        resume=resume,
    )
    engine.run()
    rep = engine.report
    checks = {
        "download_not_blocked": {
            "ok": rep.blocked is None,
            "detail": (
                "无配额阻塞"
                if rep.blocked is None
                else f"阻塞：{rep.blocked.get('limit')}，可重试于 {rep.blocked.get('next_allowed_at_utc')}"
            ),
        },
        "requests_all_ok": {
            "ok": rep.failed_requests == 0 and rep.requests > 0,
            "detail": f"请求 {rep.requests}，失败 {rep.failed_requests}，成功 {rep.ok_requests}",
        },
        "raw_blocks_committed": {
            "ok": rep.raw_blocks > 0,
            "detail": f"raw 内容寻址块 {rep.raw_blocks} 个",
        },
        "no_source_conflicts": {
            "ok": rep.conflicts == 0,
            "detail": f"同键异值 {rep.conflicts} 组（已隔离原始双方，不计入覆盖）",
        },
        "coverage_ready": {
            "ok": rep.coverage.get("status") == "ready",
            "detail": (
                f"应覆盖 {rep.coverage.get('expected_rows')}，接受 {rep.coverage.get('accepted_rows')}，"
                f"未解决 {rep.coverage.get('unresolved_count')}，"
                f"异常 {rep.coverage.get('anomaly_count')}"
            ),
        },
        "warmup_quality": {
            # 预热行是后续特征输入：缺失未解释即阻断，不能只计入请求量
            "ok": int(rep.coverage.get("warmup_unresolved", 0)) == 0,
            "detail": (
                f"预热台账 {rep.coverage.get('warmup')}，"
                f"未解决 {rep.coverage.get('warmup_unresolved')}"
            ),
        },
        "reconciliation_quality": {
            # 0.5% 是调查线：越线项必须有机械解释并披露；未解释项阻断
            "ok": rep.reconciliation.get("status") == "measured",
            "detail": (
                f"对账行 {rep.reconciliation.get('rows')}，"
                f"线内 {rep.reconciliation.get('within_line')}，"
                f"已解释异常 {rep.reconciliation.get('disclosed_anomalies')}，"
                f"未解释 {rep.reconciliation.get('unexplained')}，"
                f"处置 {rep.reconciliation.get('disposition_counts')}"
            ),
        },
    }
    data_status = "ready" if all(item["ok"] for item in checks.values()) else "data_not_ready"
    dataset_dir = work / "dataset"
    storage = {
        "raw_blocks_bytes": (
            sum(path.stat().st_size for path in (campaign / "raw").rglob("*.parquet"))
            if (campaign / "raw").exists()
            else 0
        ),
        "raw_spool_bytes": rep.raw_spool_bytes,
        "standardized_bytes": (
            sum(path.stat().st_size for path in (dataset_dir / "minute").rglob("*.parquet"))
            if (dataset_dir / "minute").exists()
            else 0
        ),
        "aux_bytes": (
            sum(
                path.stat().st_size
                for path in dataset_dir.rglob("*.parquet")
                if "minute" not in path.parts
            )
            if dataset_dir.exists()
            else 0
        ),
    }
    storage["total_bytes"] = int(sum(storage.values()))
    if resume:
        # 续跑会重算这些派生件；它们属于未发布的工作态，允许覆盖（发布件仍不可变）
        for name in ("standardized_manifest.json", "summary.json", "report.md"):
            stale = dataset_dir / name
            if stale.exists():
                stale.unlink()
    readiness = artifacts.write_readiness(
        dataset_dir, checks, data_status=data_status, allow_update=resume
    )
    elapsed_hours = round((time.perf_counter() - started) / 3600.0, 3)
    machine_hours = {
        "r1": elapsed_hours,
        "r1_download_hours": rep.duration_hours,
        "budget_total": spec.BUDGET["total_machine_hours"],
        "budget_r1_cap": spec.BUDGET["stage_upper_hours"]["r1"],
        "within_cap": elapsed_hours <= float(spec.BUDGET["stage_upper_hours"]["r1"]),
    }
    files = artifacts.file_digests(dataset_dir)
    artifacts.write_json_checked(
        dataset_dir / "standardized_manifest.json",
        {
            "files": files,
            "requests": src.requests,
            "retries": src.retries,
            "download": rep.as_payload(),
            "readiness": {k: v for k, v in readiness.items() if k != "sha256"},
        },
    )
    summary = report.summary_payload(
        identity={
            "candidate_id": spec.CANDIDATE_ID,
            "campaign_id": campaign.name,
            "run_id": run_dir.name,
            "spec_sha256": json.loads(
                (inputs_dir / "specification.json").read_text(encoding="utf-8")
            ).get("spec_sha256"),
            "r0_run_id": upstream,
        },
        engineering_status="valid" if rep.blocked is None else "partial",
        data_status=data_status,
        targets_by_capital={},
        pressure_status={"status": "not_run"},
        decision=report.decide(
            engineering_status="valid",
            data_status=data_status,
            targets=None,
            pressure=None,
            recent_flag=None,
        ),
        limitations=[
            "R1 只产出分钟面板与对账证据，不含经济读数",
            "对账异常线用于调查分类，不是删除整库的零容忍门",
        ],
        unmet_items=[
            f"{name}: {item['detail']}" for name, item in checks.items() if not item["ok"]
        ],
        machine_hours=machine_hours,
        requests=int(src.requests),
        bytes_written=int(storage["total_bytes"]),
        new_data_cost=0,
        verification_scope={
            "download": "ok" if rep.blocked is None else "blocked",
            "coverage": rep.coverage.get("status"),
            "reconciliation": rep.reconciliation.get("status"),
            "r2_plus": "not_run",
        },
        budget=artifacts.budget_status(campaign, float(spec.BUDGET["total_machine_hours"])),
        storage=storage,
    )
    report.write_report(
        dataset_dir,
        summary,
        {"increment": ["见 raw_index/、reconciliation.parquet 与 readiness.json"]},
    )
    if rep.blocked is not None:
        # 阻塞=未完成：**不发布** artifacts，保留工作态供同一 run 续跑；
        # 已下载的 raw 已在 raw_spool/ 与 campaign/raw 中逐请求落盘，不会因中断丢失。
        blocked_payload = {
            "blocked": rep.blocked,
            "requests_done": rep.ok_requests,
            "requests_total": rep.requests,
            "next_allowed_at_utc": rep.blocked.get("next_allowed_at_utc"),
            "resume_command": (
                f"python -m scripts.v2_p7.minute_repair_ml.run --stage r1 "
                f"--campaign-id {campaign.name} --run-id {run_dir.name} "
                f"--upstream {upstream} --resume"
            ),
            "finished_at": artifacts.utc_now(),
        }
        staged = work / "blocked.partial.json"
        staged.write_text(
            json.dumps(blocked_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staged.replace(work / "blocked.json")
        artifacts.write_state(
            run_dir,
            {
                "stage": "r1",
                "status": "blocked",
                "data_status": "data_not_ready",
                "reason": "rate_limit",
                "next_allowed_at_utc": rep.blocked.get("next_allowed_at_utc"),
                "machine_hours": machine_hours,
                "finished_at": artifacts.utc_now(),
            },
        )
        log(
            f"r1 blocked: 已保存 {rep.ok_requests}/{rep.requests} 请求，"
            f"下次可试 {rep.blocked.get('next_allowed_at_utc')}；未发布 artifacts"
        )
        return {
            "requests": src.requests,
            "data_status": "blocked",
            "resume": run_dir.name,
        }
    published = artifacts.publish_artifacts(
        work, run_dir, "r1", STAGE_ARTIFACTS["r1"], {"r0": inputs_dir.name}
    )
    _write_run_manifest(
        run_dir,
        {
            "stage": "r1",
            "run_id": run_dir.name,
            "campaign_id": campaign.name,
            "generated_at": artifacts.utc_now(),
            "upstream": {"r0": upstream},
            "artifacts": published,
            "state": {"data_status": data_status},
        },
    )
    artifacts.record_machine_hours(
        campaign,
        run_dir.name,
        {
            "run_id": run_dir.name,
            "stage": "r1",
            "machine_hours": elapsed_hours,
            "requests": int(src.requests),
            "measured_at": artifacts.utc_now(),
        },
    )
    artifacts.write_state(
        run_dir,
        {
            "stage": "r1",
            "status": "completed" if rep.blocked is None else "blocked",
            "data_status": data_status,
            "machine_hours": machine_hours,
            "finished_at": artifacts.utc_now(),
        },
    )
    log(f"r1 done: data_status={data_status} requests={src.requests}")
    return {"requests": src.requests, "years": sorted(rep.years), "data_status": data_status}


def run_r2(
    campaign: Path, run_dir: Path, *, upstream: str | None, log: Callable[[str], None]
) -> dict[str, Any]:
    """R2：24 列/标签/16 折/64 次拟合/完整 OOS 排名。"""
    r1_dir = _require_upstream(campaign, upstream, "dataset")
    minute_dir = r1_dir / "minute"
    calendar = panel.load_calendar(REPO_ROOT)
    dataset_dir = run_dir / "work" / "dataset"
    dataset.build_dataset(
        REPO_ROOT,
        minute_dir=minute_dir,
        feature_start=str(spec.WINDOW["start"]),
        end=str(spec.WINDOW["end"]),
        out_dir=dataset_dir,
        source_manifest_sha=sha256_file(r1_dir / "standardized_manifest.json"),
        log=log,
    )
    folds = splits.build_folds(calendar, str(spec.WINDOW["end"]))
    splits.assert_fold_count(folds)
    panel_frame = training.load_panel(dataset_dir)
    n_jobs = spec.n_jobs(os.cpu_count() or 1)
    registry = _model_registry(campaign)
    fold_results = []
    for fold in folds:
        fold_results.append(training.run_fold(panel_frame, fold, registry, n_jobs, log=log))
    training.fold_manifest(dataset_dir / "fold_models.json", {"folds": fold_results})

    predictions = []
    for fold, result_row in zip(folds, fold_results):
        for variant in ("main", "daily"):
            predictions.append(
                prediction.score_fold(
                    panel_frame, fold, registry, result_row["models"], variant, log=log
                )
            )
    frozen = prediction.freeze_predictions(predictions, dataset_dir)
    ranked_main = prediction.ranked_from_predictions(frozen[frozen["variant"] == "main"])
    ranked_daily = prediction.ranked_from_predictions(frozen[frozen["variant"] == "daily"])
    artifacts.write_json_checked(
        dataset_dir / "ranked_manifest.json",
        {
            "main_days": len(ranked_main),
            "daily_days": len(ranked_daily),
            "skip_days_main": sorted(prediction.skip_days(frozen[frozen["variant"] == "main"])),
            "skip_days_daily": sorted(prediction.skip_days(frozen[frozen["variant"] == "daily"])),
        },
    )
    published = artifacts.publish_artifacts(
        run_dir / "work", run_dir, "r2", STAGE_ARTIFACTS["r2"], {"r1": upstream}
    )
    _write_run_manifest(
        run_dir,
        {
            "stage": "r2",
            "run_id": run_dir.name,
            "campaign_id": campaign.name,
            "generated_at": artifacts.utc_now(),
            "upstream": {"r1": upstream},
            "artifacts": published,
        },
    )
    return {"folds": len(folds), "predictions": int(len(frozen))}


def _model_registry(campaign: Path) -> Any:
    from src.lazybull.v2.common.model_registry import ModelRegistry

    models_dir = Path(campaign) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return ModelRegistry(models_dir=str(models_dir))


def run_r3(
    campaign: Path, run_dir: Path, *, upstream: str | None, log: Callable[[str], None]
) -> dict[str, Any]:
    """R3：两资本账户、A 基准组合、对照、压力、独立审计与报告。"""
    r2_dir = _require_upstream(campaign, upstream, "predictions")
    frozen = pd.concat(
        [pd.read_parquet(path) for path in sorted(r2_dir.glob("predictions_*.parquet"))],
        ignore_index=True,
    )
    ranked = {
        "main": prediction.ranked_from_predictions(frozen[frozen["variant"] == "main"]),
        "daily": prediction.ranked_from_predictions(frozen[frozen["variant"] == "daily"]),
    }
    days = native_replay.replay_days(REPO_ROOT)
    prices, price_identity = native_replay.load_prices_with_warmup(REPO_ROOT, days)
    from src.lazybull.v2.core.execution.price_index import prepare_price_index

    indexes = prepare_price_index(prices)
    pit, l1_names = native_replay.load_industry(REPO_ROOT)
    corporate_book, corporate_audit = native_replay.prepare_corporate(REPO_ROOT, campaign, days)
    baseline = _verify_baseline(REPO_ROOT)
    if baseline["problems"]:
        raise RuntimeError(f"冻结 A 基准身份核验失败：{baseline['problems'][:3]}")

    accounts_dir = run_dir / "work" / "accounts"
    diagnostics_dir = run_dir / "work" / "diagnostics"
    reports_dir = run_dir / "work" / "reports"
    for directory in (accounts_dir, diagnostics_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    targets_by_capital: dict[str, Any] = {}
    for capital in spec.TARGETS["capital_scenarios"]:
        b_capital = float(capital) * float(spec.TARGETS["initial_b_fraction"])
        arms: dict[str, pd.Series] = {}
        for label, ranked_map, slippage in (
            ("LR20", ranked["main"], None),
            ("DAILY20", ranked["daily"], None),
        ):
            result = native_replay.run_arm(
                REPO_ROOT,
                ranked=ranked_map,
                days=days,
                capital=b_capital,
                corporate_book=corporate_book,
                pit=pit,
                l1_names=l1_names,
                anchor=str(spec.WINDOW["cash_anchor"]),
                slippage=slippage,
                prices=prices,
                indexes=indexes,
                log=log,
            )
            tag = f"{capital}_{label}"
            result["daily"].to_parquet(accounts_dir / f"{tag}_daily.parquet", index=False)
            result["trades"].to_parquet(accounts_dir / f"{tag}_trades.parquet", index=False)
            artifacts.write_json_checked(
                accounts_dir / f"{tag}_summary.json",
                {
                    "validation": result["validation"],
                    "diagnostics": result["diagnostics"],
                    "ending": result["ending"],
                },
            )
            arms[label] = result["daily"].set_index("trade_date")["total_value"].astype(float)
        a100 = native_replay.load_a_series(REPO_ROOT, capital, "A100")
        a80 = native_replay.load_a_series(REPO_ROOT, capital, "A80")
        combined = native_replay.combine_arms(a100, a80, arms, b_capital)
        metrics = analysis.arm_metrics(combined)
        targets = analysis.target_table(metrics)
        targets_by_capital[str(capital)] = {"metrics": metrics, "targets": targets}
        combined.to_parquet(accounts_dir / f"{capital}_combined.parquet", index=False)
        artifacts.write_json_checked(
            accounts_dir / f"{capital}_targets.json", {"metrics": metrics, "targets": targets}
        )
    pressure = {"status": "not_run_pending_main_table"}
    summary = report.summary_payload(
        identity={
            "candidate_id": spec.CANDIDATE_ID,
            "campaign_id": campaign.name,
            "run_id": run_dir.name,
            "protocol_sha256": spec.protocol_sha256(),
            "spec_sha256": spec.spec_sha256(),
            "code_commit": code_identity()["code_commit"],
            "dirty_patch_sha": code_identity()["dirty_patch_sha"],
            "baseline_sha": baseline.get("manifest_sha256"),
        },
        engineering_status="valid",
        data_status="ready",
        targets_by_capital=targets_by_capital,
        pressure_status=pressure,
        decision={
            "research_decision": "pending",
            "resource_action": "pending",
            "reason": "R3 未完成",
        },
        limitations=["历史证据等级 H；不是前向验证"],
        unmet_items=[],
        machine_hours={"r3": 0.0},
        requests=0,
        bytes_written=0,
    )
    report.write_report(reports_dir, summary, {"account": [f"corporate={corporate_audit}"]})
    published = artifacts.publish_artifacts(
        run_dir / "work", run_dir, "r3", STAGE_ARTIFACTS["r3"], {"r2": upstream}
    )
    _write_run_manifest(
        run_dir,
        {
            "stage": "r3",
            "run_id": run_dir.name,
            "campaign_id": campaign.name,
            "generated_at": artifacts.utc_now(),
            "upstream": {"r2": upstream},
            "artifacts": published,
        },
    )
    return {"targets": targets_by_capital}


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数。"""
    parser = argparse.ArgumentParser(description="LR5M-ML-10-v1 分钟量价修复 ML 研究包")
    parser.add_argument("--stage", choices=list(spec.STAGES), required=True)
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--artifact-base", default=None)
    parser.add_argument("--no-network", action="store_true", help="R0 只做本地核验")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="仅 R1：按已校验批次续跑同一未发布 run（已发布工件一律拒绝）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """入口。"""
    args = build_parser().parse_args(argv)
    log = _emit(args.stage)
    if args.resume:
        if args.stage != "r1":
            log("--resume 只支持 r1（其余阶段必须新 run）")
            return 2
        if not args.run_id:
            log("--resume 需要显式 --run-id")
            return 2
    campaign, run_dir = _resolve_campaign(args)
    started = time.perf_counter()
    try:
        if args.stage == "r0":
            result = run_r0(campaign, run_dir, log=log, use_network=not args.no_network)
            log(
                f"R0 结论：data_status={result['summary']['data_status']} "
                f"access={result['access']['access_status']}"
            )
        elif args.stage == "r1":
            r1_result = run_r1(
                campaign, run_dir, upstream=args.upstream, log=log, resume=args.resume
            )
            log(f"R1 结论：{r1_result}")
        elif args.stage == "r2":
            r2_result = run_r2(campaign, run_dir, upstream=args.upstream, log=log)
            log(f"R2 结论：{r2_result}")
        else:
            r3_result = run_r3(campaign, run_dir, upstream=args.upstream, log=log)
            log(f"R3 结论：{r3_result}")
    except Exception as error:  # noqa: BLE001 - 阶段失败必须落 state 并如实退出
        import traceback

        traceback.print_exc()
        artifacts.write_state(
            run_dir,
            {
                "stage": args.stage,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "finished_at": artifacts.utc_now(),
            },
        )
        log(f"FAILED: {type(error).__name__}: {error}")
        return 1
    elapsed = time.perf_counter() - started
    log(f"{args.stage} 完成，墙钟 {elapsed/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())

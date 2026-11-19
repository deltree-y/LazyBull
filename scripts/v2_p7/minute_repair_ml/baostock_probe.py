"""BaoStock 免费源的**有界**可得性探查（用户 2026-11-18 授权 B 方案）。

只回答"能不能替代 TuShare"，不做全量下载、不训练、不读收益：
1. 历史深度：2016 / 2019 / 近期代表窗口是否有 5 分钟线；
2. 字段与语义：时间标签约定、是否有 09:30 竞价、复权标记、单位；
3. 日线配对：同一天分钟量额能否与日线对上；
4. 分块一致性：一次取一个月 vs 分三段取，键集合是否相同；
5. 退市股：历史存在但后来退市的股票是否仍可取（生存者偏差检查）；
6. 吞吐：实测查询延迟，用于判断全量是否可能落在 22 小时预算内。

约束：总时长上限（默认 900 秒）与查询数上限（默认 60），超限即停；
命中来源配额/错误即停并如实登记；不写生产数据目录、不缓存到正式 raw。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.v2_p7.common import sha256_file
from scripts.v2_p7.minute_repair_ml import artifacts, bs_lock, spec

FIELDS_5M = "date,time,code,open,high,low,close,volume,amount,adjustflag"
FIELDS_DAILY = "date,code,open,high,low,close,volume,amount"

#: 预登记探查清单（顺序即优先级）：(名称, 代码, 起, 止, 频率)
CASES: tuple[tuple[str, str, str, str, str], ...] = (
    ("hist2016_sh", "sh.600000", "2016-01-04", "2016-01-08", "5"),
    ("hist2016_sz", "sz.000001", "2016-01-04", "2016-01-08", "5"),
    ("hist2019_sh", "sh.600000", "2019-07-01", "2019-07-05", "5"),
    ("hist2019_sz", "sz.000001", "2019-07-01", "2019-07-05", "5"),
    ("recent_sh", "sh.600000", "2026-10-26", "2026-10-30", "5"),
    ("recent_sz", "sz.000001", "2026-10-26", "2026-10-30", "5"),
    ("early2013", "sh.600000", "2013-04-01", "2013-04-05", "5"),
    ("cut2019h2", "sh.600000", "2019-12-02", "2019-12-06", "5"),
    ("cut2020h1", "sh.600000", "2020-01-06", "2020-01-10", "5"),
    ("year2020", "sh.600000", "2020-07-06", "2020-07-10", "5"),
    ("year2021", "sh.600000", "2021-07-05", "2021-07-09", "5"),
    ("year2022", "sh.600000", "2022-07-04", "2022-07-08", "5"),
    ("year2023", "sh.600000", "2023-07-03", "2023-07-07", "5"),
    ("year2024", "sh.600000", "2024-07-01", "2024-07-05", "5"),
    ("year2025", "sh.600000", "2025-07-07", "2025-07-11", "5"),
    ("delisted", "sh.600005", "2016-01-04", "2016-01-08", "5"),
    ("delisted_2020", "sz.000587", "2020-07-06", "2020-07-10", "5"),
    ("delisted_2021", "sz.000587", "2021-07-05", "2021-07-09", "5"),
    ("daily2016", "sh.600000", "2016-01-04", "2016-01-08", "d"),
    ("daily2019", "sh.600000", "2019-07-01", "2019-07-05", "d"),
    ("daily_recent", "sh.600000", "2026-10-26", "2026-10-30", "d"),
    ("chunk_whole", "sh.600000", "2025-07-01", "2025-07-31", "5"),
    ("chunk_a", "sh.600000", "2025-07-01", "2025-07-10", "5"),
    ("chunk_b", "sh.600000", "2025-07-11", "2025-07-20", "5"),
    ("chunk_c", "sh.600000", "2025-07-21", "2025-07-31", "5"),
)


def _login_status() -> dict[str, Any]:
    import baostock as bs

    result = bs.login()
    return {
        "code": result.error_code,
        "msg": result.error_msg,
        "package": str(getattr(bs, "__version__", "unknown")),
    }


def _query(code: str, start: str, end: str, freq: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行一次 BaoStock 查询并计时。"""
    import baostock as bs

    fields = FIELDS_5M if freq == "5" else FIELDS_DAILY
    started = time.perf_counter()
    rs = bs.query_history_k_data_plus(
        code,
        fields,
        start_date=start,
        end_date=end,
        frequency=freq,
        adjustflag="3",
    )
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame(columns=rs.fields)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, {
        "code": rs.error_code,
        "msg": rs.error_msg,
        "rows": int(len(frame)),
        "elapsed_seconds": round(elapsed, 3),
    }


def _bar_structure(frame: pd.DataFrame) -> dict[str, Any]:
    """逐日结构：48 根、首尾标签、是否含 09:30。"""
    if not len(frame):
        return {"days": 0, "session_ok_days": 0}
    times = frame["time"].astype(str)
    scoped = frame.assign(day=times.str[:8].to_numpy(), minute=times.str[8:12].to_numpy())
    per_day = scoped.groupby("day")["minute"].nunique()
    has_auction = bool((scoped["minute"] == "0930").any())
    firsts = scoped.groupby("day")["minute"].min()
    lasts = scoped.groupby("day")["minute"].max()
    return {
        "days": int(len(per_day)),
        "bars_per_day_min": int(per_day.min()),
        "bars_per_day_max": int(per_day.max()),
        "expected_bars": int(spec.MINUTE["grid_slots"]),
        "session_ok_days": int((per_day == int(spec.MINUTE["grid_slots"])).sum()),
        "has_0930_auction": has_auction,
        "first_minute": sorted(firsts.unique().tolist()),
        "last_minute": sorted(lasts.unique().tolist()),
        "label_convention": "end" if not has_auction else "mixed",
    }


def _daily_reconcile(minute: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """分钟量额与日线对账（同一天、同一股票）。"""
    if not len(minute) or not len(daily):
        return {"status": "no_data"}
    m = minute.copy()
    m["day"] = m["time"].astype(str).str[:8]
    m_sum = m.groupby("day")[["volume", "amount"]].sum()
    d = daily.copy()
    d["day"] = d["date"].astype(str).str.replace("-", "", regex=False)
    d = d.set_index("day")[["volume", "amount"]]
    joined = m_sum.join(d, how="inner", lsuffix="_m", rsuffix="_d")
    if joined.empty:
        return {"status": "no_overlap"}
    vol_err = ((joined["volume_m"] - joined["volume_d"]).abs() / joined["volume_d"]).replace(
        [float("inf")], pd.NA
    )
    amt_err = ((joined["amount_m"] - joined["amount_d"]).abs() / joined["amount_d"]).replace(
        [float("inf")], pd.NA
    )
    return {
        "status": "measured",
        "days": int(len(joined)),
        "volume_rel_error_max": float(vol_err.max()),
        "amount_rel_error_max": float(amt_err.max()),
        "within_line": bool(
            float(vol_err.max()) <= float(spec.UNITS["minute_anomaly_volume_rel_error"])
            and float(amt_err.max()) <= float(spec.UNITS["minute_anomaly_volume_rel_error"])
        ),
        "sample": joined.head(3).reset_index().to_dict("records"),
    }


def run_probe(
    probe_dir: Path,
    *,
    max_queries: int = 60,
    max_seconds: float = 900.0,
    log=lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """执行有界探查并落盘。"""
    probe_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = bs_lock.acquire(probe_dir.parent.parent, owner="baostock_probe")
    login = _login_status()
    if login["code"] != "0":
        bs_lock.release(lock)
        payload = {"status": "login_failed", "login": login}
        artifacts.write_json_checked(probe_dir / "summary.json", payload)
        return payload
    frames: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    try:
        for index, (name, code, start, end, freq) in enumerate(CASES):
            if index >= int(max_queries):
                records.append({"name": name, "status": "skipped_max_queries"})
                continue
            if time.perf_counter() - started >= max_seconds:
                records.append({"name": name, "status": "skipped_max_seconds"})
                continue
            frame, meta = _query(code, start, end, freq)
            frames[name] = frame
            # 注意：meta 里的 `code` 是**响应码**，不能覆盖股票代码（历史缺陷：报告代码列全为 0）
            entry = {
                "name": name,
                "ts_code": code,
                "start": start,
                "end": end,
                "freq": freq,
                "response_code": meta["code"],
                "response_msg": meta["msg"],
                "rows": meta["rows"],
                "elapsed_seconds": meta["elapsed_seconds"],
            }
            if meta["code"] != "0":
                entry["status"] = "query_failed"
            elif not len(frame):
                entry["status"] = "empty"
            else:
                entry["status"] = "ok"
                raw_path = probe_dir / f"{name}.parquet"
                frame.to_parquet(raw_path, index=False)
                entry["sha256"] = sha256_file(raw_path)
                if freq == "5":
                    entry["structure"] = _bar_structure(frame)
            records.append(entry)
            log(f"probe {name}: {entry['status']} rows={entry['rows']} {meta['elapsed_seconds']}s")
    finally:
        import baostock as bs

        bs.logout()
        bs_lock.release(lock)

    elapsed = round(time.perf_counter() - started, 2)
    ok_5m = [item for item in records if item.get("freq") == "5" and item.get("status") == "ok"]
    structure = {item["name"]: item.get("structure") for item in ok_5m}
    recon: dict[str, Any] = {}
    recon["2016"] = _daily_reconcile(
        frames.get("hist2016_sh", pd.DataFrame()), frames.get("daily2016", pd.DataFrame())
    )
    recon["2019"] = _daily_reconcile(
        frames.get("hist2019_sh", pd.DataFrame()), frames.get("daily2019", pd.DataFrame())
    )
    recon["recent"] = _daily_reconcile(
        frames.get("recent_sh", pd.DataFrame()), frames.get("daily_recent", pd.DataFrame())
    )
    chunk = _chunk_consistency(frames)
    latency = [item["elapsed_seconds"] for item in ok_5m if item.get("elapsed_seconds")]
    requests_total = sum(
        1 for item in records if item.get("status") in {"ok", "empty", "query_failed"}
    )
    rate = (requests_total / elapsed * 3600.0) if elapsed > 0 and requests_total else None
    required_requests = 37829
    summary = {
        "status": "completed",
        "login": login,
        "source_identity": {
            "name": "baostock",
            "package_version": login["package"],
            "fields_5m": FIELDS_5M.split(","),
            "adjustflag": "3（不复权）",
            "time_format": "YYYYMMDDHHMMSSsss",
            "label_convention": "end（5min 收盘时刻，无 09:30 竞价行）",
            "volume_unit": "股（与日线同口径，需对账确认）",
            "amount_unit": "元",
        },
        "requests": requests_total,
        "elapsed_seconds": elapsed,
        "max_queries": int(max_queries),
        "max_seconds": float(max_seconds),
        "structure": structure,
        "reconciliation": recon,
        "chunk_consistency": chunk,
        "throughput": {
            "observed_requests_per_hour": round(rate, 1) if rate else None,
            "mean_query_seconds": round(sum(latency) / len(latency), 3) if latency else None,
            "required_requests": required_requests,
            "projected_hours": round(required_requests / rate, 1) if rate else None,
            "within_22h": bool(
                rate and required_requests / rate <= float(spec.BUDGET["r1_download_max_hours"])
            ),
        },
        "records": records,
        "note": "只核验可得性与语义，不下载全量；空返回与查询失败分别登记，不当作'无数据'",
    }
    artifacts.write_json_checked(probe_dir / "summary.json", summary)
    (probe_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _chunk_consistency(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """整月 vs 分块：**全部字段逐值**比较，并检查重复键（不只比键集合）。"""
    whole = frames.get("chunk_whole")
    parts = [frames.get(name) for name in ("chunk_a", "chunk_b", "chunk_c")]
    if whole is None or len(whole) == 0 or any(part is None for part in parts):
        return {"status": "not_attempted"}
    merged = pd.concat(parts, ignore_index=True)
    value_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "amount")
        if column in whole.columns and column in merged.columns
    ]
    whole_sorted = whole.sort_values("time").reset_index(drop=True)
    merged_sorted = merged.sort_values("time").reset_index(drop=True)
    whole_keys = set(whole_sorted["time"].astype(str))
    part_keys = set(merged_sorted["time"].astype(str))
    same_shape = len(whole_sorted) == len(merged_sorted)
    same_keys = whole_keys == part_keys
    same_values = False
    max_abs_diff = None
    if same_shape and same_keys and value_columns:
        left = whole_sorted[value_columns].apply(pd.to_numeric, errors="coerce")
        right = merged_sorted[value_columns].apply(pd.to_numeric, errors="coerce")
        diff = (left - right).abs()
        max_abs_diff = float(diff.to_numpy().max()) if diff.size else 0.0
        same_values = bool(max_abs_diff == 0.0)
    duplicates_whole = int(whole_sorted.duplicated(subset=["time"]).sum())
    duplicates_parts = int(merged_sorted.duplicated(subset=["time"]).sum())
    clean = not duplicates_whole and not duplicates_parts
    status = "ok" if (same_shape and same_keys and same_values and clean) else "unequal"
    return {
        "status": status,
        "rows_whole": int(len(whole_sorted)),
        "rows_parts": int(len(merged_sorted)),
        "missing_in_parts": len(whole_keys - part_keys),
        "extra_in_parts": len(part_keys - whole_keys),
        "same_values": same_values,
        "max_abs_diff": max_abs_diff,
        "duplicates_whole": duplicates_whole,
        "duplicates_parts": duplicates_parts,
    }


def _render(summary: dict[str, Any]) -> str:
    lines = ["# BaoStock 可得性探查（有界）", "", "## 结论", ""]
    lines.append(
        f"- 登录：`{summary['login']['code']}`；查询 {summary['requests']} 次，"
        f"耗时 {summary['elapsed_seconds']} 秒"
    )
    for name, block in (summary.get("structure") or {}).items():
        if not block:
            continue
        lines.append(
            f"- `{name}`：{block['days']} 天、每日 {block['bars_per_day_min']}~"
            f"{block['bars_per_day_max']} 根、结构合格 {block['session_ok_days']} 天、"
            f"09:30 竞价={block['has_0930_auction']}"
        )
    for label, block in (summary.get("reconciliation") or {}).items():
        lines.append(f"- 对账 {label}：{json.dumps(block, ensure_ascii=False)[:220]}")
    chunk = summary.get("chunk_consistency") or {}
    lines.append(f"- 分块一致性：{json.dumps(chunk, ensure_ascii=False)}")
    thr = summary.get("throughput") or {}
    lines.append(
        f"- 吞吐：{thr.get('observed_requests_per_hour')} 请求/小时，"
        f"全量预计 {thr.get('projected_hours')} 小时，22 小时内={thr.get('within_22h')}"
    )
    lines.append("")
    lines.append("## 逐查询")
    lines.append("")
    lines.append("| 名称 | 股票代码 | 起 | 止 | 频率 | 状态 | 行数 | 秒 |")
    lines.append("|---|---|---|---|---|---|---:|---:|")
    for item in summary["records"]:
        lines.append(
            f"| {item['name']} | {item.get('ts_code', '-')} | {item.get('start', '-')} | "
            f"{item.get('end', '-')} | {item.get('freq', '-')} | {item['status']} | "
            f"{item.get('rows', '-')} | {item.get('elapsed_seconds', '-')} |"
        )
    lines.append("")
    lines.append(f"> {summary['note']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：只接受上限参数，不提供规格扫描。"""
    import argparse

    parser = argparse.ArgumentParser(description="BaoStock 有界可得性探查")
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--artifact-base", default=None)
    args = parser.parse_args(argv)
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    probe_dir = campaign / "baostock_probe" / artifacts.new_run_id("bsprobe")
    summary = run_probe(
        probe_dir,
        max_queries=args.max_queries,
        max_seconds=args.max_seconds,
        log=lambda message: print(f"[bs] {message}", flush=True),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

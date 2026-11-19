"""零采购费的有界可得性探查（用户 2026-11-18 授权的 A 方案：≤8 请求）。

只回答三个问题，不做全量下载、不做经济推断：
1. 现有授权**能拿到多早**的 5min / 1min（历史深度）；
2. 拿到的单日结构是否与协议一致（48 槽 + 09:30 竞价、1min 241 行）；
3. 配额窗口是否足以支撑 37,829 请求（只记录观测到的上限，不推算）。

规则：
- 每个请求之间间隔 65 秒（若配额是 1 次/分钟仍能连续取；若是 1 次/小时则会中断）；
- **一旦命中已知配额立即停止**，不重试、不等待整窗；
- 每个成功响应按字节存证（parquet + SHA），失败只记状态；
- 全程不打印 token；不写生产数据目录。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.v2_p7.common import sha256_file
from scripts.v2_p7.minute_repair_ml import artifacts, source, spec

#: 预登记探查清单（顺序即优先级：最决定性的先跑）
CASES: tuple[tuple[str, str, str], ...] = (
    ("600000.SH", "20160105", "5min"),  # 5min 历史下限
    ("600000.SH", "20261029", "5min"),  # 5min 近期（结构核验）
    ("000001.SZ", "20190701", "5min"),  # 中期 + 深市
    ("600000.SH", "20160105", "1min"),  # 1min 历史下限（配对对账前提）
    ("600000.SH", "20261029", "1min"),  # 1min 近期
    ("600519.SH", "20160105", "5min"),  # 第二只股票的历史
    ("600000.SH", "20130401", "5min"),  # 早于协议起点，判断真实起点
    ("000001.SZ", "20261029", "5min"),  # 深市近期
)


def _window(day: str) -> tuple[str, str]:
    return (
        f"{day[:4]}-{day[4:6]}-{day[6:]} 09:00:00",
        f"{day[:4]}-{day[4:6]}-{day[6:]} 15:31:00",
    )


def _structure(frame: pd.DataFrame, freq: str) -> dict[str, Any]:
    """单日结构核验：行数、首尾时间、竞价、常规槽数。"""
    times = sorted(frame["trade_time"].astype(str).tolist())
    has_auction = any(item.endswith(spec.MINUTE["auction_time"]) for item in times)
    core = [item for item in times if not item.endswith(spec.MINUTE["auction_time"])]
    expected_core = int(spec.MINUTE["grid_slots"]) if freq == "5min" else 240
    return {
        "rows": int(len(frame)),
        "first_time": times[0] if times else None,
        "last_time": times[-1] if times else None,
        "has_0930_auction": has_auction,
        "core_rows": len(core),
        "expected_core_rows": expected_core,
        "session_ok": len(core) == expected_core,
    }


def run_probe(
    campaign: Path,
    probe_dir: Path,
    *,
    max_requests: int = 8,
    interval_seconds: float = 65.0,
    log=print,
) -> dict[str, Any]:
    """执行有界探查并落盘（命中配额即停）。"""
    probe_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        src = source.MinuteSource(max_retries=0, log=lambda _m: None)
    except source.SourceError as error:
        payload = {"status": "token_missing", "error": str(error)[:200]}
        artifacts.write_json_checked(probe_dir / "summary.json", payload)
        return payload
    records: list[dict[str, Any]] = []
    blocked: dict[str, Any] | None = None
    for index, (code, day, freq) in enumerate(CASES[: max(0, int(max_requests))]):
        if blocked is not None:
            records.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "freq": freq,
                    "status": "skipped_after_block",
                }
            )
            continue
        if index:
            time.sleep(interval_seconds)
        start, end = _window(day)
        try:
            result = src.fetch(code, start, end, freq)
        except source.RateLimitBlocked as error:
            blocked = error.as_record()
            records.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "freq": freq,
                    "status": source.RATE_LIMIT,
                    "blocked": blocked,
                }
            )
            log(
                f"probe {code} {day} {freq}: rate_limit，停止（下次 {blocked['next_allowed_at_utc']}）"
            )
            continue
        entry: dict[str, Any] = {
            "ts_code": code,
            "trade_date": day,
            "freq": freq,
            "status": result.status,
            "requests": int(result.requests),
            "splits": int(result.splits),
        }
        if result.status == source.OK and len(result.frame):
            entry.update(_structure(result.frame, freq))
            raw_path = probe_dir / f"{code}_{day}_{freq}.parquet"
            result.frame.to_parquet(raw_path, index=False)
            entry["sha256"] = sha256_file(raw_path)
            entry["vol_sum"] = float(pd.to_numeric(result.frame["vol"], errors="coerce").sum())
            entry["amount_sum"] = float(
                pd.to_numeric(result.frame["amount"], errors="coerce").sum()
            )
        elif result.status != source.OK:
            entry["error"] = result.error
        records.append(entry)
        log(f"probe {code} {day} {freq}: {result.status} rows={entry.get('rows')}")

    elapsed = round(time.perf_counter() - started, 2)
    by_freq: dict[str, dict[str, Any]] = {}
    for freq in ("5min", "1min"):
        ok = [
            item
            for item in records
            if item["freq"] == freq and item.get("status") == source.OK and item.get("rows")
        ]
        by_freq[freq] = {
            # 只统计真正发出的请求；skipped_after_block 不计入"尝试"
            "attempted": sum(
                1
                for item in records
                if item["freq"] == freq and item.get("status") != "skipped_after_block"
            ),
            "skipped": sum(
                1
                for item in records
                if item["freq"] == freq and item.get("status") == "skipped_after_block"
            ),
            "ok_samples": len(ok),
            "earliest_date": min((item["trade_date"] for item in ok), default=None),
            "latest_date": max((item["trade_date"] for item in ok), default=None),
            "session_ok_samples": sum(1 for item in ok if item.get("session_ok")),
        }
    summary = {
        "status": "blocked" if blocked else "completed",
        "max_requests": int(max_requests),
        "requests_sent": int(src.requests),
        "interval_seconds": float(interval_seconds),
        "elapsed_seconds": elapsed,
        "blocked": blocked,
        "quota_observed": (blocked or {}).get("limit"),
        "by_freq": by_freq,
        "history_5m_reaches_protocol_start": bool(
            by_freq["5min"]["earliest_date"] and by_freq["5min"]["earliest_date"] <= "20160105"
        ),
        "history_1m_reaches_protocol_start": bool(
            by_freq["1min"]["earliest_date"] and by_freq["1min"]["earliest_date"] <= "20160105"
        ),
        "throughput": {
            "required_requests": 37829,
            "observation": (
                "命中配额后停止，未做持续采样；配额见 quota_observed"
                if blocked
                else "样本过少，不足以估计吞吐"
            ),
            "measured_success_per_hour": None,
            "status": "not_measured",
        },
        "records": records,
        "note": (
            "只核验可得性与单日结构，不下载全量、不读收益；"
            "样本失败或缺失一律按状态登记，不当作'无数据'"
        ),
    }
    artifacts.write_json_checked(probe_dir / "summary.json", summary)
    (probe_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(summary: dict[str, Any]) -> str:
    lines = ["# 分钟数据可得性探查（有界）", "", "## 结论", ""]
    lines.append(f"- 状态：`{summary['status']}`；发送请求 {summary['requests_sent']} 次")
    if summary.get("quota_observed"):
        lines.append(
            f"- 观测配额：`{json.dumps(summary['quota_observed'], ensure_ascii=False)}`；"
            f"下次可试 `{(summary.get('blocked') or {}).get('next_allowed_at_utc')}`"
        )
    for freq, block in summary["by_freq"].items():
        lines.append(
            f"- {freq}：成功样本 {block['ok_samples']}/{block['attempted']}，"
            f"日期范围 {block['earliest_date']}..{block['latest_date']}，"
            f"结构合格 {block['session_ok_samples']} 个"
        )
    lines.append("")
    lines.append("## 逐请求")
    lines.append("")
    lines.append("| 股票 | 日期 | 频率 | 状态 | 行数 | 常规槽 | 竞价 |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for item in summary["records"]:
        lines.append(
            f"| {item['ts_code']} | {item['trade_date']} | {item['freq']} | {item['status']} | "
            f"{item.get('rows', '-')} | {item.get('core_rows', '-')} | "
            f"{item.get('has_0930_auction', '-')} |"
        )
    lines.append("")
    lines.append(f"> {summary['note']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：只接受 campaign 与请求上限，不提供任何规格扫描参数。"""
    parser = argparse.ArgumentParser(description="分钟数据可得性有界探查")
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--interval-seconds", type=float, default=65.0)
    parser.add_argument("--artifact-base", default=None)
    args = parser.parse_args(argv)
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    probe_dir = campaign / "access_probe" / artifacts.new_run_id("probe")
    summary = run_probe(
        campaign,
        probe_dir,
        max_requests=args.max_requests,
        interval_seconds=args.interval_seconds,
        log=lambda message: print(f"[probe] {message}", flush=True),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False))
    return 0 if summary.get("status") == "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

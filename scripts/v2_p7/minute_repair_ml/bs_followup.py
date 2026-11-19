"""BaoStock 覆盖验收的**补样**：退市股样本 + 真实 120 交易日块吞吐（≤15 查询）。

第一轮（`bs_coverage`，285 查询）已确认 2020/2023/2026 三个锚点无缺失、无未解释对账；
本补样只补两件第一轮没覆盖的事：
1. 5 只"历史在市、后来退市"的股票在 2020-01 与 2021-06 是否可取；
2. 5 个 120 交易日块的实测吞吐（第一轮在 logout 后执行，未取得有效读数）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.common import sha256_file
from scripts.v2_p7.minute_repair_ml import (
    artifacts,
    bs_coverage as cov,
    bs_lock,
    domain,
    panel,
    spec,
)

DELISTED_SAMPLE = 5
DELISTED_ANCHORS: tuple[str, ...] = ("202001", "202106")
BLOCK_CODES = 5


def _last_dates(inputs: dict[str, Any]) -> dict[str, str]:
    """逐股最后一个有效收盘日（用于识别"已退市/长期停牌"）。"""
    close = inputs["prices"]["close_adj"]
    last: dict[str, str] = {}
    for code in close.columns:
        values = close[code].notna().to_numpy()
        indexes = np.flatnonzero(values)
        if indexes.size:
            last[str(code)] = str(close.index[indexes[-1]])
    return last


def _pick_delisted(
    inputs: dict[str, Any], result: Any, days: Sequence[str], *, size: int
) -> pd.DataFrame:
    """选取"历史合格、后来不再交易"的股票，兼顾沪深与合格天数。"""
    last = _last_dates(inputs)
    position = {day: index for index, day in enumerate(inputs["dates"])}
    positions = [position[day] for day in days]
    rows = []
    for index, code in enumerate(inputs["codes"]):
        end = last.get(str(code))
        if end is None or end >= "20250630":
            continue
        count = int(result.eligible[positions, index].sum())
        if count < 200:
            continue
        rows.append(
            {
                "ts_code": str(code),
                "market": str(code).split(".")[-1].upper(),
                "last_trade_date": end,
                "eligible_days": count,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    picked: list[int] = []
    for market in ("SH", "SZ"):
        scoped = table[table["market"] == market].sort_values(
            ["eligible_days", "ts_code"], ascending=[False, True]
        )
        picked.extend(scoped.index[: max(1, size // 2)].tolist())
    return table.loc[sorted(set(picked))[:size]].reset_index(drop=True)


def run_followup(
    probe_dir: Path,
    *,
    max_queries: int = 15,
    max_seconds: float = 1800.0,
    log=lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """补样执行：退市股覆盖 + 吞吐块。"""
    import baostock as bs

    probe_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calendar = panel.load_calendar(cov.REPO_ROOT)
    inputs = cov.dataset_inputs()
    days = [d for d in inputs["dates"] if cov.WINDOW_START <= str(d) <= cov.WINDOW_END]
    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    delisted = _pick_delisted(inputs, result, days, size=DELISTED_SAMPLE)
    plan = [
        {
            "ts_code": row["ts_code"],
            "bs_code": cov._bs_code(row["ts_code"]),
            "anchor": anchor,
            "start": cov._month_bounds(anchor)[0],
            "end": cov._month_bounds(anchor)[1],
            "last_trade_date": row["last_trade_date"],
        }
        for row in delisted.to_dict("records")
        for anchor in DELISTED_ANCHORS
    ]
    artifacts.write_json_checked(
        probe_dir / "plan.json",
        {
            "delisted_sample": delisted.to_dict("records"),
            "anchors": list(DELISTED_ANCHORS),
            "queries": len(plan) + BLOCK_CODES,
            "max_queries": int(max_queries),
            "note": "只补第一轮缺的退市样本与吞吐块",
        },
    )
    log(f"followup plan: {len(plan)} 退市查询 + {BLOCK_CODES} 吞吐块")

    lock = bs_lock.acquire(probe_dir.parent.parent, owner="bs_followup")
    login = bs.login()
    if login.error_code != "0":
        bs_lock.release(lock)
        payload = {"status": "login_failed"}
        artifacts.write_json_checked(probe_dir / "summary.json", payload)
        return payload
    local = cov._local_state(delisted["ts_code"].tolist(), days)
    records: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(plan):
            if index >= int(max_queries) or time.perf_counter() - started >= max_seconds:
                records.append({**item, "status": "skipped_limit"})
                continue
            frame, meta = cov._query(item["bs_code"], item["start"], item["end"])
            entry = {**item, "rows": meta["rows"], "elapsed_seconds": meta["elapsed_seconds"]}
            if meta["response_code"] != "0":
                entry["status"] = "query_failed"
            elif not len(frame):
                entry["status"] = "empty"
                entry["expected_in_month"] = len(
                    [
                        day
                        for day in days
                        if day[:6] == item["anchor"]
                        and (item["ts_code"], day) in local
                        and int(local[(item["ts_code"], day)].get("tradable", 1) or 0) == 1
                        and int(local[(item["ts_code"], day)].get("is_suspended", 0) or 0) == 0
                    ]
                )
            else:
                raw_path = probe_dir / f"{item['ts_code']}_{item['anchor']}.parquet"
                frame.to_parquet(raw_path, index=False)
                entry["sha256"] = sha256_file(raw_path)
                expected = [
                    day
                    for day in days
                    if day[:6] == item["anchor"]
                    and (item["ts_code"], day) in local
                    and int(local[(item["ts_code"], day)].get("tradable", 1) or 0) == 1
                    and int(local[(item["ts_code"], day)].get("is_suspended", 0) or 0) == 0
                ]
                entry.update(
                    cov._analyze_month(frame, item["ts_code"], item["anchor"], expected, local)
                )
            records.append(entry)
    finally:
        bs.logout()
        bs_lock.release(lock)

    block_codes = delisted["ts_code"].tolist()[:BLOCK_CODES]
    if not block_codes:
        block_codes = [str(code) for code in inputs["codes"][:BLOCK_CODES]]
    throughput = cov.throughput_blocks(
        block_codes,
        days=120,
        end="2025-12-31",
        log=log,
        campaign=probe_dir.parent.parent,
        guard=False,
    )
    summary = {
        "status": "completed",
        "mode": "followup",
        "queries_sent": len(records) + int(throughput.get("ok_blocks", 0)),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "delisted_sample": delisted.to_dict("records"),
        "records": records,
        "throughput": throughput,
        "verdict": _verdict(records, throughput),
        "note": "退市样本只核验「当时在市」的覆盖；不据此推断全市场早年完整性",
    }
    artifacts.write_json_checked(probe_dir / "summary.json", summary)
    (probe_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _verdict(records: Sequence[dict[str, Any]], throughput: dict[str, Any]) -> dict[str, Any]:
    ok = [item for item in records if item.get("status") == "ok"]
    missing = int(sum(len(item.get("missing_days", [])) for item in ok))
    partial = int(sum(int(item.get("partial_days", 0)) for item in ok))
    unexplained = int(sum(int(item.get("unexplained", 0)) for item in ok))
    empty = sum(1 for item in records if item.get("status") == "empty")
    return {
        "delisted_with_data": len(ok),
        "delisted_empty": int(empty),
        "missing_days": missing,
        "partial_days": partial,
        "unexplained_recon": unexplained,
        "throughput": throughput.get("status"),
        "verdict": (
            "usable"
            if ok
            and missing == 0
            and partial == 0
            and unexplained == 0
            and throughput.get("status") == "measured"
            else "not_proven"
        ),
    }


def _render(summary: dict[str, Any]) -> str:
    verdict = summary["verdict"]
    lines = ["# BaoStock 覆盖验收补样（退市股 + 吞吐块）", "", "## 结论", ""]
    lines.append(
        f"- 退市样本有数据 {verdict['delisted_with_data']} / 空 {verdict['delisted_empty']}；"
        f"缺失 {verdict['missing_days']} 天、不完整 {verdict['partial_days']} 天、"
        f"未解释对账 {verdict['unexplained_recon']}"
    )
    thr = summary["throughput"]
    lines.append(
        f"- 吞吐：{thr.get('ok_blocks')}/{thr.get('blocks')} 块成功，"
        f"{thr.get('observed_queries_per_hour')} 查询/小时，"
        f"{thr.get('rows_per_query')} 行/块"
    )
    lines.append(f"- 判定：`{verdict['verdict']}`")
    lines.append("")
    lines.append("## 逐查询")
    lines.append("")
    lines.append("| 股票 | 锚点 | 状态 | 应有日 | 实取日 | 完整日 | 缺失 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for item in summary["records"]:
        missing = item.get("missing_days")
        missing_text = "-" if missing is None else str(len(missing))
        lines.append(
            f"| {item['ts_code']} | {item.get('anchor', '-')} | {item['status']} | "
            f"{item.get('expected_days', '-')} | {item.get('present_days', '-')} | "
            f"{item.get('complete_days', '-')} | {missing_text} |"
        )
    lines.append("")
    lines.append(f"> {summary['note']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI。"""
    import argparse

    parser = argparse.ArgumentParser(description="BaoStock 覆盖验收补样")
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--max-queries", type=int, default=15)
    parser.add_argument("--artifact-base", default=None)
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="只测 120 交易日块吞吐（用当前在市的大盘股，避免退市样本空返回）",
    )
    args = parser.parse_args(argv)
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    if args.throughput_only:
        probe_dir = campaign / "bs_followup" / artifacts.new_run_id("bsthr")
        probe_dir.mkdir(parents=True, exist_ok=True)
        codes = ["600000.SH", "000001.SZ", "600519.SH", "000002.SZ", "601318.SH"]
        result = cov.throughput_blocks(
            codes, days=120, end="2025-12-31", log=print, campaign=campaign
        )
        payload = {"mode": "throughput_only", "codes": codes, **result}
        artifacts.write_json_checked(probe_dir / "summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    probe_dir = campaign / "bs_followup" / artifacts.new_run_id("bsfollow")
    summary = run_followup(probe_dir, max_queries=args.max_queries)
    print(json.dumps(summary.get("verdict", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

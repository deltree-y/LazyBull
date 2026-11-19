"""BaoStock 2020 起**目标域覆盖验收**（用户 2026-11-18 授权，≤300 查询 / ≤2 机器小时）。

只回答一个问题：**历史合格域内的股票，2020-01 起能否取全**。不做全量、不训练、不读收益。

方法（预登记，先写计划再取数）：
1. 用本地 `clean/daily` + `daily_basic` + PIT 行业重建 2020-01-01—2026-10-29 的**历史合格域**；
2. 按市场 × 行业 × 流动性分位分层抽取 95 只（含 15 只"历史在市、现已退市"），
   每只取 3 个锚点月：2020-01、2023-06、2026-09；
3. 逐查询核对：应有交易日（本地日历 ∩ 该股在市且未停牌）vs 实取天数、
   每日 48 槽完整性、量额与本地日线对账（0.5% 调查线 + 机械处置）；
4. 另用 10 个 120 交易日块测真实吞吐，并据此重算 2020 起全量的请求数与耗时；
5. 任何限频/失败即停并如实登记，不改判据、不缩窗。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.minute_repair_ml import (
    artifacts,
    bs_lock,
    domain,
    download,
    panel,
    quality,
    spec,
)

ANCHORS: tuple[str, ...] = ("202001", "202306", "202609")
SAMPLE_SIZE = 95
DELISTED_QUOTA = 15
THROUGHPUT_BLOCKS = 10
THROUGHPUT_BLOCK_DAYS = 120
WINDOW_START = "20200101"
WINDOW_END = "20261029"
FIELDS_5M = "date,time,code,open,high,low,close,volume,amount,adjustflag"


def _bs_code(ts_code: str) -> str:
    code, market = str(ts_code).split(".")
    return f"{'sh' if market.upper() == 'SH' else 'sz'}.{code}"


def _month_bounds(anchor: str) -> tuple[str, str]:
    year, month = int(anchor[:4]), int(anchor[4:])
    start = f"{year}-{month:02d}-01"
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    return start, end


def _query(code: str, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import baostock as bs

    started = time.perf_counter()
    rs = bs.query_history_k_data_plus(
        code, FIELDS_5M, start_date=start, end_date=end, frequency="5", adjustflag="3"
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
        "response_code": rs.error_code,
        "response_msg": rs.error_msg,
        "rows": int(len(frame)),
        "elapsed_seconds": round(elapsed, 3),
    }


def _local_state(
    codes: Sequence[str], days: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """本地日线状态：在市/停牌 + 量额高低（用于"应有交易日"与对账）。"""
    wanted = {str(code) for code in codes}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for day in sorted(set(days)):
        path = REPO_ROOT / "data/clean/daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["ts_code", "vol", "amount", "high", "low", "is_suspended", "tradable"]
        )
        frame = frame[frame["ts_code"].astype(str).isin(wanted)]
        for row in frame.to_dict("records"):
            out[(str(row["ts_code"]), str(day))] = row
    return out


def _stock_table(inputs: dict[str, Any], result: Any, days: Sequence[str]) -> pd.DataFrame:
    """逐股属性：市场、行业、流动性、合格日数、是否已退市。"""
    codes = list(inputs["codes"])
    amount = inputs["prices"]["amount"]
    close = inputs["prices"]["close_adj"]
    position = {day: index for index, day in enumerate(inputs["dates"])}
    positions = [position[day] for day in days]
    rows = []
    basic = inputs["stock_basic"]
    industry = dict(zip(basic["ts_code"].astype(str), basic.get("industry", pd.Series())))
    # 退市判定不能看 stock_basic 是否含该代码（本地表含历史退市），要看"最后交易日"
    for index, code in enumerate(codes):
        column = result.eligible[positions, index]
        count = int(column.sum())
        if count == 0:
            continue
        values = amount[code].iloc[positions] if code in amount.columns else None
        liquidity = (
            float(np.nanmean(values.to_numpy(dtype=float))) if values is not None else np.nan
        )
        last_trade = None
        if code in close.columns:
            valid_days = close[code].notna().to_numpy()
            indexes = np.flatnonzero(valid_days)
            if indexes.size:
                last_trade = str(close.index[indexes[-1]])
        rows.append(
            {
                "ts_code": str(code),
                "market": str(code).split(".")[-1].upper(),
                "industry": str(industry.get(str(code), "unknown")),
                "eligible_days": count,
                "liquidity": liquidity,
                "last_trade_date": last_trade,
                "delisted": bool(last_trade and last_trade < "20250630"),
            }
        )
    table = pd.DataFrame(rows)
    valid = table["liquidity"].notna()
    table["liquidity_tercile"] = "na"
    if valid.any():
        table.loc[valid, "liquidity_tercile"] = pd.qcut(
            table.loc[valid, "liquidity"], 3, labels=["low", "mid", "high"]
        ).astype(str)
    return table


def _sample(table: pd.DataFrame, size: int, delisted_quota: int) -> pd.DataFrame:
    """分层＋确定序抽样：先保退市配额，再按 (市场 → 行业 → 流动性) 轮转补齐。

    历史缺陷：直接按 `(market, industry, tercile)` 排序轮转，会让"市场"先被填满
    （SH 排在 SZ 前），Step 1 的 40 只因此全是沪市。现在**先按市场分轮**，
    每轮每个市场各取一只，保证沪深都有。
    """
    ordered = table.sort_values(["ts_code"]).reset_index(drop=True)
    picked: list[int] = []
    delisted = ordered[ordered["delisted"]].index.tolist()
    picked.extend(delisted[:delisted_quota])
    remaining = ordered.drop(index=picked)
    if "liquidity_tercile" not in remaining.columns:
        remaining = remaining.assign(liquidity_tercile="na")
    by_market: dict[str, dict[tuple, list[int]]] = {}
    for key, group in remaining.groupby(["market", "industry", "liquidity_tercile"]):
        by_market.setdefault(str(key[0]), {})[key] = group.sort_values("ts_code").index.tolist()
    cursors = {market: {key: 0 for key in groups} for market, groups in by_market.items()}
    markets = sorted(by_market)
    while len(picked) < size:
        progressed = False
        for market in markets:
            groups = by_market[market]
            for key in sorted(groups):
                cursor = cursors[market][key]
                if cursor < len(groups[key]):
                    picked.append(groups[key][cursor])
                    cursors[market][key] = cursor + 1
                    progressed = True
                    break
            if len(picked) >= size:
                break
        if not progressed:
            break
    return ordered.loc[sorted(set(picked))].reset_index(drop=True)


def _analyze_month(
    frame: pd.DataFrame,
    code: str,
    anchor: str,
    expected_days: Sequence[str],
    local: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """单月单股：应有 vs 实取、48 槽完整性、量额对账。"""
    if not len(frame):
        return {
            "status": "empty",
            "expected_days": len(expected_days),
            "present_days": 0,
            "complete_days": 0,
            "missing_days": list(expected_days)[:50],
        }
    frame = frame.copy()
    frame["day"] = frame["time"].astype(str).str[:8]
    counts = frame.groupby("day")["time"].nunique()
    present = set(counts.index)
    expected = set(expected_days)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    complete = int((counts == int(spec.MINUTE["grid_slots"])).sum())
    partial = int(((counts > 0) & (counts < int(spec.MINUTE["grid_slots"]))).sum())
    recon_rows = []
    for day, group in frame.groupby("day"):
        row = local.get((code, day))
        if row is None:
            recon_rows.append({"trade_date": day, "disposition": "not_reconcilable"})
            continue
        minute_vol = float(group["volume"].sum())
        minute_amount = float(group["amount"].sum())
        daily_vol = float(row["vol"]) * 100.0
        daily_amount = float(row["amount"]) * 1000.0
        check = {
            "reconcilable": daily_vol > 0 and daily_amount > 0,
            "minute_volume": minute_vol,
            "daily_volume": daily_vol,
            "minute_amount": minute_amount,
            "daily_amount": daily_amount,
            "volume_rel_error": abs(minute_vol - daily_vol) / daily_vol if daily_vol else None,
            "amount_rel_error": (
                abs(minute_amount - daily_amount) / daily_amount if daily_amount else None
            ),
            "high_gap": abs(float(group["high"].max()) - float(row["high"])),
            "low_gap": abs(float(group["low"].min()) - float(row["low"])),
        }
        recon_rows.append(
            {"trade_date": day, "disposition": quality.reconciliation_disposition(check)}
        )
    recon = pd.DataFrame(recon_rows)
    dispositions = recon["disposition"].value_counts().to_dict() if len(recon) else {}
    unexplained = sum(
        value for name, value in dispositions.items() if name.startswith("unexplained")
    )
    return {
        "status": "ok",
        "anchor": anchor,
        "expected_days": len(expected),
        "present_days": len(present),
        "complete_days": complete,
        "partial_days": partial,
        "missing_days": missing,
        "extra_days": extra,
        "dispositions": {str(k): int(v) for k, v in dispositions.items()},
        "unexplained": int(unexplained),
    }


def run_coverage(
    probe_dir: Path,
    *,
    max_queries: int = 300,
    max_seconds: float = 7200.0,
    sample_size: int = SAMPLE_SIZE,
    log=lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """执行覆盖验收并落盘（先写预登记计划，再取数）。"""
    import baostock as bs

    probe_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calendar = panel.load_calendar(REPO_ROOT)
    inputs = dataset_inputs()
    days = [day for day in inputs["dates"] if WINDOW_START <= str(day) <= WINDOW_END]
    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    table = _stock_table(inputs, result, days)
    sampled = _sample(table, int(sample_size), DELISTED_QUOTA)
    anchor_days = {anchor: [day for day in days if day[:6] == anchor] for anchor in ANCHORS}
    plan = []
    for row in sampled.to_dict("records"):
        for anchor in ANCHORS:
            if not anchor_days[anchor]:
                continue
            plan.append(
                {
                    "ts_code": row["ts_code"],
                    "bs_code": _bs_code(row["ts_code"]),
                    "anchor": anchor,
                    "start": _month_bounds(anchor)[0],
                    "end": _month_bounds(anchor)[1],
                    "expected_days": len(anchor_days[anchor]),
                }
            )
    artifacts.write_json_checked(
        probe_dir / "plan.json",
        {
            "window": [WINDOW_START, WINDOW_END],
            "anchors": list(ANCHORS),
            "sample_size": int(sample_size),
            "delisted_in_sample": int(sampled["delisted"].sum()),
            "queries": len(plan),
            "max_queries": int(max_queries),
            "max_seconds": float(max_seconds),
            "strata": "market × industry × liquidity_tercile，含退市配额",
            "sample": sampled[
                ["ts_code", "market", "industry", "eligible_days", "delisted"]
            ].to_dict("records"),
            "query_plan": plan,
        },
    )
    log(f"plan: {len(plan)} 查询，样本 {len(sampled)} 只（退市 {int(sampled['delisted'].sum())}）")

    lock = bs_lock.acquire(probe_dir.parent.parent, owner="bs_coverage")
    login = bs.login()
    if login.error_code != "0":
        bs_lock.release(lock)
        payload = {"status": "login_failed", "login_code": login.error_code}
        artifacts.write_json_checked(probe_dir / "summary.json", payload)
        return payload
    local = _local_state(sampled["ts_code"].tolist(), days)
    records: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(plan):
            if index >= int(max_queries) or time.perf_counter() - started >= max_seconds:
                records.append({**item, "status": "skipped_limit"})
                continue
            frame, meta = _query(item["bs_code"], item["start"], item["end"])
            entry = {
                **item,
                "response_code": meta["response_code"],
                "rows": meta["rows"],
                "elapsed_seconds": meta["elapsed_seconds"],
            }
            if meta["response_code"] != "0":
                entry["status"] = "query_failed"
            elif not len(frame):
                entry["status"] = "empty"
                entry["expected_in_month"] = len(
                    [day for day in anchor_days[item["anchor"]] if (item["ts_code"], day) in local]
                )
            else:
                raw_path = probe_dir / f"{item['ts_code']}_{item['anchor']}.parquet"
                frame.to_parquet(raw_path, index=False)
                entry["sha256"] = sha256_file(raw_path)
                entry.update(
                    _analyze_month(
                        frame,
                        item["ts_code"],
                        item["anchor"],
                        [
                            day
                            for day in anchor_days[item["anchor"]]
                            if (item["ts_code"], day) in local
                            and int(local[(item["ts_code"], day)].get("tradable", 1) or 0) == 1
                            and int(local[(item["ts_code"], day)].get("is_suspended", 0) or 0) == 0
                        ],
                        local,
                    )
                )
            records.append(entry)
    finally:
        bs.logout()
        bs_lock.release(lock)

    throughput = throughput_blocks(
        sampled.sort_values("ts_code").head(THROUGHPUT_BLOCKS)["ts_code"].tolist(),
        days=THROUGHPUT_BLOCK_DAYS,
        end="2025-12-31",
        log=log,
    )
    requests_total = len(
        download.build_coverage(
            result.eligible[[inputs["dates"].index(day) for day in days]],
            int(spec.MINUTE["chunk_trading_days"]),
            int(spec.WINDOW["minute_preheat_trading_days"]),
        )
    )
    rate = throughput.get("observed_queries_per_hour")
    summary = {
        "status": "completed",
        "window": [WINDOW_START, WINDOW_END],
        "queries_planned": len(plan),
        "queries_sent": sum(1 for item in records if item.get("status") != "skipped_limit"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "sample_stocks": int(len(sampled)),
        "delisted_in_sample": int(sampled["delisted"].sum()),
        "anchor_summary": _anchor_summary(records),
        "empty_by_anchor": _empty_by_anchor(records),
        "dispositions": _disposition_totals(records),
        "throughput": throughput,
        "projected": {
            "requests_2020_window": int(requests_total),
            "hours_at_measured_rate": (round(requests_total / rate, 2) if rate else None),
            "within_22h": bool(
                rate and requests_total / rate <= float(spec.BUDGET["r1_download_max_hours"])
            ),
        },
        "records": records,
        "note": "只核验覆盖与语义；空返回、缺失、未解释对账分别登记，不做经济推断",
    }
    summary["verdict"] = _verdict(summary)
    artifacts.write_json_checked(probe_dir / "summary.json", summary)
    (probe_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def dataset_inputs() -> dict[str, Any]:
    """加载 2020 起输入面板（与 R0/R1 同一实现）。"""
    from scripts.v2_p7.minute_repair_ml import dataset

    return dataset.load_inputs(REPO_ROOT, WINDOW_START, WINDOW_END)


def throughput_blocks(
    codes: Sequence[str],
    *,
    days: int,
    end: str,
    log,
    campaign: Path | None = None,
    guard: bool = True,
) -> dict[str, Any]:
    """真实 120 交易日块吞吐（排除空返回与日线查询的混淆；自行登录/登出）。

    `guard=False` 供调用方已持有单连接锁时使用（禁止嵌套加锁）。
    """
    import baostock as bs

    picks = list(codes)[:THROUGHPUT_BLOCKS]
    end_ts = pd.Timestamp(end)
    start = end_ts - pd.offsets.BDay(int(days))
    rows = 0
    ok = 0
    empty = 0
    failed = 0
    seconds = 0.0
    bytes_total = 0
    lock = (
        bs_lock.acquire(Path(campaign), owner="bs_throughput")
        if guard and campaign is not None
        else None
    )
    login = bs.login()
    if login.error_code != "0":
        bs_lock.release(lock)
        return {"status": "login_failed", "blocks": len(picks)}
    try:
        for code in picks:
            frame, meta = _query(
                _bs_code(code), start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
            )
            seconds += float(meta["elapsed_seconds"])
            if meta["response_code"] != "0":
                failed += 1
                continue
            if meta["rows"] == 0:
                empty += 1
                continue
            ok += 1
            rows += meta["rows"]
            bytes_total += int(frame.memory_usage(deep=True).sum())
    finally:
        bs.logout()
        bs_lock.release(lock)
    rate = (ok / seconds * 3600.0) if seconds > 0 and ok else None
    log(f"throughput blocks: ok={ok} empty={empty} rows={rows} seconds={seconds:.1f}")
    return {
        "blocks": len(picks),
        "ok_blocks": ok,
        "empty_blocks": empty,
        "failed_blocks": failed,
        "rows": rows,
        "in_memory_bytes": bytes_total,
        "seconds": round(seconds, 2),
        "observed_queries_per_hour": round(rate, 1) if rate else None,
        "rows_per_query": round(rows / ok, 1) if ok else None,
        "status": "measured" if ok and not empty and not failed else "insufficient",
    }


def _anchor_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for anchor in ANCHORS:
        scoped = [
            item for item in records if item.get("anchor") == anchor and item.get("status") == "ok"
        ]
        if not scoped:
            out[anchor] = {"stocks": 0}
            continue
        out[anchor] = {
            "stocks_with_data": len(scoped),
            "expected_days": int(sum(item["expected_days"] for item in scoped)),
            "present_days": int(sum(item["present_days"] for item in scoped)),
            "complete_days": int(sum(item["complete_days"] for item in scoped)),
            "partial_days": int(sum(item["partial_days"] for item in scoped)),
            "missing_days": int(sum(len(item["missing_days"]) for item in scoped)),
            "unexplained": int(sum(item["unexplained"] for item in scoped)),
        }
    return out


def _empty_by_anchor(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in records:
        if item.get("status") == "empty":
            counts[str(item["anchor"])] = counts.get(str(item["anchor"]), 0) + 1
    return counts


def _disposition_totals(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in records:
        for name, value in (item.get("dispositions") or {}).items():
            totals[name] = totals.get(name, 0) + int(value)
    return totals


def _verdict(summary: dict[str, Any]) -> dict[str, Any]:
    anchors = summary["anchor_summary"]
    first = anchors.get("202001") or {}
    missing = int(first.get("missing_days", 0))
    partial = int(first.get("partial_days", 0))
    unexplained = (
        int(summary["dispositions"].get("unexplained_volume", 0))
        + int(summary["dispositions"].get("unexplained_amount", 0))
        + int(summary["dispositions"].get("unexplained_both", 0))
    )
    full = (
        first.get("stocks_with_data", 0) > 0 and missing == 0 and partial == 0 and unexplained == 0
    )
    return {
        "coverage_2020": "full" if full else "gaps",
        "missing_days_2020": missing,
        "partial_days_2020": partial,
        "unexplained_recon": unexplained,
        "empty_2020": int(summary["empty_by_anchor"].get("202001", 0)),
        "throughput_ok": summary["throughput"].get("status") == "measured",
        "verdict": (
            "usable" if full and summary["throughput"].get("status") == "measured" else "not_proven"
        ),
    }


def _render(summary: dict[str, Any]) -> str:
    lines = ["# BaoStock 2020 起目标域覆盖验收", "", "## 结论", ""]
    verdict = summary["verdict"]
    lines.append(
        f"- 2020 锚覆盖：`{verdict['coverage_2020']}`；缺口 {verdict['missing_days_2020']} 天、"
        f"不完整 {verdict['partial_days_2020']} 天、未解释对账 {verdict['unexplained_recon']}"
    )
    lines.append(f"- 空返回（2020 锚）：{verdict['empty_2020']}")
    lines.append(f"- 判定：`{verdict['verdict']}`")
    lines.append("")
    lines.append("## 分锚点")
    lines.append("")
    lines.append("| 锚点 | 有数据股票 | 应有日 | 实取日 | 完整日 | 不完整 | 缺失 | 未解释 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for anchor, block in summary["anchor_summary"].items():
        lines.append(
            f"| {anchor} | {block.get('stocks_with_data', 0)} | {block.get('expected_days', 0)} | "
            f"{block.get('present_days', 0)} | {block.get('complete_days', 0)} | "
            f"{block.get('partial_days', 0)} | {block.get('missing_days', 0)} | "
            f"{block.get('unexplained', 0)} |"
        )
    lines.append("")
    thr = summary["throughput"]
    lines.append(
        f"- 吞吐：{thr.get('ok_blocks')}/{thr.get('blocks')} 块成功，"
        f"{thr.get('observed_queries_per_hour')} 查询/小时；全量 "
        f"{summary['projected']['requests_2020_window']} 请求预计 "
        f"{summary['projected']['hours_at_measured_rate']} 小时"
    )
    lines.append(f"- 查询：计划 {summary['queries_planned']}，实发 {summary['queries_sent']}")
    lines.append("")
    lines.append(f"> {summary['note']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：只接受上限参数。"""
    import argparse

    parser = argparse.ArgumentParser(description="BaoStock 2020 起覆盖验收")
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--artifact-base", default=None)
    args = parser.parse_args(argv)
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    probe_dir = campaign / "bs_coverage" / artifacts.new_run_id("bscover")
    summary = run_coverage(
        probe_dir,
        max_queries=args.max_queries,
        max_seconds=args.max_seconds,
        sample_size=args.sample_size,
        log=lambda message: print(f"[cover] {message}", flush=True),
    )
    print(json.dumps(summary.get("verdict", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

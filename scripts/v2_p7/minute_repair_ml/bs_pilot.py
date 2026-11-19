"""Step 1：BaoStock 小样本**端到端管道验证**（40 只 × 2020-04…2026-10，单连接）。

目的不是研究，而是用**真实的 R1 DownloadEngine**跑一遍 BaoStock 数据：
逐请求落盘 → 标准化 → 48 槽质量 → 日线对账 → 覆盖裁定 → 断点续跑，
确认管道在免费源上可用、并得到可信的真实吞吐。

约束：单进程单连接（`bs_lock`），分批 10 只，可随时中断续跑。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from scripts.v2_p7.common import REPO_ROOT
from scripts.v2_p7.minute_repair_ml import (
    artifacts,
    baostock_source,
    bs_coverage,
    bs_lock,
    dataset,
    domain,
    download,
    panel,
    spec,
)

WINDOW: tuple[str, str] = ("20200401", "20261029")
DEFAULT_SAMPLE = 40
DELISTED_QUOTA = 5


def run_pilot(
    pilot_dir: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE,
    batch_codes: int = 10,
    log=lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """执行小样本管道验证并落盘。"""
    started = time.perf_counter()
    pilot_dir.mkdir(parents=True, exist_ok=True)
    campaign = pilot_dir.parent.parent
    calendar = panel.load_calendar(REPO_ROOT)
    inputs = dataset.load_inputs(REPO_ROOT, WINDOW[0], WINDOW[1])
    days = [str(day) for day in inputs["dates"] if WINDOW[0] <= str(day) <= WINDOW[1]]
    result = domain.build_domain(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    table = bs_coverage._stock_table(inputs, result, days)
    sampled = bs_coverage._sample(table, int(sample_size), DELISTED_QUOTA)
    codes = sampled["ts_code"].astype(str).tolist()
    positions = [inputs["dates"].index(day) for day in days]
    column_index = [inputs["codes"].index(code) for code in codes]
    eligible = result.eligible[positions][:, column_index]
    covered = download.build_coverage(
        eligible,
        int(spec.MINUTE["chunk_trading_days"]),
        int(spec.WINDOW["minute_preheat_trading_days"]),
    )
    log(
        f"pilot plan: {len(codes)} 只 / {len(days)} 交易日 / {len(covered)} 请求"
        f"（退市 {int(sampled['delisted'].sum())}）"
    )
    artifacts.write_json_checked(
        pilot_dir / "plan.json",
        {
            "window": list(WINDOW),
            "stocks": codes,
            "sample_meta": sampled.to_dict("records"),
            "trading_days": len(days),
            "requests": len(covered),
            "batch_codes": int(batch_codes),
        },
    )

    lock = bs_lock.acquire(campaign, owner="bs_pilot")
    src = baostock_source.BaoStockSource(log=log)
    engine = download.DownloadEngine(
        campaign=campaign,
        work_dir=pilot_dir / "work",
        src=src,  # type: ignore[arg-type]
        days=days,
        codes=codes,
        eligible=eligible,
        log=log,
        batch_codes=int(batch_codes),
        resume=True,
    )
    try:
        rep = engine.run()
    finally:
        src.close()
        bs_lock.release(lock)

    elapsed = round(time.perf_counter() - started, 1)
    storage = {
        "raw_blocks_bytes": (
            sum(path.stat().st_size for path in (campaign / "raw").rglob("*.parquet"))
            if (campaign / "raw").exists()
            else 0
        ),
        "raw_spool_bytes": rep.raw_spool_bytes,
        "standardized_bytes": (
            sum(
                path.stat().st_size
                for path in (pilot_dir / "work" / "dataset" / "minute").rglob("*.parquet")
            )
            if (pilot_dir / "work" / "dataset" / "minute").exists()
            else 0
        ),
    }
    storage["total_bytes"] = int(sum(storage.values()))
    summary = {
        "status": "completed",
        "stage": "step1_pilot",
        "window": list(WINDOW),
        "stocks": len(codes),
        "delisted_in_sample": int(sampled["delisted"].sum()),
        "trading_days": len(days),
        "requests_planned": len(covered),
        "requests_ok": rep.ok_requests,
        "requests_failed": rep.failed_requests,
        "requests_replayed": rep.replayed_requests,
        "blocked": rep.blocked,
        "elapsed_seconds": elapsed,
        "download_hours": rep.duration_hours,
        "standardized_rows": rep.standardized_rows,
        "auction_rows": rep.auction_rows,
        "conflicts": rep.conflicts,
        "conflict_rows": rep.conflict_rows,
        "exclusion_rows": rep.exclusion_rows,
        "coverage": rep.coverage,
        "reconciliation": rep.reconciliation,
        "throughput": src.stats(),
        "storage": storage,
        "verdict": _verdict(rep),
    }
    artifacts.write_json_checked(pilot_dir / "summary.json", summary)
    (pilot_dir / "report.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _verdict(rep: download.DownloadReport) -> dict[str, Any]:
    coverage = rep.coverage or {}
    recon = rep.reconciliation or {}
    unresolved = int(coverage.get("unresolved_count", 0))
    warmup_unresolved = int(coverage.get("warmup_unresolved", 0))
    unexplained = int(recon.get("unexplained", 0))
    counts = coverage.get("counts") or {}
    # 槽位完整性：只有"行存在但核心槽不全"才判失败（k=1..47）
    slots_ok = int(counts.get("unresolved_incomplete", 0)) == 0
    checks = {
        "no_block": rep.blocked is None,
        "requests_all_ok": rep.failed_requests == 0 and rep.ok_requests > 0,
        "coverage_unresolved_zero": unresolved == 0,
        "warmup_unresolved_zero": warmup_unresolved == 0,
        "reconciliation_no_unexplained": unexplained == 0 and int(recon.get("rows", 0)) > 0,
        "slot_completeness": slots_ok,
    }
    return {
        "checks": checks,
        "coverage_status": coverage.get("status"),
        "reconciliation_status": recon.get("status"),
        "pass": all(checks.values()),
    }


def _render(summary: dict[str, Any]) -> str:
    verdict = summary["verdict"]
    lines = ["# Step 1：BaoStock 小样本管道验证", "", "## 结论", ""]
    lines.append(f"- 判定：`{'PASS' if verdict['pass'] else 'FAIL'}`")
    for name, ok in verdict["checks"].items():
        lines.append(f"- {name}: {'ok' if ok else 'fail'}")
    lines.append("")
    lines.append("## 规模与耗时")
    lines.append("")
    lines.append(
        f"- 窗口 {summary['window'][0]}…{summary['window'][1]}，{summary['trading_days']} 个交易日"
    )
    lines.append(f"- 股票 {summary['stocks']} 只（退市 {summary['delisted_in_sample']}）")
    lines.append(
        f"- 请求 计划 {summary['requests_planned']} / 成功 {summary['requests_ok']} / "
        f"失败 {summary['requests_failed']} / 重放 {summary['requests_replayed']}"
    )
    lines.append(
        f"- 标准化 {summary['standardized_rows']} 行、竞价 {summary['auction_rows']} 行、"
        f"冲突 {summary['conflicts']}、排除台账 {summary['exclusion_rows']} 行"
    )
    thr = summary["throughput"]
    lines.append(
        f"- 吞吐：{thr.get('requests')} 次查询、{thr.get('rows')} 行，"
        f"中位 {thr.get('latency_median')} 秒、最大 {thr.get('latency_max')} 秒，"
        f"{thr.get('queries_per_hour')} 查询/小时"
    )
    lines.append(f"- 墙钟：{summary['elapsed_seconds']} 秒")
    cov = summary["coverage"]
    lines.append(
        f"- 覆盖：{cov.get('status')}，应有 {cov.get('expected_rows')}、接受 {cov.get('accepted_rows')}、"
        f"未解决 {cov.get('unresolved_count')}、异常 {cov.get('anomaly_count')}"
    )
    recon = summary["reconciliation"]
    lines.append(
        f"- 对账：{recon.get('status')}，行 {recon.get('rows')}，"
        f"未解释 {recon.get('unexplained')}，处置 {recon.get('disposition_counts')}"
    )
    lines.append("")
    lines.append("> 只验证管道与数据门，不做经济研究；结果不外推到正式 R1。")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：`--sample-size` 可调，其余固定。"""
    import argparse

    parser = argparse.ArgumentParser(description="BaoStock Step 1 管道验证")
    parser.add_argument("--campaign-id", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--batch-codes", type=int, default=10)
    parser.add_argument("--artifact-base", default=None)
    args = parser.parse_args(argv)
    base = Path(args.artifact_base) if args.artifact_base else artifacts.default_artifact_base()
    campaign = artifacts.campaign_root(args.campaign_id, base)
    artifacts.ensure_layout(campaign, base)
    pilot_dir = campaign / "bs_pilot" / artifacts.new_run_id("bspilot")
    summary = run_pilot(
        pilot_dir,
        sample_size=args.sample_size,
        batch_codes=args.batch_codes,
        log=lambda message: print(f"[pilot] {message}", flush=True),
    )
    print(json.dumps(summary["verdict"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

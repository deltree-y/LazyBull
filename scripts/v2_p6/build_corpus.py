# -*- coding: utf-8 -*-
"""P6 T0.5 合成标准/压力语料生成器（设计 §7.3）。

定位：**性能规模语料**，只保证行数/字节规模、确定性与可双跑比对；不冒充 P6 输入语义契约
（P4/runs/P5b 的语义夹具归 T1/T3 的 pytest 合成夹具）。语料写入用户显式指定的**新目录**，
默认落 ``temp/``，不写生产数据根。

用法::

    python scripts/v2_p6/build_corpus.py --out temp/p6_t05_corpus/run_a --profile all
    python scripts/v2_p6/build_corpus.py --out temp/p6_t05_corpus/run_b --profile all

双跑一致性：``corpus_manifest.json:content_sha256`` 与 ``profiles.*.files[*].sha256`` 全等。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import contracts, validation  # noqa: E402
from src.lazybull.v2.store.store_identity import canonical_json_sha256  # noqa: E402

__all__ = ["CorpusError", "build", "main"]

#: 固定语料种子（变更即新语料，须重新冻结 SHA）
SEED = 20261105
#: 合成日期基准（不读取系统时钟）
BASE_DATE = date(2026, 1, 5)


class CorpusError(RuntimeError):
    """语料输出目录已存在或语料参数不合法。"""


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random(":".join((str(seed), *parts)))


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
            count += 1
    return count


def _write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_bytes((text + "\n").encode("utf-8"))
    return len(payload) if isinstance(payload, (list, dict)) else 0


def _fold_counts(total: int, folds: int) -> list[int]:
    base, remainder = divmod(total, folds)
    return [base + (1 if index < remainder else 0) for index in range(folds)]


def _dashboard(root: Path, counts: dict[str, int], observed: dict[str, int], seed: int) -> None:
    rng = _rng(seed, "dashboard")
    positions = []
    for index in range(counts["positions"]):
        price = round(rng.uniform(5.0, 80.0), 2)
        shares = 100 * rng.randint(1, 40)
        positions.append(
            {
                "ts_code": f"{600000 + index:06d}.SH",
                "shares": shares,
                "buy_price": price,
                "buy_cost": round(shares * price * 0.0003, 2),
                "buy_date": (BASE_DATE + timedelta(days=index)).isoformat(),
                "buy_pnl_price": round(price * rng.uniform(0.95, 1.05), 2),
                "status": "holding" if index % 11 else "pending_exit",
            }
        )
    observed["dashboard/positions.json"] = _write_json(
        root / "dashboard" / "positions.json", positions
    )

    instructions = []
    for index in range(counts["instructions"]):
        action = "buy" if index % 3 else "sell"
        price = round(rng.uniform(5.0, 80.0), 2)
        shares = 100 * rng.randint(1, 20)
        instructions.append(
            (
                "20260603",
                "20260602",
                f"{600000 + index:06d}.SH",
                action,
                shares,
                price,
                round(shares * price, 2),
                "signal_rank",
                "limit" if index % 5 else "market",
                round(rng.uniform(0.0, 0.2), 6),
                index,
                "synthetic",
            )
        )
    observed["dashboard/instructions.csv"] = _write_csv(
        root / "dashboard" / "instructions.csv",
        (
            "trade_date,source_date,ts_code,action,shares,price,amount,reason,"
            "order_type,target_weight,priority,note"
        ).split(","),
        instructions,
    )

    run_index = 0.0
    nav_rows = []
    current = BASE_DATE
    for index in range(counts["nav_rows"]):
        current += timedelta(days=1)
        while current.weekday() >= 5:
            current += timedelta(days=1)
        run_index *= 1.0 + rng.uniform(-0.012, 0.014)
        cash = run_index * rng.uniform(0.02, 0.12)
        nav_rows.append(
            (
                current.strftime("%Y%m%d"),
                f"{run_index:.6f}",
                f"{run_index - cash:.6f}",
                f"{cash:.6f}",
                f"{run_index / 500000.0:.8f}",
            )
        )
    observed["dashboard/nav.csv"] = _write_csv(
        root / "dashboard" / "nav.csv",
        ("trade_date", "total_value", "position_value", "cash", "nav"),
        nav_rows,
    )

    quality_rows = []
    for index in range(counts["quality_abnormal_rows"]):
        quality_rows.append(
            (
                "clean" if index % 2 else "raw",
                f"dataset_{index % 7}",
                f"metric_{index % 13}",
                f"2026{index % 12 + 1:02d}",
                str(index),
                "error" if index % 2 else "warning",
            )
        )
    for dataset in ("daily", "adj_factor", "features", "labels"):
        quality_rows.append(("clean", dataset, "partition_count", "", "17", "ok"))
    observed["dashboard/quality_metrics.csv"] = _write_csv(
        root / "dashboard" / "quality_metrics.csv",
        ("layer", "dataset", "metric", "partition", "value", "status"),
        quality_rows,
    )

    half = counts["instructions"] // 2
    pending_buy = [
        {"ts_code": f"{600000 + index:06d}.SH", "shares": 100 * (index + 1), "price": 10.0 + index}
        for index in range(half)
    ]
    pending_sell = [
        {"ts_code": f"{601000 + index:06d}.SH", "shares": 100 * (index + 1), "price": 12.0 + index}
        for index in range(counts["instructions"] - half)
    ]
    observed["dashboard/pending_buy.json"] = _write_json(
        root / "dashboard" / "pending_buy.json", pending_buy
    )
    observed["dashboard/pending_sell.json"] = _write_json(
        root / "dashboard" / "pending_sell.json", pending_sell
    )
    observed["dashboard/rebalance_state.json"] = _write_json(
        root / "dashboard" / "rebalance_state.json",
        {
            "rebalance_day_index": 3,
            "rebalance_cycle": 2,
            "next_rebalance_date": "20260610",
            "last_rebalance_date": "20260603",
            "pending_signal_count": half,
        },
    )


@dataclass
class _ArmContext:
    folder: Path
    folds: int
    counts: dict[str, int]
    rng: random.Random
    observed: dict[str, int]
    prefix: str


def _write_summary(ctx: _ArmContext) -> None:
    rows = []
    for index in range(ctx.folds):
        rows.append(
            (
                index,
                (BASE_DATE + timedelta(days=index * 20)).strftime("%Y%m%d"),
                (BASE_DATE + timedelta(days=index * 20 + 19)).strftime("%Y%m%d"),
                round(ctx.rng.uniform(-0.2, 0.3), 6),
                round(ctx.rng.uniform(-0.3, 0.5), 6),
                round(-ctx.rng.uniform(0.02, 0.25), 6),
                round(ctx.rng.uniform(-1.0, 2.0), 6),
                round(ctx.rng.uniform(0.3, 0.8), 6),
                round(ctx.rng.uniform(0.2, 0.7), 6),
                round(ctx.rng.uniform(0.2, 0.7), 6),
            )
        )
    header = (
        "fold,test_start,test_end,bt_total_return,bt_annual_return,bt_max_drawdown,"
        "bt_sharpe,bt_win_rate,KEY_Top20_hit_rate,KEY_Top30_hit_rate"
    ).split(",")
    ctx.observed[f"{ctx.prefix}/summary.csv"] = _write_csv(ctx.folder / "summary.csv", header, rows)


def _write_chain(ctx: _ArmContext) -> None:
    def rows() -> Iterator[tuple[Any, ...]]:
        for fold, count in enumerate(_fold_counts(ctx.counts["chain_rows_per_arm"], ctx.folds)):
            nav = 1.0
            for row_index in range(count):
                nav *= 1.0 + ctx.rng.uniform(-0.01, 0.011)
                day = (BASE_DATE + timedelta(days=fold * 20 + row_index)).strftime("%Y%m%d")
                yield fold, day, f"{nav:.8f}"

    ctx.observed[f"{ctx.prefix}/chain_nav.csv"] = _write_csv(
        ctx.folder / "chain_nav.csv", ("fold", "trade_date", "nav"), rows()
    )


def _write_daily(ctx: _ArmContext) -> None:
    def rows() -> Iterator[tuple[Any, ...]]:
        for fold, count in enumerate(_fold_counts(ctx.counts["daily_rows_per_arm"], ctx.folds)):
            for row_index in range(count):
                day = (BASE_DATE + timedelta(days=fold * 20 + row_index)).strftime("%Y%m%d")
                yield (
                    fold,
                    day,
                    round(ctx.rng.uniform(0.0, 5_000_000.0), 2),
                    round(ctx.rng.uniform(0.0, 2_000.0), 2),
                )

    ctx.observed[f"{ctx.prefix}/daily.csv"] = _write_csv(
        ctx.folder / "daily.csv", ("fold", "trade_date", "turnover_amount", "cost"), rows()
    )


def _write_trades(ctx: _ArmContext) -> None:
    def rows() -> Iterator[tuple[Any, ...]]:
        for fold, count in enumerate(_fold_counts(ctx.counts["trade_rows_per_arm"], ctx.folds)):
            for row_index in range(count):
                price = round(ctx.rng.uniform(5.0, 80.0), 2)
                shares = 100 * ctx.rng.randint(1, 30)
                day = (BASE_DATE + timedelta(days=fold * 20 + row_index % 20)).strftime("%Y%m%d")
                yield (
                    fold,
                    day,
                    f"{600000 + row_index % 500:06d}.SH",
                    "buy" if row_index % 2 else "sell",
                    shares,
                    price,
                    round(shares * price, 2),
                    round(shares * price * 0.0003, 2),
                )

    ctx.observed[f"{ctx.prefix}/trades.csv"] = _write_csv(
        ctx.folder / "trades.csv",
        ("fold", "trade_date", "ts_code", "action", "shares", "price", "amount", "cost"),
        rows(),
    )


def _write_attribution(ctx: _ArmContext) -> None:
    def rows() -> Iterator[tuple[Any, ...]]:
        for fold, count in enumerate(
            _fold_counts(ctx.counts["attribution_rows_per_arm"], ctx.folds)
        ):
            for row_index in range(count):
                day = (BASE_DATE + timedelta(days=fold * 20 + row_index % 20)).strftime("%Y%m%d")
                yield (
                    fold,
                    day,
                    f"{600000 + row_index % 500:06d}.SH",
                    "filled",
                    "signal_rank",
                    round(ctx.rng.uniform(-0.05, 0.05), 8),
                )

    ctx.observed[f"{ctx.prefix}/attribution.csv"] = _write_csv(
        ctx.folder / "attribution.csv",
        ("fold", "trade_date", "ts_code", "status", "reason", "signal_to_buy_return"),
        rows(),
    )


def _write_topk(ctx: _ArmContext) -> None:
    if ctx.counts["topk_detail_rows_per_arm"] <= 0:
        return

    def rows() -> Iterator[tuple[Any, ...]]:
        for fold, count in enumerate(
            _fold_counts(ctx.counts["topk_detail_rows_per_arm"], ctx.folds)
        ):
            for row_index in range(count):
                day = (BASE_DATE + timedelta(days=fold * 20 + row_index % 20)).strftime("%Y%m%d")
                yield (
                    fold,
                    day,
                    f"{600000 + row_index % 500:06d}.SH",
                    row_index % 20,
                    round(ctx.rng.uniform(-0.03, 0.03), 8),
                )

    ctx.observed[f"{ctx.prefix}/topk_detail.csv"] = _write_csv(
        ctx.folder / "topk_detail.csv",
        ("fold", "trade_date", "ts_code", "rank", "ret"),
        rows(),
    )


def _arm_files(
    base: Path,
    suffix: str,
    counts: dict[str, int],
    observed: dict[str, int],
    seed: int,
    prefix: str,
) -> None:
    rng = _rng(seed, "experiment", suffix)
    ctx = _ArmContext(
        folder=base / suffix,
        folds=counts["folds"],
        counts=counts,
        rng=rng,
        observed=observed,
        prefix=prefix,
    )
    for writer in (
        _write_summary,
        _write_chain,
        _write_daily,
        _write_trades,
        _write_attribution,
        _write_topk,
    ):
        writer(ctx)


def _experiment(root: Path, counts: dict[str, int], observed: dict[str, int], seed: int) -> None:
    for index in range(counts["arms"]):
        suffix = f"arm_{chr(ord('a') + index)}"
        _arm_files(
            root / "experiment" / "runs",
            suffix,
            counts,
            observed,
            seed,
            f"experiment/runs/{suffix}",
        )
        _arm_files(
            root / "experiment" / "legacy_wf",
            suffix,
            counts,
            observed,
            seed,
            f"experiment/legacy_wf/{suffix}",
        )


def _run_index_row(index: int, profile: str, prefix: str) -> tuple[Any, ...]:
    window_end = (BASE_DATE + timedelta(days=index * 7)).strftime("%Y%m%d")
    run_id = f"{prefix}_run_{index:05d}"
    digest = f"{(index * 2654435761) & 0xFFFFFFFFFFFFFFFF:016x}" * 4
    return (
        f"run_key_{profile}_{index:05d}",
        run_id,
        f"batch_{index % 37:03d}",
        "" if index % 5 else f"campaign_{index % 11:03d}",
        "research",
        "backtest",
        "v2",
        (BASE_DATE + timedelta(days=index * 7 - 120)).strftime("%Y%m%d"),
        window_end,
        "trade_date",
        f"data_state_{index % 5}",
        f"code_state_{index % 3}",
        f"runs/{run_id}",
        digest,
        "",
        digest,
        "full",
        "available",
        "",
    )


def _trend(
    root: Path,
    counts: dict[str, int],
    observed: dict[str, int],
    profile: str,
    seed: int,
) -> None:
    header = (
        "run_key,run_id,batch_id,campaign_id,evidence_class,host_mode,implementation,"
        "window_start,window_end,date_semantics,data_state_id,code_state_id,source_ref,"
        "source_identity_sha256,source_manifest_sha256,source_sha256,index_mode,"
        "report_status,degradation_reason_code"
    ).split(",")
    trend_header = (
        "run_key,metric_name,value,unit,window_start,window_end,date_semantics,source_ref,"
        "source_sha256,report_status,degradation_reason_code"
    ).split(",")

    previous_root = root / "trend" / "previous"
    previous_rows = [
        _run_index_row(index, profile, "prev") for index in range(counts["previous_runs"])
    ]
    observed["trend/previous/run_index.csv"] = _write_csv(
        previous_root / "run_index.csv", header, previous_rows
    )

    def previous_trend() -> Iterator[tuple[Any, ...]]:
        for index in range(counts["previous_runs"]):
            run_key = f"run_key_{profile}_{index:05d}"
            cagr = random.Random(f"{seed}:trend:{index}:cagr").uniform(-0.2, 0.5)
            drawdown = -random.Random(f"{seed}:trend:{index}:dd").uniform(0.02, 0.5)
            sharpe = random.Random(f"{seed}:trend:{index}:sharpe").uniform(-1.0, 2.0)
            for metric, value, unit in (
                ("cagr", round(cagr, 8), "fraction"),
                ("max_drawdown", round(drawdown, 8), "fraction"),
                ("sharpe", round(sharpe, 8), "ratio"),
            ):
                yield (
                    run_key,
                    metric,
                    value,
                    unit,
                    (BASE_DATE + timedelta(days=index * 7 - 120)).strftime("%Y%m%d"),
                    (BASE_DATE + timedelta(days=index * 7)).strftime("%Y%m%d"),
                    "trade_date",
                    f"trend/previous/run_{index:05d}",
                    f"{(index * 40503) & 0xFFFFFFFF:08x}" * 8,
                    "available",
                    "",
                )

    observed["trend/previous/trend.csv"] = _write_csv(
        previous_root / "trend.csv", trend_header, previous_trend()
    )

    new_root = root / "trend" / "new"
    new_rows = [
        _run_index_row(counts["previous_runs"] + index, profile, "new")
        for index in range(counts["new_runs"])
    ]
    observed["trend/new/run_index.csv"] = _write_csv(new_root / "run_index.csv", header, new_rows)
    new_trend = []
    for index in range(counts["new_runs"]):
        run_key = f"run_key_{profile}_{counts['previous_runs'] + index:05d}"
        for metric, value, unit in (
            ("cagr", 0.18, "fraction"),
            ("max_drawdown", -0.09, "fraction"),
            ("sharpe", 1.1, "ratio"),
        ):
            new_trend.append(
                (
                    run_key,
                    metric,
                    value,
                    unit,
                    (BASE_DATE + timedelta(days=counts["previous_runs"] * 7 - 120)).strftime(
                        "%Y%m%d"
                    ),
                    (BASE_DATE + timedelta(days=counts["previous_runs"] * 7)).strftime("%Y%m%d"),
                    "trade_date",
                    f"trend/new/run_{index:05d}",
                    f"{(index * 7919) & 0xFFFFFFFF:08x}" * 8,
                    "available",
                    "",
                )
            )
    observed["trend/new/trend.csv"] = _write_csv(new_root / "trend.csv", trend_header, new_trend)


def _file_entries(root: Path, base: Path, observed: dict[str, int]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        entry: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": validation.sha256_file(path),
        }
        key = relative.split("/", 1)[1] if "/" in relative else relative
        if key in observed:
            entry["rows"] = observed[key]
        entries.append(entry)
    return entries


def build(out: Path | str, *, profile: str = "all", seed: int = SEED) -> dict[str, Any]:
    """生成语料到新目录并返回 manifest；目录已存在即拒绝（不覆盖历史语料）。"""
    target = Path(out)
    if target.exists():
        raise CorpusError(f"语料输出目录已存在，拒绝覆盖：{target}")
    profiles = ["standard", "stress"] if profile == "all" else [profile]
    if any(name not in contracts.CORPUS_PROFILES for name in profiles):
        raise CorpusError(f"未知语料 profile：{profile!r}")
    target.mkdir(parents=True, exist_ok=False)

    manifest_profiles: dict[str, Any] = {}
    for name in profiles:
        counts = contracts.CORPUS_PROFILES[name]
        root = target / name
        root.mkdir()
        observed: dict[str, int] = {}
        _dashboard(root, counts["dashboard"], observed, seed)
        _experiment(root, counts["experiment"], observed, seed)
        _trend(root, counts["trend"], observed, name, seed)
        entries = _file_entries(root, target, observed)
        manifest_profiles[name] = {
            "spec": {section: dict(values) for section, values in counts.items()},
            "files": entries,
            "content_sha256": canonical_json_sha256(entries),
        }

    payload = {
        "schema": contracts.CORPUS_SCHEMA,
        "seed": int(seed),
        "profiles": manifest_profiles,
        "content_sha256": canonical_json_sha256(
            {name: entry["content_sha256"] for name, entry in manifest_profiles.items()}
        ),
        "notes": {
            "scope": "performance_scale_corpus_only",
            "semantic_fixtures": "T1/T3 合成夹具，不在本语料内",
            "p5b_bundles": "T1 pytest 合成 sealed bundle，本语料不生成",
            "no_real_config": True,
        },
    }
    validation.write_json_new(target / "corpus_manifest.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 P6 T0.5 合成标准/压力语料（默认写新目录）")
    parser.add_argument("--out", required=True, help="新的语料输出目录（已存在即失败）")
    parser.add_argument("--profile", choices=["all", "standard", "stress"], default="all")
    parser.add_argument("--seed", type=int, default=SEED, help="固定语料种子（默认 20261105）")
    parser.add_argument("--json", action="store_true", help="打印 manifest 摘要 JSON")
    args = parser.parse_args(argv)
    try:
        payload = build(args.out, profile=args.profile, seed=args.seed)
    except CorpusError as exc:
        print(f"[P6-T0.5] 语料生成失败：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {"content_sha256": payload["content_sha256"], "out": str(args.out)},
                ensure_ascii=False,
            )
        )
    else:
        print(f"语料已生成：{args.out}（content_sha256={payload['content_sha256']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

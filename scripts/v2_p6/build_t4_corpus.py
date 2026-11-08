# -*- coding: utf-8 -*-
"""P6 T4 趋势语料构建（确定性、规模语料；不引用真实生产数据）。

语料规模（设计 §7.3）：

- 标准档：research 500 run（runs 400 + 旧 WF 100）+ production 纸面 24 run；
- 压力档：research 2,000 run（runs 1,600 + 旧 WF 400）+ production 纸面 60 run。

每个 research run 都满足 runs 契约（``load_runs_batch`` 可读）并带 ``manifest.json``
（覆盖“runs 有 manifest ⇒ 身份 = manifest 字节 SHA”分支）；旧 WF run 无 manifest
（覆盖“排序后的消费文件 path+sha canonical 哈希 + manifest_absent”分支）。

``upto`` 支持**增量加 run**：先建 N-1 个 run 发布上一快照，再补最后 1 个 run 做增量生成。
所有内容由 ``index`` 确定性派生（无随机数、无墙钟、无绝对路径）。
"""

# 复杂度豁免：阶段专用语料构建（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

if __name__ == "__main__":  # 语料进程禁写 __pycache__
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.lazybull.v2.evidence.fingerprint_keys import fingerprint  # noqa: E402
from src.lazybull.v2.evidence.runs_schema import (  # noqa: E402
    DAILY_CONTRACT_COLS,
    TRADES_CONTRACT_COLS,
)

__all__ = ["CORPUS_SCHEMA", "PROFILE_COUNTS", "build", "content_digest"]

CORPUS_SCHEMA = "p6_t4_corpus/v1"
#: 语料规模（run 数；``paper`` 为 production 纸面 run 数）
PROFILE_COUNTS: dict[str, dict[str, int]] = {
    "standard": {"runs": 400, "legacy": 100, "paper": 24},
    "stress": {"runs": 1600, "legacy": 400, "paper": 60},
    # pytest 合成夹具（语义覆盖用，规模不代表验收档）
    "tiny": {"runs": 3, "legacy": 1, "paper": 3},
}
#: 历史 run 的轻量负载（只用于规模，不参与计时判定）
CHAIN_ROWS = 12
#: 计时新 run 的负载（设计 §7.3：14 折、1,700/5,000 chain+daily、2,000/10,000 trades；R3-07）
PROFILE_LOAD: dict[str, dict[str, int]] = {
    "standard": {"chain_rows": 1700, "folds": 14, "trades": 2000},
    "stress": {"chain_rows": 5000, "folds": 14, "trades": 10000},
}
_NAV_START = 1.0


def _json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """新文件写 canonical JSON（已存在即拒绝覆盖：语料幂等由存在性保证）。"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """新文件写 parquet（已存在即跳过，保证 ``upto`` 增量语义）。"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _business_days(start: str, count: int) -> list[str]:
    """从 ``start`` 起的连续自然日序列（YYYYMMDD；不依赖真实交易日历）。"""
    base = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    return [(base + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(count)]


def _load_for(index: int, heavy_index: int, profile: str) -> dict[str, int]:
    """计时新 run 用重负载，其余历史 run 保持轻量（规模只体现在 run 数上）。"""
    if index == heavy_index and profile in PROFILE_LOAD:
        return dict(PROFILE_LOAD[profile])
    return {"chain_rows": CHAIN_ROWS, "folds": 1, "trades": 2}


def _chain_frames(
    index: int, load: Mapping[str, int]
) -> tuple[pd.DataFrame, list[tuple[int, list[str]]]]:
    """多折链式净值（折界重复点：后折首行 = 前折末行同日同 NAV；总行数守恒）。

    R3-07-R1：``runs_loader._check_fold_seams`` 要求折界重复点语义，自然日/NAV 直接
    切折会被 load 拒绝，导致新 run 在指标计算前早退。这里为每折（除首折）复写前折末行，
    总行数仍等于 ``load["chain_rows"]``。
    """
    rows_total = int(load["chain_rows"])
    folds = max(1, min(int(load["folds"]), max(1, rows_total - int(load["folds"]) + 1)))
    budget = rows_total - (folds - 1)
    days = _business_days("20240101", budget)
    step = 0.001 + index % 7 * 0.0001
    per = max(1, budget // folds)
    frames: list[pd.DataFrame] = []
    spans: list[tuple[int, list[str]]] = []
    cursor = 0
    previous_last: dict[str, Any] | None = None
    for fold in range(folds):
        take = per if fold < folds - 1 else budget - per * (folds - 1)
        seg_days = days[cursor:cursor + take]
        seg_navs = [_NAV_START + step * (cursor + offset) for offset in range(take)]
        fold_days = list(seg_days)
        fold_navs = list(seg_navs)
        if previous_last is not None:
            fold_days.insert(0, previous_last["date"])
            fold_navs.insert(0, previous_last["nav"])
        frames.append(
            pd.DataFrame(
                {"date": fold_days, "nav": fold_navs, "split_index": [fold] * len(fold_days)}
            )
        )
        spans.append((fold, fold_days))
        previous_last = {"date": seg_days[-1], "nav": seg_navs[-1]}
        cursor += take
    return pd.concat(frames, ignore_index=True), spans


def _summary_frame(
    index: int, chain: pd.DataFrame, spans: Sequence[tuple[int, list[str]]], data_state_id: str
) -> pd.DataFrame:
    """逐折 summary（必需列 + 逐折窗口；契约 §3 允许配置列扩展）。"""
    rows = []
    for fold, days in spans:
        slice_ = chain[chain["split_index"] == fold]
        first = float(slice_["nav"].iloc[0])
        last = float(slice_["nav"].iloc[-1])
        total_return = last / first - 1.0
        rows.append(
            {
                "batch_id": f"p6t4_run_{index:04d}",
                "split_index": fold,
                "test_start": days[0],
                "test_end": days[-1],
                "bt_total_return": total_return,
                "bt_annual_return": total_return,
                "bt_max_drawdown": -0.02,
                "bt_sharpe": 1.0,
                "data_state_id": data_state_id,
                "git_commit": "corpus",
                "git_dirty": False,
            }
        )
    return pd.DataFrame(rows)


def _fold_trades(index: int, split: int, days: Sequence[str], count: int) -> pd.DataFrame:
    """单折成交（契约 26 列；买卖交替，卖出行 buy_date 合法）。"""
    price = 10.0 + index % 5
    shares = 1000
    rows: list[dict[str, Any]] = []
    for offset in range(count):
        buy = offset % 2 == 0
        day = days[offset % len(days)]
        signal = days[max(0, offset % len(days) - 1)]
        rows.append(
            {
                "wf_run_id": f"p6t4_run_{index:04d}",
                "split_index": split,
                "model_version": 24001,
                "trade_date": day,
                "signal_date": signal,
                # 同一折内 (trade_date, action, ts_code) 必须唯一（契约 §9.4）：逐行给独立代码
                "ts_code": f"{600000 + offset:06d}.SH",
                "action": "buy" if buy else "sell",
                "price": price if buy else price * 1.01,
                "shares": shares,
                "amount": (price if buy else price * 1.01) * shares,
                "cost": 5.0,
                "buy_date": None if buy else days[max(0, offset % len(days) - 1)],
                "buy_price": None if buy else price,
                "buy_pnl_price": None if buy else price,
                "sell_pnl_price": None if buy else price * 1.01,
                "pnl_profit_amount": None if buy else price * shares * 0.01,
                "pnl_profit_pct": None if buy else 0.01,
                "sell_type": None if buy else "hold_period",
                "sell_timing": None if buy else "close",
                "sell_reason": None if buy else "hold_period",
                "trigger_type": None if buy else "scheduled",
                "buy_type": None if buy else "scheduled",
                "buy_reason": None if buy else "signal",
                "lot_id": None if buy else "L0",
                "tranche_idx": None if buy else 0,
            }
        )
    return pd.DataFrame(rows, columns=list(TRADES_CONTRACT_COLS))


def _daily_frame(index: int, chain: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """单折逐日（契约 11 列；与 trades 计数、chain 起止比一致）。"""
    navs = list(chain["nav"])
    base_value = 1_000_000.0
    buys = trades[trades["action"] == "buy"].groupby("trade_date").size()
    sells = trades[trades["action"] == "sell"].groupby("trade_date").size()
    rows = []
    for day, nav in zip(chain["date"], navs):
        total_value = base_value * float(nav)
        rows.append(
            {
                "trade_date": day,
                "total_value": total_value,
                "cash": total_value * 0.2,
                "market_value": total_value * 0.8,
                "nav": float(nav),
                "daily_return": 0.0,
                "n_positions": 1,
                "n_buys": int(buys.get(day, 0)),
                "n_sells": int(sells.get(day, 0)),
                "turnover_amount": 12_345.0 + index % 11,
                "exposure_lambda": 1.0,
            }
        )
    return pd.DataFrame(rows)[DAILY_CONTRACT_COLS]


def _build_run(root: Path, index: int, load: Mapping[str, int]) -> None:
    """一个规范化 runs 批次（含 manifest.json：身份 = manifest 字节 SHA）。"""
    batch = root / "runs" / f"p6t4_run_{index:04d}"
    if batch.exists():
        return
    data_state_id = f"ds_{index:04d}"
    config = {"initial_capital": 1_000_000.0, "seed": index}
    batch_meta = {
        "schema_version": 1,
        "batch_id": batch.name,
        "created_at": "2024-01-01T00:00:00",
        "host_mode": "backtest",
        "code_state": {"git_commit": "corpus", "git_dirty": False},
        "data_state": {"data_state_id": data_state_id, "sources": {"daily": "20240112"}},
        "config": config,
        "config_fingerprint": fingerprint(config),
        "baseline_ref": None,
        "arms": ["base"],
    }
    chain, spans = _chain_frames(index, load)
    summary = _summary_frame(index, chain, spans, data_state_id)
    _json_new(
        batch / "manifest.json",
        {"schema": "p6t4_corpus_run/v1", "run_id": batch.name, "index": index},
    )
    _json_new(batch / "batch_meta.json", batch_meta)
    _write_parquet(batch / "summary.parquet", summary)
    _write_parquet(batch / "chain_nav.parquet", chain)
    trade_counts = _split_counts(int(load["trades"]), len(spans))
    for (fold, days), count in zip(spans, trade_counts):
        trades = _fold_trades(index, fold, days, count)
        fold_chain = chain[chain["split_index"] == fold]
        _write_parquet(batch / "folds" / f"split{fold}" / "trades.parquet", trades)
        _write_parquet(
            batch / "folds" / f"split{fold}" / "daily.parquet",
            _daily_frame(index, fold_chain, trades),
        )


def _split_counts(total: int, parts: int) -> list[int]:
    """把 ``total`` 均匀分到 ``parts`` 份（余数给最后一份；每份至少 1）。"""
    if parts <= 0:
        return []
    base = max(1, total // parts)
    counts = [base] * parts
    counts[-1] = max(1, total - base * (parts - 1))
    return counts


def _build_legacy(root: Path, index: int) -> None:
    """一个旧 WF 运行目录（无 manifest；chain 用折内序数、summary 用自然日窗口）。"""
    run_dir = root / "legacy" / f"wf_{index:04d}"
    if run_dir.exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    days = _business_days("20240101", CHAIN_ROWS)
    step = 0.001 + index % 5 * 0.0002
    navs = [_NAV_START + step * offset for offset in range(CHAIN_ROWS)]
    pd.DataFrame(
        {"date": list(range(CHAIN_ROWS)), "nav": navs, "split_index": [0] * CHAIN_ROWS}
    ).to_csv(run_dir / "chain_nav_wf_corpus.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        [
            {
                "split_index": 0,
                "test_start": days[0],
                "test_end": days[-1],
                "KEY_Top20_hit_rate": 0.5 + (index % 9) * 0.01,
                "KEY_Top30_hit_rate": 0.4 + (index % 7) * 0.01,
            }
        ]
    ).to_csv(run_dir / "walk_forward_summary_wf_corpus.csv", index=False, lineterminator="\n")
    _json_new(
        run_dir / "data_state_wf_corpus.json",
        {"data_state_id": f"wf_ds_{index:04d}", "git_commit": "corpus", "git_dirty": False},
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """新文件写 YAML（已存在即跳过）。"""
    if path.exists():
        return
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )


def _build_paper(root: Path, count: int) -> None:
    """production 纸面根（配置 + NAV + 成交 + 逐日 run 记录）。"""
    paper = root / "paper"
    days = _business_days("20260101", count)
    initial = 500_000.0
    _write_yaml(
        paper / "config.yaml",
        {"engine": "v2", "initial_capital": initial, "account_start_date": days[0]},
    )
    nav_rows = []
    for offset, day in enumerate(days):
        total = initial * (1.0 + 0.001 * (offset + 1))
        nav_rows.append(
            {
                "trade_date": day,
                "cash": total * 0.3,
                "position_value": total * 0.7,
                "total_value": total,
                "nav": total / initial,
            }
        )
    _write_parquet(paper / "nav" / "nav.parquet", pd.DataFrame(nav_rows))
    _write_engine_state(paper, days, initial)
    trade_rows = [
        {
            "trade_date": days[1],
            "ts_code": "600000.SH",
            "action": "buy",
            "shares": 1000,
            "price": 10.0,
            "amount": 10_000.0,
            "commission": 3.0,
            "stamp_tax": 0.0,
            "slippage": 2.0,
            "total_cost": 5.0,
            "reason": "signal",
        },
        {
            "trade_date": days[2],
            "ts_code": "600000.SH",
            "action": "sell",
            "shares": 1000,
            "price": 10.2,
            "amount": 10_200.0,
            "commission": 3.0,
            "stamp_tax": 5.1,
            "slippage": 2.0,
            "total_cost": 10.1,
            "reason": "hold_period",
        },
    ]
    _write_parquet(paper / "trades" / "trades.parquet", pd.DataFrame(trade_rows))
    for day in days:
        _json_new(
            paper / "runs" / f"t1_{day}.json",
            {
                "trade_date": day,
                "run_kind": "t1",
                "engine": "v2",
                "buy_price_type": "close",
                "sell_price_type": "open",
                "instructions_count": 1,
                "pending_buys_count": 0,
                "orders_count": 1,
                "fills_count": 1,
                "timestamp": "2026-01-01T00:00:00",
            },
        )


def _write_engine_state(paper: Path, days: Sequence[str], initial: float) -> None:
    """写 P4 权威代（paper_generation + portfolio_values），production 数值的前置门（R3-01）。"""
    path = paper / "state" / "engine_state.bin"
    if path.exists():
        return
    from src.lazybull.v2.core.accounting.engine_state import (
        EngineState,
        dumps_blob,
        state_to_blob,
    )

    values = []
    for offset, day in enumerate(days):
        total = initial * (1.0 + 0.001 * (offset + 1))
        values.append(
            {
                # 权威 portfolio_values 的键名与作者核对口径一致（capital/market_value/portfolio_value）
                "date": day,
                "capital": total * 0.3,
                "market_value": total * 0.7,
                "portfolio_value": total,
            }
        )
    state = EngineState(
        current_capital=initial,
        positions={},
        pending_signals={},
        pending_stop_loss_sells={},
        pending_condition_sells={},
        _cycle_anchor_idx=0,
        portfolio_values=values,
        trades=[],
        execution_attribution_records=[],
        holdings_snapshots=[],
        unfilled_slots={},
        completion_stats={},
        _normalize_log_count=0,
        _last_ranked_candidates=[],
        _last_signal_date=None,
        _last_rebalance_nav=None,
        _signal_dates={},
        ledger={"applied_keys": [], "last_fill_date": None},
    )
    blob = state_to_blob(state)
    blob["paper_generation"] = {"trade_date": days[-1], "run_kind": "t1"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps_blob(blob))


def build(root: Path, profile: str = "standard", *, upto: int | None = None) -> dict[str, Any]:
    """构建（或增量补齐到 ``upto`` 个）语料；已存在文件一律不覆盖。"""
    if profile not in PROFILE_COUNTS:
        raise ValueError(f"未知语料档：{profile!r}")
    counts = dict(PROFILE_COUNTS[profile])
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    runs_limit = counts["runs"] if upto is None else min(int(upto), counts["runs"])
    heavy_index = runs_limit - 1
    for index in range(runs_limit):
        _build_run(base, index, _load_for(index, heavy_index, profile))
    for index in range(counts["legacy"]):
        _build_legacy(base, index)
    _build_paper(base, counts["paper"])
    return {
        "schema": CORPUS_SCHEMA,
        "profile": profile,
        "root": ".",
        "counts": {**counts, "runs_materialized": runs_limit},
        "roots": {"runs": "runs", "legacy": "legacy", "paper": "paper"},
        "content_sha256": content_digest(base),
    }


def content_digest(root: Path) -> str:
    """语料内容身份（全部文件相对路径 + 字节 SHA，按路径排序；不含 mtime）。"""
    base = Path(root)
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts:
            continue
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """命令行入口（``python scripts/v2_p6/build_t4_corpus.py --root ... --profile stress``）。"""
    import argparse

    parser = argparse.ArgumentParser(description="P6 T4 趋势语料构建")
    parser.add_argument("--root", required=True)
    parser.add_argument("--profile", default="standard", choices=sorted(PROFILE_COUNTS))
    parser.add_argument("--upto", type=int, default=None, help="只物化前 N 个 runs（增量语料）")
    args = parser.parse_args(argv)
    payload = build(Path(args.root), args.profile, upto=args.upto)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""P6 T2 合成驾驶舱语料（标准 / 压力，设计 §7.3 dashboard 行）。

定位：**渲染规模语料**。构造与生产同构的 ``data/paper`` 形态只读根 + 质量长表，规模取
``contracts.CORPUS_PROFILES[<profile>]["dashboard"]``（持仓/指令/NAV 行/质量异常行），
用于体积门、打开门与生成门；不引用任何真实生产数据、不写 ``data/paper``。

用法::

    python scripts/v2_p6/build_t2_corpus.py --out temp/p6_t2_corpus --profile all

输出结构（``--out`` 必须是新目录）::

    <out>/<profile>/paper/...           # 合成纸面根（权威 blob + 账户 + NAV + 指令 + pending）
    <out>/<profile>/quality/latest_metrics.parquet

确定性：固定种子 + 固定日期生成，双跑逐文件 SHA 一致（``corpus_manifest.json`` 记录）。
"""

from __future__ import annotations

import argparse
import json
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.core.accounting.engine_state import (  # noqa: E402
    EngineState,
    dumps_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.ledger import Ledger  # noqa: E402
from src.lazybull.v2.core.execution.live_schedule import LiveAnchorState  # noqa: E402
from src.lazybull.v2.hosts.paper.instruction_projection import (  # noqa: E402
    INSTRUCTION_COLUMNS,
)
from src.lazybull.v2.hosts.paper.state_store import (  # noqa: E402
    NAV_COLUMNS,
    TRADES_COLUMNS,
    build_account_projection,
    build_rebalance_state_projection,
)
from src.lazybull.v2.reporting import contracts, validation  # noqa: E402

__all__ = ["CaseError", "build_case", "main"]

#: 固定语料种子（变更即新语料）
SEED = 20261106
#: 合成当前权威日（与真实生产日期无关）
TRADE_DATE = "20260603"
#: 未来执行日
FUTURE_DATE = "20260604"
#: 合成初始资金
INITIAL_CAPITAL = 500_000.0
#: 质量长表列（与 ``quality.project_quality`` 读取的字段一致）
QUALITY_COLUMNS = ("layer", "dataset", "metric", "partition", "value", "status")


class CaseError(RuntimeError):
    """语料输出目录已存在或 profile 未知。"""


def _business_days(count: int, *, end: date) -> list[str]:
    """回溯生成 ``count`` 个工作日（YYYYMMDD，升序；末日为 ``end``）。"""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return [day.strftime("%Y%m%d") for day in reversed(days)]


def _nav_series(days: Sequence[str]) -> list[dict[str, Any]]:
    """NAV/权威共用的三原值序列（确定性正弦+线性漂移，无随机数）。"""
    import math

    rows: list[dict[str, Any]] = []
    total = INITIAL_CAPITAL
    for index, day in enumerate(days):
        drift = 0.0004 * index + 0.02 * math.sin(index / 7.0)
        total = INITIAL_CAPITAL * (1.0 + drift)
        cash = round(total * 0.08, 6)
        rows.append(
            {
                "trade_date": day,
                "total_value": round(total, 6),
                "cash": cash,
                "position_value": round(total - cash, 6),
            }
        )
    return rows


def _positions(count: int) -> dict[str, dict[str, Any]]:
    """合成持仓（v2 键：shares/buy_date/buy_trade_price/buy_pnl_price/buy_cost_cash）。"""
    positions: dict[str, dict[str, Any]] = {}
    for index in range(count):
        code = f"{600000 + index:06d}.SH"
        price = round(5.0 + (index % 60) * 0.7, 2)
        shares = 100 * (1 + index % 40)
        positions[code] = {
            "shares": shares,
            "buy_date": "20260506",
            "signal_date": "20260505",
            "buy_trade_price": price,
            "buy_pnl_price": round(price * (0.96 + (index % 9) * 0.01), 2),
            "buy_cost_cash": round(shares * price + 5.0, 6),
        }
    return positions


def _instruction_rows(count: int) -> list[dict[str, Any]]:
    """未来指令行（12 列契约；``source_date`` = 当前权威日）。"""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "ts_code": f"{600000 + index:06d}.SH",
                "action": "buy" if index % 3 else "sell",
                "shares": 100 * (1 + index % 20),
                "price_type": "close",
                "reason": "调仓计划",
                "source_date": TRADE_DATE,
                "target_weight": round(0.01 + (index % 10) * 0.005, 6),
                "original_signal_date": TRADE_DATE,
                "desired_position_count": count,
                "retry_attempt": 0,
                "replacement_slot_code": "",
                "keep_buy_date": False,
            }
        )
    return rows


def _pending_rows(count: int, *, offset: int) -> list[dict[str, Any]]:
    """pending 队列行（买卖各半，原值）。"""
    return [
        {
            "ts_code": f"{600000 + offset + index:06d}.SH",
            "action": "buy" if offset == 0 else "sell",
            "shares": 100 * (1 + index % 15),
            "price": round(8.0 + (index % 30) * 0.5, 2),
            "reason": "调仓计划",
        }
        for index in range(count)
    ]


def _quality_rows(abnormal: int) -> list[dict[str, Any]]:
    """质量长表：``abnormal`` 条异常指标行 + 每数据集三项摘要行。"""
    datasets = ("daily", "adj_factor", "features", "labels")
    rows: list[dict[str, Any]] = []
    for index in range(abnormal):
        rows.append(
            {
                "layer": "clean" if index % 2 else "raw",
                "dataset": datasets[index % len(datasets)],
                "metric": f"metric_{index % 13}",
                "partition": f"2026{(index % 12) + 1:02d}{index % 28 + 1:02d}",
                "value": str(index % 97),
                "status": "error" if index % 2 else "warning",
            }
        )
    for index, dataset in enumerate(datasets):
        rows.extend(
            [
                {
                    "layer": "clean",
                    "dataset": dataset,
                    "metric": "partition_count",
                    "partition": "",
                    "value": str(100 + index),
                    "status": "ok",
                },
                {
                    "layer": "clean",
                    "dataset": dataset,
                    "metric": "latest_partition",
                    "partition": "",
                    "value": f"2026{index + 1:02d}28",
                    "status": "ok",
                },
                {
                    "layer": "clean",
                    "dataset": dataset,
                    "metric": "sync_watermark",
                    "partition": "",
                    "value": TRADE_DATE,
                    "status": "ok",
                },
            ]
        )
    return rows


def _engine_blob(*, positions: dict[str, dict[str, Any]], nav: Sequence[dict[str, Any]]) -> bytes:
    """权威 blob（``EngineState`` 正规往返 + 顶层 ``paper_generation``）。"""
    state = EngineState(
        current_capital=float(nav[-1]["cash"]),
        positions={key: dict(value) for key, value in positions.items()},
        pending_signals={},
        pending_stop_loss_sells={},
        pending_condition_sells={},
        _cycle_anchor_idx=0,
        portfolio_values=[
            {
                "date": pd.Timestamp(row["trade_date"]),
                "portfolio_value": row["total_value"],
                "capital": row["cash"],
                "market_value": row["position_value"],
            }
            for row in nav
        ],
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
        pending_order_manager=None,
        stop_loss_monitor=None,
        ledger=Ledger().export_state(),
        live_anchor_state=None,
    )
    blob = state_to_blob(state)
    blob["paper_generation"] = {"trade_date": TRADE_DATE, "run_kind": "t1"}
    return dumps_blob(blob)


@dataclass(frozen=True)
class _Scale:
    positions: int
    instructions: int
    nav_rows: int
    quality_abnormal_rows: int


def _write_paper_core(paper: Path, scale: _Scale, nav: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """写纸面根的核心载体（权威 blob / 配置 / 账户 / NAV / 成交 / record / 调仓）。"""
    positions = _positions(scale.positions)
    (paper / "state" / "engine_state.bin").write_bytes(
        _engine_blob(positions=positions, nav=nav)
    )
    config = {
        "engine": "v2",
        "initial_capital": INITIAL_CAPITAL,
        "model_version": 24001,
        "model_version_b": None,
        "ensemble_weight_a": 1.0,
        "account_start_date": nav[0]["trade_date"],
    }
    (paper / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    account = build_account_projection(float(nav[-1]["cash"]), positions, TRADE_DATE)
    (paper / "state" / "account.json").write_text(
        json.dumps(account, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(nav, columns=list(NAV_COLUMNS)).to_parquet(
        paper / "nav" / "nav.parquet", index=False
    )
    pd.DataFrame([], columns=list(TRADES_COLUMNS)).to_parquet(
        paper / "trades" / "trades.parquet", index=False
    )
    record = {
        "trade_date": TRADE_DATE,
        "engine": "v2",
        "run_kind": "t1",
        "t0_status": "success",
        "buy_price_type": "close",
        "sell_price_type": "open",
        "instructions_count": scale.instructions,
        "orders_count": scale.instructions,
        "fills_count": 0,
        "pending_buys_count": scale.instructions // 2,
        "pending_sells_count": scale.instructions - scale.instructions // 2,
        "nav": nav[-1]["total_value"],
        "timestamp": "2026-06-03T15:30:00",
    }
    (paper / "runs" / f"t1_{TRADE_DATE}.json").write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    anchor = LiveAnchorState(
        last_rebalance_date=None,
        last_scheduled_rebalance_date=None,
        anchor_date=TRADE_DATE,
        rebalance_freq=20,
        stagger_tranches=2,
    )
    (paper / "runs" / "rebalance_state.json").write_text(
        json.dumps(build_rebalance_state_projection(anchor), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return {"positions": len(positions), "account": account}


def _write_paper_queues(paper: Path, scale: _Scale) -> None:
    """写 pending 两路与未来指令文件（规模由 profile 决定）。"""
    half = scale.instructions // 2
    (paper / "pending_buys" / "pending_buys.json").write_text(
        json.dumps(_pending_rows(half, offset=0), ensure_ascii=False), encoding="utf-8"
    )
    (paper / "pending_sells" / "pending_sells.json").write_text(
        json.dumps(_pending_rows(scale.instructions - half, offset=5000), ensure_ascii=False),
        encoding="utf-8",
    )
    instructions = pd.DataFrame(
        _instruction_rows(scale.instructions), columns=list(INSTRUCTION_COLUMNS)
    )
    instructions.to_parquet(paper / "instructions" / f"{FUTURE_DATE}.parquet", index=False)


def _write_paper(paper: Path, scale: _Scale) -> dict[str, Any]:
    """写合成纸面根（全部目录一次建齐，避免读取期新增目录）。"""
    for relative in (
        "state",
        "nav",
        "trades",
        "runs",
        "pending_buys",
        "pending_sells",
        "instructions",
    ):
        (paper / relative).mkdir(parents=True, exist_ok=True)
    days = _business_days(scale.nav_rows, end=date(2026, 6, 3))
    nav = _nav_series(days)
    core = _write_paper_core(paper, scale, nav)
    _write_paper_queues(paper, scale)
    return {
        "trade_date": TRADE_DATE,
        "future_date": FUTURE_DATE,
        "nav_days": len(days),
        "positions": core["positions"],
        "instructions": scale.instructions,
        "account": core["account"],
    }


def _write_quality(quality: Path, scale: _Scale) -> int:
    """写质量长表 parquet；返回异常指标行数（与 profile 断言）。"""
    quality.mkdir(parents=True, exist_ok=True)
    rows = _quality_rows(scale.quality_abnormal_rows)
    pd.DataFrame(rows, columns=list(QUALITY_COLUMNS)).to_parquet(
        quality / "latest_metrics.parquet", index=False
    )
    return sum(1 for row in rows if row["status"] in ("error", "warning"))


def build_case(root: Path | str, profile: str = "standard") -> dict[str, Any]:
    """在**新目录** ``root`` 下构造指定 profile 的合成驾驶舱语料。"""
    if profile not in ("standard", "stress"):
        raise CaseError(f"未知语料 profile：{profile!r}（只支持 standard/stress）")
    target = Path(root)
    if target.exists():
        raise CaseError(f"语料输出目录已存在，拒绝覆盖：{target}")
    counts = contracts.CORPUS_PROFILES[profile]["dashboard"]
    scale = _Scale(
        positions=int(counts["positions"]),
        instructions=int(counts["instructions"]),
        nav_rows=int(counts["nav_rows"]),
        quality_abnormal_rows=int(counts["quality_abnormal_rows"]),
    )
    target.mkdir(parents=True, exist_ok=False)
    paper = target / "paper"
    quality = target / "quality"
    paper_info = _write_paper(paper, scale)
    abnormal = _write_quality(quality, scale)
    payload = {
        "schema": "p6_t2_corpus/v1",
        "profile": profile,
        "seed": SEED,
        "scale": {
            "positions": scale.positions,
            "instructions": scale.instructions,
            "nav_rows": scale.nav_rows,
            "quality_abnormal_rows": scale.quality_abnormal_rows,
        },
        "paper": str(paper),
        "quality_root": str(quality),
        "quality_abnormal_rows_written": abnormal,
        "paper_info": {
            key: value for key, value in paper_info.items() if key != "account"
        },
    }
    return payload


def _digest_case(root: Path) -> dict[str, str]:
    """逐文件 SHA（双跑一致性证据）。"""
    return {
        path.relative_to(root).as_posix(): validation.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6 T2 合成驾驶舱语料（标准/压力）")
    parser.add_argument("--out", required=True, help="新的语料输出目录（已存在即失败）")
    parser.add_argument("--profile", choices=["all", "standard", "stress"], default="all")
    args = parser.parse_args(argv)
    try:
        parent = Path(args.out)
        if parent.exists():
            raise CaseError(f"语料输出目录已存在，拒绝覆盖：{parent}")
        parent.mkdir(parents=True, exist_ok=False)
        profiles = ["standard", "stress"] if args.profile == "all" else [args.profile]
        summary: dict[str, Any] = {"schema": "p6_t2_corpus_manifest/v1", "profiles": {}}
        for name in profiles:
            case_root = parent / name
            payload = build_case(case_root, name)
            summary["profiles"][name] = {
                **payload,
                "files": _digest_case(case_root),
            }
        validation.write_json_new(parent / "corpus_manifest.json", summary)
    except CaseError as exc:
        print(f"[P6-T2] 语料生成失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(args.out), "profiles": profiles}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

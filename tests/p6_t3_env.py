# -*- coding: utf-8 -*-
"""P6 T3 合成夹具：规范化 runs 批次与旧 WF 运行（**不引用任何真实生产数据**）。

两类目录都按各自契约生成，供加载/指标/CSV/稳定性用例复用：

- ``write_runs_batch``：``batch_meta.json`` + ``summary.parquet`` + ``chain_nav.parquet`` +
  ``folds/split<N>/{trades,daily,attribution,...}.parquet``，满足 ``runs_loader`` §9 全部
  硬校验（含折界衔接、daily/trades 交叉对账、config_fingerprint 重算）；
- ``write_legacy_run``：``chain_nav_*.csv`` + ``walk_forward_summary_*.csv`` +
  ``data_state_*.json``，满足 ``scripts.compare.fold_subset.load_run`` 白名单子集。

数值口径：链式 NAV 由固定步长收益率累乘（合成，不冒充真实收益），逐折衔接严格相等；
所有随机性来自显式 ``seed``（可复现），不写真实数据根。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.compare.fold_subset import RunArtifacts
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_schema import (
    ATTR_CONTRACT_COLS,
    CHAIN_NAV_CONTRACT_COLS,
    DAILY_CONTRACT_COLS,
    TOPK_CONTRACT_COLS,
    TRADES_CONTRACT_COLS,
)
from src.lazybull.v2.reporting import experiment_sources
from src.lazybull.v2.reporting.snapshot import capture_stable_snapshot

__all__ = [
    "DEFAULT_STRATEGY",
    "arm_from_folds",
    "arm_from_nav",
    "build_arm_data",
    "fold_nav_series",
    "load_arms",
    "make_arm_specs",
    "nav_rows",
    "standard_projection",
    "summary_for",
    "write_legacy_run",
    "write_runs_batch",
]

#: 默认策略参数（稳定性用例可整体替换/增删）
DEFAULT_STRATEGY: dict[str, Any] = {
    "algorithm": "xgboost",
    "learning_rate": 0.03,
    "max_depth": 5,
    "split_count": 3,
    "final_date": "20260105",
    "stock_domain": "all_a",
    "enable_fundamental": True,
}
_BASE_DAY = date(2025, 1, 2)


def _distribute(total: int, folds: int) -> list[int]:
    """把 ``total`` 行尽量均分到 ``folds`` 折（前 remainder 折各多 1 行）。"""
    base, remainder = divmod(int(total), int(folds))
    return [base + (1 if index < remainder else 0) for index in range(folds)]


def fold_nav_series(
    *,
    folds: int = 3,
    rows_per_fold: int = 8,
    total_rows: int | None = None,
    step: float = 0.004,
    start: float = 1.0,
    dip: float = 0.0,
) -> tuple[list[tuple[int, str, float]], list[dict[str, Any]]]:
    """构造跨折严格衔接的 (split, date, nav) 序列与逐折窗口。

    折界重复点语义：后折首行 = 前折末行（日期与 NAV 都相等），因此
    ``trading_days = N - S`` 成立（折界不重复计收益区间）。

    ``total_rows`` 给出时按总行数均分（每折至少 2 行，折界共享点计入 N）；
    否则每折固定 ``rows_per_fold`` 行。
    """
    if total_rows is not None:
        per_fold = _distribute(int(total_rows) + folds - 1, folds)
        if min(per_fold) < 2:
            raise ValueError(f"total_rows={total_rows} 不足以让每折至少 2 行：{per_fold}")
    else:
        if rows_per_fold < 2:
            raise ValueError("rows_per_fold 至少为 2（折内需一次收益区间）")
        per_fold = [rows_per_fold] * folds
    total_days = sum(per_fold) - (folds - 1)
    days = [(_BASE_DAY + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(total_days)]
    navs = [float(start)]
    dip_index = total_days // 2 if dip else -1
    for index in range(1, total_days):
        navs.append(round(navs[-1] * (1.0 + step), 10))
    if 0 < dip_index < total_days:
        # 中段回撤（让两臂的 MaxDD 不同；仍保持折界衔接与 summary 对账）
        for index in range(dip_index, total_days):
            navs[index] = round(navs[index] * (1.0 - float(dip)), 10)
    rows: list[tuple[int, str, float]] = []
    windows: list[dict[str, Any]] = []
    offset = 0
    for split, count in enumerate(per_fold):
        fold_rows = [
            (split, days[offset + index], navs[offset + index]) for index in range(count)
        ]
        rows.extend(fold_rows)
        windows.append(
            {
                "split_index": split,
                "test_start": fold_rows[0][1],
                "test_end": fold_rows[-1][1],
                "bt_total_return": fold_rows[-1][2] / fold_rows[0][2] - 1.0,
            }
        )
        offset += count - 1
    return rows, windows


def _trades_frame(
    rows: Sequence[tuple[int, str, float]],
    *,
    run_id: str,
    split_models: Mapping[int, str],
    buy_counts: Mapping[tuple[int, str], int] | None = None,
) -> pd.DataFrame:
    """逐折成交（默认每个 (折,日期) 一买一卖；给出 ``buy_counts`` 时按计数只买）。

    两种形态都满足 §9.1（action 闭集、卖出行 buy_date 非空且不晚于成交日）与 §9.4
    （同日同 (action, ts_code) 唯一：多笔买入用不同代码）。
    """
    records: list[dict[str, Any]] = []
    for split, day, _nav in rows:
        if buy_counts is not None:
            actions = [("buy", index) for index in range(int(buy_counts.get((split, day), 0)))]
        else:
            actions = [("buy", 0), ("sell", 0)]
        for action, index in actions:
            code = f"6000{index:02d}.SH" if action == "buy" else "600001.SH"
            price = 10.0 if action == "buy" else 11.0
            records.append(
                {
                    "wf_run_id": run_id,
                    "split_index": split,
                    "model_version": split_models.get(split),
                    "trade_date": day,
                    "signal_date": day,
                    "ts_code": code,
                    "action": action,
                    "price": price,
                    "shares": 100,
                    "amount": price * 100,
                    "cost": round(price * 100 * 0.0003, 4),
                    "buy_date": day,
                    "buy_price": price,
                    "buy_pnl_price": price,
                    "sell_pnl_price": price if action == "sell" else None,
                    "pnl_profit_amount": 0.0,
                    "pnl_profit_pct": 0.0,
                    "sell_type": "due" if action == "sell" else None,
                    "sell_timing": "close" if action == "sell" else None,
                    "sell_reason": "holding_period" if action == "sell" else None,
                    "trigger_type": "signal",
                    "buy_type": "normal",
                    "buy_reason": "topn",
                    "lot_id": f"lot_{split}_{day}",
                    "tranche_idx": 0,
                }
            )
    return pd.DataFrame(records, columns=TRADES_CONTRACT_COLS)


def _daily_frame(
    rows: Sequence[tuple[int, str, float]],
    *,
    turnover_base: float = 1000.0,
    buy_counts: Mapping[tuple[int, str], int] | None = None,
) -> pd.DataFrame:
    """逐日净值（total_value 与 nav 同比，满足 §9.8 交叉对账）。"""
    records: list[dict[str, Any]] = []
    for split, day, nav in rows:
        total = nav * 1_000_000.0
        if buy_counts is None:
            n_buys, n_sells = 1, 1
        else:
            n_buys, n_sells = int(buy_counts.get((split, day), 0)), 0
        records.append(
            {
                "trade_date": day,
                "total_value": total,
                "cash": total * 0.1,
                "market_value": total * 0.9,
                "nav": nav,
                "daily_return": 0.0,
                "n_positions": 2,
                "n_buys": n_buys,
                "n_sells": n_sells,
                "turnover_amount": turnover_base + split * 1000.0,
                "exposure_lambda": 1.0,
            }
        )
    return pd.DataFrame(records, columns=DAILY_CONTRACT_COLS)


def _split_days(rows: Sequence[tuple[int, str, float]]) -> dict[int, list[str]]:
    """按折分组的日期列表（用于独立行数的归因/TopK 生成）。"""
    grouped: dict[int, list[str]] = {}
    for split, day, _nav in rows:
        grouped.setdefault(split, []).append(day)
    return grouped


def _attribution_frame(
    rows: Sequence[tuple[int, str, float]],
    *,
    run_id: str,
    split_models: Mapping[int, str],
    total: int | None = None,
) -> pd.DataFrame:
    """执行归因（19 列原值；含非成功状态，验证原值守恒）。

    ``total`` 给出时按折均分该行数（压力语料 10,000 行不再与链行数耦合）；否则一链行一行。
    """
    records: list[dict[str, Any]] = []
    if total is None:
        items = [(split, day) for split, day, _nav in rows]
    else:
        grouped = _split_days(rows)
        items = []
        for split, count in zip(sorted(grouped), _distribute(total, len(grouped))):
            days = grouped[split]
            items.extend((split, days[index % len(days)]) for index in range(count))
    for index, (split, day) in enumerate(items):
        status = "filled" if index % 3 else ("rejected" if index % 2 else "skipped")
        records.append(
            {
                "wf_run_id": run_id,
                "split_index": split,
                "model_version": split_models.get(split),
                "signal_date": day,
                "ranking_date": day,
                "execution_date": day,
                "execution_stage": "open",
                "tranche_idx": 0,
                "planned_ts_code": f"6000{index % 50:02d}.SH",
                "actual_ts_code": f"6000{index % 50:02d}.SH" if status == "filled" else None,
                "planned_rank": index % 10,
                "actual_rank": index % 10 if status == "filled" else None,
                "pred_score": 0.01 * (index % 7),
                "target_weight": 0.05,
                "status": status,
                "reason": "signal_rank" if status == "filled" else "insufficient_cash",
                "buy_price": 10.0 + index * 0.01,
                "signal_price": 9.9 + index * 0.01,
                "signal_to_buy_return": 0.01,
            }
        )
    return pd.DataFrame(records, columns=ATTR_CONTRACT_COLS)


def _topk_frame(
    rows: Sequence[tuple[int, str, float]],
    *,
    run_id: str,
    split_models: Mapping[int, str],
    total: int | None = None,
) -> pd.DataFrame:
    """TopK 明细（只用于“必须被忽略”的负例：P6 不得据此私算正日占比）。"""
    records: list[dict[str, Any]] = []
    if total is None:
        items = [(split, day) for split, day, _nav in rows]
    else:
        grouped = _split_days(rows)
        items = []
        for split, count in zip(sorted(grouped), _distribute(total, len(grouped))):
            days = grouped[split]
            items.extend((split, days[index % len(days)]) for index in range(count))
    for index, (split, day) in enumerate(items):
        records.append(
            {
                "wf_run_id": run_id,
                "split_index": split,
                "test_start": day,
                "test_end": day,
                "model_version": split_models.get(split),
                "trade_date": day,
                "topk": 20,
                "rank": index % 20,
                "ts_code": f"6000{index % 50:02d}.SH",
                "pred_score": 0.01 * (index % 9),
                "true_return": 0.001 * (index % 5),
                "score_column": "final_score",
                "ml_score": 0.01,
                "risk_score": 0.0,
                "final_score": 0.01,
            }
        )
    return pd.DataFrame(records, columns=TOPK_CONTRACT_COLS)


def write_runs_batch(
    root: Path | str,
    *,
    batch_id: str,
    config: Mapping[str, Any] | None = None,
    folds: int = 3,
    rows_per_fold: int = 8,
    total_chain_rows: int | None = None,
    trades_rows: int | None = None,
    attribution_rows: int | None = None,
    topk_rows: int | None = None,
    nav_step: float = 0.004,
    nav_dip: float = 0.0,
    data_state_id: str = "ds000001",
    host_mode: str = "backtest",
    git_commit: str = "abc1234",
    git_dirty: bool = False,
    with_daily: bool = True,
    with_attribution: bool = True,
    with_topk: bool = False,
    with_trades: bool = True,
) -> Path:
    """写出一个符合 runs 契约的合成批次，返回批次目录。

    ``total_chain_rows``/``trades_rows``/``attribution_rows``/``topk_rows`` 用于对齐
    §7.3 语料规模（标准/压力）；不给时按 ``folds × rows_per_fold`` 与“每链行一行”生成。
    """
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    params = dict(DEFAULT_STRATEGY if config is None else config)
    rows, windows = fold_nav_series(
        folds=folds,
        rows_per_fold=rows_per_fold,
        total_rows=total_chain_rows,
        step=nav_step,
        dip=nav_dip,
    )
    buy_counts: dict[tuple[int, str], int] | None = None
    if trades_rows is not None:
        days = [(split, day) for split, day, _nav in rows]
        counts = [int(item) for item in _distribute(int(trades_rows), len(days))]
        buy_counts = dict(zip(days, counts))
    split_models = {item["split_index"]: str(1000 + item["split_index"]) for item in windows}
    meta = {
        "schema_version": 1,
        "batch_id": batch_id,
        "host_mode": host_mode,
        "source": "p6_t3_synthetic",
        "code_state": {"git_commit": git_commit, "git_dirty": git_dirty},
        "data_state": {
            "data_state_id": data_state_id,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "raw_latest_partitions": {"daily": "2026-07-31"},
            "features_cs_train_latest": "20260702",
            "dividend_coverage": {"data": 10, "empty": 0, "failed": 0, "pending": 0, "total": 10},
        },
        "config": params,
        "config_fingerprint": fingerprint(params),
    }
    (target / "batch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    chain = pd.DataFrame(
        {
            "date": [item[1] for item in rows],
            "nav": [item[2] for item in rows],
            "split_index": [item[0] for item in rows],
        },
        columns=CHAIN_NAV_CONTRACT_COLS,
    )
    chain.to_parquet(target / "chain_nav.parquet", index=False)
    summary = pd.DataFrame(
        {
            "batch_id": batch_id,
            "split_index": [item["split_index"] for item in windows],
            "test_start": [item["test_start"] for item in windows],
            "test_end": [item["test_end"] for item in windows],
            "bt_total_return": [item["bt_total_return"] for item in windows],
            "bt_annual_return": [0.12 for _ in windows],
            "bt_max_drawdown": [-0.08 for _ in windows],
            "bt_sharpe": [1.2 for _ in windows],
            "bt_win_rate": [0.55 for _ in windows],
            "data_state_id": [data_state_id for _ in windows],
            "model_version": [split_models[item["split_index"]] for item in windows],
            "wf_run_id": [batch_id for _ in windows],
        }
    )
    summary.to_parquet(target / "summary.parquet", index=False)
    folds_dir = target / "folds"
    folds_dir.mkdir(exist_ok=True)
    for item in windows:
        split = int(item["split_index"])
        fold_dir = folds_dir / f"split{split}"
        fold_dir.mkdir(exist_ok=True)
        fold_rows = [row for row in rows if row[0] == split]
        if with_trades:
            _trades_frame(
                fold_rows, run_id=batch_id, split_models=split_models, buy_counts=buy_counts
            ).to_parquet(fold_dir / "trades.parquet", index=False)
        if with_daily:
            _daily_frame(fold_rows, buy_counts=buy_counts).to_parquet(
                fold_dir / "daily.parquet", index=False
            )
        if with_attribution:
            fold_total = None if attribution_rows is None else _distribute(attribution_rows, folds)[split]
            _attribution_frame(
                fold_rows,
                run_id=batch_id,
                split_models=split_models,
                total=fold_total,
            ).to_parquet(fold_dir / "attribution.parquet", index=False)
        if with_topk:
            fold_topk = None if topk_rows is None else _distribute(topk_rows, folds)[split]
            _topk_frame(
                fold_rows, run_id=batch_id, split_models=split_models, total=fold_topk
            ).to_parquet(fold_dir / "topk_detail.parquet", index=False)
    return target


def write_legacy_run(
    root: Path | str,
    *,
    run_id: str,
    batch_run_id: str | None = None,
    batch_period_label: str | None = None,
    params: Mapping[str, Any] | None = None,
    folds: int = 3,
    rows_per_fold: int = 8,
    data_state_id: str = "ds000001",
    git_commit: str = "abc1234",
    git_dirty: bool = False,
    topk: bool = True,
    model_base: int = 1000,
) -> Path:
    """写出一个旧 WF 合成运行目录（chain/summary/data_state 三件）。"""
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    material = dict(DEFAULT_STRATEGY if params is None else params)
    rows, windows = fold_nav_series(folds=folds, rows_per_fold=rows_per_fold)
    chain = pd.DataFrame(
        {
            "date": [index for index, _row in enumerate(rows)],
            "nav": [item[2] for item in rows],
            "split_index": [item[0] for item in rows],
        }
    )
    chain.to_csv(target / f"chain_nav_{run_id}.csv", index=False)
    summary = pd.DataFrame(
        {
            "split_index": [item["split_index"] for item in windows],
            "test_start": [item["test_start"] for item in windows],
            "test_end": [item["test_end"] for item in windows],
            "bt_total_return": [item["bt_total_return"] for item in windows],
            "bt_annual_return": [0.1 for _ in windows],
            "bt_max_drawdown": [-0.07 for _ in windows],
            "bt_sharpe": [1.1 for _ in windows],
            "bt_win_rate": [0.5 for _ in windows],
            "model_version": [model_base + int(item["split_index"]) for item in windows],
            "wf_run_id": [run_id for _ in windows],
            "batch_run_id": [batch_run_id or run_id for _ in windows],
            "batch_period_label": [
                batch_period_label or f"P{item['split_index']}" for item in windows
            ],
            "wf_start_date": [item["test_start"] for item in windows],
            "wf_end_date": [item["test_end"] for item in windows],
            "algorithm": [material.get("algorithm", "xgboost") for _ in windows],
            "learning_rate": [material.get("learning_rate", 0.03) for _ in windows],
            "max_depth": [material.get("max_depth", 5) for _ in windows],
            "split_count": [folds for _ in windows],
            "final_date": [material.get("final_date", "20260105") for _ in windows],
            "stock_domain": [material.get("stock_domain", "all_a") for _ in windows],
        }
    )
    if topk:
        summary["KEY_Top20_hit_rate"] = [0.55 for _ in windows]
        summary["KEY_Top30_hit_rate"] = [0.6 for _ in windows]
    summary.to_csv(target / f"walk_forward_summary_{run_id}.csv", index=False)
    state = {
        "data_state_id": data_state_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "raw_latest_partitions": {"daily": "2026-07-31"},
        "features_cs_train_latest": "20260702",
        "dividend_coverage": {"data": 10, "empty": 0, "failed": 0, "pending": 0, "total": 10},
        "batch_run_id": batch_run_id or run_id,
    }
    (target / f"data_state_{run_id}.json").write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def make_arm_specs(mapping: Sequence[tuple[str, str, Path]]) -> list[experiment_sources.ArmSpec]:
    """由 ``(arm_id, kind, path)`` 三元组构造臂规格（root_id 按顺序）。"""
    return [
        experiment_sources.ArmSpec(
            arm_id=arm_id, source_kind=kind, path=Path(path), root_id=f"arm_{index}"
        )
        for index, (arm_id, kind, path) in enumerate(mapping)
    ]


def load_arms(
    tmp_path: Path, mapping: Sequence[tuple[str, str, Path]]
) -> list[experiment_sources.ArmData]:
    """走一次真实快照 + 加载（与生成链同一路径，不绕过快照）。"""
    specs = make_arm_specs(mapping)
    roots = {spec.root_id: spec.path for spec in specs}
    plans: dict[str, Any] = {}

    def reader(snapshot) -> None:
        plans.clear()
        for spec in specs:
            plans[spec.arm_id] = experiment_sources.capture_arm_files(snapshot, spec)

    snapshot, _audit = capture_stable_snapshot(
        roots=roots,
        # 每次加载用独立 staging 根：同一测试内多次 load_arms 不会复用尝试目录
        staging_root=tmp_path / f"staging_{uuid.uuid4().hex[:8]}",
        attempt_ids=["r1", "r2", "r3"],
        reader=reader,
    )
    return [
        experiment_sources.load_arm(snapshot, spec, plans.get(spec.arm_id, {})) for spec in specs
    ]


def standard_projection(tmp_path: Path) -> tuple[list[Any], dict[str, Any]]:
    """标准三臂投影（两条规范化 runs + 一条旧 WF），用于 CSV/视图/性能用例。"""
    from src.lazybull.v2.reporting import experiment_projection

    base = write_runs_batch(tmp_path / "runs_base", batch_id="batch_base")
    arm = write_runs_batch(tmp_path / "runs_arm", batch_id="batch_arm")
    legacy = write_legacy_run(tmp_path / "legacy_a", run_id="wf_20260101_000000_a")
    arms = load_arms(
        tmp_path,
        [("base", "runs", base), ("arm", "runs", arm), ("legacy", "legacy_wf", legacy)],
    )
    built = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base", "arm_1": "arm", "arm_2": "legacy"},
    )
    return arms, built


def nav_rows(navs: Sequence[float], *, split: int = 0, day_step: int = 1) -> list[tuple[int, str, float]]:
    """由手写 NAV 序列造 (split, date, nav) 行（用于 Δ 方向/回撤用例）。"""
    return [
        (
            split,
            (_BASE_DAY + timedelta(days=index * day_step)).strftime("%Y%m%d"),
            float(nav),
        )
        for index, nav in enumerate(navs)
    ]


def summary_for(
    rows: Sequence[tuple[int, str, float]],
    *,
    run_id: str = "run",
    data_state_id: str = "ds000001",
    topk: bool = False,
    **extra: Any,
) -> pd.DataFrame:
    """由链行造逐折 summary（bt_total_return 与折内起止一致）。"""
    records: list[dict[str, Any]] = []
    for split in sorted({item[0] for item in rows}):
        segment = [item for item in rows if item[0] == split]
        record: dict[str, Any] = {
            "split_index": split,
            "test_start": segment[0][1],
            "test_end": segment[-1][1],
            "bt_total_return": segment[-1][2] / segment[0][2] - 1.0,
            "bt_annual_return": 0.1,
            "bt_max_drawdown": -0.05,
            "bt_sharpe": 1.0,
            "bt_win_rate": 0.5,
            "wf_run_id": run_id,
            "model_version": 1000 + split,
        }
        if topk:
            record["KEY_Top20_hit_rate"] = 0.5 + 0.01 * split
            record["KEY_Top30_hit_rate"] = 0.6 + 0.01 * split
        record.update(extra)
        records.append(record)
    return pd.DataFrame(records)


def arm_from_folds(
    tmp_path: Path,
    *,
    arm_id: str,
    folds: int = 3,
    rows_per_fold: int = 8,
    nav_step: float = 0.004,
    nav_dip: float = 0.0,
    implementation: str = "runs",
    data_state_id: str = "ds000001",
    config: Mapping[str, Any] | None = None,
    topk: bool = False,
) -> experiment_sources.ArmData:
    """由多折合成链造一个臂（子集/稳定性/Δ 用例）。"""
    rows, _windows = fold_nav_series(
        folds=folds, rows_per_fold=rows_per_fold, step=nav_step, dip=nav_dip
    )
    summary = summary_for(rows, run_id=arm_id, data_state_id=data_state_id, topk=topk)
    return build_arm_data(
        tmp_path,
        arm_id=arm_id,
        rows=rows,
        summary=summary,
        implementation=implementation,
        data_state_id=data_state_id,
        config=config,
    )


def arm_from_nav(
    tmp_path: Path,
    *,
    arm_id: str,
    navs: Sequence[float],
    implementation: str = "runs",
    data_state_id: str = "ds000001",
    host_mode: str = "backtest",
    config: Mapping[str, Any] | None = None,
    topk: bool = False,
) -> experiment_sources.ArmData:
    """由手写 NAV 造一个可直接投影的臂（只用于指标/Δ 用例）。"""
    rows = nav_rows(navs)
    summary = summary_for(rows, run_id=arm_id, data_state_id=data_state_id, topk=topk)
    return build_arm_data(
        tmp_path,
        arm_id=arm_id,
        rows=rows,
        summary=summary,
        implementation=implementation,
        host_mode=host_mode,
        data_state_id=data_state_id,
        config=config,
    )


def build_arm_data(
    tmp_path: Path,
    *,
    arm_id: str,
    rows: Sequence[tuple[int, str, float]],
    summary: pd.DataFrame,
    implementation: str = "runs",
    host_mode: str = "backtest",
    data_state_id: str = "ds000001",
    config: Mapping[str, Any] | None = None,
) -> experiment_sources.ArmData:
    """直接构造 ArmData（用于只测指标/稳定性、不落盘的快速用例）。"""
    chain = pd.DataFrame(
        {
            "split_index": [item[0] for item in rows],
            "date": [item[1] for item in rows],
            "nav": [item[2] for item in rows],
        }
    )
    chain["source_row_index"] = chain.groupby("split_index").cumcount()
    spec = experiment_sources.ArmSpec(
        arm_id=arm_id, source_kind=implementation, path=tmp_path, root_id=f"arm_{arm_id}"
    )
    artifacts = RunArtifacts(
        label=arm_id,
        directory=tmp_path,
        chain=chain[["split_index", "date", "nav"]].copy(),
        windows=summary[["split_index", "test_start", "test_end"]].copy(),
        data_state_id=data_state_id,
    )
    return experiment_sources.ArmData(
        spec=spec,
        status={"report_status": "available", "degradation_reason_code": None},
        label=arm_id,
        batch_id=arm_id,
        host_mode=host_mode,
        implementation=implementation,
        data_state_id=data_state_id,
        config_fingerprint="f" * 64 if implementation == "runs" else None,
        config=dict(config or DEFAULT_STRATEGY),
        chain=chain,
        summary=summary,
        windows=artifacts.windows.copy(),
        folds={},
        artifacts=artifacts,
        source_refs={},
        degradations=[],
        watermarks={"data_state_id": data_state_id, "git_commit": "abc1234", "git_dirty": False},
        rows_total=len(chain),
        rows_used=len(chain),
        split_count=int(chain["split_index"].nunique()),
    )

# -*- coding: utf-8 -*-
"""P7 T2.5 单折试点：A 袖逐日信号再生 + 相位目标 + 与锚批次一致性核验。

只推理不训练；产物落 ``data/paper_shadow/p7_testbed/<run_id>/``（测试床工作根），
不写生产根、不产生任何 pass/fail 判定（``alpha_consumed=false``）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.v2_p3.t5_arm_executors import (  # noqa: E402
    _signal_selection_mapping,
    _signal_universe_config_mapping,
)
from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    load_main_board_codes,
    score_model_fingerprint,
)
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    collect_model_identity,
    list_expected_oos_days,
    resolve_score_days,
)

ANCHOR = Path("data/runs/p2a_b0replay_20261008_144307_d79f3f")


def _progress(message: str) -> None:
    print(f"[t25] {message}", flush=True)


def _anchor_fold(anchor: Path, split_index: int) -> Dict[str, Any]:
    summary = pd.read_parquet(anchor / "summary.parquet")
    rows = summary.loc[summary["split_index"].astype(int) == int(split_index)]
    if rows.empty:
        raise ValueError(f"锚批次缺少 split{int(split_index):02d}")
    row = rows.iloc[0]
    return {
        "split_index": int(split_index),
        "model_version": int(row["model_version"]),
        "test_start": str(row["test_start"]),
        "test_end": str(row["test_end"]),
        "train_start": str(row.get("train_start", "")),
        "train_end": str(row.get("train_end", "")),
    }


def _tranche_schedule(days: Sequence[str], rebalance_freq: int, tranches: int) -> Dict[str, int]:
    from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule

    schedule = compute_tranche_schedule(list(days), rebalance_freq, tranches)
    return {str(day): int(idx) for day, idx in schedule.items()}


def _compare_with_anchor(
    anchor: Path, split_index: int, predictions: pd.DataFrame, topk: pd.DataFrame
) -> Dict[str, Any]:
    path = anchor / "folds" / f"split{int(split_index):02d}" / "attribution.parquet"
    attribution = pd.read_parquet(path)
    planned = attribution[
        ["signal_date", "planned_ts_code", "pred_score", "planned_rank", "tranche_idx"]
    ]
    planned = planned.dropna(subset=["pred_score"])
    merged = planned.merge(
        predictions,
        left_on=["signal_date", "planned_ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
    )
    diff = (merged["pred_score"] - merged["score"]).abs()
    scores = {
        "anchor_rows": int(len(planned)),
        "matched_rows": int(merged["score"].notna().sum()),
        "max_abs_diff": None if diff.notna().sum() == 0 else float(diff.max()),
        "exact_rows": int((diff <= 1e-9).sum()),
    }
    selection_ok = 0
    selection_total = 0
    if not topk.empty:
        topk_dates = topk["trade_date"].astype(str)
        for (date, tranche), group in planned.groupby(["signal_date", "tranche_idx"]):
            mine = topk[topk_dates == str(date)]
            if mine.empty:
                continue
            selection_total += 1
            ranks = group["planned_rank"].astype(int)
            picked = set(mine[mine["rank"].astype(int).isin(ranks)]["ts_code"].astype(str))
            if picked == set(group["planned_ts_code"].astype(str)):
                selection_ok += 1
    scores["topk_compared_dates"] = selection_total
    scores["topk_exact_dates"] = selection_ok
    return scores


def _targets(
    topk: pd.DataFrame, schedule: Dict[str, int], tranche_count: int, top_n: int
) -> pd.DataFrame:
    from src.lazybull.v2.common.rules.stagger import get_tranche_target_count

    if topk.empty:
        return pd.DataFrame(columns=["signal_date", "tranche_idx", "planned_rank", "ts_code"])
    frame = topk.copy()
    frame["signal_date"] = frame["trade_date"].astype(str)
    frame["tranche_idx"] = frame["signal_date"].map(lambda day: schedule.get(day, -1))
    frame = frame[frame["tranche_idx"] >= 0]
    frame["planned_rank"] = (
        frame.groupby("signal_date")["score"].rank(ascending=False, method="first").astype(int)
    )
    limits = {
        idx: int(get_tranche_target_count(idx, top_n, tranche_count))
        for idx in range(tranche_count)
    }
    keep = frame.apply(lambda row: row["planned_rank"] <= limits[int(row["tranche_idx"])], axis=1)
    selected = frame[keep]
    return selected[["signal_date", "tranche_idx", "planned_rank", "ts_code", "score"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P7 T2.5 单折试点")
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--anchor-root", default=str(ANCHOR))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default="data/paper_shadow/p7_testbed")
    parser.add_argument("--run-id", default="p7_t25_pilot_r1")
    args = parser.parse_args(argv)

    started = time.time()
    anchor = Path(args.anchor_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_root) / args.run_id
    if out_dir.exists():
        parser.error(f"输出根已存在（禁止覆盖）：{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    fold = _anchor_fold(anchor, args.split_index)
    _progress(
        f"折 split{fold['split_index']:02d}: v{fold['model_version']} "
        f"test {fold['test_start']}~{fold['test_end']}"
    )

    expected = list_expected_oos_days(data_root, fold["test_start"], fold["test_end"])
    score_days = resolve_score_days(expected, data_root / "features")
    _progress(f"预期交易日 {len(expected)}、可打分日 {len(score_days)}")

    models_dir = data_root / "models" / "stock_selection"
    main_board = load_main_board_codes(data_root, stock_domain="main")
    predictions_path = out_dir / "predictions.parquet"
    score_started = time.time()
    fingerprint = score_model_fingerprint(
        models_dir=models_dir,
        model_version=fold["model_version"],
        data_root=data_root,
        score_days=score_days,
        main_board_codes=main_board,
        producer_identity={"task": "p7_t2.5_pilot", "run_id": args.run_id},
        persist_predictions=True,
        predictions_path=predictions_path,
        signal_selection=_signal_selection_mapping(),
        signal_universe_config=_signal_universe_config_mapping(),
    )
    score_seconds = time.time() - score_started
    _progress(
        f"打分完成：{fingerprint['n_days']} 日 {fingerprint['n_rows']} 行，"
        f"{score_seconds:.1f}s，指纹 {fingerprint['fingerprint'][:12]}…"
    )

    predictions = pd.read_parquet(predictions_path)
    raw_topk = Path(str(fingerprint.get("signal_topk_path") or ""))
    candidates = [raw_topk, _REPO_ROOT / raw_topk, out_dir / raw_topk.name]
    topk_path = next((path for path in candidates if path.exists()), None)
    if topk_path is None:
        raise FileNotFoundError(f"未找到信号 top-N 产物：{raw_topk}")
    topk = pd.read_parquet(topk_path)
    topk["trade_date"] = topk["trade_date"].astype(str)

    consistency = _compare_with_anchor(anchor, fold["split_index"], predictions, topk)
    _progress(
        "锚一致性：分数 exact "
        f"{consistency['exact_rows']}/{consistency['matched_rows']}，Top-N exact "
        f"{consistency['topk_exact_dates']}/{consistency['topk_compared_dates']}"
    )

    schedule = _tranche_schedule(score_days, 20, 2)
    targets = _targets(topk, schedule, 2, int(_signal_selection_mapping()["top_n"]))
    targets.to_parquet(out_dir / "targets.parquet", index=False)

    model_identity = collect_model_identity(data_root, fold["model_version"])
    payload = {
        "schema": "p7_t25_fold_pilot/v1",
        "run_id": args.run_id,
        "split_index": fold["split_index"],
        "model_version": fold["model_version"],
        "test_window": [fold["test_start"], fold["test_end"]],
        "signal_domain": [score_days[0], score_days[-1]] if score_days else [],
        "score_days": len(score_days),
        "expected_days": len(expected),
        "predictions_rows": int(len(predictions)),
        "fingerprint": fingerprint["fingerprint"],
        "predictions_sha256": fingerprint.get("predictions_sha256"),
        "signal_topk_rows": int(len(topk)),
        "targets_rows": int(len(targets)),
        "tranche_dates": {
            str(idx): sorted(day for day, value in schedule.items() if value == idx)
            for idx in range(2)
        },
        "anchor_consistency": consistency,
        "model_identity": model_identity,
        "timing": {
            "score_seconds": round(score_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
            "rows_per_second": round(len(predictions) / max(score_seconds, 1e-9), 1),
        },
        "alpha_consumed": False,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _progress(f"产物已落盘：{out_dir}（{payload['timing']['total_seconds']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""P6 T3 成本与双边成交金额（设计 §5.5：按 fold+date 聚合，折界重复日不归并）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_metrics  # noqa: E402
from tests.p6_t3_env import load_arms, write_runs_batch  # noqa: E402


def test_p6_t3_19_turnover_by_fold_date_not_merged_at_seams(tmp_path):
    """双边成交金额 = 逐 (折, 日期) 行求和；折界重复日各自计入（不归并）。"""
    folds, rows_per_fold = 3, 8
    batch = write_runs_batch(
        tmp_path / "runs_a", batch_id="batch_a", folds=folds, rows_per_fold=rows_per_fold
    )
    arm = load_arms(tmp_path, [("a", "runs", batch)])[0]
    cost = experiment_metrics.cost_turnover(arm)
    assert cost["status"]["report_status"] == "available"
    expected_turnover = sum(
        rows_per_fold * (1000.0 + split * 1000.0) for split in range(folds)
    )
    assert cost["turnover_total"] == expected_turnover
    # 每个 (折, 日期) 一行：折界重复日不合并（行数 = 折数 × 折内行数）
    assert len(cost["turnover_by_fold_date"]) == folds * rows_per_fold
    assert sum(item["turnover_amount"] for item in cost["turnover_by_fold_date"]) == expected_turnover
    per_fold = {
        split: sum(
            item["turnover_amount"]
            for item in cost["turnover_by_fold_date"]
            if item["split_index"] == split
        )
        for split in range(folds)
    }
    assert per_fold == {
        split: rows_per_fold * (1000.0 + split * 1000.0) for split in range(folds)
    }
    # 成本金额：每 (折, 日期) 一买（10 元×100 股）一卖（11 元×100 股）
    per_round_trip = round(10.0 * 100 * 0.0003, 4) + round(11.0 * 100 * 0.0003, 4)
    assert cost["cost_total"] == pytest.approx(per_round_trip * folds * rows_per_fold)
    # 折界重复日口径：合计与“每个 trade_date 只取一次（后折）”的差 = 前折重复计入部分（A-10）
    seam_days = set(cost["seam_duplicate_days"])
    assert len(seam_days) == folds - 1
    by_day: dict[str, list[tuple[int, float]]] = {}
    for item in cost["turnover_by_fold_date"]:
        by_day.setdefault(item["trade_date"], []).append(
            (item["split_index"], item["turnover_amount"])
        )
    earlier = sum(
        amount
        for day, items in by_day.items()
        if day in seam_days
        for split, amount in items
        if split == min(item[0] for item in items)
    )
    dedup_total = sum(
        (max(items, key=lambda pair: pair[0])[1] if day in seam_days else items[0][1])
        for day, items in by_day.items()
    )
    assert cost["turnover_total"] == pytest.approx(dedup_total + earlier)
    assert earlier > 0

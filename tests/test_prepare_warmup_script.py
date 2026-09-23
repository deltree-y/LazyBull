# -*- coding: utf-8 -*-
"""预热生成脚本（``scripts/prepare_paper_exposure_warmup.py``）轻量单元测试。

只覆盖不依赖真实数据的部分：候选扫描（``scan_batch_candidates``）与命令入口校验
（``--snapshot-dir`` 必须显式给出）。真实回放路径由真实数据冒烟覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_paper_exposure_warmup import main, scan_batch_candidates  # noqa: E402

SNAPSHOT_COLUMNS = [
    "运行标识",
    "折序号",
    "模型版本",
    "日期",
    "股票代码",
    "持仓股数",
    "持仓市值",
    "持仓权重",
    "组合总值",
    "买入 日",
    "信号日",
    "持有交易日数",
    "到期执行日",
    "剩余持有交易日",
]


def _write_snapshot(raw_dir: Path, split: int, days: list) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = len(days)
    frame = pd.DataFrame(
        {
            "运行标识": ["r1"] * count,
            "折序号": [split] * count,
            "模型版本": [1] * count,
            "日期": days,
            "股票代码": ["600000.SH"] * count,
            "持仓股数": [100] * count,
            "持仓市值": [1000.0] * count,
            "持仓权重": [0.1] * count,
            "组合总值": [10000.0] * count,
            "买入 日": [days[0]] * count,
            "信号日": [days[0]] * count,
            "持有交易日数": [0] * count,
            "到期执行日": [days[-1]] * count,
            "剩余持有交易日": [19] * count,
        }
    )
    frame.to_csv(raw_dir / f"walk_forward_持仓快照_wf_test_split{split:02d}.csv", index=False)


def test_scan_batch_candidates_collects_snapshot_info(tmp_path):
    root = tmp_path / "batches"
    _write_snapshot(root / "batch_old" / "raw", 0, ["2025-01-02", "2025-01-03"])
    _write_snapshot(root / "batch_new" / "raw", 0, ["2025-01-02", "2025-01-03", "2025-12-12"])
    _write_snapshot(root / "batch_new" / "raw", 1, ["2025-01-02", "2025-12-12"])
    (root / "empty_batch" / "raw").mkdir(parents=True)  # 无快照 → 不入选

    candidates = scan_batch_candidates(root)
    names = {item["batch"] for item in candidates}
    assert names == {"batch_new", "batch_old"}
    newest = next(item for item in candidates if item["batch"] == "batch_new")
    assert newest["snapshots"] == 2
    assert newest["first_day"] == "20250102"
    assert newest["last_day"] == "20251212"


def test_scan_batch_candidates_reads_summary_params(tmp_path):
    root = tmp_path / "batches"
    raw = root / "batch_a" / "raw"
    _write_snapshot(raw, 0, ["2025-01-02"])
    pd.DataFrame(
        [
            {
                "top_n": 20,
                "rebalance_freq": 20,
                "stagger_tranches": 2,
                "position_sizing": "score",
            }
        ]
    ).to_csv(raw / "walk_forward_summary_0101_0001.csv", index=False)

    candidates = scan_batch_candidates(root)
    assert len(candidates) == 1
    item = candidates[0]
    assert item["stagger_tranches"] == "2"
    assert item["position_sizing"] == "score"
    assert item["top_n"] == "20"


def test_scan_batch_candidates_tolerates_bad_files(tmp_path):
    """坏快照文件只降级为备注，不阻断候选列出。"""
    root = tmp_path / "batches"
    raw = root / "batch_bad" / "raw"
    raw.mkdir(parents=True)
    (raw / "walk_forward_持仓快照_wf_test_split00.csv").write_text(
        "不完整的表头\n", encoding="utf-8"
    )
    candidates = scan_batch_candidates(root)
    assert len(candidates) == 1
    assert "date_note" in candidates[0]


def test_main_requires_explicit_snapshot_dir(monkeypatch):
    """未给 --snapshot-dir 且未 --list-batches ⇒ 明确报错（返回 2），禁止隐式选批。"""
    monkeypatch.setattr(sys, "argv", ["prepare_paper_exposure_warmup.py"])
    assert main() == 2

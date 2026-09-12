# -*- coding: utf-8 -*-
"""门禁区间重判脚本的分组选择测试（合成 WF 目录，不依赖真实数据）。

背景：``--wf-root`` 下常同时存在多批实验（如旧轮次 ``*_lr0.05`` 与新轮次无后缀
目录）。若默认取 ``tuning_scores.csv`` 排名第一的组，就会把历史实验组当成
"刚跑完的那批"重判。本测试锁定分组选择规则：

- ``--select latest``（默认）：折目录写入时间最新的一组；
- ``--select best``：``tuning_scores.csv`` 排名第一的组；
- ``--signature``：显式指定，优先级最高。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_terminal_risk_gate import (  # noqa: E402
    collect_fold_lifts,
    format_groups,
    load_group_table,
    pick_signature,
)
from scripts.analyze_terminal_risk_gate import main as gate_main  # noqa: E402

_SIG_OLD = "d=6|lr=0.05|nest=1000|esr=100"
_SIG_NEW = "d=5|lr=0.04|nest=1000|esr=50"


def _write_wf_root(tmp_path: Path, old_mtime: float, new_mtime: float) -> Path:
    """构造含两批实验（旧轮次 rank1 + 新轮次 rank2）的合成 WF 根目录。"""
    folds_old = ["2022H2_lr0.05", "2023H1_lr0.05"]
    folds_new = ["2022H2", "2023H1"]
    summary = pd.DataFrame(
        [{"fold": f, "lift": 1.2, "param_signature": _SIG_OLD} for f in folds_old]
        + [{"fold": f, "lift": 1.3, "param_signature": _SIG_NEW} for f in folds_new]
    )
    summary.to_csv(tmp_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "suffix": "_lr0.05",
                "param_signature": _SIG_OLD,
                "n_folds": 2,
                "tuning_score": 1.34,
                "lift_min": 1.155,
            },
            {
                "rank": 2,
                "suffix": "(baseline)",
                "param_signature": _SIG_NEW,
                "n_folds": 2,
                "tuning_score": 1.32,
                "lift_min": 1.151,
            },
        ]
    ).to_csv(tmp_path / "tuning_scores.csv", index=False)

    for fold, mtime in [(f, old_mtime) for f in folds_old] + [(f, new_mtime) for f in folds_new]:
        fold_dir = tmp_path / fold
        fold_dir.mkdir(parents=True)
        report = fold_dir / "v1_report.json"
        report.write_text("{}", encoding="utf-8")
        os.utime(report, (mtime, mtime))
    return tmp_path


class TestGroupTable:
    def test_groups_carry_fold_dirs_and_latest_mtime_sorted(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        groups = load_group_table(tmp_path)
        assert len(groups) == 2
        # 按最新写入时间升序 → 新轮次在末行
        assert groups.iloc[-1]["param_signature"] == _SIG_NEW
        assert groups.iloc[-1]["fold_dirs"] == "2022H2,2023H1"
        assert groups.iloc[0]["fold_dirs"] == "2022H2_lr0.05,2023H1_lr0.05"
        assert groups.iloc[-1]["latest_mtime"] > groups.iloc[0]["latest_mtime"]
        # 调参分/排名按签名左连接进来
        assert groups.iloc[-1]["rank"] == 2
        assert groups.iloc[-1]["tuning_score"] == 1.32

    def test_missing_param_signature_raises(self, tmp_path):
        pd.DataFrame([{"fold": "2022H2", "lift": 1.2}]).to_csv(
            tmp_path / "summary.csv", index=False
        )
        try:
            load_group_table(tmp_path)
        except ValueError as exc:
            assert "param_signature" in str(exc)
        else:
            raise AssertionError("缺少 param_signature 时未报错")

    def test_group_without_fold_dirs_not_selected_as_latest(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        summary = pd.read_csv(tmp_path / "summary.csv")
        summary = pd.concat(
            [
                summary,
                pd.DataFrame([{"fold": "2099H1", "lift": 1.9, "param_signature": "sig-missing"}]),
            ],
            ignore_index=True,
        )
        summary.to_csv(tmp_path / "summary.csv", index=False)
        groups = load_group_table(tmp_path)
        # "sig-missing" 无折目录（latest_mtime 为 NaN）→ 不参与 latest 选择
        assert pick_signature(groups, None, "latest") == _SIG_NEW


class TestPickSignature:
    def test_latest_picks_newest_fold_dirs(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        groups = load_group_table(tmp_path)
        assert pick_signature(groups, None, "latest") == _SIG_NEW

    def test_best_picks_top_ranked_group(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_800_000_000, new_mtime=1_700_000_000)
        groups = load_group_table(tmp_path)
        # 即便旧组折目录更旧，best 仍按 tuning_scores 排名取 rank=1
        assert pick_signature(groups, None, "best") == _SIG_OLD

    def test_explicit_signature_wins(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        groups = load_group_table(tmp_path)
        assert pick_signature(groups, _SIG_OLD, "latest") == _SIG_OLD
        assert pick_signature(groups, _SIG_OLD, "best") == _SIG_OLD

    def test_empty_groups_returns_none(self, tmp_path):
        groups = pd.DataFrame(columns=["param_signature", "fold_dirs", "latest_mtime"])
        assert pick_signature(groups, None, "latest") is None

    def test_best_without_rank_falls_back_to_latest(self, tmp_path):
        groups = pd.DataFrame(
            [
                {"param_signature": "a", "fold_dirs": "f1", "latest_mtime": 1.0},
                {"param_signature": "b", "fold_dirs": "f2", "latest_mtime": 2.0},
            ]
        )
        assert pick_signature(groups, None, "best") == "b"


class TestFormatGroups:
    def test_renders_dirs_rank_and_score(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        text = format_groups(load_group_table(tmp_path))
        assert "2022H2,2023H1" in text
        assert "2022H2_lr0.05,2023H1_lr0.05" in text
        assert "rank=1" in text and "rank=2" in text
        assert _SIG_NEW in text
        assert "无折目录" not in text

    def test_renders_missing_dirs_placeholder(self):
        groups = pd.DataFrame(
            [
                {
                    "param_signature": "sig",
                    "n_folds": 1,
                    "fold_dirs": "2099H1",
                    "latest_mtime": None,
                    "rank": None,
                    "suffix": None,
                    "tuning_score": None,
                }
            ]
        )
        text = format_groups(groups)
        assert "无折目录" in text
        assert "rank=-" in text and "调参分=-" in text


class TestScoreModes:
    """双判据（raw + daynorm）必须同时写入产物（v0.110.0）。"""

    def _write_dual_root(self, tmp_path: Path) -> Path:
        folds = ["2022H2", "2023H1"]
        pd.DataFrame(
            [{"fold": f, "lift": 1.2, "param_signature": _SIG_NEW} for f in folds]
        ).to_csv(tmp_path / "summary.csv", index=False)
        rng = np.random.default_rng(0)
        for fold in folds:
            fold_dir = tmp_path / fold
            fold_dir.mkdir(parents=True)
            rows = []
            for i in range(40):
                for j in range(20):
                    p = float(np.clip(rng.normal(0.3, 0.15), 0.01, 0.99))
                    rows.append(
                        {
                            "trade_date": f"2024{(i // 20) + 1:02d}{(i % 20) + 1:02d}",
                            "ts_code": f"0000{j:02d}.SZ",
                            "h": (j % 20) + 1,
                            "loss_label": int(rng.random() < p),
                            "p_loss": p,
                        }
                    )
            pd.DataFrame(rows).to_parquet(
                fold_dir / "terminal_loss_es_predictions.parquet", index=False
            )
        return tmp_path

    def test_main_writes_both_score_modes(self, tmp_path, monkeypatch):
        root = self._write_dual_root(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "analyze_terminal_risk_gate.py",
                "--wf-root",
                str(root),
                "--score-mode",
                "both",
                "--n-resamples",
                "50",
                "--block-resamples",
                "20",
            ],
        )
        assert gate_main() == 0
        gate = pd.read_csv(root / "gate_ci.csv")
        assert set(gate["score_mode"]) == {"raw", "daynorm"}
        block = pd.read_csv(root / "gate_ci_block.csv")
        assert set(block["score_mode"]) == {"raw", "daynorm"}
        assert set(block["block_days"]) == {5, 10, 20}
        assert set(block["fold"]) == {"2022H2", "2023H1"}

    def test_main_single_mode_writes_one_criterion(self, tmp_path, monkeypatch):
        root = self._write_dual_root(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "analyze_terminal_risk_gate.py",
                "--wf-root",
                str(root),
                "--score-mode",
                "daynorm",
                "--n-resamples",
                "50",
                "--block-resamples",
                "20",
            ],
        )
        assert gate_main() == 0
        gate = pd.read_csv(root / "gate_ci.csv")
        assert set(gate["score_mode"]) == {"daynorm"}
        block = pd.read_csv(root / "gate_ci_block.csv")
        assert set(block["score_mode"]) == {"daynorm"}


class TestCollectFoldLifts:
    def test_unknown_signature_raises(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        summary = pd.read_csv(tmp_path / "summary.csv")
        try:
            collect_fold_lifts(summary, "nope")
        except ValueError as exc:
            assert "不存在签名" in str(exc)
        else:
            raise AssertionError("未知签名未报错")

    def test_subset_by_signature(self, tmp_path):
        _write_wf_root(tmp_path, old_mtime=1_700_000_000, new_mtime=1_800_000_000)
        summary = pd.read_csv(tmp_path / "summary.csv")
        subset = collect_fold_lifts(summary, _SIG_NEW)
        assert list(subset["fold"]) == ["2022H2", "2023H1"]

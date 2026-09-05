#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""walk-forward 数据态血缘与跨数据态对比告警测试。"""

import json
import os
import types

import pandas as pd
import pytest

from src.lazybull.ml.walk_forward import data_state


def _make_fake_data_root(tmpdir):
    """构造带水位的数据目录：raw/daily（带横线分区名）+ features/cs_train（8 位纯数字分区名）。"""
    daily_dir = os.path.join(str(tmpdir), "raw", "daily")
    os.makedirs(daily_dir, exist_ok=True)
    for date in ("2026-01-04", "2026-01-05"):
        with open(os.path.join(daily_dir, f"{date}.parquet"), "wb") as f:
            f.write(b"stub")
    cs_train_dir = os.path.join(str(tmpdir), "features", "cs_train")
    os.makedirs(cs_train_dir, exist_ok=True)
    with open(os.path.join(cs_train_dir, "20260105.parquet"), "wb") as f:
        f.write(b"stub")
    return str(tmpdir)


@pytest.fixture(autouse=True)
def _stable_git_state(monkeypatch):
    """固定 git 采集结果，保证测试不依赖真实仓库状态。"""
    monkeypatch.setattr(
        data_state, "_git_state", lambda: {"git_commit": "abc1234", "git_dirty": False}
    )


class TestDataStateCollection:
    def test_collect_data_state_reads_watermarks(self, tmp_path):
        """数据态快照应记录 git、raw 最新分区与 cs_train 最新分区。"""
        data_root = _make_fake_data_root(tmp_path)
        state = data_state.collect_data_state(data_root=data_root, wf_run_id="wf_test_001")

        assert state["git_commit"] == "abc1234"
        assert state["git_dirty"] is False
        assert state["wf_run_id"] == "wf_test_001"
        assert state["raw_latest_partitions"]["daily"] == "2026-01-05"
        # features 层分区名为 YYYYMMDD 纯数字格式，应与 raw 层一并识别
        assert state["features_cs_train_latest"] == "20260105"
        # 未执行 dividend 回补时覆盖状态为 None
        assert state["dividend_coverage"] is None

    def test_latest_partition_accepts_both_formats(self, tmp_path):
        """分区名识别应同时兼容 YYYY-MM-DD 与 YYYYMMDD 两种格式。"""
        target = tmp_path / "mixed"
        target.mkdir()
        for name in ("2025-12-31.parquet", "20260105.parquet", "ignore.txt", "bad1234.parquet"):
            with open(target / name, "wb") as f:
                f.write(b"stub")
        assert data_state._latest_partition_date(target) == "20260105"

    def test_collect_data_state_empty_root(self, tmp_path):
        """空数据目录下各水位应为 None，且不抛异常。"""
        state = data_state.collect_data_state(data_root=str(tmp_path))
        assert state["raw_latest_partitions"]["daily"] is None
        assert state["features_cs_train_latest"] is None

    def test_state_id_stable_ignoring_volatile_fields(self, tmp_path):
        """运行标识与采集时间不参与数据态指纹。"""
        data_root = _make_fake_data_root(tmp_path)
        first = data_state.collect_data_state(data_root=data_root, wf_run_id="wf_a")
        second = data_state.collect_data_state(data_root=data_root, wf_run_id="wf_b")
        second["collected_at"] = "2099-01-01T00:00:00"
        assert data_state.compute_data_state_id(first) == data_state.compute_data_state_id(second)

    def test_state_id_changes_with_watermark(self, tmp_path):
        """raw/daily 新增分区必须改变数据态 ID。"""
        data_root = _make_fake_data_root(tmp_path)
        first = data_state.collect_data_state(data_root=data_root)
        with open(os.path.join(str(tmp_path), "raw", "daily", "2026-01-06.parquet"), "wb") as f:
            f.write(b"stub")
        second = data_state.collect_data_state(data_root=data_root)
        assert data_state.compute_data_state_id(first) != data_state.compute_data_state_id(second)


class TestDataStateSummaryColumns:
    def test_summary_columns_format(self, tmp_path):
        """摘要列包含数据态 ID 与水位，dividend 覆盖输出为可读文本。"""
        state = data_state.collect_data_state(data_root=_make_fake_data_root(tmp_path))
        state["dividend_coverage"] = {"data": 2, "empty": 1, "pending": 0, "failed": 0, "total": 3}
        cols = data_state.data_state_summary_columns(state)

        assert set(cols) == {
            "data_state_id",
            "git_commit",
            "git_dirty",
            "data_daily_latest",
            "data_cs_train_latest",
            "data_dividend_coverage",
        }
        assert cols["data_state_id"] == data_state.compute_data_state_id(state)
        assert cols["data_daily_latest"] == "2026-01-05"
        assert cols["data_dividend_coverage"] == "data=2,empty=1,failed=0,pending=0,total=3"

    def test_summary_columns_dividend_missing(self, tmp_path):
        """dividend 覆盖缺失时摘要列应为 None 而非报错。"""
        state = data_state.collect_data_state(data_root=str(tmp_path))
        cols = data_state.data_state_summary_columns(state)
        assert cols["data_dividend_coverage"] is None


class TestDataStateFile:
    def test_write_data_state_file(self, tmp_path):
        """数据态快照应落盘为 data_state_{wf_run_id}.json 且包含指纹。"""
        state = data_state.collect_data_state(
            data_root=_make_fake_data_root(tmp_path), wf_run_id="wf_test_002"
        )
        out_dir = tmp_path / "walk_forward" / "batches" / "wf_batch_test" / "raw"
        path = data_state.write_data_state_file(out_dir, state)

        assert path is not None and path.exists()
        assert path.name == "data_state_wf_test_002.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["wf_run_id"] == "wf_test_002"
        assert "raw_latest_partitions" in payload
        # JSON 自描述：附带数据态指纹，且与摘要列一致
        assert payload["data_state_id"] == data_state.compute_data_state_id(state)

    def test_write_failure_returns_none(self, tmp_path, monkeypatch):
        """写盘失败只告警并返回 None，不抛异常阻断训练输出。"""
        state = data_state.collect_data_state(data_root=str(tmp_path), wf_run_id="wf_test_003")
        target = tmp_path / "out"
        target.mkdir()
        # 构造同名目录使写文件必然失败
        (target / "data_state_wf_test_003.json").mkdir()
        assert data_state.write_data_state_file(target, state) is None


class TestSummaryIntegration:
    def _build_mock_args(self, data_root):
        return types.SimpleNamespace(
            wf_start_date="20200101",
            wf_end_date="20230630",
            batch_run_id="wf_batch_test",
            batch_period_label="0101",
            data_root=data_root,
            algorithm="xgboost",
            train_window_years=3,
            test_window_months=3,
            val_ratio=0.2,
            label_column="y_ret_5",
            task="regression",
            label_transform="cs_zscore",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            early_stopping_rounds=50,
            early_stopping_metric="rmse",
            rank_weight_enabled=False,
            rank_weight_topk=100,
            rank_weight=2.0,
            time_decay_half_life=0,
            freshness_strategy="state_keep_event_no_decay",
            event_freshness_half_life_days=120.0,
            enable_fundamental_features=False,
            enable_alt_features=False,
            enable_margin_features=False,
            enable_cyq_features=False,
            enable_fund_features=False,
            enable_express_features=False,
            feature_stability_filter=False,
            oos_backtest=False,
        )

    _RESULTS = [
        {
            "split_index": 0,
            "train_start": "20200101",
            "train_end": "20221231",
            "test_start": "20230101",
            "test_end": "20230331",
            "model_version": 1,
            "test_daily_metrics": {},
        }
    ]

    def test_summary_contains_data_state_columns(self, tmp_path):
        """summary CSV 应包含数据态摘要列，并同目录落盘血缘 JSON。"""
        from src.lazybull.ml.walk_forward.summary import write_walk_forward_summary

        data_root = _make_fake_data_root(tmp_path)
        out_dir = tmp_path / "raw"
        output_path = str(out_dir / "walk_forward_summary_wf_test.csv")

        write_walk_forward_summary(
            self._RESULTS, output_path, self._build_mock_args(data_root), "wf_test_010"
        )

        df = pd.read_csv(output_path)
        for col in ("data_state_id", "git_commit", "data_daily_latest"):
            assert col in df.columns
        assert df.loc[0, "data_daily_latest"] == "2026-01-05"
        assert (out_dir / "data_state_wf_test_010.json").exists()

    def test_summary_survives_data_state_failure(self, tmp_path, monkeypatch):
        """数据态采集失败时 summary 照常写出，仅缺数据态列。"""
        from src.lazybull.ml.walk_forward import data_state as data_state_module
        from src.lazybull.ml.walk_forward.summary import write_walk_forward_summary

        def _boom(*args, **kwargs):
            raise RuntimeError("模拟数据态采集失败")

        monkeypatch.setattr(data_state_module, "collect_data_state", _boom)
        output_path = str(tmp_path / "walk_forward_summary_wf_test.csv")

        write_walk_forward_summary(
            self._RESULTS, output_path, self._build_mock_args(str(tmp_path)), "wf_test_011"
        )

        df = pd.read_csv(output_path)
        assert "split_index" in df.columns
        assert not any(col.startswith("data_state") for col in df.columns)


class TestWarnCrossDataState:
    @staticmethod
    def _comp_df(states):
        rows = []
        for idx, state_id in enumerate(states):
            rows.append(
                {
                    "运行ID": f"wf_run_{idx}",
                    "数据态ID": state_id,
                    "Git版本": "abc1234" if state_id else None,
                    "raw/daily水位": "2026-01-05" if state_id else None,
                }
            )
        return pd.DataFrame(rows)

    def test_single_state_no_warning(self):
        from scripts.compare.data_state import warn_cross_data_state

        assert warn_cross_data_state(self._comp_df(["a1b2c3d4"] * 3)) is False

    def test_cross_state_warns(self):
        from scripts.compare.data_state import warn_cross_data_state

        assert warn_cross_data_state(self._comp_df(["a1b2c3d4", "a1b2c3d4", "e5f6a7b8"])) is True

    def test_missing_state_column_is_silent(self):
        from scripts.compare.data_state import warn_cross_data_state

        df = pd.DataFrame({"运行ID": ["wf_a"], "综合得分": [50.0]})
        assert warn_cross_data_state(df) is False

    def test_unknown_states_tolerated(self):
        """历史运行无血缘记录（全 NaN）时归入未知态，不算跨态。"""
        from scripts.compare.data_state import warn_cross_data_state

        assert warn_cross_data_state(self._comp_df([None, None])) is False

    def test_mixed_unknown_and_known_warns(self):
        from scripts.compare.data_state import warn_cross_data_state

        assert warn_cross_data_state(self._comp_df([None, "a1b2c3d4"])) is True

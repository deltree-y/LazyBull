# -*- coding: utf-8 -*-
"""v2/train/walk_forward/data_state.py 测试（P3-T3 §5 改造）。

覆盖：旧口径函数与旧模块同输入同输出（replay_b0 改指兼容）；receipt 汇总去重 /
冲突拒绝 / 非 store 现算；运行后复核漂移拒绝（物理核验为准）；环境身份字段齐备；
增强 id 新旧不可比 + 产物旧键集不变。全部 tmp_path 合成数据。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.lazybull.ml.walk_forward import data_state as old_ds
from src.lazybull.v2.store.store_identity import InputIdentityDriftError
from src.lazybull.v2.train.walk_forward import data_state as new_ds


@dataclass(frozen=True)
class _FakeReceipt:
    files: tuple


def _file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestLegacyParity:
    """旧口径函数：与旧模块同输入同输出（逐字保留证据）。"""

    def test_git_state_same(self):
        assert new_ds._git_state() == old_ds._git_state()
        assert new_ds._PROJECT_ROOT == old_ds._PROJECT_ROOT  # 深度修正后仍指向项目根
        assert (new_ds._PROJECT_ROOT / "pyproject.toml").is_file()

    def test_partition_stem_and_latest_same(self, tmp_path):
        for stem in ("2026-01-05", "20260105", "abc", "2026-1-5", ""):
            assert new_ds._is_partition_stem(stem) == old_ds._is_partition_stem(stem)
        part_dir = tmp_path / "raw" / "daily"
        _file(part_dir / "2026-01-03.parquet", b"a")
        _file(part_dir / "2026-01-05.parquet", b"b")
        _file(part_dir / "notes.parquet", b"c")
        assert new_ds._latest_partition_date(part_dir) == old_ds._latest_partition_date(part_dir)
        assert new_ds._latest_partition_date(tmp_path / "missing") is None

    def test_state_id_and_summary_same(self):
        state = {
            "collected_at": "2026-10-22T10:00:00",
            "wf_run_id": "wf_x",
            "batch_run_id": None,
            "git_commit": "abc1234",
            "git_dirty": True,
            "raw_latest_partitions": {"daily": "2026-07-31"},
            "features_cs_train_latest": "20260731",
            "dividend_coverage": {"2025": 10},
        }
        assert new_ds.compute_data_state_id(state) == old_ds.compute_data_state_id(state)
        assert new_ds.data_state_summary_columns(state) == old_ds.data_state_summary_columns(state)

    def test_write_file_same_payload_shape(self, tmp_path):
        state = {"wf_run_id": "wf_x", "git_commit": "abc"}
        old_path = old_ds.write_data_state_file(tmp_path / "old", dict(state))
        new_path = new_ds.write_data_state_file(tmp_path / "new", dict(state))
        old_payload = json.loads(old_path.read_text(encoding="utf-8"))
        new_payload = json.loads(new_path.read_text(encoding="utf-8"))
        assert old_payload == new_payload


class TestConsumedFileIdentities:
    def test_merge_receipts_dedup_and_extra(self, tmp_path):
        f1 = _file(tmp_path / "features" / "panel" / "20260105" / "core.parquet", b"a")
        f2 = _file(tmp_path / "labels" / "y_ret_5" / "20260105.parquet", b"b")
        f3 = _file(tmp_path / "clean" / "trade_cal.parquet", b"c")
        r1 = _FakeReceipt(files=tuple([self._identity(f1, tmp_path), self._identity(f2, tmp_path)]))
        r2 = _FakeReceipt(files=tuple([self._identity(f1, tmp_path)]))  # 重叠同值 ⇒ 去重
        records = new_ds.collect_consumed_file_identities([r1, r2], [f3], base_dir=tmp_path)
        paths = [r["path"] for r in records]
        assert paths == sorted(paths)
        assert len(paths) == 3

    def test_conflicting_receipts_rejected(self, tmp_path):
        f1 = _file(tmp_path / "a.parquet", b"a")
        ident_a = self._identity(f1, tmp_path)
        ident_b = dict(ident_a, sha256="0" * 64)  # 同路径不同 sha256 ⇒ 漂移冲突
        with pytest.raises(InputIdentityDriftError, match="多次读取间漂移"):
            new_ds.collect_consumed_file_identities(
                [_FakeReceipt(files=(ident_a,)), _FakeReceipt(files=(ident_b,))]
            )

    def test_verify_after_run_clean_and_drift(self, tmp_path):
        f1 = _file(tmp_path / "a.parquet", b"a")
        records = new_ds.collect_consumed_file_identities([], [f1], base_dir=tmp_path)
        new_ds.verify_consumed_file_identities(records, base_dir=tmp_path)  # 无漂移 ⇒ 通过
        f1.write_bytes(b"a+drift")  # 登记后物理变更（登记值不变场景 ⇒ 物理核验拒绝）
        with pytest.raises(InputIdentityDriftError, match="运行后复核"):
            new_ds.verify_consumed_file_identities(records, base_dir=tmp_path)

    @staticmethod
    def _identity(path: Path, base: Path):
        from src.lazybull.v2.store.store_identity import file_identity

        return file_identity(path, base_dir=base)


class TestEnvironmentIdentity:
    def test_fields_complete(self):
        env = new_ds.collect_environment_identity()
        for key in ("python", "pandas", "numpy", "pyarrow", "xgboost"):
            assert key in env and env[key]
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f"env.{var}" in env  # 线程设置键齐备（值可为空字符串）


class TestEnhancedState:
    def test_enhanced_id_differs_from_legacy(self, tmp_path):
        f1 = _file(tmp_path / "a.parquet", b"a")
        records = new_ds.collect_consumed_file_identities([], [f1], base_dir=tmp_path)
        env = new_ds.collect_environment_identity()
        state = {"wf_run_id": "wf_x", "git_commit": "abc", "git_dirty": False}
        legacy_id = new_ds.compute_data_state_id(state)
        enhanced_id = new_ds.compute_enhanced_data_state_id(state, records, env)
        assert legacy_id != enhanced_id  # §5-4:新旧批次 data_state_id 不可比
        assert len(enhanced_id) == 8

    def test_enhanced_file_additive_keys(self, tmp_path):
        f1 = _file(tmp_path / "a.parquet", b"a")
        records = new_ds.collect_consumed_file_identities([], [f1], base_dir=tmp_path)
        env = new_ds.collect_environment_identity()
        state = {"wf_run_id": "wf_x", "git_commit": "abc"}
        path = new_ds.write_enhanced_data_state_file(tmp_path, state, records, env)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["wf_run_id"] == "wf_x" and payload["git_commit"] == "abc"  # 旧键不变
        assert payload["input_files"] == records  # additive 段
        assert "environment_identity" in payload
        assert payload["data_state_id"] == new_ds.compute_enhanced_data_state_id(
            state, records, env
        )

    def test_enhanced_id_stable_across_order(self, tmp_path):
        f1 = _file(tmp_path / "a.parquet", b"a")
        f2 = _file(tmp_path / "b.parquet", b"b")
        env = {"python": "3.12"}
        r_forward = new_ds.collect_consumed_file_identities([], [f1, f2], base_dir=tmp_path)
        r_reverse = new_ds.collect_consumed_file_identities([], [f2, f1], base_dir=tmp_path)
        state = {"wf_run_id": "wf_x"}
        assert new_ds.compute_enhanced_data_state_id(
            state, r_forward, env
        ) == new_ds.compute_enhanced_data_state_id(state, r_reverse, env)

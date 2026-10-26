# -*- coding: utf-8 -*-
"""P3 T5 阶段 A 评审处置测试：R2-T5A-02 消费集合硬门扩展 + R2-T5A-11 双跑缓存探针 + P2-3 B0' 收益路径。

**全部合成/mock，禁止真实训练与真实生产数据**。覆盖：

- R2-T5A-02（P1-1）前半：``enumerate_consumption_files(include_panel_labels=True)``
  追加枚举 features/panel 热区（按族冷热同源，冷区优先）+ panel_archive 冷区 +
  labels 三表 + manifest 族（审计，不入字节门）；冻结段末 20260702 过滤；
  新增目录零覆盖/缺失 ⇒ 显式拒绝；集合内篡改 ⇒ 续跑拒绝；集合外追加 ⇒ 放行（M5）；
  manifest 窗内登记改写 ⇒ 拒绝、窗口外追加 ⇒ 放行（沿 T1 相关性 diff 口径）；
  启用扩展后 manifest 基线缺失 + 有缓存 ⇒ 拒绝；默认（False）⇒ 与 spike 现状逐位一致；
- R2-T5A-02 后半：打分 store 实际读取 receipt ⊆ 消费集合断言正反例；
- R2-T5A-11：``arm_run_cache_exists`` 识别单/双跑布局；双跑布局有缓存 + 基线缺失
  ⇒ 拒绝且快照未重建（不带续跑参数同拒绝）；
- P2-3：``freeze_b0_prime`` 复制 runs 批收益实体（chain_nav/summary/逐折 daily），
  manifest 所列全部路径可解析断言（含幂等复用分支）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t5_arm_artifacts import (  # noqa: E402
    ARM_LEGACY_CS_TRAIN,
    ARM_V2_CS_TRAIN_VERIFY,
    ARM_V2_PANEL,
    acquire_arm_baselines,
    arm_run_cache_exists,
    assert_score_receipt_within_consumption,
    collect_score_read_receipt_files,
    freeze_b0_prime,
)
from scripts.v2_p3.t05b_baselines import (  # noqa: E402
    MANIFEST_FAMILY_NAMES,
    enumerate_consumption_files,
    flatten_consumption_files,
    verify_manifest_baseline,
)
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    collect_consumption_snapshot,
    verify_consumption_snapshot,
)
from src.lazybull.v2.store.data_store import StoreReadReceipt  # noqa: E402

_FROZEN_END = "20260702"


# ---------------------------------------------------------------------------
# 合成夹具
# ---------------------------------------------------------------------------


def _write_manifest(features_dir: Path) -> None:
    """合成有效 manifest（满足加载校验顶层键；含窗内 panel/labels 登记）。"""
    doc = {
        "manifest_version": "1",
        "updated_at": "2026-07-02T00:00:00",
        "columns": {
            "close": {
                "group": "core",
                "source": "daily",
                "definition_version": "v1",
                "available_from": None,
                "backfilled_at": None,
                "status": "active",
            }
        },
        "partitions": {
            "panel/20260601/core.parquet": {
                "sha256_16": "a" * 16,
                "rows": 2,
                "written_at": "2026-06-01T00:00:00",
                "zone": "hot",
                "row_domain": "label_filtered",
            },
            "panel_archive/2025-06/core.parquet": {
                "sha256_16": "b" * 16,
                "rows": 20,
                "written_at": "2026-07-01T00:00:00",
                "zone": "archive",
                "row_domain": "label_filtered",
            },
        },
        "dependencies": {"raw_datasets": [], "factor_functions": {}},
        "labels": {
            name: {
                "partitions": {
                    "20260601.parquet": {
                        "sha256_16": "c" * 16,
                        "rows": 2,
                        "written_at": "2026-06-01T00:00:00",
                        "row_domain": "label_filtered",
                    }
                },
                "sealed_through": _FROZEN_END,
            }
            for name in ("y_ret_5", "y_ret_10", "y_ret_20")
        },
        "row_domain_segments": {
            "label_filtered_through": _FROZEN_END,
            "structural_from": "20260703",
        },
    }
    (features_dir / "manifest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (features_dir / "manifest.json.bak").write_bytes((features_dir / "manifest.json").read_bytes())
    (features_dir / "manifest_wal.jsonl").write_bytes(b"")  # 空 WAL = 无待处理态


def _make_panel_production(tmp_path: Path) -> Path:
    """合成生产数据根：spike 契约目录 + panel 热/冷区 + labels 三表 + manifest 族。"""
    prod = tmp_path / "data"
    for sub in ("features/cs_train", "clean/daily", "raw/suspend"):
        (prod / sub).mkdir(parents=True, exist_ok=True)
    for day in ("20260601", "20260702", "20260703"):
        (prod / "features" / "cs_train" / f"{day}.parquet").write_bytes(b"cs" + day.encode())
    for day in ("2026-06-01", "2026-07-02", "2026-07-03"):
        (prod / "clean" / "daily" / f"{day}.parquet").write_bytes(b"d" + day.encode())
        (prod / "raw" / "suspend" / f"{day}.parquet").write_bytes(b"s" + day.encode())
    (prod / "raw" / "moneyflow_hsgt").mkdir(parents=True)
    (prod / "raw" / "moneyflow_hsgt" / "2026-06-01.parquet").write_bytes(b"north")
    (prod / "clean" / "stock_basic.parquet").write_bytes(b"sb")
    (prod / "raw" / "trade_cal.parquet").write_bytes(b"tc")
    (prod / "models" / "stock_selection").mkdir(parents=True, exist_ok=True)
    (prod / "models" / "stock_selection" / "sentinel.txt").write_text("prod", encoding="utf-8")

    # panel 热区：20250615（core 已被冷区封存 ⇒ 让位；technical 保留）、
    # 20260601/20260702（窗内）、20260703（冻结段外 ⇒ 排除）、非日期目录（忽略）
    hot = prod / "features" / "panel"
    for day, groups in (
        ("20250615", ("core", "technical")),
        ("20260601", ("core", "technical")),
        ("20260702", ("core",)),
        ("20260703", ("core",)),
        ("not_a_date", ("core",)),
    ):
        for group in groups:
            directory = hot / day
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{group}.parquet").write_bytes(f"hot-{day}-{group}".encode())
    # panel 冷区：2025-06/core（窗内，封存热区 20250615/core）、2026-08（冻结段外 ⇒ 排除）
    archive = prod / "features" / "panel_archive"
    for month, groups in (("2025-06", ("core",)), ("2026-08", ("core",)), ("badmonth", ("core",))):
        for group in groups:
            directory = archive / month
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{group}.parquet").write_bytes(f"arc-{month}-{group}".encode())
    # labels 三表：窗内 20260601；y_ret_5 另有冻结段外 20260703（⇒ 排除）
    for name in ("y_ret_5", "y_ret_10", "y_ret_20"):
        directory = prod / "labels" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "20260601.parquet").write_bytes(f"lab-{name}".encode())
    (prod / "labels" / "y_ret_5" / "20260703.parquet").write_bytes(b"lab-out")
    _write_manifest(prod / "features")
    return prod


def _hot_names(by_dir: Dict[str, List[Path]]) -> List[str]:
    return [f"{p.parent.name}/{p.name}" for p in by_dir["features/panel"]]


def _archive_names(by_dir: Dict[str, List[Path]]) -> List[str]:
    return [f"{p.parent.name}/{p.name}" for p in by_dir["features/panel_archive"]]


def _label_names(by_dir: Dict[str, List[Path]]) -> List[str]:
    return [f"{p.parent.name}/{p.name}" for p in by_dir["labels"]]


# ---------------------------------------------------------------------------
# R2-T5A-02 前半：扩展枚举
# ---------------------------------------------------------------------------


class TestExtendedConsumptionEnumeration:
    def test_extension_covers_panel_archive_labels_with_frozen_filter(self, tmp_path):
        """扩展开启：panel 热/冷区 + labels 三表入枚举，冻结段末过滤生效。"""
        prod = _make_panel_production(tmp_path)
        by_dir = enumerate_consumption_files(prod, include_panel_labels=True)
        # 热区：族序 core 先 technical 后；冷区封存日让位（20250615/core 不在）
        assert _hot_names(by_dir) == [
            "20260601/core.parquet",
            "20260702/core.parquet",
            "20250615/technical.parquet",
            "20260601/technical.parquet",
        ]
        assert _archive_names(by_dir) == ["2025-06/core.parquet"]
        assert _label_names(by_dir) == [
            "y_ret_10/20260601.parquet",
            "y_ret_20/20260601.parquet",
            "y_ret_5/20260601.parquet",
        ]
        # 冻结段外与非日期命名一律排除
        flat = {p.name for p in flatten_consumption_files(by_dir)}
        assert "20260703.parquet" not in flat
        assert not any("2026-08" in f"{p.parent.name}" for p in flatten_consumption_files(by_dir))
        # manifest 族单独成列（审计，不入字节门）
        manifest_files = {p.name for p in by_dir["（manifest 族，审计不入字节门）"]}
        assert manifest_files == set(MANIFEST_FAMILY_NAMES)

    def test_default_enumeration_bit_identical(self, tmp_path):
        """⑦ 默认（False）与现状逐位一致：扩展目录存在也不泄漏进默认枚举。"""
        prod = _make_panel_production(tmp_path)
        legacy = enumerate_consumption_files(prod)
        explicit_default = enumerate_consumption_files(prod, include_panel_labels=False)
        assert legacy == explicit_default
        assert set(legacy) == {"features/cs_train", "clean/daily", "raw/suspend", "（单文件）"}
        assert {p.name for p in legacy["features/cs_train"]} == {
            "20260601.parquet",
            "20260702.parquet",
        }

    def test_moneyflow_hsgt_optional_guard(self, tmp_path):
        """R2-T5AE-02：闭市日 raw 输入可入快照且改动可被守卫检出。"""
        prod = _make_panel_production(tmp_path)
        hsgt = prod / "raw" / "moneyflow_hsgt"
        hsgt.mkdir(parents=True, exist_ok=True)
        target = hsgt / "2022-05-06.parquet"
        target.write_bytes(b"north-all-null")
        assert "raw/moneyflow_hsgt" not in enumerate_consumption_files(prod)
        by_dir = enumerate_consumption_files(prod, include_moneyflow_hsgt=True)
        names = {path.name for path in by_dir["raw/moneyflow_hsgt"]}
        assert "2022-05-06.parquet" in names
        assert "2026-06-01.parquet" in names
        snapshot = tmp_path / "consumption.json"
        collect_consumption_snapshot(by_dir["raw/moneyflow_hsgt"], base_dir=prod, out_path=snapshot)
        assert verify_consumption_snapshot(snapshot, base_dir=prod) == []
        target.write_bytes(b"north-has-values")
        drift = verify_consumption_snapshot(snapshot, base_dir=prod)
        assert drift and "moneyflow_hsgt" in drift[0]

    def test_panel_zero_coverage_rejected(self, tmp_path):
        """启用扩展后 panel 热/冷区合计零覆盖 ⇒ 拒绝（防静默漏枚举）。"""
        prod = _make_panel_production(tmp_path)
        import shutil

        shutil.rmtree(prod / "features" / "panel")
        shutil.rmtree(prod / "features" / "panel_archive")
        with pytest.raises(ValueError, match="panel 热/冷区冻结段内零覆盖"):
            enumerate_consumption_files(prod, include_panel_labels=True)

    def test_labels_missing_or_empty_rejected(self, tmp_path):
        """④ 启用扩展后 labels 目录缺失 ⇒ FileNotFoundError；空覆盖 ⇒ ValueError。"""
        prod = _make_panel_production(tmp_path)
        import shutil

        shutil.rmtree(prod / "labels")
        with pytest.raises(FileNotFoundError, match="labels 目录缺失"):
            enumerate_consumption_files(prod, include_panel_labels=True)
        (prod / "labels").mkdir()
        with pytest.raises(ValueError, match="labels 冻结段内零覆盖"):
            enumerate_consumption_files(prod, include_panel_labels=True)


# ---------------------------------------------------------------------------
# R2-T5A-02 前半：基线获取/续跑复核（字节门 + manifest 相关性 diff）
# ---------------------------------------------------------------------------


class TestExtendedBaselineAcquire:
    def test_acquire_v2_panel_layout(self, tmp_path):
        """臂③首跑：manifest 基线落盘；字节门快照不含 manifest 族、含 panel/labels。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        result = acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        manifest_path = run_dir / "manifest_baseline.json"
        assert result["manifest_baseline_path"] == manifest_path
        assert manifest_path.exists()
        baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert baseline["frozen_segment_end"] == _FROZEN_END
        assert baseline["wal_pending"] is False
        assert set(baseline["scope_days"]) == {"20250601", "20250615", "20260601", "20260702"}
        assert len(baseline["audit_identities"]) == len(MANIFEST_FAMILY_NAMES)
        records = json.loads((run_dir / "consumption_snapshot.json").read_text(encoding="utf-8"))
        display_paths = [str(r["path"]) for r in records]
        assert any("features/panel/20260601/core.parquet" in p for p in display_paths)
        assert any("features/panel_archive/2025-06/core.parquet" in p for p in display_paths)
        assert any("labels/y_ret_5/20260601.parquet" in p for p in display_paths)
        assert not any(name in p for p in display_paths for name in MANIFEST_FAMILY_NAMES)
        counts = result["consumption_counts"]
        assert counts["features/panel"] == 4 and counts["features/panel_archive"] == 1
        assert counts["labels"] == 3

    def test_default_arm_acquire_bit_identical(self, tmp_path):
        """⑦ 默认臂（①②）基线获取与 spike 现状逐位一致：无 manifest 基线、快照相同。"""
        prod = _make_panel_production(tmp_path)
        (tmp_path / "r1").mkdir(parents=True)
        (tmp_path / "r2").mkdir(parents=True)
        first = acquire_arm_baselines(tmp_path / "r1", prod, False)
        second = acquire_arm_baselines(tmp_path / "r2", prod, False, arm=ARM_V2_CS_TRAIN_VERIFY)
        assert first["manifest_baseline_path"] is None
        assert second["manifest_baseline_path"] is None
        assert not (tmp_path / "r1" / "manifest_baseline.json").exists()
        assert (tmp_path / "r1" / "consumption_snapshot.json").read_bytes() == (
            tmp_path / "r2" / "consumption_snapshot.json"
        ).read_bytes()

    def test_tampered_panel_and_labels_rejected_on_resume(self, tmp_path):
        """② 集合内 panel/labels 文件篡改 ⇒ 续跑复核拒绝（字节门覆盖扩展目录）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        for target in (
            prod / "features" / "panel" / "20260601" / "core.parquet",
            prod / "features" / "panel_archive" / "2025-06" / "core.parquet",
            prod / "labels" / "y_ret_5" / "20260601.parquet",
        ):
            original = target.read_bytes()
            target.write_bytes(b"tampered")
            drift = verify_consumption_snapshot(
                run_dir / "consumption_snapshot.json", base_dir=tmp_path
            )
            assert any(target.name in item for item in drift), target
            with pytest.raises(RuntimeError, match="续跑基线漂移"):
                acquire_arm_baselines(run_dir, prod, True, arm=ARM_V2_PANEL)
            target.write_bytes(original)

    def test_out_of_set_append_passes(self, tmp_path):
        """③ 集合外追加（冻结段外更晚日期分区）⇒ 续跑放行（M5 口径沿用到 panel/labels）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        (prod / "features" / "panel" / "20260704").mkdir(parents=True)
        (prod / "features" / "panel" / "20260704" / "core.parquet").write_bytes(b"new")
        (prod / "labels" / "y_ret_5" / "20260704.parquet").write_bytes(b"new")
        (prod / "features" / "cs_train" / "20260704.parquet").write_bytes(b"new")
        assert (
            verify_consumption_snapshot(run_dir / "consumption_snapshot.json", base_dir=tmp_path)
            == []
        )
        result = acquire_arm_baselines(run_dir, prod, True, arm=ARM_V2_PANEL)
        assert result["resumed"] is True

    def test_manifest_in_window_rewrite_rejected(self, tmp_path):
        """⑤ 窗内登记改写 ⇒ 相关性 diff 拒绝（panel/冷区/labels/sealed_through 全列）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        baseline_path = run_dir / "manifest_baseline.json"
        manifest_path = prod / "features" / "manifest.json"
        original = manifest_path.read_text(encoding="utf-8")

        def _rewrite(mutate) -> List[str]:
            doc = json.loads(original)
            mutate(doc)
            manifest_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            try:
                return verify_manifest_baseline(baseline_path, prod)
            finally:
                manifest_path.write_text(original, encoding="utf-8")

        issues = _rewrite(
            lambda doc: doc["partitions"]["panel/20260601/core.parquet"].update(sha256_16="f" * 16)
        )
        assert any("panel/20260601/core.parquet" in item for item in issues)
        issues = _rewrite(
            lambda doc: doc["partitions"]["panel_archive/2025-06/core.parquet"].update(rows=21)
        )
        assert any("panel_archive/2025-06/core.parquet" in item for item in issues)
        issues = _rewrite(
            lambda doc: doc["labels"]["y_ret_5"]["partitions"]["20260601.parquet"].update(rows=3)
        )
        assert any("y_ret_5/20260601.parquet" in item for item in issues)
        issues = _rewrite(lambda doc: doc["labels"]["y_ret_5"].update(sealed_through="20260701"))
        assert any("sealed_through" in item for item in issues)
        issues = _rewrite(
            lambda doc: doc["row_domain_segments"].update(label_filtered_through="20260701")
        )
        assert any("row_domain_segments" in item for item in issues)

    def test_manifest_out_of_window_append_passes(self, tmp_path):
        """⑤ 窗口外正常追加登记（> 冻结段末）⇒ 放行（T1 reviewer2-N2 口径沿用）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        manifest_path = prod / "features" / "manifest.json"
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        doc["partitions"]["panel/20260710/core.parquet"] = {
            "sha256_16": "e" * 16,
            "rows": 2,
            "written_at": "2026-07-10T00:00:00",
            "zone": "hot",
            "row_domain": "structural",
        }
        doc["labels"]["y_ret_5"]["partitions"]["20260710.parquet"] = {
            "sha256_16": "d" * 16,
            "rows": 2,
            "written_at": "2026-07-10T00:00:00",
            "row_domain": "structural",
        }
        manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        assert verify_manifest_baseline(run_dir / "manifest_baseline.json", prod) == []
        result = acquire_arm_baselines(run_dir, prod, True, arm=ARM_V2_PANEL)
        assert result["resumed"] is True

    def test_manifest_wal_corruption_rejected(self, tmp_path):
        """WAL 待处理态/内容运行期损坏 ⇒ 复核解析失败判漂移（不静默放行）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        (prod / "features" / "manifest_wal.jsonl").write_bytes(b"not-json\n")
        issues = verify_manifest_baseline(run_dir / "manifest_baseline.json", prod)
        assert any("复核解析失败" in item for item in issues)
        with pytest.raises(RuntimeError, match="续跑基线漂移"):
            acquire_arm_baselines(run_dir, prod, True, arm=ARM_V2_PANEL)

    def test_manifest_baseline_missing_with_cache_rejected(self, tmp_path):
        """启用扩展后基线齐备口径 = 三份：缺 manifest 基线且有缓存 ⇒ 拒绝（不重建）。"""
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        (run_dir / "manifest_baseline.json").unlink()
        (run_dir / "folds" / "split00").mkdir(parents=True)
        (run_dir / "folds" / "split00" / "fingerprint.json").write_text("{}", encoding="utf-8")
        for skip in (False, True):
            with pytest.raises(RuntimeError, match="manifest_baseline"):
                acquire_arm_baselines(run_dir, prod, skip, arm=ARM_V2_PANEL)
            assert not (run_dir / "manifest_baseline.json").exists()


# ---------------------------------------------------------------------------
# R2-T5A-02 后半：打分 store 实际读取 receipt ⊆ 消费集合
# ---------------------------------------------------------------------------


def _receipt(kind: str, paths: List[str]) -> StoreReadReceipt:
    files = tuple({"path": p, "sha256": "0" * 64, "size": 1} for p in paths)
    return StoreReadReceipt(
        kind=kind,
        start_date="20250615",
        end_date=_FROZEN_END,
        files=files,
        collected_at="2026-10-24T00:00:00",
    )


class TestScoreReceiptGate:
    def _baseline(self, tmp_path: Path):
        prod = _make_panel_production(tmp_path)
        run_dir = tmp_path / "frozen" / "r"
        run_dir.mkdir(parents=True)
        acquire_arm_baselines(run_dir, prod, False, arm=ARM_V2_PANEL)
        return prod, run_dir / "consumption_snapshot.json"

    def test_receipt_within_consumption_passes(self, tmp_path):
        """⑥ 正例：实际读取（多次读取累积）全部落在消费集合内 ⇒ 放行。"""
        prod, snapshot = self._baseline(tmp_path)
        store = SimpleNamespace(
            read_receipts=(
                _receipt(
                    "features",
                    [
                        "features/panel/20260601/core.parquet",
                        "features/panel_archive/2025-06/core.parquet",
                    ],
                ),
                _receipt("labels", ["labels/y_ret_5/20260601.parquet"]),
            )
        )
        report = assert_score_receipt_within_consumption(
            store, snapshot, base_dir=tmp_path, production_data_root=prod
        )
        assert report["pass"] is True and report["receipt_files"] == 3
        assert report["receipt_count"] == 2
        assert collect_score_read_receipt_files(store)[0] == (
            "features/panel/20260601/core.parquet"
        )

    def test_receipt_outside_consumption_rejected(self, tmp_path):
        """⑥ 反例：读取了集合外文件（冻结段外分区 / 从未枚举的族）⇒ 拒绝并列出。"""
        prod, snapshot = self._baseline(tmp_path)
        store = SimpleNamespace(
            read_receipts=(
                _receipt(
                    "features",
                    [
                        "features/panel/20260601/core.parquet",
                        "features/panel/20260703/core.parquet",  # 存在但冻结段外，未入集合
                        "features/panel/20260601/moneyflow.parquet",  # 从未存在
                    ],
                ),
            )
        )
        with pytest.raises(RuntimeError, match="越出消费集合") as exc_info:
            assert_score_receipt_within_consumption(
                store, snapshot, base_dir=tmp_path, production_data_root=prod
            )
        assert "20260703" in str(exc_info.value) and "moneyflow" in str(exc_info.value)

    def test_empty_receipts_pass_with_note(self, tmp_path):
        """全折复用未打分 ⇒ 无读取 receipt，空通过并登记 note。"""
        prod, snapshot = self._baseline(tmp_path)
        store = SimpleNamespace(read_receipts=())
        report = assert_score_receipt_within_consumption(
            store, snapshot, base_dir=tmp_path, production_data_root=prod
        )
        assert report["pass"] is True and "空通过" in str(report["note"])


# ---------------------------------------------------------------------------
# R2-T5A-11：双跑缓存探针
# ---------------------------------------------------------------------------


class TestArmRunCacheProbeDoubleRun:
    def test_probe_detects_single_and_double_run_layers(self, tmp_path):
        """当前层（单跑）与 run1/run2 子层（双跑）任一缓存形态 ⇒ 有缓存。"""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert arm_run_cache_exists(run_dir) is False
        # 单跑：当前层逐折指纹
        (run_dir / "folds" / "split00").mkdir(parents=True)
        (run_dir / "folds" / "split00" / "fingerprint.json").write_text("{}", encoding="utf-8")
        assert arm_run_cache_exists(run_dir) is True

        double_dir = tmp_path / "double"
        (double_dir / "run1" / "folds" / "split00").mkdir(parents=True)
        fp = double_dir / "run1" / "folds" / "split00" / "fingerprint.json"
        fp.write_text("{}", encoding="utf-8")
        assert arm_run_cache_exists(double_dir) is True
        fp.unlink()
        (double_dir / "run2" / "runs" / "batch_x").mkdir(parents=True)
        assert arm_run_cache_exists(double_dir) is True  # run2/runs 批次目录非空

    def test_double_run_cache_missing_baselines_rejected_no_snapshot_rebuilt(self, tmp_path):
        """双跑布局有缓存 + 基线缺失 ⇒ 拒绝（不带续跑参数同拒绝）且快照未重建。"""
        prod = _make_panel_production(tmp_path)
        for layer in ("run1", "run2"):
            run_dir = tmp_path / f"frozen_{layer}" / "r"
            cache = run_dir / layer / "folds" / "split00"
            cache.mkdir(parents=True)
            (cache / "fingerprint.json").write_text("{}", encoding="utf-8")
            for skip in (False, True):
                with pytest.raises(RuntimeError, match="基线缺失"):
                    acquire_arm_baselines(run_dir, prod, skip)
                assert not (run_dir / "before_guard.json").exists()
                assert not (run_dir / "consumption_snapshot.json").exists()


# ---------------------------------------------------------------------------
# P2-3：B0' 冻结 runs 收益路径可解析
# ---------------------------------------------------------------------------


def _make_freezable_arm_dir(tmp_path: Path) -> Path:
    """合成臂①全产物（批指纹 + 链式指标 + 逐折三件套 + runs 批收益实体）。"""
    arm_dir = tmp_path / "frozen" / "t5_legacy"
    fold = arm_dir / "folds" / "split00"
    fold.mkdir(parents=True, exist_ok=True)
    (fold / "model_version.txt").write_text("1\n", encoding="utf-8")
    (fold / "boundary.json").write_text("{}", encoding="utf-8")
    (fold / "fingerprint.json").write_text(json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")
    (arm_dir / "batch_fingerprint.json").write_text(
        json.dumps({"folds": {"split00": "a" * 64}, "batch_fingerprint": "b" * 64}),
        encoding="utf-8",
    )
    (arm_dir / "chain_metrics.json").write_text(
        json.dumps({"cagr": 0.1, "trading_days": 3}), encoding="utf-8"
    )
    batch = arm_dir / "runs" / "batch_x"
    (batch / "folds" / "split00").mkdir(parents=True, exist_ok=True)
    (batch / "batch_meta.json").write_text(
        json.dumps(
            {
                "data_state": {"data_state_id": "ds_x"},
                "code_state": {"git_commit": "t", "git_dirty": False},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2026-01-05"], "nav": [1.0]}).to_parquet(
        batch / "chain_nav.parquet", index=False
    )
    pd.DataFrame({"split_index": [0]}).to_parquet(batch / "summary.parquet", index=False)
    pd.DataFrame({"date": ["2026-01-05"], "nav": [1.0]}).to_parquet(
        batch / "folds" / "split00" / "daily.parquet", index=False
    )
    return arm_dir


class TestB0PrimeRunsPaths:
    def _freeze(self, tmp_path: Path) -> Path:
        return freeze_b0_prime(
            _make_freezable_arm_dir(tmp_path),
            run_name="t5_legacy",
            producer_identity={"arm": ARM_LEGACY_CS_TRAIN},
            config_digest="c" * 64,
        )

    def test_freeze_copies_runs_payload_and_manifest_paths_resolve(self, tmp_path):
        """冻结复制 runs 批收益实体；manifest 所列全部相对路径在 b0_prime/ 下可解析。"""
        target = self._freeze(tmp_path)
        manifest = json.loads((target / "freeze_manifest.json").read_text(encoding="utf-8"))
        batch_id = manifest["runs_batch"]["batch_id"]
        assert (target / "runs" / batch_id / "chain_nav.parquet").is_file()
        assert (target / "runs" / batch_id / "summary.parquet").is_file()
        assert (target / "runs" / batch_id / "folds" / "split00" / "daily.parquet").is_file()
        listed = [manifest["runs_batch"]["chain_nav_path"]]
        for entry in manifest["folds"].values():
            listed += [entry["fingerprint_path"], entry["boundary_path"], entry["daily_nav_path"]]
        for rel in listed:
            assert (target / rel).is_file(), rel

    def test_idempotent_reuse_requires_manifest_paths(self, tmp_path):
        """幂等复用分支同样断言路径可解析：收益实体被删 ⇒ 拒绝复用（防伪冻结）。"""
        target = self._freeze(tmp_path)
        again = freeze_b0_prime(
            _make_freezable_arm_dir(tmp_path),
            run_name="t5_legacy",
            producer_identity={"arm": ARM_LEGACY_CS_TRAIN},
            config_digest="c" * 64,
        )
        assert again == target  # 身份一致 + 路径齐全 ⇒ 幂等复用
        (target / "runs" / "batch_x" / "chain_nav.parquet").unlink()
        with pytest.raises(RuntimeError, match="不可解析"):
            self._freeze(tmp_path)
        # daily.parquet 缺失同样拒绝
        target2 = self._freeze(tmp_path / "second")
        (target2 / "runs" / "batch_x" / "folds" / "split00" / "daily.parquet").unlink()
        with pytest.raises(RuntimeError, match="不可解析"):
            freeze_b0_prime(
                _make_freezable_arm_dir(tmp_path / "second"),
                run_name="t5_legacy",
                producer_identity={"arm": ARM_LEGACY_CS_TRAIN},
                config_digest="c" * 64,
            )

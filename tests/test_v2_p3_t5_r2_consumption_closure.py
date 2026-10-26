# -*- coding: utf-8 -*-
"""v2 P3 T5 评审复审（reviewer2）处置回归测试：消费集合闭合核验 + 读取门。

覆盖（全合成数据，不读取真实生产数据/产物）：
- 共享核验（``t05b_isolated_root``，R2-T5AE-R1-02）：候选缺失态登记（``missing``）+
  闭合核验三情形（快照后新增=重枚举发现 / 候选由缺转现 / 已有文件内容变更）+
  旧格式快照向后兼容；
- 补评分入口（``t5_guarded_rescore``）：reviewer2 贯通反例复现——评分期间新建
  采集时缺失的 ``raw/moneyflow_hsgt`` 候选 ⇒ FAIL（对照：保持缺失 ⇒ PASS）；
- 主臂运行后复核（``t5_arm_artifacts.verify_after_guards``）：运行后重枚举发现
  消费目录内新增文件 ⇒ ``consumption_drift`` 非空（与基线采集同源同口径）；
- 读取门与台账（``t05b_stock_connect``，R2-T5AE-R2-01）：读取动作绑定冻结快照
  身份——「创建→真实闭市读取→移除」在读取动作处**立即拒绝**（不再依赖收尾
  目录重枚举）；快照外实读 / 读时内容与登记身份不符 ⇒ 立即拒绝；未激活 =
  现行为逐位不变。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3 import t5_guarded_rescore as rescore_mod  # noqa: E402
from scripts.v2_p3.t5_arm_artifacts import verify_after_guards  # noqa: E402
from scripts.v2_p3.t05b_baselines import enumerate_consumption_files  # noqa: E402
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    collect_consumption_snapshot,
    snapshot_production_models_guard,
    verify_consumption_snapshot_closed,
)
from scripts.v2_p3.t05b_stock_connect import (  # noqa: E402
    raw_read_guard,
    resolve_stock_connect_closure_days,
)

_DAY = "20211224"
_DAY_DASH = "2021-12-24"
_FP = "a" * 64
_REQUIRED_COLUMNS = ("north_net_buy", "ret_5")


def _write_trade_cal(root: Path) -> None:
    path = root / "clean" / "trade_cal.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"cal_date": [_DAY], "is_open": [1]})
    frame.to_parquet(path, index=False)


def _dated_parquet(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 共享核验单元（t05b_isolated_root）
# ---------------------------------------------------------------------------


class TestClosedVerificationUnit:
    def _snapshot(self, tmp_path: Path, *, with_missing: bool = True) -> tuple[Path, Path]:
        root = tmp_path / "data"
        _dated_parquet(root / "features" / "cs_train" / f"{_DAY}.parquet")
        _dated_parquet(root / "clean" / "daily" / f"{_DAY_DASH}.parquet")
        missing = [root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet"] if with_missing else []
        snapshot = tmp_path / "consumption_before.json"
        collect_consumption_snapshot(
            [root / "features" / "cs_train" / f"{_DAY}.parquet"],
            base_dir=root,
            out_path=snapshot,
            missing_files=missing,
        )
        return root, snapshot

    def test_missing_candidate_stays_missing_passes(self, tmp_path):
        """采集时缺失的候选保持缺失 ⇒ 不误报（对照基线）。"""
        root, snapshot = self._snapshot(tmp_path)
        drift = verify_consumption_snapshot_closed(snapshot, base_dir=root)
        assert drift == []

    def test_missing_candidate_appears_fails(self, tmp_path):
        """采集时缺失的候选在核验前出现 ⇒ 快照后新增实读候选，FAIL。"""
        root, snapshot = self._snapshot(tmp_path)
        _dated_parquet(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        drift = verify_consumption_snapshot_closed(snapshot, base_dir=root)
        assert any("采集时缺失现已出现" in item for item in drift)
        assert any("moneyflow_hsgt" in item for item in drift)

    def test_rescan_finds_new_file_fails(self, tmp_path):
        """评分后重枚举发现快照外现存文件（消费目录内新增）⇒ FAIL。"""
        root, snapshot = self._snapshot(tmp_path, with_missing=False)
        new_file = _dated_parquet(root / "features" / "cs_train" / "20211225.parquet")
        drift = verify_consumption_snapshot_closed(snapshot, base_dir=root, rescan_paths=[new_file])
        assert any("重枚举发现" in item for item in drift)

    def test_content_change_still_detected(self, tmp_path):
        """已有文件内容变更 ⇒ 清单身份哈希漂移仍 FAIL（原语义保持）。"""
        root, snapshot = self._snapshot(tmp_path, with_missing=False)
        target = root / "features" / "cs_train" / f"{_DAY}.parquet"
        pd.DataFrame({"x": [2]}).to_parquet(target, index=False)
        drift = verify_consumption_snapshot_closed(snapshot, base_dir=root)
        assert len(drift) == 1

    def test_legacy_snapshot_without_missing_records_compatible(self, tmp_path):
        """旧格式快照（无 missing 记录）向后兼容；rescan 与清单一致 ⇒ 零漂移。"""
        root, snapshot = self._snapshot(tmp_path, with_missing=False)
        records = json.loads(snapshot.read_text(encoding="utf-8"))
        assert all(not record.get("missing") for record in records)
        rescan = [root / "features" / "cs_train" / f"{_DAY}.parquet"]
        drift = verify_consumption_snapshot_closed(snapshot, base_dir=root, rescan_paths=rescan)
        assert drift == []


# ---------------------------------------------------------------------------
# 补评分入口端到端（reviewer2 反例复现转 FAIL）
# ---------------------------------------------------------------------------


def _build_arm(tmp_path: Path) -> Path:
    """最小臂布局：arm_meta + split06 台账/边界/指纹 + 隔离根最小数据。"""
    arm_dir = tmp_path / "arm"
    root = arm_dir / "root"
    (arm_dir / "folds" / "split06").mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    (arm_dir / "arm_meta.json").write_text(json.dumps({"arm": "legacy_cs_train"}), encoding="utf-8")
    (arm_dir / "folds" / "split06" / "model_version.txt").write_text("1\n", encoding="utf-8")
    (arm_dir / "folds" / "split06" / "boundary.json").write_text(
        json.dumps({"test_start": _DAY, "test_end": _DAY}), encoding="utf-8"
    )
    (arm_dir / "folds" / "split06" / "fingerprint.json").write_text(
        json.dumps({"fingerprint": _FP, "excluded_days": []}), encoding="utf-8"
    )
    _write_trade_cal(root)
    _dated_parquet(root / "features" / "cs_train" / f"{_DAY}.parquet")
    models = root / "models" / "stock_selection"
    models.mkdir(parents=True, exist_ok=True)
    (models / "v1_features.json").write_text("{}", encoding="utf-8")
    (models / "v1_metadata.json").write_text("{}", encoding="utf-8")
    (models / "v1_model.joblib").write_bytes(b"model")
    return arm_dir


class TestGuardedRescoreLateAppearingFile:
    @pytest.fixture()
    def arm_dir(self, tmp_path: Path) -> Path:
        return _build_arm(tmp_path)

    def _run(self, arm_dir: Path, tmp_path: Path, stub: Callable[..., Any]) -> dict[str, Any]:
        return rescore_mod.guarded_rescore_fold(arm_dir, "split06", tmp_path / "out", score_fn=stub)

    def test_candidates_without_file_passes(self, arm_dir, tmp_path, monkeypatch):
        """对照：缺失候选（moneyflow/suspend/clean daily）保持缺失 ⇒ PASS 不误报。"""
        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        verdict = self._run(
            arm_dir, tmp_path, lambda **_k: {"fingerprint": _FP, "excluded_days": []}
        )
        assert verdict["verdict"] == "PASS", verdict["after_drift"]
        assert verdict["missing_candidates"] >= 3

    def test_late_appearing_raw_file_fails(self, arm_dir, tmp_path, monkeypatch):
        """reviewer2 反例：评分期间新建采集时缺失的 raw 候选 ⇒ FAIL（快照后新增实读候选）。"""

        def _create_during_scoring(**_kwargs: Any) -> dict[str, Any]:
            _dated_parquet(arm_dir / "root" / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
            return {"fingerprint": _FP, "excluded_days": []}

        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        verdict = self._run(arm_dir, tmp_path, _create_during_scoring)
        assert verdict["verdict"] == "FAIL"
        assert any("采集时缺失现已出现" in item for item in verdict["after_drift"])
        assert any("moneyflow_hsgt" in item for item in verdict["after_drift"])

    def test_out_of_window_new_file_not_flagged(self, arm_dir, tmp_path, monkeypatch):
        """评分窗外日期的新增文件（日更追加到评分范围之外）不触发（reviewer2 口径：
        重枚举闭合 = 本次评分候选域；窗外追加不属于实读候选）。"""

        def _create_out_of_window(**_kwargs: Any) -> dict[str, Any]:
            _dated_parquet(arm_dir / "root" / "features" / "cs_train" / "20211227.parquet")
            return {"fingerprint": _FP, "excluded_days": []}

        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        verdict = self._run(arm_dir, tmp_path, _create_out_of_window)
        assert verdict["verdict"] == "PASS"

    def test_in_window_file_replaced_fails(self, arm_dir, tmp_path, monkeypatch):
        """评分期间替换窗内已有候选（同路径不同内容）⇒ 清单身份漂移 FAIL。"""

        def _replace_during_scoring(**_kwargs: Any) -> dict[str, Any]:
            path = arm_dir / "root" / "features" / "cs_train" / f"{_DAY}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"x": [2]}).to_parquet(path, index=False)
            return {"fingerprint": _FP, "excluded_days": []}

        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        verdict = self._run(arm_dir, tmp_path, _replace_during_scoring)
        assert verdict["verdict"] == "FAIL"
        assert len(verdict["after_drift"]) == 1


# ---------------------------------------------------------------------------
# 主臂运行后复核接线（verify_after_guards）
# ---------------------------------------------------------------------------


def _build_production(tmp_path: Path) -> Path:
    production = tmp_path / "prod" / "data"
    _dated_parquet(production / "features" / "cs_train" / f"{_DAY}.parquet")
    _dated_parquet(production / "clean" / "daily" / f"{_DAY_DASH}.parquet")
    _dated_parquet(production / "raw" / "suspend" / f"{_DAY_DASH}.parquet")
    _dated_parquet(production / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
    (production / "clean" / "stock_basic.parquet").write_bytes(b"sb")
    _write_trade_cal(production)
    models = production / "models" / "stock_selection"
    models.mkdir(parents=True, exist_ok=True)
    (models / "v1_model.joblib").write_bytes(b"model")
    (models / "v1_metadata.json").write_text("{}", encoding="utf-8")
    (models / "ml_train_runs.csv").write_text("a,b\n1,1\n", encoding="utf-8")
    return production


def _build_baselines(tmp_path: Path, production: Path) -> dict[str, Any]:
    by_dir = enumerate_consumption_files(production, include_moneyflow_hsgt=True)
    files = [path for paths in by_dir.values() for path in paths]
    snapshot = tmp_path / "consumption_snapshot.json"
    collect_consumption_snapshot(files, base_dir=production.resolve().parent, out_path=snapshot)
    return {
        "guard_before": snapshot_production_models_guard(production),
        "consumption_snapshot_path": snapshot,
        "base_dir": production.resolve().parent,
        "production_data_root": production,
    }


class TestVerifyAfterGuardsClosed:
    def test_no_drift_passes(self, tmp_path):
        """正常路径：生产模型守卫 + 消费闭合核验均零漂移。"""
        production = _build_production(tmp_path)
        arm_dir = tmp_path / "arm"
        arm_dir.mkdir()
        (arm_dir / "arm_meta.json").write_text(
            json.dumps({"arm": "legacy_cs_train"}), encoding="utf-8"
        )
        report = verify_after_guards(arm_dir, _build_baselines(tmp_path, production))
        assert report["consumption_drift"] == []
        assert report["guard_drift"] == []

    def test_new_consumption_file_after_run_fails(self, tmp_path):
        """运行后消费目录内新增文件（快照采集后重枚举发现）⇒ consumption_drift 非空。"""
        production = _build_production(tmp_path)
        arm_dir = tmp_path / "arm"
        arm_dir.mkdir()
        (arm_dir / "arm_meta.json").write_text(
            json.dumps({"arm": "legacy_cs_train"}), encoding="utf-8"
        )
        baselines = _build_baselines(tmp_path, production)
        _dated_parquet(production / "features" / "cs_train" / "20211225.parquet")
        report = verify_after_guards(arm_dir, baselines)
        assert any("重枚举发现" in item for item in report["consumption_drift"])


# ---------------------------------------------------------------------------
# 读取门与台账（R2-T5AE-R2-01：读取动作绑定冻结快照身份）
# ---------------------------------------------------------------------------


def _write_closure_raw(path: Path) -> Path:
    """合法闭市 raw 分区（北向三列全空）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"hgt": [None], "sgt": [None], "north_money": [None]}).to_parquet(
        path, index=False
    )
    return path


def _resolve_closure(root: Path, days: tuple[str, ...]) -> dict[str, str]:
    """真实闭市判定（required_columns 含 north_net_buy 才走 raw 读取通路）。"""
    return resolve_stock_connect_closure_days(Path(root), list(days), list(_REQUIRED_COLUMNS))


def _root_with_cs_train_north_null(tmp_path: Path) -> Path:
    """合成根：cs_train 分区 north_net_buy 全空（源侧校验可通过）。"""
    root = tmp_path / "root"
    path = root / "features" / "cs_train" / f"{_DAY}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"north_net_buy": [None], "x": [1.0]}).to_parquet(path, index=False)
    return root


class TestRawReadGuard:
    """读取门单元（真实闭市判定通路；未激活 = 现行为逐位不变）。"""

    def test_missing_candidate_read_rejected_immediately(self, tmp_path):
        """跑前缺失候选一旦被实际读取（真实闭市判定）⇒ 立即拒绝，与收尾核验无关。"""
        root = _root_with_cs_train_north_null(tmp_path)
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        registered = {
            (root / "features" / "cs_train" / f"{_DAY}.parquet").resolve().as_posix(): "0" * 64
        }
        with pytest.raises(RuntimeError, match="跑前缺失候选被实际读取"):
            with raw_read_guard(missing_paths=[raw_path], registered_identities=registered):
                _resolve_closure(root, (_DAY,))

    def test_out_of_snapshot_read_rejected(self, tmp_path):
        """快照外实读候选（不在登记身份集合）⇒ 立即拒绝。"""
        root = _root_with_cs_train_north_null(tmp_path)
        _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        with pytest.raises(RuntimeError, match="快照外实读候选"):
            with raw_read_guard(missing_paths=[], registered_identities={}):
                _resolve_closure(root, (_DAY,))

    def test_content_mismatch_rejected(self, tmp_path):
        """已存在 raw 读取时点内容与登记身份不符（读时已被替换）⇒ 立即拒绝。"""
        root = _root_with_cs_train_north_null(tmp_path)
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        registered = {raw_path.resolve().as_posix(): "0" * 64}
        with pytest.raises(RuntimeError, match="读取内容与跑前登记身份不符"):
            with raw_read_guard(missing_paths=[], registered_identities=registered):
                _resolve_closure(root, (_DAY,))

    def test_ledger_records_passing_reads(self, tmp_path):
        """正常读取（在册且身份相符）⇒ 台账登记 (路径, 实际解析字节哈希)。"""
        root = _root_with_cs_train_north_null(tmp_path)
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        north_path = root / "features" / "cs_train" / f"{_DAY}.parquet"
        registered = {
            raw_path.resolve().as_posix(): hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            north_path.resolve().as_posix(): hashlib.sha256(north_path.read_bytes()).hexdigest(),
        }
        with raw_read_guard(missing_paths=[], registered_identities=registered) as ledger:
            excluded = _resolve_closure(root, (_DAY,))
        assert excluded == {_DAY: "stock_connect_closed"}
        # raw 与源侧 cs_train 各一条；台账哈希 = 实际解析字节（R2-T5AE-R3-01）
        assert len(ledger) == 2
        paths = {record.path for record in ledger}
        assert paths == {raw_path.resolve().as_posix(), north_path.resolve().as_posix()}

    def test_inactive_guard_behavior_unchanged(self, tmp_path):
        """未激活读取门：闭市判定现行为逐位不变（正常读取、正常返回）。"""
        root = _root_with_cs_train_north_null(tmp_path)
        _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        assert _resolve_closure(root, (_DAY,)) == {_DAY: "stock_connect_closed"}


class TestReadByteBinding:
    """R2-T5AE-R3-01：校验/使用分离消除（单次读取 → 校验字节 → 解析同一缓冲）。"""

    def test_activated_parse_uses_verified_buffer_not_path(self, tmp_path, monkeypatch):
        """激活态解析消费已校验字节缓冲：按路径的二次读取为零（分离消除证明）。"""
        root = _root_with_cs_train_north_null(tmp_path)
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        north_path = root / "features" / "cs_train" / f"{_DAY}.parquet"
        registered = {
            raw_path.resolve().as_posix(): hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            north_path.resolve().as_posix(): hashlib.sha256(north_path.read_bytes()).hexdigest(),
        }
        sources: list[object] = []
        original = pd.read_parquet

        def _spy(source, *args, **kwargs):
            sources.append(type(source).__name__)
            return original(source, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", _spy)
        with raw_read_guard(missing_paths=[], registered_identities=registered):
            excluded = _resolve_closure(root, (_DAY,))
        assert excluded == {_DAY: "stock_connect_closed"}
        assert "BytesIO" in sources and "Path" not in sources
        assert sources.count("BytesIO") == 2  # raw + 源侧各一次

    def test_cs_train_content_swap_rejected_immediately(self, tmp_path):
        """reviewer2 §2.2 反例：raw 合法闭市 + 评分期间 cs_train 分区被改空 ⇒ ③立即拒绝。

        修复前成员门只查路径——消费未登记字节（改空 → 排除日误判）后恢复即 PASS；
        ③内容身份落地后在已校验字节处拒绝，与收尾核验无关。
        """
        root = tmp_path / "root"
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        north_path = root / "features" / "cs_train" / f"{_DAY}.parquet"
        north_path.parent.mkdir(parents=True, exist_ok=True)
        # 跑前登记内容：north_net_buy 有值（按此内容哈希登记）
        pd.DataFrame({"north_net_buy": [1.0], "x": [1.0]}).to_parquet(north_path, index=False)
        registered = {
            raw_path.resolve().as_posix(): hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            north_path.resolve().as_posix(): hashlib.sha256(north_path.read_bytes()).hexdigest(),
        }
        # 评分期间临时替换：north_net_buy 全空（若被消费将误判为合法排除日）
        pd.DataFrame({"north_net_buy": [None], "x": [1.0]}).to_parquet(north_path, index=False)
        with pytest.raises(RuntimeError, match="读取内容与跑前登记身份不符"):
            with raw_read_guard(missing_paths=[], registered_identities=registered):
                _resolve_closure(root, (_DAY,))
        # 对照：恢复跑前内容后同门通过（源侧有值 ⇒ 不排除）
        pd.DataFrame({"north_net_buy": [1.0], "x": [1.0]}).to_parquet(north_path, index=False)
        with raw_read_guard(missing_paths=[], registered_identities=registered):
            assert _resolve_closure(root, (_DAY,)) == {}

    def test_raw_removed_after_existence_check_not_excluded(self, tmp_path):
        """竞态安全：is_file 通过后文件被移除 ⇒ 按原「文件缺失不排除」语义返回空。"""
        root = _root_with_cs_train_north_null(tmp_path)
        raw_path = _write_closure_raw(root / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet")
        registered = {raw_path.resolve().as_posix(): "0" * 64}
        with raw_read_guard(missing_paths=[], registered_identities=registered):
            raw_path.unlink()
            assert _resolve_closure(root, (_DAY,)) == {}


class TestRescoreCreateReadRemoveRejected:
    """reviewer2 复审二 §2.1 反例复现：「创建→真实闭市读取→移除」立即拒绝。"""

    @pytest.fixture()
    def arm_dir(self, tmp_path: Path) -> Path:
        return _build_arm(tmp_path)

    def test_create_read_remove_rejected_immediately(self, arm_dir, tmp_path, monkeypatch):
        """新建缺失候选→真实闭市判定读取→收尾前移除：读取动作处立即拒绝。"""
        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        raw_path = arm_dir / "root" / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet"

        def _create_read_remove(**_kwargs: Any) -> dict[str, Any]:
            _write_closure_raw(raw_path)
            excluded = _resolve_closure(arm_dir / "root", (_DAY,))
            raw_path.unlink()  # 收尾前恢复跑前缺失态（reviewer2 反例关键动作）
            return {"fingerprint": _FP, "excluded_days": sorted(excluded)}

        with pytest.raises(RuntimeError, match="跑前缺失候选被实际读取"):
            rescore_mod.guarded_rescore_fold(
                arm_dir, "split06", tmp_path / "out", score_fn=_create_read_remove
            )

    def test_normal_rescore_records_raw_reads(self, arm_dir, tmp_path, monkeypatch):
        """正例保持：现存 moneyflow 经真实闭市判定读取 ⇒ gate 通过 + 台账登记。"""
        monkeypatch.setattr(rescore_mod, "load_main_board_codes", lambda *a, **k: set())
        raw_path = _write_closure_raw(
            arm_dir / "root" / "raw" / "moneyflow_hsgt" / f"{_DAY_DASH}.parquet"
        )
        north_path = arm_dir / "root" / "features" / "cs_train" / f"{_DAY}.parquet"
        north_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [1], "north_net_buy": [None]}).to_parquet(north_path, index=False)
        # 真实闭市判定将排除 20211224——同步既有折指纹的排除日（防 days_match 误判）
        fp_path = arm_dir / "folds" / "split06" / "fingerprint.json"
        fp_path.write_text(
            json.dumps({"fingerprint": _FP, "excluded_days": [_DAY]}), encoding="utf-8"
        )

        def _score(**_kwargs: Any) -> dict[str, Any]:
            excluded = _resolve_closure(arm_dir / "root", (_DAY,))
            return {"fingerprint": _FP, "excluded_days": sorted(excluded)}

        verdict = rescore_mod.guarded_rescore_fold(
            arm_dir, "split06", tmp_path / "out", score_fn=_score
        )
        assert verdict["verdict"] == "PASS", verdict["after_drift"]
        # 台账 = raw + 源侧 cs_train 各一条（候选日两条读取路径）
        assert len(verdict["raw_reads"]) == 2
        read_paths = {entry["path"] for entry in verdict["raw_reads"]}
        assert raw_path.resolve().as_posix() in read_paths

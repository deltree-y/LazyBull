# -*- coding: utf-8 -*-
"""v2 P3 T1 冻结段小窗对账工具测试（合成数据，不触碰真实 data/）。

覆盖（P3 规划 §4.2 三类结论 + G-code 数据前提 + R2-P3-T1-01 输入稳定硬门）：
- 全等情形：panel 腿 381 列容差内一致、键集一致、cs_train 验证源腿逐位一致 ⇒ PASS；
- 已批准差异项：D-12 豁免列（days_to_unlock）注入差异 ⇒ 登记为已批准、verdict 仍 PASS；
- 拒绝项：普通列注入差异（待调查/越界）⇒ FAIL；键集不一致 ⇒ FAIL；
- 结构负例：missing_ref 日 / cols_only 越界日 / 两侧重复键 ⇒ FAIL（reviewer1-P3-5）；
- 输入稳定硬门（R2-P3-T1-01）：两腿之间篡改 cs_train / manifest / 输入水位 ⇒ FAIL；
  消费集合外追加 ⇒ 放行；按族冷热混合清单覆盖；panel_reconcile 私有依赖签名锁定。
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t1_reconcile_adapter import (  # noqa: E402
    _panel_partition_files,
    enumerate_window_input_files,
    run_window_reconcile,
)
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import (  # noqa: E402
    LABEL_TABLES,
    PANEL_GROUPS,
    group_of_column,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_reconcile import PANEL_FEATURE_COLUMNS  # noqa: E402

_DAYS = ["20260105", "20260106"]
_CODES = ["600000.SH", "000001.SZ"]
_LABEL_COLUMNS = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]


def _build_dual_root(
    tmp_path: Path,
    *,
    panel_overrides: dict[str, float] | None = None,
    cs_train_overrides: dict[str, float] | None = None,
    extra_cs_train_code: str | None = None,
    extra_cs_train_column: str | None = None,
    duplicate_cs_train_key: bool = False,
    skip_cs_train_days: tuple[str, ...] = (),
) -> Path:
    """构建同时含 panel store（八族 + 三表标签）与 cs_train 分区的合成数据根。"""
    calendar = pd.date_range("2026-01-05", periods=70, freq="B").strftime("%Y%m%d").tolist()
    clean = tmp_path / "clean"
    (clean / "daily").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": calendar, "is_open": [1] * len(calendar)}).to_parquet(
        clean / "trade_cal.parquet", index=False
    )
    for day in calendar:
        pd.DataFrame({"trade_date": [day], "close": [1.0]}).to_parquet(
            clean / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )

    store = PanelDataStore(tmp_path)
    store.manifest.register_columns(
        {col: {"group": group_of_column(col), "source": "test"} for col in PANEL_FEATURE_COLUMNS}
    )
    store.manifest.save()

    panel_overrides = panel_overrides or {}
    cs_train_overrides = cs_train_overrides or {}
    for day in _DAYS:
        _append_panel_day(store, day, panel_overrides)
        _append_labels(store, day)
        if day not in skip_cs_train_days:
            _write_cs_train_day(
                tmp_path,
                day,
                cs_train_overrides,
                extra_cs_train_code,
                extra_cs_train_column,
                duplicate_cs_train_key,
            )
    return tmp_path


def _append_panel_day(store: PanelDataStore, day: str, overrides: dict[str, float]) -> None:
    for group in PANEL_GROUPS:
        payload: dict[str, list] = {"trade_date": [day] * len(_CODES), "ts_code": list(_CODES)}
        if group == "core":
            payload["ret_1"] = [0.01, 0.02]
            payload["sw_industry"] = ["银行", "地产"]
        if group == "announcement" and "days_to_unlock" in overrides:
            payload["days_to_unlock"] = [overrides["days_to_unlock"]] * len(_CODES)
        store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))


def _append_labels(store: PanelDataStore, day: str) -> None:
    for offset, name in enumerate(LABEL_TABLES):
        store.append_labels(
            name,
            pd.DataFrame(
                {
                    "ts_code": list(_CODES),
                    "trade_date": [day] * len(_CODES),
                    "label_value": [0.01 * (offset + 1)] * len(_CODES),
                    "neu_label_value": [0.02 * (offset + 1)] * len(_CODES),
                    "maturity_status": ["sealed"] * len(_CODES),
                },
                columns=_LABEL_COLUMNS,
            ),
        )


def _write_cs_train_day(
    root: Path,
    day: str,
    overrides: dict[str, float],
    extra_code: str | None,
    extra_column: str | None = None,
    duplicate_key: bool = False,
) -> None:
    cs_train = root / "features" / "cs_train"
    cs_train.mkdir(parents=True, exist_ok=True)
    codes = list(_CODES) + ([extra_code] if extra_code else [])
    if duplicate_key:
        codes = codes + [codes[0]]
    n = len(codes)
    payload: dict[str, list] = {"trade_date": [day] * n, "ts_code": codes}
    for col in PANEL_FEATURE_COLUMNS:
        if col == "ret_1":
            values = [overrides.get("ret_1", 0.01), overrides.get("ret_1", 0.02)]
            payload[col] = values + [float("nan")] * (n - 2)
        elif col in overrides:
            payload[col] = [overrides[col]] * n
        elif col == "sw_industry":
            payload[col] = ["银行", "地产"] + [None] * (n - 2)
        else:
            payload[col] = [float("nan")] * n
    for offset, (name, (value_col, neu_col)) in enumerate(LABEL_TABLES.items()):
        payload[value_col] = [0.01 * (offset + 1)] * n
        payload[neu_col] = [0.02 * (offset + 1)] * n
    if extra_column:
        payload[extra_column] = [1.0] * n
    pd.DataFrame(payload).to_parquet(cs_train / f"{day}.parquet", index=False)


class TestReconcilePass:
    def test_identical_pass(self, tmp_path):
        root = _build_dual_root(tmp_path)
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1], rationale="合成全等用例")
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is True
        assert leg["verdict_pass"] is True
        assert leg["rejected_differences"] == {}
        assert leg["key_diff_days"] == []
        assert leg["duplicate_key_days"] == []
        assert len(leg["identical_columns"]) == 381  # 375 特征 + 6 标签全部容差内一致
        assert "ret_1" in leg["identical_columns"]
        assert "y_ret_20" in leg["identical_columns"]
        assert report["cs_train_verify_parity"]["pass"] is True
        assert (
            report["adapter_evidence"]["watermarks"]["label_sealed_through"]["y_ret_20"]
            == _DAYS[-1]
        )
        # 输入稳定硬门证据（R2-P3-T1-01）
        stability = report["input_stability"]
        assert stability["checked"] is True and stability["pass"] is True
        assert stability["watermark_stable"] is True
        assert stability["locked_file_count"] + stability["manifest_audit_count"] == len(
            report["input_identities"]
        )
        assert stability["manifest_audit_count"] >= 1  # manifest.json 纳入审计
        assert "运行前快照" in stability["manifest_audit_note"]  # reviewer1-N-4 口径
        coverage = report["coverage_consistency"]
        assert coverage["consistent"] is True
        assert coverage["excluded_days"] == []
        assert report["adapter_evidence"]["adapter_dates"] == _DAYS
        assert report["adapter_evidence"]["excluded_days"] == []
        assert report["window"]["frozen_through"] == "20260702"
        assert report["window"]["rationale"] == "合成全等用例"
        assert "atol=1e-6" in report["comparison_legend"]

    def test_d12_exempt_difference_approved(self, tmp_path):
        """D-12 豁免列差异 ⇒ 已批准差异项（F11 词表登记），verdict 仍 PASS。"""
        root = _build_dual_root(
            tmp_path,
            panel_overrides={"days_to_unlock": 100.0},
            cs_train_overrides={"days_to_unlock": 50.0},
        )
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is True
        approved = leg["approved_differences"]
        assert "days_to_unlock" in approved
        assert "D-12" in approved["days_to_unlock"]["attribution"]
        assert leg["rejected_differences"] == {}


class TestReconcileReject:
    def test_unattributed_difference_fails(self, tmp_path):
        """普通列注入差异（不在已批准清单）⇒ 拒绝项，verdict FAIL（§7.4-7 停止）。"""
        root = _build_dual_root(tmp_path, cs_train_overrides={"ret_1": 0.99})
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is False
        assert "ret_1" in leg["rejected_differences"]

    def test_key_set_mismatch_fails(self, tmp_path):
        """键集不一致（cs_train 多出一股）⇒ FAIL。"""
        root = _build_dual_root(tmp_path, extra_cs_train_code="999999.XX")
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is False
        assert leg["key_diff_days"]
        # 键集差异不影响 cs_train 验证源腿（同源自读仍逐位一致）
        assert report["cs_train_verify_parity"]["pass"] is True

    def test_missing_ref_day_fails(self, tmp_path):
        """missing_ref 日（cs_train 缺分区）⇒ 结构违例 FAIL。"""
        root = _build_dual_root(tmp_path, skip_cs_train_days=(_DAYS[1],))
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is False
        assert leg["missing_ref_days"] == [_DAYS[1]]

    def test_cols_only_late_day_fails(self, tmp_path):
        """cs_train 多出幽灵列（晚于 2012 schema 演进边界）⇒ 结构违例 FAIL。"""
        root = _build_dual_root(tmp_path, extra_cs_train_column="ghost_col")
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is False
        assert leg["cols_only_cs_train_days"]

    def test_duplicate_key_fails(self, tmp_path):
        """参照侧同日重复键 ⇒ 双侧重键 fail-fast（set 键域比较掩盖的错位防线）。"""
        root = _build_dual_root(tmp_path, duplicate_cs_train_key=True)
        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        leg = report["panel_vs_cs_train"]
        assert report["verdict"]["pass"] is False
        assert leg["duplicate_key_days"] == [
            {"date": _DAYS[0], "side": "cs_train"},
            {"date": _DAYS[1], "side": "cs_train"},
        ]


class TestInputStabilityGate:
    """R2-P3-T1-01：消费前后输入身份硬门（事后身份无权追认先前读取）。"""

    def test_between_legs_tamper_cs_train_fails(self, tmp_path):
        """评审复现反例：panel 腿后篡改 cs_train —— 两腿各自仍过，整体必须 FAIL。"""
        root = _build_dual_root(tmp_path)

        def tamper(target_root: Path) -> None:
            path = target_root / "features" / "cs_train" / f"{_DAYS[0]}.parquet"
            frame = pd.read_parquet(path)
            frame["ret_1"] = [999.0, 999.0]
            frame.to_parquet(path, index=False)

        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=tamper)
        assert report["panel_vs_cs_train"]["verdict_pass"] is True  # 比较的是旧值
        assert report["cs_train_verify_parity"]["pass"] is True  # 两次读的都是新值
        assert report["input_stability"]["pass"] is False
        assert report["input_stability"]["drift_files"]
        assert report["verdict"]["pass"] is False  # 漂移 ⇒ 整体拒绝

    def test_between_legs_tamper_manifest_registration_fails(self, tmp_path):
        """窗内 manifest 登记被改写（标签分区指纹）⇒ 相关性 diff 拒绝（N2 查询相关变更）。"""
        root = _build_dual_root(tmp_path)

        def tamper(target_root: Path) -> None:
            manifest = target_root / "features" / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["labels"]["y_ret_20"]["partitions"][f"{_DAYS[0]}.parquet"]["sha256_16"] = (
                "0" * 16
            )
            manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=tamper)
        assert report["input_stability"]["pass"] is False
        issues = report["input_stability"]["manifest_relevance_issues"]
        assert any("标签分区登记变更" in issue and "y_ret_20" in issue for issue in issues)
        assert report["verdict"]["pass"] is False

    def test_between_legs_corrupt_manifest_fails(self, tmp_path):
        """manifest 运行期损坏（非法 JSON）⇒ 复核解析失败即漂移（fail-closed）。"""
        root = _build_dual_root(tmp_path)

        def tamper(target_root: Path) -> None:
            manifest = target_root / "features" / "manifest.json"
            manifest.write_text(manifest.read_text(encoding="utf-8")[:50], encoding="utf-8")

        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=tamper)
        assert report["input_stability"]["pass"] is False
        assert report["verdict"]["pass"] is False

    def test_between_legs_watermark_shift_fails(self, tmp_path):
        """输入水位（clean/daily 更晚分区）运行间变化 ⇒ FAIL（影响成熟度判定的对象固定证据）。"""
        root = _build_dual_root(tmp_path)

        def tamper(target_root: Path) -> None:
            pd.DataFrame({"trade_date": ["20261231"], "close": [1.0]}).to_parquet(
                target_root / "clean" / "daily" / "2026-12-31.parquet", index=False
            )

        report = run_window_reconcile(root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=tamper)
        assert report["input_stability"]["watermark_stable"] is False
        assert report["verdict"]["pass"] is False

    def test_external_append_passes(self, tmp_path):
        """消费集合外追加（窗口外标签分区）⇒ 不放射性失败（既有契约保留）。"""
        root = _build_dual_root(tmp_path)

        def append_outside(target_root: Path) -> None:
            pd.DataFrame(
                {
                    "ts_code": _CODES,
                    "trade_date": ["20260108"] * len(_CODES),
                    "label_value": [0.1] * len(_CODES),
                    "neu_label_value": [0.2] * len(_CODES),
                    "maturity_status": ["forming"] * len(_CODES),
                }
            ).to_parquet(target_root / "labels" / "y_ret_20" / "20260108.parquet", index=False)

        report = run_window_reconcile(
            root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=append_outside
        )
        assert report["input_stability"]["pass"] is True
        assert report["verdict"]["pass"] is True

    def test_official_store_window_external_append_passes(self, tmp_path):
        """reviewer2-N2 正例：经正式 store 写 API 追加完整窗口外日 ⇒ 放行。

        追加 20260703 八族 + 三表 forming 标签（合法更新全局 manifest），
        查询相关输入与水位不变 ⇒ 不得误判漂移。
        """
        root = _build_dual_root_long_calendar(tmp_path)

        def append_via_store(target_root: Path) -> None:
            store = PanelDataStore(target_root)
            for group in PANEL_GROUPS:
                store.append_features(
                    TradeDate.from_str("20260703"),
                    group,
                    pd.DataFrame({"trade_date": ["20260703"] * 2, "ts_code": _CODES}),
                )
            for name in LABEL_TABLES:
                store.append_labels(
                    name,
                    pd.DataFrame(
                        {
                            "ts_code": _CODES,
                            "trade_date": ["20260703"] * 2,
                            "label_value": [0.1] * 2,
                            "neu_label_value": [0.2] * 2,
                            "maturity_status": ["forming"] * 2,
                        },
                        columns=_LABEL_COLUMNS,
                    ),
                )

        report = run_window_reconcile(
            root, _DAYS[0], _DAYS[-1], _post_panel_leg_hook=append_via_store
        )
        stability = report["input_stability"]
        assert stability["manifest_relevance_issues"] == []
        assert stability["wal_pending_before"] == stability["wal_pending_after"]
        assert stability["watermark_stable"] is True
        assert report["verdict"]["pass"] is True


class TestFirstReadGuard:
    """reviewer2-N1：首次日历读取必须受守卫覆盖（一致输入上下文 / 干净重读 / 拒绝）。"""

    @staticmethod
    def _flip_calendar(root: Path) -> None:
        calendar = root / "clean" / "trade_cal.parquet"
        frame = pd.read_parquet(calendar)
        mask = frame["cal_date"] == _DAYS[1]
        frame.loc[mask, "is_open"] = 1 - frame.loc[mask, "is_open"]
        frame.to_parquet(calendar, index=False)

    def test_calendar_shift_during_acquire_rejected(self, tmp_path, monkeypatch):
        """日历在每次采集尝试中都变化 ⇒ 连续不一致 ⇒ 拒绝（不保留旧清单再换基线）。"""
        root = _build_dual_root(tmp_path)
        from scripts.v2_p3 import t1_reconcile_adapter as tool

        original = tool.resolve_trading_dates

        def always_shift(path):
            days = original(path)
            self._flip_calendar(Path(path))
            return days

        monkeypatch.setattr(tool, "resolve_trading_dates", always_shift)
        with pytest.raises(RuntimeError, match="连续两次采集不一致"):
            tool.run_window_reconcile(root, _DAYS[0], _DAYS[-1])

    def test_calendar_shift_once_restarts_clean(self, tmp_path, monkeypatch):
        """首次采集期间变化一次 ⇒ 丢弃干净重读：选日/readiness/两腿全部基于新日历一致通过。"""
        root = _build_dual_root(tmp_path)
        from scripts.v2_p3 import t1_reconcile_adapter as tool

        original = tool.resolve_trading_dates
        state = {"tampered": False}

        def shift_once(path):
            days = original(path)
            if not state["tampered"]:
                state["tampered"] = True
                calendar = Path(path) / "clean" / "trade_cal.parquet"
                frame = pd.read_parquet(calendar)
                frame.loc[frame["cal_date"] == _DAYS[1], "is_open"] = 0
                frame.to_parquet(calendar, index=False)
            return days

        monkeypatch.setattr(tool, "resolve_trading_dates", shift_once)
        report = tool.run_window_reconcile(root, _DAYS[0], _DAYS[-1])
        # 重启后全部基于新日历（仅首日开市）：不得出现「2 日比较 / 1 日 readiness」混合
        assert report["window"]["trade_days"] == 1
        coverage = report["coverage_consistency"]
        assert coverage["consistent"] is True
        assert coverage["expected_days"] == coverage["readiness_days"] == 1
        assert report["cs_train_verify_parity"]["trade_days"] == 1
        # reviewer2-N1 余项：实际适配帧日期 ⊆ 已校验日域（区间读取带入的 20260106 被剔除登记）
        assert report["adapter_evidence"]["adapter_dates"] == [_DAYS[0]]
        assert report["adapter_evidence"]["excluded_days"] == [_DAYS[1]]
        assert report["verdict"]["pass"] is True

    def test_coverage_same_count_different_dates_fails(self, tmp_path):
        """同数不同日 ⇒ 覆盖门拒绝（日期集合核对，计数一致不等于日域一致）。"""
        from scripts.v2_p3.t1_reconcile_adapter import _coverage_consistency

        frame = SimpleNamespace(
            day_readiness=({"trade_date": _DAYS[1]},),
            df=pd.DataFrame({"trade_date": [_DAYS[1]]}),
            excluded_days=(),
        )
        coverage = _coverage_consistency(
            [_DAYS[0]],
            frame,
            {"missing_ref_days": [], "duplicate_key_days": [], "trade_days_compared": 1},
            {"dates": [_DAYS[1]]},
        )
        assert coverage["consistent"] is False
        assert coverage["date_mismatches"]["readiness_minus_expected"] == [_DAYS[1]]

    def test_empty_window_rejected(self, tmp_path):
        """窗口内无交易日 ⇒ 明确 ValueError（reviewer1-N-1：先于枚举的 IndexError）。"""
        root = _build_dual_root(tmp_path)
        with pytest.raises(ValueError, match="无交易日"):
            run_window_reconcile(root, "20260110", "20260111")  # 周末两日


def _build_dual_root_long_calendar(tmp_path: Path) -> Path:
    """N2 正例专用：日历与 clean/daily 覆盖 20260703 之后的合成数据根。"""
    calendar = pd.date_range("2026-01-05", periods=200, freq="B").strftime("%Y%m%d").tolist()
    clean = tmp_path / "clean"
    (clean / "daily").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": calendar, "is_open": [1] * len(calendar)}).to_parquet(
        clean / "trade_cal.parquet", index=False
    )
    for day in calendar:
        pd.DataFrame({"trade_date": [day], "close": [1.0]}).to_parquet(
            clean / "daily" / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )
    store = PanelDataStore(tmp_path)
    store.manifest.register_columns(
        {col: {"group": group_of_column(col), "source": "test"} for col in PANEL_FEATURE_COLUMNS}
    )
    store.manifest.save()
    for day in _DAYS:
        _append_panel_day(store, day, {})
        _append_labels(store, day)
        _write_cs_train_day(tmp_path, day, {}, None)
    return tmp_path


class TestConsumptionEnumeration:
    def test_enumerate_window_input_files(self, tmp_path):
        root = _build_dual_root(tmp_path)
        files = enumerate_window_input_files(root, _DAYS, _DAYS[0], _DAYS[-1])
        names = {path.relative_to(root).as_posix() for path in files}
        assert "clean/trade_cal.parquet" in names
        assert "features/manifest.json" in names  # R2-P3-T1-01 清单缺口
        assert {f"features/cs_train/{day}.parquet" for day in _DAYS} <= names
        assert {f"labels/y_ret_20/{day}.parquet" for day in _DAYS} <= names
        assert {f"features/panel/{day}/core.parquet" for day in _DAYS} <= names
        daily = [name for name in names if name.startswith("clean/daily/")]
        assert len(daily) == 1  # 仅最新分区（W_in 物理来源）

    def test_mixed_hot_cold_per_group(self, tmp_path):
        """族级冷热混合：core 当月冷区、moneyflow 热区 ⇒ 清单按族同源解析（R2-P3-T1-01）。"""
        root = _build_dual_root(tmp_path)
        month = f"{_DAYS[0][:4]}-{_DAYS[0][4:6]}"
        archive_dir = tmp_path / "features" / "panel_archive" / month
        archive_dir.mkdir(parents=True)
        # 仅 core 落冷区月分区（其余族仍热区）
        pd.read_parquet(tmp_path / "features" / "panel" / _DAYS[0] / "core.parquet").to_parquet(
            archive_dir / "core.parquet", index=False
        )
        files = _panel_partition_files(root, _DAYS[0], _DAYS[-1])
        names = {path.relative_to(root).as_posix() for path in files}
        assert f"features/panel_archive/{month}/core.parquet" in names
        assert f"features/panel/{_DAYS[0]}/core.parquet" not in names  # 冷区为准跳过该族热区
        assert f"features/panel/{_DAYS[0]}/moneyflow.parquet" in names  # 其余族热区保留


class TestPanelReconcileFacadeLock:
    """对账工具复用 panel_reconcile 私有实现（有意复用，reviewer1-P3-5）：签名锁定防静默重构。"""

    def test_private_facade_signatures(self):
        from src.lazybull.v2.store import panel_reconcile

        assert list(inspect.signature(panel_reconcile._compare_day).parameters) == [
            "date",
            "ref",
            "panel_day",
        ]
        assert list(inspect.signature(panel_reconcile._merge_and_judge).parameters) == [
            "day_results",
            "mode",
        ]
        assert list(inspect.signature(panel_reconcile._build_verdict).parameters) == [
            "merged",
            "diff_cols",
            "mode",
        ]
        assert len(panel_reconcile.PANEL_FEATURE_COLUMNS) == 375
        # reviewer1-N-3：工具对 store 私有静态方法的依赖同样纳入签名锁定
        from src.lazybull.v2.store.data_store import PanelDataStore

        assert list(inspect.signature(PanelDataStore._months_in_range).parameters) == [
            "start_str",
            "end_str",
        ]

# -*- coding: utf-8 -*-
"""v2 P3 T5 阶段 A R1 评审处置回归测试（全合成数据，不读取真实生产数据/产物）。

处置清单映射（全部接纳；合成夹具复用 test_v2_p3_t5_data_impact_attribution）：
- R2-T5A-06（P1 预测行逐值归因）：b 节读两臂 ``folds/splitNN/predictions.parquet``——
  键集对称差 + 交集键 ``%.17g`` 逐值 diff；同日换股与同键改分可区分；产物/指纹增键
  缺失或漂移 ⇒ FAIL（必需证据，不得跳过）；
- R2-T5A-04（P1 零比较日/缺证据可闭合）：a 节完整覆盖前置（trade_cal ``is_open=1`` ∩
  boundary 窗口；``days_compared == 预期日数`` 且非零；缺日即 missing + FAIL；非开市日
  excluded 登记）；消费快照当前态复核（缺失/漂移 ⇒ FAIL）；batch_meta 空/缺
  data_state_id、fingerprint ``{}`` 或缺字段、daily/holdings 文件缺失 ⇒ FAIL（与
  合法零行分开登记）；
- R2-T5A-05（P1 输入闭合误当输出归因闭合）：归因闭合 = a 节闭合 ∧ 输出级一致性
  （规则① 零输入差异 ⇒ 零输出差异；规则② 逐折对应）；反例：输入零差异 + 输出
  劣化 ⇒ FAIL；输入有差异 + 输出差异落在有差异折 ⇒ 可闭合；
- P2-7（归因成本）：跨折单日结果缓存键 = ``(day, 消费列集指纹)``；缓存命中与不命中
  结果一致；命中率登记 CLI/报告；
- P3-2（g 节身份补全）：B0' 从 ``freeze_manifest.json`` 读代码态/数据态；两臂环境身份
  从 ``arm_meta.json`` 的 ``producer_identity.environment_identity`` 读；缺失 ⇒ FAIL 前置；
- P3-3（c 节重复键笛卡尔展开）：同键多笔先聚合（shares/amount 求和、price 推导），
  重复键计数登记；
- P3-8（fold dtype）：``compare_chain_navs`` 的 ``split_index`` 显式 ``astype(int)``。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3 import t5_attribution_sections as sections  # noqa: E402
from scripts.v2_p3.run_data_impact_attribution import build_report, main  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from tests.test_v2_p3_t5_data_impact_attribution import (  # noqa: E402
    _CONSUMED_COLUMNS,
    _DAYS,
    _FOLDS,
    _OBSERVER,
    _build_arm,
    _build_b0p,
    _build_data_root,
    _prediction_frame_fingerprint,
    _register_columns,
    _sha256_file,
    _write_consumption_binding,
    _write_trade_calendar,
)


def _synth_env(tmp_path: Path, **arm3_kwargs: Any) -> dict[str, Path]:
    """默认闭合环境（同基线测试夹具；arm3 差异经 arm3_kwargs 注入）。"""
    data_root = _build_data_root(tmp_path)
    return {
        "data_root": data_root,
        "arm2": _build_arm(tmp_path, "arm2", data_root=data_root),
        "arm3": _build_arm(tmp_path, "arm3", data_root=data_root, **arm3_kwargs),
        "b0p": _build_b0p(tmp_path),
    }


def _run(env: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return build_report(
        env["arm2"],
        env["arm3"],
        env["b0p"],
        env["data_root"],
        run_id="r1_fix_test",
        observer_identity=_OBSERVER,
        **kwargs,
    )


def _rewrite_predictions(
    arm: Path, fold: str, rows: list[dict[str, Any]], *, sync_fingerprint: bool = True
) -> None:
    """重写一折预测行并同步全部派生登记（sha256/行数/n_rows/score_days/行集真指纹）。

    ``sync_fingerprint=False`` 时保留旧 ``fingerprint`` 字段（构造①行集自洽反例）。
    """
    fold_dir = arm / "folds" / fold
    if rows:
        frame = pd.DataFrame(rows).astype({"trade_date": str, "ts_code": str, "score": float})
        frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=["trade_date", "ts_code", "score"]).astype(
            {"trade_date": str, "ts_code": str, "score": float}
        )
    frame.to_parquet(fold_dir / "predictions.parquet", index=False)
    fp_path = fold_dir / "fingerprint.json"
    doc = json.loads(fp_path.read_text(encoding="utf-8"))
    doc["predictions_sha256"] = _sha256_file(fold_dir / "predictions.parquet")
    doc["predictions_rows"] = len(frame)
    doc["n_rows"] = len(frame)
    doc["score_days"] = sorted({str(d) for d in frame["trade_date"]})
    if sync_fingerprint:
        doc["fingerprint"] = _prediction_frame_fingerprint(frame)
    fp_path.write_text(json.dumps(doc), encoding="utf-8")


def _recollect_snapshots(env: dict[str, Path]) -> None:
    """数据态变更后两臂重采快照并回填 arm_meta 摘要绑定（合法数据态演进路径）。"""
    for key in ("arm2", "arm3"):
        digest = _write_consumption_binding(env[key], env["data_root"])
        meta_path = env[key] / "arm_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["producer_identity"]["input_snapshot_digest"] = digest
        meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _tamper_cs_train_day(env: dict[str, Path], day: str) -> None:
    """篡改一个 cs_train 日分区内容（改消费列值 ⇒ 内容哈希变化、键集不变）。"""
    path = env["data_root"] / "features" / "cs_train" / f"{day}.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], "ret_1"] = float(frame.loc[frame.index[0], "ret_1"]) + 1.0
    frame.to_parquet(path, index=False)


def _error_text(report: dict[str, Any]) -> str:
    return "\n".join(report["verdict"]["errors"])


# ---------------------------------------------------------------------------
# R2-T5A-06：b 节预测行逐值归因
# ---------------------------------------------------------------------------


class TestPredictionRowAttribution:
    def test_key_sym_diff_distinguishes_same_day_swap(self, tmp_path):
        """同日换股：臂③同日以 300750.SZ 替换 600000.SH ⇒ 键集对称差 + 换股候选日登记。"""
        env = _synth_env(tmp_path)
        _rewrite_predictions(
            env["arm3"],
            "split00",
            [
                {"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.11},
                {"trade_date": "20260106", "ts_code": "300750.SZ", "score": 0.22},
            ],
        )
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        pred = _fold_b(report, "split00")["predictions"]
        assert pred["key_sym_diff_count"] == 2
        assert pred["keys_only_arm2"] == 1 and pred["keys_only_arm3"] == 1
        assert pred["sample_only_arm2"] == [["20260106", "600000.SH"]]
        assert pred["sample_only_arm3"] == [["20260106", "300750.SZ"]]
        assert pred["same_day_swap_dates_count"] == 1
        assert pred["same_day_swap_dates"] == ["20260106"]
        assert pred["value_diff_rows"] == 0  # 交集键无改分
        # 键差异折有 a 节已登记消费列差异（默认 D-12/D-04③）⇒ 规则② 逐折对应满足
        detail = report["sections"]["f_gdata_judgment_table"]["attribution_closure_detail"]
        assert detail["consistent"] is True
        assert "split00" in detail["degraded_output_folds"]

    def test_same_key_value_diff_17g(self, tmp_path):
        """同键改分：交集键 score 第 16 位有效数字级差异 ⇒ %.17g 逐值 diff 检出。"""
        env = _synth_env(tmp_path)
        _rewrite_predictions(
            env["arm3"],
            "split01",
            [
                {"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.11 + 1e-16},
                {"trade_date": "20260106", "ts_code": "600000.SH", "score": 0.22},
            ],
        )
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        pred = _fold_b(report, "split01")["predictions"]
        assert pred["key_sym_diff_count"] == 0
        assert pred["intersection_keys"] == 2
        assert pred["value_diff_rows"] == 1
        assert 0.0 < pred["value_max_abs_diff"] < 1e-15
        sample = pred["value_diff_samples"][0]
        assert sample["ts_code"] == "000001.SZ"
        assert sample["score_arm2"] == "0.11"
        assert sample["score_arm3"] == format(0.11 + 1e-16, ".17g")
        assert pred["value_caliber"].startswith("交集键 score 按 %.17g")

    def test_missing_predictions_parquet_fails(self, tmp_path):
        """任一折缺 predictions.parquet ⇒ 必需证据缺失 FAIL（不得跳过）。"""
        env = _synth_env(tmp_path)
        (env["arm3"] / "folds" / "split01" / "predictions.parquet").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "预测行产物缺失" in _error_text(report)

    def test_missing_predictions_fingerprint_keys_fails(self, tmp_path):
        """fingerprint.json 缺 predictions_* 增键 ⇒ 该折产物不完整 ⇒ FAIL。"""
        env = _synth_env(tmp_path)
        fp_path = env["arm2"] / "folds" / "split00" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        for key in ("predictions_path", "predictions_sha256", "predictions_rows"):
            del doc[key]
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "predictions_*" in _error_text(report)

    def test_predictions_sha_drift_fails(self, tmp_path):
        """预测行落盘后漂移（sha256 与指纹登记不符）⇒ FAIL。"""
        env = _synth_env(tmp_path)
        fold_dir = env["arm3"] / "folds" / "split00"
        frame = pd.DataFrame(
            [
                {"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.99},
                {"trade_date": "20260106", "ts_code": "600000.SH", "score": 0.22},
            ]
        )
        frame.to_parquet(fold_dir / "predictions.parquet", index=False)  # 不同步指纹登记
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "sha256 不一致" in _error_text(report)

    def test_negative_zero_normalized(self):
        """-0.0 归一 0.0（跨路契约口径）：-0.0 vs 0.0 不计逐值差异。"""
        left = pd.DataFrame([{"trade_date": "20260106", "ts_code": "000001.SZ", "score": -0.0}])
        right = pd.DataFrame([{"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.0}])
        result = sections.compare_prediction_frames(left, right)
        assert result["value_diff_rows"] == 0
        assert result["key_sym_diff_count"] == 0

    def test_nan_score_rejected(self):
        """NaN/Inf 防御性拒绝（生产侧已拒，消费侧复核证据完整性）。"""
        good = pd.DataFrame([{"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.1}])
        bad = pd.DataFrame(
            [{"trade_date": "20260106", "ts_code": "000001.SZ", "score": float("nan")}]
        )
        with pytest.raises(ValueError, match="NaN/Inf"):
            sections.compare_prediction_frames(good, bad)

    def test_fingerprint_empty_object_fails(self, tmp_path):
        """R2-T5A-04：逐折 fingerprint.json 为 {} ⇒ FAIL（修掉 None==None 误判一致）。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            for fold in _FOLDS:
                (arm / "folds" / fold / "fingerprint.json").write_text("{}", encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "指纹文件为空对象" in _error_text(report)
        assert report["sections"]["b_prediction_impact"]["folds"] == []

    def test_fingerprint_missing_fingerprint_field_fails(self, tmp_path):
        """fingerprint.json 缺 fingerprint 字段 ⇒ FAIL（field 级缺失，非空对象）。"""
        env = _synth_env(tmp_path)
        fp_path = env["arm2"] / "folds" / "split00" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        del doc["fingerprint"]
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "缺 fingerprint 字段" in _error_text(report)


def _fold_b(report: dict[str, Any], fold: str) -> dict[str, Any]:
    return next(
        entry
        for entry in report["sections"]["b_prediction_impact"]["folds"]
        if entry["fold"] == fold
    )


# ---------------------------------------------------------------------------
# R2-T5A-04：完整覆盖前置 + 消费态核验 + 必需输出/身份缺失即 FAIL
# ---------------------------------------------------------------------------


class TestCoverageGate:
    def test_zero_compared_days_never_closed(self, tmp_path):
        """评审复现：数据根仅登记消费列、无任何分区 ⇒ days_compared=0，a 节不得闭合。"""
        data_root = tmp_path / "data_root"
        store = PanelDataStore(data_root)
        _register_columns(store, list(_CONSUMED_COLUMNS))
        _write_trade_calendar(data_root, tuple(_DAYS), ())
        env = {
            "data_root": data_root,
            "arm2": _build_arm(tmp_path, "arm2", data_root=data_root),
            "arm3": _build_arm(tmp_path, "arm3", data_root=data_root),
            "b0p": _build_b0p(tmp_path),
        }
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        section_a = report["sections"]["a_column_input_diffs"]
        assert section_a["closed"] is False
        fold0 = section_a["folds"][0]
        assert fold0["days_compared"] == 0
        assert fold0["expected_days_count"] == 2
        assert fold0["missing_cs_train_days"] == list(_DAYS)
        assert fold0["missing_panel_days"] == list(_DAYS)  # 双源同缺双侧入册
        assert fold0["coverage_gap_count"] == 2

    def test_missing_cs_train_partition_fails(self, tmp_path):
        """预期日域内任一源缺一日分区 ⇒ missing 登记 + FAIL（覆盖门）。"""
        env = _synth_env(tmp_path)
        (env["data_root"] / "features" / "cs_train" / f"{_DAYS[1]}.parquet").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        fold0 = report["sections"]["a_column_input_diffs"]["folds"][0]
        assert fold0["missing_cs_train_days"] == [_DAYS[1]]
        assert fold0["days_compared"] == 1
        assert fold0["expected_days_count"] == 2
        assert fold0["closed"] is False

    def test_non_open_day_excluded_registered(self, tmp_path):
        """合法剔除：窗口内非开市日（双源同缺）显式登记 excluded 计数，不破坏闭合。"""
        data_root = _build_data_root(tmp_path, open_days=(_DAYS[0],), closed_days=(_DAYS[1],))
        env = {
            "data_root": data_root,
            "arm2": _build_arm(tmp_path, "arm2", data_root=data_root),
            "arm3": _build_arm(tmp_path, "arm3", data_root=data_root),
            "b0p": _build_b0p(tmp_path),
        }
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        fold0 = report["sections"]["a_column_input_diffs"]["folds"][0]
        assert fold0["expected_days_count"] == 1
        assert fold0["days_compared"] == 1
        assert fold0["excluded_non_open_days"] == 1
        # 非开市日分区仍存在（夹具两日均写分区）⇒ 预期域外分区信息性登记，不计差异
        assert fold0["unexpected_partition_days"] == [_DAYS[1]]
        assert fold0["closed"] is True

    def test_missing_calendar_fails(self, tmp_path):
        """交易日历不可得 ⇒ 完整覆盖前置不可核验 ⇒ FAIL。"""
        env = _synth_env(tmp_path)
        (env["data_root"] / "clean" / "trade_cal.parquet").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "交易日历加载失败" in _error_text(report)

    def test_consumption_snapshot_missing_fails(self, tmp_path):
        """消费快照文件本身缺失 ⇒ FAIL 前置（评审复现：此前仍 PASS 属缺陷）。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            (arm / "consumption_snapshot.json").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "缺消费快照" in _error_text(report)
        section_g = report["sections"]["g_identity_appendix"]
        assert section_g["arm2"]["consumption_snapshot_verified"] is False

    def test_consumption_snapshot_drift_fails(self, tmp_path):
        """消费快照当前态复核：快照采集后消费文件内容漂移 ⇒ FAIL 并登记漂移清单。"""
        env = _synth_env(tmp_path)
        _tamper_cs_train_day(env, _DAYS[0])  # 快照采集后内容漂移
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "消费快照当前态复核漂移" in _error_text(report)
        section_g = report["sections"]["g_identity_appendix"]
        assert section_g["arm2"]["consumption_snapshot_verified"] is False
        assert section_g["arm2"]["consumption_drift_count"] == 1

    @pytest.mark.parametrize("content", ["{}", json.dumps({"batch_id": "batch_arm2"})])
    def test_batch_meta_missing_identity_fails(self, tmp_path, content):
        """batch_meta.json 为 {} 或缺 data_state_id ⇒ FAIL（评审复现：此前仍 PASS）。"""
        env = _synth_env(tmp_path)
        batch = env["arm2"] / "runs" / "batch_arm2"
        (batch / "batch_meta.json").write_text(content, encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        text = _error_text(report)
        assert "batch_meta.json 为空对象" in text or "缺 data_state.data_state_id" in text

    def test_daily_holdings_missing_fail(self, tmp_path):
        """daily/holdings 文件缺失（双臂同缺同罪）⇒ FAIL，与合法零行分开登记。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            for fold in _FOLDS:
                fold_dir = arm / "runs" / f"batch_{arm.name}" / "folds" / fold
                (fold_dir / "daily.parquet").unlink()
                (fold_dir / "holdings_snapshot.parquet").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        text = _error_text(report)
        assert "daily 文件缺失" in text and "holdings_snapshot 文件缺失" in text
        assert report["sections"]["d_holdings_impact"]["folds"] == []
        assert report["sections"]["e_nav_impact"]["daily_folds"] == []

    def test_empty_daily_holdings_legal_pass(self, tmp_path):
        """合法空：文件在但 0 行 ⇒ 登记 empty 标记且不判缺失，总判定 PASS。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            for fold in _FOLDS:
                fold_dir = arm / "runs" / f"batch_{arm.name}" / "folds" / fold
                pd.DataFrame(columns=["trade_date", "nav", "total_value"]).to_parquet(
                    fold_dir / "daily.parquet", index=False
                )
                pd.DataFrame(
                    columns=["trade_date", "ts_code", "shares", "market_value", "weight"]
                ).to_parquet(fold_dir / "holdings_snapshot.parquet", index=False)
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        daily0 = report["sections"]["e_nav_impact"]["daily_folds"][0]
        assert daily0["days_arm2"] == 0 and daily0["empty_arm2"] is True
        holdings0 = report["sections"]["d_holdings_impact"]["folds"][0]
        assert holdings0["rows_arm2"] == 0 and holdings0["empty_arm3"] is True


# ---------------------------------------------------------------------------
# R2-T5A-05：归因闭合 = a 节闭合 ∧ 输出级一致性
# ---------------------------------------------------------------------------


def _zero_diff_data_root(tmp_path: Path) -> Path:
    """两源消费列完全一致的合成数据根（a 节全列零差异）。"""
    return _build_data_root(
        tmp_path,
        panel_overrides={"days_to_unlock": [100.0, 100.0]},
        cs_train_overrides={"dividend_payout_ratio": [0.3, 0.3]},
    )


class TestOutputLevelClosure:
    def test_zero_input_diff_with_output_degradation_fails(self, tmp_path):
        """评审复现：输入零差异 + 输出显著劣化 ⇒ 规则① 不闭合 FAIL（反例）。"""
        data_root = _zero_diff_data_root(tmp_path)
        env = {
            "data_root": data_root,
            "arm2": _build_arm(tmp_path, "arm2", data_root=data_root),
            "arm3": _build_arm(
                tmp_path,
                "arm3",
                data_root=data_root,
                nav_last_delta=-0.5,  # nav 深跌 ⇒ 派生 cagr 大幅为负 + maxdd 深负
                daily_nav_delta=0.01,
            ),
            "b0p": _build_b0p(tmp_path),
        }
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        section_a = report["sections"]["a_column_input_diffs"]
        assert section_a["closed"] is True  # a 节自身闭合（零差异且覆盖完整）
        assert section_a["diff_column_count"] == 0
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["attribution_closed"] is False
        assert section_f["inputs"]["maxdd_attribution_closed"] is False
        assert section_f["cagr_judgment"]["branch"] == "overshoot_unattributed"
        assert section_f["maxdd_judgment"]["branch"] == "deterioration_unattributed"
        detail = section_f["attribution_closure_detail"]
        assert detail["zero_input"] is True
        assert detail["zero_input_zero_output_violations"]
        assert "输出级一致性" in _error_text(report)

    def test_input_diff_with_matching_output_folds_closes(self, tmp_path):
        """正例：输入有差异（词表在册）+ 输出差异落在有差异折 ⇒ 可闭合 PASS。"""
        env = _synth_env(tmp_path, nav_last_delta=0.05, daily_nav_delta=0.001)
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["attribution_closed"] is True
        detail = section_f["attribution_closure_detail"]
        assert detail["consistent"] is True
        assert detail["zero_input"] is False
        assert detail["degraded_output_folds"] == ["split00", "split01"]
        assert detail["unattributed_output_violations"] == []

    def test_maxdd_deterioration_fold_correspondence(self, tmp_path):
        """ΔMaxDD 劣化逐折归因闭合（§13.3.6 按规则②）：输入在册差异覆盖劣化折 ⇒ 放行。"""
        env = _synth_env(tmp_path, daily_nav_delta=0.002, nav_last_delta=-0.05)
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        section_f = report["sections"]["f_gdata_judgment_table"]
        assert section_f["maxdd_judgment"]["branch"] == "deterioration_attributed"
        assert section_f["inputs"]["maxdd_attribution_closed"] is True

    def test_registered_structure_is_input_diff_and_matches_output(self):
        """已登记结构差异（CYQ5 单侧列）既是输入差异、也可作为逐折对应证据。"""
        result = sections.evaluate_output_consistency(
            a_folds=[
                {
                    "fold": "split00",
                    "registered_cols_only_panel": {"winner_rate": {"days": 1}},
                }
            ],
            b_folds=[{"fold": "split00", "predictions": {"key_sym_diff_count": 1}}],
            c_folds=[],
            d_folds=[],
            e_chain=None,
            e_daily_folds=[],
        )
        assert result["zero_input"] is False
        assert result["consistent"] is True
        assert result["unattributed_output_violations"] == []

    def test_registered_structure_only_without_output_is_consistent(self):
        """仅已登记结构变化、输出无差异 ⇒ 不得误报规则①。"""
        result = sections.evaluate_output_consistency(
            a_folds=[
                {
                    "fold": "split00",
                    "registered_cols_only_panel": {"winner_rate": {"days": 1}},
                }
            ],
            b_folds=[],
            c_folds=[],
            d_folds=[],
            e_chain=None,
            e_daily_folds=[],
        )
        assert result["zero_input"] is False
        assert result["consistent"] is True

    def test_output_diff_in_fold_without_input_diff_fails(self, tmp_path):
        """规则② 反例：split01 消费列仅 ret_1（零输入差异）但其 daily 输出劣化 ⇒ FAIL。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            models = arm / "root" / "models" / "stock_selection"
            (models / "v2_features.json").write_text(json.dumps(["ret_1"]), encoding="utf-8")
        daily_path = env["arm3"] / "runs" / "batch_arm3" / "folds" / "split01" / "daily.parquet"
        daily = pd.read_parquet(daily_path)
        daily["nav"] = daily["nav"] + 0.01
        daily.to_parquet(daily_path, index=False)
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        detail = report["sections"]["f_gdata_judgment_table"]["attribution_closure_detail"]
        assert detail["consistent"] is False
        assert any("split01" in text for text in detail["unattributed_output_violations"])
        # split00 输入差异在册且无输出劣化 ⇒ 不违规；违规仅来自 split01
        assert detail["zero_input_zero_output_violations"] == []


# ---------------------------------------------------------------------------
# P2-7：跨折单日结果缓存（命中与不命中结果一致 + 命中率登记）
# ---------------------------------------------------------------------------


def _section_a_without_cache_stats(report: dict[str, Any]) -> str:
    section = dict(report["sections"]["a_column_input_diffs"])
    section.pop("cache_stats", None)
    return json.dumps(section, ensure_ascii=False, sort_keys=True, default=str)


class TestAttributionCache:
    def test_cache_hit_miss_identical_results(self, tmp_path):
        """缓存键 = (day, 消费列集指纹)：唯一日只比较一次（跨折复用），命中与否结果一致。"""
        env = _synth_env(tmp_path)
        cache_dir = tmp_path / "attribution_cache"
        cold = _run(env, cache_dir=cache_dir)
        cold_stats = _day_stats(cold)
        assert cold_stats["misses"] == 2  # 唯一日 2 个（消费列集指纹相同）
        assert cold_stats["hits"] == 2  # 第二折跨折命中
        assert cold["verdict"]["verdict"] == "PASS", _error_text(cold)
        version = sections.DayCompareCache.FORMAT_VERSION
        ns = cold_stats["namespace"]
        cached_files = sorted((cache_dir / version / ns).glob("*.json"))
        assert len(cached_files) == 2  # 每唯一日一文件（命名空间隔离目录）

        warm = _run(env, cache_dir=cache_dir)
        warm_stats = _day_stats(warm)
        assert warm_stats["hits"] == 4 and warm_stats["misses"] == 0  # 跨运行磁盘命中
        assert warm_stats["hit_rate"] == 1.0
        assert warm_stats["namespace"] == ns  # 同数据态 ⇒ 同命名空间
        assert _section_a_without_cache_stats(warm) == _section_a_without_cache_stats(cold)

    def test_panel_month_lru_registered(self, tmp_path):
        """P3-N4：panel 月帧 (month, 列集指纹) LRU 跨折复用，命中率登记且结果不变。"""
        env = _synth_env(tmp_path)
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        month_stats = report["sections"]["a_column_input_diffs"]["cache_stats"]["panel_month"]
        assert month_stats["misses"] == 1  # 唯一月 2026-01（两折同列集）
        assert month_stats["hits"] == 1  # 第二折命中
        assert month_stats["hit_rate"] == pytest.approx(0.5)

    def test_panel_month_lru_eviction(self):
        """LRU 容量语义：超出容量逐出最久未用条目（容量 2 放入 3 条）。"""
        cache = sections.PanelMonthCache(capacity=2)
        frame = pd.DataFrame()
        cache.put("2026-01", "fp", frame, "id1")
        cache.put("2026-02", "fp", frame, "id2")
        assert cache.get("2026-01", "fp") is not None  # 刷新 2026-01 为最近使用
        cache.put("2026-03", "fp", frame, "id3")  # 逐出 2026-02
        assert cache.get("2026-02", "fp") is None
        assert cache.get("2026-01", "fp") is not None

    def test_cache_stats_registered_in_cli_report(self, tmp_path, monkeypatch):
        """CLI/报告登记缓存命中率：缓存目录落 out-json 同级 attribution_cache/。"""
        monkeypatch.setattr(
            "src.lazybull.v2.store.store_identity.workspace_identity", lambda root: _OBSERVER
        )
        env = _synth_env(tmp_path)
        out_json = tmp_path / "out" / "report.json"
        exit_code = main(
            [
                "--arm2-dir",
                str(env["arm2"]),
                "--arm3-dir",
                str(env["arm3"]),
                "--b0p-dir",
                str(env["b0p"]),
                "--production-data-root",
                str(env["data_root"]),
                "--run-id",
                "cache_cli_test",
                "--out-json",
                str(out_json),
                "--out-md",
                str(tmp_path / "out" / "report.md"),
            ]
        )
        assert exit_code == 0
        doc = json.loads(out_json.read_text(encoding="utf-8"))
        stats = doc["sections"]["a_column_input_diffs"]["cache_stats"]
        day_stats = stats["day_compare"]
        assert day_stats["cache_dir"] == str(out_json.parent / "attribution_cache")
        assert day_stats["hits"] + day_stats["misses"] == 4
        assert stats["panel_month"]["hits"] + stats["panel_month"]["misses"] == 2
        ns_dir = (
            out_json.parent
            / "attribution_cache"
            / sections.DayCompareCache.FORMAT_VERSION
            / day_stats["namespace"]
        )
        assert ns_dir.is_dir()


def _day_stats(report: dict[str, Any]) -> dict[str, Any]:
    return report["sections"]["a_column_input_diffs"]["cache_stats"]["day_compare"]


# ---------------------------------------------------------------------------
# P3-2：g 节身份补全（freeze_manifest / producer_identity.environment_identity）
# ---------------------------------------------------------------------------


class TestIdentityCompletion:
    def test_b0p_missing_freeze_manifest_fails(self, tmp_path):
        """B0' 缺 freeze_manifest.json ⇒ FAIL 前置（g 节 B0' 身份唯一来源）。"""
        env = _synth_env(tmp_path)
        (env["b0p"] / "freeze_manifest.json").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "缺 freeze_manifest.json" in _error_text(report)

    def test_b0p_manifest_missing_digest_fails(self, tmp_path):
        """freeze_manifest.json 缺 producer_identity.source_digest ⇒ FAIL 前置。"""
        env = _synth_env(tmp_path)
        manifest_path = env["b0p"] / "freeze_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["producer_identity"]["source_digest"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "producer_identity.source_digest" in _error_text(report)

    def test_arm_missing_environment_identity_fails(self, tmp_path):
        """臂 arm_meta 缺 producer_identity.environment_identity（且无兜底）⇒ FAIL 前置。"""
        env = _synth_env(tmp_path)
        meta_path = env["arm3"] / "arm_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["producer_identity"]["environment_identity"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "身份字段缺失" in _error_text(report)


# ---------------------------------------------------------------------------
# P3-3：c 节重复键聚合（同日同股同方向多笔成交）
# ---------------------------------------------------------------------------


def _trade_rows(*rows: tuple[str, float, float]) -> pd.DataFrame:
    """(action, price, shares) 简表 → trades 帧（同日同股）。"""
    return pd.DataFrame(
        [
            {
                "trade_date": "20260106",
                "ts_code": "000001.SZ",
                "action": action,
                "price": price,
                "shares": shares,
                "amount": price * shares,
            }
            for action, price, shares in rows
        ]
    )


class TestKeyedAggregation:
    def test_duplicate_keys_aggregated_before_compare(self):
        """同键多笔先聚合：两臂经济口径一致（Σshares/Σamount/推导价）⇒ 零差异零失真。"""
        frame2 = _trade_rows(("buy", 10.0, 100.0), ("buy", 11.0, 100.0))
        frame3 = _trade_rows(("buy", 10.5, 200.0))
        result = sections.compare_trade_frames(frame2, frame3)
        assert result["rows_arm2"] == 2 and result["rows_arm3"] == 1
        assert result["duplicate_key_groups_arm2"] == 1
        assert result["duplicate_key_groups_arm3"] == 0
        assert result["sample_duplicate_keys_arm2"] == [["20260106", "000001.SZ", "buy"]]
        assert result["key_sym_diff_count"] == 0
        assert result["intersection_rows"] == 1  # 无多×多笛卡尔展开
        for col in ("price", "shares", "amount"):
            assert result["value_max_abs_diff"][col]["max_abs_diff"] == 0.0

    def test_duplicate_keys_aggregated_mismatch_detected(self):
        """聚合后真实差异仍可检出：price=Σamount/Σshares 推导口径下 Δprice=0.1。"""
        frame2 = _trade_rows(("buy", 10.0, 100.0), ("buy", 11.0, 100.0))
        frame3 = _trade_rows(("buy", 10.6, 200.0))
        result = sections.compare_trade_frames(frame2, frame3)
        assert result["intersection_rows"] == 1
        assert result["value_max_abs_diff"]["shares"]["max_abs_diff"] == 0.0
        assert result["value_max_abs_diff"]["price"]["max_abs_diff"] == pytest.approx(0.1)
        assert result["value_max_abs_diff"]["amount"]["max_abs_diff"] == pytest.approx(20.0)

    def test_duplicate_key_counts_registered_in_report(self, tmp_path):
        """重复键分组计数随节登记（报告 c 节 fold 条目直读）。"""
        env = _synth_env(tmp_path)
        fold_dir2 = env["arm2"] / "runs" / "batch_arm2" / "folds" / "split00"
        _trade_rows(("buy", 10.0, 100.0), ("buy", 11.0, 100.0)).to_parquet(
            fold_dir2 / "trades.parquet", index=False
        )
        fold_dir3 = env["arm3"] / "runs" / "batch_arm3" / "folds" / "split00"
        _trade_rows(("buy", 10.5, 200.0)).to_parquet(fold_dir3 / "trades.parquet", index=False)
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        fold0 = next(
            entry
            for entry in report["sections"]["c_trade_impact"]["folds"]
            if entry["fold"] == "split00"
        )
        assert fold0["duplicate_key_groups_arm2"] == 1
        assert fold0["intersection_rows"] == 1
        assert fold0["value_max_abs_diff"]["amount"]["max_abs_diff"] == 0.0


# ---------------------------------------------------------------------------
# P3-8：compare_chain_navs 的 split_index dtype 统一
# ---------------------------------------------------------------------------


class TestChainNavDtype:
    def test_split_index_string_vs_int_unified(self):
        """split_index 字符串 vs 整数混合 dtype ⇒ 显式 astype(int) 后逐折正确对齐。"""
        nav2 = pd.DataFrame(
            [
                {"date": "20260105", "nav": 1.0, "split_index": "0"},
                {"date": "20260106", "nav": 1.01, "split_index": "0"},
                {"date": "20260106", "nav": 1.01, "split_index": "1"},
                {"date": "20260107", "nav": 1.02, "split_index": "1"},
            ]
        )
        nav3 = pd.DataFrame(
            [
                {"date": "20260105", "nav": 1.0, "split_index": 0},
                {"date": "20260106", "nav": 1.01, "split_index": 0},
                {"date": "20260106", "nav": 1.01, "split_index": 1},
                {"date": "20260107", "nav": 1.07, "split_index": 1},
            ]
        )
        result = sections.compare_chain_navs(nav2, nav3)
        split0 = next(f for f in result["folds"] if f["split_index"] == 0)
        split1 = next(f for f in result["folds"] if f["split_index"] == 1)
        assert split0["days_arm2"] == 2 and split0["common_days"] == 2
        assert split0["max_abs_diff"] == pytest.approx(0.0)
        assert split1["max_abs_diff"] == pytest.approx(0.05)
        assert result["max_abs_diff"] == pytest.approx(0.05)
        assert result["final_fold_last_date_diff"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# R2-T5A-R1-03：实读绑定（快照登记集合 ↔ a 节实际读取集合）
# ---------------------------------------------------------------------------


class TestReadBinding:
    def test_other_data_root_substitution_fails(self, tmp_path):
        """评审复现：归因实读根被 other_data_root 顶替（同内容不同路径）⇒ 实读越界 FAIL。"""
        env = _synth_env(tmp_path / "case_a")
        other_root = tmp_path / "case_b" / "data_root"
        other_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(env["data_root"], other_root)
        report = build_report(
            env["arm2"],
            env["arm3"],
            env["b0p"],
            other_root,
            run_id="binding_substitution",
            observer_identity=_OBSERVER,
        )
        assert report["verdict"]["verdict"] == "FAIL"
        assert "实读越界" in _error_text(report)

    def test_read_content_drift_fails(self, tmp_path):
        """实读内容漂移：快照采集后消费文件被改 ⇒ 复核漂移 + 实读漂移 FAIL。"""
        env = _synth_env(tmp_path)
        _tamper_cs_train_day(env, _DAYS[0])
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        text = _error_text(report)
        assert "实读内容漂移" in text or "消费快照当前态复核漂移" in text

    def test_snapshot_digest_binding_mismatch_fails(self, tmp_path):
        """摘要绑定：快照文件被改写但 arm_meta 摘要未同步 ⇒ 摘要不一致 FAIL。"""
        env = _synth_env(tmp_path)
        snapshot_path = env["arm3"] / "consumption_snapshot.json"
        snapshot_path.write_text(
            snapshot_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )  # 合法 JSON 但字节已变 ⇒ sha256 不等
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "摘要绑定失败" in _error_text(report)

    def test_empty_snapshot_fails(self, tmp_path):
        """快照为空清单（[]）⇒ FAIL 前置（无登记集合可供实读绑定）。"""
        env = _synth_env(tmp_path)
        (env["arm2"] / "consumption_snapshot.json").write_text("[]", encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "空清单" in _error_text(report)

    def test_missing_anchor_fails(self, tmp_path):
        """实读绑定锚点缺失（before_guard.json 不存在）⇒ FAIL 前置。"""
        env = _synth_env(tmp_path)
        (env["arm3"] / "before_guard.json").unlink()
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "缺 before_guard.json 消费锚点" in _error_text(report)

    def test_normal_binding_pass(self, tmp_path):
        """正常绑定：实读 ⊆ 登记集合且逐文件哈希一致 + 摘要绑定 ⇒ 无绑定类错误。"""
        report = _run(_synth_env(tmp_path))
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        section_g = report["sections"]["g_identity_appendix"]
        assert section_g["arm2"]["consumption_snapshot_verified"] is True
        assert "实读" not in _error_text(report) and "摘要绑定" not in _error_text(report)


# ---------------------------------------------------------------------------
# R2-T5A-R1-04：缓存命名空间隔离与条目内容身份
# ---------------------------------------------------------------------------


class TestCacheNamespace:
    def test_different_root_not_hit(self, tmp_path):
        """不同根（显示路径不同 ⇒ 快照字节不同 ⇒ 命名空间不同）：旧缓存不得命中。"""
        env_a = _synth_env(tmp_path / "a")
        # 环境 B：数据根目录名不同（other_data_root）⇒ 快照显示路径不同 ⇒ 摘要不同
        root_b = tmp_path / "b"
        data_root_b = root_b / "other_data_root"
        shutil.copytree(env_a["data_root"], data_root_b)
        env_b = {
            "data_root": data_root_b,
            "arm2": _build_arm(root_b, "arm2", data_root=data_root_b),
            "arm3": _build_arm(root_b, "arm3", data_root=data_root_b),
            "b0p": _build_b0p(root_b),
        }
        cache_dir = tmp_path / "shared_cache"
        report_a = _run(env_a, cache_dir=cache_dir)
        ns_a = _day_stats(report_a)["namespace"]
        assert _day_stats(report_a)["misses"] == 2

        report_b = _run(env_b, cache_dir=cache_dir)
        stats_b = _day_stats(report_b)
        assert stats_b["namespace"] != ns_a  # 不同根 ⇒ 命名空间隔离
        assert stats_b["misses"] == 2  # 唯一日重算（未命中 A 环境条目）
        assert report_b["verdict"]["verdict"] == "PASS", _error_text(report_b)

    def test_new_data_state_not_hit(self, tmp_path):
        """同根新数据态（内容变 + 快照重采 ⇒ 摘要变 ⇒ 命名空间变）：旧缓存不得命中。"""
        env = _synth_env(tmp_path)
        cache_dir = tmp_path / "attribution_cache"
        cold = _run(env, cache_dir=cache_dir)
        ns_before = _day_stats(cold)["namespace"]
        _tamper_cs_train_day(env, _DAYS[0])
        _recollect_snapshots(env)  # 合法数据态演进：重采快照并回填摘要绑定
        warm = _run(env, cache_dir=cache_dir)
        stats = _day_stats(warm)
        assert stats["namespace"] != ns_before
        assert stats["misses"] == 2  # 全量重算，旧条目零命中

    def test_entry_identity_mismatch_recomputes(self, tmp_path):
        """条目内容身份（②）：文件变更但快照未重采（命名空间不变）⇒ 身份不符重算。"""
        env = _synth_env(tmp_path)
        cache_dir = tmp_path / "attribution_cache"
        cold = _run(env, cache_dir=cache_dir)
        assert _day_stats(cold)["misses"] == 2
        _tamper_cs_train_day(env, _DAYS[0])  # 内容变、快照不重采 ⇒ 命名空间不变
        warm = _run(env, cache_dir=cache_dir)
        stats = _day_stats(warm)
        assert stats["namespace"] == _day_stats(cold)["namespace"]
        assert stats["misses"] == 1  # 被改日身份不符 ⇒ 未命中重算（其余日命中）
        assert warm["verdict"]["verdict"] == "FAIL"  # 快照漂移/实读漂移门拦截

    def test_format_version_bump_not_hit(self, tmp_path, monkeypatch):
        """规则/实现变更 ⇒ FORMAT_VERSION 递增 ⇒ 命名空间变化 ⇒ 旧缓存不得命中。"""
        env = _synth_env(tmp_path)
        cache_dir = tmp_path / "attribution_cache"
        cold = _run(env, cache_dir=cache_dir)
        assert _day_stats(cold)["misses"] == 2
        monkeypatch.setattr(sections.DayCompareCache, "FORMAT_VERSION", "v9")
        warm = _run(env, cache_dir=cache_dir)
        stats = _day_stats(warm)
        assert stats["namespace"] != _day_stats(cold)["namespace"]
        assert stats["misses"] == 2  # 版本 bump 后全量重算
        assert stats["format_version"] == "v9"


# ---------------------------------------------------------------------------
# R2-T5A-R1-05：预测证据自洽（行集指纹/行数/日集/非空）+ signal_topk 登记
# ---------------------------------------------------------------------------


class TestPredictionSelfConsistency:
    def test_empty_predictions_fails(self, tmp_path):
        """评审复现一：零行 parquet（同步哈希与全部登记）⇒ 空产物拒绝 FAIL。"""
        env = _synth_env(tmp_path)
        _rewrite_predictions(env["arm3"], "split00", [])
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "空产物" in _error_text(report)

    def test_fingerprint_not_self_consistent_fails(self, tmp_path):
        """行集自洽：parquet 改写后 sha/rows 同步但 fingerprint 字段未同步 ⇒ ① FAIL。"""
        env = _synth_env(tmp_path)
        _rewrite_predictions(
            env["arm3"],
            "split00",
            [{"trade_date": "20260106", "ts_code": "000001.SZ", "score": 0.99}],
            sync_fingerprint=False,
        )
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "行集自洽" in _error_text(report)

    def test_days_outside_test_window_fails(self, tmp_path):
        """日集越窗：预测行落在折 boundary test 窗外（登记同步一致）⇒ ③ FAIL。"""
        env = _synth_env(tmp_path)
        _rewrite_predictions(
            env["arm3"],
            "split01",
            [{"trade_date": "20260107", "ts_code": "000001.SZ", "score": 0.11}],
        )
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "越窗" in _error_text(report)

    def test_self_consistent_predictions_pass(self, tmp_path):
        """正常：行集指纹/行数/日集/越窗全部自洽 ⇒ 无自洽类错误。"""
        report = _run(_synth_env(tmp_path))
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        assert "自洽" not in _error_text(report) and "越窗" not in _error_text(report)

    def test_signal_topk_registered(self, tmp_path):
        """signal_topk.parquet 产出时 b 节登记 rows/days/empty_days（展示，不设门）。"""
        env = _synth_env(tmp_path)
        for arm in (env["arm2"], env["arm3"]):
            for fold in _FOLDS:
                fold_dir = arm / "folds" / fold
                topk = pd.DataFrame(
                    [
                        {
                            "trade_date": "20260106",
                            "rank": 1,
                            "ts_code": "600000.SH",
                            "score": 0.22,
                        },
                        {
                            "trade_date": "20260106",
                            "rank": 2,
                            "ts_code": "000001.SZ",
                            "score": 0.11,
                        },
                    ]
                )
                topk.to_parquet(fold_dir / "signal_topk.parquet", index=False)
                fp_path = fold_dir / "fingerprint.json"
                doc = json.loads(fp_path.read_text(encoding="utf-8"))
                doc["signal_topk_path"] = f"folds/{fold}/signal_topk.parquet"
                doc["signal_topk_sha256"] = _sha256_file(fold_dir / "signal_topk.parquet")
                doc["signal_topk_rows"] = len(topk)
                doc["signal_topk_days"] = ["20260106"]
                doc["signal_topk_empty_days"] = ["20260105"]
                fp_path.write_text(json.dumps(doc), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "PASS", _error_text(report)
        fold0 = _fold_b(report, "split00")
        for key in ("signal_topk_arm2", "signal_topk_arm3"):
            entry = fold0[key]
            assert entry["available"] is True
            assert entry["rows"] == 2 and entry["days"] == 1
            assert entry["empty_days"] == ["20260105"]
            assert entry["sha256_match"] is True


# ---------------------------------------------------------------------------
# R2-T5A-R1-07：净值有限性前置 + 派生指标一致性核对
# ---------------------------------------------------------------------------


class TestNavFiniteAndDerived:
    def test_nan_chain_nav_fails(self, tmp_path):
        """评审复现：chain_nav 终点 NaN ⇒ 有限性前置拒绝 FAIL（不得登记为零差异）。"""
        env = _synth_env(tmp_path)
        nav_path = env["arm3"] / "runs" / "batch_arm3" / "chain_nav.parquet"
        nav = pd.read_parquet(nav_path)
        nav.loc[nav.index[-1], "nav"] = float("nan")
        nav.to_parquet(nav_path, index=False)
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "NaN/Inf" in _error_text(report)

    def test_nan_daily_fails(self, tmp_path):
        """逐折 daily nav 含 NaN ⇒ 有限性前置拒绝 FAIL。"""
        env = _synth_env(tmp_path)
        daily_path = env["arm3"] / "runs" / "batch_arm3" / "folds" / "split00" / "daily.parquet"
        daily = pd.read_parquet(daily_path)
        daily.loc[daily.index[-1], "nav"] = float("nan")
        daily.to_parquet(daily_path, index=False)
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "NaN/Inf" in _error_text(report)

    def test_derived_metrics_mismatch_fails(self, tmp_path):
        """派生指标核对：chain_metrics.json 登记值与 chain_nav 重算不一致 ⇒ FAIL。"""
        env = _synth_env(tmp_path)
        metrics_path = env["arm3"] / "chain_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["cagr"] = float(metrics["cagr"]) + 0.5
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        report = _run(env)
        assert report["verdict"]["verdict"] == "FAIL"
        assert "派生指标核对失败" in _error_text(report)


# ---------------------------------------------------------------------------
# R2-T5AE-R1-01（reviewer2 复审一 P1）：归因门明确类越界保留否决效力
# ---------------------------------------------------------------------------


def _aggregator_single_column(
    col: str, *, rows: int, over: int, peak: float, over_nan_rows: int = 0, date: str = "20220505"
) -> dict[str, Any]:
    """单列单日差异直入正式聚合器（reviewer2 复现口径；单日统计结构）。"""
    agg = sections.ColumnImpactAggregator()
    agg.add_day(
        {
            "date": date,
            "n_only_ref": 0,
            "n_only_panel": 0,
            "sample_only_ref": [],
            "sample_only_panel": [],
            "cols_only_ref": [],
            "cols_only_panel": [],
            "col_rows": {col: rows},
            "columns": {
                col: {
                    "over_rows": over,
                    "rows": rows,
                    "max_abs_diff": peak,
                    "over_nan_rows": over_nan_rows,
                    "samples": [],
                }
            },
        }
    )
    return agg.finalize()


class TestD14SpecialVetoOverSparseFallback:
    """strict 稀疏伴随兜底不得覆盖归因门 D-14 专项明确越界（R2-T5AE-R1-01）。

    兜底仅对归因门第六类（``ATTRIB_VIOLATION_TEXT``，词表无类）的伴随缺口启用；
    归因门已给出明确类而越界者（D-14 三上限 / D-04 登记窗）保留否决效力——
    strict 稀疏伴随分支无超幅检查，不区分拒绝原因会重新打开「稀疏但超幅」敞口。
    """

    def test_zscore_macd_sparse_over_peak_rejected(self):
        """1/3000 行、maxΔ=21.0 > 专项上限 20.0 ⇒ 拒绝（稀疏伴随分支不得覆盖）。"""
        report = _aggregator_single_column("zscore_macd_hist", rows=3000, over=1, peak=21.0)
        entry = report["columns"]["zscore_macd_hist"]
        assert entry["boundary_violation"] is True
        assert entry["attribution_mode"] == "attrib"
        assert report["closed"] is False

    def test_skewness_sparse_over_peak_rejected(self):
        """1/3000 行、maxΔ=6.0 > 专项上限 5.0 ⇒ 拒绝。"""
        report = _aggregator_single_column("skewness_20", rows=3000, over=1, peak=6.0)
        entry = report["columns"]["skewness_20"]
        assert entry["boundary_violation"] is True
        assert entry["attribution_mode"] == "attrib"
        assert report["closed"] is False

    def test_amplitude_sparse_companion_still_fallback(self):
        """实际显现类保持放行：amplitude 0.0/NaN 伴随单点（第六类）仍走 strict 兜底。"""
        report = _aggregator_single_column(
            "amplitude", rows=3000, over=1, peak=0.0, over_nan_rows=1
        )
        entry = report["columns"]["amplitude"]
        assert entry["attribution_mode"] == "strict_d11_fallback"
        assert entry["boundary_violation"] is False
        assert report["closed"] is True

    def test_dense_anchor_sensitive_column_rejected(self):
        """密集主分支反例保持拒绝（R2-T5AE-01 修复不回退：D-11 主分支不入归因词表）。"""
        report = _aggregator_single_column("volatility_20", rows=3000, over=100, peak=0.5)
        assert report["columns"]["volatility_20"]["boundary_violation"] is True
        assert report["closed"] is False

    def test_dense_anchor_sensitive_single_day_over_peak(self):
        """单日密集且超限的长记忆列（skewness_20 26 行 > 20 行上限）保持拒绝。"""
        report = _aggregator_single_column("skewness_20", rows=3000, over=26, peak=1.0)
        assert report["columns"]["skewness_20"]["boundary_violation"] is True

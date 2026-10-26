# -*- coding: utf-8 -*-
"""P3 T5 阶段 A 评审意见处置的收紧语义测试（合成数据，不读取真实生产数据/产物）。

覆盖（逐条映射处置清单）：
- R2-T5A-03-① 折集交叉核验：两臂 runs 折集必须恰为 14 折（== EXPECTED_SPLITS，
  与指纹折集同一集合），缺一/多一即 FAIL 并指出折号；旧「2 折 runs + 14 折指纹
  ⇒ PASS」语义收紧为 FAIL；
- R2-T5A-R1-01 信号路径 top-k 辅证：读两臂 folds/splitNN/signal_topk.parquet
  （真实信号过滤通路产物）——指纹增键缺失/文件缺失/哈希漂移/rows 交叉核对不符 ⇒
  FAIL；完整性门（signal_topk_days ∪ empty_days == predictions 日集 == score_days，
  缺日 FAIL）；全折 rows 合计 == 0 ⇒ FAIL（零辅证不得过总门）；
  **评估路径 topk CSV 双臂全折同缺 ⇒ FAIL**（取代 R1 前 skip 登记语义）；
- R2-T5A-03-③ 空摘要不可过：batch_fingerprint 缺字段/None/非法 ⇒ FAIL（双缺同非法
  亦 FAIL）；逐折指纹值先验合法 sha256；
- P2-2 + R2-T5A-R1-02 容差收口：chain_nav/summary（绩效统计允许清单列）/
  daily(nav,total_value) 三面同一 1e-6 容差投影，各面 max|Δ| 登记；**允许清单外
  bt_* 配置列投影前逐位精确**（0.15 vs 0.1500005 ⇒ FAIL）；任一面超容差 ⇒ FAIL；
- R2-T5A-07 有限性门：G-code 与回归边两路 chain_nav/daily/summary 任一 NaN/Inf
  ⇒ FAIL（非「容差内」，NaN 不得借 NaN==NaN 精确口径绕过）；
- R2-T5A-09 + R2-T5A-R1-06 同分排名互换：分组身份配对口径——多 K 出口同股跨 K
  合法重复不误报；真实同分互换计数不判 mismatch；**异分反转（排名与分数矛盾）⇒
  selection_mismatch**；跨 K 边界 ⇒ selection_mismatch；>入选数 5% 登记判据保持。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t5_compare_gates import (  # noqa: E402
    build_gcode_verdict,
    compare_gcode_fingerprints,
    compare_gcode_runs,
    compare_regression_edge,
    compare_signal_topk,
    compare_topk_details,
)
from scripts.v2_p3.t5_compare_gates_signal import _compare_topk_fold  # noqa: E402
from src.lazybull.v2.evidence.fingerprint_keys import (  # noqa: E402
    config_fingerprint_keys,
    fingerprint,
)
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch  # noqa: E402


def _sha(text: str) -> str:
    """合成合法 64 位 sha256 hex 指纹值。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 合成夹具（与 test_v2_p3_t5_compare_gates.py 同族，本文件自包含不跨测试 import）
# ---------------------------------------------------------------------------

_SYNTH_CONFIG: Dict[str, object] = {
    "bt_top_n": 20,
    "bt_max_weight_per_stock": 0.15,  # 配置浮点列实例（R2-T5A-R1-02：非统计列，逐位精确面）
    "stock_domain": "main",
    "wf_start_date": "20240102",
    "wf_end_date": "20241231",
}


def _synth_nav_curve(dates: List[str], values: List[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": list(dates),
            "portfolio_value": list(values),
            "capital": [v * 0.4 for v in values],
            "market_value": [v * 0.6 for v in values],
        }
    )
    frame["nav"] = frame["portfolio_value"] / 1_000_000.0
    frame["return"] = frame["nav"] - 1.0
    return frame


def _synth_fold(split: int, model_version: int) -> Dict[str, object]:
    """最小合法合成折：1 笔买入 + 两日净值（bt_total_return 与 nav 比自洽）。"""
    day0, day1 = ("2024-01-02", "2024-01-03") if split == 0 else ("2024-07-01", "2024-07-02")
    trades = pd.DataFrame(
        [
            {
                "date": day0,
                "signal_date": "2023-12-29",
                "stock": "000001.SZ",
                "action": "buy",
                "price": 10.0,
                "shares": 100,
                "amount": 1000.0,
                "cost": 5.0,
            }
        ]
    )
    nav = _synth_nav_curve([day0, day1], [1_000_000.0, 1_010_000.0])
    return {
        "split_index": split,
        "model_version": model_version,
        "train_start": "20180101",
        "train_end": "20231229",
        "test_start": day0.replace("-", ""),
        "test_end": day1.replace("-", ""),
        "metrics": {
            "bt_total_return": 0.01,
            "bt_annual_return": 0.02,
            "bt_max_drawdown": -0.01,
            "bt_volatility": 0.1,
            "bt_sharpe": 1.0,
            "bt_calmar": 2.0,
            "bt_trading_days": 2,
            "bt_start": day0.replace("-", ""),
            "bt_end": day1.replace("-", ""),
            "bt_top_n": 20,
        },
        "nav_curve": nav,
        "trades": trades,
        "attribution": None,
        "holdings_snapshot": None,
    }


def _write_batch(
    runs_root: Path,
    batch_id: str,
    model_version: int = 24008,
    data_state_id: str = "ds-syn",
    *,
    split_count: int = 14,
) -> Path:
    """写出合成 runs 批次（默认 14 折 = G-code ② 折集交叉核验收紧后口径）。"""
    return write_runs_batch(
        [_synth_fold(split, model_version) for split in range(split_count)],
        batch_id=batch_id,
        runs_root=runs_root,
        config=dict(_SYNTH_CONFIG),
        code_state={"git_commit": "syn0000", "git_dirty": False},
        data_state={"data_state_id": data_state_id, "sources": {"daily": "2024-07-02"}},
    )


def _arm_fold_days(split: int) -> List[str]:
    """合成折的信号/打分日清单（与 _synth_predictions 日集一致）。"""
    return ["20240102", "20240103"] if split == 0 else ["20240701", "20240702"]


def _synth_predictions(split: int) -> pd.DataFrame:
    """合成一折 predictions.parquet 帧（跨路契约形态；每日 25 股 ⇒ top-20 截断）。"""
    rows = [
        {"trade_date": day, "ts_code": f"T{idx:04d}", "score": 1.0 - idx * 0.01}
        for day in _arm_fold_days(split)
        for idx in range(25)
    ]
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _synth_signal_topk(split: int) -> pd.DataFrame:
    """合成一折 signal_topk.parquet 帧（跨路契约形态：trade_date/rank/ts_code/score，
    按 (trade_date,rank) 升序；每日 top-20，rank=1..20）。"""
    rows = []
    for day in _arm_fold_days(split):
        for rank in range(1, 21):
            rows.append(
                {
                    "trade_date": day,
                    "rank": rank,
                    "ts_code": f"T{rank - 1:04d}",
                    "score": 1.0 - (rank - 1) * 0.01,
                }
            )
    return pd.DataFrame(rows)


def _write_signal_fold(
    arm_dir: Path,
    split: int,
    frame: pd.DataFrame,
    *,
    days: Optional[List[str]] = None,
    empty_days: Optional[List[str]] = None,
    write_predictions: bool = True,
    write_fingerprint: bool = True,
) -> Path:
    """写一折信号辅证产物 + 指纹增键（sha256 自重算文件兑现登记一致）。"""
    name = f"split{split:02d}"
    fold_dir = arm_dir / "folds" / name
    fold_dir.mkdir(parents=True, exist_ok=True)
    if write_predictions:
        _synth_predictions(split).to_parquet(fold_dir / "predictions.parquet", index=False)
    path = fold_dir / "signal_topk.parquet"
    frame.to_parquet(path, index=False)
    covered_days = list(days if days is not None else _arm_fold_days(split))
    if write_fingerprint:
        fp_path = fold_dir / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.exists() else {}
        doc.update(
            {
                "fingerprint": doc.get("fingerprint", _sha(f"fp-{name}")),
                "score_days": covered_days,
                "signal_topk_path": f"folds/{name}/signal_topk.parquet",
                "signal_topk_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "signal_topk_rows": int(len(frame)),
                "signal_topk_days": [d for d in covered_days if d not in (empty_days or [])],
                "signal_topk_empty_days": list(empty_days or []),
            }
        )
        fp_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def _write_arm_signal_artifacts(arm_dir: Path, *, skip_folds: tuple = ()) -> None:
    """写一臂 14 折信号辅证产物（R2-T5A-R1-01 口径：predictions 日集锚 +
    signal_topk.parquet + fingerprint.json 增键五件套）。"""
    for split in range(14):
        if f"split{split:02d}" in skip_folds:
            continue
        _write_signal_fold(arm_dir, split, _synth_signal_topk(split))


def _make_arm_fingerprints(root: Path, arm: str) -> Path:
    """合成一臂 14 折指纹 + 批次汇总（值一律合法 sha256 hex）。"""
    arm_dir = root / arm
    folds_map: Dict[str, str] = {}
    for split in range(14):
        name = f"split{split:02d}"
        fingerprint = _sha(f"fp-{name}")
        folds_map[name] = fingerprint
        fold_dir = arm_dir / "folds" / name
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "fingerprint.json").write_text(
            json.dumps({"fingerprint": fingerprint, "n_days": 2, "n_rows": 50}),
            encoding="utf-8",
        )
    (arm_dir / "batch_fingerprint.json").write_text(
        json.dumps({"folds": folds_map, "batch_fingerprint": _sha("batch-fp")}),
        encoding="utf-8",
    )
    return arm_dir


def _shift_parquet_column(path: Path, column: str, delta: float) -> None:
    """parquet 数值列整体平移 delta（容差反例夹具；写出参数与仓库惯例一致）。"""
    frame = pd.read_parquet(path)
    frame[column] = frame[column].astype(float) + delta
    frame.to_parquet(path, index=False)


def _set_parquet_value(path: Path, column: str, row: int, value: float) -> None:
    """parquet 数值列单点置值（NaN/Inf 反例夹具）。"""
    frame = pd.read_parquet(path)
    frame.loc[row, column] = value
    frame.to_parquet(path, index=False)


def _rewrite_batch_meta_config(batch_dir: Path) -> None:
    """summary 篡改后同步重算 batch_meta 的 config 与 config_fingerprint（契约 §9.3
    自校验口径：config 键 = 指纹键 ∩ summary 首行，指纹 = fingerprint_keys.fingerprint）。"""
    meta_path = Path(batch_dir) / "batch_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = pd.read_parquet(Path(batch_dir) / "summary.parquet")
    row = summary.iloc[0]
    config = {
        key: (row[key].item() if hasattr(row[key], "item") else row[key])
        for key in config_fingerprint_keys(summary.columns.tolist())
        if key in summary.columns and pd.notna(row[key])
    }
    meta["config"] = config
    meta["config_fingerprint"] = fingerprint(config)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# R2-T5A-03-① 折集交叉核验：两臂 runs 折集必须恰为 14 折
# ---------------------------------------------------------------------------


class TestRunsFoldSetCrossCheck:
    """收紧后口径：runs 折集 == set(range(14))（== 指纹折集），缺一/多一即 FAIL。"""

    def test_two_fold_runs_fail_names_missing_splits(self, tmp_path):
        """2 折 runs（旧组装测试口径）⇒ FAIL 并逐一指出缺折折号。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a", split_count=2)
        dir_b = _write_batch(tmp_path / "b", "batch_b", split_count=2)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        fold_errors = [m for m in result["errors"] if "折集" in m]
        assert fold_errors, result["errors"]
        assert any("split02" in m and "split13" in m for m in fold_errors)

    def test_extra_fold_fail_names_split(self, tmp_path):
        """15 折（多 split14）⇒ FAIL 并指出多折折号。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a", split_count=15)
        dir_b = _write_batch(tmp_path / "b", "batch_b", split_count=15)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("多折" in m and "split14" in m for m in result["errors"])

    def test_single_side_short_fail(self, tmp_path):
        """一臂 14 折、另一臂 13 折 ⇒ FAIL（缺折侧被指名）。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b", split_count=13)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("臂②" in m and "split13" in m for m in result["errors"])

    def test_assembly_two_fold_runs_now_fail(self, tmp_path):
        """旧 TestGcodeVerdictAssembly 语义收紧：14 折指纹 + 2 折 runs ⇒ 总 FAIL。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        _write_batch(arm_a / "runs", "batch_arm1", split_count=2)
        _write_batch(arm_b / "runs", "batch_arm2", split_count=2)
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        verdict = build_gcode_verdict(arm_a, arm_b)
        assert verdict["verdict"] == "FAIL"
        assert verdict["sub_gates"]["fingerprint"]["verdict"] == "PASS"
        assert verdict["sub_gates"]["signal_topk_auxiliary"]["verdict"] == "PASS"
        runs_gate = verdict["sub_gates"]["runs"]
        assert runs_gate["verdict"] == "FAIL"
        assert any("折集" in m for m in runs_gate["errors"])


# ---------------------------------------------------------------------------
# R2-T5A-03-③ 空摘要不可过 + 逐折指纹值 sha256 先验
# ---------------------------------------------------------------------------


def _write_batch_doc(arm_dir: Path, doc: Dict[str, object]) -> None:
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "batch_fingerprint.json").write_text(json.dumps(doc), encoding="utf-8")


class TestFingerprintSha256Precondition:
    """空摘要/非法值 ⇒ FAIL（双侧同缺同非法亦 FAIL，禁止双 None 误判 PASS）。"""

    def _arms_with_fold_fps(self, tmp_path: Path) -> tuple:
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        return arm_a, arm_b

    def test_batch_fingerprint_field_missing_both_sides_fail(self, tmp_path):
        arm_a, arm_b = self._arms_with_fold_fps(tmp_path)
        for arm_dir in (arm_a, arm_b):
            doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
            del doc["batch_fingerprint"]
            _write_batch_doc(arm_dir, doc)
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert result["batch_fingerprint"]["match"] is False
        assert any("缺 batch_fingerprint 字段或值非法" in m for m in result["errors"])

    def test_batch_fingerprint_none_both_sides_fail(self, tmp_path):
        arm_a, arm_b = self._arms_with_fold_fps(tmp_path)
        for arm_dir in (arm_a, arm_b):
            doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
            doc["batch_fingerprint"] = None
            _write_batch_doc(arm_dir, doc)
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert result["batch_fingerprint"]["match"] is False

    def test_batch_fingerprint_illegal_value_fail(self, tmp_path):
        """非 64 位 hex（双侧同非法值亦 FAIL）。"""
        arm_a, arm_b = self._arms_with_fold_fps(tmp_path)
        for arm_dir in (arm_a, arm_b):
            doc = json.loads((arm_dir / "batch_fingerprint.json").read_text(encoding="utf-8"))
            doc["batch_fingerprint"] = "batch-fp"
            _write_batch_doc(arm_dir, doc)
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert result["batch_fingerprint"]["match"] is False

    def test_fold_fingerprint_illegal_value_fail_names_fold(self, tmp_path):
        """逐折指纹值非法 ⇒ FAIL 并指出折号（先验合法再比较）。"""
        arm_a, arm_b = self._arms_with_fold_fps(tmp_path)
        path = arm_b / "folds" / "split04" / "fingerprint.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["fingerprint"] = "fp-split04"
        path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split04" in m and "指纹值非法" in m for m in result["errors"])


# ---------------------------------------------------------------------------
# R2-T5A-R1-01 信号路径 top-k 辅证（signal_topk.parquet 真实信号通路产物）
# ---------------------------------------------------------------------------


class TestSignalTopkAuxiliary:
    """读 signal_topk.parquet + 指纹增键/sha256/rows 核验 + 完整性门；零辅证 FAIL。"""

    def test_identical_artifacts_pass(self, tmp_path):
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]
        assert len(result["folds"]) == 14
        assert all(fold["status"] == "match" for fold in result["folds"])

    def test_missing_signal_topk_file_fail(self, tmp_path):
        """缺 signal_topk.parquet 文件 ⇒ FAIL 指出折号（不得静默跳过）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        (arm_b / "folds" / "split03" / "signal_topk.parquet").unlink()
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split03" in m and "缺 signal_topk 产物文件" in m for m in result["errors"])

    def test_missing_fingerprint_keys_fail(self, tmp_path):
        """fingerprint.json 缺信号 topk 增键 ⇒ FAIL。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        fp_path = arm_b / "folds" / "split02" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        for key in ("signal_topk_path", "signal_topk_sha256"):
            del doc[key]
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split02" in m and "缺信号 topk 增键" in m for m in result["errors"])

    def test_hash_drift_fail(self, tmp_path):
        """产物写出后被篡改（sha256 登记不更新）⇒ 哈希漂移 FAIL。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        path = arm_b / "folds" / "split00" / "signal_topk.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "score"] = frame.loc[0, "score"] + 1e-9
        frame.to_parquet(path, index=False)  # 文件变、登记 sha256 不变
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split00" in m and "哈希漂移" in m for m in result["errors"])

    def test_rows_crosscheck_mismatch_fail(self, tmp_path):
        """登记 signal_topk_rows 与实际行数不符 ⇒ FAIL。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        fp_path = arm_b / "folds" / "split01" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        doc["signal_topk_rows"] = int(doc["signal_topk_rows"]) - 1
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split01" in m and "交叉核对不符" in m for m in result["errors"])

    def test_days_set_unlink_predictions_fail(self, tmp_path):
        """覆盖日集与主证据（predictions.parquet）日集不符 ⇒ FAIL（关联性核验）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        fp_path = arm_b / "folds" / "split00" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        doc["signal_topk_days"] = doc["signal_topk_days"][:1]  # 漏报一日
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split00" in m and "主证据日集不符" in m for m in result["errors"])

    def test_missing_score_days_fail(self, tmp_path):
        """覆盖日集缺 score_days 日 ⇒ FAIL（缺日；预期日域一致性核验）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        fp_path = arm_b / "folds" / "split00" / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8"))
        doc["score_days"] = doc["score_days"] + ["20240109"]  # 预期日域多出一天
        fp_path.write_text(json.dumps(doc), encoding="utf-8")
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split00" in m and "缺日 FAIL" in m for m in result["errors"])

    def test_missing_predictions_integrity_fail(self, tmp_path):
        """缺 predictions.parquet ⇒ 关联性核验不可行 ⇒ FAIL。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        (arm_b / "folds" / "split05" / "predictions.parquet").unlink()
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split05" in m and "关联性核验不可行" in m for m in result["errors"])

    def test_zero_auxiliary_rows_fail(self, tmp_path):
        """全折 signal_topk_rows 合计 == 0 ⇒ FAIL（零辅证不得过总门）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        for arm in (arm_a, arm_b):
            for split in range(14):
                days = _arm_fold_days(split)
                empty = pd.DataFrame(columns=["trade_date", "rank", "ts_code", "score"])
                _write_signal_fold(arm, split, empty, empty_days=days)
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert sum("零辅证不得过总门" in m for m in result["errors"]) == 2

    def test_empty_days_fold_legitimate_pass(self, tmp_path):
        """部分折合法零选股（empty_days 承载）但全臂 rows 合计 > 0 ⇒ 可过。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        for arm in (arm_a, arm_b):
            _write_arm_signal_artifacts(arm)
            empty = pd.DataFrame(columns=["trade_date", "rank", "ts_code", "score"])
            _write_signal_fold(arm, 0, empty, empty_days=_arm_fold_days(0))
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]

    def test_score_difference_fail(self, tmp_path):
        """臂②某股分数真实差异（臂内单调保持）⇒ FAIL（多重集不一致，非互换）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        frame_b = _synth_signal_topk(0)
        mask = (frame_b["trade_date"] == "20240102") & (frame_b["ts_code"] == "T0005")
        frame_b.loc[mask, "score"] = 0.949  # 原 0.95 ⇒ 微差且臂内单调保持
        _write_signal_fold(arm_b, 0, frame_b)  # 重写 split00（sha256 重算）
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split00" in m and "多重集不一致" in m for m in result["errors"])

    def test_tie_swap_pass_registered(self, tmp_path):
        """同分并列位次互换（两臂同分）⇒ PASS + 互换计数 + >5% 登记（信息性）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        tied = _synth_signal_topk(0)
        day_mask = tied["trade_date"] == "20240102"
        tied.loc[day_mask, "score"] = 0.5  # 20240102 全日 20 股同分（两臂同分才成立）
        _write_signal_fold(arm_a, 0, tied)
        swapped = tied.copy()
        swapped.loc[day_mask, "rank"] = tied.loc[day_mask, "rank"].to_numpy()[::-1]
        _write_signal_fold(arm_b, 0, swapped)  # 同分段内 rank 完全反转
        result = compare_signal_topk(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]
        fold = next(f for f in result["folds"] if f["fold"] == "split00")
        assert fold["status"] == "tie_swaps" and fold["swap_count"] == 20
        assert fold["registered"] is True  # 20 > 40 × 5%
        assert any("split00" in item for item in result["tie_swap_registrations"])

    def test_invalid_signal_topk_contract_raises(self, tmp_path):
        """signal_topk 缺契约列 ⇒ 契约违反抛错（不静默）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_arm_signal_artifacts(arm_a)
        bad = _synth_signal_topk(0).drop(columns=["score"])
        _write_signal_fold(arm_b, 0, bad)
        with pytest.raises(ValueError, match="缺契约列"):
            compare_signal_topk(arm_a, arm_b)


# ---------------------------------------------------------------------------
# R2-T5A-R1-01 评估路径辅证缺失 = FAIL（取代 R1 前 skip 登记语义）
# ---------------------------------------------------------------------------


class TestTopkAuxiliaryMissingFail:
    def test_all_folds_missing_fail(self, tmp_path):
        """全折双臂同缺 ⇒ FAIL（辅证缺失 = FAIL，R2-T5A-R1-01）。"""
        result = compare_topk_details(tmp_path / "arm1", tmp_path / "arm2")
        assert result["verdict"] == "FAIL"
        assert any("辅证缺失 = FAIL" in message for message in result["errors"])

    def test_partial_present_fail(self, tmp_path):
        """只比较一折不得替代完整 14 折辅证，逐一指出双侧缺失折。"""
        rows = [
            {
                "wf_run_id": "wf-a",
                "split_index": 0,
                "trade_date": "20240102",
                "topk": 20,
                "rank": 1,
                "ts_code": "T0001",
                "pred_score": 0.9,
            }
        ]
        for arm in ("arm1", "arm2"):
            raw_dir = tmp_path / arm / "root" / "walk_forward" / "raw"
            raw_dir.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(
                raw_dir / "walk_forward_topk_details_wf-x_split00.csv",
                index=False,
                encoding="utf-8-sig",
            )
        result = compare_topk_details(tmp_path / "arm1", tmp_path / "arm2")
        assert result["verdict"] == "FAIL"
        assert len(result["folds"]) == 1
        assert any("split01" in error for error in result["errors"])
        assert any("split13" in error for error in result["errors"])

    def test_full_other_surfaces_cannot_cover_missing_eval_folds(self, tmp_path):
        """14 折指纹/runs/信号辅证齐全，评估只留一折，G-code 总门也必须拒绝。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        for arm, batch in ((arm_a, "batch_a"), (arm_b, "batch_b")):
            _write_batch(arm / "runs", batch)
            _write_arm_signal_artifacts(arm)
            raw = arm / "root" / "walk_forward" / "raw"
            raw.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "trade_date": "20240102",
                        "topk": 20,
                        "rank": 1,
                        "ts_code": "T0001",
                        "pred_score": 0.9,
                    }
                ]
            ).to_csv(raw / "walk_forward_topk_details_wf-syn_split00.csv", index=False)
        result = build_gcode_verdict(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert result["sub_gates"]["fingerprint"]["verdict"] == "PASS"
        assert result["sub_gates"]["runs"]["verdict"] == "PASS"
        assert result["sub_gates"]["signal_topk_auxiliary"]["verdict"] == "PASS"
        eval_gate = result["sub_gates"]["topk_auxiliary"]
        assert len(eval_gate["folds"]) == 1
        assert len(eval_gate["errors"]) == 13


# ---------------------------------------------------------------------------
# P2-2 容差收口：三面 1e-6 容差投影（chain_nav + summary bt_* + daily nav/total_value）
# ---------------------------------------------------------------------------


class TestToleranceProjectionSurfaces:
    """冻结判据 = 成交逐笔精确 + 净值类 1e-6 容差；容差内投影、各面 max|Δ| 登记。"""

    def _tampered_pair(self, tmp_path: Path) -> tuple:
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b")
        return dir_a, dir_b

    def test_within_tolerance_all_surfaces_pass_with_max_diff(self, tmp_path):
        """容差内 chain_nav + summary + daily 同步微差 ⇒ PASS 且三面 max|Δ| 登记。"""
        dir_a, dir_b = self._tampered_pair(tmp_path)
        _shift_parquet_column(dir_b / "chain_nav.parquet", "nav", 5e-7)
        _shift_parquet_column(dir_b / "summary.parquet", "bt_sharpe", 5e-7)
        _shift_parquet_column(dir_b / "summary.parquet", "bt_annual_return", 2e-7)
        for split in range(14):
            daily_path = dir_b / "folds" / f"split{split:02d}" / "daily.parquet"
            _shift_parquet_column(daily_path, "nav", 5e-7)
            _shift_parquet_column(daily_path, "total_value", 5e-7)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "PASS", result["errors"]
        assert result["nav_tolerance_check"]["max_abs_diff"] == pytest.approx(5e-7)
        # total_value 为 1e6 量级，+5e-7 存在 float64 表示偏差 ⇒ rel 容差断言
        assert result["summary_tolerance_check"]["max_abs_diff"] == pytest.approx(5e-7)
        assert result["daily_tolerance_check"]["max_abs_diff"] == pytest.approx(5e-7, rel=1e-3)
        column_diffs = result["summary_tolerance_check"]["column_max_abs_diff"]
        assert column_diffs["bt_sharpe"] == pytest.approx(5e-7)
        assert column_diffs["bt_annual_return"] == pytest.approx(2e-7)
        # 容差列只含绩效统计允许清单列（R2-T5A-R1-02）；配置列零差异精确通过
        assert set(result["summary_tolerance_check"]["columns"]) == {
            "bt_annual_return",
            "bt_calmar",
            "bt_max_drawdown",
            "bt_sharpe",
            "bt_total_return",
            "bt_volatility",
        }
        config_check = result["summary_config_exact_check"]
        assert config_check["match"] is True
        assert config_check["columns"] == ["bt_max_weight_per_stock"]
        projected_notes = [note for note in result["notes"] if "投影对齐" in note]
        assert len(projected_notes) == 3  # chain_nav / summary / daily 三面各登记一次

    def test_config_column_micro_diff_fail(self, tmp_path):
        """R2-T5A-R1-02 反例：配置列 0.15 vs 0.1500005（<1e-6）⇒ FAIL（不得容差抹平）。"""
        dir_a, dir_b = self._tampered_pair(tmp_path)
        _set_parquet_value(dir_b / "summary.parquet", "bt_max_weight_per_stock", 0, 0.1500005)
        _rewrite_batch_meta_config(dir_b)  # 配置键入指纹 ⇒ 同步重算保持 loader 自洽
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        config_check = result["summary_config_exact_check"]
        assert config_check["match"] is False
        assert "bt_max_weight_per_stock" in config_check["column_diffs"]
        assert any("配置列" in m and "bt_max_weight_per_stock" in m for m in result["errors"])
        # 配置列不在容差面 ⇒ 无投影掩盖；summary 绩效容差面不受影响仍 PASS 档
        assert result["summary_tolerance_check"]["within_tolerance"] is True

    def test_chain_nav_over_tolerance_fail(self, tmp_path):
        dir_a, dir_b = self._tampered_pair(tmp_path)
        _shift_parquet_column(dir_b / "chain_nav.parquet", "nav", 2e-6)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert result["nav_tolerance_check"]["within_tolerance"] is False
        assert any("chain_nav" in m and "容差比较失败" in m for m in result["errors"])

    def test_summary_over_tolerance_fail(self, tmp_path):
        """summary 绩效列超容差 ⇒ FAIL（容差面收口后不再靠逐位比较兜底）。"""
        dir_a, dir_b = self._tampered_pair(tmp_path)
        _shift_parquet_column(dir_b / "summary.parquet", "bt_sharpe", 2e-6)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert result["summary_tolerance_check"]["within_tolerance"] is False
        assert any("summary" in m and "容差比较失败" in m for m in result["errors"])

    def test_daily_over_tolerance_fail(self, tmp_path):
        dir_a, dir_b = self._tampered_pair(tmp_path)
        daily_path = dir_b / "folds" / "split00" / "daily.parquet"
        _shift_parquet_column(daily_path, "nav", 2e-6)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert result["daily_tolerance_check"]["within_tolerance"] is False
        assert any("daily" in m and "容差比较失败" in m for m in result["errors"])

    def test_exact_surfaces_still_exact(self, tmp_path):
        """容差面之外的产物仍逐位精确：trades 改一笔 ⇒ FAIL（不受容差投影影响）。"""
        dir_a, dir_b = self._tampered_pair(tmp_path)
        _shift_parquet_column(dir_b / "chain_nav.parquet", "nav", 5e-7)
        trades_path = dir_b / "folds" / "split00" / "trades.parquet"
        _set_parquet_value(trades_path, "price", 0, 99.0)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("trades" in m for m in result["errors"])


# ---------------------------------------------------------------------------
# R2-T5A-07 有限性门：NaN/Inf ⇒ FAIL（G-code 与回归边两路）
# ---------------------------------------------------------------------------


class TestFiniteGuard:
    def test_nan_chain_nav_single_side_fail(self, tmp_path):
        """单侧 chain_nav nav NaN ⇒ FAIL（非「容差内」）。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b")
        _set_parquet_value(dir_b / "chain_nav.parquet", "nav", 0, float("nan"))
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m for m in result["errors"])

    def test_nan_chain_nav_both_sides_fail(self, tmp_path):
        """双侧同位置 NaN ⇒ FAIL：NaN 不得借 NaN==NaN 精确口径判「容差内/一致」。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b")
        _set_parquet_value(dir_a / "chain_nav.parquet", "nav", 0, float("nan"))
        _set_parquet_value(dir_b / "chain_nav.parquet", "nav", 0, float("nan"))
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m for m in result["errors"])

    def test_inf_daily_nav_fail(self, tmp_path):
        """单侧 daily nav Inf ⇒ FAIL。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b")
        daily_path = dir_b / "folds" / "split00" / "daily.parquet"
        _set_parquet_value(daily_path, "nav", 0, float("inf"))
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m for m in result["errors"])

    def test_nan_summary_metric_fail(self, tmp_path):
        """单侧 summary bt_* 绩效列 NaN ⇒ FAIL。"""
        dir_a = _write_batch(tmp_path / "a", "batch_a")
        dir_b = _write_batch(tmp_path / "b", "batch_b")
        _set_parquet_value(dir_b / "summary.parquet", "bt_sharpe", 0, float("nan"))
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m and "bt_sharpe" in m for m in result["errors"])

    def test_regression_edge_nan_fail(self, tmp_path):
        """回归边同口径：冻结侧 chain_nav NaN ⇒ FAIL。"""
        frozen = _write_batch(tmp_path / "frozen", "p2b_t5_closed_syn")
        rerun = _write_batch(tmp_path / "new", "p3_rerun_syn")
        _set_parquet_value(frozen / "chain_nav.parquet", "nav", 0, float("nan"))
        result = compare_regression_edge(rerun, frozen)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m for m in result["errors"])

    def test_regression_edge_nan_both_sides_fail(self, tmp_path):
        """回归边双侧同位置 NaN ⇒ FAIL（精确比较 NaN==NaN 视等，有限性门拦截）。"""
        frozen = _write_batch(tmp_path / "frozen", "p2b_t5_closed_syn")
        rerun = _write_batch(tmp_path / "new", "p3_rerun_syn")
        _set_parquet_value(frozen / "chain_nav.parquet", "nav", 0, float("nan"))
        _set_parquet_value(rerun / "chain_nav.parquet", "nav", 0, float("nan"))
        result = compare_regression_edge(rerun, frozen)
        assert result["verdict"] == "FAIL"
        assert any("非有限值" in m for m in result["errors"])


# ---------------------------------------------------------------------------
# R2-T5A-09 + R2-T5A-R1-06 同分排名互换：分组身份配对口径（_compare_topk_fold 直射）
# ---------------------------------------------------------------------------


def _topk_frame(rows: List[Dict[str, object]], *, topk: int = 20) -> pd.DataFrame:
    """合成 topk 明细帧（正式出口列形态：trade_date/topk/rank/ts_code/score；
    行自带 topk 键时不覆盖，支持多 K 组）。"""
    enriched = [dict(row) if "topk" in row else dict(row, topk=topk) for row in rows]
    return pd.DataFrame(enriched, columns=["trade_date", "topk", "rank", "ts_code", "score"])


def _fold_result(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> Dict[str, object]:
    """直射调用（score 列名以信号/合成口径为准）。"""
    return _compare_topk_fold(frame_a, frame_b, score_col="score")


class TestTieSwapIdentityPairing:
    """按 (trade_date, topk) 分组、组内 ts_code 配对：组内同分 + rank 不同 ⇒ 互换计数。"""

    def test_true_tie_swap_counted_not_mismatch(self):
        """真实同分互换（rank 5/6 同分换股）⇒ tie_swaps 计数 2，不判 mismatch。"""
        frame_a = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 5, "ts_code": "T0005", "score": 0.5},
                {"trade_date": "20240102", "rank": 6, "ts_code": "T0006", "score": 0.5},
            ]
        )
        frame_b = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 5, "ts_code": "T0006", "score": 0.5},
                {"trade_date": "20240102", "rank": 6, "ts_code": "T0005", "score": 0.5},
            ]
        )
        result = _fold_result(frame_a, frame_b)
        assert result["status"] == "tie_swaps"
        assert result["swap_count"] == 2  # 互换涉及两个位次
        assert result["errors"] == []

    def test_reverse_rank_without_tie_is_mismatch(self):
        """R2-T5A-R1-06 反例：异分反转（A=0.9/rank1 与 B=0.8/rank2 换位，
        逐键分数跨臂相等但组内排名与分数矛盾）⇒ selection_mismatch（不判并列）。"""
        frame_a = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 1, "ts_code": "T0001", "score": 0.9},
                {"trade_date": "20240102", "rank": 2, "ts_code": "T0002", "score": 0.8},
            ]
        )
        frame_b = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 1, "ts_code": "T0002", "score": 0.8},
                {"trade_date": "20240102", "rank": 2, "ts_code": "T0001", "score": 0.9},
            ]
        )
        result = _fold_result(frame_a, frame_b)
        assert result["status"] == "selection_mismatch"
        assert any("排名与分数矛盾" in m for m in result["errors"])

    def test_rank_swap_without_tie_is_mismatch(self):
        """rank 不同但分数不等 ⇒ selection_mismatch（真实差异保持硬失败）。"""
        frame_a = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 5, "ts_code": "T0005", "score": 0.5},
                {"trade_date": "20240102", "rank": 6, "ts_code": "T0006", "score": 0.4},
            ]
        )
        frame_b = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 5, "ts_code": "T0006", "score": 0.4},
                {"trade_date": "20240102", "rank": 6, "ts_code": "T0005", "score": 0.45},
            ]
        )
        result = _fold_result(frame_a, frame_b)
        assert result["status"] == "selection_mismatch"

    def test_cross_topk_boundary_is_mismatch(self):
        """跨 top-K 边界（一臂入选一臂落选）⇒ selection_mismatch。"""
        frame_a = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 19, "ts_code": "T0019", "score": 0.21},
                {"trade_date": "20240102", "rank": 20, "ts_code": "T0020", "score": 0.20},
            ]
        )
        frame_b = _topk_frame(
            [
                {"trade_date": "20240102", "rank": 19, "ts_code": "T0019", "score": 0.21},
                {"trade_date": "20240102", "rank": 20, "ts_code": "T9999", "score": 0.20},
            ]
        )
        result = _fold_result(frame_a, frame_b)
        assert result["status"] == "selection_mismatch"
        assert "T9999" in result["errors"][0]

    def test_multi_k_same_stock_legal_duplicate(self):
        """R2-T5A-R1-06 多 K 正式出口形态：同股同日跨 topk=20/30 合法重复 ⇒
        不抛配对键重复；topk=20 组内同分互换 ⇒ 计数正确不判 mismatch。"""
        rows_a = [
            {"trade_date": "20240102", "topk": 20, "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "topk": 20, "rank": 2, "ts_code": "T0002", "score": 0.5},
            {"trade_date": "20240102", "topk": 20, "rank": 3, "ts_code": "T0003", "score": 0.5},
            {"trade_date": "20240102", "topk": 30, "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "topk": 30, "rank": 2, "ts_code": "T0002", "score": 0.5},
            {"trade_date": "20240102", "topk": 30, "rank": 3, "ts_code": "T0003", "score": 0.5},
            {"trade_date": "20240102", "topk": 30, "rank": 4, "ts_code": "T0004", "score": 0.4},
        ]
        rows_b = [dict(row) for row in rows_a]
        # topk=20 组内 T0002/T0003 同分换股（rank 对调）；topk=30 组不动
        rows_b[1]["rank"], rows_b[2]["rank"] = 3, 2
        result = _fold_result(_topk_frame(rows_a), _topk_frame(rows_b))
        assert result["status"] == "tie_swaps"
        assert result["swap_count"] == 2  # 仅 topk=20 组内两个位次
        assert result["errors"] == []

    def test_cross_k_boundary_is_mismatch(self):
        """R2-T5A-R1-06 跨 K 边界：一臂 topk=20 入选股在另一臂跌出 top20
        （仅现于 topk=30 组）⇒ selection_mismatch。"""
        rows_a = [
            {"trade_date": "20240102", "topk": 20, "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "topk": 30, "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "topk": 30, "rank": 2, "ts_code": "T0002", "score": 0.8},
        ]
        rows_b = [
            {"trade_date": "20240102", "topk": 20, "rank": 1, "ts_code": "T0002", "score": 0.8},
            {"trade_date": "20240102", "topk": 30, "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "topk": 30, "rank": 2, "ts_code": "T0002", "score": 0.8},
        ]
        result = _fold_result(_topk_frame(rows_a), _topk_frame(rows_b))
        assert result["status"] == "selection_mismatch"
        assert any("多重集不一致" in m for m in result["errors"])

    def test_row_order_only_difference_is_match(self):
        """仅物理行序不同（逐键整行全等含 rank）⇒ match（行序差不构成排名互换）。"""
        rows = [
            {"trade_date": "20240102", "rank": 1, "ts_code": "T0001", "score": 0.9},
            {"trade_date": "20240102", "rank": 2, "ts_code": "T0002", "score": 0.8},
        ]
        frame_a = _topk_frame(rows)
        frame_b = _topk_frame([rows[1], rows[0]])
        result = _fold_result(frame_a, frame_b)
        assert result["status"] == "match"
        assert result["swap_count"] == 0

    def test_register_ratio_kept(self):
        """>入选数 5% 登记判据保持：2/20 ⇒ 登记；2/100 ⇒ 不登记。"""
        base = [
            {
                "trade_date": "20240102",
                "rank": rank,
                "ts_code": f"T{rank:04d}",
                "score": 1.0 - rank * 0.001,
            }
            for rank in range(1, 101)
        ]
        tied = [dict(row) for row in base]
        tied[4]["score"] = tied[5]["score"]  # rank5/rank6 并列同分（两臂同分才成立）
        frame_a = _topk_frame(tied)
        swapped = [dict(row) for row in tied]
        swapped[4]["rank"], swapped[5]["rank"] = swapped[5]["rank"], swapped[4]["rank"]
        frame_b = _topk_frame(swapped)
        result_100 = _fold_result(frame_a, frame_b)
        assert result_100["status"] == "tie_swaps" and result_100["swap_count"] == 2
        assert result_100["registered"] is False  # 2 ≤ 100 × 5%
        result_20 = _fold_result(frame_a.head(20), frame_b.head(20))
        assert result_20["status"] == "tie_swaps" and result_20["swap_count"] == 2
        assert result_20["registered"] is True  # 2 > 20 × 5%

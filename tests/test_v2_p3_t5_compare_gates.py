# -*- coding: utf-8 -*-
"""v2 P3 T5 三道比较边判定工具测试（合成数据，不读取真实生产数据/产物）。

覆盖（§19.4 / §19.8 测试清单映射）：
- G-data 判定表边界（C02）：恰好 ΔCAGR = −8pp 带内 PASS；−8pp−ε 超带
  （有归因闭合 ⇒ PASS 登记 / 无 ⇒ FAIL）；ΔMaxDD 劣化 0/有归因/无归因三分支；
  ΔCAGR 改善 >+8pp 登记不精读；缺指标输入契约违反；
- G-code ① 指纹主门：逐折一致 + 汇总一致 ⇒ PASS；任一折不一致 ⇒ FAIL 并指出折号；
  缺折 ⇒ FAIL；汇总不一致 / 汇总与逐折不自洽 ⇒ FAIL；指纹值一律为合法 64 位
  sha256 hex（R2-T5A-03-③ 收紧后口径，非法值反例见 test_v2_p3_t5_r1_gate_fixes）；
- 配对投影归一化（§7.2/§13.3.3）：归一化键不同但内容同 ⇒ PASS；非归一化键差异 ⇒ FAIL；
- 链式净值 1e-6 容差边界（逐位差 ≤1e-6 容差内 / 超容差 / 折集错位）；
- 回归边薄包装：合成两 runs 批一致 ⇒ PASS / 改一行 ⇒ FAIL；
- G-code ③ 辅证（评估路径）：并列互换计数正确且 >入选数 5% 时登记字段出现；≤5% 不登记；
  选股多重集差异 ⇒ FAIL；并列互换夹具为**真实排名位次互换**（配对键 rank 值对调，
  R2-T5A-09 身份配对口径）；**双臂全折同缺 ⇒ FAIL**（R2-T5A-R1-01 辅证缺失）；
- 四门组装：build_gcode_verdict 合成臂目录端到端 PASS（14 折 runs + 14 折评估路径
  CSV + 14 折信号产物：signal_topk.parquet + 指纹增键 + predictions 日集锚，
  R2-T5A-R1-01 收紧后口径）。
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
    PAIR_PROJECTION_KEYS,
    build_gcode_verdict,
    compare_chain_nav_tolerance,
    compare_excluded_days,
    compare_gcode_fingerprints,
    compare_gcode_runs,
    compare_regression_edge,
    compare_topk_details,
    judge_gdata_table,
)
from src.lazybull.v2.hosts.backtest.runs_writer import write_runs_batch  # noqa: E402

_B0P_CHAIN: Dict[str, object] = {"cagr": 0.18, "max_drawdown": -0.12, "sharpe": 1.5}


def _sha(text: str) -> str:
    """合成合法 64 位 sha256 hex 指纹值（R2-T5A-03-③：非法值先验拒绝）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _arm3_chain(**overrides: float) -> Dict[str, object]:
    chain: Dict[str, object] = {"cagr": 0.18, "max_drawdown": -0.12, "sharpe": 1.5}
    chain.update(overrides)
    return chain


# ---------------------------------------------------------------------------
# G-data 判定表（Q9-A/Q11-A/Q12-A；C02 边界）
# ---------------------------------------------------------------------------


class TestGdataTable:
    """判定表边界语义：等号带内、严格大于才超带、ΔMaxDD 零容忍 + 归因出口。"""

    def test_exact_minus_8pp_is_within_band(self):
        """C02 边界：恰好 ΔCAGR = −8pp ⇒ 带内 PASS（等号带内）。"""
        table = judge_gdata_table(_arm3_chain(cagr=0.10), dict(_B0P_CHAIN))
        assert table["verdict"] == "PASS"
        assert table["cagr_judgment"]["branch"] == "within_band"
        assert table["cagr_judgment"]["overshoot"] is False
        assert table["deltas"]["cagr"] == pytest.approx(-0.08)

    def test_overshoot_epsilon_without_attribution_fails(self):
        """C02 边界：−8pp−ε ⇒ 超带；无归因闭合 ⇒ FAIL。"""
        table = judge_gdata_table(_arm3_chain(cagr=0.10 - 1e-9), dict(_B0P_CHAIN))
        assert table["verdict"] == "FAIL"
        assert table["cagr_judgment"]["branch"] == "overshoot_unattributed"
        assert table["cagr_judgment"]["passed"] is False

    def test_overshoot_epsilon_with_attribution_passes_registered(self):
        """−8pp−ε 超带 + 归因闭合 ⇒ PASS 并登记（放行并登记表达式）。"""
        table = judge_gdata_table(
            _arm3_chain(cagr=0.10 - 1e-9), dict(_B0P_CHAIN), cagr_attribution_closed=True
        )
        assert table["verdict"] == "PASS"
        assert table["cagr_judgment"]["branch"] == "overshoot_attributed"
        assert any("ΔCAGR" in item and "放行并登记" in item for item in table["registrations"])

    def test_improvement_over_8pp_registered_not_scrutinized(self):
        """ΔCAGR 改善 >+8pp ⇒ PASS 登记不精读（MVP 禁宣称突破）。"""
        table = judge_gdata_table(_arm3_chain(cagr=0.18 + 0.09), dict(_B0P_CHAIN))
        assert table["verdict"] == "PASS"
        assert table["cagr_judgment"]["branch"] == "improvement"
        assert any("改善登记不精读" in item for item in table["registrations"])

    def test_maxdd_no_deterioration_passes(self):
        """ΔMaxDD = 0（不劣化）⇒ PASS，无需归因标记。"""
        table = judge_gdata_table(_arm3_chain(), dict(_B0P_CHAIN))
        assert table["verdict"] == "PASS"
        assert table["maxdd_judgment"]["branch"] == "no_deterioration"
        assert table["maxdd_judgment"]["deterioration"] is False

    def test_maxdd_deterioration_with_attribution_passes_registered(self):
        """ΔMaxDD 更负（劣化）+ 归因闭合 ⇒ PASS 并登记（零容忍 + 归因出口）。"""
        table = judge_gdata_table(
            _arm3_chain(max_drawdown=-0.121), dict(_B0P_CHAIN), maxdd_attribution_closed=True
        )
        assert table["verdict"] == "PASS"
        assert table["maxdd_judgment"]["branch"] == "deterioration_attributed"
        assert any("ΔMaxDD" in item and "放行并登记" in item for item in table["registrations"])

    def test_maxdd_deterioration_without_attribution_fails(self):
        """ΔMaxDD 劣化且无归因闭合 ⇒ FAIL（零容忍）。"""
        table = judge_gdata_table(_arm3_chain(max_drawdown=-0.121), dict(_B0P_CHAIN))
        assert table["verdict"] == "FAIL"
        assert table["maxdd_judgment"]["branch"] == "deterioration_unattributed"

    def test_maxdd_improvement_needs_no_attribution(self):
        """ΔMaxDD 改善（更浅）⇒ PASS，不要求归因标记。"""
        table = judge_gdata_table(_arm3_chain(max_drawdown=-0.11), dict(_B0P_CHAIN))
        assert table["verdict"] == "PASS"
        assert table["maxdd_judgment"]["branch"] == "no_deterioration"

    def test_missing_metric_raises(self):
        """判定表输入契约：缺指标 / 非有限值 ⇒ 报错（禁止静默判 PASS）。"""
        with pytest.raises(ValueError, match="cagr"):
            judge_gdata_table({"max_drawdown": -0.1, "sharpe": 1.0}, dict(_B0P_CHAIN))
        with pytest.raises(ValueError, match="非有限值"):
            judge_gdata_table(_arm3_chain(cagr=float("nan")), dict(_B0P_CHAIN))


# ---------------------------------------------------------------------------
# G-code ① 指纹主门
# ---------------------------------------------------------------------------


def _make_arm_fingerprints(
    root: Path,
    arm: str,
    *,
    fold_overrides: Optional[Dict[str, str]] = None,
    batch_fingerprint: Optional[str] = None,
    skip_folds: tuple = (),
) -> Path:
    """合成一臂的指纹产物：folds/splitNN/fingerprint.json × 14 + batch_fingerprint.json。

    指纹值一律为合法 64 位 sha256 hex（R2-T5A-03-③ 收紧后口径：非法值先验拒绝）。
    """
    arm_dir = root / arm
    overrides = fold_overrides or {}
    folds_map: Dict[str, str] = {}
    for split in range(14):
        name = f"split{split:02d}"
        fingerprint = overrides.get(name, _sha(f"fp-{name}"))
        folds_map[name] = fingerprint
        if name in skip_folds:
            continue
        fold_dir = arm_dir / "folds" / name
        fold_dir.mkdir(parents=True)
        doc = {
            "model_version": 24008,
            "n_days": 5,
            "n_rows": 100,
            "score_days": ["20240102"],
            "fingerprint": fingerprint,
            "per_seed_diagnostic": [
                {"seed": 42, "n_rows": 100, "fingerprint": _sha(f"seed-{name}")}
            ],
        }
        (fold_dir / "fingerprint.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )
    (arm_dir / "batch_fingerprint.json").write_text(
        json.dumps(
            {"folds": folds_map, "batch_fingerprint": batch_fingerprint or _sha("batch-fp")}
        ),
        encoding="utf-8",
    )
    return arm_dir


class TestExcludedDaysGate:
    """R1-P2-1：闭市排除日必须由臂间子门机器校验，不能只靠覆盖门。"""

    def _set_days(self, arm: Path, days: list[str]) -> None:
        path = arm / "folds" / "split06" / "fingerprint.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["excluded_days"] = days
        doc["excluded_day_reasons"] = {day: "stock_connect_closed" for day in days}
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    def test_same_days_pass(self, tmp_path):
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        days = ["20211224", "20220414", "20220506"]
        self._set_days(arm_a, days)
        self._set_days(arm_b, days)
        result = compare_excluded_days(arm_a, arm_b)
        assert result["verdict"] == "PASS"
        fold = next(row for row in result["folds"] if row["fold"] == "split06")
        assert fold["excluded_days_a"] == days

    def test_mismatch_fails_and_names_fold(self, tmp_path):
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        self._set_days(arm_a, ["20211224", "20220414", "20220506"])
        self._set_days(arm_b, ["20211224"])
        result = compare_excluded_days(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split06" in message and "集合不一致" in message for message in result["errors"])

    def test_missing_reason_mismatch_fails(self, tmp_path):
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        self._set_days(arm_a, ["20220506"])
        path = arm_b / "folds" / "split06" / "fingerprint.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["excluded_days"] = ["20220506"]
        doc["excluded_day_reasons"] = {}
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        result = compare_excluded_days(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("原因映射不一致" in message for message in result["errors"])


class TestGcodeFingerprints:
    """① 主门：逐折指纹 14/14 逐位一致 + 全批汇总逐位一致。"""

    def test_all_match_pass(self, tmp_path):
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]
        assert len(result["fold_results"]) == 14
        assert result["batch_fingerprint"]["match"] is True

    def test_single_fold_mismatch_fail_names_fold(self, tmp_path):
        """任一折不一致 ⇒ FAIL 并指出折号。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(
            tmp_path, "arm2", fold_overrides={"split07": _sha("fp-DIFFERENT")}
        )
        # 汇总映射同步改写（隔离 fold 级不一致这一变量）
        batch_doc = json.loads((arm_b / "batch_fingerprint.json").read_text(encoding="utf-8"))
        batch_doc["folds"]["split07"] = _sha("fp-DIFFERENT")
        (arm_b / "batch_fingerprint.json").write_text(json.dumps(batch_doc), encoding="utf-8")
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split07" in message for message in result["errors"])
        mismatched = [f for f in result["fold_results"] if not f["match"]]
        assert [f["fold"] for f in mismatched] == ["split07"]

    def test_missing_fold_fail(self, tmp_path):
        """缺折 ⇒ FAIL 并指出折号。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2", skip_folds=("split03",))
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("split03" in message for message in result["errors"])

    def test_batch_fingerprint_mismatch_fail(self, tmp_path):
        """逐折一致但全批汇总不一致 ⇒ FAIL。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2", batch_fingerprint=_sha("batch-OTHER"))
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("全批汇总指纹不一致" in message for message in result["errors"])

    def test_batch_fold_map_inconsistent_fail(self, tmp_path):
        """汇总 folds 映射与逐折文件不自洽 ⇒ FAIL（输入身份校验）。"""
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        batch_doc = json.loads((arm_b / "batch_fingerprint.json").read_text(encoding="utf-8"))
        batch_doc["folds"]["split05"] = _sha("fp-TAMPERED")
        (arm_b / "batch_fingerprint.json").write_text(json.dumps(batch_doc), encoding="utf-8")
        result = compare_gcode_fingerprints(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("不自洽" in message and "split05" in message for message in result["errors"])


# ---------------------------------------------------------------------------
# 合成 runs 批次（runs_writer 正式出口写出，读入桥自校验过硬契约）
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


def _synth_fold(split: int, model_version: int, *, price: float = 10.0) -> Dict[str, object]:
    """最小合法合成折：1 笔买入 + 两日净值（bt_total_return 与 nav 比自洽）。"""
    day0, day1 = ("2024-01-02", "2024-01-03") if split == 0 else ("2024-07-01", "2024-07-02")
    trades = pd.DataFrame(
        [
            {
                "date": day0,
                "signal_date": "2023-12-29",
                "stock": "000001.SZ",
                "action": "buy",
                "price": price,
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
    model_version: int,
    data_state_id: str,
    *,
    price: float = 10.0,
    split_count: int = 14,
) -> Path:
    """写出合成 runs 批次（默认 14 折：G-code ② 折集交叉核验收紧后口径，R2-T5A-03-①）。"""
    return write_runs_batch(
        [_synth_fold(split, model_version, price=price) for split in range(split_count)],
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
    """合成一折 predictions.parquet 帧（跨路契约形态：trade_date/ts_code/score，
    键 (trade_date,ts_code) 升序；每日 25 股 ⇒ top-20 截断真实发生）。"""
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


def _write_arm_signal_artifacts(
    arm_dir: Path,
    *,
    skip_folds: tuple = (),
    frame_overrides: Optional[Dict[str, pd.DataFrame]] = None,
    empty_days: Optional[List[str]] = None,
    write_fingerprint_keys: bool = True,
) -> None:
    """写一臂 14 折信号辅证产物（R2-T5A-R1-01 口径）：

    predictions.parquet（主证据日集锚）+ signal_topk.parquet（真实信号通路产物
    形态）+ 回写 fingerprint.json 增键五件套（path/sha256/rows/days/empty_days，
    sha256 自重算文件兑现登记一致）；score_days 与覆盖日集保持一致。
    """
    overrides = frame_overrides or {}
    for split in range(14):
        name = f"split{split:02d}"
        if name in skip_folds:
            continue
        fold_dir = arm_dir / "folds" / name
        fold_dir.mkdir(parents=True, exist_ok=True)
        days = _arm_fold_days(split)
        _synth_predictions(split).to_parquet(fold_dir / "predictions.parquet", index=False)
        frame = overrides.get(name)
        if frame is None:
            frame = _synth_signal_topk(split)
        path = fold_dir / "signal_topk.parquet"
        frame.to_parquet(path, index=False)
        if not write_fingerprint_keys:
            continue
        fp_path = fold_dir / "fingerprint.json"
        doc = json.loads(fp_path.read_text(encoding="utf-8")) if fp_path.exists() else {}
        doc.update(
            {
                "score_days": days,
                "signal_topk_path": f"folds/{name}/signal_topk.parquet",
                "signal_topk_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "signal_topk_rows": int(len(frame)),
                "signal_topk_days": [d for d in days if d not in (empty_days or [])],
                "signal_topk_empty_days": list(empty_days or []),
            }
        )
        fp_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 配对投影归一化 + G-code ② runs 比较 + 链式净值容差
# ---------------------------------------------------------------------------


class TestPairProjectionRuns:
    """配对投影（§7.2）：归一化键不同但内容同 ⇒ PASS；非归一化键差异 ⇒ FAIL。"""

    def test_projection_keys_differ_but_content_same_pass(self, tmp_path):
        """model_version / data_state_id / batch_id（wf_run_id）不同、内容同 ⇒ PASS。"""
        dir_a = _write_batch(tmp_path / "a", "batch_arm1", 24008, "ds-aaa")
        dir_b = _write_batch(tmp_path / "b", "batch_arm2", 99999, "ds-bbb")
        assert PAIR_PROJECTION_KEYS == (
            "model_version",
            "registered_at",
            "data_state_id",
            "batch_id",
            "wf_run_id",
        )
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "PASS", result["errors"]
        assert result["nav_tolerance_check"]["max_abs_diff"] == 0.0

    def test_non_projection_difference_fail(self, tmp_path):
        """非归一化键差异（成交价）⇒ FAIL 并指明产物。"""
        dir_a = _write_batch(tmp_path / "a", "batch_arm1", 24008, "ds-aaa")
        dir_b = _write_batch(tmp_path / "b", "batch_arm2", 24008, "ds-aaa", price=10.5)
        result = compare_gcode_runs(dir_a, dir_b)
        assert result["verdict"] == "FAIL"
        assert any("trades" in message for message in result["errors"])


class TestChainNavTolerance:
    """链式净值 1e-6 容差边界（§9.1 口径；G-code ② 的 chain_nav 判据）。"""

    def _chain(self, nav_last: float, split: int = 0) -> pd.DataFrame:
        return pd.DataFrame({"date": [0, 1], "nav": [1.0, nav_last], "split_index": [split, split]})

    def test_identical_within_tolerance(self):
        result = compare_chain_nav_tolerance(self._chain(1.01), self._chain(1.01))
        assert result["within_tolerance"] is True and result["aligned"] is True
        assert result["max_abs_diff"] == 0.0

    def test_diff_below_tolerance_within(self):
        result = compare_chain_nav_tolerance(self._chain(1.01), self._chain(1.01 + 5e-7))
        assert result["within_tolerance"] is True
        assert result["max_abs_diff"] == pytest.approx(5e-7)

    def test_diff_above_tolerance_fails(self):
        result = compare_chain_nav_tolerance(self._chain(1.01), self._chain(1.01 + 2e-6))
        assert result["within_tolerance"] is False
        assert any("超容差" in message for message in result["errors"])

    def test_split_set_misaligned_fails(self):
        result = compare_chain_nav_tolerance(self._chain(1.01, split=0), self._chain(1.01, split=1))
        assert result["aligned"] is False and result["within_tolerance"] is False


# ---------------------------------------------------------------------------
# 回归边薄包装
# ---------------------------------------------------------------------------


class TestRegressionEdge:
    """回归边（R2-R1-04）：两 runs 批成对精确比较薄包装 + 配对投影归一化。"""

    def test_identical_batches_pass(self, tmp_path):
        frozen = _write_batch(tmp_path / "frozen", "p2b_t5_closed_syn", 24008, "ds-syn")
        rerun = _write_batch(tmp_path / "new", "p3_rerun_20261023", 24008, "ds-syn")
        result = compare_regression_edge(rerun, frozen)
        assert result["verdict"] == "PASS", result["errors"]
        assert result["projection_keys"] == list(PAIR_PROJECTION_KEYS)

    def test_modified_row_fail(self, tmp_path):
        frozen = _write_batch(tmp_path / "frozen", "p2b_t5_closed_syn", 24008, "ds-syn")
        rerun = _write_batch(tmp_path / "new", "p3_rerun_20261023", 24008, "ds-syn")
        trades_path = rerun / "folds" / "split00" / "trades.parquet"
        trades = pd.read_parquet(trades_path)
        trades.loc[0, "price"] = 99.0
        trades.to_parquet(trades_path, index=False)
        result = compare_regression_edge(rerun, frozen)
        assert result["verdict"] == "FAIL"
        assert any("trades" in message for message in result["errors"])


# ---------------------------------------------------------------------------
# G-code ③ 辅证：topk 明细 + 并列互换计数
# ---------------------------------------------------------------------------


def _topk_rows(count: int, wf_run_id: str, model_version: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank in range(1, count + 1):
        rows.append(
            {
                "wf_run_id": wf_run_id,
                "split_index": 0,
                "test_start": "20240102",
                "test_end": "20240628",
                "model_version": model_version,
                "trade_date": "20240102",
                "topk": 20,
                "rank": rank,
                "ts_code": f"T{rank:04d}",
                "pred_score": 1.0 - rank * 0.001,
                "true_return": 0.01,
                "score_column": "pred_score",
                "ml_score": None,
                "risk_score": None,
                "final_score": None,
            }
        )
    return rows


def _write_topk_csv(
    arm_dir: Path, fold: str, rows: List[Dict[str, object]], wf_run_id: str
) -> Path:
    raw_dir = arm_dir / "root" / "walk_forward" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"walk_forward_topk_details_{wf_run_id}_{fold}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    # 每折辅证都必需：聚焦某折的用例同样提供其余折的完整合法输入。
    for split in range(14):
        other = raw_dir / f"walk_forward_topk_details_{wf_run_id}_split{split:02d}.csv"
        if not other.exists():
            pd.DataFrame(rows).to_csv(other, index=False, encoding="utf-8-sig")
    return path


class TestTopkAuxiliary:
    """③ 辅证：逐行对比 + 并列互换计数（>入选数 5% 登记，信息性不硬失败）。"""

    def test_identical_csv_pass(self, tmp_path):
        """逐行一致（归一化键不同）⇒ PASS，互换计数 0。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_topk_csv(arm_a, "split00", _topk_rows(20, "wf-a", 24008), "wf-a")
        _write_topk_csv(arm_b, "split00", _topk_rows(20, "wf-b", 99999), "wf-b")
        result = compare_topk_details(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]
        fold = result["folds"][0]
        assert fold["status"] == "match" and fold["swap_count"] == 0

    def test_tie_swaps_counted_and_registered_over_5pct(self, tmp_path):
        """同分排名互换：计数正确且 >入选数 5% 时登记字段出现（信息性，不硬失败）。

        夹具 = 真实排名位次互换（R2-T5A-09 身份配对口径）：配对键 (trade_date,
        ts_code) 的 rank 值在两臂间对调、分数相等（并列成立）；仅交换物理行序而
        rank 不动不构成本口径的互换（逐键整行全等 ⇒ match）。
        """
        rows_a = _topk_rows(20, "wf-a", 24008)
        rows_a[4]["pred_score"] = rows_a[5]["pred_score"]  # rank5/rank6 并列同分
        rows_b = [dict(row) for row in rows_a]
        rows_b[4]["rank"], rows_b[5]["rank"] = rows_a[5]["rank"], rows_a[4]["rank"]
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_topk_csv(arm_a, "split00", rows_a, "wf-a")
        _write_topk_csv(arm_b, "split00", rows_b, "wf-b")
        result = compare_topk_details(arm_a, arm_b)
        assert result["verdict"] == "PASS", result["errors"]  # 信息性，不作硬失败
        fold = result["folds"][0]
        assert fold["status"] == "tie_swaps"
        assert fold["swap_count"] == 2  # 互换涉及两个位次
        assert fold["registered"] is True  # 2 > 20 × 5%
        assert any("split00" in item for item in result["tie_swap_registrations"])

    def test_tie_swaps_below_threshold_not_registered(self, tmp_path):
        """互换计数 ≤入选数 5% ⇒ 不登记（仍 PASS）。"""
        rows_a = _topk_rows(100, "wf-a", 24008)
        rows_a[4]["pred_score"] = rows_a[5]["pred_score"]
        rows_b = [dict(row) for row in rows_a]
        rows_b[4]["rank"], rows_b[5]["rank"] = rows_a[5]["rank"], rows_a[4]["rank"]
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_topk_csv(arm_a, "split00", rows_a, "wf-a")
        _write_topk_csv(arm_b, "split00", rows_b, "wf-b")
        result = compare_topk_details(arm_a, arm_b)
        assert result["verdict"] == "PASS"
        fold = result["folds"][0]
        assert fold["status"] == "tie_swaps" and fold["swap_count"] == 2
        assert fold["registered"] is False  # 2 ≤ 100 × 5%
        assert result["tie_swap_registrations"] == []

    def test_selection_mismatch_fail(self, tmp_path):
        """选股多重集差异（真实差异）⇒ FAIL。"""
        rows_a = _topk_rows(20, "wf-a", 24008)
        rows_b = _topk_rows(20, "wf-b", 24008)
        rows_b[0]["ts_code"] = "T9999"
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_topk_csv(arm_a, "split00", rows_a, "wf-a")
        _write_topk_csv(arm_b, "split00", rows_b, "wf-b")
        result = compare_topk_details(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("多重集不一致" in message for message in result["errors"])

    def test_single_side_missing_fail(self, tmp_path):
        """单侧缺 topk 明细 ⇒ FAIL；双臂全折同缺 ⇒ FAIL（R2-T5A-R1-01 辅证缺失）。"""
        arm_a, arm_b = tmp_path / "arm1", tmp_path / "arm2"
        _write_topk_csv(arm_a, "split00", _topk_rows(20, "wf-a", 24008), "wf-a")
        result = compare_topk_details(arm_a, arm_b)
        assert result["verdict"] == "FAIL"
        assert any("单侧缺失" in message and "split00" in message for message in result["errors"])
        both_missing = compare_topk_details(tmp_path / "arm3", tmp_path / "arm4")
        assert both_missing["verdict"] == "FAIL"
        assert any("辅证缺失 = FAIL" in message for message in both_missing["errors"])


# ---------------------------------------------------------------------------
# 四门组装端到端（合成臂目录：14 折指纹 + 14 折 runs + 14 折评估路径 CSV +
# 14 折信号产物（signal_topk.parquet + 指纹增键 + predictions 日集锚）；
# R2-T5A-R1-01 收紧后口径）
# ---------------------------------------------------------------------------


class TestGcodeVerdictAssembly:
    def test_build_gcode_verdict_pass(self, tmp_path):
        arm_a = _make_arm_fingerprints(tmp_path, "arm1")
        arm_b = _make_arm_fingerprints(tmp_path, "arm2")
        _write_batch(arm_a / "runs", "batch_arm1", 24008, "ds-aaa")
        _write_batch(arm_b / "runs", "batch_arm2", 99999, "ds-bbb")
        _write_arm_signal_artifacts(arm_a)
        _write_arm_signal_artifacts(arm_b)
        for split in range(14):
            fold = f"split{split:02d}"
            _write_topk_csv(arm_a, fold, _topk_rows(20, "wf-a", 24008), "wf-a")
            _write_topk_csv(arm_b, fold, _topk_rows(20, "wf-b", 99999), "wf-b")
        verdict = build_gcode_verdict(arm_a, arm_b)
        assert verdict["gate"] == "gcode"
        assert verdict["verdict"] == "PASS"
        assert set(verdict["sub_gates"]) == {
            "fingerprint",
            "runs",
            "excluded_days",
            "topk_auxiliary",
            "signal_topk_auxiliary",
        }
        assert verdict["projection_keys"] == list(PAIR_PROJECTION_KEYS)
        # 落盘 JSON 可序列化
        text = json.dumps(verdict, ensure_ascii=False, default=str)
        assert json.loads(text)["verdict"] == "PASS"

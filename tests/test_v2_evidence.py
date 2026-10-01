# -*- coding: utf-8 -*-
"""v2 证据机器（P5a-1）单元测试：合成数据，不依赖真实批次。

覆盖：配对制度重排的配对性（两臂同排列集）、功效标定的平移构造、
判据自洽性（换种子恒等自检 + 平移可检出）、runs 转换器字段映射与丢弃列清单。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.evidence.criterion_calibration import criterion_self_consistency
from src.lazybull.v2.evidence.power_calibration import (
    power_calibration_curve,
    shift_fold_returns_annual_pp,
)
from src.lazybull.v2.evidence.regime_resample import paired_regime_bootstrap
from src.lazybull.v2.evidence.runs_convert import convert_wf_batch


def _make_run(label: str, seed: int, n_splits: int = 4, days_per_split: int = 20):
    """构造一个合成 walk-forward 产物（RunArtifacts）。"""
    from scripts.compare.fold_subset import RunArtifacts

    rng = np.random.default_rng(seed)
    rows = []
    for split in range(n_splits):
        # 逐日收益 ~ N(0.0008, 0.015)（微正漂移）
        rets = rng.normal(0.0008, 0.015, days_per_split)
        nav = np.cumprod(1.0 + rets)
        dates = pd.date_range("2024-01-01", periods=days_per_split).strftime("%Y%m%d") + f""
        for i, d in enumerate(dates):
            rows.append({"date": d, "nav": nav[i], "split_index": split})
    chain = pd.DataFrame(rows)
    windows = pd.DataFrame(
        {
            "split_index": list(range(n_splits)),
            "test_start": ["20240101"] * n_splits,
            "test_end": ["20240201"] * n_splits,
        }
    )
    return RunArtifacts(
        label=label,
        directory=Path("."),
        chain=chain,
        windows=windows,
    )


# ========== 配对制度重排 ==========


class TestPairedRegimeBootstrap:
    def test_identical_arm_delta_zero(self):
        """同一臂对自身配对 ⇒ Δ 恒为 0（配对性自检）"""
        a = _make_run("A", seed=1)
        res = paired_regime_bootstrap(a, a, splits=[0, 1, 2, 3], n_boot=50, seed=7)
        for key in ("cagr", "max_drawdown", "sharpe"):
            samples = np.asarray(res.delta_samples[key])
            assert np.allclose(samples, 0.0), f"{key} 同臂配对 Δ 应恒为 0"

    def test_shifted_arm_delta_positive(self):
        """+5pp 平移臂 vs 基线 ⇒ ΔCAGR 分布应显著为正"""
        a = _make_run("A", seed=1)
        # 平移臂：逐日收益加 5pp 年化对应日度偏移
        daily_delta = (1.05) ** (1 / 252) - 1.0
        b_chain = a.chain.copy()
        b_chain["nav"] = b_chain.groupby("split_index")["nav"].transform(
            lambda s: np.cumprod((1 + s.pct_change().fillna(s.iloc[0] - 1) + daily_delta))
        )
        from scripts.compare.fold_subset import RunArtifacts

        b = RunArtifacts(label="B", directory=Path("."), chain=b_chain, windows=a.windows)
        res = paired_regime_bootstrap(a, b, splits=[0, 1, 2, 3], n_boot=200, seed=7)
        s = res.summary("cagr")
        assert s["median"] > 0, "平移臂 ΔCAGR 中位应为正"


# ========== 功效标定 ==========


class TestPowerCalibration:
    def test_shift_returns_scale(self):
        """+2pp 年化平移 ⇒ 链式 CAGR 抬升 ≈ 0.02"""
        from scripts.compare.fold_subset import chain_metrics_from_fold_returns

        a = _make_run("A", seed=2)
        base = a.chain.groupby("split_index")["nav"].apply(
            lambda s: (s.iloc[1:].to_numpy() / s.iloc[:-1].to_numpy() - 1.0)
        ).to_dict()
        base = {int(k): np.asarray(v) for k, v in base.items()}
        shifted = shift_fold_returns_annual_pp(base, 2.0)
        m_base = chain_metrics_from_fold_returns(list(base.values()))
        m_shift = chain_metrics_from_fold_returns(list(shifted.values()))
        # 复利平移 + 折间几何拼接在小样本上略偏名义值（2pp 年化）⇒ 容差放宽到 6‰
        #（判定目标是「抬升方向正确且量级吻合」，不是精确复现名义值）
        assert m_shift["cagr"] - m_base["cagr"] == pytest.approx(0.02, abs=0.006)

    def test_power_curve_real_noise_low_detection(self):
        """真实噪声口径（A1 修复后）：block bootstrap 代理下小位移检出率低（不再系统性 100%）。

        A1 核心：等值平移构造 Δ 波动为 0 ⇒ 检出率虚高；改噪声驱动后，+1pp 在真实/代理
        噪声宽度下检出率应显著低于 100%（尺子的真实刻度）。
        """
        a = _make_run("A", seed=3, n_splits=6, days_per_split=40)
        points = power_calibration_curve(
            a, splits=[0, 1, 2, 3, 4, 5], shift_grid_pp=(1.0, 5.0), n_boot=200, seed=11
        )
        assert len(points) == 2
        assert points[0].noise_source == "block_bootstrap_proxy"  # 无换种子批 ⇒ 代理口径
        # 代理噪声下 +1pp 检出率应 < 100%（修复前等值平移恒为 100%）
        assert points[0].detection_prob_cagr < 1.0
        # 双指标字段存在（A1：覆盖北极星 ΔMaxDD）
        assert hasattr(points[0], "detection_prob_maxdd")


# ========== 判据自洽性 ==========


class TestCriterionSelfConsistency:
    def test_identity_self_check_marked_not_counted(self):
        """arm_a_alt_seed=None（恒等自检）⇒ 不作数：criterion_passes 恒 False 并标注。"""
        a = _make_run("A", seed=4)
        res = criterion_self_consistency(
            a, None, splits=[0, 1, 2, 3], shift_pp=2.0, n_boot=200, seed=5
        )
        assert res.seed_swap_indistinguishable is True  # 恒等对照 Δ 恒 0
        assert res.seed_swap_is_real is False  # 非真实换种子批
        assert res.criterion_passes is False  # 恒等自检不作数（A2）
        assert "恒等自检" in res.note

    def test_no_shift_may_fail_detection(self):
        """shift_pp=0（无改进）⇒ 可检出性应为 False（判据不把零改进当改进）"""
        a = _make_run("A", seed=6)
        res = criterion_self_consistency(
            a, None, splits=[0, 1, 2, 3], shift_pp=0.0, n_boot=200, seed=8
        )
        assert res.shifted_detectable is False
        assert res.criterion_passes is False

    def test_real_seed_swap_populates_result(self):
        """真实换种子批（独立 seed 合成）⇒ seed_swap_is_real=True，结果字段齐全。"""
        a = _make_run("A", seed=4)
        a_alt = _make_run("A'", seed=99)  # 不同 seed 的独立噪声实现（合成换种子对照）
        res = criterion_self_consistency(
            a, a_alt, splits=[0, 1, 2, 3], shift_pp=2.0, n_boot=200, seed=5
        )
        assert res.seed_swap_is_real is True
        # 真实噪声下 criterion_passes 由分布决定（不预设方向），但字段必须齐全
        assert isinstance(res.seed_swap_indistinguishable, bool)
        assert isinstance(res.shifted_detectable, bool)


# ========== runs 转换器 ==========


def _make_legacy_batch(root: Path) -> Path:
    """造一个最小旧 WF 批次（chain_nav + summary + trades + attribution + 快照 + topk）。"""
    raw = root / "raw"
    raw.mkdir(parents=True)
    # chain_nav
    pd.DataFrame(
        {"date": ["20240102", "20240103"], "nav": [1.0, 1.01], "split_index": [0, 0]}
    ).to_csv(raw / "chain_nav_wf_20240101_120000_abcd1234.csv", index=False, encoding="utf-8-sig")
    # summary（含 KEY_ 前缀列）
    pd.DataFrame(
        {
            "KEY_说明": ["x"],
            "KEY_Top20_hit_rate": [0.5],
            "split_index": [0],
            "test_start": ["20240101"],
            "test_end": ["20240201"],
            "bt_top_n": [20],
            "wf_run_id": ["wf_20240101_120000_abcd1234"],
        }
    ).to_csv(raw / "walk_forward_summary_test_0001.csv", index=False, encoding="utf-8-sig")
    # trades（含 F2 补的 4 列）
    pd.DataFrame(
        {
            "wf_run_id": ["wf1"],
            "split_index": [0],
            "model_version": ["v1"],
            "date": ["20240103"],
            "signal_date": ["20240102"],
            "stock": ["600000.SH"],
            "action": ["buy"],
            "price": [10.0],
            "shares": [100],
            "amount": [1000.0],
            "cost": [5.0],
            "buy_date": ["20240103"],
            "buy_price": [10.0],
            "buy_pnl_price": [10.0],
            "sell_pnl_price": [np.nan],
            "pnl_profit_amount": [np.nan],
            "pnl_profit_pct": [np.nan],
            "sell_type": [""],
            "sell_timing": [""],
            "sell_reason": ["expiry"],
            "trigger_type": ["normal"],
            "buy_type": ["initial"],
            "buy_reason": ["signal"],
        }
    ).to_csv(raw / "walk_forward_trades_wf_20240101_120000_abcd1234_split00.csv", index=False, encoding="utf-8-sig")
    # attribution
    pd.DataFrame(
        {
            "wf_run_id": ["wf1"],
            "split_index": [0],
            "model_version": ["v1"],
            "signal_date": ["20240102"],
            "ranking_date": ["20240102"],
            "execution_date": ["20240103"],
            "execution_stage": ["T1"],
            "tranche_idx": [0],
            "planned_stock": ["600000.SH"],
            "actual_stock": ["600000.SH"],
            "planned_rank": [1],
            "actual_rank": [1],
            "pred_score": [0.9],
            "target_weight": [0.05],
            "status": ["filled"],
            "reason": [""],
            "buy_price": [10.0],
            "signal_price": [9.9],
            "signal_to_buy_return": [0.01],
        }
    ).to_csv(raw / "walk_forward_execution_attribution_wf_20240101_120000_abcd1234_split00.csv", index=False, encoding="utf-8-sig")
    # 持仓快照（中文表头，含「买入 日」带空格）
    pd.DataFrame(
        {
            "运行标识": ["wf1"],
            "折序号": [0],
            "模型版本": ["v1"],
            "日期": ["20240103"],
            "股票代码": ["600000.SH"],
            "持仓股数": [100],
            "持仓市值": [1000.0],
            "持仓权重": [0.05],
            "组合总值": [20000.0],
            "买入 日": ["20240103"],
            "信号日": ["20240102"],
            "持有交易日数": [1],
            "到期执行日": ["20240131"],
            "剩余持有交易日": [19],
        }
    ).to_csv(raw / "walk_forward_持仓快照_wf_20240101_120000_abcd1234_split00.csv", index=False, encoding="utf-8-sig")
    # topk_detail
    pd.DataFrame(
        {
            "wf_run_id": ["wf1"],
            "split_index": [0],
            "test_start": ["20240101"],
            "test_end": ["20240201"],
            "model_version": ["v1"],
            "trade_date": ["20240103"],
            "topk": [20],
            "rank": [1],
            "ts_code": ["600000.SH"],
            "pred_score": [0.9],
            "true_return": [0.01],
            "score_column": ["final_score"],
            "ml_score": [0.9],
            "risk_score": [0.0],
            "final_score": [0.9],
        }
    ).to_csv(raw / "walk_forward_topk_details_wf_20240101_120000_abcd1234_split00.csv", index=False, encoding="utf-8-sig")
    # data_state
    (raw / "data_state_wf_20240101_120000_abcd1234.json").write_text(
        '{"data_state_id": "test123"}', encoding="utf-8"
    )
    return raw


class TestRunsConvert:
    def test_convert_full_batch(self, tmp_path):
        raw = _make_legacy_batch(tmp_path)
        out_root = tmp_path / "runs"
        report = convert_wf_batch(raw, out_root, "test_batch")
        assert report.row_check_pass is True
        assert not report.errors
        assert not report.missing_required

        out_dir = out_root / "test_batch"
        # chain_nav / summary
        assert (out_dir / "chain_nav.parquet").exists()
        summary = pd.read_parquet(out_dir / "summary.parquet")
        # KEY_ 前缀转小写蛇形
        assert "key_top20_hit_rate" in summary.columns
        assert "KEY_Top20_hit_rate" not in summary.columns
        # trades：stock→ts_code、date→trade_date、4 列保留
        trades = pd.read_parquet(out_dir / "folds" / "split00" / "trades.parquet")
        assert "ts_code" in trades.columns and "trade_date" in trades.columns
        for c in ("sell_reason", "trigger_type", "buy_type", "buy_reason"):
            assert c in trades.columns
        # 行数不变（数值逐位不动）
        assert len(trades) == 1
        # attribution：planned_stock→planned_ts_code
        attr = pd.read_parquet(out_dir / "folds" / "split00" / "attribution.parquet")
        assert "planned_ts_code" in attr.columns
        # 快照：中文表头映射
        snap = pd.read_parquet(out_dir / "folds" / "split00" / "holdings_snapshot.parquet")
        assert "ts_code" in snap.columns and "buy_date" in snap.columns
        # topk_detail
        assert (out_dir / "folds" / "split00" / "topk_detail.parquet").exists()
        # batch_meta
        import json

        meta = json.loads((out_dir / "batch_meta.json").read_text(encoding="utf-8"))
        assert meta["data_state"]["data_state_id"] == "test123"
        assert "topk_detail.missing" not in report.meta_notes

    def test_missing_topk_marks_meta_note(self, tmp_path):
        raw = _make_legacy_batch(tmp_path)
        for p in raw.glob("walk_forward_topk_details_*.csv"):
            p.unlink()
        report = convert_wf_batch(raw, tmp_path / "runs", "no_topk")
        assert report.meta_notes.get("topk_detail.missing") == "true"

    def test_missing_required_fails(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        pd.DataFrame({"date": ["20240102"], "nav": [1.0], "split_index": [0]}).to_csv(
            raw / "chain_nav_x.csv", index=False
        )
        report = convert_wf_batch(raw, tmp_path / "runs", "bad")
        assert report.row_check_pass is False
        assert "summary" in report.missing_required


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

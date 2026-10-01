# -*- coding: utf-8 -*-
"""v2 信号层尺子（P5a-1 交付物④）单元测试：合成数据，不依赖真实批次。

覆盖：同日配对与对齐硬校验、必需列与 rank 过滤、点估计判据口径（历史四轮语义）、
块长敏感性、命中率噪声带标注、topk 缺失降级、ruler_from_runs 折集合/数据态校验。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.evidence.signal_ruler import (
    NOISE_BAND_HIT_RATE_PP,
    NOISE_BAND_RETURN_BPS,
    RELATIVE_GATE,
    ruler_from_runs,
    ruler_verdict,
)
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_schema import TOPK_CONTRACT_COLS, TRADES_CONTRACT_COLS


def _mk_topk(days, seed, ret_mean, topk=20, split=0):
    """合成一臂 topk_detail（每日 topk 只）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for r in range(1, topk + 1):
            rows.append({
                "split_index": split, "trade_date": d, "topk": topk, "rank": r,
                "ts_code": f"600{r:04d}.SH",
                "true_return": rng.normal(ret_mean, 0.02),
                "pred_score": 0.5, "score_column": "final_score",
                "ml_score": 0.5, "risk_score": 0.0, "final_score": 0.5,
            })
    return pd.DataFrame(rows)


class TestRulerVerdict:
    def test_identical_arms_delta_zero(self):
        """同一臂 ⇒ Δ=0、相对降幅 0、过尺（不触发阻断）。"""
        days = pd.date_range("2024-01-01", periods=40).strftime("%Y%m%d")
        a = _mk_topk(days, seed=1, ret_mean=0.001)
        v = ruler_verdict(a, a.copy(), topk=20, block_days_list=[20], seed=1)
        assert v.delta_return_bps == pytest.approx(0.0, abs=1e-6)
        assert v.relative_drop == pytest.approx(0.0, abs=1e-9)
        assert v.passes is True
        assert v.n_signal_days == 40

    def test_better_arm_positive_delta(self):
        """臂明显更优 ⇒ Δ收益为正。"""
        days = pd.date_range("2024-01-01", periods=60).strftime("%Y%m%d")
        base = _mk_topk(days, seed=2, ret_mean=0.001)
        arm = _mk_topk(days, seed=2, ret_mean=0.0015)  # 同 seed 同噪声，仅均值高 0.05pp/日
        v = ruler_verdict(base, arm, topk=20, block_days_list=[20], seed=3)
        assert v.delta_return_bps > 0
        assert v.relative_drop < 0  # 负 = 变好
        assert v.n_signal_days == 60

    def test_no_common_days_raises(self):
        """两臂无共同（折, 信号日）⇒ 报错。"""
        base = _mk_topk(pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d"), seed=1, ret_mean=0.001)
        arm = _mk_topk(pd.date_range("2024-03-01", periods=10).strftime("%Y%m%d"), seed=1, ret_mean=0.001)
        with pytest.raises(ValueError, match="配对设计失效"):
            ruler_verdict(base, arm, topk=20, block_days_list=[20])

    def test_partial_misalignment_raises(self):
        """臂缺部分信号日 ⇒ 未完全对齐报错（配对设计失效不得静默继续）。"""
        days = pd.date_range("2024-01-01", periods=30).strftime("%Y%m%d")
        base = _mk_topk(days, seed=1, ret_mean=0.001)
        arm = _mk_topk(days[3:], seed=1, ret_mean=0.001)  # 缺前 3 天
        with pytest.raises(ValueError, match="配对设计失效"):
            ruler_verdict(base, arm, topk=20, block_days_list=[20])
        # 显式降级：仅告警并放行
        v = ruler_verdict(base, arm, topk=20, block_days_list=[20], allow_day_mismatch=True)
        assert v.n_signal_days == 27

    def test_missing_required_column_raises(self):
        """缺必需列 ⇒ 明确 ValueError（不裸 KeyError）。"""
        days = pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d")
        base = _mk_topk(days, seed=1, ret_mean=0.001).drop(columns=["true_return"])
        arm = _mk_topk(days, seed=1, ret_mean=0.001)
        with pytest.raises(ValueError, match="缺必需列"):
            ruler_verdict(base, arm, topk=20, block_days_list=[20])

    def test_rank_filter_applied(self):
        """rank > topk 的行被过滤（与生产 build_day_panels 双过滤一致）。"""
        days = pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d")
        base = _mk_topk(days, seed=1, ret_mean=0.001, topk=20)
        arm = _mk_topk(days, seed=1, ret_mean=0.001, topk=20)
        # 追加 rank>20 的极端行：若不过滤会显著拉偏均值
        junk = _mk_topk(days, seed=9, ret_mean=-0.5, topk=20)
        junk["rank"] = 25
        arm_dirty = pd.concat([arm, junk], ignore_index=True)
        v_clean = ruler_verdict(base, arm, topk=20, block_days_list=[5], seed=5)
        v_dirty = ruler_verdict(base, arm_dirty, topk=20, block_days_list=[5], seed=5)
        assert v_dirty.delta_return_bps == pytest.approx(v_clean.delta_return_bps, abs=1e-9)

    def test_point_estimate_governs_not_ci_low(self):
        """判据口径锁定（历史四轮语义）：点估计带内、相对 <9% 即过尺——
        即使区间下界破 −10bps（区间只作可读性参考，不进判定）。
        确定性构造：逐日差值交替 ±5pp（均值 −3bps、方差大 ⇒ 区间必宽）。"""
        days = pd.date_range("2024-01-01", periods=30).strftime("%Y%m%d")
        rows_base, rows_arm = [], []
        for i, d in enumerate(days):
            d_ret = 0.05 if i % 2 else -0.0506  # 均值 −3bps，std≈5pp
            for r in range(1, 21):
                rows_base.append({"split_index": 0, "trade_date": d, "topk": 20, "rank": r,
                                  "ts_code": f"600{r:04d}.SH", "true_return": 0.01})
                rows_arm.append({"split_index": 0, "trade_date": d, "topk": 20, "rank": r,
                                 "ts_code": f"600{r:04d}.SH", "true_return": 0.01 + d_ret})
        v = ruler_verdict(pd.DataFrame(rows_base), pd.DataFrame(rows_arm),
                          topk=20, block_days_list=[5], seed=42)
        assert abs(v.delta_return_bps) < NOISE_BAND_RETURN_BPS  # 点估计 −3bps 带内
        assert abs(v.relative_drop) < RELATIVE_GATE  # 相对 3%
        assert v.delta_return_ci95[0] < -NOISE_BAND_RETURN_BPS  # 区间下界确已破带
        assert v.passes is True  # 点估计口径：过尺（ci_low 口径则会误拦）

    def test_point_estimate_breach_blocks(self):
        """点估计 ≤ −10bps ⇒ 不过尺（无需看区间）。"""
        days = pd.date_range("2024-01-01", periods=60).strftime("%Y%m%d")
        base = _mk_topk(days, seed=4, ret_mean=0.01)
        arm = _mk_topk(days, seed=4, ret_mean=0.0085)  # 同 seed：Δ 恒 −15bps，区间极窄
        v = ruler_verdict(base, arm, topk=20, block_days_list=[20], seed=6)
        assert v.delta_return_bps < -NOISE_BAND_RETURN_BPS
        assert v.passes is False

    def test_block_sensitivity_reported(self):
        """块长敏感性：三档区间齐全，主块长区间 = by_block[主块长]。"""
        days = pd.date_range("2024-01-01", periods=90).strftime("%Y%m%d")
        a = _mk_topk(days, seed=7, ret_mean=0.005)
        b = _mk_topk(days, seed=8, ret_mean=0.0052)
        v = ruler_verdict(a, b, topk=20, seed=9)
        assert sorted(v.delta_return_ci95_by_block) == [20, 40, 60]
        assert v.delta_return_ci95 == v.delta_return_ci95_by_block[v.block_days]
        lo, hi = v.delta_return_ci95
        assert lo <= hi

    def test_hit_rate_band_annotated_not_gating(self):
        """Δ命中率超带只进 note 标注，不进 passes（历史四轮从未以命中率阻断）。"""
        days = pd.date_range("2024-01-01", periods=40).strftime("%Y%m%d")
        base = _mk_topk(days, seed=3, ret_mean=0.0)
        arm = _mk_topk(days, seed=3, ret_mean=0.0)
        # 臂全样本取正：命中率 ≈100% vs 基线 ≈50%（Δ≈+50pp 超带），均值收益同时为正
        arm["true_return"] = arm["true_return"].abs() + 0.0001
        v = ruler_verdict(base, arm, topk=20, block_days_list=[20], seed=3)
        assert abs(v.delta_hit_rate_pp) > NOISE_BAND_HIT_RATE_PP
        assert "超带" in v.note
        assert v.passes is True  # 命中率不进判定（Δ收益为正 ⇒ 过尺）


# ---- ruler_from_runs（经 runs_loader 全校验） ----


def _write_min_runs_batch(root: Path, name: str, ds_id: str, topk_by_split: dict) -> Path:
    """造最小合法 runs 批（2 折；无 daily——可缺；含 topk_detail）。"""
    batch = root / name
    batch.mkdir(parents=True)
    chain = pd.DataFrame({
        "date": [0, 1, 0, 1], "nav": [1.0, 1.02, 1.02, 1.0302],
        "split_index": [0, 0, 1, 1],
    })
    summary = pd.DataFrame({
        "batch_id": [name, name], "split_index": [0, 1],
        "test_start": ["20240101", "20240201"], "test_end": ["20240131", "20240229"],
        "bt_total_return": [1.02 / 1.0 - 1.0, 1.0302 / 1.02 - 1.0],
        "data_state_id": [ds_id, ds_id],
    })
    chain.to_parquet(batch / "chain_nav.parquet", index=False)
    summary.to_parquet(batch / "summary.parquet", index=False)
    cfg = {"bt_top_n": 20}
    (batch / "batch_meta.json").write_text(json.dumps({
        "batch_id": name, "config": cfg, "config_fingerprint": fingerprint(cfg),
        "data_state": {"data_state_id": ds_id}, "baseline_ref": None, "arms": [],
    }, ensure_ascii=False), encoding="utf-8")
    for split in (0, 1):
        fold_dir = batch / "folds" / f"split{split:02d}"
        fold_dir.mkdir(parents=True)
        # 最小 trades（全契约列）：1 买 1 卖
        row = {c: None for c in TRADES_CONTRACT_COLS}
        row.update({"wf_run_id": "wf1", "split_index": split, "model_version": "v1",
                    "signal_date": f"20240{split + 1}01", "ts_code": "600000.SH",
                    "price": 10.0, "shares": 100, "amount": 1000.0, "cost": 5.0,
                    "buy_price": 10.0, "sell_type": "", "sell_timing": "",
                    "lot_id": "l0"})
        buy_day = f"20240{split + 1}02"
        sell_day = f"20240{split + 1}03"
        buy = {**row, "trade_date": buy_day, "action": "buy", "buy_date": buy_day}
        sell = {**row, "trade_date": sell_day, "action": "sell", "buy_date": buy_day}
        pd.DataFrame([buy, sell])[TRADES_CONTRACT_COLS].to_parquet(
            fold_dir / "trades.parquet", index=False)
        if split in topk_by_split:
            topk_by_split[split].to_parquet(fold_dir / "topk_detail.parquet", index=False)
    return batch


def _mk_topk_detail(split: int, days, seed, ret_mean, topk=20) -> pd.DataFrame:
    df = _mk_topk(days, seed, ret_mean, topk=topk, split=split)
    df["wf_run_id"] = "wf1"
    df["test_start"] = "20240101"
    df["test_end"] = "20240131"
    df["model_version"] = "v1"
    return df[TOPK_CONTRACT_COLS]


class TestRulerFromRuns:
    def test_happy_path(self, tmp_path):
        d0 = pd.date_range("2024-01-02", periods=30).strftime("%Y%m%d")
        d1 = pd.date_range("2024-02-01", periods=30).strftime("%Y%m%d")
        kw = {"topk_by_split": {}}
        base = _write_min_runs_batch(tmp_path, "base", "ds1", {
            0: _mk_topk_detail(0, d0, 1, 0.001), 1: _mk_topk_detail(1, d1, 1, 0.001)})
        # 同 seed：噪声序列相同，仅均值偏移 +1bps ⇒ Δ 恒正（不受采样噪声影响）
        arm = _write_min_runs_batch(tmp_path, "arm", "ds1", {
            0: _mk_topk_detail(0, d0, 1, 0.0011), 1: _mk_topk_detail(1, d1, 1, 0.0011)})
        v = ruler_from_runs(base, arm, topk=20, block_days_list=[5])
        assert v.n_signal_days == 60
        assert v.delta_return_bps > 0

    def test_data_state_mismatch_raises(self, tmp_path):
        d0 = pd.date_range("2024-01-02", periods=10).strftime("%Y%m%d")
        d1 = pd.date_range("2024-02-01", periods=10).strftime("%Y%m%d")
        base = _write_min_runs_batch(tmp_path, "base", "ds1", {
            0: _mk_topk_detail(0, d0, 1, 0.001), 1: _mk_topk_detail(1, d1, 1, 0.001)})
        arm = _write_min_runs_batch(tmp_path, "arm", "ds2", {
            0: _mk_topk_detail(0, d0, 1, 0.001), 1: _mk_topk_detail(1, d1, 1, 0.001)})
        with pytest.raises(ValueError, match="数据态"):
            ruler_from_runs(base, arm, topk=20, block_days_list=[5])
        v = ruler_from_runs(base, arm, topk=20, block_days_list=[5], allow_state_mismatch=True)
        assert v is not None

    def test_fold_set_mismatch_raises(self, tmp_path):
        d0 = pd.date_range("2024-01-02", periods=10).strftime("%Y%m%d")
        d1 = pd.date_range("2024-02-01", periods=10).strftime("%Y%m%d")
        base = _write_min_runs_batch(tmp_path, "base", "ds1", {
            0: _mk_topk_detail(0, d0, 1, 0.001), 1: _mk_topk_detail(1, d1, 1, 0.001)})
        arm = _write_min_runs_batch(tmp_path, "arm", "ds1", {
            0: _mk_topk_detail(0, d0, 1, 0.001)})  # 折1 缺 topk_detail
        with pytest.raises(ValueError, match="折集合不一致"):
            ruler_from_runs(base, arm, topk=20)

    def test_all_folds_missing_topk_raises_contract_error(self, tmp_path):
        """全折缺 topk_detail ⇒ 契约降级错误（不是 pandas 原生 concat 错误）。"""
        base = _write_min_runs_batch(tmp_path, "base", "ds1", {})
        arm = _write_min_runs_batch(tmp_path, "arm", "ds1", {})
        with pytest.raises(ValueError, match="topk_detail 缺失"):
            ruler_from_runs(base, arm, topk=20)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

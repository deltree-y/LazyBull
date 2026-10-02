# -*- coding: utf-8 -*-
"""P5a-2 入场点敏感度计算口径测试（合成净值序列，不依赖真实批次）。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.v2_p5a2.run_entry_sensitivity import entry_distribution, summarize


class TestEntryDistribution:
    def test_monotonic_nav_all_positive_zero_dd(self):
        nav = np.linspace(1.0, 2.0, 1000)  # 单调上行
        dist = entry_distribution(nav, min_days=252)
        assert len(dist) == 1000 - 252  # t ≤ N-1-min_days
        assert (dist["cagr"] > 0).all()
        assert (dist["maxdd"] == 0).all()

    def test_min_days_filter_drops_tail(self):
        nav = np.linspace(1.0, 1.5, 300)
        dist = entry_distribution(nav, min_days=252)
        assert len(dist) == 48  # 300-252；末端 2 日窗口的伪年化被过滤
        assert dist["days"].min() >= 252

    def test_known_drawdown(self):
        # 先平后跌 50% 再平：从顶部入场的 MaxDD = -50%
        nav = np.concatenate([np.ones(500), np.linspace(1.0, 0.5, 300), np.full(500, 0.5)])
        dist = entry_distribution(nav, min_days=252)
        row = dist[dist["start_idx"] == 400].iloc[0]  # 顶部前入场
        assert row["maxdd"] == pytest.approx(-0.5, abs=1e-9)


class TestSummarize:
    def test_median_and_worst_k(self):
        nav = np.linspace(1.0, 2.0, 1000)
        splits = np.zeros(1000, dtype=int)
        dist = entry_distribution(nav, min_days=252)
        s = summarize(dist, splits, top_k=5)
        assert s["n_entries"] == len(dist)
        assert len(s["worst_k_by_cagr"]) == 5
        # 最差起点 = 最迟入场（累积涨幅最少）
        worst = s["worst_k_by_cagr"][0]
        assert worst["start_idx"] == dist["start_idx"].max()
        assert "有效自由度" in s["dof_note"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

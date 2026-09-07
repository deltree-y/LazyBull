"""期末异常亏损标签构建测试（合成数据，不依赖真实配置与真实数据）

覆盖方案 8.1 第一阶段验证项：open/open 端点、h 边界、缺数据、停牌、
期末未成熟、标签数学、执行受阻标记、输入不可变。
"""

from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.lazybull.factors.risk.volatility_factors import compute_sigma_daily_panel
from src.lazybull.risk.terminal_loss import (
    LABEL_STATUS_ENDPOINT_MISSING,
    LABEL_STATUS_IMMATURE,
    LABEL_STATUS_SIGMA_UNAVAILABLE,
    LABEL_STATUS_VALID,
    TerminalLossLabelConfig,
    build_terminal_loss_labels,
    summarize_label_coverage,
)

# 10 个合成交易日（连续字符串日历，labels 按日历位置推导端点）
CAL: List[str] = [f"2024010{d}" for d in range(2, 6)] + [
    "20240108", "20240109", "20240110", "20240111", "20240112", "20240115"
]

# 单股 A：第 2 日开盘大跌 10%，其余持平
OPEN_A = [100.0, 100.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]


def _panels(open_dict, sigma_value=0.05):
    """构建 open_adj / sigma 面板（sigma 固定值，隔离标签数学）。"""
    open_panel = pd.DataFrame(open_dict, index=CAL)
    sigma_panel = pd.DataFrame(
        {c: [sigma_value] * len(CAL) for c in open_panel.columns}, index=CAL
    )
    return open_panel, sigma_panel


class TestBuildTerminalLossLabels:
    def test_open_open_endpoint_and_label_math(self):
        """h=1 时 R = open(E)/open(T+1)-1，label 按 -k·σ·√h 比较。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=1)
        )
        row = labels[
            (labels["trade_date"] == CAL[0]) & (labels["ts_code"] == "A")
        ].iloc[0]
        # T=0: P0=open(T+1)=100, E=idx2 open=90 → R=-0.1
        assert row["reference_date"] == CAL[1]
        assert row["planned_exit_date"] == CAL[2]
        assert row["label_end_date"] == CAL[2]
        assert row["terminal_return"] == pytest.approx(-0.1)
        # bound = -1 * 0.05 * sqrt(1) = -0.05；-0.1 < -0.05 → 正例
        assert row["loss_label"] == 1
        assert row["label_status"] == LABEL_STATUS_VALID

        # T=2 之后端点持平 → R=0 → 非正例
        row2 = labels[
            (labels["trade_date"] == CAL[2]) & (labels["ts_code"] == "A")
        ].iloc[0]
        assert row2["terminal_return"] == pytest.approx(0.0)
        assert row2["loss_label"] == 0

    def test_h_grid_and_bounds_scale(self):
        """同一 T 不同 h 的端点按日历位置展开；h=2 界限按 sqrt(2) 放宽。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=2)
        )
        assert set(labels["h"]) == {1, 2}
        t0_h2 = labels[
            (labels["trade_date"] == CAL[0])
            & (labels["h"] == 2)
            & (labels["ts_code"] == "A")
        ].iloc[0]
        # h=2: E = idx(0)+1+2 = idx3，open=100 → R=0
        assert t0_h2["planned_exit_date"] == CAL[3]
        assert t0_h2["terminal_return"] == pytest.approx(0.0)
        assert t0_h2["loss_label"] == 0

    def test_immature_when_exit_beyond_calendar_end(self):
        """E 超出日历末端 → immature，价格列 NaN，不标记安全。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=2)
        )
        imm = labels[
            (labels["ts_code"] == "A")
            & (labels["label_status"] == LABEL_STATUS_IMMATURE)
        ]
        # h=1: T+1+h > idx9 → T ∈ {idx8, idx9}；h=2: T ∈ {idx7, idx8, idx9}
        imm_h1 = imm[imm["h"] == 1]
        assert set(imm_h1["trade_date"]) == {CAL[8], CAL[9]}
        imm_h2 = imm[imm["h"] == 2]
        assert set(imm_h2["trade_date"]) == {CAL[7], CAL[8], CAL[9]}
        assert imm["terminal_return"].isna().all()
        assert imm["loss_label"].isna().all()

    def test_endpoint_missing_when_t1_or_e_suspended(self):
        """T+1 或 E 停牌缺行 → endpoint_missing；T 日停牌不生成行。"""
        open_b = list(OPEN_A)
        open_b[1] = np.nan  # T=idx0 的 T+1 停牌
        open_b[5] = np.nan  # T=idx4 的 T+1 停牌；同时 idx5 本身停牌不生成行
        open_panel, sigma_panel = _panels({"B": open_b})
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=1)
        )
        by_t = {
            t: g.iloc[0]
            for t, g in labels[labels["ts_code"] == "B"].groupby("trade_date")
        }
        assert by_t[CAL[0]]["label_status"] == LABEL_STATUS_ENDPOINT_MISSING
        assert by_t[CAL[4]]["label_status"] == LABEL_STATUS_ENDPOINT_MISSING
        # idx5 当日停牌 → 不生成该 T 的行
        assert CAL[5] not in by_t

    def test_sigma_unavailable_marks_row_not_safe(self):
        """sigma 缺失或零波动 → sigma_unavailable，不得产出标签。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        sigma_panel.iloc[0, 0] = np.nan
        sigma_panel.iloc[2, 0] = 0.0  # 零波动（长期一字板）同样不可用
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=1)
        )
        by_t = {
            t: g.iloc[0]
            for t, g in labels[labels["ts_code"] == "A"].groupby("trade_date")
        }
        assert by_t[CAL[0]]["label_status"] == LABEL_STATUS_SIGMA_UNAVAILABLE
        assert by_t[CAL[2]]["label_status"] == LABEL_STATUS_SIGMA_UNAVAILABLE
        assert by_t[CAL[0]]["loss_label"] is np.nan or pd.isna(by_t[CAL[0]]["loss_label"])

    def test_status_priority_immature_over_endpoint(self):
        """immature 行内即使 T+1 也缺，状态仍是 immature（优先级最高）。"""
        open_c = list(OPEN_A)
        open_c[9] = np.nan
        open_panel, sigma_panel = _panels({"C": open_c})
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=1)
        )
        # T=idx8: T+1=idx9 有报价、E=idx10 超界 → immature（而非 endpoint）
        row = labels[
            (labels["ts_code"] == "C") & (labels["trade_date"] == CAL[8])
        ].iloc[0]
        assert row["label_status"] == LABEL_STATUS_IMMATURE

    def test_execution_blocked_keeps_label(self):
        """E 日跌停：标签保留且单独标记，不剔除高风险样本。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        limit_down = pd.DataFrame(0, index=CAL, columns=["A"])
        limit_down.loc[CAL[2], "A"] = 1  # T=0/h=1 的 E 日跌停
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel,
            TerminalLossLabelConfig(h_min=1, h_max=1),
            limit_down_panel=limit_down,
        )
        row = labels[(labels["trade_date"] == CAL[0]) & (labels["ts_code"] == "A")].iloc[0]
        assert row["label_status"] == LABEL_STATUS_VALID
        assert bool(row["execution_blocked"]) is True
        assert row["loss_label"] == 1

    def test_input_panels_not_mutated(self):
        """更改 T 后价格只影响标签：输入面板（X/sigma 来源）不被修改。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        open_before = open_panel.copy()
        sigma_before = sigma_panel.copy()
        build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=2)
        )
        assert_frame_equal(open_panel, open_before)
        assert_frame_equal(sigma_panel, sigma_before)

    def test_panel_misalignment_rejected(self):
        """sigma 面板与 open 面板未对齐必须报错，不静默 reindex。"""
        open_panel, sigma_panel = _panels({"A": OPEN_A})
        bad = sigma_panel.iloc[:-1]
        with pytest.raises(ValueError, match="对齐"):
            build_terminal_loss_labels(
                open_panel, bad, TerminalLossLabelConfig(h_min=1, h_max=1)
            )


class TestSummarizeCoverage:
    def test_coverage_pivot_and_event_rate(self):
        open_panel, sigma_panel = _panels({"A": OPEN_A, "B": OPEN_A})
        sigma_panel.iloc[0, 1] = np.nan
        labels = build_terminal_loss_labels(
            open_panel, sigma_panel, TerminalLossLabelConfig(h_min=1, h_max=1)
        )
        cov = summarize_label_coverage(labels)
        assert cov.loc[1, LABEL_STATUS_SIGMA_UNAVAILABLE] == 1
        # A 股 valid = 10 - immature2；B 股 valid = 10 - immature2 - sigma1
        assert cov.loc[1, LABEL_STATUS_VALID] == (len(CAL) - 2) + (len(CAL) - 3)
        assert 0.0 <= cov.loc[1, "event_rate"] <= 1.0


class TestComputeSigmaDailyPanel:
    def _long(self, code: str, closes: List[float], dates: List[str]):
        return pd.DataFrame(
            {"ts_code": code, "trade_date": dates, "close_adj": closes}
        )

    def test_strict_calendar_alignment_no_window_compression(self):
        """停牌缺行不压缩：停牌后 20 个日历槽内 sigma 必须为 NaN。"""
        n = 30
        dates = [f"2024010{d:02d}" for d in range(1, 10)] + [
            f"202401{d}" for d in range(10, 31)
        ]
        closes = [100.0 * (1.0 + 0.01 * ((-1) ** i)) for i in range(n)]  # ±1% 交替
        # 停牌股：第 10 日缺行（日历有槽、长表无行）
        suspended = self._long(
            "S", closes[:9] + closes[10:], dates[:9] + dates[10:]
        )
        normal = self._long("N", closes, dates)
        sigma = compute_sigma_daily_panel(
            pd.concat([suspended, normal], ignore_index=True), dates, window=20
        )
        # 正常股：pct_change 首行收益为 NaN 占一个槽，idx20 起窗口满 20 个有效收益
        assert sigma["N"].iloc[:19].isna().all()
        rets = pd.Series(closes).pct_change()
        expected = rets.iloc[1:21].std(ddof=1)
        assert sigma.loc[dates[20], "N"] == pytest.approx(expected)
        # 停牌股：缺行落在任何窗口内即 NaN——第 9 日（停牌日）起连续 NaN，
        # 直到停牌日退出 20 日窗口（idx9 + 20 = idx29 之外才有值，本例内恒 NaN）
        assert sigma["S"].iloc[19:].isna().all()
        assert sigma["S"].iloc[:8].isna().all()

    def test_adj_price_used_directly(self):
        """sigma 消费复权价：除息不产生伪波动。"""
        n = 22
        dates = [f"2024010{d:02d}" for d in range(1, 10)] + [
            f"202401{d}" for d in range(10, 23)
        ]
        assert len(dates) == n
        # 复权后价格恒定（真实价格除息下跌但 adj 平滑）→ 收益全 0 → sigma=0
        closes = [100.0] * n
        sigma = compute_sigma_daily_panel(
            self._long("F", closes, dates), dates, window=20
        )
        # idx20 起窗口满 20 个零收益 → sigma=0（由调用方按零波动无效处理）
        assert sigma["F"].iloc[20] == pytest.approx(0.0)
        assert sigma["F"].iloc[:19].isna().all()

    def test_empty_input(self):
        sigma = compute_sigma_daily_panel(pd.DataFrame(), ["20240102"])
        assert sigma.empty

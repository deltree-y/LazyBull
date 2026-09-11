"""覆盖审计测试：占比、代理条件事件率、endpoint_delayed 敏感性（合成数据）"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.terminal_loss import (
    AUDIT_PROXY_COLUMNS,
    DELAYED_SENSITIVITY_SHARE_THRESHOLD,
    LabelCoverageAccumulator,
    ProxyProfileAccumulator,
    TerminalLossLabelConfig,
    coverage_audit_required,
    delayed_endpoint_sensitivity,
    summarize_label_coverage,
)
from src.lazybull.risk.terminal_loss.labels import (
    LABEL_STATUS_ENDPOINT_MISSING,
    LABEL_STATUS_IMMATURE,
    LABEL_STATUS_SIGMA_UNAVAILABLE,
    LABEL_STATUS_VALID,
)

CAL = [f"202401{d:02d}" for d in range(1, 21)]


def _labels(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "ts_code",
            "trade_date",
            "h",
            "planned_exit_date",
            "label_status",
            "loss_label",
            "execution_blocked",
            "sigma_at_t",
        ],
    )


def _row(code, date, h, status, label=0.0, blocked=0, sigma=0.02):
    return {
        "ts_code": code,
        "trade_date": date,
        "h": h,
        "planned_exit_date": CAL[-1],
        "label_status": status,
        "loss_label": label,
        "execution_blocked": blocked,
        "sigma_at_t": sigma,
    }


def _audit_frame(rows) -> pd.DataFrame:
    """审计输入帧：标签表的 sigma_at_t → 审计口径 sigma_daily_20。"""
    return _labels(rows).rename(columns={"sigma_at_t": "sigma_daily_20"})


class TestLabelCoverageAccumulator:
    def test_chunked_accumulation_equals_single_frame(self):
        """分块累计必须等于单帧口径：均值不可累加，比率须由总数重算。"""
        chunk_a = _labels(
            [
                _row("A", CAL[0], 1, LABEL_STATUS_VALID, label=1),
                _row("B", CAL[0], 1, LABEL_STATUS_VALID, label=0),
            ]
        )
        chunk_b = _labels(
            [
                _row("A", CAL[1], 1, LABEL_STATUS_VALID, label=1),
                _row("B", CAL[1], 1, LABEL_STATUS_VALID, label=1),
                _row("C", CAL[1], 1, LABEL_STATUS_VALID, label=1),
            ]
        )
        acc = LabelCoverageAccumulator().add(chunk_a).add(chunk_b)
        frame = acc.to_frame()
        row = frame[(frame["h"] == 1) & (frame["label_status"] == LABEL_STATUS_VALID)].iloc[0]
        # 单帧口径 = 4/5 = 0.8；若错误地平均块内率会得到 (1/2 + 3/3)/2 = 0.75
        assert row["count"] == 5
        assert row["event_rate"] == pytest.approx(4 / 5)
        assert acc.valid_count == 5
        assert acc.valid_event_sum == 4

    def test_pivot_entry_matches_accumulator(self):
        labels = _labels(
            [
                _row("A", CAL[0], 2, LABEL_STATUS_VALID, label=1),
                _row("B", CAL[0], 2, LABEL_STATUS_ENDPOINT_MISSING),
                _row("C", CAL[0], 2, LABEL_STATUS_SIGMA_UNAVAILABLE),
            ]
        )
        pivot = summarize_label_coverage(labels)
        assert pivot.loc[2, LABEL_STATUS_VALID] == 1
        assert pivot.loc[2, LABEL_STATUS_ENDPOINT_MISSING] == 1
        assert pivot.loc[2, LABEL_STATUS_SIGMA_UNAVAILABLE] == 1
        assert pivot.loc[2, "event_rate"] == pytest.approx(1.0)

    def test_execution_blocked_counted_for_valid_only(self):
        labels = _labels(
            [
                _row("A", CAL[0], 3, LABEL_STATUS_VALID, label=1, blocked=1),
                _row("B", CAL[0], 3, LABEL_STATUS_VALID, label=0, blocked=0),
            ]
        )
        frame = LabelCoverageAccumulator().add(labels).to_frame()
        valid = frame[frame["label_status"] == LABEL_STATUS_VALID].iloc[0]
        assert valid["execution_blocked_count"] == 1
        missing = (
            LabelCoverageAccumulator()
            .add(_labels([_row("A", CAL[0], 3, LABEL_STATUS_IMMATURE)]))
            .to_frame()
        )
        assert np.isnan(missing.iloc[0]["execution_blocked_count"])

    def test_status_share_sums_to_one_per_h(self):
        labels = _labels(
            [
                _row("A", CAL[0], 4, LABEL_STATUS_VALID),
                _row("B", CAL[0], 4, LABEL_STATUS_IMMATURE),
                _row("C", CAL[0], 4, LABEL_STATUS_SIGMA_UNAVAILABLE),
                _row("D", CAL[0], 4, LABEL_STATUS_ENDPOINT_MISSING),
            ]
        )
        share = LabelCoverageAccumulator().add(labels).status_share()
        assert share[share["h"] == 4]["share"].sum() == pytest.approx(1.0)


class TestProxyProfile:
    def _proxies(self) -> pd.DataFrame:
        rows = []
        for code, ret20 in zip(["A", "B", "C", "D"], [-0.2, -0.1, 0.1, 0.2]):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": CAL[0],
                    "ret_20": ret20,
                    "cvar_95_20": ret20 / 10,
                    "amihud_illiq_20": 1.0,
                }
            )
        return pd.DataFrame(rows)

    def test_profile_separates_status_groups(self):
        labels = _audit_frame(
            [
                _row("A", CAL[0], 1, LABEL_STATUS_VALID, label=1, sigma=0.05),
                _row("B", CAL[0], 1, LABEL_STATUS_VALID, label=0, sigma=0.03),
                _row("D", CAL[0], 1, LABEL_STATUS_ENDPOINT_MISSING, sigma=0.09),
            ]
        )
        acc = ProxyProfileAccumulator().add(labels, self._proxies())
        profile = acc.profile_table()
        valid_sigma = profile[
            (profile["label_status"] == LABEL_STATUS_VALID) & (profile["proxy"] == "sigma_daily_20")
        ].iloc[0]
        missing_sigma = profile[
            (profile["label_status"] == LABEL_STATUS_ENDPOINT_MISSING)
            & (profile["proxy"] == "sigma_daily_20")
        ].iloc[0]
        assert valid_sigma["n"] == 2
        assert valid_sigma["mean"] == pytest.approx(0.04)
        assert missing_sigma["mean"] == pytest.approx(0.09)

    def test_conditional_event_rate_uses_day_cross_section_buckets(self):
        labels = _audit_frame(
            [
                _row("A", CAL[0], 1, LABEL_STATUS_VALID, label=1),
                _row("B", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("C", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("D", CAL[0], 1, LABEL_STATUS_VALID, label=0),
            ]
        )
        acc = ProxyProfileAccumulator().add(labels, self._proxies())
        cond = acc.conditional_table()
        ret20 = cond[cond["proxy"] == "ret_20"].sort_values("bucket")
        assert ret20["n_valid"].sum() == 4
        # 桶 0 只含 ret_20 最低的股票 A，且 A 是唯一事件 → 事件率 1.0
        assert ret20.iloc[0]["event_rate"] == pytest.approx(1.0)

    def test_implied_rate_between_bucket_rates(self):
        labels = _audit_frame(
            [
                _row("A", CAL[0], 1, LABEL_STATUS_VALID, label=1),
                _row("B", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("C", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("D", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("D", CAL[0], 1, LABEL_STATUS_ENDPOINT_MISSING),
            ]
        )
        acc = ProxyProfileAccumulator().add(labels, self._proxies())
        implied = acc.implied_missing_event_rate()
        row = implied[implied["proxy"] == "ret_20"].iloc[0]
        assert row["missing_n"] == 1
        assert 0.0 <= row["implied_event_rate"] <= 1.0
        assert row["delta"] == pytest.approx(row["implied_event_rate"] - row["valid_event_rate"])

    def test_requires_proxy_columns_gracefully(self):
        """代理列缺失（合成数据无量额）时不报错，只是该代理不入表。"""
        labels = _audit_frame([_row("A", CAL[0], 1, LABEL_STATUS_VALID, label=1)])
        acc = ProxyProfileAccumulator(proxies=tuple(AUDIT_PROXY_COLUMNS)).add(labels, None)
        profile = acc.profile_table()
        assert list(profile["proxy"]) == ["sigma_daily_20"]


class TestCoverageAuditRequired:
    def test_share_above_threshold_requires_sensitivity(self):
        share = pd.DataFrame(
            {
                "h": [1, 1, 1],
                "label_status": [
                    LABEL_STATUS_VALID,
                    LABEL_STATUS_ENDPOINT_MISSING,
                    LABEL_STATUS_SIGMA_UNAVAILABLE,
                ],
                "count": [90, 5, 5],
                "share": [0.9, 0.05, 0.05],
            }
        )
        result = coverage_audit_required(share)
        assert result["share"] == pytest.approx(0.10)
        assert result["required"] is True
        assert result["threshold"] == DELAYED_SENSITIVITY_SHARE_THRESHOLD
        assert result["by_status"][LABEL_STATUS_ENDPOINT_MISSING] == pytest.approx(0.05)

    def test_small_share_does_not_require(self):
        share = pd.DataFrame(
            {
                "h": [1, 1],
                "label_status": [LABEL_STATUS_VALID, LABEL_STATUS_ENDPOINT_MISSING],
                "count": [999, 1],
                "share": [0.999, 0.001],
            }
        )
        assert coverage_audit_required(share)["required"] is False

    def test_empty_share_is_not_required(self):
        result = coverage_audit_required(pd.DataFrame())
        assert result["required"] is False
        assert result["share"] is None


class TestDelayedEndpointSensitivity:
    def _panel(self, values: dict) -> pd.DataFrame:
        """values: {ts_code: [逐日开盘价]}（缺失日用 np.nan）。"""
        return pd.DataFrame(values, index=CAL, dtype=float)

    def test_delayed_event_counted_when_endpoint_missing(self):
        """E 起停牌两日、随后复牌下跌：delayed 端点应把该行算进来。"""
        n = len(CAL)
        # A：全程 100 开盘，E(=CAL[2]) 与 CAL[3] 停牌，CAL[4]=90（跌 10%）
        base = [100.0] * n
        base[4] = 90.0
        base[2] = np.nan
        base[3] = np.nan
        panel = self._panel({"A": base})
        labels = _labels([_row("A", CAL[0], 1, LABEL_STATUS_ENDPOINT_MISSING, sigma=0.01)])
        result = delayed_endpoint_sensitivity(labels, panel, TerminalLossLabelConfig())
        total = result["total"]
        assert total["n_endpoint_missing"] == 1
        assert total["n_evaluated"] == 1
        assert total["n_delay_unavailable"] == 0
        # R = 90/100(T+1=CAL[1]) - 1 = -10% < -1% → 事件
        assert total["n_event"] == 1
        assert total["event_rate"] == pytest.approx(1.0)

    def test_unavailable_when_delay_exceeds_window(self):
        n = len(CAL)
        base = [100.0] * n
        for i in range(2, n):  # E 起全部停牌
            base[i] = np.nan
        panel = self._panel({"A": base})
        labels = _labels([_row("A", CAL[0], 1, LABEL_STATUS_ENDPOINT_MISSING)])
        result = delayed_endpoint_sensitivity(
            labels, panel, TerminalLossLabelConfig(), max_delay_days=3
        )
        assert result["total"]["n_evaluated"] == 0
        assert result["total"]["n_delay_unavailable"] == 1

    def test_t1_missing_counted_separately(self):
        n = len(CAL)
        base = [100.0] * n
        base[1] = np.nan  # T+1 停牌：基准价不可得
        panel = self._panel({"A": base})
        labels = _labels([_row("A", CAL[0], 1, LABEL_STATUS_ENDPOINT_MISSING)])
        result = delayed_endpoint_sensitivity(labels, panel, TerminalLossLabelConfig())
        assert result["total"]["n_t1_missing"] == 1
        assert result["total"]["n_evaluated"] == 0

    def test_only_endpoint_missing_rows_are_audited(self):
        panel = self._panel({"A": [100.0] * len(CAL)})
        labels = _labels(
            [
                _row("A", CAL[0], 1, LABEL_STATUS_VALID, label=0),
                _row("A", CAL[1], 1, LABEL_STATUS_SIGMA_UNAVAILABLE),
            ]
        )
        result = delayed_endpoint_sensitivity(labels, panel, TerminalLossLabelConfig())
        assert result["total"]["n_endpoint_missing"] == 0

    def test_missing_required_column_raises(self):
        panel = self._panel({"A": [100.0] * len(CAL)})
        with pytest.raises(ValueError, match="需要 ts_code/trade_date/h"):
            delayed_endpoint_sensitivity(pd.DataFrame({"ts_code": ["A"]}), panel)

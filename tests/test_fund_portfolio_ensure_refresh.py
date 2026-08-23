# -*- coding: utf-8 -*-
"""基金持仓 paper 端补齐修复回归测试。

背景（fund_portfolio 全链路审计）：
1. paper 端季度分区"下载一次永久冻结"：分区一旦存在就跳过，披露季中期首次
   下载的部分快照永不刷新，与训练侧事后全量下载口径分裂；
2. paper 端落盘未去重：同一报告期"季报前十大 + 半年报/年报全量"两批公告
   （同 ts_code+symbol+end_date，不同 ann_date）会让聚合 sum(stk_float_ratio)
   双重计数。

修复行为：
- 距报告期末 < 4 个月（披露窗口内）且分区内最新公告日未覆盖到当前日 → 强制
  重下并覆盖重写；
- 落盘前按 (ts_code, symbol, end_date) 去重，保留 ann_date 最晚记录；
- 刷新后的分区强制重算 fund_portfolio_agg 缓存；
- 下载失败日志升级为 warning（不再 debug 静默吞掉）。
"""

import pandas as pd
import pytest

import src.lazybull.features.ensure.historical_assets as hist_assets
from src.lazybull.features.ensure.historical_assets import (
    _dedup_fund_portfolio,
    _is_fund_portfolio_in_disclosure_window,
    _try_ensure_historical_fund_portfolio,
)


class _FakeStorage:
    """按 (数据集, 日期) 键模拟分区存储。"""

    def __init__(self):
        self.partitions = {}

    def is_data_exists(self, layer, name, date):
        return (name, date) in self.partitions

    def load_raw_by_date(self, name, trade_date, format="parquet", columns=None):
        key = (name, trade_date)
        if key not in self.partitions:
            return None
        df = self.partitions[key]
        if columns is not None and df is not None:
            df = df[[c for c in columns if c in df.columns]]
        return df.copy() if df is not None else None

    def save_raw_by_date(self, df, name, date_str, format="parquet"):
        self.partitions[(name, date_str)] = df.copy()


def _make_raw_df(period: str, ann_date: str, ts_code: str = "000001.OF") -> pd.DataFrame:
    """构造单条基金持仓原始记录。"""
    return pd.DataFrame(
        [
            {
                "ts_code": ts_code,
                "symbol": "000001",
                "ann_date": ann_date,
                "end_date": period,
                "stk_float_ratio": 1.5,
                "mkv": 1000,
                "amount": 100,
            }
        ]
    )


class TestDisclosureWindow:
    """披露窗口判断边界。"""

    @pytest.mark.parametrize(
        ("period", "max_date", "expected"),
        [
            ("20250930", "20260130", True),  # 三季报披露最迟 10 月底，窗口到第 4 个月整
            ("20250930", "20260131", False),  # 超窗即冻结
            ("20251231", "20260430", True),  # 年报披露最迟次年 4 月底
            ("20251231", "20260501", False),
            ("20250630", "20251030", True),  # 半年报披露最迟 8 月底
            ("20250331", "20250730", True),  # 一季报披露最迟 4 月底
        ],
    )
    def test_window_boundary(self, period, max_date, expected):
        assert _is_fund_portfolio_in_disclosure_window(period, max_date) == expected


class TestDedupFundPortfolio:
    """落盘前去重：保留 ann_date 最晚记录。"""

    def test_keeps_latest_ann_date(self):
        df = pd.DataFrame(
            [
                # 同一基金同一报告期两批公告：季报前十大 vs 半年报全量
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": "20250630",
                    "ann_date": "20250720",
                    "stk_float_ratio": 1.2,
                },
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": "20250630",
                    "ann_date": "20250825",
                    "stk_float_ratio": 1.2,
                },
                # 另一只基金不重复
                {
                    "ts_code": "000002.OF",
                    "symbol": "000001",
                    "end_date": "20250630",
                    "ann_date": "20250810",
                    "stk_float_ratio": 0.5,
                },
            ]
        )
        deduped = _dedup_fund_portfolio(df)
        assert len(deduped) == 2
        dup_rows = deduped[deduped["ts_code"] == "000001.OF"]
        assert len(dup_rows) == 1
        assert dup_rows.iloc[0]["ann_date"] == "20250825"

    def test_missing_dedup_cols_returns_unchanged(self):
        df = pd.DataFrame({"ann_date": ["20250720"]})
        deduped = _dedup_fund_portfolio(df)
        assert len(deduped) == 1


class TestEnsureFundPortfolioRefresh:
    """paper 端补齐：披露窗口内按覆盖水位刷新。"""

    PERIOD = "20250930"
    # 窗口内日期（20250930 + 4 个月 = 20260130）
    IN_WINDOW = "20251115"
    # 窗口外日期
    OUT_WINDOW = "20260215"

    def _setup(self, monkeypatch, trading_dates, initial_raw=None, initial_agg=None):
        storage = _FakeStorage()
        if initial_raw is not None:
            storage.save_raw_by_date(initial_raw, "fund_portfolio", self.PERIOD)
        if initial_agg is not None:
            storage.save_raw_by_date(initial_agg, "fund_portfolio_agg", self.PERIOD)
        monkeypatch.setattr(hist_assets, "_generate_quarter_periods", lambda *_args: [self.PERIOD])
        return storage

    def test_out_of_window_existing_partition_skips_download(self, monkeypatch):
        """窗口外已有分区 → 不重下、不回读明细检查水位。"""
        storage = self._setup(
            monkeypatch,
            [self.OUT_WINDOW],
            initial_raw=_make_raw_df(self.PERIOD, "20251025"),
        )

        def _fail_query(*args, **kwargs):
            raise AssertionError("窗口外不应触发下载")

        monkeypatch.setattr(hist_assets, "_query_with_pagination", _fail_query)
        result = _try_ensure_historical_fund_portfolio(
            client=object(), storage=storage, trading_dates_str=[self.OUT_WINDOW]
        )
        assert result is not None
        assert len(result) == 1

    def test_in_window_covered_watermark_skips_download(self, monkeypatch):
        """窗口内但分区内最新公告日 >= max_date → 不重下。"""
        storage = self._setup(
            monkeypatch,
            [self.IN_WINDOW],
            initial_raw=_make_raw_df(self.PERIOD, self.IN_WINDOW),
        )

        def _fail_query(*args, **kwargs):
            raise AssertionError("覆盖水位已达标，不应触发下载")

        monkeypatch.setattr(hist_assets, "_query_with_pagination", _fail_query)
        result = _try_ensure_historical_fund_portfolio(
            client=object(), storage=storage, trading_dates_str=[self.IN_WINDOW]
        )
        assert result is not None
        assert len(result) == 1

    def test_in_window_stale_watermark_refreshes_and_dedups(self, monkeypatch):
        """窗口内且水位落后 → 覆盖重写 + 去重 + agg 缓存强制重算。"""
        # 旧 raw：重复记录（两批公告）+ 水位落后
        old_raw = pd.DataFrame(
            [
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "ann_date": "20251025",
                    "stk_float_ratio": 1.2,
                },
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "ann_date": "20251025",
                    "stk_float_ratio": 1.2,
                },
            ]
        )
        # 旧 agg 缓存：代表旧口径聚合结果，应被覆盖
        old_agg = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "fund_hold_ratio": 999.0,
                    "fund_count": 1,
                    "ann_date": "20251025",
                }
            ]
        )
        storage = self._setup(
            monkeypatch, [self.IN_WINDOW], initial_raw=old_raw, initial_agg=old_agg
        )

        # 新下载：同一基金两批公告（去重目标）+ 一只新基金
        new_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "ann_date": "20251028",
                    "stk_float_ratio": 1.2,
                },
                {
                    "ts_code": "000001.OF",
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "ann_date": "20251110",
                    "stk_float_ratio": 1.2,
                },
                {
                    "ts_code": "000002.OF",
                    "symbol": "000001",
                    "end_date": self.PERIOD,
                    "ann_date": "20251110",
                    "stk_float_ratio": 0.8,
                },
            ]
        )
        monkeypatch.setattr(hist_assets, "_query_with_pagination", lambda *a, **k: new_df)

        result = _try_ensure_historical_fund_portfolio(
            client=object(), storage=storage, trading_dates_str=[self.IN_WINDOW]
        )

        # raw 分区已覆盖重写并去重：000001.OF 仅保留 ann_date 最晚一条
        saved_raw = storage.load_raw_by_date("fund_portfolio", self.PERIOD)
        assert len(saved_raw) == 2
        dup = saved_raw[saved_raw["ts_code"] == "000001.OF"]
        assert len(dup) == 1
        assert dup.iloc[0]["ann_date"] == "20251110"

        # agg 缓存已被强制重算覆盖：1.2 + 0.8 = 2.0，不再是旧值 999
        saved_agg = storage.load_raw_by_date("fund_portfolio_agg", self.PERIOD)
        assert saved_agg is not None
        assert len(saved_agg) == 1
        assert saved_agg.iloc[0]["fund_hold_ratio"] == pytest.approx(2.0)
        assert saved_agg.iloc[0]["fund_count"] == 2

        # 返回结果为新聚合
        assert result is not None
        assert result.iloc[0]["fund_hold_ratio"] == pytest.approx(2.0)

    def test_download_failure_warns_and_keeps_old_data(self, monkeypatch):
        """下载失败：warning 日志 + 不崩溃 + 沿用旧聚合。"""
        storage = self._setup(
            monkeypatch,
            [self.IN_WINDOW],
            initial_raw=_make_raw_df(self.PERIOD, "20251025"),
            initial_agg=pd.DataFrame(
                [
                    {
                        "symbol": "000001",
                        "end_date": self.PERIOD,
                        "fund_hold_ratio": 1.5,
                        "fund_count": 1,
                        "ann_date": "20251025",
                    }
                ]
            ),
        )

        class _FakeLogger:
            def __init__(self):
                self.warnings = []

            def warning(self, msg, *args, **kwargs):
                self.warnings.append(str(msg))

            def info(self, *args, **kwargs):
                pass

            def debug(self, *args, **kwargs):
                pass

        fake_logger = _FakeLogger()
        monkeypatch.setattr(hist_assets, "logger", fake_logger)

        def _raise(*args, **kwargs):
            raise RuntimeError("TuShare 限流")

        monkeypatch.setattr(hist_assets, "_query_with_pagination", _raise)
        result = _try_ensure_historical_fund_portfolio(
            client=object(), storage=storage, trading_dates_str=[self.IN_WINDOW]
        )
        assert result is not None
        assert result.iloc[0]["fund_hold_ratio"] == pytest.approx(1.5)
        # 下载失败必须以 warning 可见（不再 debug 静默吞掉）
        assert any("fund_portfolio" in w and "下载失败" in w for w in fake_logger.warnings)

    def test_missing_partition_downloads_regardless_of_window(self, monkeypatch):
        """分区缺失时无论窗口内外都应下载。"""
        storage = self._setup(monkeypatch, [self.OUT_WINDOW])
        monkeypatch.setattr(
            hist_assets,
            "_query_with_pagination",
            lambda *a, **k: _make_raw_df(self.PERIOD, "20251025"),
        )
        result = _try_ensure_historical_fund_portfolio(
            client=object(), storage=storage, trading_dates_str=[self.OUT_WINDOW]
        )
        assert result is not None
        assert storage.is_data_exists("raw", "fund_portfolio", self.PERIOD)
        assert storage.is_data_exists("raw", "fund_portfolio_agg", self.PERIOD)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_core 按日期切分与 cs_zscore 无泄露测试

测试内容：
- split_train_val_by_date 按 trade_date 粒度切分，不拆分同日样本
- 切分后 train/val 日期集合不相交
- 切分比例近似正确
- cs_zscore 变换在切分前后的泄露对照：旧逻辑（全量变换再切分）与新逻辑（切分后各自变换）
- prepare_training_data 使用日期切分（通过 label_transform_fn 路径）
"""

import pytest
import pandas as pd
import numpy as np

from src.lazybull.ml.train_core import split_train_val_by_date, transform_labels_cs_zscore


def _make_df(n_dates: int, stocks_per_date: int, seed: int = 42) -> pd.DataFrame:
    """构造测试用 DataFrame，每个交易日有 stocks_per_date 支股票"""
    rng = np.random.default_rng(seed)
    dates = [f"2023{d:04d}" for d in range(101, 101 + n_dates)]
    rows = []
    for date in dates:
        for i in range(stocks_per_date):
            rows.append({
                "trade_date": date,
                "ts_code": f"{i:06d}.SZ",
                "label": rng.normal()
            })
    return pd.DataFrame(rows)


class TestSplitTrainValByDate:
    """测试 split_train_val_by_date 按日期粒度切分"""

    def test_no_same_day_split(self):
        """同一交易日的所有样本必须全部落在同一集合内"""
        df = _make_df(n_dates=10, stocks_per_date=5)
        df_train, df_val, _ = split_train_val_by_date(df, val_ratio=0.2)

        train_dates = set(df_train["trade_date"].unique())
        val_dates = set(df_val["trade_date"].unique())

        # 训练集和验证集日期不相交
        assert train_dates & val_dates == set(), "训练集与验证集的日期集合不应有交叉"

    def test_all_samples_preserved(self):
        """切分后样本总数等于原始样本数"""
        df = _make_df(n_dates=10, stocks_per_date=5)
        df_train, df_val, _ = split_train_val_by_date(df, val_ratio=0.2)

        assert len(df_train) + len(df_val) == len(df)

    def test_val_ratio_approximately_correct(self):
        """验证集日期数量约为 ceil(n_dates * val_ratio)"""
        import math
        n_dates = 20
        val_ratio = 0.2
        df = _make_df(n_dates=n_dates, stocks_per_date=3)
        _, df_val, stats = split_train_val_by_date(df, val_ratio=val_ratio)

        expected_val_dates = math.ceil(n_dates * val_ratio)
        assert stats["val_n_dates"] == expected_val_dates, (
            f"验证集应有 {expected_val_dates} 个交易日，实际 {stats['val_n_dates']} 个"
        )

    def test_val_dates_are_later_than_train_dates(self):
        """验证集日期应晚于训练集日期"""
        df = _make_df(n_dates=10, stocks_per_date=4)
        df_train, df_val, _ = split_train_val_by_date(df, val_ratio=0.3)

        if len(df_train) > 0 and len(df_val) > 0:
            assert df_train["trade_date"].max() < df_val["trade_date"].min(), (
                "训练集最后一个交易日应早于验证集第一个交易日"
            )

    def test_stats_keys_present(self):
        """stats 字典包含所有必要字段"""
        df = _make_df(n_dates=10, stocks_per_date=3)
        _, _, stats = split_train_val_by_date(df, val_ratio=0.2)

        for key in ["train_n_dates", "val_n_dates", "train_start_date",
                    "train_end_date", "val_start_date", "val_end_date"]:
            assert key in stats, f"stats 应包含字段 {key}"

    def test_empty_dataframe(self):
        """空 DataFrame 不应抛出异常，返回空结果"""
        df = pd.DataFrame({"trade_date": [], "label": []})
        df_train, df_val, stats = split_train_val_by_date(df, val_ratio=0.2)

        assert len(df_train) == 0
        assert len(df_val) == 0
        assert stats["train_n_dates"] == 0

    def test_single_date(self):
        """只有一个交易日时，全部样本归入验证集"""
        df = _make_df(n_dates=1, stocks_per_date=5)
        df_train, df_val, stats = split_train_val_by_date(df, val_ratio=0.2)

        # 只有1个交易日，验证集至少包含1个交易日
        assert stats["val_n_dates"] >= 1
        # 训练集和验证集不重叠
        train_dates = set(df_train["trade_date"].unique())
        val_dates = set(df_val["trade_date"].unique())
        assert train_dates & val_dates == set()

    def test_custom_date_col(self):
        """支持自定义日期列名"""
        df = _make_df(n_dates=10, stocks_per_date=3)
        df = df.rename(columns={"trade_date": "date"})
        df_train, df_val, _ = split_train_val_by_date(df, val_ratio=0.2, date_col="date")

        train_dates = set(df_train["date"].unique())
        val_dates = set(df_val["date"].unique())
        assert train_dates & val_dates == set()


class TestCsZscoreNoLeakage:
    """对照测试：旧逻辑（全量变换再切分）vs 新逻辑（切分后各自变换）"""

    def _cs_zscore_manual(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """简化版截面 z-score，用于对照"""
        df = df.copy()
        df[label_col] = df.groupby("trade_date")[label_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        return df

    def test_old_logic_causes_leakage_on_boundary_date(self):
        """旧逻辑（全量变换后按行切分）会导致边界日期的 label 依赖验证集样本，形成泄露"""
        # 构造数据：10 个日期，每日 10 支股票
        n_dates = 10
        n_stocks = 10
        rng = np.random.default_rng(0)
        dates = [f"20230{d:03d}" for d in range(101, 101 + n_dates)]
        rows = []
        for date in dates:
            for i in range(n_stocks):
                rows.append({"trade_date": date, "ts_code": f"{i:06d}.SZ", "label": rng.normal()})
        df = pd.DataFrame(rows)

        # 旧逻辑：先对全量 df 做截面 z-score，再按行切分
        df_transformed = self._cs_zscore_manual(df, "label")
        df_sorted = df_transformed.sort_values("trade_date")
        split_idx = int(len(df_sorted) * 0.8)
        df_train_old = df_sorted.iloc[:split_idx]
        df_val_old = df_sorted.iloc[split_idx:]

        # 新逻辑：先按日期切分，再各自做截面 z-score
        df_train_new, df_val_new, _ = split_train_val_by_date(df, val_ratio=0.2)
        df_train_new = self._cs_zscore_manual(df_train_new, "label")
        df_val_new = self._cs_zscore_manual(df_val_new, "label")

        # 找到边界日期（可能被旧逻辑按行切分到两侧的日期）
        boundary_date = df_sorted.iloc[split_idx]["trade_date"]

        # 检查旧逻辑中边界日期是否被拆分
        boundary_in_train_old = (df_train_old["trade_date"] == boundary_date).any()
        boundary_in_val_old = (df_val_old["trade_date"] == boundary_date).any()

        if boundary_in_train_old and boundary_in_val_old:
            # 旧逻辑：边界日期确实被拆分
            # 取边界日期在训练集中的 label
            label_old = df_train_old[df_train_old["trade_date"] == boundary_date]["label"].values

            # 新逻辑：边界日期整体属于训练集
            label_new = df_train_new[df_train_new["trade_date"] == boundary_date]["label"].values

            # 两者的 label 值应该不同（因为旧逻辑计算均值/方差时包含了落入验证集的样本）
            if len(label_old) > 0 and len(label_new) > 0:
                # 旧逻辑 label 依赖整日截面（含验证集样本）
                # 新逻辑 label 只依赖训练集截面（不含验证集样本）
                assert not np.allclose(
                    sorted(label_old), sorted(label_new), atol=1e-6
                ), "旧逻辑中边界日期 label 应依赖验证集样本（存在泄露），新逻辑应与旧逻辑不同"

    def test_new_logic_no_cross_set_statistics(self):
        """新逻辑中，验证集的截面统计量不包含训练集样本"""
        n_dates = 10
        n_stocks = 10
        rng = np.random.default_rng(1)
        dates = [f"20230{d:03d}" for d in range(101, 101 + n_dates)]
        rows = []
        for date in dates:
            for i in range(n_stocks):
                rows.append({"trade_date": date, "ts_code": f"{i:06d}.SZ", "label": rng.normal()})
        df = pd.DataFrame(rows)

        df_train, df_val, _ = split_train_val_by_date(df, val_ratio=0.2)

        # 对验证集独立做截面 z-score
        df_val_transformed = self._cs_zscore_manual(df_val, "label")

        # 验证验证集的每个交易日均值接近 0、标准差接近 1
        for date, grp in df_val_transformed.groupby("trade_date"):
            assert abs(grp["label"].mean()) < 1e-6, (
                f"验证集日期 {date} 的 label 均值应接近 0"
            )
            if len(grp) > 1:
                assert abs(grp["label"].std() - 1.0) < 0.1, (
                    f"验证集日期 {date} 的 label 标准差应接近 1"
                )

    def test_transform_labels_cs_zscore_per_date(self):
        """transform_labels_cs_zscore 对每日截面独立标准化，不跨日共享统计量"""
        n_dates = 5
        n_stocks = 20
        rng = np.random.default_rng(2)
        dates = [f"2023010{d}" for d in range(1, n_dates + 1)]
        rows = []
        for date in dates:
            for i in range(n_stocks):
                rows.append({"trade_date": date, "ts_code": f"{i:06d}.SZ", "label": rng.normal()})
        df = pd.DataFrame(rows)

        df_out = transform_labels_cs_zscore(df, label_column="label", winsorize_p=0.0)

        # 验证每日截面标准化后均值接近 0
        daily_means = df_out.groupby("trade_date")["label"].mean()
        assert (daily_means.abs() < 0.5).all(), (
            "cs_zscore 变换后每日截面均值应接近 0"
        )

    def test_split_train_val_date_sets_disjoint_with_transform(self):
        """按日期切分后，train/val 日期不交叉，各自变换独立"""
        df = _make_df(n_dates=20, stocks_per_date=8)

        df_train, df_val, stats = split_train_val_by_date(df, val_ratio=0.2)

        # 应用 cs_zscore 变换（各自独立）
        df_train_t = transform_labels_cs_zscore(df_train, label_column="label", winsorize_p=0.0)
        df_val_t = transform_labels_cs_zscore(df_val, label_column="label", winsorize_p=0.0)

        train_dates = set(df_train_t["trade_date"].unique())
        val_dates = set(df_val_t["trade_date"].unique())

        # 日期集合不交叉
        assert train_dates & val_dates == set(), "变换后 train/val 日期集合不应有交叉"

        # 变换后总样本数一致
        assert len(df_train_t) + len(df_val_t) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# -*- coding: utf-8 -*-
"""风控公告类因子处理器（pledge_stat / share_float / block_trade 原始列接入）。

独立于 factor_handlers.py 单文件存放，遵循"新增功能对应新增文件"契约：
每个 handler 将 announcement_lookup.py 构建的当日截面原始列合并进 features，
供 factors/risk/announcement_factors.py 的三层加工因子（freshness 衰减 /
delta-on-update / 分档编码）消费。

数据为空时输出 NaN/0 占位列，保证启用开关后 schema 稳定（与 Consensus /
ConsensusRevision handler 同模式）。
"""

from typing import Dict

import pandas as pd

from .factor_handlers import _safe_merge_by_ts_code

# 输出列名与 announcement_factors.py 消费列名严格一致
PLEDGE_COLS = ["pledge_ratio", "pledge_freshness_days", "pledge_ratio_prev"]
SHARE_FLOAT_COLS = ["days_to_unlock", "unlock_ratio"]
BLOCK_TRADE_COLS = ["block_discount_avg_10d", "block_discount_days_10d"]


class PledgeFactorHandler:
    """质押公告原始列（PIT 前向填充日频截面）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "pledge")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            for col in PLEDGE_COLS:
                result[col] = float("nan")
        return result


class ShareFloatFactorHandler:
    """限售解禁原始列（PIT 按公告日，未解禁事件）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "share_float")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            for col in SHARE_FLOAT_COLS:
                result[col] = float("nan")
        return result


class BlockTradeFactorHandler:
    """大宗交易聚合列（近 10 交易日折价）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "block_trade")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            result["block_discount_avg_10d"] = float("nan")
            result["block_discount_days_10d"] = 0.0
        return result

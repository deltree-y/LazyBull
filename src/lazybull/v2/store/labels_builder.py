# -*- coding: utf-8 -*-
"""v2 labels 构建器：从构建完成的单日大表抽 3 张标签表并落 store。

列名适配（登记）：冻结文档 §3 的物理列 ``y_ret_N / neu_y_ret_N`` 在 v2 store 层
适配为协议形态 ``label_value / neu_label_value``（`docs/contracts/protocols.md` §1.1
load_labels 返回 columns=[label_value, maturity_status]；neu 变体随表同存）。
本模块是适配唯一发生处，下游一律只见适配后列名。

成熟度生命周期（冻结 §3）：封存时点 = T + max(h) = T+20 端点可算日——
T 日 sealed ⟺ 交易日历中 idx(T)+1+20 存在（T+1 起算第 20 个交易日存在）；
否则 forming（封存前幂等可重写，封存后拒绝改写，由 store 侧强制）。
"""

from __future__ import annotations

import bisect
from typing import Sequence

import pandas as pd
from loguru import logger

from src.lazybull.v2.store.column_groups import LABEL_TABLES

__all__ = ["MAX_LABEL_HORIZON", "maturity_status_for", "extract_labels", "append_labels_for_day"]

#: 最大标签 horizon（冻结 §3：max(h)=20；horizons={5,10,20} 冻结于 §1）
MAX_LABEL_HORIZON = 20

#: 标签表输出列（协议形态 + neu 变体 + 成熟度）
_LABEL_OUT_COLUMNS = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]


def maturity_status_for(date: str, trading_dates: Sequence[str]) -> str:
    """T 日成熟度：交易日历中 idx(T)+1+MAX_LABEL_HORIZON 存在 ⇒ sealed，否则 forming。

    Args:
        date: 交易日（YYYYMMDD，须在 trading_dates 中）
        trading_dates: 交易日历（YYYYMMDD 升序；须覆盖 T 之后足够窗口，
            回填应传全量日历而非构建区间子集）
    """
    dates = [str(d) for d in trading_dates]
    idx = bisect.bisect_left(dates, str(date))
    if idx >= len(dates) or dates[idx] != str(date):
        raise ValueError(f"日期 {date} 不在交易日历中（无法判定标签成熟度）")
    return "sealed" if idx + 1 + MAX_LABEL_HORIZON < len(dates) else "forming"


def extract_labels(df_day: pd.DataFrame, trading_dates: Sequence[str]) -> dict[str, pd.DataFrame]:
    """从单日大表抽 3 张标签表（按 LABEL_TABLES），列名适配为 store 协议形态。

    Args:
        df_day: 构建完成的单日大表（须含 ts_code/trade_date + 6 个标签列；单日）
        trading_dates: 交易日历（成熟度判定用，见 maturity_status_for）

    Returns:
        ``{label_name: DataFrame[ts_code, trade_date, label_value, neu_label_value,
        maturity_status]}``（3 表同一成熟度）。
    """
    for col in ("ts_code", "trade_date"):
        if col not in df_day.columns:
            raise ValueError(f"标签抽取缺键列: {col}")
    dates = df_day["trade_date"].astype(str).unique()
    if len(dates) != 1:
        raise ValueError(f"标签抽取要求单日大表，实得日期 {sorted(dates)[:5]}")
    status = maturity_status_for(dates[0], trading_dates)
    tables: dict[str, pd.DataFrame] = {}
    for label_name, (value_col, neu_col) in LABEL_TABLES.items():
        missing = [c for c in (value_col, neu_col) if c not in df_day.columns]
        if missing:
            raise ValueError(f"标签 {label_name} 抽取缺列 {missing}（大表标签列不全）")
        out = df_day[["ts_code", "trade_date", value_col, neu_col]].rename(
            columns={value_col: "label_value", neu_col: "neu_label_value"}
        )
        out = out.assign(maturity_status=status)
        tables[label_name] = out[_LABEL_OUT_COLUMNS].reset_index(drop=True)
    return tables


def append_labels_for_day(
    store: object, df_day: pd.DataFrame, trading_dates: Sequence[str]
) -> None:
    """抽取并逐表落 store（store.append_labels 的 sealed/forming 语义随之强制生效）。

    Args:
        store: PanelDataStore（DataStore 协议实现）
        df_day: 构建完成的单日大表
        trading_dates: 交易日历（全量）
    """
    for label_name, table in extract_labels(df_day, trading_dates).items():
        store.append_labels(label_name, table)
        logger.debug(
            f"标签落盘: {label_name}（{table['maturity_status'].iloc[0]}，{len(table)} 行）"
        )

# -*- coding: utf-8 -*-
"""测试 TushareClient 全市场查询自动分页（审计问题3）。

对应 2026-08-09 审计修复：daily_basic / moneyflow / stk_limit
单日全市场行数逼近或超过 TuShare 单次 6000 上限，改为自动分页取全，
避免静默截断造成缺口永久驻留。
"""

from unittest.mock import Mock

import pandas as pd

from src.lazybull.data import TushareClient


def _make_client(query_side_effect):
    """绕过 __init__ 创建真实客户端实例，仅替换 query 为 Mock。

    list 视为分页侧效序列（side_effect），其余视为固定返回值（return_value），
    避免 DataFrame 被 Mock 当作可迭代 side_effect 逐列弹出。
    """
    client = TushareClient.__new__(TushareClient)
    if isinstance(query_side_effect, list):
        client.query = Mock(side_effect=query_side_effect)
    else:
        client.query = Mock(return_value=query_side_effect)
    return client


def _page(n: int, start: int = 0) -> pd.DataFrame:
    """构造 n 行的模拟全市场单页数据。"""
    return pd.DataFrame(
        {
            "ts_code": [f"s{i}" for i in range(start, start + n)],
            "trade_date": ["20250121"] * n,
        }
    )


def test_get_daily_full_market_pagination():
    """daily 全市场查询自动分页（堵住 daily 自身截断源头，审计问题2）。"""
    client = _make_client([_page(6000), _page(2, 6000)])

    result = client.get_daily(trade_date="20250121")

    assert len(result) == 6002
    assert client.query.call_count == 2
    assert client.query.call_args_list[1].kwargs["offset"] == 6000


def test_get_daily_basic_full_market_pagination():
    """全市场查询超过单次上限时自动翻页拼接。"""
    client = _make_client([_page(6000), _page(2, 6000)])

    result = client.get_daily_basic(trade_date="20250121")

    assert len(result) == 6002
    assert client.query.call_count == 2
    # 第二次查询应带上 offset 继续翻页
    assert client.query.call_args_list[1].kwargs["offset"] == 6000
    assert client.query.call_args_list[1].kwargs["limit"] == 6000


def test_get_moneyflow_full_market_pagination():
    """moneyflow 全市场查询自动分页。"""
    client = _make_client([_page(6000), _page(1, 6000)])

    result = client.get_moneyflow(trade_date="20250121")

    assert len(result) == 6001
    assert client.query.call_count == 2


def test_get_stk_limit_full_market_pagination():
    """stk_limit 含指数超 6000 上限，全市场查询自动分页。"""
    client = _make_client([_page(6000), _page(3, 6000)])

    result = client.get_stk_limit(trade_date="20250121")

    assert len(result) == 6003
    assert client.query.call_count == 2


def test_get_daily_basic_single_stock_no_pagination():
    """指定 ts_code 的单股查询不翻页，保持原路径（不传 limit/offset）。"""
    client = _make_client(
        pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20250121"]})
    )

    result = client.get_daily_basic(ts_code="000001.SZ", trade_date="20250121")

    assert len(result) == 1
    _, kwargs = client.query.call_args
    assert "limit" not in kwargs
    assert "offset" not in kwargs


def test_get_daily_basic_full_market_empty():
    """全市场查询返回空（如非交易日）时优雅降级为空 DataFrame。"""
    client = _make_client(pd.DataFrame())

    result = client.get_daily_basic(trade_date="20250121")

    assert result.empty

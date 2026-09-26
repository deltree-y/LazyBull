# -*- coding: utf-8 -*-
"""选股域开关（C 臂域扩展）单元测试。

硬契约（universe/domains.py + 两侧接线）：
1. 域注册表封闭：未知域名 fail-fast，禁止静默回退 main；
2. **两侧同源**：markets 同时作用于训练池（training_core）与回测 universe（backtest）；
   市值上下限只作用于 MLSignal 推理过滤；
3. **main_small 训练池 == main**（主板）⇒ 可 skip-training 复用模型；
   **main_gem 训练池扩展**（+创业板）⇒ 必须重训；
4. 默认（main）与历史行为逐位一致：_build_main_board_codes 默认参数 = 纯主板。
"""

import pandas as pd
import pytest

from src.lazybull.ml.walk_forward.training_core import _build_main_board_codes
from src.lazybull.universe.domains import (
    STOCK_DOMAINS,
    domain_market_whitelist,
    domain_training_pool_unchanged,
    resolve_stock_domain,
)


def _stock_basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ", "300001.SZ", "688001.SH"],
            "market": ["主板", "主板", "创业板", "科创板"],
        }
    )


# ---------------------------------------------------------------- 注册表契约
def test_registry_is_closed():
    assert set(STOCK_DOMAINS) == {"main", "main_small", "main_gem"}


def test_unknown_domain_fails_fast():
    with pytest.raises(ValueError, match="未知选股域"):
        resolve_stock_domain("all_market")
    with pytest.raises(ValueError, match="未知选股域"):
        resolve_stock_domain("")


def test_domain_definitions():
    main = resolve_stock_domain("main")
    assert main["markets"] == ["主板"]
    assert main["min_total_mv_wan"] == 500000.0  # 50 亿
    small = resolve_stock_domain("main_small")
    assert small["markets"] == ["主板"]
    assert small["min_total_mv_wan"] == 250000.0  # 25 亿（唯一差异）
    assert small["max_total_mv_wan"] == main["max_total_mv_wan"]
    gem = resolve_stock_domain("main_gem")
    assert gem["markets"] == ["主板", "创业板"]
    assert gem["min_total_mv_wan"] == main["min_total_mv_wan"]


# ---------------------------------------------------------------- 训练池契约
def test_main_small_training_pool_unchanged():
    """C1（main_small）训练池与 main 相同 ⇒ 可 skip-training 复用生产模型。"""
    assert domain_training_pool_unchanged("main")
    assert domain_training_pool_unchanged("main_small")
    assert not domain_training_pool_unchanged("main_gem")


def test_build_codes_default_is_main_board_only():
    """默认参数 = 纯主板（历史行为逐位一致）。"""
    codes = _build_main_board_codes(_stock_basic())
    assert codes == {"600000.SH", "000001.SZ"}


def test_build_codes_with_gem_market():
    """main_gem 域池含创业板、不含科创板。"""
    codes = _build_main_board_codes(
        _stock_basic(), markets=tuple(domain_market_whitelist("main_gem"))
    )
    assert codes == {"600000.SH", "000001.SZ", "300001.SZ"}
    assert "688001.SH" not in codes


def test_build_codes_empty_domain_raises():
    df = pd.DataFrame({"ts_code": ["688001.SH"], "market": ["科创板"]})
    with pytest.raises(ValueError, match="无法构建域股票池"):
        _build_main_board_codes(df, markets=("创业板",))

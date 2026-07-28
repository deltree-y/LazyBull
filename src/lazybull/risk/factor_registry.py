"""风控因子注册表

提供装饰器风格的因子注册机制，所有风控因子模块通过 @register_risk_factor
自行注册，builder.py 仅需调用 compute_all_risk_factors() 一个入口即可获取全部因子。

使用方式：
    from .factor_registry import register_risk_factor

    @register_risk_factor("downside_vol_20")
    def compute_downside_vol_20(df, daily_adj, market_state, **kwargs):
        ...

架构约束：每个因子的计算函数签名统一为 `fn(df, daily_adj, market_state, **kwargs) -> pd.Series`，
其中 df 是当日截面的 DataFrame（含 ts_code 列），返回与 df 等长的 Series。
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------------------------

_RISK_FACTOR_REGISTRY: Dict[str, Callable[..., pd.Series]] = {}

# 已导入标记：各因子模块首次导入时设为 True，避免重复导入
_IMPORTED_MODULES: Dict[str, bool] = {}

# 风控因子所需的最大历史窗口（交易日数）
# 22 个因子使用 daily_adj，最大窗口为 252（vol_regime_percentile / turnover_percentile）
_MAX_RISK_WINDOW = 252


def register_risk_factor(name: str):
    """装饰器：将函数注册为风控因子计算器。

    Args:
        name: 因子名，同时也是最终写入特征 DataFrame 的列名。

    Example:
        @register_risk_factor("downside_vol_20")
        def compute_downside_vol_20(df, daily_adj, market_state, **kwargs):
            ...
    """

    def decorator(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        if name in _RISK_FACTOR_REGISTRY:
            logger.warning(f"风控因子 '{name}' 被重复注册，将覆盖前一个")
        _RISK_FACTOR_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# 统一入口（builder.py 唯一调用点）
# ---------------------------------------------------------------------------


def _ensure_all_modules_imported():
    """确保所有因子模块已被导入（触发注册）。

    首次调用 compute_all_risk_factors() 时自动执行。
    使用包相对导入，兼容 `src.lazybull` 与 `lazybull` 两种导入路径。
    """
    import importlib

    modules_to_import = [
        "downside_factors",
        "volatility_factors",
        "liquidity_factors",
        "announcement_factors",
    ]

    for mod_name in modules_to_import:
        if mod_name not in _IMPORTED_MODULES:
            try:
                importlib.import_module(f".{mod_name}", package=__package__)
                _IMPORTED_MODULES[mod_name] = True
                logger.debug(f"已加载风控因子模块: {__package__}.{mod_name}")
            except ImportError as e:
                logger.warning(f"无法加载风控因子模块 {mod_name}: {e}")


def _prefilter_daily_adj(
    daily_adj: pd.DataFrame,
    trade_date: str,
    max_window: int = _MAX_RISK_WINDOW,
    trading_dates: Optional[list] = None,
    trade_date_idx: Optional[int] = None,
) -> pd.DataFrame:
    """预过滤 daily_adj：仅保留 trade_date 及之前、每只股票最多 max_window 条记录。

    将 ~1300 万行缩减到 ~5000×252≈126 万行，避免 22 个因子各自重复做
    全量过滤 + groupby.tail 操作。

    当提供 trading_dates + trade_date_idx 时，使用 O(1) isin 日期窗口过滤
    （直接定位最近 max_window 个交易日），跳过昂贵的 groupby.tail 扫描。
    否则回退到 groupby.tail 路径。

    前置条件：daily_adj 必须已按 (ts_code, trade_date) 排序。
    """
    if trading_dates is not None and trade_date_idx is not None:
        # 快速路径：用双边界比较定位最近 max_window 个交易日（纯向量化，无哈希开销）
        start_date = trading_dates[max(0, trade_date_idx - max_window + 1)]
        mask = (daily_adj["trade_date"] >= start_date) & (daily_adj["trade_date"] <= trade_date)
        return daily_adj.loc[mask]

    # 回退路径：mask + groupby.tail（无 trading_dates 信息时使用）
    mask = daily_adj["trade_date"] <= trade_date
    filtered = daily_adj.loc[mask]
    filtered = filtered.groupby("ts_code", sort=False).tail(max_window)
    return filtered


def compute_all_risk_factors(
    df: pd.DataFrame,
    daily_adj: Optional[pd.DataFrame] = None,
    market_state: Optional[Dict[str, Any]] = None,
    exclude: Optional[set] = None,
    **kwargs,
) -> Dict[str, pd.Series]:
    """计算全部已注册的风控因子（builder.py 唯一调用入口）。

    Args:
        df: 当日截面 DataFrame，必须包含 ts_code 列
        daily_adj: 全量后复权日线数据（部分因子需要历史窗口计算）
        market_state: 市场状态字典（mkt_ret / mkt_vol / mkt_drawdown 等）
        exclude: 需跳过的因子名集合（已由批量预计算提供的因子）
        **kwargs: 传递给各因子计算函数的额外参数

    Returns:
        {因子名: Series} 字典，每个 Series 与 df 等长、索引对齐
    """
    _ensure_all_modules_imported()

    if not _RISK_FACTOR_REGISTRY:
        logger.debug("风控因子注册表为空，跳过计算")
        return {}

    # ---- 性能优化：预过滤 daily_adj 到最大窗口 ----
    # 使用 trading_dates 做双边界比较过滤，替代昂贵的 groupby.tail 扫描
    trade_date = kwargs.get("trade_date", "")
    _td_list = kwargs.get("trading_dates", None)
    _td_idx = kwargs.get("trade_date_idx", None)
    if daily_adj is not None and trade_date and "ts_code" in daily_adj.columns:
        daily_adj = _prefilter_daily_adj(
            daily_adj,
            trade_date,
            _MAX_RISK_WINDOW,
            trading_dates=_td_list,
            trade_date_idx=_td_idx,
        )

    logger.debug(f"开始计算 {len(_RISK_FACTOR_REGISTRY)} 个风控因子...")
    results: Dict[str, pd.Series] = {}

    for name, fn in _RISK_FACTOR_REGISTRY.items():
        if exclude and name in exclude:
            continue
        try:
            series = fn(df, daily_adj=daily_adj, market_state=market_state, **kwargs)
            if isinstance(series, pd.Series):
                results[name] = series
            else:
                logger.warning(f"风控因子 '{name}' 返回类型异常: {type(series)}，已跳过")
        except Exception as e:
            logger.error(f"风控因子 '{name}' 计算失败: {e}", exc_info=True)

    logger.debug(f"风控因子计算完成: {len(results)}/{len(_RISK_FACTOR_REGISTRY)} 个成功")
    return results


def get_registered_factor_names() -> list:
    """返回所有已注册的因子名列表（用于调试/检查）。"""
    _ensure_all_modules_imported()
    return sorted(_RISK_FACTOR_REGISTRY.keys())


def reset_registry():
    """重置注册表（仅用于测试）。"""
    _RISK_FACTOR_REGISTRY.clear()
    _IMPORTED_MODULES.clear()

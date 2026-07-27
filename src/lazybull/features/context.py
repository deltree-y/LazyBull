"""特征构建上下文 —— 收敛 build_features_for_day() 的 20+ 参数。

将散列参数统一打包为 dataclass，方便跨模块传递与并行化 pickle。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class FeatureContext:
    """单日特征构建所需全部输入数据与控制开关。

    调用方可继续使用 FeatureBuilder.build_features_for_day(**kwargs) 原签名；
    该方法内部会将 kwargs 打包为 FeatureContext 再委托给无状态函数。
    """

    # ── 核心必填 ──
    trade_date: str
    trade_cal: pd.DataFrame
    daily_data: pd.DataFrame
    adj_factor: pd.DataFrame
    stock_basic: pd.DataFrame

    # ── 可选行情/辅助数据 ──
    daily_basic_data: Optional[pd.DataFrame] = None
    moneyflow_data: Optional[pd.DataFrame] = None
    suspend_info: Optional[pd.DataFrame] = None
    limit_info: Optional[pd.DataFrame] = None
    shenwan_industry: Optional[pd.DataFrame] = None

    # ── 中性化开关 ──
    apply_industry_neutralization: bool = False
    apply_size_neutralization: bool = False

    # ── 可选因子数据（当日已前向填充的截面）──
    fundamental_data: Optional[pd.DataFrame] = None
    margin_data: Optional[pd.DataFrame] = None
    holder_data: Optional[pd.DataFrame] = None
    earnings_data: Optional[pd.DataFrame] = None
    cyq_perf_data: Optional[pd.DataFrame] = None
    express_data: Optional[pd.DataFrame] = None
    fund_portfolio_data: Optional[pd.DataFrame] = None
    north_flow_data: Optional[Dict[str, float]] = None
    lhb_data: Optional[pd.DataFrame] = None
    consensus_data: Optional[pd.DataFrame] = None
    cashflow_data: Optional[pd.DataFrame] = None
    consensus_revision_data: Optional[pd.DataFrame] = None

    # ── 预计算缓存引用（并行模式下从主进程传入，只读共享）──
    daily_adj_dict: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = field(default=None, repr=False)
    market_state_cache: Optional[pd.DataFrame] = field(default=None, repr=False)
    trading_dates_list: Optional[List[str]] = field(default=None, repr=False)
    trading_date_index: Optional[Dict[str, int]] = field(default=None, repr=False)
    daily_adj_precomputed: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ── 配置（从 FeatureBuilder 透传）──
    horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    horizon: int = 20
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    require_label: bool = True
    label_filter_mode: str = "all"
    min_list_days: int = 365
    shenwan_level: str = "l2"
    verbose: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs) -> "FeatureContext":
        """从 build_features_for_day() 的关键字参数构建上下文。"""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        ctx_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        return cls(**ctx_kwargs)

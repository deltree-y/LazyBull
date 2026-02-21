"""因子库模块

提供可复用的技术指标、K线形态、波动率、行业等因子计算
"""

from .technical_indicators import (
    calculate_rsi,
    calculate_kdj,
    calculate_macd,
    calculate_bollinger_bands,
)

from .candlestick import (
    calculate_amplitude,
    calculate_shadows,
)

from .volatility import (
    calculate_volatility,
)

from .industry import (
    add_industry_features,
    calculate_industry_alpha_windows,
    generate_industry_encoding,
)

from .momentum import (
    calculate_acceleration,
)

from .volume import (
    calculate_volume_burst,
)

from .normalization import (
    cross_sectional_zscore,
    industry_neutralization,
)

from .market_state import compute_market_state_features, precompute_market_state_features
from .precompute_technical_factors import precompute_technical_factors

__all__ = [
    # 技术指标
    'calculate_rsi',
    'calculate_kdj',
    'calculate_macd',
    'calculate_bollinger_bands',
    # K线形态
    'calculate_amplitude',
    'calculate_shadows',
    # 波动率
    'calculate_volatility',
    # 行业
    'add_industry_features',
    'calculate_industry_alpha_windows',
    'generate_industry_encoding',
    # 动量
    'calculate_acceleration',
    # 量能
    'calculate_volume_burst',
    # 标准化与中性化
    'cross_sectional_zscore',
    'industry_neutralization',
    # 市场状态特征
    'compute_market_state_features',
    'precompute_market_state_features',
    # 技术指标与波动率批量预计算
    'precompute_technical_factors',
]

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
from .returns import compute_ret_1
from .fundamental import build_fundamental_lookup_by_date, FUNDA_COLS
from .cashflow_quality import build_cashflow_quality_lookup_by_date, CASHFLOW_COLS
from .consensus_revision import build_consensus_revision_lookup_by_date, CONSENSUS_REVISION_COLS

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
    # 收益率构造工具
    'compute_ret_1',
    # 基本面因子
    'build_fundamental_lookup_by_date',
    'FUNDA_COLS',
    # 现金流质量因子
    'build_cashflow_quality_lookup_by_date',
    'CASHFLOW_COLS',
    # 一致预期修正因子
    'build_consensus_revision_lookup_by_date',
    'CONSENSUS_REVISION_COLS',
]

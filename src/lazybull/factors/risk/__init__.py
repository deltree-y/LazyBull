"""风控因子子包（src/lazybull/factors/risk）

风控模型专用的因子模块，统一归入 factors/risk 子目录，与选股因子库
（src/lazybull/factors 根目录）保持命名空间隔离。

模块说明：
  - factor_registry       : 因子注册表 + compute_all_risk_factors() 统一入口
  - downside_factors      : 下行风险因子（VaR/CVaR/偏度/峰度/下行β 等）
  - volatility_factors    : 波动结构因子（Parkinson/波动率之波动/GARCH 等）
  - liquidity_factors     : 流动性风险因子（Amihud/量价背离/缩量等）
  - announcement_factors  : 公告类低频因子（质押/解禁/大宗/融券，三层加工）
  - derived_factors       : 衍生因子（momentum_decay/earnings_yield 等）
  - position_features     : 持仓上下文特征（持有天数/浮盈/组合内排名等）
"""

from .factor_registry import (
    compute_all_risk_factors,
    get_registered_factor_names,
    register_risk_factor,
)

__all__ = [
    'compute_all_risk_factors',
    'get_registered_factor_names',
    'register_risk_factor',
]

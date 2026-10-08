"""回测引擎目标持仓/分批/买入阈值参数推导（P2a-T4，engine.py 函数级拆分件）。

来源：旧 ``backtest/engine.py`` 的 ``_get_target_position_count`` /
``_get_tranche_target_count`` / ``_get_tranche_capital_fraction`` /
``_get_min_buy_value_threshold``（T0 规划 §3.3）。方法转函数、显式入参
替代 self；公开语义、数值操作顺序、默认值逐字保留。

依赖处置（§3.5 登记表）：
- 分批槽位/预算比例委托 v2 ``common.rules.stagger``（T1 已迁，改指 v2 路径）；
- 最小买入阈值委托 ``core/decision/sizing``（T3 已归位，改指 v2 复制件；
  原 T4 期的旧 ``trading.sizing`` 只读过渡依赖随之去除）。
"""

from typing import Dict, Optional

from src.lazybull.v2.common.rules.stagger import (
    get_tranche_capital_fraction as _shared_tranche_capital_fraction,
)
from src.lazybull.v2.common.rules.stagger import (
    get_tranche_target_count as _shared_tranche_target_count,
)
from src.lazybull.v2.core.decision.sizing import compute_min_buy_value_threshold


def get_target_position_count(signal, positions: Dict[str, Dict]) -> int:
    """获取组合当前期望的目标持仓数。

    Args:
        signal: 信号生成器（读取其 top_n 属性）
        positions: 当前持仓 {股票代码: 持仓信息}

    Returns:
        目标持仓数；信号未配置有效 top_n 时回退为当前持仓数
    """
    target_n = getattr(signal, "top_n", None)
    if isinstance(target_n, int) and target_n > 0:
        return target_n
    return len(positions)


def get_tranche_target_count(
    tranche_idx: int,
    stagger_tranches: int,
    signal,
    positions: Dict[str, Dict],
    target_count: Optional[int] = None,
) -> int:
    """获取当前批次应占用的目标持仓槽位数（委托 common.rules.stagger 共享实现）。

    Args:
        tranche_idx: 批次索引
        stagger_tranches: 分批调仓批次数
        signal: 信号生成器（target_count 缺省时用于推导目标持仓数）
        positions: 当前持仓（target_count 缺省时用于推导目标持仓数）
        target_count: 目标持仓数；None 时按信号/当前持仓推导
    """
    if target_count is None:
        target_count = get_target_position_count(signal, positions)
    return _shared_tranche_target_count(tranche_idx, target_count, stagger_tranches)


def get_tranche_capital_fraction(
    tranche_idx: int,
    stagger_tranches: int,
    signal,
    positions: Dict[str, Dict],
) -> float:
    """获取当前批次占组合总资产的预算比例（委托 common.rules.stagger 共享实现）。

    Args:
        tranche_idx: 批次索引
        stagger_tranches: 分批调仓批次数
        signal: 信号生成器（用于推导目标持仓数）
        positions: 当前持仓（用于推导目标持仓数）
    """
    target_count = get_target_position_count(signal, positions)
    return _shared_tranche_capital_fraction(tranche_idx, target_count, stagger_tranches)


def get_min_buy_value_threshold(
    min_buy_value_ratio: float,
    total_assets: float,
    target_count: int,
) -> float:
    """计算最小买入后市值阈值（与纸面交易共用 trading.sizing 口径）。

    原方法内部以 ``_calculate_portfolio_value(date)`` 与
    ``_get_target_position_count()`` 取这两个输入；纯函数形态改为显式入参，
    由调用方（组装体）按同一口径供给。

    Args:
        min_buy_value_ratio: 最小持仓市值占平均仓位市值比例（0=关闭）
        total_assets: 组合总市值（原 ``_calculate_portfolio_value(date)`` 口径）
        target_count: 目标持仓数（原 ``_get_target_position_count()`` 口径）
    """
    ratio = float(min_buy_value_ratio or 0.0)
    if ratio <= 0:
        return 0.0
    return compute_min_buy_value_threshold(
        total_assets=float(total_assets),
        target_count=int(target_count or 0),
        ratio=ratio,
    )

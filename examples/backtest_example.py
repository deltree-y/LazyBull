"""回测引擎使用示例

展示如何使用重构后的回测引擎：
1. 价格口径分离（成交 vs 绩效）
2. 风险预算/波动率缩放
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from lazybull.backtest import BacktestEngine
from lazybull.common.cost import CostModel
from lazybull.signals.base import Signal
from lazybull.universe.base import Universe


class SimpleUniverse(Universe):
    """简单股票池：返回固定股票列表"""
    
    def __init__(self, stocks):
        self.stocks = stocks
    
    def get_stocks(self, date):
        return self.stocks


class EqualWeightSignal(Signal):
    """等权信号：均分资金到所有股票"""
    
    def generate(self, date, universe, data):
        if not universe:
            return {}
        return {stock: 1.0 / len(universe) for stock in universe}


def generate_sample_data():
    """生成模拟价格数据
    
    包含不复权价格（close）和后复权价格（close_adj）
    模拟真实场景：除权除息后不复权价格会跳空，但后复权价格连续
    """
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for i, date in enumerate(dates):
        for j, stock in enumerate(stocks):
            # 不复权价格：在除权日会跳空
            close = 10.0
            if i == 50:  # 模拟除权
                close = 8.0  # 除权后价格下跌
            
            # 后复权价格：连续增长，适合计算收益率
            close_adj = 10.0 + i * 0.05 + j * 0.02
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': close,
                'close_adj': close_adj
            })
    
    return pd.DataFrame(data)


def main():
    """主函数：运行回测示例"""
    
    print("=" * 60)
    print("回测引擎使用示例")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    price_data = generate_sample_data()
    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-01', periods=100, freq='B')]
    
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    universe = SimpleUniverse(stocks)
    signal = EqualWeightSignal()
    
    # 2. 运行回测（不启用风险预算）
    print("\n2. 运行回测（不启用风险预算）...")
    engine1 = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        cost_model=CostModel(),
        rebalance_freq="M",
        holding_period=20,
        enable_risk_budget=False,
        verbose=False
    )
    
    nav1 = engine1.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data
    )
    
    print(f"   最终净值: {nav1['nav'].iloc[-1]:.4f}")
    print(f"   总收益率: {nav1['return'].iloc[-1]:.2%}")
    print(f"   交易笔数: {len(engine1.get_trades())} 笔")
    
    # 3. 运行回测（启用风险预算）
    print("\n3. 运行回测（启用风险预算）...")
    engine2 = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        cost_model=CostModel(),
        rebalance_freq="M",
        holding_period=20,
        enable_risk_budget=True,  # 启用风险预算
        vol_window=20,
        verbose=False
    )
    
    nav2 = engine2.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data
    )
    
    print(f"   最终净值: {nav2['nav'].iloc[-1]:.4f}")
    print(f"   总收益率: {nav2['return'].iloc[-1]:.2%}")
    print(f"   交易笔数: {len(engine2.get_trades())} 笔")
    
    # 4. 查看交易记录（包含绩效价格字段）
    print("\n4. 查看交易记录...")
    trades2 = engine2.get_trades()
    sell_trades = trades2[trades2['action'] == 'sell']
    
    if len(sell_trades) > 0:
        print("\n   示例卖出交易（前3笔）:")
        for idx, trade in sell_trades.head(3).iterrows():
            print(f"\n   交易 #{idx + 1}:")
            print(f"     股票: {trade['stock']}")
            print(f"     日期: {trade['date'].date()}")
            print(f"     卖出成交价: {trade['price']:.2f}")
            print(f"     买入绩效价格: {trade['buy_pnl_price']:.2f}")
            print(f"     卖出绩效价格: {trade['sell_pnl_price']:.2f}")
            print(f"     绩效收益率: {trade['pnl_profit_pct']:.2%}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("\n主要特性:")
    print("  ✓ 价格口径分离：成交使用不复权价格，绩效使用后复权价格")
    print("  ✓ 风险预算：根据波动率动态调整权重")
    print("  ✓ 高性能：使用 MultiIndex 代替嵌套字典")
    print("  ✓ 向后兼容：保留原有 API，默认不启用风险预算")


if __name__ == "__main__":
    main()

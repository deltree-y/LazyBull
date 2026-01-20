"""测试止损功能集成到回测引擎

验证止损触发、T+1 卖出执行、交易记录等功能
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine, BacktestEngineML
from src.lazybull.common.cost import CostModel
from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse
from src.lazybull.risk.stop_loss import StopLossConfig


class MockMLModel:
    """模拟 ML 模型（用于测试）"""
    
    def predict(self, X):
        """返回模拟预测值"""
        if len(X.columns) > 0:
            return X.iloc[:, 0].values * 0.1
        return np.random.randn(len(X))


@pytest.fixture
def temp_models_dir():
    """创建临时模型目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def trained_model(temp_models_dir):
    """创建一个训练好的模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    model = MockMLModel()
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100}
    )
    
    return temp_models_dir, version


@pytest.fixture
def mock_stock_basic():
    """模拟股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'symbol': ['000001', '000002', '600000'],
        'name': ['股票A', '股票B', '股票C'],
        'market': ['主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101']
    })


def test_stop_loss_t1_execution(trained_model, mock_stock_basic):
    """测试止损触发后的 T+1 卖出执行
    
    验证点：
    1. T 日触发止损，记录到待卖出队列
    2. T+1 日执行卖出
    3. 交易记录中包含 sell_type='stop_loss'
    4. 止损监控器状态被正确清理
    """
    models_dir, version = trained_model
    
    # 创建 ML 信号生成器
    signal = MLSignal(
        top_n=2,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal"
    )
    
    # 创建股票池
    universe = BasicUniverse(
        stock_basic=mock_stock_basic,
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    # 创建止损配置（20% 回撤止损）
    stop_loss_config = StopLossConfig(
        enabled=True,
        drawdown_pct=20.0,
        trailing_stop_enabled=False,
        consecutive_limit_down_days=2,
        post_trigger_action='hold_cash'
    )
    
    # 创建价格数据
    # 场景：000001.SZ 在第3天触发止损（跌幅超过20%），应在第4天卖出
    dates = ['20230601', '20230602', '20230605', '20230606', '20230607']
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    price_data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            if stock == '000001.SZ':
                # 000001.SZ: 买入价10.0，第3天跌到7.5（跌幅25%，触发止损）
                if i == 0:
                    close_price = 10.0
                elif i == 1:
                    close_price = 10.0
                elif i == 2:
                    close_price = 7.5  # 触发止损（T日）
                elif i == 3:
                    close_price = 7.3  # T+1日卖出
                else:
                    close_price = 7.0
            elif stock == '000002.SZ':
                # 000002.SZ: 正常持有，不触发止损
                close_price = 10.0 + i * 0.2
            else:
                # 600000.SH: 正常持有
                close_price = 10.0 + i * 0.1
            
            price_data.append({
                'ts_code': stock,
                'trade_date': date,
                'close': close_price,
                'close_adj': close_price,
                'high': close_price + 0.5,
                'low': close_price - 0.5,
                'pct_chg': 0.0,
                'is_limit_down': False,
                'is_limit_up': False,
                'is_suspended': False,
                'vol': 1000000
            })
    
    price_data_df = pd.DataFrame(price_data)
    
    # 创建特征数据（第一天选中 000001.SZ 和 000002.SZ）
    features_by_date = {
        '20230601': pd.DataFrame({
            'ts_code': stocks,
            'f1': [20, 18, 10],  # 000001.SZ 和 000002.SZ 排名前2
            'f2': np.random.randn(3),
            'f3': np.random.randn(3)
        })
    }
    
    # 创建 ML 回测引擎（启用止损）
    engine = BacktestEngineML(
        features_by_date=features_by_date,
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=20,  # 较长的调仓周期，确保不会正常卖出
        holding_period=20,
        stop_loss_config=stop_loss_config,
        verbose=True
    )
    
    # 准备交易日列表
    trading_dates = [pd.Timestamp(d) for d in dates]
    
    # 运行回测
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data_df
    )
    
    # 获取交易记录
    trades = engine.get_trades()
    
    # 验证1：应该有买入和卖出记录
    assert len(trades) > 0
    buy_trades = trades[trades['action'] == 'buy']
    sell_trades = trades[trades['action'] == 'sell']
    
    print(f"\n买入交易: {len(buy_trades)} 笔")
    print(f"卖出交易: {len(sell_trades)} 笔")
    
    # 验证2：应该在 20230602（T+1）买入 000001.SZ 和 000002.SZ
    assert len(buy_trades) == 2
    assert set(buy_trades['stock'].tolist()) == {'000001.SZ', '000002.SZ'}
    
    # 验证3：应该有止损卖出记录
    stop_loss_sells = sell_trades[sell_trades.get('sell_type') == 'stop_loss']
    assert len(stop_loss_sells) > 0
    
    # 验证4：止损卖出应该是 000001.SZ
    stop_loss_stock = stop_loss_sells.iloc[0]['stock']
    assert stop_loss_stock == '000001.SZ'
    
    # 验证5：止损卖出日期应该是 20230606（T+1日）
    stop_loss_date = stop_loss_sells.iloc[0]['date']
    assert stop_loss_date == pd.Timestamp('2023-06-06')
    
    # 验证6：卖出记录应该包含止损原因
    assert 'sell_reason' in stop_loss_sells.columns
    assert stop_loss_sells.iloc[0]['sell_reason'] is not None
    
    # 验证7：止损监控器状态应该被清理（000001.SZ 已卖出）
    assert stop_loss_stock not in engine.positions
    
    # 验证8：待止损卖出队列应该为空
    assert len(engine.pending_stop_loss_sells) == 0
    
    print("\n✓ 止损触发和 T+1 卖出验证通过")
    print(f"✓ 止损触发日期: T日（20230605）")
    print(f"✓ 止损执行日期: T+1日（20230606）")
    print(f"✓ 止损原因: {stop_loss_sells.iloc[0]['sell_reason']}")
    print(f"✓ 触发类型: {stop_loss_sells.iloc[0].get('trigger_type', 'N/A')}")


def test_stop_loss_no_duplicate_trigger(trained_model, mock_stock_basic):
    """测试止损触发后不会重复生成卖出信号
    
    验证点：
    1. T 日触发止损后，加入待卖出队列
    2. T 日之后的日期不会重复触发
    3. T+1 日执行卖出后，从队列中移除
    """
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=1,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal"
    )
    
    universe = BasicUniverse(
        stock_basic=mock_stock_basic,
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    stop_loss_config = StopLossConfig(
        enabled=True,
        drawdown_pct=20.0,
        trailing_stop_enabled=False,
        consecutive_limit_down_days=2,
        post_trigger_action='hold_cash'
    )
    
    # 创建价格数据：000001.SZ 持续下跌
    dates = ['20230601', '20230602', '20230605', '20230606', '20230607']
    price_data = []
    for i, date in enumerate(dates):
        if i == 0:
            close_price = 10.0
        elif i == 1:
            close_price = 10.0
        elif i == 2:
            close_price = 7.5  # T 日触发止损
        elif i == 3:
            close_price = 7.0  # T+1 日执行卖出
        else:
            close_price = 6.5  # 继续下跌
        
        price_data.append({
            'ts_code': '000001.SZ',
            'trade_date': date,
            'close': close_price,
            'close_adj': close_price,
            'high': close_price + 0.5,
            'low': close_price - 0.5,
            'pct_chg': 0.0,
            'is_limit_down': False,
            'is_limit_up': False,
            'is_suspended': False,
            'vol': 1000000
        })
    
    price_data_df = pd.DataFrame(price_data)
    
    features_by_date = {
        '20230601': pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'f1': [20],
            'f2': [1.0],
            'f3': [1.0]
        })
    }
    
    engine = BacktestEngineML(
        features_by_date=features_by_date,
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=20,
        holding_period=20,
        stop_loss_config=stop_loss_config,
        verbose=True
    )
    
    trading_dates = [pd.Timestamp(d) for d in dates]
    
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data_df
    )
    
    trades = engine.get_trades()
    
    # 验证：只应该有1笔止损卖出（不会重复触发）
    stop_loss_sells = trades[(trades['action'] == 'sell') & (trades.get('sell_type') == 'stop_loss')]
    assert len(stop_loss_sells) == 1
    
    print("\n✓ 止损不会重复触发验证通过")
    print(f"✓ 止损卖出次数: {len(stop_loss_sells)} 次（符合预期）")


def test_backward_compatibility_no_stop_loss(trained_model, mock_stock_basic):
    """测试向后兼容性：不启用止损时，行为不变
    
    验证点：
    1. 不传入 stop_loss_config 时，止损功能不启用
    2. 回测正常运行
    3. 交易记录不包含止损相关字段
    """
    models_dir, version = trained_model
    
    signal = MLSignal(
        top_n=2,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal"
    )
    
    universe = BasicUniverse(
        stock_basic=mock_stock_basic,
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    # 创建价格数据
    dates = ['20230601', '20230602', '20230605']
    stocks = ['000001.SZ', '000002.SZ']
    
    price_data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            price_data.append({
                'ts_code': stock,
                'trade_date': date,
                'close': 10.0,
                'close_adj': 10.0,
                'high': 11.0,
                'low': 9.0,
                'pct_chg': 0.0,
                'is_limit_down': False,
                'is_limit_up': False,
                'is_suspended': False,
                'vol': 1000000
            })
    
    price_data_df = pd.DataFrame(price_data)
    
    features_by_date = {
        '20230601': pd.DataFrame({
            'ts_code': stocks,
            'f1': [20, 18],
            'f2': np.random.randn(2),
            'f3': np.random.randn(2)
        })
    }
    
    # 不传入 stop_loss_config
    engine = BacktestEngineML(
        features_by_date=features_by_date,
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=1,
        verbose=True
    )
    
    trading_dates = [pd.Timestamp(d) for d in dates]
    
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data_df
    )
    
    # 验证：止损监控器应该为 None
    assert engine.stop_loss_monitor is None
    
    # 验证：待止损卖出队列应该为空
    assert len(engine.pending_stop_loss_sells) == 0
    
    # 验证：回测正常完成
    assert len(nav_curve) == 3
    
    print("\n✓ 向后兼容性验证通过")
    print("✓ 不启用止损时，回测正常运行")

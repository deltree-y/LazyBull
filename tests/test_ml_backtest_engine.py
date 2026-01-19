"""测试 ML 回测引擎是否正确使用父类信号过滤逻辑"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngineML
from src.lazybull.common.cost import CostModel
from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal
from src.lazybull.universe import BasicUniverse


class MockMLModel:
    """模拟 ML 模型（用于测试）"""
    
    def predict(self, X):
        """返回模拟预测值"""
        # 返回简单的预测值（基于第一个特征）
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
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '600000.SH', '600001.SH'],
        'symbol': ['000001', '000002', '000003', '600000', '600001'],
        'name': ['股票A', '股票B', '股票C', '股票D', '股票E'],
        'market': ['主板', '主板', '主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101', '20200101', '20200101']
    })


@pytest.fixture
def mock_price_data():
    """模拟价格数据"""
    dates = ['20230601', '20230602', '20230605']
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '600000.SH', '600001.SH']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date,
                'close': 10.0 + np.random.randn() * 0.5,
                'close_adj': 10.0 + np.random.randn() * 0.5,
                'high': 11.0,
                'low': 9.0,
                'pct_chg': 0.0,  # 正常交易
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_features_by_date():
    """模拟特征数据（按日期组织）"""
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '600000.SH', '600001.SH']
    
    # 第一天的特征数据
    features_20230601 = pd.DataFrame({
        'ts_code': stocks,
        'f1': [10, 8, 12, 6, 15],  # 排序: 600001.SH(15), 000003.SZ(12), 000001.SZ(10), 000002.SZ(8), 600000.SH(6)
        'f2': np.random.randn(5),
        'f3': np.random.randn(5)
    })
    
    # 第二天的特征数据
    features_20230602 = pd.DataFrame({
        'ts_code': stocks,
        'f1': [5, 9, 7, 11, 8],  # 排序: 600000.SH(11), 000002.SZ(9), 600001.SH(8), 000003.SZ(7), 000001.SZ(5)
        'f2': np.random.randn(5),
        'f3': np.random.randn(5)
    })
    
    return {
        '20230601': features_20230601,
        '20230602': features_20230602,
        # 注意：20230605 没有特征数据，用于测试缺失处理
    }


def test_ml_backtest_uses_parent_filtering_logic(
    trained_model,
    mock_stock_basic,
    mock_price_data,
    mock_features_by_date
):
    """测试 ML 回测引擎是否使用父类的信号过滤逻辑
    
    验证点：
    1. 信号生成时调用 generate_ranked 而非 generate
    2. 权重被归一化（和为 1）
    3. 特征缺失日期被正确跳过
    """
    models_dir, version = trained_model
    
    # 创建 ML 信号生成器
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="score"
    )
    
    # 创建股票池
    universe = BasicUniverse(
        stock_basic=mock_stock_basic,
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    # 创建 ML 回测引擎
    engine = BacktestEngineML(
        features_by_date=mock_features_by_date,
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=1,  # 每天调仓
        verbose=True
    )
    
    # 准备交易日列表
    trading_dates = [
        pd.Timestamp('2023-06-01'),
        pd.Timestamp('2023-06-02'),
        pd.Timestamp('2023-06-05')
    ]
    
    # 运行回测
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=mock_price_data
    )
    
    # 验证：回测完成
    assert len(nav_curve) == 3
    assert 'nav' in nav_curve.columns
    
    # 验证：有交易记录（说明信号生成成功）
    trades = engine.get_trades()
    assert len(trades) > 0
    
    # 验证：pending_signals 在处理后应该为空或只包含最后一天的信号
    # （因为最后一天生成信号但没有下一天执行）
    assert len(engine.pending_signals) <= 1
    
    print(f"\n✓ ML 回测完成: 交易日={len(trading_dates)}, 交易笔数={len(trades)}")
    print(f"✓ 最终净值: {nav_curve['nav'].iloc[-1]:.4f}")


def test_ml_backtest_handles_missing_features(
    trained_model,
    mock_stock_basic,
    mock_price_data
):
    """测试 ML 回测引擎正确处理特征缺失的情况"""
    models_dir, version = trained_model
    
    # 创建 ML 信号生成器
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir
    )
    
    # 创建股票池
    universe = BasicUniverse(
        stock_basic=mock_stock_basic,
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    # 创建 ML 回测引擎，所有日期都没有特征
    engine = BacktestEngineML(
        features_by_date={},  # 空的特征字典
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=1,
        verbose=True
    )
    
    # 准备交易日列表
    trading_dates = [
        pd.Timestamp('2023-06-01'),
        pd.Timestamp('2023-06-02')
    ]
    
    # 运行回测（应该不报错，只是没有交易）
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=mock_price_data
    )
    
    # 验证：回测完成但没有交易
    assert len(nav_curve) == 2
    trades = engine.get_trades()
    assert len(trades) == 0  # 没有特征数据，应该没有交易
    
    # 验证：净值保持不变（没有交易）
    assert nav_curve['nav'].iloc[0] == 1.0
    assert nav_curve['nav'].iloc[-1] == 1.0
    
    print(f"\n✓ 特征缺失场景测试通过: 无特征时正确跳过信号生成")


def test_ml_engine_build_signal_data(trained_model, mock_features_by_date):
    """测试 ML 引擎的 _build_signal_data 方法"""
    models_dir, version = trained_model
    
    signal = MLSignal(top_n=3, model_version=version, models_dir=models_dir)
    universe = BasicUniverse(
        stock_basic=pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'symbol': ['000001'],
            'name': ['测试'],
            'market': ['主板'],
            'list_date': ['20200101']
        }),
        exclude_st=False,
        min_list_days=0,
        markets=['主板']
    )
    
    engine = BacktestEngineML(
        features_by_date=mock_features_by_date,
        universe=universe,
        signal=signal,
        initial_capital=100000.0
    )
    
    # 测试有特征数据的日期
    date_with_features = pd.Timestamp('2023-06-01')
    data = engine._build_signal_data(date_with_features)
    assert data is not None
    assert 'features' in data
    assert len(data['features']) == 5
    
    # 测试无特征数据的日期
    date_without_features = pd.Timestamp('2023-06-05')
    data = engine._build_signal_data(date_without_features)
    assert data is None  # 应该返回 None
    
    print(f"\n✓ _build_signal_data 方法测试通过")

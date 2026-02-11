"""测试多 horizon 标签功能"""

import pandas as pd
import pytest
import json
from pathlib import Path

from src.lazybull.features import FeatureBuilder


@pytest.fixture
def mock_trade_cal():
    """模拟交易日历（30 个交易日）"""
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    return pd.DataFrame({
        'exchange': ['SSE'] * len(dates),
        'cal_date': dates.strftime('%Y%m%d').tolist(),
        'is_open': [1] * len(dates)
    })


@pytest.fixture
def mock_stock_basic():
    """模拟股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'name': ['平安银行', '万科A', '浦发银行'],
        'list_date': ['20100101', '20100101', '20100101']
    })


@pytest.fixture
def mock_daily_data():
    """模拟日线行情数据（30 天）"""
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for i, stock in enumerate(stocks):
            # 模拟价格：每天随机波动
            base_price = 10.0 + i
            close = base_price * (1 + 0.01 * ((date.day + i) % 10 - 5))  # -5% 到 +5% 波动
            pre_close = base_price
            pct_chg = ((close - pre_close) / pre_close) * 100
            
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'close': close,
                'close_adj': close,  # clean 层已包含复权价格
                'pre_close': pre_close,
                'pct_chg': pct_chg,
                'vol': 1000000,
                'amount': 1000000 * close,
                'is_st': 0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'list_days': 5000,
                'tradable': 1
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_adj_factor():
    """模拟复权因子"""
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'adj_factor': 1.0
            })
    
    return pd.DataFrame(data)


class TestMultiHorizonLabels:
    """测试多 horizon 标签功能"""
    
    def test_feature_builder_init_with_horizons(self):
        """测试 FeatureBuilder 初始化支持 horizons 参数"""
        builder = FeatureBuilder(horizons=[5, 10, 20])
        
        assert builder.horizons == [5, 10, 20]
        assert builder.horizon == 5  # 默认使用第一个
    
    def test_feature_builder_default_horizons(self):
        """测试 FeatureBuilder 默认 horizons"""
        builder = FeatureBuilder()
        
        assert builder.horizons == [5, 10, 20]
    
    def test_feature_builder_backward_compatible(self):
        """测试向后兼容：使用旧的 horizon 参数"""
        builder = FeatureBuilder(horizon=7, horizons=[7])
        
        # 应使用 horizons 参数，horizon 也被设置
        assert builder.horizons == [7]
        assert builder.horizon == 7
    
    def test_build_features_multiple_labels(
        self,
        mock_trade_cal,
        mock_stock_basic,
        mock_daily_data,
        mock_adj_factor
    ):
        """测试特征构建生成多个 horizon 的标签"""
        builder = FeatureBuilder(horizons=[5, 10, 20])
        
        # 构建第 5 个交易日的特征（确保有足够的历史和未来数据）
        # 需要 t+20 的数据，所以选择 30-20-5 = 第 5 个交易日
        trade_date = mock_trade_cal.iloc[5]['cal_date']
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic
        )
        
        # 验证包含所有 horizon 的标签
        assert 'y_ret_5' in features.columns
        assert 'y_ret_10' in features.columns
        assert 'y_ret_20' in features.columns
        
        # 验证标签不为空（至少 y_ret_5 应该有值）
        assert features['y_ret_5'].notna().sum() > 0
    
    def test_labels_calculation_correctness(
        self,
        mock_trade_cal,
        mock_stock_basic,
        mock_daily_data,
        mock_adj_factor
    ):
        """测试标签计算的正确性"""
        builder = FeatureBuilder(horizons=[5])
        
        # 使用第 5 个交易日
        trade_date = mock_trade_cal.iloc[5]['cal_date']
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic
        )
        
        # 手动验证第一只股票的标签
        stock = features.iloc[0]['ts_code']
        
        # 获取 t 时刻价格
        price_t = mock_daily_data[
            (mock_daily_data['trade_date'] == trade_date) &
            (mock_daily_data['ts_code'] == stock)
        ]['close_adj'].iloc[0]
        
        # 获取 t+5 时刻价格
        future_date = mock_trade_cal.iloc[10]['cal_date']  # 5+5=10
        price_t5 = mock_daily_data[
            (mock_daily_data['trade_date'] == future_date) &
            (mock_daily_data['ts_code'] == stock)
        ]['close_adj'].iloc[0]
        
        # 计算预期收益率
        expected_ret = (price_t5 / price_t) - 1
        actual_ret = features.iloc[0]['y_ret_5']
        
        # 验证（允许小误差）
        assert abs(actual_ret - expected_ret) < 1e-6
    
    def test_label_missing_at_end(
        self,
        mock_trade_cal,
        mock_stock_basic,
        mock_daily_data,
        mock_adj_factor
    ):
        """测试数据末尾标签缺失的处理"""
        builder = FeatureBuilder(horizons=[5, 10, 20], require_label=False)
        
        # 使用倒数第 22 个交易日（20 日标签可能缺失）
        # 30 个交易日，-22 位置是第 8 个（index 7）
        trade_date = mock_trade_cal.iloc[-22]['cal_date']
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic
        )
        
        # y_ret_5 应该有值
        assert features['y_ret_5'].notna().sum() > 0
        # y_ret_20 可能为空（取决于数据范围）
    
    def test_filter_requires_at_least_one_label(
        self,
        mock_trade_cal,
        mock_stock_basic,
        mock_daily_data,
        mock_adj_factor
    ):
        """测试过滤逻辑：至少一个标签非空"""
        builder = FeatureBuilder(horizons=[5, 10, 20], require_label=True)
        
        # 使用倒数第 8 个交易日（y_ret_20 会缺失，但 y_ret_5/10 存在）
        trade_date = mock_trade_cal.iloc[-8]['cal_date']
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic
        )
        
        # 应该有样本（因为 y_ret_5 存在）
        assert len(features) > 0
        # 所有样本至少有一个标签非空
        for _, row in features.iterrows():
            has_label = any([
                pd.notna(row.get('y_ret_5')),
                pd.notna(row.get('y_ret_10')),
                pd.notna(row.get('y_ret_20'))
            ])
            assert has_label
    
    def test_custom_horizons(
        self,
        mock_trade_cal,
        mock_stock_basic,
        mock_daily_data,
        mock_adj_factor
    ):
        """测试自定义 horizons"""
        builder = FeatureBuilder(horizons=[3, 7])
        
        trade_date = mock_trade_cal.iloc[10]['cal_date']
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic
        )
        
        # 验证只包含指定的 horizon 标签
        assert 'y_ret_3' in features.columns
        assert 'y_ret_7' in features.columns
        assert 'y_ret_5' not in features.columns
        assert 'y_ret_10' not in features.columns
        assert 'y_ret_20' not in features.columns


class TestTrainModelLabelSelection:
    """测试训练脚本的 label 选择功能"""
    
    def test_label_parameter_parsing(self):
        """测试 --label 参数解析"""
        import argparse
        
        # 模拟 train_ml_model.py 的参数解析器
        parser = argparse.ArgumentParser()
        parser.add_argument("--label", type=str, choices=["y_ret_5", "y_ret_10", "y_ret_20"])
        parser.add_argument("--label-column", type=str, default="y_ret_5")
        
        # 测试 --label 参数
        args = parser.parse_args(["--label", "y_ret_10"])
        assert args.label == "y_ret_10"
        
        # 测试默认值
        args = parser.parse_args([])
        assert args.label is None
        assert args.label_column == "y_ret_5"
    
    def test_model_metadata_records_label(self, tmp_path):
        """测试模型元数据记录 label_column"""
        from src.lazybull.ml import ModelRegistry
        import joblib
        import xgboost as xgb
        
        # 创建临时模型目录
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # 创建真实的 XGBoost 模型（而非嵌套类）
        import numpy as np
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model = xgb.XGBRegressor(n_estimators=10)
        model.fit(X, y)
        
        # 注册模型
        registry = ModelRegistry(models_dir=str(models_dir))
        version = registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date="20230101",
            train_end_date="20231231",
            feature_columns=["feat1", "feat2"],
            label_column="y_ret_10",  # 使用 y_ret_10
            n_samples=1000,
            train_params={"n_estimators": 100},
            performance_metrics={"ic": 0.05}
        )
        
        # 验证元数据
        registry_data = registry._load_registry()
        model_metadata = registry_data["models"][0]
        
        assert model_metadata["version"] == version
        assert model_metadata["label_column"] == "y_ret_10"


class TestBacktestLabelSelection:
    """测试回测脚本的 label 选择和自动调仓频率"""
    
    def test_auto_rebalance_freq(self):
        """测试自动调仓频率设置"""
        import re
        
        # 模拟自动设置逻辑
        def auto_set_rebalance_freq(label):
            if label == 'y_ret_5':
                return 5
            elif label == 'y_ret_10':
                return 10
            elif label == 'y_ret_20':
                return 20
            else:
                match = re.search(r'(\d+)', label)
                if match:
                    return int(match.group(1))
                return 10
        
        # 测试标准标签
        assert auto_set_rebalance_freq('y_ret_5') == 5
        assert auto_set_rebalance_freq('y_ret_10') == 10
        assert auto_set_rebalance_freq('y_ret_20') == 20
        
        # 测试自定义标签
        assert auto_set_rebalance_freq('y_ret_15') == 15
        
        # 测试无效标签
        assert auto_set_rebalance_freq('invalid_label') == 10
    
    def test_model_label_consistency_check(self, tmp_path):
        """测试模型版本与 label 一致性校验"""
        from src.lazybull.ml import ModelRegistry
        import xgboost as xgb
        import numpy as np
        
        # 创建临时模型目录
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # 创建真实的 XGBoost 模型
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model = xgb.XGBRegressor(n_estimators=10)
        model.fit(X, y)
        
        # 注册使用 y_ret_5 训练的模型
        registry = ModelRegistry(models_dir=str(models_dir))
        version = registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date="20230101",
            train_end_date="20231231",
            feature_columns=["feat1"],
            label_column="y_ret_5",
            n_samples=1000,
            train_params={},
            performance_metrics={}
        )
        
        # 加载模型元数据
        _, metadata = registry.load_model(version=version)
        model_label = metadata['label_column']
        
        # 测试一致性校验
        def check_consistency(model_version, specified_label, model_label):
            if model_version is not None and specified_label is not None:
                if model_label != specified_label:
                    return False, f"模型训练标签 {model_label} 与指定标签 {specified_label} 不一致"
            return True, ""
        
        # 一致的情况
        is_valid, msg = check_consistency(version, "y_ret_5", model_label)
        assert is_valid
        
        # 不一致的情况
        is_valid, msg = check_consistency(version, "y_ret_10", model_label)
        assert not is_valid
        assert "不一致" in msg
        
        # 只指定 label（不指定 version）
        is_valid, msg = check_consistency(None, "y_ret_10", model_label)
        assert is_valid
        
        # 只指定 version（不指定 label）
        is_valid, msg = check_consistency(version, None, model_label)
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

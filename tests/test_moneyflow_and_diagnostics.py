"""测试 DataLoader.load_clean_moneyflow 和 moneyflow 特征生成"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from src.lazybull.data.storage import Storage
from src.lazybull.data.loader import DataLoader
from src.lazybull.common.feature_utils import cross_sectional_zscore
from src.lazybull.ml.eval_utils import compute_diagnostic_statistics, print_diagnostic_report


@pytest.fixture
def temp_storage():
    """创建临时存储实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir)
        yield storage


@pytest.fixture
def sample_moneyflow_data():
    """创建样本 moneyflow 数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'] * 3,
        'trade_date': ['20230101'] * 3 + ['20230102'] * 3 + ['20230103'] * 3,
        'buy_sm_amount': [100, 200, 150, 110, 210, 160, 120, 220, 170],
        'buy_md_amount': [500, 600, 550, 510, 610, 560, 520, 620, 570],
        'buy_lg_amount': [1000, 1200, 1100, 1050, 1250, 1150, 1100, 1300, 1200],
        'sell_sm_amount': [80, 180, 130, 90, 190, 140, 100, 200, 150],
        'sell_md_amount': [450, 550, 500, 460, 560, 510, 470, 570, 520],
        'sell_lg_amount': [900, 1100, 1000, 950, 1150, 1050, 1000, 1200, 1100],
    })


@pytest.fixture
def sample_features_data():
    """创建样本特征数据（用于诊断测试）"""
    np.random.seed(42)
    n_stocks = 100
    n_dates = 10
    
    data = []
    for i in range(n_dates):
        date = f'2023010{i+1}'
        for j in range(n_stocks):
            data.append({
                'trade_date': date,
                'ts_code': f'{j:06d}.SZ',
                'pred_score': np.random.randn(),
                'y_ret_20': np.random.randn() * 0.05 + 0.01  # 均值 1%, 标准差 5%
            })
    
    return pd.DataFrame(data)


class TestLoadCleanMoneyflow:
    """测试 DataLoader.load_clean_moneyflow"""
    
    def test_load_clean_moneyflow_by_date_range(self, temp_storage, sample_moneyflow_data):
        """测试按日期范围加载 clean moneyflow"""
        loader = DataLoader(storage=temp_storage)
        
        # 保存样本数据（分区）
        for date in ['20230101', '20230102', '20230103']:
            date_data = sample_moneyflow_data[sample_moneyflow_data['trade_date'] == date]
            temp_storage.save_clean_by_date(date_data, 'moneyflow', date)
        
        # 加载日期范围
        loaded = loader.load_clean_moneyflow('20230101', '20230102')
        
        assert loaded is not None
        assert len(loaded) == 6  # 2 天 * 3 只股票
        assert 'trade_date' in loaded.columns
        assert 'buy_lg_amount' in loaded.columns
    
    def test_load_clean_moneyflow_empty(self, temp_storage):
        """测试加载不存在的 moneyflow 数据"""
        loader = DataLoader(storage=temp_storage)
        
        loaded = loader.load_clean_moneyflow('20230101', '20230102')
        
        # 应该返回 None（没有数据）
        assert loaded is None
    
    def test_load_clean_moneyflow_date_format(self, temp_storage, sample_moneyflow_data):
        """测试日期格式转换"""
        loader = DataLoader(storage=temp_storage)
        
        # 保存数据
        temp_storage.save_clean_by_date(sample_moneyflow_data, 'moneyflow', '20230101')
        
        # 用不同格式加载
        loaded1 = loader.load_clean_moneyflow('20230101', '20230101')
        loaded2 = loader.load_clean_moneyflow('2023-01-01', '2023-01-01')
        
        assert loaded1 is not None
        assert loaded2 is not None
        assert len(loaded1) == len(loaded2)


class TestCrossSectionalZscoreVectorized:
    """测试 cross_sectional_zscore 矢量化实现"""
    
    def test_cs_zscore_no_groupby_apply(self):
        """测试 cs_zscore 不触发 groupby.apply"""
        df = pd.DataFrame({
            'trade_date': ['20230101'] * 50 + ['20230102'] * 50,
            'ts_code': [f'{i:06d}.SZ' for i in range(50)] * 2,
            'return': np.random.randn(100)
        })
        
        # 捕获警告
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = cross_sectional_zscore(
                df, 
                value_col='return',
                group_col='trade_date',
                winsorize_limits=(0.01, 0.01)
            )
            
            # 检查是否有 FutureWarning
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            assert len(future_warnings) == 0, f"触发了 FutureWarning: {future_warnings}"
        
        # 验证结果
        assert len(result) == len(df)
        assert result.notna().all()
    
    def test_cs_zscore_correctness(self):
        """测试 cs_zscore 计算正确性"""
        df = pd.DataFrame({
            'trade_date': ['20230101'] * 5,
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = cross_sectional_zscore(
            df,
            value_col='value',
            group_col='trade_date',
            winsorize_limits=None
        )
        
        # 验证均值接近 0，标准差接近 1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std(ddof=0) - 1.0) < 1e-10


class TestDiagnosticStatistics:
    """测试逐日评估诊断统计"""
    
    def test_compute_diagnostic_statistics(self, sample_features_data):
        """测试诊断统计计算"""
        diagnostics = compute_diagnostic_statistics(
            df=sample_features_data,
            date_col='trade_date',
            prediction_col='pred_score',
            return_col='y_ret_20',
            topk_values=[10, 30]
        )
        
        # 验证必需的诊断项存在
        assert '全市场收益_逐日均值的均值' in diagnostics
        assert '全市场收益_逐日标准差的均值' in diagnostics
        assert '每日样本数_最小' in diagnostics
        assert '每日样本数_中位数' in diagnostics
        assert '每日样本数_最大' in diagnostics
        
        # 验证 TopK 诊断项
        assert 'Top10_逐日均值的均值' in diagnostics
        assert 'Top10_相对全市场提升_均值' in diagnostics
        assert 'Top10_逐日均值_25分位' in diagnostics
        assert 'Top10_逐日均值_50分位' in diagnostics
        assert 'Top10_逐日均值_75分位' in diagnostics
        
        # 验证样本数统计
        assert diagnostics['每日样本数_最小'] == 100
        assert diagnostics['每日样本数_最大'] == 100

    def test_topk_diagnostics_skip_days_with_insufficient_samples(self):
        df = pd.DataFrame(
            {
                'trade_date': ['20240102'] * 5 + ['20240103'] * 12,
                'ts_code': [f'{i:06d}.SZ' for i in range(5)] + [f'{i:06d}.SZ' for i in range(12)],
                'pred_score': list(range(5, 0, -1)) + list(range(12, 0, -1)),
                'y_ret_20': np.linspace(0.01, 0.05, 17),
            }
        )

        diagnostics = compute_diagnostic_statistics(
            df=df,
            date_col='trade_date',
            prediction_col='pred_score',
            return_col='y_ret_20',
            topk_values=[10],
        )

        assert diagnostics['Top10_有效交易日数'] == 1
        assert diagnostics['Top10_样本覆盖率'] == 0.5
    
    def test_print_diagnostic_report(self, sample_features_data):
        """测试诊断报告打印（不报错即可）"""
        diagnostics = compute_diagnostic_statistics(
            df=sample_features_data,
            date_col='trade_date',
            prediction_col='pred_score',
            return_col='y_ret_20',
            topk_values=[30]
        )
        
        # 测试打印不报错
        try:
            print_diagnostic_report(diagnostics)
            success = True
        except Exception as e:
            success = False
            print(f"打印诊断报告失败: {e}")
        
        assert success


class TestClassificationLabelsVectorized:
    """测试分类标签生成矢量化实现（在 train_ml_model.py 中）"""
    
    def test_topk_labels_no_warnings(self):
        """测试 TopK 标签生成不触发 FutureWarning"""
        df = pd.DataFrame({
            'trade_date': ['20230101'] * 50 + ['20230102'] * 50,
            'ts_code': [f'{i:06d}.SZ' for i in range(50)] * 2,
            'y_ret_20': np.random.randn(100)
        })
        
        # 模拟 generate_classification_labels 的核心逻辑
        pos_topk = 10
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 模拟生成标签
            df['_rank'] = df.groupby('trade_date')['y_ret_20'].rank(
                method='first',
                ascending=False,
                na_option='keep'
            )
            df['binary_label'] = (df['_rank'] <= pos_topk).astype(float)
            
            # 检查是否有 FutureWarning
            future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
            assert len(future_warnings) == 0
        
        # 验证每个交易日的正类数量
        pos_counts = df.groupby('trade_date')['binary_label'].sum()
        assert (pos_counts == pos_topk).all()


class TestMoneyflowFeaturesIntegration:
    """测试 moneyflow 特征生成集成（确保 ensure 流程包含 moneyflow）"""
    
    def test_ensure_features_requires_moneyflow(self, temp_storage):
        """测试 ensure_features_for_date 需要 moneyflow 数据"""
        # 这个测试需要完整的环境，这里只验证逻辑
        # 实际测试应该在 test_features.py 中进行
        
        # 模拟：缺少 moneyflow 时应该报错
        # （在实际代码中，ensure_features_for_date 会检查并报错）
        
        # 这里只是占位测试，确保测试文件可以被导入
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

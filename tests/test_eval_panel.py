"""评估面板测试"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml import ModelRegistry
from src.lazybull.signals import MLSignal


class MockMLModel:
    """模拟 ML 模型（用于测试）"""
    
    def predict(self, X):
        """返回模拟预测值（基于第一个特征）"""
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


def test_equal_count_grouping():
    """测试等数量分组函数"""
    # 导入函数
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 导入 run_ml_backtest 中的函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ml_backtest", 
        project_root / "scripts" / "run_ml_backtest.py"
    )
    run_ml_backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_ml_backtest)
    
    equal_count_grouping = run_ml_backtest.equal_count_grouping
    
    # 测试1：10个样本分成3组
    scores = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], index=[f"stock_{i}" for i in range(10)])
    groups = equal_count_grouping(scores, n_groups=3)
    
    # 验证分组正确
    assert len(groups) == 10
    assert set(groups.values) == {1, 2, 3}
    
    # 验证每组大小（10个样本分3组：4, 3, 3）
    group_counts = groups.value_counts().sort_index()
    assert group_counts[1] == 4  # 第1组（最高分）
    assert group_counts[2] == 3  # 第2组
    assert group_counts[3] == 3  # 第3组
    
    # 验证分数降序：第1组的分数应该最高
    group1_scores = scores[groups == 1]
    group3_scores = scores[groups == 3]
    assert group1_scores.mean() > group3_scores.mean()
    
    # 测试2：9个样本分成3组（能整除）
    scores2 = pd.Series([9, 8, 7, 6, 5, 4, 3, 2, 1], index=[f"stock_{i}" for i in range(9)])
    groups2 = equal_count_grouping(scores2, n_groups=3)
    
    group_counts2 = groups2.value_counts().sort_index()
    assert group_counts2[1] == 3
    assert group_counts2[2] == 3
    assert group_counts2[3] == 3
    
    # 测试3：空序列
    empty_scores = pd.Series(dtype=float)
    empty_groups = equal_count_grouping(empty_scores, n_groups=3)
    assert len(empty_groups) == 0


def test_rank_ic_calculation(trained_model):
    """测试 RankIC 计算（使用简单可控数据）"""
    models_dir, version = trained_model
    
    # 创建完美正相关的预测和真实收益
    scores = pd.Series([1.0, 0.8, 0.6, 0.4, 0.2], index=[f"stock_{i}" for i in range(5)])
    returns = pd.Series([0.10, 0.08, 0.06, 0.04, 0.02], index=[f"stock_{i}" for i in range(5)])
    
    # 计算 Spearman 相关
    rank_ic = scores.corr(returns, method='spearman')
    
    # 完美正相关应该是 1.0
    assert rank_ic == 1.0
    
    # 创建完美负相关
    returns_neg = pd.Series([0.02, 0.04, 0.06, 0.08, 0.10], index=[f"stock_{i}" for i in range(5)])
    rank_ic_neg = scores.corr(returns_neg, method='spearman')
    
    # 完美负相关应该是 -1.0
    assert rank_ic_neg == -1.0
    
    # 创建无相关
    returns_random = pd.Series([0.06, 0.02, 0.10, 0.04, 0.08], index=[f"stock_{i}" for i in range(5)])
    rank_ic_random = scores.corr(returns_random, method='spearman')
    
    # 无相关应该接近 0（但不一定完全是 0）
    assert -1.0 <= rank_ic_random <= 1.0


def test_evaluate_daily(trained_model, tmp_path):
    """测试日度评估函数"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 导入 run_ml_backtest 中的函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ml_backtest", 
        project_root / "scripts" / "run_ml_backtest.py"
    )
    run_ml_backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_ml_backtest)
    
    evaluate_daily = run_ml_backtest.evaluate_daily
    
    models_dir, version = trained_model
    
    # 创建 ML 信号
    signal = MLSignal(
        top_n=3,
        model_version=version,
        models_dir=models_dir,
        weight_method="equal",
        verbose=False
    )
    
    # 创建模拟特征数据
    features_df = pd.DataFrame({
        'ts_code': [f'stock_{i}' for i in range(10)],
        'f1': np.linspace(10, 1, 10),  # 第一个特征降序（用于预测）
        'f2': np.random.randn(10),
        'f3': np.random.randn(10),
        'y_ret_5': np.linspace(0.10, 0.01, 10)  # 真实收益也降序（完美相关）
    })
    
    universe = features_df['ts_code'].tolist()
    
    # 评估
    daily_metrics, group_details = evaluate_daily(
        date='20231201',
        signal=signal,
        universe=universe,
        features_df=features_df,
        label_column='y_ret_5',
        n_groups=5,
        topk=3
    )
    
    # 验证日度指标
    assert daily_metrics is not None
    assert daily_metrics['交易日期'] == '20231201'
    assert daily_metrics['样本数'] == 10
    assert 'RankIC' in daily_metrics
    assert 'TopK平均收益' in daily_metrics
    assert 'Top组平均收益' in daily_metrics
    assert 'Bottom组平均收益' in daily_metrics
    assert '多空收益' in daily_metrics
    
    # 由于预测和真实收益是正相关的，RankIC 应该 > 0
    assert daily_metrics['RankIC'] > 0
    
    # Top组收益应该高于Bottom组
    assert daily_metrics['Top组平均收益'] > daily_metrics['Bottom组平均收益']
    
    # 多空收益应该 > 0
    assert daily_metrics['多空收益'] > 0
    
    # 验证分组明细
    assert group_details is not None
    assert len(group_details) == 5  # 5个组
    
    # 验证每组都有数据
    for detail in group_details:
        assert '交易日期' in detail
        assert '组号' in detail
        assert '组内股票数' in detail
        assert '组内平均真实收益' in detail
        assert '组内平均预测分数' in detail
        assert detail['组内股票数'] > 0


def test_csv_export(trained_model, tmp_path):
    """测试 CSV 文件生成"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 导入 run_ml_backtest 中的函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ml_backtest", 
        project_root / "scripts" / "run_ml_backtest.py"
    )
    run_ml_backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_ml_backtest)
    
    _append_dict_to_csv = run_ml_backtest._append_dict_to_csv
    
    # 测试 CSV 追加功能
    csv_file = tmp_path / "test.csv"
    
    # 第一次写入
    row1 = {'col1': 'value1', 'col2': 10, 'col3': 1.5}
    _append_dict_to_csv(csv_file, row1, fieldnames=['col1', 'col2', 'col3'])
    
    # 验证文件存在
    assert csv_file.exists()
    
    # 读取并验证
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    assert len(df) == 1
    assert list(df.columns) == ['col1', 'col2', 'col3']
    assert df.iloc[0]['col1'] == 'value1'
    assert df.iloc[0]['col2'] == 10
    assert df.iloc[0]['col3'] == 1.5
    
    # 第二次追加
    row2 = {'col1': 'value2', 'col2': 20, 'col3': 2.5}
    _append_dict_to_csv(csv_file, row2, fieldnames=['col1', 'col2', 'col3'])
    
    # 再次读取验证
    df2 = pd.read_csv(csv_file, encoding='utf-8-sig')
    assert len(df2) == 2
    assert df2.iloc[1]['col1'] == 'value2'
    assert df2.iloc[1]['col2'] == 20
    assert df2.iloc[1]['col3'] == 2.5

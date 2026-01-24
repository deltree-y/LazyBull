"""测试机器学习评估模块

测试IC/RankIC统计功能：
- 按日IC序列计算
- IC统计指标计算
- 完整评估流程
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml.evaluation import (
    calculate_daily_ic,
    calculate_ic_statistics,
    evaluate_model_ic,
    print_ic_evaluation_report,
)


@pytest.fixture
def sample_predictions():
    """生成测试用的预测和真实值数据
    
    包含3个交易日，每日20只股票
    模拟有一定相关性的预测值
    """
    np.random.seed(42)
    
    dates = ['20230101', '20230102', '20230103']
    n_stocks_per_day = 20
    
    data = []
    for date in dates:
        # 生成真实标签（收益率）
        y_true = np.random.normal(0, 0.05, n_stocks_per_day)
        
        # 生成预测值（与真实值有一定相关性）
        # 预测 = 0.6 * 真实 + 0.4 * 噪音
        noise = np.random.normal(0, 0.03, n_stocks_per_day)
        y_pred = 0.6 * y_true + 0.4 * noise
        
        for i in range(n_stocks_per_day):
            data.append({
                'trade_date': date,
                'ts_code': f'{i:06d}.SZ',
                'y_true': y_true[i],
                'y_pred': y_pred[i]
            })
    
    df = pd.DataFrame(data)
    return df


def test_calculate_daily_ic_pearson(sample_predictions):
    """测试按日IC计算（Pearson相关系数）"""
    df = sample_predictions
    
    # 计算按日IC
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 验证返回类型和长度
    assert isinstance(ic_series, pd.Series)
    assert len(ic_series) == df['trade_date'].nunique()
    
    # 验证IC值范围（-1到1之间）
    assert (ic_series >= -1.0).all() and (ic_series <= 1.0).all()
    
    # 验证IC值不全为NaN
    assert ic_series.notna().sum() > 0
    
    # 由于我们构造了有相关性的数据，IC应该是正的
    valid_ics = ic_series.dropna()
    assert valid_ics.mean() > 0, "预期IC为正值（因为构造了正相关数据）"


def test_calculate_daily_ic_spearman(sample_predictions):
    """测试按日IC计算（Spearman相关系数）"""
    df = sample_predictions
    
    # 计算按日RankIC
    rank_ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='spearman'
    )
    
    # 验证返回类型和长度
    assert isinstance(rank_ic_series, pd.Series)
    assert len(rank_ic_series) == df['trade_date'].nunique()
    
    # 验证RankIC值范围
    assert (rank_ic_series >= -1.0).all() and (rank_ic_series <= 1.0).all()
    
    # 由于我们构造了有相关性的数据，RankIC应该是正的
    valid_rank_ics = rank_ic_series.dropna()
    assert valid_rank_ics.mean() > 0


def test_calculate_ic_statistics(sample_predictions):
    """测试IC统计指标计算"""
    df = sample_predictions
    
    # 先计算按日IC序列
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 计算统计指标
    stats = calculate_ic_statistics(ic_series, include_quantiles=True)
    
    # 验证包含所有必需的统计指标
    required_keys = ['ic_mean', 'ic_std', 'ic_ir', 'ic_positive_rate', 'ic_min', 'ic_max']
    for key in required_keys:
        assert key in stats, f"缺少统计指标: {key}"
    
    # 验证分位数
    assert 'ic_p10' in stats
    assert 'ic_p50' in stats
    assert 'ic_p90' in stats
    
    # 验证统计值的合理性
    assert stats['ic_min'] <= stats['ic_p10'] <= stats['ic_p50'] <= stats['ic_p90'] <= stats['ic_max']
    assert 0.0 <= stats['ic_positive_rate'] <= 1.0
    
    # 验证IR（信息比率）计算
    if stats['ic_std'] > 0:
        expected_ir = stats['ic_mean'] / stats['ic_std']
        assert abs(stats['ic_ir'] - expected_ir) < 1e-6


def test_evaluate_model_ic_without_dates(sample_predictions):
    """测试不提供日期的总体IC评估"""
    df = sample_predictions
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    
    # 评估（不提供日期）
    metrics = evaluate_model_ic(
        y_true=y_true,
        y_pred=y_pred,
        trade_dates=None
    )
    
    # 应该只包含总体IC和RankIC
    assert 'ic' in metrics
    assert 'rank_ic' in metrics
    
    # 不应该包含按日统计
    assert 'ic_mean' not in metrics
    assert 'rank_ic_mean' not in metrics
    
    # 验证IC值合理性
    assert -1.0 <= metrics['ic'] <= 1.0
    assert -1.0 <= metrics['rank_ic'] <= 1.0


def test_evaluate_model_ic_with_dates(sample_predictions):
    """测试提供日期的完整IC评估"""
    df = sample_predictions
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    trade_dates = df['trade_date']
    
    # 完整评估
    metrics = evaluate_model_ic(
        y_true=y_true,
        y_pred=y_pred,
        trade_dates=trade_dates,
        return_daily_series=False
    )
    
    # 应该包含总体IC
    assert 'ic' in metrics
    assert 'rank_ic' in metrics
    
    # 应该包含按日统计
    assert 'ic_mean' in metrics
    assert 'ic_std' in metrics
    assert 'ic_ir' in metrics
    assert 'ic_positive_rate' in metrics
    
    assert 'rank_ic_mean' in metrics
    assert 'rank_ic_std' in metrics
    assert 'rank_ic_ir' in metrics
    assert 'rank_ic_positive_rate' in metrics
    
    # 应该包含分位数
    assert 'ic_p10' in metrics
    assert 'ic_p50' in metrics
    assert 'ic_p90' in metrics
    
    # 不应该包含序列（因为return_daily_series=False）
    assert 'ic_series' not in metrics
    assert 'rank_ic_series' not in metrics


def test_evaluate_model_ic_return_series(sample_predictions):
    """测试返回按日IC序列"""
    df = sample_predictions
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    trade_dates = df['trade_date']
    
    # 评估并返回序列
    metrics = evaluate_model_ic(
        y_true=y_true,
        y_pred=y_pred,
        trade_dates=trade_dates,
        return_daily_series=True
    )
    
    # 应该包含序列
    assert 'ic_series' in metrics
    assert 'rank_ic_series' in metrics
    
    # 验证序列类型和长度
    assert isinstance(metrics['ic_series'], pd.Series)
    assert isinstance(metrics['rank_ic_series'], pd.Series)
    assert len(metrics['ic_series']) == df['trade_date'].nunique()
    assert len(metrics['rank_ic_series']) == df['trade_date'].nunique()


def test_print_ic_evaluation_report(sample_predictions, caplog):
    """测试IC评估报告打印功能"""
    df = sample_predictions
    
    y_true = df['y_true']
    y_pred = df['y_pred']
    trade_dates = df['trade_date']
    
    # 评估
    metrics = evaluate_model_ic(
        y_true=y_true,
        y_pred=y_pred,
        trade_dates=trade_dates
    )
    
    # 打印报告（应该不抛出异常）
    print_ic_evaluation_report(metrics, title="测试报告")
    
    # 验证日志中包含关键信息（通过caplog捕获）
    # 注意：实际测试时可能需要配置loguru的捕获


def test_handle_missing_predictions():
    """测试处理包含缺失值的预测"""
    data = {
        'trade_date': ['20230101'] * 10,
        'ts_code': [f'{i:06d}.SZ' for i in range(10)],
        'y_true': [0.01, 0.02, 0.03, -0.01, -0.02, 0.00, 0.04, -0.03, 0.01, 0.02],
        'y_pred': [0.02, np.nan, 0.02, -0.02, np.nan, 0.01, 0.03, -0.02, np.nan, 0.01]
    }
    df = pd.DataFrame(data)
    
    # 计算IC（应该自动过滤缺失值）
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 应该返回有效的IC（基于非缺失样本）
    assert len(ic_series) == 1
    assert ic_series.notna().all()


def test_insufficient_samples():
    """测试样本数不足的情况"""
    # 每天只有1个样本
    data = {
        'trade_date': ['20230101', '20230102'],
        'ts_code': ['000001.SZ', '000002.SZ'],
        'y_true': [0.01, 0.02],
        'y_pred': [0.02, 0.01]
    }
    df = pd.DataFrame(data)
    
    # 计算IC（样本数不足，应该返回NaN）
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 每天只有1个样本，无法计算相关系数，应该全为NaN
    assert ic_series.isna().all()


def test_perfect_correlation():
    """测试完美相关的情况（IC=1）"""
    data = {
        'trade_date': ['20230101'] * 10,
        'ts_code': [f'{i:06d}.SZ' for i in range(10)],
        'y_true': np.arange(10) * 0.01,  # 线性序列
        'y_pred': np.arange(10) * 0.01   # 完全相同
    }
    df = pd.DataFrame(data)
    
    # 计算IC
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 完美相关，IC应该接近1
    assert abs(ic_series.iloc[0] - 1.0) < 1e-6


def test_negative_correlation():
    """测试负相关的情况"""
    data = {
        'trade_date': ['20230101'] * 10,
        'ts_code': [f'{i:06d}.SZ' for i in range(10)],
        'y_true': np.arange(10) * 0.01,
        'y_pred': -np.arange(10) * 0.01  # 完全负相关
    }
    df = pd.DataFrame(data)
    
    # 计算IC
    ic_series = calculate_daily_ic(
        df,
        pred_col='y_pred',
        label_col='y_true',
        date_col='trade_date',
        method='pearson'
    )
    
    # 完美负相关，IC应该接近-1
    assert abs(ic_series.iloc[0] - (-1.0)) < 1e-6


def test_ic_statistics_empty_series():
    """测试空IC序列的统计"""
    empty_series = pd.Series(dtype=float)
    
    stats = calculate_ic_statistics(empty_series)
    
    # 应该返回NaN
    assert pd.isna(stats['ic_mean'])
    assert pd.isna(stats['ic_std'])
    assert pd.isna(stats['ic_ir'])


def test_ic_statistics_all_nan():
    """测试全为NaN的IC序列"""
    nan_series = pd.Series([np.nan, np.nan, np.nan])
    
    stats = calculate_ic_statistics(nan_series)
    
    # 应该返回NaN
    assert pd.isna(stats['ic_mean'])
    assert pd.isna(stats['ic_std'])

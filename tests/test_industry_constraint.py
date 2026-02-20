"""行业约束模块测试"""

import pytest
import pandas as pd
from src.lazybull.portfolio.industry_constraint import (
    load_industry_mapping,
    apply_industry_constraint
)


def test_load_industry_mapping_basic():
    """测试基本的行业映射加载"""
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'sw_industry': ['国有大型银行', '房地产开发', '国有大型银行']
    })
    
    mapping = load_industry_mapping(stock_basic)
    
    assert mapping['000001.SZ'] == '国有大型银行'
    assert mapping['000002.SZ'] == '房地产开发'
    assert mapping['600000.SH'] == '国有大型银行'
    assert len(mapping) == 3


def test_load_industry_mapping_with_missing():
    """测试行业缺失的情况"""
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'sw_industry': ['国有大型银行', None, '']
    })
    
    mapping = load_industry_mapping(stock_basic)
    
    assert mapping['000001.SZ'] == '国有大型银行'
    assert mapping['000002.SZ'] == '未知行业'
    assert mapping['600000.SH'] == '未知行业'


def test_load_industry_mapping_empty():
    """测试空 DataFrame"""
    stock_basic = pd.DataFrame()
    mapping = load_industry_mapping(stock_basic)
    assert mapping == {}


def test_load_industry_mapping_missing_columns():
    """测试缺少必需列的情况"""
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ'],
    })
    
    with pytest.raises(ValueError, match="必须包含 sw_industry 列"):
        load_industry_mapping(stock_basic)
    
    stock_basic = pd.DataFrame({
        'sw_industry': ['银行I'],
    })
    
    with pytest.raises(ValueError, match="必须包含 ts_code 列"):
        load_industry_mapping(stock_basic)


def test_apply_industry_constraint_basic():
    """测试基本的行业约束"""
    ranked_candidates = [
        ('stock_a', 0.9),  # 银行
        ('stock_b', 0.8),  # 银行
        ('stock_c', 0.7),  # 房地产
        ('stock_d', 0.6),  # 银行（应被跳过）
        ('stock_e', 0.5),  # 房地产
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '银行',
        'stock_c': '房地产',
        'stock_d': '银行',
        'stock_e': '房地产',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=4
    )
    
    # 应选中 4 只股票
    assert len(result) == 4
    
    # 应选中 stock_a, stock_b, stock_c, stock_e
    selected_stocks = [stock for stock, score in result]
    assert selected_stocks == ['stock_a', 'stock_b', 'stock_c', 'stock_e']
    
    # stock_d 因银行行业已满应被跳过
    assert 'stock_d' not in selected_stocks


def test_apply_industry_constraint_exact_limit():
    """测试刚好达到目标数量的情况"""
    ranked_candidates = [
        ('stock_a', 0.9),
        ('stock_b', 0.8),
        ('stock_c', 0.7),
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '房地产',
        'stock_c': '制造',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=1,
        target_n=3
    )
    
    assert len(result) == 3
    assert [stock for stock, score in result] == ['stock_a', 'stock_b', 'stock_c']


def test_apply_industry_constraint_insufficient_candidates():
    """测试候选不足的情况"""
    ranked_candidates = [
        ('stock_a', 0.9),
        ('stock_b', 0.8),
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '房地产',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=5
    )
    
    # 只有 2 个候选，最多选 2 个
    assert len(result) == 2


def test_apply_industry_constraint_single_industry_limit():
    """测试单个行业达到上限的情况"""
    ranked_candidates = [
        ('stock_a', 0.9),  # 银行
        ('stock_b', 0.8),  # 银行
        ('stock_c', 0.7),  # 银行
        ('stock_d', 0.6),  # 银行
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '银行',
        'stock_c': '银行',
        'stock_d': '银行',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=3
    )
    
    # 只能选 2 只（行业上限）
    assert len(result) == 2
    assert [stock for stock, score in result] == ['stock_a', 'stock_b']


def test_apply_industry_constraint_unknown_industry():
    """测试未知行业的处理"""
    ranked_candidates = [
        ('stock_a', 0.9),  # 银行
        ('stock_b', 0.8),  # 未在映射中（未知行业）
        ('stock_c', 0.7),  # 未在映射中（未知行业）
        ('stock_d', 0.6),  # 未在映射中（未知行业）应被跳过
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        # stock_b, stock_c, stock_d 不在映射中
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=3
    )
    
    # 应选中 3 只：stock_a, stock_b, stock_c
    assert len(result) == 3
    selected_stocks = [stock for stock, score in result]
    assert selected_stocks == ['stock_a', 'stock_b', 'stock_c']


def test_apply_industry_constraint_empty_candidates():
    """测试空候选列表"""
    result = apply_industry_constraint(
        [],
        {},
        max_per_industry=2,
        target_n=5
    )
    
    assert result == []


def test_apply_industry_constraint_zero_target():
    """测试目标数量为0"""
    ranked_candidates = [
        ('stock_a', 0.9),
        ('stock_b', 0.8),
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '房地产',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=0
    )
    
    assert result == []


def test_apply_industry_constraint_invalid_max():
    """测试无效的 max_per_industry"""
    ranked_candidates = [('stock_a', 0.9)]
    industry_mapping = {'stock_a': '银行'}
    
    with pytest.raises(ValueError):
        apply_industry_constraint(
            ranked_candidates,
            industry_mapping,
            max_per_industry=0,
            target_n=1
        )
    
    with pytest.raises(ValueError):
        apply_industry_constraint(
            ranked_candidates,
            industry_mapping,
            max_per_industry=-1,
            target_n=1
        )


def test_apply_industry_constraint_diverse_industries():
    """测试多样化的行业分布"""
    ranked_candidates = [
        ('stock_a', 0.95),  # 银行
        ('stock_b', 0.90),  # 房地产
        ('stock_c', 0.85),  # 制造
        ('stock_d', 0.80),  # 银行
        ('stock_e', 0.75),  # 房地产
        ('stock_f', 0.70),  # 科技
        ('stock_g', 0.65),  # 银行（应被跳过）
        ('stock_h', 0.60),  # 科技
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '房地产',
        'stock_c': '制造',
        'stock_d': '银行',
        'stock_e': '房地产',
        'stock_f': '科技',
        'stock_g': '银行',
        'stock_h': '科技',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=2,
        target_n=7
    )
    
    # 应选中 7 只股票
    assert len(result) == 7
    
    selected_stocks = [stock for stock, score in result]
    
    # stock_g 应被跳过（银行已有 2 只）
    assert 'stock_g' not in selected_stocks
    
    # 检查行业分布
    selected_industries = [industry_mapping[stock] for stock in selected_stocks]
    from collections import Counter
    industry_counts = Counter(selected_industries)
    
    # 每个行业不超过 2 只
    assert all(count <= 2 for count in industry_counts.values())


def test_apply_industry_constraint_preserve_order():
    """测试保持候选顺序（按分数降序）"""
    ranked_candidates = [
        ('stock_a', 0.9),
        ('stock_b', 0.8),
        ('stock_c', 0.7),
    ]
    
    industry_mapping = {
        'stock_a': '银行',
        'stock_b': '房地产',
        'stock_c': '制造',
    }
    
    result = apply_industry_constraint(
        ranked_candidates,
        industry_mapping,
        max_per_industry=1,
        target_n=3
    )
    
    # 结果应保持原始顺序
    assert result == ranked_candidates

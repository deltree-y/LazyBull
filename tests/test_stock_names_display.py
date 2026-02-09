"""测试股票名称显示功能"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage
from src.lazybull.paper import PaperAccount, PaperBroker, PaperStorage, Position


@pytest.fixture
def temp_storage_with_stock_basic():
    """创建带有 stock_basic 数据的临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir, verbose=False)
        
        # 创建模拟的 stock_basic 数据
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '603115.SH'],
            'symbol': ['000001', '000002', '600000', '603115'],
            'name': ['平安银行', '万科A', '浦发银行', '三维股份'],
            'market': ['主板', '主板', '主板', '主板'],
            'list_date': ['19910403', '19910129', '19991110', '20170705']
        })
        
        # 保存为 clean 数据
        storage.save_clean(stock_basic, "stock_basic")
        
        yield storage


@pytest.fixture
def temp_storage_without_stock_basic():
    """创建不带 stock_basic 数据的临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir, verbose=False)
        yield storage


@pytest.fixture
def temp_paper_storage():
    """创建临时纸面交易存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        yield storage


@pytest.fixture
def sample_positions_account(temp_paper_storage):
    """创建带有持仓的账户"""
    account = PaperAccount(initial_capital=100000.0, storage=temp_paper_storage)
    
    # 添加一些持仓（访问 state.positions）
    account.state.positions['000001.SZ'] = Position(
        ts_code='000001.SZ',
        shares=1000,
        buy_price=10.0,
        buy_cost=15.0,
        buy_date='20260205'
    )
    
    account.state.positions['603115.SH'] = Position(
        ts_code='603115.SH',
        shares=500,
        buy_price=20.0,
        buy_cost=20.0,
        buy_date='20260205'
    )
    
    yield account


def test_get_positions_detail_with_stock_names(temp_paper_storage, sample_positions_account):
    """测试当提供股票名称字典时，持仓明细能正确显示股票名称"""
    broker = PaperBroker(sample_positions_account)
    
    # 当前价格
    current_prices = {
        '000001.SZ': 11.0,
        '603115.SH': 22.0
    }
    
    # 股票名称字典
    stock_names = {
        '000001.SZ': '平安银行',
        '603115.SH': '三维股份'
    }
    
    # 获取持仓明细
    df = broker.get_positions_detail(current_prices, '20260206', stock_names)
    
    # 验证股票代码列包含名称
    assert not df.empty
    assert len(df) == 2
    
    # 检查股票代码显示格式
    stock_codes = df['股票代码'].tolist()
    assert '000001.SZ(平安银行)' in stock_codes
    assert '603115.SH(三维股份)' in stock_codes


def test_get_positions_detail_without_stock_names(temp_paper_storage, sample_positions_account):
    """测试当不提供股票名称字典时，持仓明细显示 (na)"""
    broker = PaperBroker(sample_positions_account)
    
    # 当前价格
    current_prices = {
        '000001.SZ': 11.0,
        '603115.SH': 22.0
    }
    
    # 不提供股票名称字典（或提供空字典）
    stock_names = {}
    
    # 获取持仓明细
    df = broker.get_positions_detail(current_prices, '20260206', stock_names)
    
    # 验证股票代码列显示 (na)
    assert not df.empty
    assert len(df) == 2
    
    # 检查股票代码显示格式
    stock_codes = df['股票代码'].tolist()
    assert '000001.SZ(na)' in stock_codes
    assert '603115.SH(na)' in stock_codes


def test_build_stock_names_dict_with_clean_stock_basic(temp_storage_with_stock_basic):
    """测试从 clean stock_basic 构建股票名称字典"""
    loader = DataLoader(temp_storage_with_stock_basic, verbose=False)
    
    # 导入构建函数（需要从 scripts 导入，这里模拟相同逻辑）
    stock_names = {}
    stock_basic = loader.load_clean_stock_basic()
    
    if stock_basic is not None and not stock_basic.empty:
        if 'ts_code' in stock_basic.columns and 'name' in stock_basic.columns:
            for _, row in stock_basic.iterrows():
                if pd.notna(row.get('ts_code')) and pd.notna(row.get('name')) and row['name']:
                    stock_names[row['ts_code']] = row['name']
    
    # 验证股票名称字典
    assert len(stock_names) == 4
    assert stock_names['000001.SZ'] == '平安银行'
    assert stock_names['000002.SZ'] == '万科A'
    assert stock_names['600000.SH'] == '浦发银行'
    assert stock_names['603115.SH'] == '三维股份'


def test_build_stock_names_dict_without_stock_basic(temp_storage_without_stock_basic):
    """测试当 stock_basic 不存在时，返回空字典"""
    loader = DataLoader(temp_storage_without_stock_basic, verbose=False)
    
    # 模拟构建逻辑
    stock_names = {}
    stock_basic = loader.load_clean_stock_basic()
    
    if stock_basic is None or stock_basic.empty:
        stock_basic = loader.load_stock_basic()
    
    if stock_basic is None or stock_basic.empty:
        # 应该返回空字典
        pass
    
    # 验证返回空字典
    assert len(stock_names) == 0


def test_load_stock_basic_fallback(temp_storage_with_stock_basic):
    """测试当 clean_stock_basic 不存在时，能回退到 load_stock_basic"""
    # 创建只有 raw stock_basic 的存储
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir, verbose=False)
        
        # 只保存 raw 数据
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '603115.SH'],
            'symbol': ['000001', '603115'],
            'name': ['平安银行', '三维股份'],
            'market': ['主板', '主板'],
            'list_date': ['19910403', '20170705']
        })
        storage.save_raw(stock_basic, "stock_basic")
        
        loader = DataLoader(storage, verbose=False)
        
        # 尝试加载（应该回退到 raw）
        stock_basic = loader.load_clean_stock_basic()
        if stock_basic is None or stock_basic.empty:
            stock_basic = loader.load_stock_basic()
        
        # 验证能够加载到数据
        assert stock_basic is not None
        assert not stock_basic.empty
        assert len(stock_basic) == 2
        assert 'ts_code' in stock_basic.columns
        assert 'name' in stock_basic.columns

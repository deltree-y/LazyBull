"""停牌日历工具类测试"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.lazybull.common.suspend_calendar import SuspendCalendar
from src.lazybull.data.storage import Storage


class TestSuspendCalendar:
    """测试停牌日历工具类"""
    
    @pytest.fixture
    def temp_storage(self):
        """创建临时存储"""
        temp_dir = tempfile.mkdtemp()
        storage = Storage(root_path=temp_dir, verbose=False)
        yield storage
        # 清理临时目录
        shutil.rmtree(temp_dir)
    
    def _create_suspend_data(self, storage: Storage, trade_date: str, records: list):
        """创建模拟停牌数据
        
        Args:
            storage: Storage 实例
            trade_date: 交易日期
            records: 记录列表，每个记录是 (ts_code, suspend_type) 元组
        """
        df = pd.DataFrame([
            {'ts_code': ts_code, 'suspend_type': suspend_type, 'trade_date': trade_date}
            for ts_code, suspend_type in records
        ])
        storage.save_raw_by_date(df, "suspend", trade_date)
    
    def test_is_suspended_with_s_type(self, temp_storage):
        """测试：记录存在且 suspend_type='S' => 停牌"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "S"), ("000002.SZ", "R")]
        )
        
        # 测试停牌股票
        assert calendar.is_suspended("000001.SZ", "20260120") is True
        assert calendar.get_status_reason("000001.SZ", "20260120") == "停牌"
    
    def test_is_suspended_with_r_type(self, temp_storage):
        """测试：记录存在且 suspend_type='R' => 非停牌"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "R"), ("000002.SZ", "S")]
        )
        
        # 测试复牌股票
        assert calendar.is_suspended("000001.SZ", "20260120") is False
        assert calendar.get_status_reason("000001.SZ", "20260120") == "复牌"
    
    def test_is_suspended_no_record(self, temp_storage):
        """测试：当日无记录 => 非停牌"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据（只有部分股票）
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "S")]
        )
        
        # 测试无记录的股票
        assert calendar.is_suspended("000003.SZ", "20260120") is False
        assert calendar.get_status_reason("000003.SZ", "20260120") == "无记录"
    
    def test_suspend_file_missing_raises_exception(self, temp_storage):
        """测试：suspend 数据文件缺失时抛出异常（严格模式）"""
        calendar = SuspendCalendar(temp_storage)
        
        # 不创建任何停牌数据，直接查询
        with pytest.raises(FileNotFoundError) as exc_info:
            calendar.is_suspended("000001.SZ", "20260120")
        
        # 验证异常消息包含关键信息
        assert "停牌数据文件缺失" in str(exc_info.value)
        assert "20260120" in str(exc_info.value)
    
    def test_batch_is_suspended(self, temp_storage):
        """测试：批量判断停牌"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [
                ("000001.SZ", "S"),  # 停牌
                ("000002.SZ", "R"),  # 复牌
                ("000003.SZ", "S"),  # 停牌
            ]
        )
        
        # 批量查询
        ts_codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        result = calendar.batch_is_suspended(ts_codes, "20260120")
        
        # 验证结果
        assert result["000001.SZ"] is True   # 停牌
        assert result["000002.SZ"] is False  # 复牌
        assert result["000003.SZ"] is True   # 停牌
        assert result["000004.SZ"] is False  # 无记录
    
    def test_cache_mechanism(self, temp_storage):
        """测试：缓存机制（同一日期多次查询只加载一次）"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "S")]
        )
        
        # 第一次查询
        is_suspended_1 = calendar.is_suspended("000001.SZ", "20260120")
        
        # 第二次查询（应该使用缓存）
        is_suspended_2 = calendar.is_suspended("000001.SZ", "20260120")
        
        # 验证结果一致
        assert is_suspended_1 is True
        assert is_suspended_2 is True
        
        # 验证缓存中有数据
        assert "20260120" in calendar._cache
    
    def test_date_format_yyyymmdd(self, temp_storage):
        """测试：支持 YYYYMMDD 格式的日期"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据（使用 YYYYMMDD 格式）
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "S")]
        )
        
        # 使用 YYYYMMDD 格式查询
        assert calendar.is_suspended("000001.SZ", "20260120") is True
    
    def test_date_format_yyyy_mm_dd(self, temp_storage):
        """测试：支持 YYYY-MM-DD 格式的日期"""
        calendar = SuspendCalendar(temp_storage)
        
        # 创建停牌数据（使用 YYYYMMDD 格式存储）
        self._create_suspend_data(
            temp_storage, 
            "20260120", 
            [("000001.SZ", "S")]
        )
        
        # 使用 YYYY-MM-DD 格式查询（Storage 会自动转换）
        assert calendar.is_suspended("000001.SZ", "2026-01-20") is True

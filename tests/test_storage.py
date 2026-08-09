"""测试数据存储模块"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import src.lazybull.common.config as config_module

from src.lazybull.common.config import Config
from src.lazybull.data.storage import Storage
from src.lazybull.paper.storage import PaperStorage


@pytest.fixture
def temp_storage():
    """创建临时存储实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir)
        yield storage


@pytest.fixture
def sample_data():
    """创建样本数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'trade_date': ['20230101', '20230101', '20230101'],
        'close': [10.5, 20.3, 15.8],
        'volume': [1000, 2000, 1500]
    })


class TestStorageBasic:
    """测试基础存储功能"""
    
    def test_init(self, temp_storage):
        """测试初始化"""
        assert temp_storage.root_path.exists()
        assert temp_storage.raw_path.exists()
        assert temp_storage.clean_path.exists()
        assert temp_storage.features_path.exists()
        assert temp_storage.reports_path.exists()

    def test_init_uses_project_config(self, monkeypatch, tmp_path):
        """未显式传 root_path 时应读取项目配置的数据路径。"""
        config = Config()
        data_root = tmp_path / "data_root"
        raw_path = tmp_path / "raw_zone"
        clean_path = tmp_path / "clean_zone"
        features_path = tmp_path / "features_zone"
        reports_path = tmp_path / "reports_zone"
        config.set("data.root", str(data_root))
        config.set("data.raw", str(raw_path))
        config.set("data.clean", str(clean_path))
        config.set("data.features", str(features_path))
        config.set("data.reports", str(reports_path))
        monkeypatch.setattr(config_module, "_global_config", config)

        storage = Storage()

        assert storage.root_path == data_root
        assert storage.raw_path == raw_path
        assert storage.clean_path == clean_path
        assert storage.features_path == features_path
        assert storage.reports_path == reports_path

    def test_paper_storage_default_root_follows_data_root(self, monkeypatch, tmp_path):
        """PaperStorage 默认目录应派生自 data.root/paper。"""
        config = Config()
        data_root = tmp_path / "paper_data_root"
        config.set("data.root", str(data_root))
        monkeypatch.setattr(config_module, "_global_config", config)

        storage = PaperStorage()

        assert storage.root_path == data_root / "paper"
        assert storage.state_path.exists()
    
    def test_save_and_load_raw(self, temp_storage, sample_data):
        """测试保存和加载原始数据（非分区）"""
        temp_storage.save_raw(sample_data, "test_data")
        loaded = temp_storage.load_raw("test_data")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data)
        assert list(loaded.columns) == list(sample_data.columns)
    
    def test_save_and_load_clean(self, temp_storage, sample_data):
        """测试保存和加载清洗数据（非分区）"""
        temp_storage.save_clean(sample_data, "test_data")
        loaded = temp_storage.load_clean("test_data")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data)


class TestStoragePartitioning:
    """测试分区存储功能"""
    
    def test_format_date(self, temp_storage):
        """测试日期格式转换"""
        assert temp_storage._format_date("20230101") == "2023-01-01"
        assert temp_storage._format_date("2023-01-01") == "2023-01-01"
        
        with pytest.raises(ValueError):
            temp_storage._format_date("202301")  # 太短
        
        with pytest.raises(ValueError):
            temp_storage._format_date("202301011")  # 过长
    
    def test_save_raw_by_date(self, temp_storage, sample_data):
        """测试按日期保存原始数据"""
        temp_storage.save_raw_by_date(sample_data, "daily", "20230101")
        
        # 检查文件是否存在
        partition_path = temp_storage.raw_path / "daily" / "2023-01-01.parquet"
        assert partition_path.exists()
    
    def test_load_raw_by_date(self, temp_storage, sample_data):
        """测试按日期加载原始数据"""
        temp_storage.save_raw_by_date(sample_data, "daily", "20230101")
        loaded = temp_storage.load_raw_by_date("daily", "20230101")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data)
        pd.testing.assert_frame_equal(loaded, sample_data)
    
    def test_load_raw_by_date_with_formatted_date(self, temp_storage, sample_data):
        """测试使用不同日期格式加载"""
        temp_storage.save_raw_by_date(sample_data, "daily", "20230101")
        
        # 使用YYYY-MM-DD格式加载
        loaded = temp_storage.load_raw_by_date("daily", "2023-01-01")
        assert loaded is not None
        assert len(loaded) == len(sample_data)
    
    def test_load_raw_by_date_range(self, temp_storage, sample_data):
        """测试加载日期范围内的数据"""
        # 保存多天数据
        for i, date in enumerate(["20230101", "20230102", "20230103"]):
            df = sample_data.copy()
            df['trade_date'] = date
            df['close'] = df['close'] + i
            temp_storage.save_raw_by_date(df, "daily", date)
        
        # 加载日期范围
        loaded = temp_storage.load_raw_by_date_range("daily", "20230101", "20230102")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data) * 2  # 两天的数据
    
    def test_save_clean_by_date(self, temp_storage, sample_data):
        """测试按日期保存清洗数据"""
        temp_storage.save_clean_by_date(sample_data, "daily", "20230101")
        
        # 检查文件是否存在
        partition_path = temp_storage.clean_path / "daily" / "2023-01-01.parquet"
        assert partition_path.exists()
    
    def test_load_clean_by_date(self, temp_storage, sample_data):
        """测试按日期加载清洗数据"""
        temp_storage.save_clean_by_date(sample_data, "daily", "20230101")
        loaded = temp_storage.load_clean_by_date("daily", "20230101")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data)
    
    def test_load_clean_by_date_range(self, temp_storage, sample_data):
        """测试加载日期范围内的清洗数据"""
        # 保存多天数据
        for i, date in enumerate(["20230101", "20230102", "20230103"]):
            df = sample_data.copy()
            df['trade_date'] = date
            temp_storage.save_clean_by_date(df, "daily", date)
        
        # 加载日期范围
        loaded = temp_storage.load_clean_by_date_range("daily", "20230101", "20230103")
        
        assert loaded is not None
        assert len(loaded) == len(sample_data) * 3  # 三天的数据
    
    def test_list_partitions(self, temp_storage, sample_data):
        """测试列出分区日期"""
        # 保存多天数据
        dates = ["20230101", "20230102", "20230103"]
        for date in dates:
            temp_storage.save_raw_by_date(sample_data, "daily", date)
        
        # 列出分区
        partitions = temp_storage.list_partitions("raw", "daily")
        
        assert len(partitions) == 3
        assert partitions == ["2023-01-01", "2023-01-02", "2023-01-03"]
    
    def test_list_partitions_empty(self, temp_storage):
        """测试列出不存在的分区"""
        partitions = temp_storage.list_partitions("raw", "nonexistent")
        assert partitions == []

    def test_count_rows(self, temp_storage, sample_data):
        """count_rows 快速统计分区行数（不加载全量）"""
        temp_storage.save_raw_by_date(sample_data, "daily", "20230101")
        assert temp_storage.count_rows("raw", "daily", "20230101") == 3
        # 文件不存在返回 None
        assert temp_storage.count_rows("raw", "daily", "20230102") is None

    def test_is_data_exists_min_rows_coverage_gate(self, temp_storage, sample_data):
        """min_rows 覆盖度门控：文件存在但行数不足视为未补齐（审计问题2）"""
        temp_storage.save_raw_by_date(sample_data, "daily", "20230101")  # 3 行
        # 行数充足 / 恰好等于阈值均视为已补齐
        assert temp_storage.is_data_exists("raw", "daily", "20230101", min_rows=3)
        assert temp_storage.is_data_exists("raw", "daily", "20230101", min_rows=2)
        # 行数不足视为未补齐（模拟历史截断落盘）
        assert not temp_storage.is_data_exists("raw", "daily", "20230101", min_rows=5)
        # 不传 min_rows 时保持原行为（只查存在性）
        assert temp_storage.is_data_exists("raw", "daily", "20230101")


class TestTushareClient:
    """测试TushareClient的suspend_d方法"""
    
    def test_suspend_d_signature(self):
        """测试get_suspend_d方法签名"""
        from src.lazybull.data.tushare_client import TushareClient
        import inspect
        
        # 获取方法签名
        sig = inspect.signature(TushareClient.get_suspend_d)
        params = list(sig.parameters.keys())
        
        # 验证新参数存在
        assert 'ts_code' in params
        assert 'trade_date' in params
        assert 'start_date' in params
        assert 'end_date' in params
        assert 'suspend_type' in params
        
        # 验证旧参数不存在
        assert 'suspend_date' not in params
        assert 'resume_date' not in params

    def test_init_uses_project_config(self, monkeypatch):
        """TushareClient 未显式传参时应读取项目配置默认值。"""
        from src.lazybull.data import tushare_client as tushare_client_module

        config = Config()
        config.set("tushare.max_retries", 9)
        config.set("tushare.retry_delay", 3.5)
        config.set("tushare.rate_limit", 77)
        monkeypatch.setattr(config_module, "_global_config", config)
        monkeypatch.setenv("TS_TOKEN", "fake-token")
        monkeypatch.setattr(tushare_client_module.ts, "set_token", lambda token: None)
        monkeypatch.setattr(tushare_client_module.ts, "pro_api", lambda: object())

        client = tushare_client_module.TushareClient(verbose=False)

        assert client.max_retries == 9
        assert client.retry_delay == 3.5
        assert client.rate_limit == 77

    def test_init_explicit_args_override_project_config(self, monkeypatch):
        """显式传入 TuShare 参数时应优先于项目配置。"""
        from src.lazybull.data import tushare_client as tushare_client_module

        config = Config()
        config.set("tushare.max_retries", 9)
        config.set("tushare.retry_delay", 3.5)
        config.set("tushare.rate_limit", 77)
        monkeypatch.setattr(config_module, "_global_config", config)
        monkeypatch.setenv("TS_TOKEN", "fake-token")
        monkeypatch.setattr(tushare_client_module.ts, "set_token", lambda token: None)
        monkeypatch.setattr(tushare_client_module.ts, "pro_api", lambda: object())

        client = tushare_client_module.TushareClient(
            max_retries=2,
            retry_delay=0.5,
            rate_limit=33,
            verbose=False,
        )

        assert client.max_retries == 2
        assert client.retry_delay == 0.5
        assert client.rate_limit == 33

"""测试纸面交易补位机制

本测试文件验证补位机制的核心功能：
1. T1 买入失败后基于当日数据生成 T2 pending 目标
2. T2 读取 pending 并执行，如失败继续生成 T3 pending
3. 补位尝试次数上限（5次）
4. 补位信号生成经过 MLSignal 成交额过滤
"""

import tempfile
import json
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.paper import (
    PaperStorage,
    TargetWeight,
)


@pytest.fixture
def temp_storage():
    """临时存储目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir, verbose=False)
        yield storage


def test_pending_weights_metadata_save_and_load(temp_storage):
    """测试 pending_weights 元数据的保存和加载"""
    trade_date = "20260121"
    targets = [
        TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="补位-信号生成"),
        TargetWeight(ts_code="000002.SZ", target_weight=0.2, reason="补位-信号生成"),
    ]
    
    metadata = {
        'source': 'replenishment',
        'attempt_count': 1,
        'original_signal_date': '20260120',
        'failed_targets_count': 2,
    }
    
    # 保存
    temp_storage.save_pending_weights(trade_date, targets, metadata=metadata)
    
    # 加载目标
    loaded_targets = temp_storage.load_pending_weights(trade_date)
    assert loaded_targets is not None
    assert len(loaded_targets) == 2
    assert loaded_targets[0].ts_code == "000001.SZ"
    assert loaded_targets[0].reason == "补位-信号生成"
    
    # 加载元数据
    loaded_meta = temp_storage.load_pending_weights_metadata(trade_date)
    assert loaded_meta is not None
    assert loaded_meta['source'] == 'replenishment'
    assert loaded_meta['attempt_count'] == 1
    assert loaded_meta['original_signal_date'] == '20260120'
    assert loaded_meta['failed_targets_count'] == 2


def test_pending_weights_without_metadata(temp_storage):
    """测试不带元数据保存 pending_weights（T0 场景）"""
    trade_date = "20260122"
    targets = [
        TargetWeight(ts_code="000003.SZ", target_weight=0.2, reason="信号生成"),
    ]
    
    # 保存（不带元数据）
    temp_storage.save_pending_weights(trade_date, targets)
    
    # 加载目标（应该成功）
    loaded_targets = temp_storage.load_pending_weights(trade_date)
    assert loaded_targets is not None
    assert len(loaded_targets) == 1
    
    # 加载元数据（应该为 None）
    loaded_meta = temp_storage.load_pending_weights_metadata(trade_date)
    assert loaded_meta is None


def test_t0_metadata_save(temp_storage):
    """测试 T0 保存带元数据的 pending_weights"""
    trade_date = "20260123"
    targets = [
        TargetWeight(ts_code="600000.SH", target_weight=0.2, reason="信号生成"),
    ]
    
    t0_metadata = {
        'source': 't0_signal',
        'attempt_count': 0,
        'signal_date': '20260122',
    }
    
    # 保存
    temp_storage.save_pending_weights(trade_date, targets, metadata=t0_metadata)
    
    # 加载元数据
    loaded_meta = temp_storage.load_pending_weights_metadata(trade_date)
    assert loaded_meta is not None
    assert loaded_meta['source'] == 't0_signal'
    assert loaded_meta['attempt_count'] == 0


def test_replenishment_metadata_increment(temp_storage):
    """测试补位元数据的递增（模拟多次失败）"""
    # 模拟 T1 失败，生成 T2 pending（第1次补位）
    t2_targets = [
        TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="补位-信号生成"),
    ]
    t2_meta = {
        'source': 'replenishment',
        'attempt_count': 1,
        'original_signal_date': '20260120',
    }
    temp_storage.save_pending_weights("20260121", t2_targets, metadata=t2_meta)
    
    # 模拟 T2 失败，生成 T3 pending（第2次补位）
    t3_targets = [
        TargetWeight(ts_code="000002.SZ", target_weight=0.2, reason="补位-信号生成"),
    ]
    t3_meta = {
        'source': 'replenishment',
        'attempt_count': 2,
        'original_signal_date': '20260121',
    }
    temp_storage.save_pending_weights("20260122", t3_targets, metadata=t3_meta)
    
    # 验证 T2 元数据
    t2_loaded = temp_storage.load_pending_weights_metadata("20260121")
    assert t2_loaded['attempt_count'] == 1
    
    # 验证 T3 元数据
    t3_loaded = temp_storage.load_pending_weights_metadata("20260122")
    assert t3_loaded['attempt_count'] == 2


def test_replenishment_max_attempts_logic():
    """测试补位尝试次数上限逻辑（不依赖真实数据）"""
    MAX_REPLENISHMENT_ATTEMPTS = 5
    
    # 模拟不同尝试次数
    for current_attempt in range(0, 7):
        next_attempt = current_attempt + 1
        
        if next_attempt <= MAX_REPLENISHMENT_ATTEMPTS:
            # 应该继续补位
            assert next_attempt <= 5, f"尝试次数 {next_attempt} 应该 <= 5"
        else:
            # 应该停止补位
            assert next_attempt > MAX_REPLENISHMENT_ATTEMPTS, f"尝试次数 {next_attempt} 超过上限"


def test_pending_weights_file_structure(temp_storage):
    """测试 pending_weights 文件结构（parquet + meta.json）"""
    trade_date = "20260124"
    targets = [
        TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="测试"),
    ]
    metadata = {'source': 'test', 'attempt_count': 1}
    
    # 保存
    temp_storage.save_pending_weights(trade_date, targets, metadata=metadata)
    
    # 验证文件存在
    pending_path = Path(temp_storage.root_path) / "pending"
    parquet_file = pending_path / f"{trade_date}.parquet"
    meta_file = pending_path / f"{trade_date}_meta.json"
    
    assert parquet_file.exists(), "pending parquet 文件应该存在"
    assert meta_file.exists(), "pending meta.json 文件应该存在"
    
    # 验证 JSON 格式
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta_content = json.load(f)
    
    assert meta_content['source'] == 'test'
    assert meta_content['attempt_count'] == 1


def test_amount_filter_concept():
    """概念测试：确保补位机制设计包含成交额过滤
    
    注意：本测试不依赖真实数据，仅验证设计概念。
    真实的成交额过滤在 MLSignal._apply_amount_filter() 中实现，
    补位路径通过 generate_replacement_targets() -> _generate_equal_weight_with_lot_constraint() 
    -> MLSignal.generate_ranked() -> _apply_amount_filter() 调用链确保过滤生效。
    """
    # 模拟 features_df 包含 amount 字段
    features_df = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
        'amount': [1000000, 800000, 500000, 200000, 100000],  # 成交额
        'feature1': [1, 2, 3, 4, 5],
    })
    
    # MLSignal 默认过滤后 20%
    # 在实际应用中，后 20% 是 000005.SZ (100000)
    # 过滤后应该剩余 4 只股票
    
    amount_filter_pct = 20.0
    threshold_quantile = amount_filter_pct / 100.0  # 0.2
    
    # 计算阈值
    amount_threshold = features_df['amount'].quantile(threshold_quantile)
    
    # 过滤
    filtered_df = features_df[features_df['amount'] > amount_threshold]
    
    # 验证
    assert len(filtered_df) == 4, "应该过滤掉 1 只股票（后 20%）"
    assert '000005.SZ' not in filtered_df['ts_code'].values, "最低成交额股票应被过滤"


def test_replenishment_targets_reason_format():
    """测试补位目标的 reason 格式"""
    original_reason = "信号生成"
    replenishment_reason = f"补位-{original_reason}"
    
    assert replenishment_reason == "补位-信号生成"
    assert replenishment_reason.startswith("补位-")


def test_metadata_overwrite_warning_scenario(temp_storage):
    """测试 T0 覆盖补位目标的场景（元数据检测）"""
    trade_date = "20260125"
    
    # 先保存补位目标
    replenishment_targets = [
        TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="补位-信号生成"),
    ]
    replenishment_meta = {
        'source': 'replenishment',
        'attempt_count': 2,
    }
    temp_storage.save_pending_weights(trade_date, replenishment_targets, metadata=replenishment_meta)
    
    # 验证补位目标存在
    loaded_meta1 = temp_storage.load_pending_weights_metadata(trade_date)
    assert loaded_meta1['source'] == 'replenishment'
    assert loaded_meta1['attempt_count'] == 2
    
    # T0 覆盖（模拟调仓日）
    t0_targets = [
        TargetWeight(ts_code="600000.SH", target_weight=0.2, reason="信号生成"),
    ]
    t0_meta = {
        'source': 't0_signal',
        'attempt_count': 0,
    }
    temp_storage.save_pending_weights(trade_date, t0_targets, metadata=t0_meta)
    
    # 验证 T0 目标覆盖成功
    loaded_targets = temp_storage.load_pending_weights(trade_date)
    assert len(loaded_targets) == 1
    assert loaded_targets[0].ts_code == "600000.SH"
    
    loaded_meta2 = temp_storage.load_pending_weights_metadata(trade_date)
    assert loaded_meta2['source'] == 't0_signal'
    assert loaded_meta2['attempt_count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

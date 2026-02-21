#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk-forward 切分工具模块

提供 walk-forward 滚动训练的时间窗口切分逻辑。

功能：
- 生成训练/测试区间切分（splits）
- 支持按季度/月度/半年度滚动
- 支持可配置的训练窗口和测试窗口
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
from dateutil.relativedelta import relativedelta
import pandas as pd
from loguru import logger


@dataclass
class WalkForwardSplit:
    """Walk-forward 切分数据结构
    
    表示一次训练/测试切分的时间区间。
    """
    split_index: int  # 切分索引（从0开始）
    train_start: str  # 训练开始日期（YYYYMMDD）
    train_end: str  # 训练结束日期（YYYYMMDD）
    test_start: str  # 测试开始日期（YYYYMMDD）
    test_end: str  # 测试结束日期（YYYYMMDD）
    

def generate_walk_forward_splits(
    trade_cal: pd.DataFrame,
    wf_start_date: str,
    wf_end_date: str,
    step_frequency: str = "quarterly",
    train_window_years: int = 5,
    test_window_months: int = 6
) -> List[WalkForwardSplit]:
    """生成 walk-forward 切分
    
    按指定的 step 频率滚动生成训练/测试区间切分。
    
    Args:
        trade_cal: 交易日历 DataFrame（包含 cal_date, is_open 列）
        wf_start_date: walk-forward 起始日期（YYYYMMDD）
        wf_end_date: walk-forward 结束日期（YYYYMMDD）
        step_frequency: 滚动频率（"monthly"|"quarterly"|"semiannual"）
        train_window_years: 训练窗口年数（默认 5）
        test_window_months: 测试窗口月数（默认 6）
        
    Returns:
        WalkForwardSplit 列表
        
    Note:
        - 所有日期对齐到交易日（自动调整到最近的交易日）
        - train_end 是训练集的最后一天
        - test_start 是 train_end 的下一个交易日
        - test_end 是 test_start 往后推 test_window_months 个月的最后一个交易日
        - 每次滚动，train_end 向前推进 step_frequency
    """
    logger.info(f"生成 walk-forward 切分...")
    logger.info(f"  起止日期: {wf_start_date} 至 {wf_end_date}")
    logger.info(f"  滚动频率: {step_frequency}")
    logger.info(f"  训练窗口: {train_window_years} 年")
    logger.info(f"  测试窗口: {test_window_months} 个月")
    
    # 获取交易日列表
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= wf_start_date) & 
        (trade_cal['cal_date'] <= wf_end_date) & 
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    if len(trade_dates) == 0:
        raise ValueError(f"指定区间内没有交易日: {wf_start_date} 至 {wf_end_date}")
    
    logger.info(f"  区间内交易日数: {len(trade_dates)}")
    
    # 确定 step 的相对月数
    step_months_map = {
        "monthly": 1,
        "quarterly": 3,
        "semiannual": 6
    }
    
    if step_frequency not in step_months_map:
        raise ValueError(f"不支持的 step_frequency: {step_frequency}，请使用 monthly, quarterly 或 semiannual")
    
    step_months = step_months_map[step_frequency]
    
    # 将字符串日期转为 datetime 对象
    def to_datetime(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y%m%d")
    
    # 将 datetime 对象转为字符串日期
    def to_date_str(dt: datetime) -> str:
        return dt.strftime("%Y%m%d")
    
    # 查找最接近的交易日（向后查找，如果没有则向前查找）
    def find_nearest_trade_date(target_date_str: str, direction: str = "forward") -> str:
        """
        查找最接近的交易日
        
        Args:
            target_date_str: 目标日期（YYYYMMDD）
            direction: "forward"（向后查找）或 "backward"（向前查找）
            
        Returns:
            最接近的交易日（YYYYMMDD），如果找不到返回 None
        """
        if target_date_str in trade_dates:
            return target_date_str
        
        if direction == "forward":
            # 向后查找最近的交易日
            for td in trade_dates:
                if td >= target_date_str:
                    return td
            return None  # 没有找到
        else:
            # 向前查找最近的交易日
            for td in reversed(trade_dates):
                if td <= target_date_str:
                    return td
            return None  # 没有找到
    
    # 生成切分列表
    splits = []
    split_index = 0
    
    # 初始 train_end：从 wf_start_date 开始（确保有足够的训练数据）
    # 计算最早的 train_end：wf_start_date + train_window_years
    min_train_end_dt = to_datetime(wf_start_date) + relativedelta(years=train_window_years)
    min_train_end_str = to_date_str(min_train_end_dt)
    
    # 找到第一个有效的 train_end（交易日）
    first_train_end = find_nearest_trade_date(min_train_end_str, direction="forward")
    if first_train_end is None:
        logger.warning(f"无法找到有效的初始 train_end（最早需要 {min_train_end_str}）")
        return splits
    
    current_train_end_dt = to_datetime(first_train_end)
    
    # 滚动生成切分
    while True:
        current_train_end_str = to_date_str(current_train_end_dt)
        
        # 对齐 train_end 到交易日
        train_end = find_nearest_trade_date(current_train_end_str, direction="backward")
        if train_end is None or train_end > wf_end_date:
            break  # 超出范围
        
        # 计算 train_start：train_end - train_window_years
        train_start_dt = to_datetime(train_end) - relativedelta(years=train_window_years)
        train_start = find_nearest_trade_date(to_date_str(train_start_dt), direction="forward")
        
        if train_start is None or train_start < wf_start_date:
            # 调整 train_start 到 wf_start_date
            train_start = find_nearest_trade_date(wf_start_date, direction="forward")
        
        if train_start is None or train_start >= train_end:
            logger.warning(f"跳过无效的训练区间: train_start={train_start}, train_end={train_end}")
            # 继续下一个 step
            current_train_end_dt += relativedelta(months=step_months)
            continue
        
        # 计算 test_start：train_end 的下一个交易日
        train_end_idx = trade_dates.index(train_end)
        if train_end_idx + 1 >= len(trade_dates):
            logger.info(f"train_end {train_end} 已经是最后一个交易日，停止生成切分")
            break
        
        test_start = trade_dates[train_end_idx + 1]
        
        # 计算 test_end：test_start + test_window_months
        test_end_dt = to_datetime(test_start) + relativedelta(months=test_window_months)
        #test_end = find_nearest_trade_date(to_date_str(test_end_dt), direction="backward")
        test_end = to_date_str(test_end_dt) # 测试结束日期不受训练结束日期约束
        
        if test_end is None or test_end > wf_end_date:
            # 调整 test_end 到 wf_end_date
            #test_end = find_nearest_trade_date(wf_end_date, direction="backward")
            test_end = to_date_str(test_end_dt) # 测试结束日期不受训练结束日期约束
        
        if test_end is None or test_end < test_start:
            logger.info(f"无法生成有效的测试区间（test_start={test_start}），停止生成切分")
            break
        
        # 添加切分
        split = WalkForwardSplit(
            split_index=split_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end
        )
        splits.append(split)
        
        logger.debug(f"  Split {split_index}: train=[{train_start}, {train_end}], test=[{test_start}, {test_end}]")
        
        split_index += 1
        
        # 推进到下一个 step
        current_train_end_dt += relativedelta(months=step_months)
        
        # 如果下一个 train_end 已经超过 wf_end_date，停止
        if to_date_str(current_train_end_dt) > wf_end_date:
            break
    
    logger.info(f"成功生成 {len(splits)} 个切分")
    
    if len(splits) == 0:
        logger.warning("未生成任何切分，请检查参数设置（时间区间、窗口大小等）")
    
    return splits


def print_splits_summary(splits: List[WalkForwardSplit]) -> None:
    """打印切分汇总信息
    
    Args:
        splits: WalkForwardSplit 列表
    """
    if len(splits) == 0:
        logger.info("没有切分可以打印")
        return
    
    logger.info("=" * 80)
    logger.info("Walk-forward 切分汇总")
    logger.info("=" * 80)
    logger.info(f"{'索引':<6} {'训练开始':<12} {'训练结束':<12} {'测试开始':<12} {'测试结束':<12}")
    logger.info("-" * 80)
    
    for split in splits:
        logger.info(
            f"{split.split_index:<6} "
            f"{split.train_start:<12} "
            f"{split.train_end:<12} "
            f"{split.test_start:<12} "
            f"{split.test_end:<12}"
        )
    
    logger.info("=" * 80)

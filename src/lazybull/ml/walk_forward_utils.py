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
from typing import List, Optional, Tuple
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
    test_window_months: int = 6,
    rebalance_freq: Optional[int] = None
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
        - test_end 是 test_start 往后推 test_window_months 个月的最后一个交易日，
          若提供 rebalance_freq，则进一步向后对齐到第一个不早于该日期的调仓日
        - 每次滚动，train_end 向前推进 step_frequency
        - 最后一个 split 的 test_end 不超过 wf_end_date
    """
    logger.info(f"生成 walk-forward 切分...")
    logger.info(f"  起止日期: {wf_start_date} 至 {wf_end_date}")
    logger.info(f"  滚动频率: {step_frequency}")
    logger.info(f"  训练窗口: {train_window_years} 年")
    logger.info(f"  测试窗口: {test_window_months} 个月")
    
    # 获取 wf 区间内的交易日列表（用于 train 期操作）
    trade_dates = trade_cal[
        (trade_cal['cal_date'] >= wf_start_date) &
        (trade_cal['cal_date'] <= wf_end_date) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()

    if len(trade_dates) == 0:
        raise ValueError(f"指定区间内没有交易日: {wf_start_date} 至 {wf_end_date}")

    # 获取所有可用交易日（不受 wf_end_date 限制，用于 test 期查找）
    all_trade_dates = trade_cal[
        trade_cal['is_open'] == 1
    ]['cal_date'].tolist()
    
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
    def find_nearest_trade_date(target_date_str: str, direction: str = "forward", date_list=None) -> str:
        """
        查找最接近的交易日

        Args:
            target_date_str: 目标日期（YYYYMMDD）
            direction: "forward"（向后查找）或 "backward"（向前查找）
            date_list: 使用的交易日列表，默认使用 wf 区间内的 trade_dates

        Returns:
            最接近的交易日（YYYYMMDD），如果找不到返回 None
        """
        if date_list is None:
            date_list = trade_dates

        if target_date_str in date_list:
            return target_date_str

        if direction == "forward":
            # 向后查找最近的交易日
            for td in date_list:
                if td >= target_date_str:
                    return td
            return None  # 没有找到
        else:
            # 向前查找最近的交易日
            for td in reversed(date_list):
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
        
        # 计算 test_start：train_end 的下一个交易日（使用全量交易日）
        train_end_idx_all = all_trade_dates.index(train_end)
        if train_end_idx_all + 1 >= len(all_trade_dates):
            logger.info(f"train_end {train_end} 已经是最后一个可用交易日，停止生成切分")
            break

        test_start = all_trade_dates[train_end_idx_all + 1]

        # 计算 test_end：test_start + test_window_months，向前对齐至最近交易日（使用全量交易日，不受 wf_end_date 限制）
        test_end_dt = to_datetime(test_start) + relativedelta(months=test_window_months)
        test_end = find_nearest_trade_date(to_date_str(test_end_dt), direction="backward", date_list=all_trade_dates)

        if test_end is None or test_end < test_start:
            logger.info(f"无法生成有效的测试区间（test_start={test_start}），停止生成切分")
            break

        # 若提供了调仓频率，将 test_end 向后对齐到第一个不早于当前 test_end 的调仓日
        if rebalance_freq is not None and rebalance_freq > 0:
            test_start_idx = all_trade_dates.index(test_start)
            k = 1
            while True:
                rebal_idx = test_start_idx + k * rebalance_freq - 1
                if rebal_idx >= len(all_trade_dates):
                    break
                candidate = all_trade_dates[rebal_idx]
                if candidate >= test_end:
                    test_end = candidate
                    break
                k += 1

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

    # 将最后一个 split 的 test_end 限制在 wf_end_date（不允许超出）
    wf_end_capped = find_nearest_trade_date(wf_end_date, direction="backward", date_list=all_trade_dates)
    if wf_end_capped and splits[-1].test_end > wf_end_capped:
        old_end = splits[-1].test_end
        splits[-1] = WalkForwardSplit(
            split_index=splits[-1].split_index,
            train_start=splits[-1].train_start,
            train_end=splits[-1].train_end,
            test_start=splits[-1].test_start,
            test_end=wf_end_capped
        )
        logger.info(f"  最后一个 split 的 test_end 从 {old_end} 限制到 {wf_end_capped}（wf_end_date）")

    return splits


def generate_walk_forward_splits_by_count(
    trade_cal: pd.DataFrame,
    split_count: int,
    final_date: str,
    step_frequency: str = "quarterly",
    train_window_years: int = 5,
    test_window_months: int = 6,
    rebalance_freq: Optional[int] = None,
) -> List[WalkForwardSplit]:
    """按切分数量和最终日期反推 walk-forward 切分。

    Args:
        trade_cal: 交易日历 DataFrame（包含 cal_date, is_open 列）
        split_count: 切分数量（必须 > 0）
        final_date: 最终日期（YYYYMMDD）
        step_frequency: 滚动频率（"monthly"|"quarterly"|"semiannual"）
        train_window_years: 训练窗口年数
        test_window_months: 测试窗口月数
        rebalance_freq: 调仓频率（交易日）。若提供则将 test_end 向后对齐到调仓边界

    Returns:
        WalkForwardSplit 列表（按 split_index 升序）

    Note:
        - 不做“末尾强制截断”补丁，避免出现 test_start > test_end 的无效切分
        - 最后一段 split 的 test_end 会尽量贴近 final_date 且不超过 final_date
    """
    if split_count <= 0:
        raise ValueError(f"split_count 必须为正整数，当前值: {split_count}")

    step_months_map = {
        "monthly": 1,
        "quarterly": 3,
        "semiannual": 6,
    }
    if step_frequency not in step_months_map:
        raise ValueError(
            f"不支持的 step_frequency: {step_frequency}，请使用 monthly, quarterly 或 semiannual"
        )

    all_trade_dates = trade_cal[
        trade_cal['is_open'] == 1
    ]['cal_date'].sort_values().tolist()
    if len(all_trade_dates) == 0:
        raise ValueError("交易日历为空，无法生成切分")

    step_months = step_months_map[step_frequency]

    def to_datetime(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y%m%d")

    def to_date_str(dt: datetime) -> str:
        return dt.strftime("%Y%m%d")

    def find_nearest_trade_date(target_date_str: str, direction: str) -> Optional[str]:
        if target_date_str in all_trade_dates:
            return target_date_str

        if direction == "forward":
            for td in all_trade_dates:
                if td >= target_date_str:
                    return td
            return None

        for td in reversed(all_trade_dates):
            if td <= target_date_str:
                return td
        return None

    def previous_trade_date(date_str: str) -> Optional[str]:
        try:
            idx = all_trade_dates.index(date_str)
        except ValueError:
            return None
        if idx <= 0:
            return None
        return all_trade_dates[idx - 1]

    def build_split_from_train_end(
        train_end: str,
        test_end_upper_bound: Optional[str] = None,
    ) -> Optional[Tuple[str, str, str]]:
        """根据 train_end 计算 train_start/test_start/test_end。"""
        try:
            train_end_idx = all_trade_dates.index(train_end)
        except ValueError:
            return None

        if train_end_idx + 1 >= len(all_trade_dates):
            return None

        test_start = all_trade_dates[train_end_idx + 1]

        # test_end 先按窗口长度回溯（与正向逻辑保持一致）
        test_end_dt = to_datetime(test_start) + relativedelta(months=test_window_months)
        test_end = find_nearest_trade_date(to_date_str(test_end_dt), direction="backward")

        if test_end is None or test_end < test_start:
            return None

        if test_end_upper_bound is not None and test_end > test_end_upper_bound:
            return None

        # 若提供调仓频率，将 test_end 向后对齐到第一个不早于当前 test_end 的调仓日
        if rebalance_freq is not None and rebalance_freq > 0:
            test_start_idx = all_trade_dates.index(test_start)
            k = 1
            while True:
                rebal_idx = test_start_idx + k * rebalance_freq - 1
                if rebal_idx >= len(all_trade_dates):
                    break
                candidate = all_trade_dates[rebal_idx]
                if candidate >= test_end:
                    test_end = candidate
                    break
                k += 1

            if test_end_upper_bound is not None and test_end > test_end_upper_bound:
                return None

        train_start_dt = to_datetime(train_end) - relativedelta(years=train_window_years)
        train_start = find_nearest_trade_date(to_date_str(train_start_dt), direction="forward")

        if train_start is None or train_start >= train_end:
            return None

        return train_start, test_start, test_end

    def find_latest_valid_train_end(
        search_start: str,
        test_end_upper_bound: str,
    ) -> Tuple[str, Tuple[str, str, str]]:
        try:
            start_idx = all_trade_dates.index(search_start)
        except ValueError:
            aligned_search_start = find_nearest_trade_date(search_start, direction="backward")
            if aligned_search_start is None:
                raise ValueError(f"无法对齐 train_end 搜索起点: {search_start}")
            start_idx = all_trade_dates.index(aligned_search_start)

        for idx in range(start_idx, -1, -1):
            candidate_train_end = all_trade_dates[idx]
            built = build_split_from_train_end(candidate_train_end, test_end_upper_bound)
            if built is None:
                continue

            train_start, test_start, test_end = built
            if train_start < candidate_train_end and test_start <= test_end:
                return candidate_train_end, built

        raise ValueError(
            f"无法在 train_end<={search_start} 且 test_end<={test_end_upper_bound} 条件下找到有效切分"
        )

    aligned_final_date = find_nearest_trade_date(final_date, direction="backward")
    if aligned_final_date is None:
        raise ValueError(f"无法将 final_date 对齐到交易日: {final_date}")

    reverse_splits: List[WalkForwardSplit] = []
    # 从最后一个 split 开始，向前逐段回推并确保测试区间不重叠
    latest_search_start = previous_trade_date(aligned_final_date)
    if latest_search_start is None:
        raise ValueError(
            f"final_date={aligned_final_date} 之前没有可用交易日，无法生成切分"
        )

    current_train_end, current_built = find_latest_valid_train_end(
        search_start=latest_search_start,
        test_end_upper_bound=aligned_final_date,
    )
    newer_split_test_start = None

    for rev_idx in range(split_count):
        if rev_idx > 0:
            if newer_split_test_start is None:
                raise ValueError("内部状态错误：缺少下一段 test_start 上界")

            no_overlap_upper_bound = previous_trade_date(newer_split_test_start)
            if no_overlap_upper_bound is None:
                raise ValueError(
                    "切分数量过大：无法为更早 split 提供不重叠测试区间"
                )

            prev_train_end_target = to_datetime(current_train_end) - relativedelta(months=step_months)
            prev_train_end_target_str = to_date_str(prev_train_end_target)
            search_start = find_nearest_trade_date(prev_train_end_target_str, direction="backward")

            if search_start is None:
                raise ValueError(
                    f"切分数量过大，无法继续回推（当前 split_count={split_count}）"
                )

            if search_start >= current_train_end:
                search_start = previous_trade_date(current_train_end)
                if search_start is None:
                    raise ValueError(
                        f"切分数量过大，无法继续回推（当前 split_count={split_count}）"
                    )

            current_train_end, current_built = find_latest_valid_train_end(
                search_start=search_start,
                test_end_upper_bound=no_overlap_upper_bound,
            )

        train_start, test_start, test_end = current_built
        newer_split_test_start = test_start

        split_index = split_count - 1 - rev_idx
        reverse_splits.append(
            WalkForwardSplit(
                split_index=split_index,
                train_start=train_start,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    splits = sorted(reverse_splits, key=lambda x: x.split_index)

    logger.info(f"成功生成 {len(splits)} 个切分（按数量反推）")
    if len(splits) > 0:
        logger.info(
            f"  末尾约束: final_date={final_date}（对齐后={aligned_final_date}）, "
            f"last_split.test_end={splits[-1].test_end}"
        )

    return splits


def resolve_deploy_train_window(
    trade_cal: pd.DataFrame,
    deploy_train_end: str,
    train_window_years: int,
) -> Tuple[Optional[str], Optional[str]]:
    """解析部署模型训练区间并对齐到交易日。

    Args:
        trade_cal: 交易日历 DataFrame（包含 cal_date, is_open 列）
        deploy_train_end: 部署训练目标结束日期（YYYYMMDD）
        train_window_years: 训练窗口年数

    Returns:
        (train_start, train_end)；任一无法解析时返回 None
    """
    all_trade_dates = trade_cal[
        trade_cal['is_open'] == 1
    ]['cal_date'].sort_values().tolist()

    if len(all_trade_dates) == 0:
        return None, None

    train_start_dt = datetime.strptime(deploy_train_end, "%Y%m%d") - relativedelta(
        years=train_window_years
    )
    train_start_str = train_start_dt.strftime("%Y%m%d")

    train_start = None
    for td in all_trade_dates:
        if td >= train_start_str:
            train_start = td
            break

    train_end = None
    for td in reversed(all_trade_dates):
        if td <= deploy_train_end:
            train_end = td
            break

    return train_start, train_end


def print_splits_summary(
    splits: List[WalkForwardSplit],
    deploy_train_start: Optional[str] = None,
    deploy_train_end: Optional[str] = None,
) -> None:
    """打印切分汇总信息
    
    Args:
        splits: WalkForwardSplit 列表
        deploy_train_start: 部署训练开始日期（可选）
        deploy_train_end: 部署训练结束日期（可选）
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

    if deploy_train_start and deploy_train_end:
        logger.info("-" * 80)
        logger.info(
            f"{'部署训练':<6} "
            f"{deploy_train_start:<12} "
            f"{deploy_train_end:<12} "
            f"{'-':<12} "
            f"{'-':<12}"
        )
    
    logger.info("=" * 80)

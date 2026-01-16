#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征构建脚本

按日生成截面训练特征和标签，保存到 data/features/cs_train/{YYYYMMDD}.parquet
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
from src.lazybull.features import FeatureBuilder


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建日频截面特征与标签")
    
    parser.add_argument(
        "--start_date",
        type=str,
        default="20230101",
        help="开始日期，格式YYYYMMDD（默认：20230101）"
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        default="20231231",
        help="结束日期，格式YYYYMMDD（默认：20231231）"
    )
    
    parser.add_argument(
        "--min_list_days",
        type=int,
        default=60,
        help="最小上市天数（默认：60）"
    )
    
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="预测时间窗口（交易日）（默认：5）"
    )
    
    parser.add_argument(
        "--pull_data",
        action="store_true",
        help="是否先拉取所需数据（默认：否）"
    )
    
    parser.add_argument(
        "--auto_build_clean",
        action="store_true",
        default=True,
        help="当 clean 数据不存在时自动从 raw 构建（默认：是）"
    )
    
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="强制使用 raw 数据而不是 clean 数据（默认：否）"
    )
    
    return parser.parse_args()


def pull_required_data(client: TushareClient, storage: Storage, start_date: str, end_date: str):
    """拉取构建特征所需的数据
    
    Args:
        client: TuShare客户端
        storage: 存储实例
        start_date: 开始日期
        end_date: 结束日期
    """
    logger.info("=" * 60)
    logger.info("开始拉取所需数据")
    logger.info("=" * 60)
    
    # 1. 交易日历（需要包含足够的历史和未来数据）
    logger.info("拉取交易日历...")
    # 扩展日期范围：前后各6个月
    import pandas as pd
    start_dt = pd.to_datetime(start_date, format='%Y%m%d') - pd.DateOffset(months=6)
    end_dt = pd.to_datetime(end_date, format='%Y%m%d') + pd.DateOffset(months=6)
    
    trade_cal = client.get_trade_cal(
        start_date=start_dt.strftime('%Y%m%d'),
        end_date=end_dt.strftime('%Y%m%d'),
        exchange="SSE"
    )
    storage.save_raw(trade_cal, "trade_cal")
    logger.info(f"交易日历拉取完成: {len(trade_cal)} 条记录")
    
    # 2. 股票基本信息
    logger.info("拉取股票基本信息...")
    stock_basic = client.get_stock_basic(list_status="L")
    storage.save_raw(stock_basic, "stock_basic")
    logger.info(f"股票基本信息拉取完成: {len(stock_basic)} 条记录")
    
    # 3. 日线行情（需要包含历史数据用于计算特征）
    logger.info(f"拉取日线行情 ({start_dt.strftime('%Y%m%d')} - {end_dt.strftime('%Y%m%d')})...")
    daily_data = client.get_daily(
        start_date=start_dt.strftime('%Y%m%d'),
        end_date=end_dt.strftime('%Y%m%d')
    )
    storage.save_raw(daily_data, "daily")
    logger.info(f"日线行情拉取完成: {len(daily_data)} 条记录")
    
    # 4. 复权因子
    logger.info(f"拉取复权因子 ({start_dt.strftime('%Y%m%d')} - {end_dt.strftime('%Y%m%d')})...")
    adj_factor = client.get_adj_factor(
        start_date=start_dt.strftime('%Y%m%d'),
        end_date=end_dt.strftime('%Y%m%d')
    )
    storage.save_raw(adj_factor, "adj_factor")
    logger.info(f"复权因子拉取完成: {len(adj_factor)} 条记录")
    
    logger.info("=" * 60)
    logger.info("数据拉取完成")
    logger.info("=" * 60)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("开始构建特征")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"最小上市天数: {args.min_list_days}")
    logger.info(f"预测时间窗口: {args.horizon} 个交易日")
    logger.info("=" * 60)
    
    try:
        # 初始化组件
        storage = Storage()
        loader = DataLoader(storage)
        builder = FeatureBuilder(
            min_list_days=args.min_list_days,
            horizon=args.horizon
        )
        
        # 如果需要，先拉取数据
        if args.pull_data:
            client = TushareClient()
            pull_required_data(client, storage, args.start_date, args.end_date)
        
        # 加载数据：优先从 clean 层加载，如不存在则从 raw 层加载或自动构建
        logger.info("加载基础数据...")
        
        use_clean_data = not args.use_raw
        daily_data = None
        trade_cal = None
        stock_basic = None
        
        if use_clean_data:
            logger.info("尝试从 clean 层加载数据...")
            
            # 加载 clean 层的交易日历和股票基本信息
            trade_cal_clean = loader.load_clean_trade_cal()
            stock_basic_clean = loader.load_clean_stock_basic()
            
            # 检查 clean 数据是否完整
            clean_daily_exists = storage.load_clean("daily") is not None or \
                                len(storage.list_partitions("clean", "daily")) > 0
            
            if trade_cal_clean is not None and stock_basic_clean is not None and clean_daily_exists:
                logger.info("✓ clean 层数据可用")
                trade_cal = trade_cal_clean
                stock_basic = stock_basic_clean
                
                # 转换日期格式以兼容 loader.get_trading_dates
                if 'cal_date' in trade_cal.columns:
                    if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                        trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
                
                # 加载 clean daily（将在特征构建循环中使用）
                # 这里先标记使用 clean 数据
                daily_data = "CLEAN"  # 占位符，实际加载在循环中
                
            else:
                logger.warning("clean 层数据不完整")
                
                if args.auto_build_clean:
                    logger.info("尝试自动构建 clean 数据...")
                    
                    try:
                        # 加载 raw 数据
                        trade_cal_raw = storage.load_raw("trade_cal")
                        stock_basic_raw = storage.load_raw("stock_basic")
                        
                        if trade_cal_raw is None or stock_basic_raw is None:
                            raise ValueError("缺少 raw 层基础数据")
                        
                        # 构建 clean 数据
                        cleaner = DataCleaner()
                        
                        logger.info("清洗交易日历...")
                        trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
                        storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
                        
                        logger.info("清洗股票基本信息...")
                        stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
                        storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
                        
                        # 简化：直接使用 build_clean 脚本的逻辑
                        import pandas as pd
                        from scripts.build_clean import build_clean_for_date_range
                        
                        start_dt = pd.to_datetime(args.start_date, format='%Y%m%d') - pd.DateOffset(months=1)
                        end_dt = pd.to_datetime(args.end_date, format='%Y%m%d') + pd.DateOffset(months=1)
                        
                        build_clean_for_date_range(
                            storage,
                            loader,
                            cleaner,
                            start_dt.strftime('%Y%m%d'),
                            end_dt.strftime('%Y%m%d'),
                            use_partitioning=True
                        )
                        
                        logger.info("✓ clean 数据自动构建完成")
                        
                        # 重新加载
                        trade_cal = trade_cal_clean
                        stock_basic = stock_basic_clean
                        
                        # 转换日期格式
                        if 'cal_date' in trade_cal.columns:
                            if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                                trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
                        
                        daily_data = "CLEAN"
                        
                    except Exception as e:
                        logger.warning(f"自动构建 clean 数据失败: {str(e)}")
                        logger.info("回退到使用 raw 数据")
                        use_clean_data = False
                else:
                    logger.info("auto_build_clean 未启用，回退到使用 raw 数据")
                    use_clean_data = False
        
        # 如果不使用 clean 或 clean 构建失败，使用 raw 数据
        if not use_clean_data or daily_data != "CLEAN":
            logger.info("从 raw 层加载数据...")
            trade_cal = loader.load_trade_cal()
            stock_basic = loader.load_stock_basic()
            daily_data = storage.load_raw("daily")
            adj_factor = storage.load_raw("adj_factor")
            
            # 检查数据是否存在
            if trade_cal is None:
                raise ValueError("交易日历数据不存在，请先运行 python scripts/pull_data.py")
            
            if stock_basic is None:
                raise ValueError("股票基本信息不存在，请先运行 python scripts/pull_data.py")
            
            if daily_data is None:
                raise ValueError("日线行情数据不存在，请先运行 python scripts/pull_data.py")
            
            if adj_factor is None:
                raise ValueError("复权因子数据不存在，请使用 --pull_data 参数拉取")
            
            logger.info(f"交易日历: {len(trade_cal)} 条")
            logger.info(f"股票基本信息: {len(stock_basic)} 条")
            logger.info(f"日线行情: {len(daily_data)} 条")
            logger.info(f"复权因子: {len(adj_factor)} 条")
        else:
            logger.info(f"交易日历: {len(trade_cal)} 条")
            logger.info(f"股票基本信息: {len(stock_basic)} 条")
            logger.info("日线行情: 将从 clean 层按需加载")
            adj_factor = None  # clean 数据已包含复权价格
        
        # 获取交易日列表
        trading_dates = loader.get_trading_dates(
            args.start_date[:4] + '-' + args.start_date[4:6] + '-' + args.start_date[6:8],
            args.end_date[:4] + '-' + args.end_date[4:6] + '-' + args.end_date[6:8]
        )
        
        if len(trading_dates) == 0:
            raise ValueError(f"指定日期范围内没有交易日: {args.start_date} - {args.end_date}")
        
        # 转换为YYYYMMDD格式
        import pandas as pd
        trading_dates_str = [
            d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
            for d in trading_dates
        ]
        
        logger.info(f"共 {len(trading_dates_str)} 个交易日需要构建特征")
        logger.info("=" * 60)
        
        # 遍历交易日构建特征
        success_count = 0
        skip_count = 0
        error_count = 0
        
        # 如果使用 clean 数据，需要加载全量数据或使用不同方式
        if use_clean_data and daily_data == "CLEAN":
            logger.info("使用 clean 层数据构建特征")
            
            # 尝试加载 clean 日线数据（含复权价格）
            # 扩展日期范围以包含历史数据
            start_dt = pd.to_datetime(args.start_date, format='%Y%m%d') - pd.DateOffset(months=1)
            end_dt = pd.to_datetime(args.end_date, format='%Y%m%d') + pd.DateOffset(months=1)
            
            daily_clean = loader.load_clean_daily(
                start_dt.strftime('%Y%m%d'),
                end_dt.strftime('%Y%m%d')
            )
            
            if daily_clean is None:
                raise ValueError("无法加载 clean 日线数据")
            
            logger.info(f"clean 日线数据: {len(daily_clean)} 条记录")
            
            # clean 数据已包含复权价格，使用空的 adj_factor（FeatureBuilder 会检测 close_adj 列）
            daily_data = daily_clean
            adj_factor = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])
        
        for i, trade_date in enumerate(trading_dates_str, 1):
            logger.info(f"[{i}/{len(trading_dates_str)}] 构建 {trade_date} 特征...")
            
            try:
                # 构建特征
                features_df = builder.build_features_for_day(
                    trade_date=trade_date,
                    trade_cal=trade_cal,
                    daily_data=daily_data,
                    adj_factor=adj_factor,
                    stock_basic=stock_basic,
                    suspend_info=None,  # 可选：停复牌信息
                    limit_info=None     # 可选：涨跌停信息
                )
                
                # 保存结果
                if len(features_df) > 0:
                    storage.save_cs_train_day(features_df, trade_date)
                    success_count += 1
                else:
                    logger.warning(f"{trade_date} 没有有效样本，跳过保存")
                    skip_count += 1
                    
            except Exception as e:
                logger.error(f"{trade_date} 特征构建失败: {str(e)}")
                error_count += 1
                # 继续处理下一个日期
                continue
        
        # 统计结果
        logger.info("=" * 60)
        logger.info("特征构建完成")
        logger.info("=" * 60)
        logger.info(f"成功: {success_count} 个交易日")
        logger.info(f"跳过: {skip_count} 个交易日（无有效样本）")
        logger.info(f"失败: {error_count} 个交易日")
        logger.info(f"保存位置: {storage.features_path / 'cs_train'}")
        logger.info("=" * 60)
        
    except ValueError as e:
        logger.error("=" * 60)
        logger.error("特征构建失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请确保已运行以下命令拉取数据:")
        logger.error("  python scripts/pull_data.py")
        logger.error("")
        logger.error("或使用 --pull_data 参数自动拉取:")
        logger.error("  python scripts/build_features.py --pull_data")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"特征构建过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


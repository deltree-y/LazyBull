# -*- coding: utf-8 -*-
"""clean 层批量构建：从 scripts/build_clean_features.py 下沉的 build_clean_data。

负责按交易日历逐日清洗 raw 层日线/复权/停牌/涨跌停/ST 数据并保存到 clean 分区。
"""

from loguru import logger

from . import DataCleaner, DataLoader, Storage

def build_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    start_date: str,
    end_date: str,
    force: bool = False,
    min_list_days: int = 365,
) -> None:
    """构建clean层数据

    Args:
        storage: Storage实例
        loader: DataLoader实例
        cleaner: DataCleaner实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新构建
        min_list_days: 最小上市天数
    """
    logger.info("=" * 60)
    logger.info("开始构建clean层数据")
    logger.info("=" * 60)

    # 1. 检查并处理trade_cal
    logger.info("处理交易日历...")
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is None:
        raise ValueError(
            "缺少raw层trade_cal数据，请先运行: python scripts/download_raw.py --only-basic"
        )

    trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
    storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)
    logger.info(f"交易日历清洗完成: {len(trade_cal_clean)} 条记录")

    # 2. 检查并处理stock_basic
    logger.info("处理股票基本信息...")
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is None:
        raise ValueError(
            "缺少raw层stock_basic数据，请先运行: python scripts/download_raw.py --only-basic"
        )

    stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
    storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)
    logger.info(f"股票基本信息清洗完成: {len(stock_basic_clean)} 条记录")

    # 3. 按日期分区处理日线数据
    logger.info("使用分区模式处理日线数据...")

    # 获取交易日列表
    trading_dates = trade_cal_clean[
        (trade_cal_clean["cal_date"] >= start_date)
        & (trade_cal_clean["cal_date"] <= end_date)
        & (trade_cal_clean["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"共 {len(trading_dates)} 个交易日需要处理")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")

        try:
            # 检查clean数据是否已存在
            if not force and storage.is_data_exists("clean", "daily", trade_date):
                logger.info(f"  clean daily已存在，跳过")
                skip_count += 1
                continue

            # 加载raw数据
            daily_raw = storage.load_raw_by_date("daily", trade_date)
            if daily_raw is None or len(daily_raw) == 0:
                logger.warning(f"  未找到raw层daily数据，跳过")
                error_count += 1
                continue

            adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
            if adj_factor_raw is None or len(adj_factor_raw) == 0:
                logger.warning(f"  未找到复权因子，使用默认值1.0")
                adj_factor_raw = daily_raw[["ts_code", "trade_date"]].copy()
                adj_factor_raw["adj_factor"] = 1.0

            # 清洗日线数据
            daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)

            # 添加可交易标记
            suspend_raw = storage.load_raw_by_date("suspend", trade_date)
            limit_raw = storage.load_raw_by_date("stk_limit", trade_date)
            stock_st_raw = storage.load_raw_by_date("stock_st", trade_date)

            suspend_clean = None
            limit_clean = None
            stock_st_clean = None

            if suspend_raw is not None and len(suspend_raw) > 0:
                suspend_clean = cleaner.clean_suspend_info(suspend_raw)

            if limit_raw is not None and len(limit_raw) > 0:
                limit_clean = cleaner.clean_limit_info(limit_raw)

            if stock_st_raw is not None and len(stock_st_raw) > 0:
                stock_st_clean = cleaner.clean_stock_st(stock_st_raw)

            daily_clean = cleaner.add_tradable_universe_flag(
                daily_clean,
                stock_basic_clean,
                stock_st_df=stock_st_clean,
                suspend_info_df=suspend_clean,
                limit_info_df=limit_clean,
                min_list_days=min_list_days,
            )

            # 保存clean数据
            storage.save_clean_by_date(daily_clean, "daily", trade_date)
            success_count += 1
            logger.info(f"  已保存 {len(daily_clean)} 条clean记录")

            # 处理daily_basic
            daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
            if daily_basic_raw is not None and len(daily_basic_raw) > 0:
                if force or not storage.is_data_exists("clean", "daily_basic", trade_date):
                    daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                    storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)

            # 处理moneyflow
            moneyflow_raw = storage.load_raw_by_date("moneyflow", trade_date)
            if moneyflow_raw is not None and len(moneyflow_raw) > 0:
                if force or not storage.is_data_exists("clean", "moneyflow", trade_date):
                    moneyflow_clean = cleaner.clean_moneyflow(moneyflow_raw)
                    storage.save_clean_by_date(moneyflow_clean, "moneyflow", trade_date)
            else:
                logger.warning(f"  未找到资金流向数据（moneyflow 为强制依赖项）")

        except Exception as e:
            logger.error(f"处理 {trade_date} 失败: {str(e)}")
            error_count += 1
            continue

    logger.info("=" * 60)
    logger.info("clean层数据构建完成")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 个交易日")
    logger.info(f"跳过: {skip_count} 个交易日（已存在）")
    logger.info(f"失败: {error_count} 个交易日")

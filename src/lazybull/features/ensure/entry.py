# -*- coding: utf-8 -*-
"""ensure 子包：ensure_features_for_date 主入口编排。"""

import gc
from typing import List, Tuple

import pandas as pd
from loguru import logger

from ...data import DataCleaner, DataLoader, Storage, TushareClient
from ...data.ensure import ensure_basic_data, ensure_clean_data_for_date
from ..builder import FeatureBuilder
from .constants import FEATURE_DATA_FUTURE_MONTHS, FEATURE_DATA_HISTORY_MONTHS
from .factor_load import _load_factor_data
from .historical import _ensure_historical_clean_data
from .industry import _ensure_shenwan_industry
from .schema import _check_features_schema


def ensure_features_for_date(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool = False,
) -> Tuple[bool, List[str], str]:
    """确保指定日期的 features 数据存在，不存在则构建

    若发现 clean 数据缺失，会自动调用 clean 模块的 ensure 函数
    若发现 raw 数据缺失，会进一步触发 raw 模块的下载

    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        builder: FeatureBuilder 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例（用于在依赖缺失时下载）
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新构建

    Returns:
        (success, missing_factors, error_detail) 元组：
        - success: 是否成功构建 features
        - missing_factors: 缺失的因子数据名称列表（空列表表示全部加载）
        - error_detail: 失败原因描述（成功时为空字符串）
    """
    # 纸面交易/推理场景使用独立的 cs_infer 子目录，避免与训练数据交叉污染
    _INFER_SUBDIR = "cs_infer"

    # 检查是否已存在（同时校验关键因子列，防止旧缓存缺失列）
    if not force and storage.is_feature_exists(trade_date, subdir=_INFER_SUBDIR):
        if _check_features_schema(storage, trade_date, subdir=_INFER_SUBDIR):
            logger.debug(f"features 数据已存在: {trade_date}")
            return True, [], ""
        else:
            logger.warning(f"features 缓存缺少必要因子列，将重新构建: {trade_date}")

    logger.info(f"构建 features 数据: {trade_date}")

    try:
        # 1. 确保基础数据存在
        if not ensure_basic_data(client, storage, trade_date, force=False):
            logger.error("无法获取基础数据（trade_cal/stock_basic）")
            return (
                False,
                [],
                "基础数据缺失：trade_cal/stock_basic 下载失败，请检查 TuShare 连接与积分",
            )

        # 2. 确保当日 clean 数据存在
        if not ensure_clean_data_for_date(storage, loader, cleaner, client, trade_date, force):
            logger.error(f"无法获取 clean 数据: {trade_date}")
            return (
                False,
                [],
                f"clean 数据缺失：{trade_date} 日的日线/每日指标/资金流向数据下载或清洗失败",
            )

        # 3. 确保历史 clean 数据存在（features 需要历史数据计算特征）
        if not _ensure_historical_clean_data(storage, loader, cleaner, client, trade_date, force):
            logger.warning(f"历史 clean 数据不完整，特征可能受影响: {trade_date}")
            # 不返回 False，继续尝试构建特征

        # 4. 加载基础数据
        trade_cal = loader.load_clean_trade_cal()
        stock_basic = loader.load_clean_stock_basic()

        if trade_cal is None or stock_basic is None:
            logger.error("缺少 clean 基础数据")
            return False, [], "clean 基础数据缺失：trade_cal 或 stock_basic 加载失败"

        # 转换日期格式
        if "cal_date" in trade_cal.columns:
            if not pd.api.types.is_datetime64_any_dtype(trade_cal["cal_date"]):
                trade_cal["cal_date"] = pd.to_datetime(trade_cal["cal_date"], format="%Y%m%d")

        # 5. 加载 clean 日线数据（与 build_clean_features 口径对齐）
        trade_dt = pd.to_datetime(trade_date, format="%Y%m%d")
        start_dt = trade_dt - pd.DateOffset(months=FEATURE_DATA_HISTORY_MONTHS)
        end_dt = trade_dt + pd.DateOffset(months=FEATURE_DATA_FUTURE_MONTHS)

        daily_clean = loader.load_clean_daily(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )

        daily_basic_clean = loader.load_clean_daily_basic(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )

        moneyflow_clean = loader.load_clean_moneyflow(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        )

        if daily_clean is None or daily_clean.empty:
            logger.error(f"缺少 clean 日线数据: {trade_date}")
            return False, [], f"clean 日线数据缺失：{trade_date} 无日线行情数据"

        # 与 build_clean_features 对齐：moneyflow 缺失只告警，不在 ensure 阶段中断
        if moneyflow_clean is None or moneyflow_clean.empty:
            logger.warning("未找到 moneyflow 数据（强制依赖项），资金流特征将为空")

        logger.info(f"clean 日线数据: {len(daily_clean)} 条记录")
        logger.info(
            f"clean moneyflow 数据: {len(moneyflow_clean) if moneyflow_clean is not None else 0} 条记录"
        )

        # 6. 加载申万行业分类数据（缺失则自动下载）
        shenwan_industry = loader.load_shenwan_industry()
        if shenwan_industry is None:
            logger.info("自动下载申万行业分类数据...")
            shenwan_industry = _ensure_shenwan_industry(client, storage, cleaner)
        if shenwan_industry is None:
            logger.error(
                "未找到申万行业分类数据，无法构建行业中性化特征！\n"
                "模型依赖 zscore_*/neu_*/alpha_industry_*/ind_* 特征，缺失会导致推理失败。\n"
                "请运行: python scripts/update_basic_data.py --only-shenwan --force"
            )
            return False, [], "申万行业分类数据缺失，无法构建行业中性化特征"
        else:
            apply_neutralization = True
            logger.info(f"已加载申万行业分类数据: {len(shenwan_industry)} 条映射")

        # 7. 加载因子数据（基本面 + 另类数据）
        # 获取日期范围内的交易日列表（因子 lookup 构建需要）
        trading_dates_mask = (
            (trade_cal["cal_date"] >= start_dt)
            & (trade_cal["cal_date"] <= end_dt)
            & (trade_cal["is_open"] == 1)
        )
        trading_dates_str = [
            d.strftime("%Y%m%d") if isinstance(d, pd.Timestamp) else d
            for d in trade_cal[trading_dates_mask]["cal_date"].tolist()
        ]

        daily_close_lookup = {
            d: grp[["ts_code", "close", "close_adj"]].copy()
            for d, grp in daily_clean.groupby("trade_date", sort=False)
            if "close" in grp.columns and "close_adj" in grp.columns
        }
        # 未复权收盘价查询（大宗交易折价率需要与原始成交价同口径）
        block_close_lookup = {
            d: grp[["ts_code", "close"]].dropna().set_index("ts_code")["close"].to_dict()
            for d, grp in daily_clean.groupby("trade_date", sort=False)
            if "close" in grp.columns
        }

        (
            funda_today,
            margin_today,
            holder_today,
            earnings_today,
            cyq_perf_today,
            express_today,
            fund_portfolio_today,
            north_flow_today,
            lhb_today,
            consensus_today,
            cashflow_today,
            consensus_revision_today,
            pledge_today,
            share_float_today,
            block_trade_today,
            missing_factors,
        ) = _load_factor_data(
            loader,
            client,
            storage,
            trade_date,
            trading_dates_str,
            start_dt.strftime("%Y%m%d"),
            end_dt.strftime("%Y%m%d"),
            daily_close_lookup=daily_close_lookup,
            block_close_lookup=block_close_lookup,
        )

        # 与 build_clean_features 对齐：循环外预计算 daily_adj 与日期索引缓存
        adj_factor = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        builder.precompute_daily_adj(daily_clean, adj_factor)

        # 8. 构建特征（adj_factor 传空列结构，复权价来自 clean 日线）
        features_df = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_clean,
            adj_factor=adj_factor,
            stock_basic=stock_basic,
            daily_basic_data=daily_basic_clean,
            moneyflow_data=moneyflow_clean,
            suspend_info=None,
            limit_info=None,
            shenwan_industry=shenwan_industry,
            apply_industry_neutralization=apply_neutralization,
            apply_size_neutralization=True,
            fundamental_data=funda_today,
            margin_data=margin_today,
            holder_data=holder_today,
            earnings_data=earnings_today,
            cyq_perf_data=cyq_perf_today,
            express_data=express_today,
            fund_portfolio_data=fund_portfolio_today,
            north_flow_data=north_flow_today,
            lhb_data=lhb_today,
            consensus_data=consensus_today,
            cashflow_data=cashflow_today,
            consensus_revision_data=consensus_revision_today,
            pledge_data=pledge_today,
            share_float_data=share_float_today,
            block_trade_data=block_trade_today,
        )
        # 释放不再需要的历史数据，降低保存时的内存占用
        daily_clean = None
        daily_close_lookup = None
        block_close_lookup = None
        daily_basic_clean = None
        moneyflow_clean = None
        funda_today = margin_today = holder_today = earnings_today = None
        cyq_perf_today = express_today = fund_portfolio_today = None
        north_flow_today = lhb_today = consensus_today = None
        cashflow_today = consensus_revision_today = None
        pledge_today = share_float_today = block_trade_today = None
        gc.collect()

        # 9. 保存结果
        if len(features_df) > 0:
            storage.save_cs_train_day(features_df, trade_date, subdir=_INFER_SUBDIR)
            logger.info(f"已保存 features 数据: {len(features_df)} 条")
            return True, missing_factors, ""
        else:
            logger.warning(f"没有有效样本: {trade_date}")
            return (
                False,
                missing_factors,
                f"特征构建结果为空：{trade_date} 无有效样本（可能因停牌/涨跌停导致全部股票被过滤）",
            )

    except Exception as e:
        logger.error(f"构建 features 数据失败 {trade_date}: {e}")
        return False, [], f"特征构建异常：{e}"

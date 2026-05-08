"""特征确保模块

提供确保 features 数据存在的封装函数
"""

import gc
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from ..data import DataCleaner, DataLoader, Storage, TushareClient
from ..data.ensure import ensure_basic_data, ensure_clean_data_for_date
from .builder import FeatureBuilder

# 常量定义
# 与 build_clean_features.py 保持一致：
# - 过去约 7 个月用于覆盖 120 交易日 warmup
# - 向后扩展 1 个月保持离线构建口径（require_label=False 时不会因标签使用未来数据）
FEATURE_DATA_HISTORY_MONTHS = 7
FEATURE_DATA_FUTURE_MONTHS = 1
HISTORICAL_DATA_MONTHS = FEATURE_DATA_HISTORY_MONTHS
# 最多检查最近 N 个交易日 clean 分区，确保 warmup 期间缺口可被自动补齐
MAX_HISTORICAL_DAYS = 180

# 因子数据最低记录数阈值，低于此值视为数据不足，触发全量下载
# 这些因子是 point-in-time 查询，需要全量历史才有意义
_MIN_FINA_RECORDS = 1000       # 财务指标：全量应有 10 万+ 条
_MIN_HOLDER_RECORDS = 500      # 股东人数：全量应有数万条
_MIN_FORECAST_RECORDS = 500    # 业绩预告：全量应有数万条
_MIN_EXPRESS_RECORDS = 500        # 业绩快报：全量应有数万条
_MIN_REPORT_RC_RECORDS = 1000     # 一致预期研报：全量应有数万条


def ensure_features_for_date(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool = False
) -> Tuple[bool, List[str]]:
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
        (success, missing_factors) 元组：
        - success: 是否成功构建 features
        - missing_factors: 缺失的因子数据名称列表（空列表表示全部加载）
    """
    # 纸面交易/推理场景使用独立的 cs_infer 子目录，避免与训练数据交叉污染
    _INFER_SUBDIR = "cs_infer"

    # 检查是否已存在（同时校验关键因子列，防止旧缓存缺失列）
    if not force and storage.is_feature_exists(trade_date, subdir=_INFER_SUBDIR):
        if _check_features_schema(storage, trade_date, subdir=_INFER_SUBDIR):
            logger.debug(f"features 数据已存在: {trade_date}")
            return True, []
        else:
            logger.warning(
                f"features 缓存缺少必要因子列，将重新构建: {trade_date}"
            )
    
    logger.info(f"构建 features 数据: {trade_date}")
    
    try:
        # 1. 确保基础数据存在
        if not ensure_basic_data(client, storage, trade_date, force=False):
            logger.error("无法获取基础数据（trade_cal/stock_basic）")
            return False, []
        
        # 2. 确保当日 clean 数据存在
        if not ensure_clean_data_for_date(
            storage, loader, cleaner, client, trade_date, force
        ):
            logger.error(f"无法获取 clean 数据: {trade_date}")
            return False, []
        
        # 3. 确保历史 clean 数据存在（features 需要历史数据计算特征）
        if not _ensure_historical_clean_data(
            storage, loader, cleaner, client, trade_date, force
        ):
            logger.warning(f"历史 clean 数据不完整，特征可能受影响: {trade_date}")
            # 不返回 False，继续尝试构建特征
        
        # 4. 加载基础数据
        trade_cal = loader.load_clean_trade_cal()
        stock_basic = loader.load_clean_stock_basic()
        
        if trade_cal is None or stock_basic is None:
            logger.error("缺少 clean 基础数据")
            return False, []
        
        # 转换日期格式
        if 'cal_date' in trade_cal.columns:
            if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
                trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
        
        # 5. 加载 clean 日线数据（与 build_clean_features 口径对齐）
        trade_dt = pd.to_datetime(trade_date, format='%Y%m%d')
        start_dt = trade_dt - pd.DateOffset(
            months=FEATURE_DATA_HISTORY_MONTHS
        )
        end_dt = trade_dt + pd.DateOffset(
            months=FEATURE_DATA_FUTURE_MONTHS
        )
        
        daily_clean = loader.load_clean_daily(
            start_dt.strftime('%Y%m%d'),
            end_dt.strftime('%Y%m%d')
        )

        daily_basic_clean = loader.load_clean_daily_basic(
            start_dt.strftime('%Y%m%d'),
            end_dt.strftime('%Y%m%d')
        )

        moneyflow_clean = loader.load_clean_moneyflow(
            start_dt.strftime('%Y%m%d'),
            end_dt.strftime('%Y%m%d')
        )

        
        if daily_clean is None or daily_clean.empty:
            logger.error(f"缺少 clean 日线数据: {trade_date}")
            return False, []
        
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
            return False, []
        else:
            apply_neutralization = True
            logger.info(f"已加载申万行业分类数据: {len(shenwan_industry)} 条映射")

        # 7. 加载因子数据（基本面 + 另类数据）
        # 获取日期范围内的交易日列表（因子 lookup 构建需要）
        trading_dates_mask = (
            (trade_cal['cal_date'] >= start_dt) &
            (trade_cal['cal_date'] <= end_dt) &
            (trade_cal['is_open'] == 1)
        )
        trading_dates_str = [
            d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
            for d in trade_cal[trading_dates_mask]['cal_date'].tolist()
        ]

        (funda_today, margin_today, holder_today, earnings_today,
         cyq_perf_today, express_today, fund_portfolio_today,
         north_flow_today, lhb_today, consensus_today, missing_factors) = (
            _load_factor_data(loader, client, storage, trade_date, trading_dates_str,
                              start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
        )

        # 与 build_clean_features 对齐：循环外预计算 daily_adj 与日期索引缓存
        adj_factor = pd.DataFrame(columns=['ts_code', 'trade_date', 'adj_factor'])
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
        )
        # 释放不再需要的历史数据，降低保存时的内存占用
        daily_clean = None
        daily_basic_clean = None
        moneyflow_clean = None
        funda_today = margin_today = holder_today = earnings_today = None
        cyq_perf_today = express_today = fund_portfolio_today = None
        north_flow_today = lhb_today = consensus_today = None
        gc.collect()

        # 9. 保存结果
        if len(features_df) > 0:
            storage.save_cs_train_day(features_df, trade_date, subdir=_INFER_SUBDIR)
            logger.info(f"已保存 features 数据: {len(features_df)} 条")
            return True, missing_factors
        else:
            logger.warning(f"没有有效样本: {trade_date}")
            return False, missing_factors

    except Exception as e:
        logger.error(f"构建 features 数据失败 {trade_date}: {e}")
        return False, []


def _ensure_historical_clean_data(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool
) -> bool:
    """确保历史 clean 数据存在
    
    Features 构建需要历史数据来计算动量、均值等特征。
    这里按与 build_clean_features 一致的 warmup 窗口补齐历史 clean 数据。
    
    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例
        trade_date: 当前交易日期，格式 YYYYMMDD
        force: 是否强制重新构建
        
    Returns:
        是否成功（至少部分历史数据可用）
    """
    # 获取交易日历
    trade_cal = loader.load_clean_trade_cal()
    if trade_cal is None:
        logger.warning("无法加载交易日历，跳过历史数据检查")
        return False
    
    # 确保日期格式统一
    if 'cal_date' in trade_cal.columns:
        if not pd.api.types.is_datetime64_any_dtype(trade_cal['cal_date']):
            trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'], format='%Y%m%d')
    
    # 获取 warmup 窗口内的历史交易日
    start_dt = pd.to_datetime(trade_date, format='%Y%m%d') - pd.DateOffset(
        months=HISTORICAL_DATA_MONTHS
    )
    
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_dt) &
        (trade_cal['cal_date'] < pd.to_datetime(trade_date, format='%Y%m%d')) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    if not trading_dates:
        logger.warning("未找到历史交易日")
        return False
    
    # 转换为 YYYYMMDD 格式
    trading_dates_str = [
        d.strftime('%Y%m%d') if isinstance(d, pd.Timestamp) else d
        for d in trading_dates
    ]
    
    logger.info(f"检查 {len(trading_dates_str)} 个历史交易日的 clean 数据")
    
    # 检查并补齐缺失的历史数据（最多补齐最近的指定个交易日）
    missing_count = 0
    success_count = 0
    
    for hist_date in trading_dates_str[-MAX_HISTORICAL_DAYS:]:  # 最多检查最近指定个交易日
        # 检查 daily/daily_basic/moneyflow 任一缺失即需补齐
        daily_ok = storage.is_data_exists("clean", "daily", hist_date)
        daily_basic_ok = storage.is_data_exists("clean", "daily_basic", hist_date)
        moneyflow_ok = storage.is_data_exists("clean", "moneyflow", hist_date)
        if not (daily_ok and daily_basic_ok and moneyflow_ok):
            missing_count += 1
            # 尝试补齐（ensure_clean_data_for_date 内部会跳过已存在的数据集）
            if ensure_clean_data_for_date(
                storage, loader, cleaner, client, hist_date, force
            ):
                success_count += 1
    
    if missing_count > 0:
        logger.info(f"补齐了 {success_count}/{missing_count} 个历史交易日的 clean 数据")
    
    # 只要有部分数据可用就返回 True
    return True


def _load_factor_data(
    loader: DataLoader,
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    trading_dates_str: List[str],
    start_date: str,
    end_date: str,
) -> tuple:
    """加载因子数据（基本面 + 另类数据）

    尝试加载已有的 raw 因子数据并构建 lookup 表。
    若数据缺失，会自动通过 TuShare/AKShare 按日下载并追加到单文件。

    内存优化：每个因子段落处理完后立即释放原始 DataFrame 和 lookup 字典，
    配合 gc.collect() 确保在内存受限环境（如树莓派）下不会 OOM。

    Args:
        loader: DataLoader 实例
        client: TushareClient 实例（用于按日增量下载）
        storage: Storage 实例（用于保存增量数据）
        trade_date: 目标交易日，格式 YYYYMMDD
        trading_dates_str: 日期范围内的交易日列表
        start_date: 数据范围起始日期
        end_date: 数据范围结束日期

    Returns:
        (funda_today, margin_today, holder_today, earnings_today, missing_factors)
        前 4 个元素为当日的 DataFrame 或 None，最后一个为缺失因子名称列表
    """
    missing_factors = []
    # 纸面交易 T0 仅消费当日因子截面；历史窗口仍通过原始数据加载提供，
    # 无需为整段 trading_dates 物化完整的 date -> DataFrame 查询表。
    factor_output_dates = [trade_date]

    # ── 基本面因子 ──────────────────────────────────────────
    funda_today = None
    fina_indicator = loader.load_fina_indicator()
    # 数据不存在、或记录过少（之前单日增量下载的残留）均触发全量下载
    if fina_indicator is None or len(fina_indicator) < _MIN_FINA_RECORDS:
        fina_indicator = _try_download_fina_indicator(client, storage, trade_date)
    if fina_indicator is not None and len(fina_indicator) > 0:
        from ..factors.fundamental import build_fundamental_lookup_by_date
        funda_lookup = build_fundamental_lookup_by_date(fina_indicator, factor_output_dates)
        funda_today = funda_lookup.get(trade_date)
        logger.info(f"基本面因子: 已加载 ({len(fina_indicator)} 条原始记录)")
    else:
        missing_factors.append("fina_indicator（基本面）")
    # 释放基本面因子中间数据
    fina_indicator = None
    funda_lookup = None
    gc.collect()

    # ── 融资融券 ────────────────────────────────────────────
    # margin 按日分区存储，且需要 20+ 天历史数据计算滚动变动率
    margin_today = None
    # 只补齐 <= trade_date 的历史分区（未来日期实盘不可获取）
    hist_dates = [d for d in trading_dates_str if d <= trade_date]
    _try_ensure_historical_margin(client, storage, hist_dates)
    margin_detail = loader.load_margin_detail(start_date, end_date)
    if margin_detail is not None and len(margin_detail) > 0:
        from ..factors.margin import build_margin_lookup_by_date
        margin_lookup = build_margin_lookup_by_date(margin_detail, factor_output_dates)
        margin_today = margin_lookup.get(trade_date)
        # 当日 margin_detail 可能尚未发布，额外重试一次下载
        if margin_today is None:
            logger.warning(
                f"融资融券: 当日 {trade_date} 数据不在查询表中，尝试单独下载..."
            )
            try:
                df = client.query("margin_detail", trade_date=trade_date)
                if df is not None and not df.empty:
                    storage.save_raw_by_date(df, "margin_detail", trade_date)
                    # 重新加载并构建（追加当日数据）
                    margin_detail_full = loader.load_margin_detail(start_date, end_date)
                    if margin_detail_full is not None and len(margin_detail_full) > 0:
                        margin_lookup = build_margin_lookup_by_date(
                            margin_detail_full, factor_output_dates
                        )
                        margin_today = margin_lookup.get(trade_date)
                    margin_detail_full = None
            except Exception as e:
                logger.warning(f"融资融券: 当日数据下载重试失败: {e}")
        if margin_today is not None:
            logger.info(f"融资融券因子: 已加载 ({len(margin_detail)} 条)")
        else:
            raise RuntimeError(
                f"融资融券因子: 无法获取当日 {trade_date} 的 margin_detail 数据。\n"
                f"TuShare margin_detail 数据可能尚未发布，请稍后重试。"
            )
    else:
        missing_factors.append("margin_detail（融资融券）")
    # 释放融资融券中间数据
    margin_detail = None
    margin_lookup = None
    gc.collect()

    # ── 股东人数 ────────────────────────────────────────────
    holder_today = None
    stk_holdernumber = loader.load_stk_holdernumber()
    if stk_holdernumber is None or len(stk_holdernumber) < _MIN_HOLDER_RECORDS:
        stk_holdernumber = _try_download_stk_holdernumber(client, storage, trade_date)
    if stk_holdernumber is not None and len(stk_holdernumber) > 0:
        from ..factors.holder import build_holder_lookup_by_date
        holder_lookup = build_holder_lookup_by_date(stk_holdernumber, factor_output_dates)
        holder_today = holder_lookup.get(trade_date)
        logger.info(f"股东人数因子: 已加载 ({len(stk_holdernumber)} 条)")
    else:
        missing_factors.append("stk_holdernumber（股东人数）")
    # 释放股东人数中间数据
    stk_holdernumber = None
    holder_lookup = None
    gc.collect()

    # ── 业绩预告 ────────────────────────────────────────────
    earnings_today = None
    forecast_df = loader.load_forecast()
    if forecast_df is None or len(forecast_df) < _MIN_FORECAST_RECORDS:
        forecast_df = _try_download_forecast(client, storage, trade_date)
    if forecast_df is not None and len(forecast_df) > 0:
        from ..factors.earnings import build_earnings_lookup_by_date
        earnings_lookup = build_earnings_lookup_by_date(
            forecast_df, factor_output_dates
        )
        earnings_today = earnings_lookup.get(trade_date)
        logger.info(f"业绩预告因子: 已加载 ({len(forecast_df)} 条)")
    else:
        missing_factors.append("forecast（业绩预告）")
    # 释放 earnings_lookup，保留 forecast_df 供业绩快报段复用
    earnings_lookup = None
    gc.collect()

    # ── 筹码胜率（按日分区，同 margin_detail）──────────────────
    cyq_perf_today = None
    # 只补齐 <= trade_date 的历史分区（未来日期实盘不可获取）
    cyq_perf_hist_dates = [d for d in trading_dates_str if d <= trade_date]
    cyq_perf_df = _try_ensure_historical_cyq_perf(client, storage, cyq_perf_hist_dates)
    if cyq_perf_df is not None and len(cyq_perf_df) > 0:
        from ..factors.cyq_perf import build_cyq_perf_lookup_by_date
        cyq_perf_lookup = build_cyq_perf_lookup_by_date(cyq_perf_df, factor_output_dates)
        cyq_perf_today = cyq_perf_lookup.get(trade_date)
        logger.info(f"筹码胜率因子: 已加载 ({len(cyq_perf_df)} 条)")
    else:
        missing_factors.append("cyq_perf（筹码胜率）")
    # 释放筹码胜率中间数据
    cyq_perf_df = None
    cyq_perf_lookup = None
    gc.collect()

    # ── 业绩快报 ──────────────────────────────────────────────
    express_today = None
    express_df = loader.load_express()
    if express_df is None or len(express_df) < _MIN_EXPRESS_RECORDS:
        express_df = _try_download_express(client, storage, trade_date)
    if express_df is not None and len(express_df) > 0:
        from ..factors.express import build_express_lookup_by_date
        # 复用业绩预告段已加载的 forecast_df，避免重复磁盘读取
        express_lookup = build_express_lookup_by_date(
            express_df, factor_output_dates, forecast_df=forecast_df
        )
        express_today = express_lookup.get(trade_date)
        logger.info(f"业绩快报因子: 已加载 ({len(express_df)} 条)")
    else:
        missing_factors.append("express（业绩快报）")
    # 释放业绩快报 + forecast_df（业绩快报已用完，不再需要）
    express_df = None
    express_lookup = None
    forecast_df = None
    gc.collect()

    # ── 基金持仓（按季度分区）──────────────────────────────────
    fund_portfolio_today = None
    # 只补齐 <= trade_date 的历史季度（未来季度实盘不可获取）
    fund_hist_dates = [d for d in trading_dates_str if d <= trade_date]
    fund_portfolio_df = _try_ensure_historical_fund_portfolio(
        client, storage, fund_hist_dates,
    )
    if fund_portfolio_df is not None and len(fund_portfolio_df) > 0:
        from ..factors.fund_portfolio import build_fund_portfolio_lookup_by_date
        fund_lookup = build_fund_portfolio_lookup_by_date(
            fund_portfolio_df, factor_output_dates, pre_aggregated=True
        )
        fund_portfolio_today = fund_lookup.get(trade_date)
        logger.info(f"基金持仓因子: 已加载 ({len(fund_portfolio_df)} 条)")
    else:
        missing_factors.append("fund_portfolio（基金持仓）")
    # 释放基金持仓中间数据
    fund_portfolio_df = None
    fund_lookup = None
    gc.collect()

    # ── 北向资金（按日分区，市场级广播）──────────────────────────
    # 启用语义: 默认传空 dict (启用占位), 仅当全市场无任何历史数据时才回退 None
    north_flow_today = {}
    north_hist_dates = [d for d in trading_dates_str if d <= trade_date]
    hsgt_df = _try_ensure_historical_moneyflow_hsgt(client, storage, north_hist_dates)
    if hsgt_df is not None and len(hsgt_df) > 0:
        from ..factors.north_flow import build_north_flow_lookup_by_date
        north_lookup = build_north_flow_lookup_by_date(hsgt_df, factor_output_dates)
        cur = north_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            north_flow_today = cur
            logger.info(f"北向资金因子: 已加载 ({len(hsgt_df)} 条)")
        else:
            missing_factors.append("moneyflow_hsgt（北向资金, 当日无数据, 占位 NaN）")
    else:
        missing_factors.append("moneyflow_hsgt（北向资金）")
    hsgt_df = None
    north_lookup = None
    gc.collect()

    # ── 龙虎榜（按日分区）──────────────────────────────────────
    lhb_today = pd.DataFrame()
    lhb_hist_dates = [d for d in trading_dates_str if d <= trade_date]
    top_list_df = _try_ensure_historical_top_list(client, storage, lhb_hist_dates)
    if top_list_df is not None and len(top_list_df) > 0:
        from ..factors.lhb import build_lhb_lookup_by_date
        lhb_lookup = build_lhb_lookup_by_date(top_list_df, factor_output_dates)
        cur = lhb_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            lhb_today = cur
        logger.info(f"龙虎榜因子: 已加载 ({len(top_list_df)} 条)")
    else:
        # 龙虎榜稀疏, 无数据不视为错误, 只记缺失标签
        missing_factors.append("top_list（龙虎榜）")
    top_list_df = None
    lhb_lookup = None
    gc.collect()

    # ── 一致预期（单文件, 按 report_date 增量）───────────────────
    consensus_today = pd.DataFrame()
    report_rc_df = loader.load_report_rc()
    if report_rc_df is None or len(report_rc_df) < _MIN_REPORT_RC_RECORDS:
        report_rc_df = _try_download_report_rc(client, storage, trade_date)
    if report_rc_df is not None and len(report_rc_df) > 0:
        from ..factors.consensus import build_consensus_lookup_by_date
        cons_lookup = build_consensus_lookup_by_date(report_rc_df, factor_output_dates)
        cur = cons_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            consensus_today = cur
        logger.info(f"一致预期因子: 已加载 ({len(report_rc_df)} 条)")
    else:
        missing_factors.append("report_rc（一致预期）")
    report_rc_df = None
    cons_lookup = None
    gc.collect()

    # ── 汇总报告 ────────────────────────────────────────────
    total = 10
    loaded = total - len(missing_factors)
    if missing_factors:
        logger.warning(
            f"因子数据覆盖: {loaded}/{total} 组已加载，"
            f"缺失: {', '.join(missing_factors)}\n"
            f"  如需补全请运行: python scripts/download_raw.py --download <数据类型>"
        )
    else:
        logger.info(f"因子数据覆盖: {total}/{total} 组全部加载")

    return (funda_today, margin_today, holder_today, earnings_today,
            cyq_perf_today, express_today, fund_portfolio_today,
            north_flow_today, lhb_today, consensus_today, missing_factors)


# ── 因子按日增量下载辅助函数 ──────────────────────────────────────


def _append_and_save_raw(
    storage: Storage,
    dataset_name: str,
    new_df: pd.DataFrame,
    dedup_cols: List[str],
) -> pd.DataFrame:
    """将增量数据追加到已有单文件并去重保存

    Args:
        storage: Storage 实例
        dataset_name: 数据集名称（如 fina_indicator）
        new_df: 新下载的增量 DataFrame
        dedup_cols: 去重列

    Returns:
        合并后的完整 DataFrame
    """
    existing_df = storage.load_raw(dataset_name)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        result = new_df.copy()
    result = result.drop_duplicates(subset=dedup_cols, keep="last")
    storage.save_raw(result, dataset_name, is_force=True)
    return result


def _save_merged_bulk(
    storage: Storage,
    dataset_name: str,
    new_dfs: List[pd.DataFrame],
    existing_df: Optional[pd.DataFrame],
    dedup_cols: Optional[List[str]],
) -> pd.DataFrame:
    """合并新旧数据并保存（批量下载中间/最终保存用）"""
    result = pd.concat(new_dfs, ignore_index=True)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, result], ignore_index=True)
    if dedup_cols:
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")
    return result


def _normalize_date_str(date_value: object) -> Optional[str]:
    """将日期值标准化为 YYYYMMDD 字符串。"""
    if pd.isna(date_value):
        return None
    date_str = str(date_value).strip()
    if not date_str:
        return None
    date_str = date_str.replace("-", "")[:8]
    if len(date_str) != 8 or not date_str.isdigit():
        return None
    return date_str


def _get_latest_date(df: Optional[pd.DataFrame], date_col: str) -> Optional[str]:
    """从 DataFrame 中提取指定日期列的最大日期（YYYYMMDD）。"""
    if df is None or len(df) == 0 or date_col not in df.columns:
        return None
    dates = df[date_col].map(_normalize_date_str).dropna()
    if dates.empty:
        return None
    return str(dates.max())


def _iter_calendar_dates(start_date: str, end_date: str) -> List[str]:
    """生成闭区间 [start_date, end_date] 的自然日列表（YYYYMMDD）。"""
    try:
        start_ts = pd.to_datetime(start_date, format="%Y%m%d")
        end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    except Exception:
        return []
    if start_ts > end_ts:
        return []
    return [d.strftime("%Y%m%d") for d in pd.date_range(start_ts, end_ts, freq="D")]


def _incremental_catchup_by_calendar_date(
    storage: Storage,
    dataset_name: str,
    existing_df: Optional[pd.DataFrame],
    trade_date: str,
    date_col: str,
    dedup_cols: List[str],
    fetch_by_date: Callable[[str], Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """按自然日补齐公告/事件类增量数据，避免只查单日导致漏数。

    适用于 ann_date/report_date 这类“可能在非交易日发布”的数据。
    """
    if existing_df is None or len(existing_df) == 0:
        return existing_df

    target_date = _normalize_date_str(trade_date)
    if target_date is None:
        logger.warning(f"[{dataset_name}] 无法解析 trade_date={trade_date}，跳过增量补齐")
        return existing_df

    latest_date = _get_latest_date(existing_df, date_col)
    if latest_date is None:
        logger.warning(
            f"[{dataset_name}] 本地数据缺少有效 {date_col}，无法执行区间补齐，保持现有数据"
        )
        return existing_df

    if latest_date >= target_date:
        logger.info(
            f"[{dataset_name}] 本地最新 {date_col}={latest_date}，已覆盖目标日期 {target_date}"
        )
        return existing_df

    start_date = (
        pd.to_datetime(latest_date, format="%Y%m%d") + pd.Timedelta(days=1)
    ).strftime("%Y%m%d")
    pending_dates = _iter_calendar_dates(start_date, target_date)
    if not pending_dates:
        return existing_df

    logger.info(
        f"[{dataset_name}] 区间增量补齐: {date_col} {start_date}~{target_date} "
        f"(共 {len(pending_dates)} 天)"
    )

    new_dfs: List[pd.DataFrame] = []
    success_days = 0
    empty_days = 0
    failed_days = 0

    for idx, cur_date in enumerate(pending_dates, 1):
        try:
            day_df = fetch_by_date(cur_date)
            if day_df is not None and len(day_df) > 0:
                new_dfs.append(day_df)
                success_days += 1
            else:
                empty_days += 1
        except Exception as e:
            failed_days += 1
            logger.warning(f"[{dataset_name}] {date_col}={cur_date} 增量下载失败: {e}")

        if idx % 30 == 0 or idx == len(pending_dates):
            logger.info(
                f"[{dataset_name}] 增量进度 {idx}/{len(pending_dates)} "
                f"(有数据={success_days}, 空={empty_days}, 失败={failed_days})"
            )

    if not new_dfs:
        logger.info(
            f"[{dataset_name}] 区间增量完成: 无新增记录 "
            f"(空={empty_days}, 失败={failed_days})"
        )
        return existing_df

    new_merged = pd.concat(new_dfs, ignore_index=True)
    result = _append_and_save_raw(
        storage,
        dataset_name,
        new_merged,
        dedup_cols=dedup_cols,
    )
    logger.info(
        f"[{dataset_name}] 区间增量完成: 新增 {len(new_merged)} 条, "
        f"总计 {len(result)} 条"
    )
    return result


def _generate_quarter_periods(start_year: int, end_year: int) -> List[str]:
    """生成从 start_year 到 end_year 的所有季度末日期"""
    quarter_ends = ["0331", "0630", "0930", "1231"]
    return [f"{y}{q}" for y in range(start_year, end_year + 1) for q in quarter_ends]


def _query_with_pagination(
    client: TushareClient,
    api_name: str,
    page_limit: int = 50000,
    fields: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """带分页的 API 调用，自动检测并翻页获取全量数据"""
    all_pages: List[pd.DataFrame] = []
    offset = 0
    while True:
        df = client.pro.query(
            api_name, fields=fields or "",
            limit=page_limit, offset=offset, **kwargs,
        )
        if df is None or len(df) == 0:
            break
        all_pages.append(df)
        if len(df) < page_limit:
            break
        offset += page_limit
    if not all_pages:
        return pd.DataFrame()
    return pd.concat(all_pages, ignore_index=True)


def _bulk_download_by_period(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    dedup_cols: List[str],
    fields: Optional[str] = None,
    start_year: int = 2012,
) -> Optional[pd.DataFrame]:
    """按报告期(period)批量下载全量数据（自动分页）

    适用于 fina_indicator_vip, forecast_vip, express_vip, fund_portfolio。
    每季度1次 API 调用，替代逐股下载。
    当单季度数据超过上限时自动通过 offset 分页获取全量。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dataset_name: 数据集名称
        api_name: TuShare API 名称
        dedup_cols: 去重列
        fields: 返回字段（部分 API 需要）
        start_year: 起始年份

    Returns:
        下载并保存后的完整 DataFrame，或 None
    """
    import datetime as _dt
    current_year = _dt.datetime.now().year
    periods = _generate_quarter_periods(start_year, current_year)

    # 断点续传：跳过已有季度
    existing_df = storage.load_raw(dataset_name)
    existing_periods: Set[str] = set()
    if existing_df is not None and len(existing_df) > 0:
        if "end_date" in existing_df.columns:
            existing_periods = set(
                existing_df["end_date"].astype(str).str.replace("-", "").str[:8].unique()
            )

    periods_to_download = [p for p in periods if p not in existing_periods]
    if not periods_to_download:
        return existing_df

    logger.info(
        f"[{dataset_name}] 按季度批量下载: {len(periods_to_download)} 个季度"
    )

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for period in periods_to_download:
        try:
            df = _query_with_pagination(
                client, api_name, fields=fields, period=period,
            )
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.debug(f"[{dataset_name}] {period} 失败: {e}")

    if all_dfs:
        existing_df = _save_merged_bulk(
            storage, dataset_name, all_dfs, existing_df, dedup_cols
        )

    elapsed_total = time.time() - t0
    logger.info(
        f"[{dataset_name}] 全量下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total:.0f}秒"
    )
    return existing_df


def _bulk_download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    dedup_cols: Optional[List[str]] = None,
    start_year: int = 2012,
) -> Optional[pd.DataFrame]:
    """按月批量下载股东人数全量数据（单次限3000条）

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dedup_cols: 去重列
        start_year: 起始年份

    Returns:
        下载并保存后的完整 DataFrame，或 None
    """
    import calendar
    import datetime as _dt

    if dedup_cols is None:
        dedup_cols = ["ts_code", "end_date"]

    current = _dt.datetime.now()
    # 生成月范围
    month_ranges = []
    dt = _dt.datetime(start_year, 1, 1)
    while dt <= current:
        m_start = dt.strftime("%Y%m%d")
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        m_end_dt = dt.replace(day=last_day)
        if m_end_dt > current:
            m_end_dt = current
        m_end = m_end_dt.strftime("%Y%m%d")
        month_ranges.append((m_start, m_end))
        if dt.month == 12:
            dt = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            dt = dt.replace(month=dt.month + 1, day=1)

    existing_df = storage.load_raw("stk_holdernumber")
    logger.info(f"[stk_holdernumber] 按月批量下载: {len(month_ranges)} 个月")

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for i, (m_start, m_end) in enumerate(month_ranges, 1):
        try:
            df = client.get_stk_holdernumber(start_date=m_start, end_date=m_end)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.debug(f"[stk_holdernumber] {m_start}~{m_end} 失败: {e}")

        if i % 24 == 0 or i == len(month_ranges):
            logger.info(
                f"[stk_holdernumber] [{i}/{len(month_ranges)}] "
                f"成功={success} 空={empty} 失败={errors}"
            )

    if all_dfs:
        existing_df = _save_merged_bulk(
            storage, "stk_holdernumber", all_dfs, existing_df, dedup_cols
        )

    elapsed_total = time.time() - t0
    logger.info(
        f"[stk_holdernumber] 全量下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total:.0f}秒"
    )
    return existing_df


def _try_download_fina_indicator(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载财务指标数据

    数据充足（>= 阈值）：按公告日区间补齐增量公告。
    数据不足或不存在：逐股全量下载全部历史财务指标。
    """
    existing = storage.load_raw("fina_indicator")

    if existing is not None and len(existing) >= _MIN_FINA_RECORDS:
        # 增量模式：数据量充足，按公告日区间补齐
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="fina_indicator",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_fina_indicator_by_date(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 fina_indicator 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"财务指标数据不足 (当前 {cnt} 条, 阈值 {_MIN_FINA_RECORDS})，"
        f"启动按季度批量下载..."
    )
    return _bulk_download_by_period(
        client, storage,
        dataset_name="fina_indicator",
        api_name="fina_indicator_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
        fields="ts_code,ann_date,end_date,roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy",
    )


def _try_download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载股东人数数据

    数据充足（>= 阈值）：按公告日区间补齐增量。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("stk_holdernumber")

    if existing is not None and len(existing) >= _MIN_HOLDER_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="stk_holdernumber",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date"],
                fetch_by_date=lambda d: client.get_stk_holdernumber(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 stk_holdernumber 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按月批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"股东人数数据不足 (当前 {cnt} 条, 阈值 {_MIN_HOLDER_RECORDS})，"
        f"启动按月批量下载..."
    )
    return _bulk_download_stk_holdernumber(
        client, storage,
        dedup_cols=["ts_code", "end_date"],
    )


def _try_download_forecast(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩预告数据

    数据充足（>= 阈值）：按公告日区间补齐增量。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("forecast")

    if existing is not None and len(existing) >= _MIN_FORECAST_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="forecast",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_forecast_by_date(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 forecast 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）— 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩预告数据不足 (当前 {cnt} 条, 阈值 {_MIN_FORECAST_RECORDS})，"
        f"启动按季度批量下载..."
    )
    return _bulk_download_by_period(
        client, storage,
        dataset_name="forecast",
        api_name="forecast_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_ensure_historical_cyq_perf(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的筹码胜率历史数据

    cyq_perf 按日分区存储，需要 20+ 天历史数据才能计算胜率变化率。
    使用 trade_date 参数一次获取全市场当日数据。

    注意: 调用方必须确保 trading_dates_str 只包含 <= trade_date 的历史日期，
    避免下载未来数据导致前视偏差。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD，仅历史日期）

    Returns:
        合并后的 cyq_perf DataFrame，或 None
    """
    downloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "cyq_perf", dt):
            continue
        try:
            df = client.get_cyq_perf(trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "cyq_perf", dt)
                downloaded += 1
        except Exception as e:
            logger.debug(f"cyq_perf {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"筹码胜率历史补齐: 新增 {downloaded} 个交易日")

    # 重新加载完整范围
    if trading_dates_str:
        from ..data.loader import DataLoader

        loader = DataLoader(storage)
        return loader.load_cyq_perf(trading_dates_str[0], trading_dates_str[-1])
    return None


def _try_download_express(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩快报数据

    数据充足：按公告日区间补齐增量快报。
    数据不足：逐股全量下载。
    """
    existing = storage.load_raw("express")

    if existing is not None and len(existing) >= _MIN_EXPRESS_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="express",
                existing_df=existing,
                trade_date=trade_date,
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=lambda d: client.get_express_vip(ann_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 express 失败: {e}")
            return existing

    # 全量下载 — 按季度批量
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩快报数据不足 (当前 {cnt} 条, 阈值 {_MIN_EXPRESS_RECORDS})，"
        f"启动按季度批量下载..."
    )
    return _bulk_download_by_period(
        client, storage,
        dataset_name="express",
        api_name="express_vip",
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_ensure_historical_fund_portfolio(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内所需的基金持仓季度分区数据

    fund_portfolio 按季度（end_date=季度末）分区存储。
    根据 trading_dates 覆盖的时间范围，向前回溯 2 年（因子需要历史持仓），
    检查每个季度分区是否存在，缺失则通过 API 下载。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD）

    Returns:
        合并后的 fund_portfolio DataFrame，或 None
    """
    if not trading_dates_str:
        return None

    # 根据交易日范围确定需要的季度
    import datetime as _dt

    min_date = min(trading_dates_str)
    max_date = max(trading_dates_str)
    # 回溯 1 年获取历史持仓（point-in-time 只需最近季报，缩短以降低内存占用）
    start_year = int(min_date[:4]) - 1
    end_year = int(max_date[:4])
    periods = _generate_quarter_periods(start_year, end_year)

    # 只保留 <= max_date 的季度（未来季度无数据）
    periods = [p for p in periods if p <= max_date]

    downloaded = 0
    for period in periods:
        if storage.is_data_exists("raw", "fund_portfolio", period):
            continue
        try:
            df = _query_with_pagination(
                client, "fund_portfolio", page_limit=8000, period=period,
            )
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "fund_portfolio", period)
                downloaded += 1
        except Exception as e:
            logger.debug(f"fund_portfolio {period} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"基金持仓历史补齐: 新增 {downloaded} 个季度")

    # 逐分区加载+聚合，避免一次性加载全量原始数据（可达百万行级）
    from ..factors.fund_portfolio import FUND_PORTFOLIO_RAW_COLS, _aggregate_fund_portfolio

    agg_dfs = []
    agg_dataset_name = "fund_portfolio_agg"
    for period in periods:
        agg = None
        if storage.is_data_exists("raw", agg_dataset_name, period):
            agg = storage.load_raw_by_date(agg_dataset_name, period)
        elif not storage.is_data_exists("raw", "fund_portfolio", period):
            continue
        else:
            raw_df = storage.load_raw_by_date(
                "fund_portfolio",
                period,
                columns=FUND_PORTFOLIO_RAW_COLS,
            )
            if raw_df is not None and len(raw_df) > 0:
                agg = _aggregate_fund_portfolio(raw_df)
                if agg is not None and len(agg) > 0:
                    storage.save_raw_by_date(agg, agg_dataset_name, period)
            raw_df = None
            gc.collect()

        if agg is not None and len(agg) > 0:
            agg_dfs.append(agg)
            agg = None
            gc.collect()

    if not agg_dfs:
        return None
    result = pd.concat(agg_dfs, ignore_index=True)
    logger.info(f"基金持仓: 逐分区聚合完成，{len(periods)} 个季度 → {len(result)} 条个股记录")
    return result


# ── 申万行业分类自动下载 ─────────────────────────────────────────


def _ensure_shenwan_industry(
    client: TushareClient,
    storage: Storage,
    cleaner: DataCleaner,
) -> Optional[pd.DataFrame]:
    """自动下载申万三级行业分类数据

    逻辑与 scripts/update_basic_data.py 中 update_shenwan_industry() 一致，
    但集成到 ensure 链路中，纸面交易可自动触发。

    Returns:
        申万行业分类 DataFrame，或 None（失败时）
    """
    try:
        # 1. 获取申万三级行业指数列表
        logger.info("获取申万三级行业指数列表...")
        index_classify = client.get_index_classify(level="L3", src="SW2021")
        if index_classify is None or len(index_classify) == 0:
            logger.warning("未获取到申万三级行业指数")
            return None

        if "index_code" not in index_classify.columns:
            logger.warning("index_classify 缺少 index_code 字段")
            return None

        sw_l3_indices = index_classify
        logger.info(f"获取到 {len(sw_l3_indices)} 个申万三级指数")

        # 2. 逐个获取成分股
        logger.info("获取各三级行业成分股...")
        index_members: Dict[str, pd.DataFrame] = {}
        success_count = 0

        for _, row in sw_l3_indices.iterrows():
            index_code = row["index_code"]
            try:
                members = client.get_index_member(l3_code=index_code)
                if len(members) > 0:
                    index_members[index_code] = members
                    success_count += 1
            except Exception as e:
                logger.debug(f"获取 {index_code} 成分股失败: {e}")

        logger.info(f"成功获取 {success_count}/{len(sw_l3_indices)} 个三级行业成分股")

        if success_count == 0:
            logger.warning("未获取到任何行业成分股数据")
            return None

        # 3. 清洗并保存
        clean_data = cleaner.clean_shenwan_industry(
            sw_l3_indices, index_members, level_str="l3"
        )
        if len(clean_data) == 0:
            logger.warning("申万行业清洗后无有效数据")
            return None

        storage.save_raw(clean_data, "shenwan_industry", is_force=True)
        logger.info(f"申万行业分类已自动下载: {len(clean_data)} 条映射")
        return clean_data

    except Exception as e:
        logger.warning(f"自动下载申万行业分类失败: {e}")
        return None


# ── 融资融券历史数据补齐 ─────────────────────────────────────────


def _try_ensure_historical_margin(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的融资融券历史数据

    margin_detail 按日分区存储，需要 20+ 天历史数据才能计算滚动变化率。
    遍历每个交易日，若分区不存在则单独下载并保存。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD）

    Returns:
        合并后的 margin_detail DataFrame，或 None
    """
    downloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "margin_detail", dt):
            continue
        try:
            df = client.query("margin_detail", trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "margin_detail", dt)
                downloaded += 1
        except Exception as e:
            logger.debug(f"margin_detail {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"融资融券历史补齐: 新增 {downloaded} 个交易日")

    # 重新加载完整范围
    if trading_dates_str:
        from ..data.loader import DataLoader

        loader = DataLoader(storage)
        return loader.load_margin_detail(trading_dates_str[0], trading_dates_str[-1])
    return None


# ── 北向资金 / 龙虎榜 / 一致预期 历史数据补齐 ────────────────────


def _try_ensure_historical_moneyflow_hsgt(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的北向资金历史数据

    moneyflow_hsgt 按日分区存储, 需要 20+ 天历史才能计算 z-score 与 streak。
    支持单次按 start_date/end_date 批量拉取（优于逐日循环）。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD, 仅历史日期）

    Returns:
        合并后的 moneyflow_hsgt DataFrame, 或 None
    """
    if not trading_dates_str:
        return None

    missing_dates = [
        dt for dt in trading_dates_str
        if not storage.is_data_exists("raw", "moneyflow_hsgt", dt)
    ]
    if missing_dates:
        # moneyflow_hsgt 单次返回上限 300 条 (约 14 个月), 需按半年分段拉取以覆盖长历史
        missing_set = set(missing_dates)
        seg_start = missing_dates[0]
        seg_end = missing_dates[-1]
        # 生成半年段 (从 seg_start 往后, 每 6 个日历月一段)
        from datetime import datetime, timedelta
        segments: List[tuple] = []
        cursor = datetime.strptime(seg_start, "%Y%m%d")
        end_dt = datetime.strptime(seg_end, "%Y%m%d")
        while cursor <= end_dt:
            nxt = cursor + timedelta(days=180)
            if nxt > end_dt:
                nxt = end_dt
            segments.append((cursor.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
            cursor = nxt + timedelta(days=1)

        saved = 0
        for s, e in segments:
            try:
                df = client.get_moneyflow_hsgt(start_date=s, end_date=e)
                if df is None or df.empty:
                    continue
                df["trade_date"] = (
                    df["trade_date"].astype(str).str.replace("-", "").str[:8]
                )
                for dt, grp in df.groupby("trade_date"):
                    if dt in missing_set:
                        storage.save_raw_by_date(grp, "moneyflow_hsgt", dt)
                        saved += 1
            except Exception as e:
                logger.warning(f"北向资金 {s}~{e} 分段下载失败: {e}")
        if saved > 0:
            logger.info(f"北向资金历史补齐: 新增 {saved} 个交易日")

    from ..data.loader import DataLoader
    loader = DataLoader(storage)
    return loader.load_moneyflow_hsgt(
        trading_dates_str[0], trading_dates_str[-1]
    )


def _try_ensure_historical_top_list(
    client: TushareClient,
    storage: Storage,
    trading_dates_str: List[str],
) -> Optional[pd.DataFrame]:
    """补齐日期范围内缺失的龙虎榜历史数据

    top_list 按日分区, 稀疏数据（大多数日期只有个位数到几十条）。
    逐日循环下载, 下载失败或空记录都保存空标记以避免重复尝试。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trading_dates_str: 需要覆盖的交易日列表（YYYYMMDD, 仅历史日期）

    Returns:
        合并后的 top_list DataFrame, 或 None
    """
    if not trading_dates_str:
        return None

    downloaded = 0
    for dt in trading_dates_str:
        if storage.is_data_exists("raw", "top_list", dt):
            continue
        try:
            df = client.get_top_list(trade_date=dt)
            if df is not None and not df.empty:
                storage.save_raw_by_date(df, "top_list", dt)
                downloaded += 1
            else:
                # 当日无上榜股票, 保存空 DataFrame 占位避免重复下载
                storage.save_raw_by_date(
                    pd.DataFrame({"trade_date": [dt]}), "top_list", dt,
                )
        except Exception as e:
            logger.debug(f"top_list {dt} 下载失败: {e}")

    if downloaded > 0:
        logger.info(f"龙虎榜历史补齐: 新增 {downloaded} 个交易日")

    from ..data.loader import DataLoader
    loader = DataLoader(storage)
    return loader.load_top_list(trading_dates_str[0], trading_dates_str[-1])


def _try_download_report_rc(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载一致预期研报数据

    数据充足（>= 阈值）: 按 report_date 区间补齐增量研报。
    数据不足或不存在: 按年份批量回溯下载（report_rc 每次返回 2000 条, 需分页）。
    """
    existing = storage.load_raw("report_rc")

    if existing is not None and len(existing) >= _MIN_REPORT_RC_RECORDS:
        try:
            return _incremental_catchup_by_calendar_date(
                storage=storage,
                dataset_name="report_rc",
                existing_df=existing,
                trade_date=trade_date,
                date_col="report_date",
                dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
                fetch_by_date=lambda d: client.get_report_rc(report_date=d),
            )
        except Exception as e:
            logger.warning(f"增量下载 report_rc 失败: {e}")
            return existing

    # 全量下载 — 按年分页
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"一致预期数据不足 (当前 {cnt} 条, 阈值 {_MIN_REPORT_RC_RECORDS})，"
        f"启动按年批量下载..."
    )
    import datetime as _dt
    current_year = _dt.datetime.now().year
    all_pages: List[pd.DataFrame] = []
    for year in range(current_year - 5, current_year + 1):
        try:
            df = _query_with_pagination(
                client,
                "report_rc",
                start_date=f"{year}0101",
                end_date=f"{year}1231",
            )
            if df is not None and len(df) > 0:
                all_pages.append(df)
                logger.info(f"  report_rc {year} 年: {len(df)} 条")
        except Exception as e:
            logger.warning(f"  report_rc {year} 年下载失败: {e}")
    if not all_pages:
        return existing
    merged = pd.concat(all_pages, ignore_index=True)
    result = _append_and_save_raw(
        storage, "report_rc", merged,
        dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
    )
    logger.info(f"一致预期全量下载完成: 总计 {len(result)} 条")
    return result


# ── Features 缓存完整性校验 ──────────────────────────────────────

# 已缓存 features 必须包含的因子列（缺失则触发重建）
# 每个因子组至少一个代表性列，确保旧缓存或因子组缺失时自动淘汰
_REQUIRED_FACTOR_COLS = [
    "rzye_chg_5", "rzye_chg_20", "rqye_rzye_ratio",       # 融资融券
    "zscore_bp", "zscore_dv_ttm", "zscore_amount_ma20",    # 截面 z-score
    "neu_ret_5",                                            # 行业中性化收益
    "alpha_industry_5",                                     # 行业 alpha
    "ind_momentum_rank",                                    # 行业动量
    "mkt_atr_pct",                                          # 市场级 ATR 当前值
    "mkt_atr_pct_ma250",                                    # 市场级 ATR 250 日均值
    "roe_waa",                                              # 基本面因子
    "fundamental_freshness_days",                           # 基本面 freshness
    "holder_num_chg",                                       # 股东人数因子
    "holder_freshness_days",                                # 股东人数 freshness
    "forecast_type_score",                                  # 业绩预告因子
    "forecast_freshness_days",                              # 业绩预告 freshness
    "winner_rate",                                          # 筹码胜率因子
    "fund_hold_ratio",                                      # 基金持仓因子
    "fund_portfolio_freshness_days",                        # 基金持仓 freshness
    "express_revenue_yoy",                                  # 业绩快报因子
    "express_freshness_days",                               # 业绩快报 freshness
]


def _check_features_schema(
    storage: Storage, trade_date: str, subdir: str = "cs_train"
) -> bool:
    """快速检查已缓存 features 是否包含必要的因子列

    仅读取 Parquet schema（不加载数据），开销极低。
    若文件损坏或缺失必要列则返回 False，触发重建。
    """
    import pyarrow.parquet as pq

    target_path = storage.features_path / subdir
    file_path = target_path / f"{trade_date}.parquet"
    if not file_path.exists():
        return False

    try:
        schema = pq.read_schema(str(file_path))
        col_names = set(schema.names)
        missing = [c for c in _REQUIRED_FACTOR_COLS if c not in col_names]
        if missing:
            logger.debug(f"features 缓存缺失列: {missing}")
            return False
        return True
    except Exception:
        return False

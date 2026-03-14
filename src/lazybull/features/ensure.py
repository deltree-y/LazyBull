"""特征确保模块

提供确保 features 数据存在的封装函数
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from ..data import DataCleaner, DataLoader, Storage, TushareClient
from ..data.ensure import ensure_basic_data, ensure_clean_data_for_date
from .builder import FeatureBuilder

# 常量定义
FEATURE_DATA_HISTORY_MONTHS = 1  # 特征数据历史月数
FEATURE_DATA_FUTURE_MONTHS = 1   # 特征数据未来月数
HISTORICAL_DATA_MONTHS = 1       # 历史数据回看月数
MAX_HISTORICAL_DAYS = 30         # 最多检查的历史交易日数

# 逐股批量下载每批保存间隔（股票数）
_BULK_SAVE_INTERVAL = 500

# 因子数据最低记录数阈值，低于此值视为数据不足，触发全量下载
# 这些因子是 point-in-time 查询，需要全量历史才有意义
_MIN_FINA_RECORDS = 1000       # 财务指标：全量应有 10 万+ 条
_MIN_HOLDER_RECORDS = 500      # 股东人数：全量应有数万条
_MIN_FORECAST_RECORDS = 500    # 业绩预告：全量应有数万条


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
    # 检查是否已存在（同时校验关键因子列，防止旧缓存缺失列）
    if not force and storage.is_feature_exists(trade_date):
        if _check_features_schema(storage, trade_date):
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
        
        # 5. 加载 clean 日线数据（扩展范围以包含历史数据）
        start_dt = pd.to_datetime(trade_date, format='%Y%m%d') - pd.DateOffset(
            months=FEATURE_DATA_HISTORY_MONTHS
        )
        end_dt = pd.to_datetime(trade_date, format='%Y%m%d') + pd.DateOffset(
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
        
        # 强制检查 moneyflow 数据（新模型必须依赖）
        if moneyflow_clean is None or moneyflow_clean.empty:
            logger.error(
                f"缺少 clean moneyflow 数据: {trade_date}\n"
                f"新模型训练需要资金流向特征，请先补齐 moneyflow 数据。\n"
                f"推荐步骤：\n"
                f"  1. 下载 raw moneyflow: python scripts/download_raw.py --data-type moneyflow --start-date {start_dt.strftime('%Y%m%d')} --end-date {end_dt.strftime('%Y%m%d')}\n"
                f"  2. 构建 clean moneyflow: python scripts/build_clean_features.py --start-date {start_dt.strftime('%Y%m%d')} --end-date {end_dt.strftime('%Y%m%d')}"
            )
            return False, []

        logger.info(f"clean 日线数据: {len(daily_clean)} 条记录")
        logger.info(f"clean moneyflow 数据: {len(moneyflow_clean)} 条记录")

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

        funda_today, margin_today, holder_today, earnings_today, hot_rank_today, missing_factors = (
            _load_factor_data(loader, client, storage, trade_date, trading_dates_str,
                              start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
        )

        # 8. 构建特征（无需传递 adj_factor，clean 数据已包含复权价格）
        features_df = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=trade_cal,
            daily_data=daily_clean,
            adj_factor=pd.DataFrame(),  # 空 DataFrame，clean 数据已包含复权价格
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
            hot_rank_data=hot_rank_today,
        )
        
        # 9. 保存结果
        if len(features_df) > 0:
            storage.save_cs_train_day(features_df, trade_date)#, has_label=builder.require_label)
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
    
    Features 构建需要历史数据来计算动量、均值等特征
    这里确保过去一个月的交易日数据存在
    
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
    
    # 获取过去一个月的交易日
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

    Args:
        loader: DataLoader 实例
        client: TushareClient 实例（用于按日增量下载）
        storage: Storage 实例（用于保存增量数据）
        trade_date: 目标交易日，格式 YYYYMMDD
        trading_dates_str: 日期范围内的交易日列表
        start_date: 数据范围起始日期
        end_date: 数据范围结束日期

    Returns:
        (funda_today, margin_today, holder_today, earnings_today, hot_rank_today, missing_factors)
        前 5 个元素为当日的 DataFrame 或 None，最后一个为缺失因子名称列表
    """
    missing_factors = []

    # ── 基本面因子 ──────────────────────────────────────────
    funda_today = None
    fina_indicator = loader.load_fina_indicator()
    # 数据不存在、或记录过少（之前单日增量下载的残留）均触发全量下载
    if fina_indicator is None or len(fina_indicator) < _MIN_FINA_RECORDS:
        fina_indicator = _try_download_fina_indicator(client, storage, trade_date)
    if fina_indicator is not None and len(fina_indicator) > 0:
        from ..factors.fundamental import build_fundamental_lookup_by_date
        funda_lookup = build_fundamental_lookup_by_date(fina_indicator, trading_dates_str)
        funda_today = funda_lookup.get(trade_date)
        logger.info(f"基本面因子: 已加载 ({len(fina_indicator)} 条原始记录)")
    else:
        missing_factors.append("fina_indicator（基本面）")

    # ── 融资融券 ────────────────────────────────────────────
    # margin 按日分区存储，且需要 20+ 天历史数据计算滚动变动率
    margin_today = None
    # 只补齐 <= trade_date 的历史分区（未来日期实盘不可获取）
    hist_dates = [d for d in trading_dates_str if d <= trade_date]
    _try_ensure_historical_margin(client, storage, hist_dates)
    margin_detail = loader.load_margin_detail(start_date, end_date)
    if margin_detail is not None and len(margin_detail) > 0:
        from ..factors.margin import build_margin_lookup_by_date
        margin_lookup = build_margin_lookup_by_date(margin_detail, trading_dates_str)
        margin_today = margin_lookup.get(trade_date)
        logger.info(f"融资融券因子: 已加载 ({len(margin_detail)} 条)")
    else:
        missing_factors.append("margin_detail（融资融券）")

    # ── 股东人数 ────────────────────────────────────────────
    holder_today = None
    stk_holdernumber = loader.load_stk_holdernumber()
    if stk_holdernumber is None or len(stk_holdernumber) < _MIN_HOLDER_RECORDS:
        stk_holdernumber = _try_download_stk_holdernumber(client, storage, trade_date)
    if stk_holdernumber is not None and len(stk_holdernumber) > 0:
        from ..factors.holder import build_holder_lookup_by_date
        holder_lookup = build_holder_lookup_by_date(stk_holdernumber, trading_dates_str)
        holder_today = holder_lookup.get(trade_date)
        logger.info(f"股东人数因子: 已加载 ({len(stk_holdernumber)} 条)")
    else:
        missing_factors.append("stk_holdernumber（股东人数）")

    # ── 业绩预告 ────────────────────────────────────────────
    earnings_today = None
    forecast_df = loader.load_forecast()
    if forecast_df is None or len(forecast_df) < _MIN_FORECAST_RECORDS:
        forecast_df = _try_download_forecast(client, storage, trade_date)
    if forecast_df is not None and len(forecast_df) > 0:
        from ..factors.earnings import build_earnings_lookup_by_date
        earnings_lookup = build_earnings_lookup_by_date(
            forecast_df, trading_dates_str
        )
        earnings_today = earnings_lookup.get(trade_date)
        logger.info(f"业绩预告因子: 已加载 ({len(forecast_df)} 条)")
    else:
        missing_factors.append("forecast（业绩预告）")

    # ── 东财人气榜 ──────────────────────────────────────────
    hot_rank_today = None
    hot_rank_df = loader.load_hot_rank()
    if hot_rank_df is None or len(hot_rank_df) == 0:
        hot_rank_df = _try_download_hot_rank(storage)
    if hot_rank_df is not None and len(hot_rank_df) > 0:
        from ..factors.hot_rank import build_hot_rank_lookup_by_date
        hot_rank_lookup = build_hot_rank_lookup_by_date(hot_rank_df, trading_dates_str)
        hot_rank_today = hot_rank_lookup.get(trade_date)
        logger.info(f"人气榜因子: 已加载 ({len(hot_rank_df)} 条)")
    else:
        missing_factors.append("hot_rank（人气榜）")

    # ── 汇总报告 ────────────────────────────────────────────
    total = 5
    loaded = total - len(missing_factors)
    if missing_factors:
        logger.warning(
            f"因子数据覆盖: {loaded}/{total} 组已加载，"
            f"缺失: {', '.join(missing_factors)}\n"
            f"  如需补全请运行: python scripts/download_raw.py --download <数据类型>"
        )
    else:
        logger.info(f"因子数据覆盖: {total}/{total} 组全部加载")

    return funda_today, margin_today, holder_today, earnings_today, hot_rank_today, missing_factors


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


def _bulk_download_per_stock(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    stock_codes: List[str],
    fields: Optional[str] = None,
    dedup_cols: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """逐股批量下载因子数据（首次全量，支持断点续传）

    当 raw 文件不存在或为空时，需要逐股下载全量历史数据。
    这些因子（fina_indicator/stk_holdernumber/forecast）采用 point-in-time
    查询，必须有全量历史才能通过 bisect 查找到 ann_date <= trade_date 的记录。

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dataset_name: 数据集名称
        api_name: TuShare API 名称
        stock_codes: 全部股票代码列表
        fields: 返回字段
        dedup_cols: 去重列

    Returns:
        下载并保存后的完整 DataFrame，或 None
    """
    # 断点续传：跳过已有数据的股票
    existing_codes: Set[str] = set()
    existing_df = storage.load_raw(dataset_name)
    if existing_df is not None and len(existing_df) > 0:
        existing_codes = set(existing_df["ts_code"].unique())
        logger.info(f"[{dataset_name}] 已有 {len(existing_codes)} 只股票数据（断点续传）")

    codes_to_download = [c for c in stock_codes if c not in existing_codes]
    if not codes_to_download:
        logger.info(f"[{dataset_name}] 所有股票数据已存在")
        return existing_df

    logger.info(
        f"[{dataset_name}] 首次全量下载: 待下载 {len(codes_to_download)} 只股票 "
        f"(共 {len(stock_codes)} 只)"
    )

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for i, ts_code in enumerate(codes_to_download, 1):
        try:
            kwargs = {"ts_code": ts_code}
            if fields:
                df = client.query(api_name, fields=fields, **kwargs)
            else:
                df = client.query(api_name, **kwargs)

            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            errors += 1
            logger.debug(f"[{dataset_name}] {ts_code} 失败: {e}")

        if i % 100 == 0 or i == len(codes_to_download):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            remaining = (len(codes_to_download) - i) / speed if speed > 0 else 0
            logger.info(
                f"[{dataset_name}] [{i}/{len(codes_to_download)}] "
                f"成功={success} 空={empty} 失败={errors} "
                f"速度={speed:.0f}只/秒 剩余≈{remaining / 60:.1f}分钟"
            )

        # 每批保存中间结果，防止中断丢失
        if i % _BULK_SAVE_INTERVAL == 0 and all_dfs:
            existing_df = _save_merged_bulk(
                storage, dataset_name, all_dfs, existing_df, dedup_cols
            )
            all_dfs = []

    # 最终保存
    if all_dfs:
        existing_df = _save_merged_bulk(
            storage, dataset_name, all_dfs, existing_df, dedup_cols
        )

    elapsed_total = time.time() - t0
    logger.info(
        f"[{dataset_name}] 全量下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total / 60:.1f}分钟"
    )
    return existing_df


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


def _get_stock_codes(storage: Storage) -> List[str]:
    """从 stock_basic 获取全部股票代码列表"""
    stock_basic = storage.load_raw("stock_basic")
    if stock_basic is None or len(stock_basic) == 0:
        logger.warning("未找到 stock_basic 数据，无法执行逐股下载")
        return []
    return sorted(stock_basic["ts_code"].unique().tolist())


def _try_download_fina_indicator(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载财务指标数据

    数据充足（>= 阈值）：按公告日增量下载当日新公告。
    数据不足或不存在：逐股全量下载全部历史财务指标。
    """
    existing = storage.load_raw("fina_indicator")

    if existing is not None and len(existing) >= _MIN_FINA_RECORDS:
        # 增量模式：数据量充足，按公告日下载
        try:
            logger.info(f"增量下载 fina_indicator (ann_date={trade_date})...")
            new_df = client.get_fina_indicator_by_date(ann_date=trade_date)
            if new_df is not None and len(new_df) > 0:
                result = _append_and_save_raw(
                    storage, "fina_indicator", new_df,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                )
                logger.info(
                    f"  fina_indicator 增量: 新增 {len(new_df)} 条, 总计 {len(result)} 条"
                )
                return result
            else:
                logger.info(f"  fina_indicator: {trade_date} 无新公告")
                return existing
        except Exception as e:
            logger.warning(f"增量下载 fina_indicator 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）
    stock_codes = _get_stock_codes(storage)
    if not stock_codes:
        return None
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"财务指标数据不足 (当前 {cnt} 条, 阈值 {_MIN_FINA_RECORDS})，"
        f"启动全量逐股下载..."
    )
    return _bulk_download_per_stock(
        client, storage,
        dataset_name="fina_indicator",
        api_name="fina_indicator",
        stock_codes=stock_codes,
        fields="ts_code,ann_date,end_date,roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy",
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_download_stk_holdernumber(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载股东人数数据

    数据充足（>= 阈值）：按公告日增量下载。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("stk_holdernumber")

    if existing is not None and len(existing) >= _MIN_HOLDER_RECORDS:
        try:
            logger.info(f"增量下载 stk_holdernumber (ann_date={trade_date})...")
            new_df = client.get_stk_holdernumber_by_date(ann_date=trade_date)
            if new_df is not None and len(new_df) > 0:
                result = _append_and_save_raw(
                    storage, "stk_holdernumber", new_df,
                    dedup_cols=["ts_code", "end_date"],
                )
                logger.info(
                    f"  stk_holdernumber 增量: 新增 {len(new_df)} 条, 总计 {len(result)} 条"
                )
                return result
            else:
                logger.info(f"  stk_holdernumber: {trade_date} 无新公告")
                return existing
        except Exception as e:
            logger.warning(f"增量下载 stk_holdernumber 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）
    stock_codes = _get_stock_codes(storage)
    if not stock_codes:
        return None
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"股东人数数据不足 (当前 {cnt} 条, 阈值 {_MIN_HOLDER_RECORDS})，"
        f"启动全量逐股下载..."
    )
    return _bulk_download_per_stock(
        client, storage,
        dataset_name="stk_holdernumber",
        api_name="stk_holdernumber",
        stock_codes=stock_codes,
        dedup_cols=["ts_code", "end_date"],
    )


def _try_download_forecast(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
) -> Optional[pd.DataFrame]:
    """下载业绩预告数据

    数据充足（>= 阈值）：按公告日增量下载。
    数据不足或不存在：逐股全量下载。
    """
    existing = storage.load_raw("forecast")

    if existing is not None and len(existing) >= _MIN_FORECAST_RECORDS:
        try:
            logger.info(f"增量下载 forecast (ann_date={trade_date})...")
            new_df = client.get_forecast_by_date(ann_date=trade_date)
            if new_df is not None and len(new_df) > 0:
                result = _append_and_save_raw(
                    storage, "forecast", new_df,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                )
                logger.info(
                    f"  forecast 增量: 新增 {len(new_df)} 条, 总计 {len(result)} 条"
                )
                return result
            else:
                logger.info(f"  forecast: {trade_date} 无新公告")
                return existing
        except Exception as e:
            logger.warning(f"增量下载 forecast 失败: {e}")
            return existing

    # 全量下载（首次或数据不足）
    stock_codes = _get_stock_codes(storage)
    if not stock_codes:
        return None
    cnt = len(existing) if existing is not None else 0
    logger.info(
        f"业绩预告数据不足 (当前 {cnt} 条, 阈值 {_MIN_FORECAST_RECORDS})，"
        f"启动全量逐股下载..."
    )
    return _bulk_download_per_stock(
        client, storage,
        dataset_name="forecast",
        api_name="forecast",
        stock_codes=stock_codes,
        dedup_cols=["ts_code", "end_date", "ann_date"],
    )


def _try_download_hot_rank(
    storage: Storage,
) -> Optional[pd.DataFrame]:
    """尝试增量下载东财人气榜当日快照并追加保存"""
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare 未安装，跳过人气榜下载")
        return None

    try:
        logger.info("增量下载 hot_rank (当日全市场快照)...")
        df = ak.stock_hot_rank_em()
        if df is None or len(df) == 0:
            logger.info("  hot_rank: 当日快照为空（非交易时段或节假日）")
            return storage.load_raw("hot_rank")

        # 标准化列名（AKShare 返回中文列名）
        col_map = {}
        for col in df.columns:
            if "代码" in col:
                col_map[col] = "symbol"
            elif "排名" in col or "序号" in col:
                col_map[col] = "hot_rank"
        df = df.rename(columns=col_map)

        if "symbol" not in df.columns or "hot_rank" not in df.columns:
            logger.warning(f"  hot_rank 快照列名不匹配: {df.columns.tolist()}")
            return storage.load_raw("hot_rank")

        def to_ts_code(sym):
            s = str(sym).zfill(6)
            if s.startswith(("6", "9")):
                return f"{s}.SH"
            return f"{s}.SZ"

        today = pd.Timestamp.now().strftime("%Y%m%d")
        df["ts_code"] = df["symbol"].apply(to_ts_code)
        df["trade_date"] = today
        df = df[["ts_code", "trade_date", "hot_rank"]]

        result = _append_and_save_raw(
            storage, "hot_rank", df,
            dedup_cols=["ts_code", "trade_date"],
        )
        logger.info(f"  hot_rank 增量: 新增 {len(df)} 条, 总计 {len(result)} 条")
        return result
    except Exception as e:
        logger.warning(f"增量下载 hot_rank 失败: {e}")
        return storage.load_raw("hot_rank")


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


# ── Features 缓存完整性校验 ──────────────────────────────────────

# 已缓存 features 必须包含的因子列（缺失则触发重建）
# 包含融资融券 + 行业中性化特征，确保旧缓存被自动淘汰
_REQUIRED_FACTOR_COLS = [
    "rzye_chg_5", "rzye_chg_20", "rqye_rzye_ratio",       # 融资融券
    "zscore_bp", "zscore_dv_ttm", "zscore_amount_ma20",    # 截面 z-score
    "neu_ret_5",                                            # 行业中性化收益
    "alpha_industry_5",                                     # 行业 alpha
    "ind_momentum_rank",                                    # 行业动量
    "hot_rank", "hot_rank_chg_5",                           # 东财人气榜
]


def _check_features_schema(storage: Storage, trade_date: str) -> bool:
    """快速检查已缓存 features 是否包含必要的因子列

    仅读取 Parquet schema（不加载数据），开销极低。
    若文件损坏或缺失必要列则返回 False，触发重建。
    """
    import pyarrow.parquet as pq

    cs_train_path = storage.features_path / "cs_train"
    file_path = cs_train_path / f"{trade_date}.parquet"
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

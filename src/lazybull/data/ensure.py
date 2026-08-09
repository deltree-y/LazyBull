"""数据确保模块

提供确保 raw/clean 数据存在的封装函数，按模块边界分层处理依赖
"""

import math
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .cleaner import DataCleaner, has_usable_adjusted_prices
from .loader import DataLoader
from .storage import Storage
from .tushare_client import TushareClient

# 常量定义
TRADE_CAL_HISTORY_MONTHS = 6  # 交易日历历史数据月数
TRADE_CAL_FUTURE_MONTHS = 6  # 交易日历未来数据月数
MIN_LIST_DAYS = 60  # 最小上市天数（约2个月交易日，用于稳定性分析）
# daily 覆盖度阈值（占 stock_basic 全集比例）：正常交易日绝大多数股票有交易（95%+），
# 明显低于此值提示可能截断/部分返回，触发强制重下并在仍异常时 error 告警
_DAILY_COVERAGE_MIN_RATIO = 0.85
_DAILY_BASIC_CODE_DIFF_RATIO = 0.02


def _is_trade_date(storage: Storage, trade_date: str) -> Optional[bool]:
    """查询当日是否交易日（基于已缓存 trade_cal）。

    Returns:
        True/False：当日为/非交易日
        None：无法确认（trade_cal 缺失或格式异常），调用方应保守处理
    """
    try:
        trade_cal = storage.load_raw("trade_cal")
        if trade_cal is None or len(trade_cal) == 0:
            return None
        if "cal_date" not in trade_cal.columns or "is_open" not in trade_cal.columns:
            return None
        cal = trade_cal.copy()
        if pd.api.types.is_datetime64_any_dtype(cal["cal_date"]):
            cal["cal_date"] = cal["cal_date"].dt.strftime("%Y%m%d")
        else:
            cal["cal_date"] = cal["cal_date"].astype(str).str.replace("-", "")
        row = cal[cal["cal_date"] == str(trade_date)]
        if len(row) == 0:
            return None
        return int(row.iloc[0]["is_open"]) == 1
    except Exception as e:
        logger.warning(f"查询交易日历失败: {trade_date}（{e}）")
        return None


def _listed_count_by_date(stock_basic: pd.DataFrame, trade_date: str) -> Optional[int]:
    """统计截至 trade_date 已上市（list_date <= trade_date）的股票数；缺 list_date 列返回 None。"""
    if "list_date" not in stock_basic.columns:
        return None
    s = stock_basic.copy()
    if pd.api.types.is_datetime64_any_dtype(s["list_date"]):
        s["list_date"] = s["list_date"].dt.strftime("%Y%m%d")
    else:
        s["list_date"] = s["list_date"].astype(str).str.replace("-", "")
    return int(s[s["list_date"] <= str(trade_date)]["ts_code"].nunique())


def _validate_daily_keys(data: pd.DataFrame, trade_date: str, name: str) -> Optional[str]:
    """校验日频数据主键，返回错误原因；通过时返回 None。"""
    required = {"ts_code", "trade_date"}
    missing = required.difference(data.columns)
    if missing:
        return f"缺少主键列: {sorted(missing)}"
    normalized_dates = data["trade_date"].astype(str).str.replace("-", "", regex=False)
    invalid_dates = int((normalized_dates != str(trade_date)).sum())
    if invalid_dates > 0:
        return f"包含 {invalid_dates} 条非目标日期记录"
    duplicate_rows = int(data.duplicated(subset=["ts_code", "trade_date"], keep=False).sum())
    if duplicate_rows > 0:
        return f"存在 {duplicate_rows} 条重复 (ts_code, trade_date) 主键"
    if data["ts_code"].isna().any():
        return "存在空 ts_code"
    return None


def _daily_basic_confirms_daily(daily: pd.DataFrame, daily_basic: pd.DataFrame) -> bool:
    """daily_basic 代码域是否足以交叉确认 daily 完整性。"""
    daily_codes = set(daily["ts_code"].astype(str))
    basic_codes = set(daily_basic["ts_code"].astype(str))
    universe_size = max(len(daily_codes), len(basic_codes))
    allowed_diff = 0
    if universe_size >= 100:
        allowed_diff = max(1, math.ceil(universe_size * _DAILY_BASIC_CODE_DIFF_RATIO))
    return len(daily_codes.symmetric_difference(basic_codes)) <= allowed_diff


def _is_daily_coverage_low(storage: Storage, trade_date: str, daily_rows: int) -> bool:
    """daily 行数是否显著低于"截至 trade_date 已上市"股票数（截断/部分返回/停牌潮）。

    - 分母按 list_date <= trade_date 过滤，避免用当前全集误伤历史日期；
    - 无法获得基准（无 stock_basic / 无 list_date 列）时返回 False（跳过检查）。
    注意：集合级错配（缺一只混入另一只且行数相近）无法被行数发现，仍属已知局限。
    """
    if daily_rows <= 0:
        return False
    try:
        stock_basic = storage.load_raw("stock_basic")
        if stock_basic is None or len(stock_basic) == 0 or "ts_code" not in stock_basic.columns:
            return False
        total = _listed_count_by_date(stock_basic, trade_date)
        if total is None:
            logger.warning("stock_basic 缺少 list_date，跳过 daily 历史覆盖率检查")
            return False
        return total > 0 and daily_rows < int(total * _DAILY_COVERAGE_MIN_RATIO)
    except Exception as e:
        logger.warning(f"daily 覆盖度检查失败: {trade_date}（{e}）")
        return False


def _validate_daily_candidate(
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    daily: pd.DataFrame,
) -> Tuple[bool, Optional[pd.DataFrame], int]:
    """校验 daily 候选；低覆盖时使用独立 daily_basic 代码域交叉确认。"""
    key_error = _validate_daily_keys(daily, trade_date, "daily")
    if key_error is not None:
        logger.error(f"daily 主键校验失败: {trade_date}（{key_error}）")
        return False, None, 0

    unique_codes = int(daily["ts_code"].nunique())
    if not _is_daily_coverage_low(storage, trade_date, unique_codes):
        return True, None, unique_codes

    daily_basic = storage.load_raw_by_date("daily_basic", trade_date)
    prefetched = None
    if daily_basic is None or daily_basic.empty:
        daily_basic = client.get_daily_basic(trade_date=trade_date)
        prefetched = daily_basic
    if daily_basic is None or daily_basic.empty:
        logger.error(f"daily 覆盖率偏低且 daily_basic 无法用于交叉确认: {trade_date}")
        return False, None, unique_codes
    basic_error = _validate_daily_keys(daily_basic, trade_date, "daily_basic")
    if basic_error is not None:
        logger.error(f"daily_basic 主键校验失败: {trade_date}（{basic_error}）")
        return False, None, unique_codes
    if not _daily_basic_confirms_daily(daily, daily_basic):
        logger.error(
            f"daily 覆盖率偏低且与 daily_basic 代码域不一致: {trade_date}，"
            f"daily={unique_codes}，daily_basic={daily_basic['ts_code'].nunique()}"
        )
        return False, None, unique_codes
    logger.warning(f"daily 相对历史上市域覆盖率偏低，但 daily_basic 代码域已交叉确认: {trade_date}")
    return True, prefetched, unique_codes


def ensure_raw_data_for_date(
    client: TushareClient, storage: Storage, trade_date: str, force: bool = False
) -> bool:
    """确保指定日期的 raw 数据存在，不存在则下载

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新下载

    Returns:
        是否成功（True 表示数据已存在或下载成功）
    """
    daily_exists = not force and storage.is_data_exists("raw", "daily", trade_date)

    # 日线等核心数据已存在时，仍需检查可选依赖（moneyflow / daily_basic / margin_detail）
    if daily_exists:
        logger.debug(f"raw 核心数据已存在: {trade_date}，检查可选依赖")
    else:
        logger.info(f"下载 raw 数据: {trade_date}")

    daily_rows = 0
    prefetched_daily_basic: Optional[pd.DataFrame] = None
    try:
        if not daily_exists:
            # 下载日线行情
            daily_data = client.get_daily(trade_date=trade_date)
            if daily_data.empty:
                # 区分非交易日与交易日接口故障：避免把接口故障误报为成功
                is_open = _is_trade_date(storage, trade_date)
                if is_open is True:
                    logger.error(f"交易日 {trade_date} 日线接口返回空（可能为接口故障），视为失败")
                    return False
                if is_open is None:
                    logger.warning(
                        f"日线数据为空: {trade_date}，且无法从交易日历确认是否非交易日，视为失败"
                    )
                    return False
                logger.info(f"非交易日: {trade_date}，跳过当日其余数据补齐")
                return True
            valid, prefetched_daily_basic, daily_rows = _validate_daily_candidate(
                client, storage, trade_date, daily_data
            )
            if not valid:
                return False
            storage.save_raw_by_date(daily_data, "daily", trade_date)
            logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
            if prefetched_daily_basic is not None:
                storage.save_raw_by_date(prefetched_daily_basic, "daily_basic", trade_date)
        else:
            existing_daily = storage.load_raw_by_date("daily", trade_date)
            if existing_daily is None or existing_daily.empty:
                logger.error(f"daily 分区存在但无法读取有效数据: {trade_date}")
                return False
            valid, prefetched_daily_basic, daily_rows = _validate_daily_candidate(
                client, storage, trade_date, existing_daily
            )
            if not valid:
                logger.warning(f"daily 完整性校验失败: {trade_date}，强制重下一次")
                refetch = client.get_daily(trade_date=trade_date)
                if refetch.empty:
                    logger.error(
                        f"daily 重下返回空: {trade_date}（可能为接口故障），旧分区保留但视为失败"
                    )
                    return False
                valid, prefetched_daily_basic, daily_rows = _validate_daily_candidate(
                    client, storage, trade_date, refetch
                )
                if not valid:
                    logger.error(f"daily 重下后完整性仍异常: {trade_date}，旧分区保留但视为失败")
                    return False
                storage.save_raw_by_date(refetch, "daily", trade_date)
                logger.info(f"  daily 重下: 已保存 {len(refetch)} 条记录")
            if prefetched_daily_basic is not None:
                storage.save_raw_by_date(prefetched_daily_basic, "daily_basic", trade_date)

        # 以下数据独立检查补齐，不受 daily 是否存在影响：
        # 防止 daily 已落盘但某类数据因接口抖动缺失时永久无法补齐
        # 下载复权因子
        if force or not storage.is_data_exists(
            "raw", "adj_factor", trade_date, min_rows=daily_rows or None
        ):
            adj_factor = client.get_adj_factor(trade_date=trade_date)
            if not adj_factor.empty:
                storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
                logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
            else:
                logger.error(f"复权因子数据暂缺: {trade_date}，raw 补齐失败")
                return False

        # 下载停复牌信息
        if force or not storage.is_data_exists("raw", "suspend", trade_date):
            suspend = client.get_suspend_d(trade_date=trade_date)
            if not suspend.empty:
                storage.save_raw_by_date(suspend, "suspend", trade_date)
                logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
            else:
                # 空为合法（当日无停牌）：写占位空文件，避免下次 ensure 重复请求
                storage.save_raw_by_date(
                    pd.DataFrame(columns=["ts_code", "trade_date"]), "suspend", trade_date
                )
                logger.debug(f"停复牌为空（当日无停牌）: {trade_date}，已写占位")

        # 下载涨跌停信息（含指数，行数超 TuShare 单次上限，已分页）
        if force or not storage.is_data_exists(
            "raw", "stk_limit", trade_date, min_rows=daily_rows or None
        ):
            limit_up_down = client.get_stk_limit(trade_date=trade_date)
            if not limit_up_down.empty:
                storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
            else:
                logger.warning(f"涨跌停信息暂缺: {trade_date}")

        # 下载资金流向（T0 特征构建需要）
        # 注意：moneyflow 天然不覆盖全部 daily 股票（不含北交所等），不能以 daily 行数为下限
        if force or not storage.is_data_exists("raw", "moneyflow", trade_date):
            moneyflow = client.get_moneyflow(trade_date=trade_date)
            if not moneyflow.empty:
                storage.save_raw_by_date(moneyflow, "moneyflow", trade_date)
                logger.info(f"  资金流向: 已保存 {len(moneyflow)} 条记录")
            else:
                logger.warning(
                    f"资金流向数据暂缺: {trade_date}（TuShare 通常 18:00 后更新，T0 特征构建时需要）"
                )

        # 下载每日指标（pb/pe/换手率等，特征构建需要）
        # daily_basic 与 daily 代码域近似一致，按代码集合校验并容忍历史接口极少量差异
        daily_basic_existing = storage.load_raw_by_date("daily_basic", trade_date)
        daily_existing = storage.load_raw_by_date("daily", trade_date)
        daily_basic_valid = (
            daily_basic_existing is not None
            and not daily_basic_existing.empty
            and daily_existing is not None
            and _validate_daily_keys(daily_basic_existing, trade_date, "daily_basic") is None
            and _daily_basic_confirms_daily(daily_existing, daily_basic_existing)
        )
        if force or not daily_basic_valid:
            daily_basic = client.get_daily_basic(trade_date=trade_date)
            key_error = (
                None
                if daily_basic is None or daily_basic.empty
                else _validate_daily_keys(daily_basic, trade_date, "daily_basic")
            )
            if (
                daily_basic is not None
                and not daily_basic.empty
                and key_error is None
                and daily_existing is not None
                and _daily_basic_confirms_daily(daily_existing, daily_basic)
            ):
                storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
                logger.info(f"  每日指标: 已保存 {len(daily_basic)} 条记录")
            else:
                logger.warning(f"每日指标数据缺失或代码域不完整: {trade_date}")

        # 下载 ST 状态（按日口径用于写入 is_st）
        if force or not storage.is_data_exists("raw", "stock_st", trade_date):
            try:
                stock_st = client.get_stock_st(trade_date=trade_date)
                if stock_st is not None and not stock_st.empty:
                    storage.save_raw_by_date(stock_st, "stock_st", trade_date)
                    logger.info(f"  ST状态: 已保存 {len(stock_st)} 条记录")
            except Exception as e:
                logger.warning(f"ST状态数据获取失败: {trade_date}（{e}）")

        # 下载融资融券明细（按日分区，特征构建需要）
        if force or not storage.is_data_exists("raw", "margin_detail", trade_date):
            try:
                margin_detail = client.query("margin_detail", trade_date=trade_date)
                if margin_detail is not None and not margin_detail.empty:
                    storage.save_raw_by_date(margin_detail, "margin_detail", trade_date)
                    logger.info(f"  融资融券: 已保存 {len(margin_detail)} 条记录")
            except Exception as e:
                logger.warning(f"融资融券数据获取失败: {trade_date}（{e}）")

        return True

    except Exception as e:
        logger.error(f"下载 raw 数据失败 {trade_date}: {e}")
        return False


def ensure_basic_data(
    client: TushareClient, storage: Storage, end_date: str, force: bool = False
) -> bool:
    """确保基础数据（trade_cal 和 stock_basic）存在

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        end_date: 结束日期，用于判断数据是否够新，格式 YYYYMMDD
        force: 是否强制重新下载

    Returns:
        是否成功
    """
    logger.info("检查基础数据...")

    # 检查 trade_cal
    need_download_trade_cal = force or not storage.check_basic_data_freshness("trade_cal", end_date)
    if need_download_trade_cal:
        logger.info("下载交易日历...")
        try:
            # 扩展日期范围以包含足够的历史和未来数据
            start_dt = pd.to_datetime(end_date, format="%Y%m%d") - pd.DateOffset(
                months=TRADE_CAL_HISTORY_MONTHS
            )
            end_dt = pd.to_datetime(end_date, format="%Y%m%d") + pd.DateOffset(
                months=TRADE_CAL_FUTURE_MONTHS
            )

            trade_cal = client.get_trade_cal(
                start_date=start_dt.strftime("19901219"),
                end_date=f"{end_dt.year}1231",  # 直接指向目标年度最后一天
                exchange="SSE",
            )
            storage.save_raw(trade_cal, "trade_cal", is_force=True)
            logger.info(f"交易日历已下载: {len(trade_cal)} 条记录")
        except Exception as e:
            logger.error(f"下载交易日历失败: {e}")
            return False
    else:
        logger.info("交易日历已是最新")

    # 检查 stock_basic
    need_download_stock_basic = force or not storage.check_basic_data_freshness(
        "stock_basic", end_date
    )
    if need_download_stock_basic:
        logger.info("下载股票基本信息...")
        try:
            stock_basic = client.get_stock_basic(list_status="L")
            storage.save_raw(stock_basic, "stock_basic", is_force=True)
            logger.info(f"股票基本信息已下载: {len(stock_basic)} 条记录")
        except Exception as e:
            logger.error(f"下载股票基本信息失败: {e}")
            return False
    else:
        logger.info("股票基本信息已存在")

    return True


def ensure_clean_data_for_date(
    storage: Storage,
    loader: DataLoader,
    cleaner: DataCleaner,
    client: TushareClient,
    trade_date: str,
    force: bool = False,
) -> bool:
    """确保指定日期的 clean 数据存在，不存在则构建

    若发现 raw 数据缺失，会自动调用 ensure_raw_data_for_date 下载

    Args:
        storage: Storage 实例
        loader: DataLoader 实例
        cleaner: DataCleaner 实例
        client: TushareClient 实例（用于在 raw 缺失时下载）
        trade_date: 交易日期，格式 YYYYMMDD
        force: 是否强制重新构建

    Returns:
        是否成功
    """
    # 检查所有 clean 数据集是否都已存在
    daily_exists = not force and storage.is_data_exists("clean", "daily", trade_date)
    daily_basic_exists = not force and storage.is_data_exists("clean", "daily_basic", trade_date)
    moneyflow_exists = not force and storage.is_data_exists("clean", "moneyflow", trade_date)

    if daily_exists:
        existing_daily_clean = storage.load_clean_by_date("daily", trade_date)
        if not has_usable_adjusted_prices(existing_daily_clean):
            logger.warning(f"clean/daily 复权价不可用，将重新构建: {trade_date}")
            daily_exists = False

    if daily_exists and daily_basic_exists and moneyflow_exists:
        logger.debug(f"clean 数据已存在: {trade_date}")
        return True

    logger.info(f"构建 clean 数据: {trade_date}")

    # 确保 raw 数据存在（会下载 daily/daily_basic/moneyflow 等）
    if not ensure_raw_data_for_date(client, storage, trade_date, force):
        logger.error(f"无法获取 raw 数据: {trade_date}")
        return False

    try:
        # 确保基础 clean 数据存在
        _ensure_basic_clean_data(storage, cleaner)

        # 构建 clean/daily
        if not daily_exists:
            daily_raw = storage.load_raw_by_date("daily", trade_date)
            if daily_raw is None or daily_raw.empty:
                logger.error(f"未找到 raw 层 daily 数据: {trade_date}")
                return False

            adj_factor_raw = storage.load_raw_by_date("adj_factor", trade_date)
            if adj_factor_raw is None or adj_factor_raw.empty:
                logger.error(f"未找到复权因子，拒绝生成 clean/daily: {trade_date}")
                return False

            daily_clean = cleaner.clean_daily(daily_raw, adj_factor_raw)

            # 添加可交易标记
            stock_basic = loader.load_clean_stock_basic()
            if stock_basic is not None:
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
                    stock_basic,
                    stock_st_df=stock_st_clean,
                    suspend_info_df=suspend_clean,
                    limit_info_df=limit_clean,
                    min_list_days=MIN_LIST_DAYS,
                )

            storage.save_clean_by_date(daily_clean, "daily", trade_date)
            logger.info(f"已保存 clean/daily 数据: {len(daily_clean)} 条")

        # 构建 clean/daily_basic
        if not daily_basic_exists:
            daily_basic_raw = storage.load_raw_by_date("daily_basic", trade_date)
            if daily_basic_raw is not None and not daily_basic_raw.empty:
                daily_basic_clean = cleaner.clean_daily_basic(daily_basic_raw)
                storage.save_clean_by_date(daily_basic_clean, "daily_basic", trade_date)
                logger.info(f"已保存 clean/daily_basic 数据: {len(daily_basic_clean)} 条")

        # 构建 clean/moneyflow
        if not moneyflow_exists:
            moneyflow_raw = storage.load_raw_by_date("moneyflow", trade_date)
            if moneyflow_raw is not None and not moneyflow_raw.empty:
                moneyflow_clean = cleaner.clean_moneyflow(moneyflow_raw)
                storage.save_clean_by_date(moneyflow_clean, "moneyflow", trade_date)
                logger.info(f"已保存 clean/moneyflow 数据: {len(moneyflow_clean)} 条")

        return True

    except Exception as e:
        logger.error(f"构建 clean 数据失败 {trade_date}: {e}")
        return False


def _ensure_basic_clean_data(storage: Storage, cleaner: DataCleaner) -> None:
    """确保基础 clean 数据（trade_cal 和 stock_basic）存在

    内部辅助函数，不对外暴露
    """
    # 处理 trade_cal
    trade_cal_raw = storage.load_raw("trade_cal")
    if trade_cal_raw is not None:
        trade_cal_clean = cleaner.clean_trade_cal(trade_cal_raw)
        storage.save_clean(trade_cal_clean, "trade_cal", is_force=True)

    # 处理 stock_basic
    stock_basic_raw = storage.load_raw("stock_basic")
    if stock_basic_raw is not None:
        stock_basic_clean = cleaner.clean_stock_basic(stock_basic_raw)
        storage.save_clean(stock_basic_clean, "stock_basic", is_force=True)

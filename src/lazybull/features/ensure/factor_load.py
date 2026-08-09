# -*- coding: utf-8 -*-
"""ensure 子包：因子数据加载（12 组基本面 + 另类数据因子段落）。"""

import gc
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ...data import DataLoader, Storage, TushareClient
from .downloads import (
    _try_download_cashflow,
    _try_download_express,
    _try_download_fina_indicator,
    _try_download_forecast,
    _try_download_report_rc,
    _try_download_stk_holdernumber,
)
from .historical_assets import (
    _try_ensure_historical_cyq_perf,
    _try_ensure_historical_fund_portfolio,
    _try_ensure_historical_margin,
    _try_ensure_historical_moneyflow_hsgt,
    _try_ensure_historical_top_list,
)
from .incremental import _get_latest_date


def _has_announcement_gap(
    storage: Storage,
    df: Optional[pd.DataFrame],
    dataset_name: str,
    date_col: str,
    trade_date: str,
) -> bool:
    """判断公告/事件型因子是否存在数据缺口。

    缺口定义：本地数据缺失，或覆盖水位 < 目标交易日。
    覆盖水位语义（与 _incremental_catchup_by_calendar_date 保持一致）：
    - 已有同步水位（连续成功前缀）时，以水位为准；数据最新公告日可能跨过失败日，
      不可用来越过水位之后的未知区间；
    - 无水位时，以数据最新公告日初始化前缀。
    同步水位记录"成功查询至的日期"（无公告日也算已同步），避免空白日期被反复下载。
    基于覆盖判断而非记录数量——分区/单文件数据一旦齐全（无论新旧），
    记录数量永远充足，数量门控会让增量补齐永不触发。

    Args:
        storage: Storage 实例（读取同步水位）
        df: 已加载的本地因子数据
        dataset_name: 数据集名称
        date_col: 公告/事件日期列（ann_date / report_date）
        trade_date: 目标交易日 YYYYMMDD

    Returns:
        存在缺口返回 True（应触发下载/增量补齐）
    """
    latest = _get_latest_date(df, date_col)
    if latest is None:
        # 本地数据缺失/无有效日期列：即使水位存在也不能仅凭水位跳过——
        # parquet 可能被删除或损坏，需重新初始化/恢复。
        return True
    watermark = storage.load_sync_watermark(dataset_name)
    # 有水位（连续成功前缀）时，覆盖判断只认水位：数据最新公告日可能跨过
    # 失败日（后续成功落盘），不能用来越过水位之后的未知区间，否则失败日会被永久漏掉。
    # 与 _incremental_catchup_by_calendar_date 的起点语义保持一致。
    covered_to = watermark if watermark is not None else latest
    return covered_to < trade_date


def _load_factor_data(
    loader: DataLoader,
    client: TushareClient,
    storage: Storage,
    trade_date: str,
    trading_dates_str: List[str],
    start_date: str,
    end_date: str,
    daily_close_lookup: Optional[Dict[str, pd.DataFrame]] = None,
    block_close_lookup: Optional[Dict[str, Dict[str, float]]] = None,
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
        block_close_lookup: 未复权收盘价查询 {trade_date: {ts_code: close}}（大宗折价）

    Returns:
        因子当日截面与缺失列表元组。
        最后一个元素为缺失因子名称列表，其余元素为各因子当日 DataFrame/字典。
    """
    missing_factors = []
    # 纸面交易 T0 仅消费当日因子截面；历史窗口仍通过原始数据加载提供，
    # 无需为整段 trading_dates 物化完整的 date -> DataFrame 查询表。
    factor_output_dates = [trade_date]

    # ── 基本面因子 ──────────────────────────────────────────
    funda_today = None
    fina_indicator = loader.load_fina_indicator(start_date=trade_date, end_date=trade_date)
    # 无数据、或本地有效水位未覆盖目标交易日（存在缺口）→ 触发下载/增量补齐
    if _has_announcement_gap(storage, fina_indicator, "fina_indicator", "ann_date", trade_date):
        fina_indicator = _try_download_fina_indicator(client, storage, trade_date)
    if fina_indicator is not None and len(fina_indicator) > 0:
        from ...factors.fundamental import build_fundamental_lookup_by_date

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
        from ...factors.margin import build_margin_lookup_by_date

        margin_lookup = build_margin_lookup_by_date(margin_detail, factor_output_dates)
        margin_today = margin_lookup.get(trade_date)
        # 当日 margin_detail 可能尚未发布，额外重试一次下载
        if margin_today is None:
            logger.warning(f"融资融券: 当日 {trade_date} 数据不在查询表中，尝试单独下载...")
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
    if _has_announcement_gap(storage, stk_holdernumber, "stk_holdernumber", "ann_date", trade_date):
        stk_holdernumber = _try_download_stk_holdernumber(client, storage, trade_date)
    if stk_holdernumber is not None and len(stk_holdernumber) > 0:
        from ...factors.holder import build_holder_lookup_by_date

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
    if _has_announcement_gap(storage, forecast_df, "forecast", "ann_date", trade_date):
        forecast_df = _try_download_forecast(client, storage, trade_date)
    if forecast_df is not None and len(forecast_df) > 0:
        from ...factors.earnings import build_earnings_lookup_by_date

        earnings_lookup = build_earnings_lookup_by_date(forecast_df, factor_output_dates)
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
        from ...factors.cyq_perf import build_cyq_perf_lookup_by_date

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
    if _has_announcement_gap(storage, express_df, "express", "ann_date", trade_date):
        express_df = _try_download_express(client, storage, trade_date)
    if express_df is not None and len(express_df) > 0:
        from ...factors.express import build_express_lookup_by_date

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
        client,
        storage,
        fund_hist_dates,
    )
    if fund_portfolio_df is not None and len(fund_portfolio_df) > 0:
        from ...factors.fund_portfolio import build_fund_portfolio_lookup_by_date

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
        from ...factors.north_flow import build_north_flow_lookup_by_date

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
    # 滚动窗口只需近 20 个交易日; 裁剪日历避免单日推断每次对全历史重算
    lhb_hist_dates = [d for d in trading_dates_str if d <= trade_date][-40:]
    top_list_df = _try_ensure_historical_top_list(client, storage, lhb_hist_dates)
    if top_list_df is not None and len(top_list_df) > 0:
        from ...factors.lhb import build_lhb_lookup_by_date

        # 单日推断仅输出当日, 但滚动窗口需要完整历史交易日历
        lhb_lookup = build_lhb_lookup_by_date(
            top_list_df, factor_output_dates, calendar_dates=lhb_hist_dates
        )
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

    # ── 一致预期（按 report_date 增量）─────────────────────────
    consensus_today = pd.DataFrame()
    report_rc_df = loader.load_report_rc()
    if _has_announcement_gap(storage, report_rc_df, "report_rc", "report_date", trade_date):
        report_rc_df = _try_download_report_rc(client, storage, trade_date)
    if report_rc_df is not None and len(report_rc_df) > 0:
        from ...factors.consensus import build_consensus_lookup_by_date

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

    # ── 现金流质量（按 ann_date 增量）──────────────────────────
    cashflow_today = pd.DataFrame()
    cashflow_df = loader.load_cashflow(start_date=trade_date, end_date=trade_date)
    if _has_announcement_gap(storage, cashflow_df, "cashflow", "ann_date", trade_date):
        cashflow_df = _try_download_cashflow(client, storage, trade_date)
    if cashflow_df is not None and len(cashflow_df) > 0:
        from ...factors.cashflow_quality import build_cashflow_quality_lookup_by_date

        cashflow_lookup = build_cashflow_quality_lookup_by_date(cashflow_df, factor_output_dates)
        cur = cashflow_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            cashflow_today = cur
        logger.info(f"现金流质量因子: 已加载 ({len(cashflow_df)} 条)")
    else:
        missing_factors.append("cashflow（现金流质量）")
    cashflow_df = None
    cashflow_lookup = None
    gc.collect()

    # ── 一致预期修正（基于 report_rc，无额外下载）─────────────────
    consensus_revision_today = pd.DataFrame()
    report_rc_for_revision = loader.load_report_rc()
    if report_rc_for_revision is not None and len(report_rc_for_revision) > 0:
        from ...factors.consensus_revision import build_consensus_revision_lookup_by_date

        revision_lookup = build_consensus_revision_lookup_by_date(
            report_rc_for_revision,
            factor_output_dates,
            daily_data_lookup=daily_close_lookup,
        )
        cur = revision_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            consensus_revision_today = cur
        logger.info(f"一致预期修正因子: 已加载 ({len(report_rc_for_revision)} 条 report_rc)")
    else:
        missing_factors.append("consensus_revision（一致预期修正）")
    report_rc_for_revision = None
    revision_lookup = None
    gc.collect()

    # ── 风控公告类（质押，季分区 PIT 前向填充）─────────────────────
    pledge_today = pd.DataFrame()
    pledge_df = loader.load_pledge_stat(start_date, end_date)
    if pledge_df is not None and len(pledge_df) > 0:
        from ...factors.risk.announcement_lookup import build_pledge_lookup_by_date

        pledge_lookup = build_pledge_lookup_by_date(pledge_df, factor_output_dates)
        cur = pledge_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            pledge_today = cur
        logger.info(f"质押公告因子: 已加载 ({len(pledge_df)} 条原始记录)")
    else:
        missing_factors.append("pledge_stat（质押）")
    pledge_df = None
    pledge_lookup = None
    gc.collect()

    # ── 风控公告类（限售解禁，年分区 PIT 按公告日）──────────────────
    share_float_today = pd.DataFrame()
    share_float_df = loader.load_share_float(start_date, end_date)
    if share_float_df is not None and len(share_float_df) > 0:
        from ...factors.risk.announcement_lookup import build_share_float_lookup_by_date

        share_float_lookup = build_share_float_lookup_by_date(share_float_df, factor_output_dates)
        cur = share_float_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            share_float_today = cur
        logger.info(f"限售解禁因子: 已加载 ({len(share_float_df)} 条原始记录)")
    else:
        missing_factors.append("share_float（限售解禁）")
    share_float_df = None
    share_float_lookup = None
    gc.collect()

    # ── 风控公告类（大宗交易，日分区近 10 交易日折价聚合）────────────
    block_trade_today = pd.DataFrame()
    block_trade_df = loader.load_block_trade(start_date, end_date)
    if block_trade_df is not None and len(block_trade_df) > 0:
        from ...factors.risk.announcement_lookup import build_block_trade_lookup_by_date

        block_trade_lookup = build_block_trade_lookup_by_date(
            block_trade_df, factor_output_dates, close_lookup=block_close_lookup
        )
        cur = block_trade_lookup.get(trade_date)
        if cur is not None and len(cur) > 0:
            block_trade_today = cur
        logger.info(f"大宗交易因子: 已加载 ({len(block_trade_df)} 条原始记录)")
    else:
        missing_factors.append("block_trade（大宗交易）")
    block_trade_df = None
    block_trade_lookup = None
    gc.collect()

    # ── 汇总报告 ────────────────────────────────────────────
    total = 15
    loaded = total - len(missing_factors)
    if missing_factors:
        logger.warning(
            f"因子数据覆盖: {loaded}/{total} 组已加载，"
            f"缺失: {', '.join(missing_factors)}\n"
            f"  如需补全请运行: python scripts/download_raw.py --download <数据类型>"
        )
    else:
        logger.info(f"因子数据覆盖: {total}/{total} 组全部加载")

    return (
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
    )

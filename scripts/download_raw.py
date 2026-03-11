#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载原始数据脚本（仅下载raw层）

功能：
- 仅负责从TuShare/AKShare拉取原始数据并保存到raw层
- 不触发clean或feature的构建
- 支持force参数强制重新下载已存在的数据
- 支持--download参数选择下载特定数据集

数据集：
  基础数据（默认）：trade_cal, stock_basic
  日线数据（默认）：daily, daily_basic, adj_factor, suspend, stk_limit, moneyflow
  另类数据（需指定）：
    fina_indicator   - 财务指标（Tushare，逐股下载）
    margin_detail    - 融资融券明细（Tushare，按日分区）
    stk_holdernumber - 股东人数（Tushare，逐股下载）
    forecast         - 业绩预告（Tushare，逐股下载）
    express          - 业绩快报（Tushare，逐股下载）
    hot_rank         - 东财人气榜（AKShare，逐股下载）

使用示例：
    # 默认：下载基础数据 + 日线数据
    python scripts/download_raw.py

    # 下载特定另类数据
    python scripts/download_raw.py --download fina_indicator margin_detail

    # 下载全部另类数据
    python scripts/download_raw.py --download all_alt
"""

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient

if TYPE_CHECKING:
    import pandas as pd

# 所有另类数据集名称
ALT_DATASETS = [
    "fina_indicator", "margin_detail", "stk_holdernumber",
    "forecast", "express", "hot_rank",
]


def download_basic_data(
    client: TushareClient,
    storage: Storage,
    start_date: str,
    end_date: str,
    force: bool = False
) -> "pd.DataFrame":
    """下载基础数据（trade_cal和stock_basic）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
        
    Returns:
        交易日历DataFrame
    """
    # 1. 下载交易日历
    logger.info("检查交易日历...")
    if not force and storage.check_basic_data_freshness("trade_cal", end_date):
        logger.info("交易日历数据已是最新，跳过下载")
        trade_cal = storage.load_raw("trade_cal")
    else:
        logger.info(f"下载交易日历（{start_date}-{end_date}）...")
        trade_cal = client.get_trade_cal(
            start_date=start_date,
            end_date=end_date,
            exchange="SSE"
        )
        storage.save_raw(trade_cal, "trade_cal", is_force=True)
        logger.info(f"交易日历下载完成: {len(trade_cal)} 条记录")
    
    # 2. 下载股票基本信息
    logger.info("检查股票基本信息...")
    if not force and storage.check_basic_data_freshness("stock_basic", end_date):
        logger.info("股票基本信息已存在，跳过下载")
    else:
        logger.info("下载股票基本信息...")
        stock_basic = client.get_stock_basic(list_status="L")
        storage.save_raw(stock_basic, "stock_basic", is_force=True)
        logger.info(f"股票基本信息下载完成: {len(stock_basic)} 条记录")
    
    return trade_cal


def download_daily_data(
    client: TushareClient,
    storage: Storage,
    trade_cal: "pd.DataFrame",
    start_date: str,
    end_date: str,
    force: bool = False
) -> None:
    """下载日线数据（按日期分区）
    
    Args:
        client: TushareClient实例
        storage: Storage实例
        trade_cal: 交易日历DataFrame
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        force: 是否强制重新下载
    """
    import pandas as pd
    
    logger.info(f"下载日线数据（{start_date}-{end_date}）...")
    logger.info("使用按日分区存储模式")
    
    # 获取交易日列表
    trading_dates = trade_cal[
        (trade_cal['cal_date'] >= start_date) &
        (trade_cal['cal_date'] <= end_date) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].tolist()
    
    logger.info(f"共 {len(trading_dates)} 个交易日需要下载")
    
    total_daily = 0
    total_basic = 0
    skip_count = 0
    
    for i, trade_date in enumerate(trading_dates, 1):
        logger.info(f"[{i}/{len(trading_dates)}] ({i/len(trading_dates):.1%}) 处理 {trade_date}...")
        
        try:
            # 下载日线行情
            if not force and storage.is_data_exists("raw", "daily", trade_date):
                logger.info(f"  日线: 文件已存在，跳过下载")
                skip_count += 1
            else:
                daily_data = client.get_daily(trade_date=trade_date)
                if len(daily_data) > 0:
                    storage.save_raw_by_date(daily_data, "daily", trade_date)
                    total_daily += len(daily_data)
                    logger.info(f"  日线: 已保存 {len(daily_data)} 条记录")
            
            # 下载每日指标
            if not force and storage.is_data_exists("raw", "daily_basic", trade_date):
                logger.info(f"  指标: 文件已存在，跳过下载")
            else:
                daily_basic = client.get_daily_basic(trade_date=trade_date)
                if len(daily_basic) > 0:
                    storage.save_raw_by_date(daily_basic, "daily_basic", trade_date)
                    total_basic += len(daily_basic)
                    logger.info(f"  指标: 已保存 {len(daily_basic)} 条记录")

            # 下载复权因子
            if not force and storage.is_data_exists("raw", "adj_factor", trade_date):
                logger.info(f"  复权因子: 文件已存在，跳过下载")
            else:
                adj_factor = client.get_adj_factor(trade_date=trade_date)
                if len(adj_factor) > 0:
                    storage.save_raw_by_date(adj_factor, "adj_factor", trade_date)
                    logger.info(f"  复权因子: 已保存 {len(adj_factor)} 条记录")
                    
            # 下载停复牌信息
            if not force and storage.is_data_exists("raw", "suspend", trade_date):
                logger.info(f"  停复牌: 文件已存在，跳过下载")
            else:
                suspend = client.get_suspend_d(trade_date=trade_date)
                if len(suspend) > 0:
                    storage.save_raw_by_date(suspend, "suspend", trade_date)
                    logger.info(f"  停复牌: 已保存 {len(suspend)} 条记录")
                    
            # 下载涨跌停信息
            if not force and storage.is_data_exists("raw", "stk_limit", trade_date):
                logger.info(f"  涨跌停: 文件已存在，跳过下载")
            else:
                limit_up_down = client.get_stk_limit(trade_date=trade_date)
                if len(limit_up_down) > 0:
                    storage.save_raw_by_date(limit_up_down, "stk_limit", trade_date)
                    logger.info(f"  涨跌停: 已保存 {len(limit_up_down)} 条记录")
            
            # 下载资金流向
            if not force and storage.is_data_exists("raw", "moneyflow", trade_date):
                logger.info(f"  资金流向: 文件已存在，跳过下载")
            else:
                moneyflow = client.get_moneyflow(trade_date=trade_date)
                if len(moneyflow) > 0:
                    storage.save_raw_by_date(moneyflow, "moneyflow", trade_date)
                    logger.info(f"  资金流向: 已保存 {len(moneyflow)} 条记录")
                else:
                    logger.error(f"  资金流向数据缺失（moneyflow 为强制依赖项）")
                    
        except Exception as e:
            logger.error(f"下载 {trade_date} 数据失败: {str(e)}")
            continue
    
    logger.info("=" * 60)
    logger.info("日线数据下载完成")
    logger.info("=" * 60)
    logger.info(f"新下载日线行情: {total_daily} 条记录")
    logger.info(f"新下载每日指标: {total_basic} 条记录")
    logger.info(f"跳过已存在: {skip_count} 个交易日")


def download_per_stock_data(
    client: TushareClient,
    storage: Storage,
    dataset_name: str,
    api_name: str,
    stock_codes: List[str],
    fields: Optional[str] = None,
    dedup_cols: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """通用逐股下载（支持断点续传）

    适用于：fina_indicator, stk_holdernumber, forecast, express

    Args:
        client: TushareClient 实例
        storage: Storage 实例
        dataset_name: 数据集名称（用于 save_raw / load_raw）
        api_name: Tushare API 名称
        stock_codes: 待下载的全部股票代码列表
        fields: 返回字段（逗号分隔）
        dedup_cols: 去重列（不传则不做去重）
        force: 是否强制全量重下
    """
    # 断点续传：加载已有数据
    existing_codes: Set[str] = set()
    existing_df = None
    if not force:
        existing_df = storage.load_raw(dataset_name)
        if existing_df is not None and len(existing_df) > 0:
            existing_codes = set(existing_df["ts_code"].unique())
            logger.info(f"[{dataset_name}] 已有 {len(existing_codes)} 只股票数据（断点续传）")

    codes_to_download = [c for c in stock_codes if c not in existing_codes]
    if not codes_to_download:
        logger.info(f"[{dataset_name}] 所有股票数据已存在，跳过。如需重下请加 --force")
        return

    logger.info(f"[{dataset_name}] 待下载: {len(codes_to_download)} 只股票")

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
            logger.warning(f"[{dataset_name}] {ts_code} 失败: {e}")

        if i % 100 == 0 or i == len(codes_to_download):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            remaining = (len(codes_to_download) - i) / speed if speed > 0 else 0
            logger.info(
                f"[{dataset_name}] [{i}/{len(codes_to_download)}] "
                f"成功={success} 空={empty} 失败={errors} "
                f"速度={speed:.0f}只/秒 剩余≈{remaining/60:.1f}分钟"
            )

        # 每 500 只保存中间结果
        if i % 500 == 0 and all_dfs:
            _save_merged(storage, dataset_name, all_dfs, existing_df, dedup_cols)

    # 最终保存
    if all_dfs:
        _save_merged(storage, dataset_name, all_dfs, existing_df, dedup_cols)

    elapsed_total = time.time() - t0
    logger.info(
        f"[{dataset_name}] 下载完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total/60:.1f}分钟"
    )


def _save_merged(storage, dataset_name, new_dfs, existing_df, dedup_cols):
    """合并新旧数据并保存"""
    result = pd.concat(new_dfs, ignore_index=True)
    if existing_df is not None and len(existing_df) > 0:
        result = pd.concat([existing_df, result], ignore_index=True)
    if dedup_cols:
        result = result.drop_duplicates(subset=dedup_cols, keep="last")
    storage.save_raw(result, dataset_name, is_force=True)
    logger.info(f"[{dataset_name}] 已保存: {len(result)} 条记录")


def download_margin_detail(
    client: TushareClient,
    storage: Storage,
    trade_cal: pd.DataFrame,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    """下载融资融券明细（按日分区，与 daily 同模式）"""
    trading_dates = trade_cal[
        (trade_cal["cal_date"] >= start_date)
        & (trade_cal["cal_date"] <= end_date)
        & (trade_cal["is_open"] == 1)
    ]["cal_date"].tolist()

    logger.info(f"[margin_detail] {len(trading_dates)} 个交易日")
    success = skip = errors = 0

    for i, td in enumerate(trading_dates, 1):
        try:
            if not force and storage.is_data_exists("raw", "margin_detail", td):
                skip += 1
                continue
            df = client.query("margin_detail", trade_date=td)
            if df is not None and len(df) > 0:
                storage.save_raw_by_date(df, "margin_detail", td)
                success += 1
            # 融资融券覆盖面较窄，部分日期无数据属正常
        except Exception as e:
            errors += 1
            logger.warning(f"[margin_detail] {td} 失败: {e}")

        if i % 100 == 0 or i == len(trading_dates):
            logger.info(
                f"[margin_detail] [{i}/{len(trading_dates)}] "
                f"新下载={success} 跳过={skip} 失败={errors}"
            )

    logger.info(f"[margin_detail] 完成: 新下载={success} 跳过={skip} 失败={errors}")


def download_hot_rank(
    storage: Storage,
    stock_codes: List[str],
    force: bool = False,
) -> None:
    """下载东财人气榜数据（AKShare）

    策略：
    1. 首次无数据 → 逐股调用 stock_hot_rank_detail_em 回填历史（慢，~3h）
    2. 已有数据 → 调用 stock_hot_rank_em 拉取当日全市场快照（快，<10s）
       然后 append 到已有数据
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("请安装 akshare: pip install akshare")
        return

    existing_df = None
    if not force:
        existing_df = storage.load_raw("hot_rank")

    has_history = existing_df is not None and len(existing_df) > 0

    if has_history and not force:
        # ── 增量模式：批量拉取当日快照 ────────────────────────
        _download_hot_rank_daily_snapshot(ak, storage, existing_df)
    else:
        # ── 全量回填：逐股拉取历史 ────────────────────────────
        _download_hot_rank_backfill(ak, storage, stock_codes, existing_df, force)


def _download_hot_rank_daily_snapshot(ak, storage, existing_df):
    """增量模式：一次 API 调用获取全市场当日人气排名"""
    logger.info("[hot_rank] 增量模式：拉取当日全市场快照...")
    try:
        df = ak.stock_hot_rank_em()
        if df is None or len(df) == 0:
            logger.warning("[hot_rank] 当日快照为空（非交易时段或节假日）")
            return

        # 标准化列名（AKShare 返回中文列名）
        col_map = {}
        for col in df.columns:
            if "代码" in col:
                col_map[col] = "symbol"
            elif "排名" in col or "序号" in col:
                col_map[col] = "hot_rank"
        df = df.rename(columns=col_map)

        if "symbol" not in df.columns or "hot_rank" not in df.columns:
            logger.warning(f"[hot_rank] 快照列名不匹配: {df.columns.tolist()}")
            return

        # 转换代码格式: 000001 -> 000001.SZ / 600000 -> 600000.SH
        def to_ts_code(sym):
            s = str(sym).zfill(6)
            if s.startswith(("6", "9")):
                return f"{s}.SH"
            return f"{s}.SZ"

        today = pd.Timestamp.now().strftime("%Y%m%d")
        df["ts_code"] = df["symbol"].apply(to_ts_code)
        df["trade_date"] = today
        df = df[["ts_code", "trade_date", "hot_rank"]]

        # 合并去重
        result = pd.concat([existing_df, df], ignore_index=True)
        result = result.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        storage.save_raw(result, "hot_rank", is_force=True)
        logger.info(f"[hot_rank] 增量完成: 新增 {len(df)} 条, 总计 {len(result)} 条")

    except Exception as e:
        logger.error(f"[hot_rank] 快照下载失败: {e}")


def _download_hot_rank_backfill(ak, storage, stock_codes, existing_df, force):
    """全量回填：逐股拉取历史人气排名（首次使用，耗时较长）"""
    existing_codes: Set[str] = set()
    if not force and existing_df is not None and len(existing_df) > 0:
        existing_codes = set(existing_df["ts_code"].unique())
        logger.info(f"[hot_rank] 已有 {len(existing_codes)} 只股票数据（断点续传）")

    codes_to_download = [c for c in stock_codes if c not in existing_codes]
    if not codes_to_download:
        logger.info("[hot_rank] 所有股票数据已存在，跳过")
        return

    logger.info(f"[hot_rank] 全量回填: 待下载 {len(codes_to_download)} 只股票（预计 3-4 小时）")

    all_dfs: List[pd.DataFrame] = []
    success = empty = errors = 0
    t0 = time.time()

    for i, ts_code in enumerate(codes_to_download, 1):
        try:
            # 转换代码格式: 000001.SZ -> SZ000001
            code_num = ts_code.split(".")[0]
            exchange = ts_code.split(".")[1]
            symbol = f"{exchange}{code_num}"

            df = ak.stock_hot_rank_detail_em(symbol=symbol)
            if df is not None and len(df) > 0:
                df = df.rename(columns={"时间": "trade_date", "排名": "hot_rank"})
                df["ts_code"] = ts_code
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                keep_cols = [c for c in ["ts_code", "trade_date", "hot_rank"] if c in df.columns]
                df = df[keep_cols]
                all_dfs.append(df)
                success += 1
            else:
                empty += 1
        except Exception as e:
            errors += 1
            if i <= 10 or i % 500 == 0:
                logger.debug(f"[hot_rank] {ts_code} 无数据: {e}")

        if i % 200 == 0 or i == len(codes_to_download):
            elapsed = time.time() - t0
            speed = i / elapsed if elapsed > 0 else 0
            remaining = (len(codes_to_download) - i) / speed if speed > 0 else 0
            logger.info(
                f"[hot_rank] [{i}/{len(codes_to_download)}] "
                f"成功={success} 空={empty} 失败={errors} "
                f"剩余≈{remaining/60:.1f}分钟"
            )

        if i % 500 == 0 and all_dfs:
            _save_merged(storage, "hot_rank", all_dfs, existing_df,
                         dedup_cols=["ts_code", "trade_date"])

        time.sleep(0.1)

    if all_dfs:
        _save_merged(storage, "hot_rank", all_dfs, existing_df,
                     dedup_cols=["ts_code", "trade_date"])

    elapsed_total = time.time() - t0
    logger.info(
        f"[hot_rank] 回填完成: 成功={success} 空={empty} 失败={errors} "
        f"耗时={elapsed_total/60:.1f}分钟"
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载原始数据（仅raw层，不触发clean/feature构建）"
    )
    parser.add_argument(
        "--start-date",
        default="20200101",
        help="开始日期，格式YYYYMMDD（默认：20200101）"
    )
    parser.add_argument(
        "--end-date",
        default="20251231",
        help="结束日期，格式YYYYMMDD（默认：20251231）"
    )
    parser.add_argument(
        "--only-basic",
        action="store_true",
        help="仅下载基础数据（trade_cal和stock_basic）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使文件已存在"
    )
    parser.add_argument(
        "--download",
        nargs="*",
        default=None,
        help="指定下载的另类数据集，可多选。"
             "可选值: fina_indicator, margin_detail, stk_holdernumber, "
             "forecast, express, hot_rank, all_alt。"
             "不指定时仅下载基础+日线数据"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="（兼容旧参数）等效于不加 --force 的默认断点续传行为"
    )

    args = parser.parse_args()
    
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("开始下载原始数据（raw层）")
    logger.info("=" * 60)
    logger.info(f"日期范围: {args.start_date} - {args.end_date}")
    logger.info(f"仅下载基础数据: {'是' if args.only_basic else '否'}")
    logger.info(f"强制重新下载: {'是' if args.force else '否'}")
    logger.info(f"另类数据集: {args.download or '无'}")
    logger.info("=" * 60)
    
    try:
        # 初始化客户端和存储
        client = TushareClient()
        storage = Storage()
        
        # 下载基础数据
        trade_cal = download_basic_data(
            client, storage,
            args.start_date, args.end_date,
            force=args.force
        )
        
        if args.only_basic:
            logger.info("=" * 60)
            logger.info("仅下载基础数据，操作完成！")
            logger.info(f"数据保存位置: {storage.root_path}/raw")
            logger.info("=" * 60)
            sys.exit(0)
        
        # 下载日线数据
        download_daily_data(
            client, storage, trade_cal,
            args.start_date, args.end_date,
            force=args.force
        )
        
        # ── 另类数据下载 ──────────────────────────────────────────
        download_set = set(args.download) if args.download else set()
        if "all_alt" in download_set:
            download_set = set(ALT_DATASETS)

        if download_set:
            # 加载股票列表（逐股下载需要）
            stock_basic = storage.load_raw("stock_basic")
            if stock_basic is None:
                logger.error("未找到 stock_basic 数据，请先运行默认下载")
                sys.exit(1)
            stock_codes = sorted(stock_basic["ts_code"].unique().tolist())

            # 财务指标
            if "fina_indicator" in download_set:
                download_per_stock_data(
                    client, storage,
                    dataset_name="fina_indicator",
                    api_name="fina_indicator",
                    stock_codes=stock_codes,
                    fields="ts_code,ann_date,end_date,roe_waa,or_yoy,netprofit_yoy,debt_to_assets,q_gr_yoy",
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    force=args.force,
                )

            # 融资融券明细（按日分区）
            if "margin_detail" in download_set:
                download_margin_detail(
                    client, storage, trade_cal,
                    args.start_date, args.end_date,
                    force=args.force,
                )

            # 股东人数
            if "stk_holdernumber" in download_set:
                download_per_stock_data(
                    client, storage,
                    dataset_name="stk_holdernumber",
                    api_name="stk_holdernumber",
                    stock_codes=stock_codes,
                    dedup_cols=["ts_code", "end_date"],
                    force=args.force,
                )

            # 业绩预告
            if "forecast" in download_set:
                download_per_stock_data(
                    client, storage,
                    dataset_name="forecast",
                    api_name="forecast",
                    stock_codes=stock_codes,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    force=args.force,
                )

            # 业绩快报
            if "express" in download_set:
                download_per_stock_data(
                    client, storage,
                    dataset_name="express",
                    api_name="express",
                    stock_codes=stock_codes,
                    dedup_cols=["ts_code", "end_date", "ann_date"],
                    force=args.force,
                )

            # 东财人气榜（AKShare）
            if "hot_rank" in download_set:
                download_hot_rank(storage, stock_codes, force=args.force)

        logger.info("=" * 60)
        logger.info("原始数据下载完成！")
        logger.info(f"数据保存位置: {storage.root_path}/raw")
        logger.info("=" * 60)
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error("=" * 60)
        logger.error("数据下载失败")
        logger.error("=" * 60)
        logger.error(str(e))
        logger.error("")
        logger.error("请按以下步骤配置TuShare token:")
        logger.error("1. 访问 https://tushare.pro/register 注册账号")
        logger.error("2. 获取token")
        logger.error("3. 创建 .env 文件（参考 .env.example）")
        logger.error("4. 在 .env 文件中设置: TS_TOKEN=your_token_here")
        logger.error("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"数据下载过程中出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

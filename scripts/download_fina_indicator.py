#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载财务指标数据（fina_indicator）

功能：
- 从 TuShare fina_indicator API 下载全市场季度财务指标
- 保存为单个文件 data/raw/fina_indicator.parquet
- 支持 --resume 断点续传（跳过已下载的股票）
- 需要 2000 积分权限

使用示例：
    # 首次下载（全量）
    python scripts/download_fina_indicator.py

    # 断点续传（中断后继续）
    python scripts/download_fina_indicator.py --resume

    # 强制重新下载
    python scripts/download_fina_indicator.py --force
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient
from src.lazybull.common.config import get_config


def main():
    parser = argparse.ArgumentParser(description="下载财务指标数据（fina_indicator）")
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载全部数据（忽略已有文件）"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续传（跳过已下载的股票，默认行为）"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data",
        help="数据根目录，默认 ./data"
    )
    args = parser.parse_args()

    setup_logger()
    get_config()  # 确保配置已加载

    storage = Storage(root_path=args.data_root)
    client = TushareClient()

    # 加载股票列表
    stock_basic = storage.load_raw("stock_basic")
    if stock_basic is None:
        logger.error("未找到 stock_basic 数据，请先运行 python scripts/update_basic_data.py")
        sys.exit(1)

    all_codes = sorted(stock_basic['ts_code'].unique().tolist())
    logger.info(f"全市场股票数量: {len(all_codes)}")

    # 检查已有数据（断点续传）
    existing_codes = set()
    existing_df = None
    if not args.force:
        existing_df = storage.load_raw("fina_indicator")
        if existing_df is not None:
            existing_codes = set(existing_df['ts_code'].unique())
            logger.info(f"已有 {len(existing_codes)} 只股票的数据（断点续传模式）")

    codes_to_download = [c for c in all_codes if c not in existing_codes]

    if len(codes_to_download) == 0:
        logger.info("所有股票数据已下载完毕，无需操作。如需重新下载请使用 --force")
        return

    logger.info(f"待下载: {len(codes_to_download)} 只股票")
    estimated_minutes = len(codes_to_download) / 200  # ~200 req/min
    logger.info(f"预计耗时: {estimated_minutes:.0f} 分钟")

    # 逐股下载
    all_dfs = []
    success_count = 0
    empty_count = 0
    error_count = 0
    timer_start = time.time()

    for i, ts_code in enumerate(codes_to_download, 1):
        try:
            df = client.get_fina_indicator(ts_code=ts_code)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                success_count += 1
            else:
                empty_count += 1
        except Exception as e:
            error_count += 1
            logger.warning(f"下载 {ts_code} 失败: {e}")
            continue

        # 进度显示（每 100 只或最后一只）
        if i % 100 == 0 or i == len(codes_to_download):
            elapsed = time.time() - timer_start
            speed = i / elapsed if elapsed > 0 else 0
            remaining = (len(codes_to_download) - i) / speed if speed > 0 else 0
            logger.info(
                f"[{i}/{len(codes_to_download)}] "
                f"成功={success_count} 空={empty_count} 失败={error_count} "
                f"速度={speed:.0f}只/秒 "
                f"剩余≈{remaining / 60:.1f}分钟"
            )

        # 每 500 只保存一次中间结果（防止中断丢失）
        if i % 500 == 0 and all_dfs:
            _save_intermediate(storage, all_dfs, existing_df)

    # 最终保存
    if all_dfs:
        _save_intermediate(storage, all_dfs, existing_df)

    elapsed_total = time.time() - timer_start
    logger.info(
        f"\n下载完成！"
        f"\n  成功: {success_count}"
        f"\n  空数据: {empty_count}"
        f"\n  失败: {error_count}"
        f"\n  总耗时: {elapsed_total / 60:.1f} 分钟"
    )


def _save_intermediate(storage, new_dfs, existing_df):
    """保存中间结果（合并已有数据）"""
    result = pd.concat(new_dfs, ignore_index=True)
    if existing_df is not None:
        result = pd.concat([existing_df, result], ignore_index=True)
    # 去重：同一股票同一报告期只保留最新
    result = result.drop_duplicates(subset=['ts_code', 'end_date', 'ann_date'], keep='last')
    storage.save_raw(result, "fina_indicator", is_force=True)
    logger.info(f"已保存中间结果: {len(result)} 条记录")


if __name__ == "__main__":
    main()

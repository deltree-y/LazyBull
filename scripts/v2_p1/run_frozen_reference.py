# -*- coding: utf-8 -*-
"""v2 P1 单元 3：冻结参照重放（旧代码 + 数据态 B 全量重放 cs_train 等价物）。

机制：直接驱动 `features/pipeline.py::build_features_data`——
- 写侧 = ``Storage(root_path=<scratch>)``（默认 temp/p1_frozen_reference；
  分区落 ``<scratch>/features/cs_train/YYYYMMDD.parquet``，原子写沿 `_save_data` 先例；
  scratch 下 `is_feature_exists` 对未建分区恒 False ⇒ 无需 force；重跑时已建分区经
  `_check_features_schema` 校验跳过 ⇒ 断点续跑）；
- 读侧 = 生产 root 的 ``DataLoader(Storage())``（只读 raw/clean，本脚本不写生产任何分区）；
- builder = 冻结 v2 口径（min_list_days=365, horizon=20, horizons=[5,10,20],
  require_label=True, label_filter_mode="single"）；
- flags = 383 生产配置（单元 2 标定全集但 ht/rp/tfh 关闭——冻结参照复刻的是
  cs_train 383 列，物化四族不进参照）。

分块策略（--chunk-years，默认 5）：单调用全区间会把 15 年 clean 日线
（~2,000 万行）一次性载入，内存风险高；且生产 cs_train 本就是分批构建。N 年分块
起点钳制为 max(全局起点, Y−2 年) ⇒ 每个新建日有效历史 ≥ 19 个月（MA250/EMA 长尾
全收敛，与单调用全区间在 atol=1e-6 内一致；2012 首年除外，与任何构建同样受限于
数据起点），同时把查找表/风控预计算的固定成本摊到 4 个分块而非 15 个。
--chunk-years 0 = 单调用（配合 --warmup-start 给探测窗口预热段， warmup 段分区
同样落 scratch、属参照合法覆盖）。

用法：
    python scripts/v2_p1/run_frozen_reference.py [--start-date 20120104] [--end-date 20260702]
        [--out-dir temp/p1_frozen_reference] [--jobs -1] [--serial | --parallel]
        [--chunk-years 5] [--warmup-start YYYYMMDD]

构建模式：默认**串行**（对齐生产——`scripts/build_clean_features.py --parallel`
help 明示"默认串行"，cs_train 即串行产物；且串行/并行逐值一致性无生产实证）。
`--parallel` 启用并行并经 `builder._get_trading_dates` 预热交易日索引缓存——
并行路径自身不建该缓存（`_build_features_parallel` 读 `builder._trading_date_index or {}`，
空缓存时静态函数逐日返回 None：口径 B 首轮探测实证 262 天"0 成功 0 失败"零落盘，
属旧代码留白，本脚本薄入口侧补偿、不改旧文件）。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.features.builder import FeatureBuilder  # noqa: E402
from src.lazybull.features.pipeline import build_features_data  # noqa: E402

#: 383 生产开关集（单元 2 标定全集，ht/rp/tfh 三族关闭——参照复刻 383 列 cs_train）
PRODUCTION_FLAGS_383: dict[str, bool] = {
    "apply_industry_neutralization": True,
    "apply_size_neutralization": True,
    "enable_fundamental": True,
    "enable_alt": True,
    "enable_margin": True,
    "enable_cyq": True,
    "enable_fund": True,
    "enable_express": True,
    "enable_north": True,
    "enable_lhb": True,
    "enable_consensus": True,
    "enable_cashflow_quality": True,
    "enable_consensus_revision": True,
    "enable_dividend_policy": True,
    "enable_announcement_risk": True,
    "enable_holdertrade": False,
    "enable_repurchase": False,
    "enable_top10fh": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 冻结参照重放（旧代码 + 数据态 B）")
    parser.add_argument("--start-date", default="20120104", help="构建起点 YYYYMMDD")
    parser.add_argument("--end-date", default="20260702", help="构建终点 YYYYMMDD")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "temp" / "p1_frozen_reference"),
        help="scratch 写侧根（分区落 <out-dir>/features/cs_train/）",
    )
    parser.add_argument("--jobs", type=int, default=-1, help="并行 worker 数（-1 = 全核）")
    parser.add_argument(
        "--serial",
        action="store_true",
        default=True,
        help="串行构建（默认，对齐生产；--parallel 关闭）",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用多进程并行构建（需预热交易日缓存；生产默认串行，冻结参照以串行为准）",
    )
    parser.add_argument(
        "--chunk-years",
        type=int,
        default=5,
        help="分块年数（默认 5，起点钳制 Y−2 年；1 = 年度分块；0 = 单调用）",
    )
    parser.add_argument(
        "--warmup-start",
        default=None,
        help="单调用模式的管线起点前移（预热段分区同样落 scratch；仅 --chunk-years 0 生效）",
    )
    return parser.parse_args()


def _iter_chunks(start_date: str, end_date: str, chunk_years: int) -> list[tuple[str, str]]:
    """N 年分块：覆盖段 [Y, Y+N)，管线起点钳制 = max(全局起点, Y−2 年)。

    起点前移 2 年 ⇒ 每个新建日有效历史 ≥ 19 个月（EMA/MA250 长尾收敛）；
    分块内已建分区经管线跳过逻辑自然只建覆盖段年份（断点续跑）。
    """
    start_year, end_year = int(start_date[:4]), int(end_date[:4])
    chunks = []
    year = start_year
    while year <= end_year:
        span_end_year = min(year + chunk_years - 1, end_year)
        chunk_start = max(start_date, f"{year - 2}0101")
        chunk_end = min(end_date, f"{span_end_year}1231")
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        year = span_end_year + 1
    return chunks


def _run_chunk(
    storage: Storage,
    loader: DataLoader,
    builder: FeatureBuilder,
    chunk_start: str,
    chunk_end: str,
    shenwan_industry,
    use_parallel: bool,
    jobs: int,
) -> None:
    """单分块构建（force=False：scratch 已建分区经 schema 校验跳过）。"""
    build_features_data(
        storage,
        loader,
        builder,
        chunk_start,
        chunk_end,
        force=False,
        shenwan_industry=shenwan_industry,
        use_parallel=use_parallel,
        parallel_jobs=jobs,
        **PRODUCTION_FLAGS_383,
    )


def main() -> int:
    args = _parse_args()
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"v2_p1_frozen_ref_{datetime.now():%Y%m%d}.log"
    logger.add(log_path, encoding="utf-8")

    out_dir = Path(args.out_dir)
    storage = Storage(root_path=str(out_dir))  # 写侧 scratch
    loader = DataLoader(Storage())  # 读侧生产 root（只读）
    builder = FeatureBuilder(
        min_list_days=365,
        horizon=20,
        horizons=[5, 10, 20],
        require_label=True,
        label_filter_mode="single",
    )
    shenwan_industry = loader.load_shenwan_industry()
    if shenwan_industry is None:
        logger.error("缺少申万行业数据（shenwan_industry），行业中性化无法构建")
        return 1

    use_parallel = args.parallel and not args.serial
    if use_parallel:
        # 并行路径自身不建交易日索引缓存（`_build_features_parallel` 读
        # `builder._trading_date_index or {}`，缓存为空时静态函数逐日返回 None——
        # 口径 B 探测实证：262 天"0 成功 0 失败"零落盘；生产默认串行从未踩到）。
        # 经 builder._get_trading_dates 预热补齐（不改旧文件，薄入口侧补偿）。
        trade_cal = loader.load_clean_trade_cal()
        if trade_cal is None:
            logger.error("缺少 clean 层 trade_cal 数据")
            return 1
        builder._get_trading_dates(trade_cal)
        logger.info(f"并行模式：交易日缓存已预热（{len(builder._trading_dates_cache)} 天）")

    if args.chunk_years == 0:
        chunk_start = args.warmup_start or args.start_date
        chunks = [(chunk_start, args.end_date)]
        if args.warmup_start:
            logger.info(
                f"单调用 + 预热段: 管线起点 {chunk_start}（比对窗口 {args.start_date}~{args.end_date}）"
            )
    else:
        chunks = _iter_chunks(args.start_date, args.end_date, args.chunk_years)
    logger.info(
        f"冻结参照重放启动: 区间 {args.start_date}~{args.end_date}，分块 {len(chunks)} 个，"
        f"jobs={args.jobs}，use_parallel={use_parallel}，scratch={out_dir}，日志 {log_path}"
    )

    started = time.time()
    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        chunk_ts = time.time()
        logger.info(f"===== 分块 [{idx}/{len(chunks)}] {chunk_start}~{chunk_end} =====")
        try:
            _run_chunk(
                storage,
                loader,
                builder,
                chunk_start,
                chunk_end,
                shenwan_industry,
                use_parallel=use_parallel,
                jobs=args.jobs,
            )
        except Exception as exc:
            logger.error(f"分块 {chunk_start}~{chunk_end} 构建失败: {exc}")
            return 1
        elapsed = time.time() - chunk_ts
        logger.info(f"分块完成，耗时 {elapsed/60:.1f} min")

    total = time.time() - started
    partition_dir = out_dir / "features" / "cs_train"
    built = len(list(partition_dir.glob("*.parquet"))) if partition_dir.exists() else 0
    logger.info(f"重放完成: scratch 分区累计 {built} 个，总耗时 {total/60:.1f} min")
    print(f"完成: 分区 {built} 个，耗时 {total/60:.1f} min，日志 {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

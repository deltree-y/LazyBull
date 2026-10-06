# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 全量回填驱动（薄入口；统一分块约定 + keep_dates 续传防冲突）。

构造约束（主会话拍板，单元 3 报告 §8 防复发清单①）：
- 分块 = `_iter_chunks(start, end, 5)`（与冻结参照严格一致；直接 import 复用
  `scripts/v2_p1/run_frozen_reference.py`）；
- **EMA 长尾陷阱**：相邻分块 2 年重叠（2015~2016、2020~2021），同一日被两块以
  不同预热窗重建时长记忆列尾差 >1e-6，重复 append 必触发指纹冲突 ⇒ sink 只落
  「未写日」（keep_dates = 块内覆盖日 − manifest 已写日；管线仍 force=True 全量
  构建，重叠日重建后 sink 丢弃，与参照同值）；
- 已写判定（labels 不参与）：热区（>20250630）= 该日 8 族 panel 分区全登记；
  冷区（≤20250630）= 该月 8 族 panel_archive 分区全登记（缺一 ⇒ 整月未写，
  重跑时已登记族指纹一致 no-op）；
- 写窗口：panel 只写 ≥20120104 的日；串行构建（并行 `_trading_date_index` 缓存
  留白见单元 3 报告 §5-③，`--serial` 仅作显式标记）。

用法：
    python scripts/v2_p1/backfill_panel.py [--start-date 20120104] [--end-date 20260702]
        [--data-root data] [--chunk-years 5] [--serial]
日志：logs/v2_p1_backfill_panel_<date>.log。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v2_p1"))

from run_frozen_reference import _iter_chunks  # noqa: E402  # 分块约定单一来源

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import ARCHIVE_HOT_BOUNDARY, PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest  # noqa: E402

_GROUPS = tuple(PANEL_GROUPS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 panel 全量回填驱动")
    parser.add_argument("--start-date", default="20120104", help="回填起点 YYYYMMDD")
    parser.add_argument("--end-date", default="20260702", help="回填终点 YYYYMMDD")
    parser.add_argument(
        "--data-root", default=str(ROOT / "data"), help="v2 store 数据根（冒烟传 temp/ 下路径）"
    )
    parser.add_argument("--chunk-years", type=int, default=5, help="分块年数（默认 5，与参照一致）")
    parser.add_argument("--serial", action="store_true", help="显式标记串行（构建固定串行）")
    args = parser.parse_args()
    # fail-fast（R3-08）：分块参数必须 ≥1，否则 _iter_chunks 不推进/倒退 ⇒ 无限循环
    if args.chunk_years < 1:
        parser.error(
            f"--chunk-years 必须 ≥1（当前 {args.chunk_years}）：0 = 单调用语义，"
            "仅 run_frozen_reference 支持，本脚本不提供"
        )
    return args


def _written_days(store: PanelDataStore, coverage: list[str]) -> set[str]:
    """manifest 口径的已写日：热区日 8 族全登记 / 冷区月 8 族全登记 **且日内覆盖完整**
    （labels 不计）。

    冷区日级精度（D-09 修复）：月文件已登记不足以证明某日已写（分块末标签不成熟日
    曾致月文件缺日——单元 4 缺口事件）；冷区须核 core 族分区内的实际日期集合。
    """
    partitions = store.manifest.snapshot()["partitions"]
    hot_days: dict[str, set[str]] = {}
    cold_months: dict[str, set[str]] = {}
    for rel_path in partitions:
        segs = rel_path.split("/")
        if len(segs) != 3:
            continue
        zone, key, filename = segs
        group = filename.removesuffix(".parquet")
        if zone == "panel":
            hot_days.setdefault(key, set()).add(group)
        elif zone == "panel_archive":
            cold_months.setdefault(key, set()).add(group)
    all_groups = set(_GROUPS)
    cold_day_cache: dict[str, set[str]] = {}

    def _cold_days(month: str) -> set[str]:
        if month not in cold_day_cache:
            core = store.archive_dir / month / "core.parquet"
            if core.exists():
                cold_day_cache[month] = set(
                    pd.read_parquet(core, columns=["trade_date"])["trade_date"].astype(str).unique()
                )
            else:
                cold_day_cache[month] = set()
        return cold_day_cache[month]

    written: set[str] = set()

    def _hot_files_exist(date: str) -> bool:
        """热区已登记日核对 8 族文件实际存在（登记 ≠ 盘上文件，D-10 镜像教训）。"""
        day_dir = store.panel_dir / date
        missing = sorted(g for g in all_groups if not (day_dir / f"{g}.parquet").exists())
        if missing:
            logger.warning(f"热区 {date} 已登记 8 族但缺文件 {missing}，按未写处理（将重建）")
            return False
        return True

    for date in coverage:
        if date > ARCHIVE_HOT_BOUNDARY:
            if hot_days.get(date, set()) >= all_groups and _hot_files_exist(date):
                written.add(date)
        else:
            month = f"{date[:4]}-{date[4:6]}"
            if cold_months.get(month, set()) >= all_groups and date in _cold_days(month):
                written.add(date)
    return written


def _count_partitions(store: PanelDataStore) -> dict[str, int]:
    """manifest 分区计数（热区/冷区/labels 三分）。"""
    snapshot = store.manifest.snapshot()
    counts = {"hot": 0, "archive": 0, "labels": 0}
    for rel_path in snapshot["partitions"]:
        counts["hot" if rel_path.startswith("panel/") else "archive"] += 1
    for label_entry in snapshot["labels"].values():
        counts["labels"] += len(label_entry["partitions"])
    return counts


def main() -> int:
    args = _parse_args()
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"v2_p1_backfill_panel_{datetime.now():%Y%m%d}.log"
    logger.add(log_path, encoding="utf-8")

    store = PanelDataStore(args.data_root)
    bootstrap_manifest(store)  # 幂等：377+34 列 + dependencies 登记
    loader = DataLoader(Storage())  # 生产只读
    builder = V2PanelBuilder(loader)
    calendar = [d for d in builder._full_calendar() if d >= "20120104"]  # 写窗口
    chunks = _iter_chunks(args.start_date, args.end_date, args.chunk_years)
    logger.info(
        f"panel 回填启动: 区间 {args.start_date}~{args.end_date}，分块 {chunks}，"
        f"data_root={args.data_root}，日志 {log_path}"
    )

    started = time.time()
    for idx, (chunk_start, chunk_end) in enumerate(chunks, 1):
        coverage = [d for d in calendar if chunk_start <= d <= chunk_end]
        written = _written_days(store, coverage)
        keep_dates = set(coverage) - written
        logger.info(
            f"===== 分块 [{idx}/{len(chunks)}] {chunk_start}~{chunk_end}: "
            f"覆盖 {len(coverage)} 日，已写 {len(written)} 日，待写 {len(keep_dates)} 日 ====="
        )
        if not keep_dates:
            continue
        chunk_ts = time.time()
        builder.backfill(
            TradeDate.from_str(chunk_start),
            TradeDate.from_str(chunk_end),
            _GROUPS,
            store,
            keep_dates=keep_dates,
        )
        counts = _count_partitions(store)
        logger.info(
            f"分块完成，耗时 {(time.time() - chunk_ts)/60:.1f} min；"
            f"manifest 分区: 热区 {counts['hot']} / 冷区 {counts['archive']} / labels {counts['labels']}"
        )

    total = time.time() - started
    counts = _count_partitions(store)
    msg = (
        f"回填完成: 8 族分区 热区 {counts['hot']} + 冷区 {counts['archive']}，"
        f"labels {counts['labels']}，manifest 分区总数 {sum(counts.values())}，"
        f"总耗时 {total/60:.1f} min"
    )
    logger.info(msg)
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

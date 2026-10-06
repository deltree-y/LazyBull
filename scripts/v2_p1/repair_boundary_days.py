# -*- coding: utf-8 -*-
"""v2 P1 单元 4：分块边界缺口日修复性回补（一次性；用户授权，机器时间另计）。

缺口事件（单元 4 闸门预审）：分块末 5 日（20161228/29/30、20211230/31）因
「块末标签不成熟 + 已写判定月粒度盲区」未落 panel/labels。本脚本按**参照同
预热窗**构建缺口日（分块 2 口径 [20150101, 20170210] / 分块 3 口径
[20200101, 20220210] ⇒ 与冻结参照逐位一致），labels 直写日分区，panel 经
`PanelDataStore.repair_archive_partition`（行超集追加 + 理由登记）并入月归档。

用法：python scripts/v2_p1/repair_boundary_days.py [--data-root data]
"""

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
from src.lazybull.v2.store.column_groups import PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.labels_builder import append_labels_for_day  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, _split_groups  # noqa: E402

#: 缺口修复计划：(构建区间起点, 构建区间终点, [缺口日], 归档月, 参照预热口径说明)
_REPAIR_PLAN = [
    (
        "20150101",
        "20170210",
        ["20161228", "20161229", "20161230"],
        "2016-12",
        "参照分块 2（预热自 2014-06）",
    ),
    (
        "20200101",
        "20220210",
        ["20211230", "20211231"],
        "2021-12",
        "参照分块 3（预热自 2019-06）",
    ),
]

_REASON = (
    "单元 4 分块边界缺口回补（D-09）：块末标签不成熟 + 已写判定月粒度盲区致 5 日未落盘；"
    "按参照同预热窗重建，行超集追加"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 P1 分块边界缺口日修复性回补")
    parser.add_argument("--data-root", default="data", help="v2 store 数据根")
    args = parser.parse_args()

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"v2_p1_repair_boundary_{datetime.now():%Y%m%d}.log"
    logger.add(str(log_path), level="DEBUG", encoding="utf-8")

    t0 = time.time()
    store = PanelDataStore(args.data_root)
    loader = DataLoader(Storage())
    builder = V2PanelBuilder(loader=loader)
    calendar = builder._full_calendar()

    total_repaired = 0
    for build_start, build_end, target_days, month, note in _REPAIR_PLAN:
        logger.info(
            f"===== 修复 {month}：构建区间 [{build_start}, {build_end}]（{note}），"
            f"缺口日 {target_days} ====="
        )
        ti_lookups = {d: builder._build_ti_lookup_for_day(calendar, d) for d in target_days}
        captured = builder._run_capture(build_start, build_end, keep_dates=set(target_days))
        captured_days = {d for d, _ in captured}
        missing = sorted(set(target_days) - captured_days)
        if missing:
            raise RuntimeError(f"{month} 缺口日构建未捕获: {missing}")
        for trade_date, df_day in captured:
            materialized = builder._materialize_day(df_day, trade_date, ti_lookups[trade_date])
            append_labels_for_day(store, materialized, calendar)
            frames = _split_groups(materialized)
            for group in PANEL_GROUPS:
                result = store.repair_archive_partition(
                    month, group, frames[group], reason=_REASON
                )
                total_repaired += 1
                logger.info(f"  合并 {result['rel_path']}（+{len(result['added_dates'])} 日）")
        logger.info(f"{month} 修复完成")

    elapsed = (time.time() - t0) / 60
    logger.info(f"全部修复完成：{len(_REPAIR_PLAN)} 个月，合并 {total_repaired} 族次，耗时 {elapsed:.1f} min")
    logger.info(f"日志: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

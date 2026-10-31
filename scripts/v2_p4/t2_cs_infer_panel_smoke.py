# -*- coding: utf-8 -*-
"""v2 P4 T2 cs_infer ↔ panel 推理面只读冒烟（**非判据**——F12 措辞：预期等价未验证）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

用途
====
复用 ``v2/hosts/paper/daily_assembly.compare_cs_infer_vs_panel``，对同一历史日做
cs_infer 分区（旧推理特征面）vs panel 同日截面（v2 store）的**列覆盖 + 非空率对照**。
本工具只产出事实对照（列集合差集、各侧逐列非空率），**不要求逐值一致、不下等价
结论**——「cs_infer 与 panel 推理面预期等价」是未验证的措辞（F12），本冒烟是其
证据采集入口，不是验收判据。

动态选日（任务卡硬要求）：取 cs_infer 与 panel **都有数据**的最近共同交易日——
cs_infer 分区清单经 DataLoader/Storage（``features/cs_infer/*.parquet``）、panel
分区清单经 PanelDataStore（``features/panel/<YYYYMMDD>/``）实测选取；两侧无共同
分区 ⇒ 报错退出（fail-closed，返回码 2），不静默跳过。

只读纪律：不写任何生产根（data/ 全程只读）；结论打印到 stdout，日志写 logs/。
返回码：0 = 对照产出；1 = 运行期错误；2 = 无共同交易日（数据水位不满足前提）。

用法：``python scripts/v2_p4/t2_cs_infer_panel_smoke.py [--date YYYYMMDD]``
（--date 缺省 = 动态选取最近共同交易日）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from src.lazybull.common.config import get_data_root, get_logs_dir  # noqa: E402
from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.v2.hosts.paper.daily_assembly import compare_cs_infer_vs_panel  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

#: 逐列非空率对照的最大展示行数（按 |Δ非空率| 降序）
_TOP_DIVERGENT_COLUMNS = 25


def _configure_logging() -> Path:
    """loguru 文件 sink 落 logs/（ stderr 默认 sink 保留）；返回日志文件路径。"""
    log_path = (
        Path(get_logs_dir()) / f"t2_cs_infer_panel_smoke_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logger.add(str(log_path), encoding="utf-8", level="DEBUG")
    return log_path


def _force_utf8_console() -> None:
    """管道/重定向场景强制 UTF-8（Windows 默认 cp936 会乱码；真控制台不干预）。"""
    for stream in (sys.stdout, sys.stderr):
        if stream.isatty():
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _cs_infer_partition_dates(loader: DataLoader) -> List[str]:
    """cs_infer 分区交易日清单（实测：``features/cs_infer/*.parquet`` 文件主名）。"""
    cs_infer_dir = Path(loader.storage.features_path) / "cs_infer"
    if not cs_infer_dir.exists():
        return []
    return sorted(
        p.stem for p in cs_infer_dir.glob("*.parquet") if len(p.stem) == 8 and p.stem.isdigit()
    )


def _panel_partition_dates(store: PanelDataStore) -> List[str]:
    """panel 分区交易日清单（实测：``features/panel/<YYYYMMDD>/`` 目录名）。"""
    panel_dir = store.panel_dir
    if not panel_dir.exists():
        return []
    return sorted(
        p.name for p in panel_dir.iterdir() if p.is_dir() and len(p.name) == 8 and p.name.isdigit()
    )


def pick_latest_common_date(
    loader: DataLoader, store: PanelDataStore
) -> Tuple[Optional[str], List[str], List[str]]:
    """动态选日：cs_infer 与 panel 都有数据的最近共同交易日。

    Returns:
        (最近共同交易日或 None, cs_infer 分区清单, panel 分区清单)；
        任一侧无分区或两侧无共同交易日 ⇒ 首位 None（调用方 fail-closed 报错）。
    """
    cs_dates = _cs_infer_partition_dates(loader)
    panel_dates = _panel_partition_dates(store)
    common = sorted(set(cs_dates) & set(panel_dates))
    if not common:
        return None, cs_dates, panel_dates
    return common[-1], cs_dates, panel_dates


def _summarize_nonnull_rates(rates: Dict[str, float]) -> str:
    """单侧非空率分布摘要（均值/最小值/全满列占比）。"""
    if not rates:
        return "无列"
    values = list(rates.values())
    full = sum(1 for v in values if v >= 1.0)
    return (
        f"列数 {len(values)}，均值非空率 {sum(values) / len(values):.2%}，"
        f"最小 {min(values):.2%}，全满（100%）列 {full}/{len(values)}"
    )


def _divergent_common_rows(result: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    """共同列非空率逐列对照（按 |Δ| 降序）：[(列, cs_infer 率, panel 率, Δ)]。"""
    rows = []
    for col in result["common_cols"]:
        cs_rate = result["nonnull_rate_cs_infer"].get(col)
        panel_rate = result["nonnull_rate_panel"].get(col)
        if cs_rate is None or panel_rate is None:
            continue
        rows.append((col, cs_rate, panel_rate, panel_rate - cs_rate))
    rows.sort(key=lambda row: abs(row[3]), reverse=True)
    return rows


def _print_report(result: Dict[str, Any], cs_dates: List[str], panel_dates: List[str]) -> None:
    """人类可读对照报告（事实陈述，不下等价判据结论）。"""
    date = result["date"]
    print("=" * 78)
    print(f"cs_infer ↔ panel 推理面只读冒烟（非判据；F12：预期等价未验证） 日期={date}")
    print("=" * 78)
    print(
        f"分区水位：cs_infer {len(cs_dates)} 日（最新 {cs_dates[-1]}），"
        f"panel {len(panel_dates)} 日（最新 {panel_dates[-1]}）；"
        f"选日口径 = 两侧最近共同交易日"
    )
    print(
        f"列覆盖：cs_infer {len(result['cs_infer_cols'])} 列 / "
        f"panel {len(result['panel_cols'])} 列 / 共同 {len(result['common_cols'])} 列"
    )
    only_cs = result["only_cs_infer"]
    only_panel = result["only_panel"]
    print(f"仅 cs_infer（{len(only_cs)} 列）: {only_cs[:30]}{' …' if len(only_cs) > 30 else ''}")
    print(f"仅 panel（{len(only_panel)} 列）: {only_panel[:30]}{' …' if len(only_panel) > 30 else ''}")
    print(f"非空率（cs_infer 侧）: {_summarize_nonnull_rates(result['nonnull_rate_cs_infer'])}")
    print(f"非空率（panel 侧）  : {_summarize_nonnull_rates(result['nonnull_rate_panel'])}")
    rows = _divergent_common_rows(result)
    shown = rows[:_TOP_DIVERGENT_COLUMNS]
    print(f"共同列逐列非空率对照（按 |Δ| 降序，前 {len(shown)}/{len(rows)} 行）:")
    print(f"  {'列名':<40} {'cs_infer':>10} {'panel':>10} {'Δ(panel-cs)':>12}")
    for col, cs_rate, panel_rate, delta in shown:
        print(f"  {col:<40} {cs_rate:>10.2%} {panel_rate:>10.2%} {delta:>+12.2%}")
    same = sum(1 for _c, a, b, _d in rows if a == b)
    print(f"共同列非空率逐位一致列数: {same}/{len(rows)}")
    print(
        "结论口径：本对照只陈述列覆盖与非空率事实（不要求逐值一致）；"
        "推理面等价性未验证（F12），不构成验收判据。"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """入口：动态选日 → 对照 → 打印报告。只读，零写入生产根。"""
    _force_utf8_console()
    parser = argparse.ArgumentParser(
        description="v2 P4 T2 cs_infer ↔ panel 推理面只读冒烟（非判据）"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="对照交易日（YYYYMMDD）；缺省 = cs_infer 与 panel 最近共同交易日（实测选取）",
    )
    args = parser.parse_args(argv)
    log_path = _configure_logging()

    data_root = get_data_root()
    logger.info(f"cs_infer↔panel 冒烟启动（只读），数据根: {data_root}，日志: {log_path}")
    loader = DataLoader(storage=Storage(root_path=data_root), verbose=False)
    store = PanelDataStore(data_root)

    if args.date:
        date = str(args.date).strip().replace("-", "")
        cs_dates = _cs_infer_partition_dates(loader)
        panel_dates = _panel_partition_dates(store)
        if date not in set(cs_dates) or date not in set(panel_dates):
            print(
                f"错误：指定日期 {date} 不满足双侧分区前提（"
                f"在 cs_infer: {date in set(cs_dates)}，在 panel: {date in set(panel_dates)}）",
                file=sys.stderr,
            )
            return 2
    else:
        date, cs_dates, panel_dates = pick_latest_common_date(loader, store)
        if date is None:
            print(
                "错误：cs_infer 与 panel 无共同交易日分区（"
                f"cs_infer 分区 {len(cs_dates)} 个"
                f"（最新 {cs_dates[-1] if cs_dates else '无'}），"
                f"panel 分区 {len(panel_dates)} 个"
                f"（最新 {panel_dates[-1] if panel_dates else '无'}））——冒烟前提不满足",
                file=sys.stderr,
            )
            return 2
    logger.info(f"对照日期: {date}")

    result = compare_cs_infer_vs_panel(date, store, loader)
    _print_report(result, cs_dates, panel_dates)
    logger.info(f"cs_infer↔panel 冒烟完成（只读，零写入）: {date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""v2 P4 T2 单日 dry-run 形态证明（T2 验收项；阶段专用临时开发文件）。

复杂度豁免：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。

证明形态
========
**真实数据装配 + 宿主一日驱动 + 隔离根落盘**：

1. 读生产 ``data/paper/config.yaml``（经生产 ``load_paper_config``，**只读**）；
2. 在 **tmp 隔离根**（默认 ``temp/t2_daily_dryrun_<时间戳>/paper``，或
   ``--paper-root`` 指定）上经 ``create_paper_host`` 装配宿主——隔离状态根 +
   真实 panel store + 真实 clean/daily + 真实模型（config 的 model_version）+
   真实交易日历；
3. 驱动 ``run_t0(date)`` 单日（date 缺省 = 当前水位最近交易日：panel ∩ clean/daily
   ∩ 交易日历的最新共同日，实测选取）；
4. 打印 PaperDayResult 摘要（t0_status / 成交 / 次日指令 / nav / 就绪门结论）与
   隔离根产物清单。

硬约束（containment）：生产 ``data/paper/`` **零写入**——驱动前后对生产根做
文件集 + 逐文件 sha256 快照比对（口径同 ``t05b_live_anchor.snapshot_protected_root``
），零变更才 PASS。任何真实装配/驱动失败（如模型特征列在 panel 缺失）如实打印
失败点与缺口，**不伪造成功**。

返回码（fail-closed，真实结论入返回码）：
- 0 = 当日真实驱动成功（driven=True）且生产根零变更；
- 1 = 装配/驱动抛异常（如实打印异常）；
- 2 = 数据未就绪（readiness.ok=False，宿主按契约推迟、不落盘）或选不到可用交易日；
- 3 = containment 违约（生产 data/paper/ 发生变更，最严重，置顶打印）。

用法：``python scripts/v2_p4/t2_daily_dryrun.py [--date 20260731] [--paper-root <tmp>]``
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from scripts.v2_p4.t05b_live_anchor import snapshot_protected_root  # noqa: E402
from src.lazybull.common.config import get_data_root, get_logs_dir, get_paper_root  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402
from src.lazybull.v2.hosts.paper.factory import create_paper_host  # noqa: E402
from src.lazybull.v2.hosts.paper.host import PaperDayResult, PaperHost  # noqa: E402


def _configure_logging() -> Path:
    """loguru 文件 sink 落 logs/（stderr 默认 sink 保留）；返回日志文件路径。"""
    log_path = Path(get_logs_dir()) / f"t2_daily_dryrun_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _resolve_default_date(host: PaperHost) -> Optional[str]:
    """当前水位最近交易日：panel 分区 ∩ clean/daily 分区 ∩ 交易日历的最新共同日。

    实测三侧水位（不硬编码日期）；无共同日 ⇒ None（调用方按 fail-closed 处理）。
    """
    panel_dir = host.panel_store.panel_dir
    panel_dates = (
        {p.name for p in panel_dir.iterdir() if p.is_dir()} if panel_dir.exists() else set()
    )
    daily_dir = Path(host.data_storage.clean_path) / "daily"
    clean_dates = (
        {p.stem.replace("-", "") for p in daily_dir.glob("*.parquet")}
        if daily_dir.exists()
        else set()
    )
    calendar = {day.strftime("%Y%m%d") for day in host.trading_calendar}
    common = sorted(panel_dates & clean_dates & calendar)
    return common[-1] if common else None


def _print_result_summary(result: PaperDayResult) -> None:
    """PaperDayResult 摘要（成交/次日指令/nav/就绪门结论）。"""
    print("-" * 78)
    print(
        f"PaperDayResult: 请求日期={result.requested_date} 校正日期={result.corrected_date} "
        f"入口={result.run_kind} driven={result.driven} t0_status={result.t0_status}"
    )
    fills = result.fills
    buys = [row for row in fills if row.get("action") == "buy"]
    sells = [row for row in fills if row.get("action") == "sell"]
    print(f"当日成交: {len(fills)} 笔（买 {len(buys)} / 卖 {len(sells)}）")
    for row in fills:
        print(
            f"  {row['action']:<4} {row['ts_code']:<12} 量 {int(row['shares']):>8} "
            f"价 {float(row['price']):>10.3f} 费 {float(row['total_cost']):>10.2f} "
            f"因 {row.get('reason', '')}"
        )
    orders = result.orders
    order_buys = [o for o in orders if o.side.value == "buy"]
    order_sells = [o for o in orders if o.side.value == "sell"]
    print(f"次日指令: {len(orders)} 条（买 {len(order_buys)} / 卖 {len(order_sells)}）")
    for order in orders:
        print(
            f"  {order.side.value:<4} {order.ts_code.value:<12} "
            f"目标持仓 {int(order.target_shares.value):>8} 因 {order.reason.value}"
        )
    print(f"nav（当日总资产估值）: {result.nav}")
    if result.readiness is not None:
        print(
            f"就绪门结论: ok={result.readiness.ok}，缺口 {len(result.readiness.gaps)} 项: "
            f"{result.readiness.gaps}"
        )
    else:
        print("就绪门结论: 通过（driven=True ⇒ 未走 not_ready 分支）")
    if result.push_text:
        print("推送文本:")
        for line in result.push_text.splitlines():
            print(f"  {line}")
    print("-" * 78)


def _print_isolated_artifacts(isolated_root: Path) -> int:
    """隔离根产物清单（相对路径 + 字节数，排序）；返回文件数。"""
    files = sorted(p for p in isolated_root.rglob("*") if p.is_file())
    print(f"隔离根产物清单（{isolated_root}，{len(files)} 个文件）:")
    for path in files:
        print(f"  {path.relative_to(isolated_root)}  ({path.stat().st_size} B)")
    return len(files)


def _check_containment(before: Dict[str, str], protected_root: Path) -> bool:
    """生产根零变更校验（文件集 + 逐文件 sha256）；变更明细如实打印。"""
    after = snapshot_protected_root(protected_root)
    if after == before:
        print(f"containment PASS：生产根 {protected_root} 驱动前后零变更（{len(after)} 个文件）")
        return True
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in after.keys() & before.keys() if after[k] != before[k])
    print(f"containment FAIL：生产根 {protected_root} 发生变更！")
    print(f"  新增 {len(added)}: {added[:5]}")
    print(f"  删除 {len(removed)}: {removed[:5]}")
    print(f"  内容变化 {len(changed)}: {changed[:5]}")
    return False


def main(argv: Optional[List[str]] = None) -> int:
    """入口：containment 快照 → 真实装配 → 单日驱动 → 摘要 + 产物清单 + 复验。"""
    _force_utf8_console()
    parser = argparse.ArgumentParser(
        description="v2 P4 T2 单日 dry-run 形态证明（隔离根落盘 + 生产根零写入 containment）"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="驱动交易日（YYYYMMDD）；缺省 = 当前水位最近交易日（panel∩clean∩日历实测）",
    )
    parser.add_argument(
        "--paper-root",
        default=None,
        help="隔离纸面根目录；缺省 = temp/t2_daily_dryrun_<时间戳>/paper",
    )
    args = parser.parse_args(argv)
    log_path = _configure_logging()

    protected_root = Path(get_paper_root())
    if args.paper_root:
        isolated_root = Path(args.paper_root)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        isolated_root = _REPO_ROOT / "temp" / f"t2_daily_dryrun_{stamp}" / "paper"
    print(f"生产纸面根（只读）: {protected_root}")
    print(f"隔离纸面根（全部落盘位）: {isolated_root}")
    print(f"日志: {log_path}")

    # ① 驱动前生产根快照（覆盖其后全部动作：配置读取 + 装配 + 驱动）
    before = snapshot_protected_root(protected_root)
    logger.info(f"生产根驱动前快照: {len(before)} 个文件")

    result: Optional[PaperDayResult] = None
    failure_lines: List[str] = []
    try:
        # ② 生产配置只读读取（load_paper_config 经 create_layout=False 不建目录；
        # 零文件写入由 containment 复验证明）
        config_flat = load_paper_config(str(protected_root))
        # ③ 隔离根 + 真实数据面（panel store / clean / 模型 / 日历）装配
        host = create_paper_host(
            store=str(isolated_root),
            config_flat=config_flat,
            data_root=get_data_root(),
            verbose=False,
        )
        # ④ 日期解析：--date 指定或当前水位最近共同交易日
        if args.date:
            date = str(args.date).strip().replace("-", "")
        else:
            date = _resolve_default_date(host)
            if date is None:
                print("错误：panel ∩ clean/daily ∩ 交易日历无共同交易日，无法选日", file=sys.stderr)
                return 2
        print(f"驱动日期: {date}（{'命令行指定' if args.date else '水位实测最近共同交易日'}）")
        # ⑤ 单日驱动（T0 视图；首日无历史计划 ⇒ 成交为零属预期形态）
        host.run_t0(date)
        result = host.last_day_result
    except Exception as exc:  # 真实失败如实打印（不伪造成功）
        failure_lines = traceback.format_exc().splitlines()
        print(f"装配/驱动异常（如实报告，未伪造成功）: {type(exc).__name__}: {exc}")
        for line in failure_lines[-15:]:
            print(f"  {line}")
        _check_containment(before, protected_root)
        return 1

    assert result is not None  # run_t0 正常返回必有 last_day_result
    _print_result_summary(result)
    n_files = _print_isolated_artifacts(isolated_root)
    containment_ok = _check_containment(before, protected_root)
    if not containment_ok:
        return 3
    if not result.driven:
        print(f"当日未真实驱动（t0_status={result.t0_status}）——验收形态未达成，如实报告")
        return 2
    print(f"dry-run PASS：{result.corrected_date} 真实驱动完成，隔离根产物 {n_files} 个，生产根零变更")
    return 0


if __name__ == "__main__":
    sys.exit(main())

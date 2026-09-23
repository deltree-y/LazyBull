#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纸面暴露政策预热面板生成（terminal_loss P2-5 / 纸面 P2a）。

背景：暴露政策的滚动阈值来自 provider 逐日累积的日级面板（判定时用 `[t-window, t-1]`
历史）；纸面从启动日累积，启动初期（< min_window=60 交易日）不判定。本工具用
**同策略的回测持仓快照**（`batch_walk_forward` 产物 `*持仓快照*split*.csv`）回放
provider，把启动前的面板历史（默认回看 270 个交易日）导出为预热文件；纸面启动时
自动恢复（仅当面板为空，见 ``paper/exposure_policy.py::_apply_warmup_if_needed``）。

用法：
  # 步骤 1（在有回测产物的机器上执行）：列出候选批次
  py -3 scripts\\prepare_paper_exposure_warmup.py --list-batches

  # 步骤 2：选一个与纸面策略配置一致的批，显式指定生成
  py -3 scripts\\prepare_paper_exposure_warmup.py `
      --snapshot-dir "data\\walk_forward\\batches\\wf_batch_20260921_090929\\raw"
  # 输出默认：<policy_model_root>/paper_warmup/state.json
  # （模型根 / 后缀 / 策略默认从 data/paper/config.yaml 的 exposure 配置读取）

跨机器流程（纸面机可以没有回测产物）：
  预热文件是自包含的（1 MB 级 JSON，含逐日面板行，不引用任何快照文件）：
  - **生成端**（有 `data/walk_forward/batches/*` 回测产物 + 历史行情/特征）：跑本工具；
    **纸面策略参数变化后需重生**；
  - **纸面机**：把 `state.json` 拷到 `<policy_model_root>/paper_warmup/`（或任意路径 +
    `policy_warmup_file` 指向），并保证**折模型目录与生成端一致**（否则摘要失配、纸面告警忽略）；
    纸面机不需要任何回测快照/批次目录。

口径登记（与纸面运行期的差异）：
  1. 预热期面板行来自**回测持仓快照**（收盘后口径），运行期为执行前口径，
     换仓日股票集合可能有少量差异（阈值统计以 p_loss 均值为主，影响有限）；
  2. 预热回放按“按日自动选折”（与运行侧 policy_fold 无关；面板本身为多折混合语义）；
  3. 折模型重训后必须重新运行本工具（文件含折模型内容摘要，失配时纸面忽略预热并告警）；
  4. 快照批次必须**显式指定**（不自动选最新）——选与纸面策略参数（top_n / rebalance / stagger /
     仓位模式等）一致的批，否则阈值历史来自不同持仓路径。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lazybull.common.config import get_data_root  # noqa: E402
from src.lazybull.common.logger import setup_logger  # noqa: E402
from src.lazybull.risk.terminal_loss.dataset import load_trade_calendar  # noqa: E402
from src.lazybull.risk.terminal_loss.exposure_online import (  # noqa: E402
    DailyExposureProvider,
    OnlinePolicyConfig,
    folds_model_digest,
    to_date_str,
)
from src.lazybull.risk.terminal_loss.policy_sidecar import _read_snapshot  # noqa: E402

WARMUP_KIND = "paper_exposure_warmup"
WARMUP_VERSION = 1


def _paper_defaults() -> dict:
    """从 `data/paper/config.yaml` 读默认的 exposure 配置（读不到返回空）。"""
    try:
        from src.lazybull.paper import PaperStorage

        config = PaperStorage().load_config() or {}
    except Exception as exc:  # noqa: BLE001 - 默认值读取失败只影响便捷性
        logger.warning(f"读取纸面配置失败（需要显式参数）: {exc}")
        return {}
    return {
        "policy": str(config.get("exposure_policy") or "") or None,
        "model_root": str(config.get("policy_model_root") or "") or None,
        "arm_suffix": str(config.get("policy_arm_suffix") or "") or None,
    }


def scan_batch_candidates(root: Path) -> list:
    """扫描批次目录中的快照候选（供 --list-batches 显式选择）。

    Returns:
        按最近修改倒序的列表，每项含 batch / raw_dir / snapshots / first_day / last_day /
        关键参数（从同名批的 summary 首行读，读不到则省略）。
    """
    candidates = []
    for raw in sorted(root.glob("*/raw")):
        snaps = sorted(raw.glob("*持仓快照*split*.csv"))
        if not snaps:
            continue
        info = {
            "batch": raw.parent.name,
            "raw_dir": str(raw),
            "snapshots": len(snaps),
            "last_modified": max(path.stat().st_mtime for path in snaps),
        }
        try:
            head = pd.read_csv(snaps[0], usecols=["日期"], dtype=str)
            tail = pd.read_csv(snaps[-1], usecols=["日期"], dtype=str)
            info["first_day"] = str(head.iloc[:, 0].min()).replace("-", "")[:8]
            info["last_day"] = str(tail.iloc[:, 0].max()).replace("-", "")[:8]
        except Exception as exc:  # noqa: BLE001 - 日期探测失败不影响候选列出
            info["date_note"] = f"日期读取失败: {exc}"
        summaries = sorted(raw.glob("*summary*.csv"))
        if summaries:
            try:
                summary = pd.read_csv(summaries[0], nrows=1)
                for key in ("top_n", "rebalance_freq", "stagger_tranches", "position_sizing"):
                    value = summary.iloc[0].get(key) if key in summary.columns else None
                    if value is not None and str(value).strip() and str(value) != "nan":
                        info[key] = str(value)
            except Exception as exc:  # noqa: BLE001 - 参数摘要读取失败仅提示
                info["summary_note"] = f"参数读取失败: {exc}"
        candidates.append(info)
    candidates.sort(key=lambda item: item["last_modified"], reverse=True)
    return candidates


def _print_batch_table(candidates: list) -> None:
    """打印候选批次表（帮助选择与纸面策略一致的快照目录）。"""
    if not candidates:
        print("（无候选：未找到任何含 *持仓快照*split*.csv 的批次目录）")
        return
    print(f"候选批次 {len(candidates)} 个（按最近修改倒序）：")
    for item in candidates:
        params = " ".join(
            f"{key}={item[key]}"
            for key in ("top_n", "rebalance_freq", "stagger_tranches", "position_sizing")
            if key in item
        )
        print(
            f"  {item['batch']}\n"
            f"    snapshots={item['snapshots']}"
            f"  range={item.get('first_day', '?')}~{item.get('last_day', '?')}"
            f"  {params}\n"
            f"    --snapshot-dir \"{item['raw_dir']}\""
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="生成纸面暴露政策预热面板文件（P2a）")
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="（必需）持仓快照目录（含 *持仓快照*split*.csv）；用 --list-batches 查看候选",
    )
    parser.add_argument(
        "--list-batches",
        action="store_true",
        help="列出可用批次候选（批次 / 快照数 / 日期范围 / 参数摘要）并退出",
    )
    parser.add_argument(
        "--start", default=None, help="回放起始交易日 YYYYMMDD（默认按 --lookback-days）"
    )
    parser.add_argument("--end", default=None, help="回放结束日 YYYYMMDD（默认=快照最大日期）")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=270,
        help="默认回放交易日数（默认 270：覆盖 window=250 + 余量）",
    )
    parser.add_argument("--model-root", default=None, help="终损折模型根（默认取纸面配置）")
    parser.add_argument("--arm-suffix", default=None, help="折目录后缀（默认取纸面配置）")
    parser.add_argument("--policy", default=None, help="策略串（默认取纸面配置 exposure_policy）")
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件；默认 <model-root>/paper_warmup/state.json",
    )
    parser.add_argument("--limit-days", type=int, default=None, help="调试：仅回放前 N 个交易日")
    args = parser.parse_args()

    setup_logger()
    if args.list_batches:
        _print_batch_table(
            scan_batch_candidates(project_root / "data" / "walk_forward" / "batches")
        )
        return 0
    if not args.snapshot_dir:
        logger.error(
            "必须显式指定 --snapshot-dir（选择与纸面策略参数一致的批；"
            "可用 --list-batches 查看候选）"
        )
        return 2
    defaults = _paper_defaults()
    policy_text = args.policy or defaults.get("policy")
    model_root = args.model_root or defaults.get("model_root")
    arm_suffix = args.arm_suffix or defaults.get("arm_suffix")
    if not policy_text or not model_root or not arm_suffix:
        logger.error(
            "缺少策略/模型源参数，且未能在 data/paper/config.yaml 读到默认值；"
            "请显式给出 --policy / --model-root / --arm-suffix"
        )
        return 2

    snapshot_dir = Path(args.snapshot_dir)
    files = sorted(snapshot_dir.glob("*持仓快照*split*.csv"))
    if not files:
        logger.error(f"目录下未找到 *持仓快照*split*.csv: {snapshot_dir}")
        return 2
    frames = [_read_snapshot(path) for path in files]
    snapshots = pd.concat(frames, ignore_index=True)
    duplicates = int(snapshots.duplicated(subset=["date", "ts_code"]).sum())
    if duplicates:
        logger.warning(f"快照存在同日同股重复行 {duplicates} 条，按首行去重")
        snapshots = snapshots.drop_duplicates(subset=["date", "ts_code"], keep="first")
    snapshots["date"] = snapshots["date"].map(to_date_str)

    end = to_date_str(args.end) if args.end else str(snapshots["date"].max())
    day_list_all = sorted(day for day in snapshots["date"].unique() if day <= end)
    if args.start:
        start = to_date_str(args.start)
        day_list = [day for day in day_list_all if day >= start]
    else:
        day_list = day_list_all[-int(args.lookback_days) :]
    if args.limit_days:
        day_list = day_list[: int(args.limit_days)]
    if not day_list:
        logger.error("裁剪后无任何交易日可回放，请检查 --start/--end/--lookback-days")
        return 2
    first_day, last_day = day_list[0], day_list[-1]
    logger.info(
        f"预热回放区间: {first_day}~{last_day}（{len(day_list)} 个交易日；"
        f"模型根 {model_root}；后缀 {arm_suffix}）"
    )

    config = OnlinePolicyConfig.parse(policy_text)
    provider = DailyExposureProvider(
        config,
        risk_root=str(model_root),
        arm_suffix=arm_suffix,
        data_root=get_data_root(),
        book="end_of_day",
        verbose=False,
        coverage_mode="serving",
    )
    selected = snapshots[snapshots["date"].isin(set(day_list))]
    rows_total = 0
    for index, (day, part) in enumerate(selected.groupby("date", sort=True), start=1):
        rows = part[["ts_code", "weight", "remaining_intervals"]]
        provider.multiplier_for(day, rows)
        rows_total += int(len(rows))
        if index % 50 == 0 or index == len(day_list):
            logger.info(
                f"预热回放进度 {index}/{len(day_list)}（{day}；入账行 {rows_total}；"
                f"触发 {provider.stats['trigger_days']} 日）"
            )

    # 收尾：用日历上更晚的日期驱动标签成熟，避免把“本可成熟”的行带入导出状态
    h_max = max((fold.h_max for fold in provider.folds), default=1)
    calendar = load_trade_calendar(get_data_root(), "19900101", "20991231")
    if last_day in calendar:
        through = calendar[min(len(calendar) - 1, calendar.index(last_day) + h_max + 1)]
        provider.advance_label_maturation(through)
        logger.info(f"标签成熟推进至 {through}（h_max={h_max}）")
    else:
        logger.warning(f"回放末日 {last_day} 不在交易日历中，跳过成熟收尾")

    payload = {
        "kind": WARMUP_KIND,
        "version": WARMUP_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_root": str(model_root),
        "arm_suffix": arm_suffix,
        "policy": policy_text,
        "policy_fingerprint": provider.fingerprint,
        "folds_digest": folds_model_digest(str(model_root), arm_suffix),
        "source": {
            "snapshot_dir": str(snapshot_dir),
            "files": len(files),
            "duplicates_dropped": duplicates,
        },
        "coverage": {
            "first_day": first_day,
            "last_day": last_day,
            "days": len(day_list),
            "rows": rows_total,
        },
        "stats": {
            "trigger_days": provider.stats["trigger_days"],
            "judged_days": provider.stats["judged_days"],
            "dropped_invalid_rows": provider.stats["dropped_invalid_rows"],
            "pending_rows": provider.stats["pending_rows"],
        },
        "provider": provider.export_state(),
    }
    output = Path(args.output) if args.output else Path(model_root) / "paper_warmup" / "state.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    logger.info(
        f"预热面板已写出: {output}（{len(day_list)} 日 / {rows_total} 行；"
        f"指纹 {provider.fingerprint}；digest {payload['folds_digest']}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

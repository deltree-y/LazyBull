"""terminal_loss 政策旁路入口（P2-1 离线回放）。

用法（示例）：

    python scripts/analyze_policy_sidecar.py \
        --snapshot-dir data/walk_forward/raw \
        --risk-root data/walk_forward/terminal_risk_wf \
        --arm _d5_v6m_fscore \
        --out-dir data/walk_forward/terminal_risk_wf/archives/policy_sidecar_d5

产出（全部**中文表头**，utf-8-sig 可直接用 Excel 打开）：

- ``风险台账.csv``：一行 = 某持仓的某个剩余持有期（含模型读数、当日截面分位、
  日波动率、市场波动状态、事后实际收益与是否异常亏损）
- ``触发清单.csv``：双重条件命中（正确拦截 / 误杀）与漏报事件，供人工复核
- ``阈值扫描.csv``：每个 (截面分位阈值, 绝对概率阈值) 组合的拦截/误杀/漏报与收益代价

本入口只读快照与模型，**不修改任何交易链路**。
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lazybull.risk.terminal_loss.policy_sidecar import (  # noqa: E402
    build_trigger_list,
    score_holdings,
    scan_thresholds,
    summarize_ledger,
    write_ledger,
    write_threshold_scan,
    write_trigger_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="terminal_loss 政策旁路（P2-1 离线回放）")
    parser.add_argument(
        "--snapshot-dir",
        default="data/walk_forward/raw",
        help="含中文表头「持仓快照」CSV 的目录（默认 data/walk_forward/raw）",
    )
    parser.add_argument(
        "--snapshot", nargs="*", default=None, help="显式指定快照文件（覆盖目录扫描）"
    )
    parser.add_argument("--data-root", default="data", help="数据根目录（clean / features）")
    parser.add_argument(
        "--risk-root",
        default="data/walk_forward/terminal_risk_wf",
        help="terminal_loss WF 根目录（选择折模型）",
    )
    parser.add_argument(
        "--arm",
        default="_d5_v6m_fscore",
        help="折目录后缀（如 _d5_v6m_fscore / _v6m_fscore_state）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="输出目录（默认 {risk-root}/archives/policy_sidecar{arm}）",
    )
    parser.add_argument(
        "--p-hi",
        nargs="*",
        type=float,
        default=[0.90, 0.95, 0.99],
        help="截面分位阈值网格（触发清单用第一个值）",
    )
    parser.add_argument(
        "--p-abs",
        nargs="*",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20],
        help="绝对概率阈值网格（触发清单用第一个值）",
    )
    parser.add_argument(
        "--no-daypct",
        action="store_true",
        help="跳过当日截面分位计算（不产出分位列，阈值扫描将不可用）",
    )
    parser.add_argument("--no-missed", action="store_true", help="触发清单不包含漏报事件")
    return parser.parse_args()


def _collect_snapshots(args: argparse.Namespace) -> list:
    if args.snapshot:
        files = [Path(p) for p in args.snapshot]
    else:
        files = sorted(Path(args.snapshot_dir).glob("*持仓快照_*.csv"))
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError(
            f"未找到持仓快照 CSV（目录 {args.snapshot_dir}；"
            f"请先跑一次带 OOS 回测的 walk_forward，快照名形如 walk_forward_持仓快照_*_splitNN.csv）"
        )
    return files


def main() -> int:
    args = parse_args()
    files = _collect_snapshots(args)
    logger.info(
        f"待打分快照 {len(files)} 个：{[p.name for p in files[:3]]}"
        f"{'...' if len(files) > 3 else ''}"
    )

    ledger = score_holdings(
        snapshot_files=files,
        risk_root=args.risk_root,
        arm_suffix=args.arm,
        data_root=args.data_root,
        with_daypct=not args.no_daypct,
    )
    out_dir = Path(
        args.out_dir or (Path(args.risk_root) / "archives" / f"policy_sidecar{args.arm}")
    )
    write_ledger(ledger, out_dir / "风险台账.csv")

    summary = summarize_ledger(ledger)
    logger.info(
        "台账概览: "
        + "，".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in summary.items()
        )
    )

    if args.no_daypct:
        logger.warning("已跳过当日截面分位：触发清单与阈值扫描需要分位列，本次不产出")
        return 0

    triggers = build_trigger_list(
        ledger, p_hi=args.p_hi[0], p_abs=args.p_abs[0], include_missed=not args.no_missed
    )
    write_trigger_list(triggers, out_dir / "触发清单.csv")
    if not triggers.empty:
        counts = triggers["event_type"].value_counts().to_dict()
        logger.info(f"触发清单分布: {counts}")

    scan = scan_thresholds(ledger, p_hi_list=args.p_hi, p_abs_list=args.p_abs)
    write_threshold_scan(scan, out_dir / "阈值扫描.csv")
    print(scan.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

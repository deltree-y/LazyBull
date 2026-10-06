# -*- coding: utf-8 -*-
"""v2 P1 单元 4：闸门③——已写分区内容不变性验证（新增列只写新文件 / 历史分区不可变）。

D-13 后口径（2026-10-06 裁决，冻结文档 §8）：panel 解锁前瞻 3 列（days_to_unlock /
unlock_ratio / unlock_risk_flag）经 D-12 修复通道重写为全量 PIT 语义，**构建通道重建
不再复现该 3 列的生产值**（修复通道独立于构建通道，修复正确性由
`v2_p1_d12_exemption_evidence_*.json` 双恒等式证据独立治理）。故闸门③对
announcement/risk 两族剔除该 3 列后逐值比对，其余 6 族全列逐值比对。

三层设计（同上下文重跑才保证逐位一致，机制见 v1 版注释与单元 4 报告 §5）：
- 层 0：store 级往返 no-op（读回热区日分区原样重放 ⇒ 指纹不变；两阶段提交 no-op 语义）。
- 层 1：热区日按原分块上下文 [chunk_start, hot_date] 重建 **落 scratch 根**，
  与生产分区逐列比对（因果 ⇒ 逐位一致；年分区公告按分区名日期过滤，子范围终点
  截断丢整年公告的实证教训仅影响 D-12 豁免列，已由剔除口径覆盖）。
- 层 2：冷区月按原分块上下文 [chunk_start, chunk_end] 重建落 scratch 根，逐列比对。
  （范围终点必须与原分块一致；重建为全管线重算，机器耗时 ≈ 原分块耗时。）

任何构建通道列差异 ⇒ 出口 1（实现漂移 / 历史分区被改写嫌疑）。

产物：data/reports/v2_p1_immutable_partitions_<date>.json（checked/stale/missing 计数
与结论，含三层比对明细）+ 控制台汇总。

用法：
    python scripts/v2_p1/verify_immutable_partitions.py --data-root data \
        --hot-date 20260603 --cold-month 2025-05 \
        --chunk-start 20200101 --chunk-end 20260702
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.panel_reconcile import D12_EXEMPT_COLUMNS  # noqa: E402

_GROUPS = tuple(PANEL_GROUPS)
#: D-12 修复通道重写的列（构建通道重建不复现；D-13 豁免，证据独立治理）
_D12_COLS = tuple(sorted(D12_EXEMPT_COLUMNS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 P1 闸门③：已写分区内容不变性验证（D-13 后口径）")
    parser.add_argument("--data-root", required=True, help="v2 store 生产数据根")
    parser.add_argument("--hot-date", required=True, help="已写热区日 YYYYMMDD")
    parser.add_argument("--cold-month", required=True, help="已写冷区月 YYYY-MM")
    parser.add_argument(
        "--chunk-start",
        required=True,
        help="写出目标分区的分块起点 YYYYMMDD（全量回填 = 20120104/20150101/20200101 之一）",
    )
    parser.add_argument(
        "--chunk-end",
        required=True,
        help="写出目标分区的分块终点 YYYYMMDD（同上 = 20161231/20211231/20260702）",
    )
    parser.add_argument(
        "--scratch-root",
        default=str(ROOT / "temp" / "p1_gate3_scratch"),
        help="重建落盘 scratch 根（默认 temp/p1_gate3_scratch；保留可断点续传）",
    )
    return parser.parse_args()


def _fingerprints(store: PanelDataStore, rel_paths: list[str]) -> dict[str, str | None]:
    return {rel: store.manifest.partition_fingerprint(rel) for rel in rel_paths}


def _store_roundtrip_noop(store: PanelDataStore, hot_date: str) -> None:
    """store 级往返：读回热区日分区原样重放 ⇒ 指纹不变（两阶段提交 no-op 语义）。"""
    for group in _GROUPS:
        path = store.panel_dir / hot_date / f"{group}.parquet"
        df = pd.read_parquet(path)
        store.append_features(TradeDate.from_str(hot_date), group, df)


def _compare_partition_content(prod_file: Path, scratch_file: Path, group: str) -> list[str]:
    """逐列比对生产 vs scratch 分区（announcement/risk 族剔除 D-12 豁免列）。

    行序规整后 assert_frame_equal(check_exact=True)（含 dtype 与 NaN 模式；
    float32/64 尾差即差异——D-09 dtype 教训）。返回差异描述列表（空 = 一致）。
    """
    if not scratch_file.exists():
        return [f"scratch 分区未生成: {scratch_file}"]
    df_p = pd.read_parquet(prod_file)
    df_s = pd.read_parquet(scratch_file)
    drops = [c for c in _D12_COLS if c in df_p.columns or c in df_s.columns]
    if drops:
        logger.info(f"  [{group}] D-13 豁免剔除列: {drops}")
        df_p = df_p.drop(columns=[c for c in drops if c in df_p.columns])
        df_s = df_s.drop(columns=[c for c in drops if c in df_s.columns])
    problems: list[str] = []
    if set(df_p.columns) != set(df_s.columns):
        problems.append(
            f"列集不一致: prod-only={sorted(set(df_p.columns) - set(df_s.columns))} "
            f"scratch-only={sorted(set(df_s.columns) - set(df_p.columns))}"
        )
        return problems
    if len(df_p) != len(df_s):
        problems.append(f"行数不一致: 生产 {len(df_p)} vs scratch {len(df_s)}")
        return problems
    key = ["trade_date", "ts_code"]
    df_p = df_p.sort_values(key).reset_index(drop=True)
    df_s = df_s[df_p.columns].sort_values(key).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(df_p, df_s, check_exact=True)
    except AssertionError as exc:
        problems.append(str(exc)[:600])
    return problems


def _compare_all(
    prod_store: PanelDataStore,
    scratch_store: PanelDataStore,
    rel_paths: list[str],
    label: str,
) -> list[str]:
    """对一组同名单元（热区日 / 冷区月的 8 族文件）逐族比对，返回全部差异描述。"""
    all_problems: list[str] = []
    for rel in rel_paths:
        group = Path(rel).stem
        if rel.startswith("panel_archive/"):
            sub = rel.removeprefix("panel_archive/")
            prod_file = prod_store.archive_dir / sub
            scratch_file = scratch_store.archive_dir / sub
        else:
            sub = rel.removeprefix("panel/")
            prod_file = prod_store.panel_dir / sub
            scratch_file = scratch_store.panel_dir / sub
        problems = _compare_partition_content(prod_file, scratch_file, group)
        for p in problems:
            all_problems.append(f"[{label}/{rel}] {p}")
        if not problems:
            logger.info(f"  [{label}] {rel} 逐值一致 ✓")
    return all_problems


def main() -> int:
    args = _parse_args()
    store = PanelDataStore(args.data_root)
    loader = DataLoader(Storage())
    builder = V2PanelBuilder(loader)
    hot_rels = [f"panel/{args.hot_date}/{g}.parquet" for g in _GROUPS]
    cold_rels = [f"panel_archive/{args.cold_month}/{g}.parquet" for g in _GROUPS]
    targets = hot_rels + cold_rels

    # 机器产物（评审整改：闸门③ PASS 结论落盘，与其他 v2_p1 产物同风格）
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": args.data_root,
        "hot_date": args.hot_date,
        "cold_month": args.cold_month,
        "chunk_start": args.chunk_start,
        "chunk_end": args.chunk_end,
        "checked": len(targets),
        "missing": {"count": 0, "partitions": []},
        "stale": {"count": 0, "partitions": []},
        "layers": {"layer0_roundtrip": None, "layer1_hot": None, "layer2_cold": None},
        "d12_exempt_columns": list(_D12_COLS),
        "verdict": {"pass": False, "conclusion": "未完成"},
        "problems": [],
    }

    def _finalize(passed: bool, conclusion: str) -> int:
        report["verdict"] = {"pass": passed, "conclusion": conclusion}
        out = (
            ROOT / "data" / "reports"
            / f"v2_p1_immutable_partitions_{datetime.now():%Y%m%d}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"报告: {out}")
        return 0 if passed else 1

    before = _fingerprints(store, targets)
    missing = [rel for rel, fp in before.items() if fp is None]
    if missing:
        report["missing"] = {"count": len(missing), "partitions": missing}
        print(f"目标分区未登记 manifest（无法验证）: {missing[:5]}")
        return _finalize(False, f"目标分区未登记 manifest {len(missing)} 个")

    # 层 0：store 级往返 no-op（廉价 sanity）
    _store_roundtrip_noop(store, args.hot_date)
    mid = _fingerprints(store, targets)
    stale0 = [rel for rel in targets if mid[rel] != before[rel]]
    report["layers"]["layer0_roundtrip"] = {
        "stale_count": len(stale0),
        "stale_partitions": stale0,
    }
    if stale0:
        report["stale"] = {"count": len(stale0), "partitions": stale0}
        print("FAIL: store 级往返后指纹变化")
        return _finalize(False, f"store 级往返后指纹变化 {len(stale0)} 个分区")
    logger.info("层 0（store 往返 no-op）通过")

    scratch_store = PanelDataStore(args.scratch_root)
    bootstrap_manifest(scratch_store)

    # 层 1：热区日同上下文重建落 scratch（因果 ⇒ 构建通道逐位一致）
    builder.backfill(
        TradeDate.from_str(args.chunk_start),
        TradeDate.from_str(args.hot_date),
        _GROUPS,
        scratch_store,
        keep_dates={args.hot_date},
    )
    hot_problems = _compare_all(store, scratch_store, hot_rels, "层1-热区日")
    report["layers"]["layer1_hot"] = {"problem_count": len(hot_problems)}
    logger.info(f"层 1（热区日重建 {args.hot_date}）：差异 {len(hot_problems)} 项")

    # 层 2：冷区月同上下文重建落 scratch（终点必须与原分块一致）
    month_dates = [d for d in builder._full_calendar() if f"{d[:4]}-{d[4:6]}" == args.cold_month]
    if not month_dates:
        print(f"冷区月无交易日: {args.cold_month}")
        return _finalize(False, f"冷区月无交易日: {args.cold_month}")
    builder.backfill(
        TradeDate.from_str(args.chunk_start),
        TradeDate.from_str(args.chunk_end),
        _GROUPS,
        scratch_store,
        keep_dates=set(month_dates),
    )
    cold_problems = _compare_all(store, scratch_store, cold_rels, "层2-冷区月")
    report["layers"]["layer2_cold"] = {"problem_count": len(cold_problems)}
    logger.info(f"层 2（冷区月重建 {args.cold_month}）：差异 {len(cold_problems)} 项")

    problems = hot_problems + cold_problems
    report["problems"] = problems
    stale_rels = sorted({p.split("] ", 1)[0].split("/", 1)[-1] for p in problems})
    if problems:
        report["stale"] = {"count": len(stale_rels), "partitions": stale_rels}
        print(f"FAIL: 构建通道差异 {len(problems)} 项")
        for p in problems[:20]:
            print(f"  {p}")
        return _finalize(False, f"构建通道差异 {len(problems)} 项")
    conclusion = (
        f"{len(targets)} 个已写分区内容不变（store 往返 + 热区日 + 冷区月三层重建比对；"
        f"D-12 豁免列 {len(_D12_COLS)} 个按 D-13 登记剔除，证据独立治理）"
    )
    print(f"PASS: {conclusion}")
    return _finalize(True, conclusion)


if __name__ == "__main__":
    sys.exit(main())

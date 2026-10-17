# -*- coding: utf-8 -*-
"""验收门 3/4/5：v2 重放 runs 批次 vs B0 冻结批次比对。

- 门 3 配置指纹：新 summary 逐折 vs 冻结 summary（KEY_* 列小写化后）按
  ``is_fingerprint_key`` 全量键值比对，缺键/多键/错值出清单；``batch_id``
  作为迁移专用标识另列登记放行（逐项打印），不得用子集关系放行；
- 门 4 成交逐笔：9 列（date/signal_date/stock/action/price/shares/amount/
  cost + sell_type，N5 写死）日期规范化后逐折 ``assert_frame_equal(check_exact=True)``
  口径比对 + 并列互换计数 = 0——trades 其余 10 列（运行标识/展示列，含
  wf_run_id/buy_pnl_price 等）不参与门 4；execution_attribution **无条件**
  二次交叉（全列比对，唯一规范化 = wf_run_id 迁移专用运行标识）；
- 门 5 净值：逐折日净值 + 链式净值全表，容差 1e-6（冻结 chain_nav 的 date
  为折内整数序号，按位次比对）。

折集合纪律（R2-T7-03）：默认完整终验模式要求新侧折集 == 冻结折集（缺折/多折
即失败）；仅显式 ``--allow-subset`` 预检允许真子集比对交集，并豁免
wf_start_date/wf_end_date 两子集依赖键。

输出总 PASS/FAIL 与逐门差异清单。本工具读取真实产物，属机器时间任务，
不进 pytest（pytest 以合成批次驱动比对函数）。

P2b-T3 差异登记：新增可选显式参照文件集、模式键期望值与逐名身份规范化
参数，供三边比对复用既有原语；所有参数缺省时 P2a 门 3/4/5 行为不变。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

# 脚本直接运行时把仓库根插入 sys.path（同 scripts/v2_p1、v2_p5a1 既有先例）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lazybull.v2.evidence.fingerprint_keys import is_fingerprint_key
from src.lazybull.v2.evidence.runs_loader import RunsBatch, RunsFold, load_runs_batch

#: 门 4 成交逐笔比对列（N5 写死，旧 CSV 列名口径）
GATE4_TRADE_COLS = [
    "date",
    "signal_date",
    "stock",
    "action",
    "price",
    "shares",
    "amount",
    "cost",
    "sell_type",
]

#: 门 5 净值容差（契约/规划明文 1e-6）
NAV_TOL = 1e-6

#: 迁移专用标识（门 3 另列登记放行；新批次有、冻结批次无）
MIGRATION_ONLY_KEYS = ("batch_id",)


@dataclass
class CompareReport:
    """门 3/4/5 比对报告。"""

    gate3_diffs: List[str] = field(default_factory=list)
    gate4_diffs: List[str] = field(default_factory=list)
    gate5_diffs: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)  # 登记放行项 / 子集说明

    @property
    def ok(self) -> bool:
        return not (self.gate3_diffs or self.gate4_diffs or self.gate5_diffs)


@dataclass(frozen=True)
class RawReferenceFiles:
    """P2b 显式解析后的旧 B0 raw 参照文件集（禁止多命中取最后一份）。"""

    summary: Path
    chain_nav: Path
    trades: Dict[int, Path]
    attribution: Dict[int, Optional[Path]]


def _norm_date_str(value: object) -> object:
    """日期值规范化（YYYYMMDD 字符串）；空值归一 None。"""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None
    digits = text.replace("-", "")
    return digits[:8] if len(digits) >= 8 and digits[:8].isdigit() else text


def _norm_scalar(value: object) -> object:
    """指纹键值规范化：空值归一 None；数值与文本各归其位（bool 先行）。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("nan", "none"):
            return None
        if text in ("True", "False"):
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _values_equal(a: object, b: object) -> bool:
    """规范化后键值相等判定（数值精确相等，bool 不与数值混淆——False≠0.0）。"""
    na, nb = _norm_scalar(a), _norm_scalar(b)
    if na is None or nb is None:
        return na is None and nb is None
    if isinstance(na, bool) != isinstance(nb, bool):
        return False
    if isinstance(na, float) and isinstance(nb, (int, float)):
        return na == float(nb)
    if isinstance(nb, float) and isinstance(na, (int, float)):
        return float(na) == nb
    return na == nb


def _check_new_only_expected(
    new_summary: pd.DataFrame,
    key: str,
    expected: object,
    report: CompareReport,
) -> None:
    """P2b 新侧独有键必须逐折等于预登记期望值（不是无条件豁免）。"""
    bad = [
        int(row["split_index"])
        for _, row in new_summary.iterrows()
        if not _values_equal(row[key], expected)
    ]
    if bad:
        report.gate3_diffs.append(f"门3 新侧独有键 {key} 错值折={bad}，期望={expected!r}")
    else:
        report.notes.append(f"门3 登记放行: 新侧独有键 {key}={expected!r}（逐折核验）")


def _compare_shared_fingerprint_value(
    key: str,
    split: int,
    row_new: pd.Series,
    row_frozen: pd.Series,
    shared_expected_values: Dict[str, Sequence[object]],
    report: CompareReport,
) -> None:
    """共享键普通等值比较，或按 P2b 预登记的前后期望值比较。"""
    if key not in shared_expected_values:
        if not _values_equal(row_new[key], row_frozen[key]):
            report.gate3_diffs.append(
                f"门3 错值: split{split:02d}.{key}: 新={row_new[key]!r} vs "
                f"冻结={row_frozen[key]!r}"
            )
        return
    expected_new, expected_frozen = shared_expected_values[key]
    if not _values_equal(row_new[key], expected_new) or not _values_equal(
        row_frozen[key], expected_frozen
    ):
        report.gate3_diffs.append(
            f"门3 前后期望值不符: split{split:02d}.{key}: 新={row_new[key]!r} "
            f"(期望 {expected_new!r}) vs 冻结={row_frozen[key]!r} (期望 {expected_frozen!r})"
        )


def compare_fingerprint_gate(
    new_summary: pd.DataFrame,
    frozen_summary: pd.DataFrame,
    report: CompareReport,
    *,
    allow_subset: bool = False,
    migration_only_values: Optional[Dict[str, object]] = None,
    shared_expected_values: Optional[Dict[str, Sequence[object]]] = None,
    normalized_shared_keys: Sequence[str] = (),
) -> None:
    """门 3：配置指纹全量键值比对（逐折；缺键/多键/错值出清单）。

    默认参数保持 P2a 门 3 原行为。P2b 可选参数分别表达新侧独有键的期望值、
    共享键的前后期望值，以及已单独完成批内身份核验后的逐名运行身份规范化。
    """
    migration_values = migration_only_values or {}
    expected_values = shared_expected_values or {}
    frozen = frozen_summary.copy()
    frozen = frozen.rename(columns={c: c.lower() for c in frozen.columns if c.startswith("KEY_")})
    frozen = frozen[[c for c in frozen.columns if str(c).isascii()]]

    new_keys = {c for c in new_summary.columns if is_fingerprint_key(c)}
    frozen_keys = {c for c in frozen.columns if is_fingerprint_key(c)}
    for key in sorted(new_keys - frozen_keys):
        if key in MIGRATION_ONLY_KEYS:
            report.notes.append(f"门3 登记放行（迁移专用标识）: 新侧多键 {key}")
        elif key in migration_values:
            _check_new_only_expected(new_summary, key, migration_values[key], report)
        else:
            report.gate3_diffs.append(f"门3 多键: {key}（新侧有、冻结侧无）")
    for key in sorted(frozen_keys - new_keys):
        report.gate3_diffs.append(f"门3 缺键: {key}（冻结侧有、新侧无）")

    frozen_splits = set(frozen["split_index"].astype(int))
    new_splits = set(new_summary["split_index"].astype(int))
    subset_keys: set = set()
    if frozen_splits != new_splits:
        if not allow_subset:
            report.gate3_diffs.append(
                f"门3 折集合不一致: 新 {sorted(new_splits)} vs 冻结 {sorted(frozen_splits)}"
                "——完整终验模式要求折集全等；子集预检须显式 --allow-subset"
            )
        elif not new_splits < frozen_splits:
            report.gate3_diffs.append(
                f"门3 子集模式但新侧折集不是冻结折集的真子集: 新 {sorted(new_splits)} vs "
                f"冻结 {sorted(frozen_splits)}（多出折或互不包含）"
            )
        else:
            report.notes.append(
                f"门3 子集预检模式（显式启用）: 新 {sorted(new_splits)} ⊆ 冻结 "
                f"{sorted(frozen_splits)}；仅比对交集"
            )
            report.notes.append(
                "门3 子集模式：wf_start_date/wf_end_date 为子集依赖键，登记跳过值比对"
            )
            subset_keys = {"wf_start_date", "wf_end_date"}
    shared_keys = (new_keys & frozen_keys) - subset_keys - set(normalized_shared_keys)
    for key in sorted(set(normalized_shared_keys) & new_keys & frozen_keys):
        report.notes.append(f"门3 运行身份规范化（批内身份已先核验）: {key}")
    for split in sorted(frozen_splits & new_splits):
        row_new = new_summary[new_summary["split_index"].astype(int) == split].iloc[0]
        row_frozen = frozen[frozen["split_index"].astype(int) == split].iloc[0]
        for key in sorted(shared_keys):
            _compare_shared_fingerprint_value(
                key, split, row_new, row_frozen, expected_values, report
            )


def _frozen_glob(frozen_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(Path(frozen_dir).glob(pattern))
    return files[-1] if files else None


def _frame_equal_exact(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """逐位精确比对（NaN==NaN 视等；assert_frame_equal(check_exact=True) 口径）。"""
    if a.shape != b.shape:
        return False
    for col in a.columns:
        av, bv = a[col].tolist(), b[col].tolist()
        for x, y in zip(av, bv):
            x_none = x is None or (not isinstance(x, str) and pd.isna(x))
            y_none = y is None or (not isinstance(y, str) and pd.isna(y))
            if x_none or y_none:
                if not (x_none and y_none):
                    return False
            elif x != y:
                return False
    return True


def _count_swapped_ties(a: pd.DataFrame, b: pd.DataFrame) -> Optional[int]:
    """并列互换计数：两帧多重集相等但位次不同的行数；多重集不等返回 None。"""
    key_cols = list(a.columns)
    sa = a.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    sb = b.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    if not _frame_equal_exact(sa, sb):
        return None
    return sum(
        1
        for i in range(len(a))
        if not _frame_equal_exact(
            a.iloc[[i]].reset_index(drop=True), b.iloc[[i]].reset_index(drop=True)
        )
    )


def _gate4_frame(fold_dir_frame: pd.DataFrame, rename: Dict[str, str]) -> pd.DataFrame:
    df = fold_dir_frame.rename(columns=rename).copy()
    for col in ("date", "signal_date"):
        if col in df.columns:
            df[col] = df[col].map(_norm_date_str)
    missing = [c for c in GATE4_TRADE_COLS if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = pd.NA
    return df[GATE4_TRADE_COLS].reset_index(drop=True)


def _compare_attribution(
    fold: "RunsFold", attr_path: Optional[Path], tag: str, report: CompareReport
) -> None:
    """attribution 二次交叉（全列、行序、逐位；唯一规范化 wf_run_id）。"""
    if attr_path is None and fold.attribution is None:
        report.notes.append(f"门4 {tag}: attribution 双侧缺失，登记跳过")
        return
    if attr_path is None or fold.attribution is None:
        report.gate4_diffs.append(
            f"门4 {tag}: attribution 单侧缺失（冻结={'有' if attr_path else '无'} / "
            f"新侧={'有' if fold.attribution is not None else '无'}）"
        )
        return
    frozen_attr = pd.read_csv(attr_path, encoding="utf-8-sig", float_precision="round_trip").rename(
        columns={"planned_stock": "planned_ts_code", "actual_stock": "actual_ts_code"}
    )
    new_attr = fold.attribution.copy()
    for col in ("signal_date", "ranking_date", "execution_date"):
        if col in new_attr.columns:
            new_attr[col] = new_attr[col].map(_norm_date_str)
        if col in frozen_attr.columns:
            frozen_attr[col] = frozen_attr[col].map(_norm_date_str)
    if "wf_run_id" in new_attr.columns and "wf_run_id" in frozen_attr.columns:
        new_attr["wf_run_id"] = "<normalized>"
        frozen_attr["wf_run_id"] = "<normalized>"
        report.notes.append(f"门4 {tag}: attribution wf_run_id 规范化（迁移专用运行标识）")
    new_cols = set(new_attr.columns)
    frozen_cols = set(frozen_attr.columns)
    if new_cols != frozen_cols:
        report.gate4_diffs.append(
            f"门4 {tag}: attribution 列集合不一致——缺 {sorted(frozen_cols - new_cols)}，"
            f"多 {sorted(new_cols - frozen_cols)}"
        )
        return
    cols = sorted(frozen_cols)
    if not _frame_equal_exact(
        frozen_attr[cols].reset_index(drop=True),
        new_attr[cols].reset_index(drop=True),
    ):
        report.gate4_diffs.append(f"门4 {tag}: attribution 全列比对不一致")


def compare_trades_gate(
    new_batch: RunsBatch,
    frozen_dir: Path,
    report: CompareReport,
    *,
    reference_files: Optional[RawReferenceFiles] = None,
) -> None:
    """门 4：成交 9 列逐折逐位比对 + 并列互换计数；attribution 无条件二次交叉。"""
    for split_index, fold in sorted(new_batch.folds.items()):
        tag = f"split{split_index:02d}"
        if reference_files is None:
            frozen_path = _frozen_glob(frozen_dir, f"walk_forward_trades_*_{tag}.csv")
            attr_path = _frozen_glob(frozen_dir, f"walk_forward_execution_attribution_*_{tag}.csv")
        else:
            frozen_path = reference_files.trades.get(split_index)
            attr_path = reference_files.attribution.get(split_index)
        if frozen_path is None:
            report.gate4_diffs.append(f"门4 {tag}: 冻结侧缺 trades CSV")
            continue
        frozen = pd.read_csv(frozen_path, encoding="utf-8-sig", float_precision="round_trip")
        new_frame = _gate4_frame(fold.trades, {"trade_date": "date", "ts_code": "stock"})
        frozen_frame = _gate4_frame(frozen, {})
        if not _frame_equal_exact(new_frame, frozen_frame):
            swapped = _count_swapped_ties(new_frame, frozen_frame)
            if swapped is None:
                report.gate4_diffs.append(f"门4 {tag}: 成交 9 列逐位不一致（多重集亦不等）")
            elif swapped > 0:
                report.gate4_diffs.append(f"门4 {tag}: 并列互换 {swapped} 行（要求 = 0）")
        _compare_attribution(fold, attr_path, tag, report)


def compare_nav_gate(
    new_batch: RunsBatch,
    frozen_dir: Path,
    report: CompareReport,
    *,
    allow_subset: bool = False,
    reference_files: Optional[RawReferenceFiles] = None,
) -> None:
    """门 5：逐折日净值 + 链式净值全表（容差 1e-6；冻结 date 为折内序号，按位次）。

    折集合纪律同门 3（R2-T7-03）：默认完整模式要求折集全等；显式预检才允许
    真子集比对交集。P2b 可传显式参照文件集；缺省沿用 P2a glob 行为。
    """
    chain_path = (
        _frozen_glob(frozen_dir, "chain_nav_*.csv")
        if reference_files is None
        else reference_files.chain_nav
    )
    if chain_path is None:
        report.gate5_diffs.append("门5: 冻结侧缺 chain_nav_*.csv")
        return
    frozen_chain = pd.read_csv(chain_path, encoding="utf-8-sig", float_precision="round_trip")
    new_chain = new_batch.chain_nav

    frozen_splits = {int(s) for s in frozen_chain["split_index"].unique()}
    new_splits = {int(s) for s in new_chain["split_index"].unique()}
    if frozen_splits != new_splits:
        if not allow_subset:
            report.gate5_diffs.append(
                f"门5 折集合不一致: 新 {sorted(new_splits)} vs 冻结 {sorted(frozen_splits)}"
                "——完整终验模式要求折集全等；子集预检须显式 --allow-subset"
            )
        elif not new_splits < frozen_splits:
            report.gate5_diffs.append(
                f"门5 子集模式但新侧折集不是冻结折集的真子集: 新 {sorted(new_splits)} vs "
                f"冻结 {sorted(frozen_splits)}"
            )
        else:
            report.notes.append(
                f"门5 子集预检模式（显式启用）: 新 {sorted(new_splits)} ⊆ 冻结 "
                f"{sorted(frozen_splits)}；仅比对交集"
            )
    for split in sorted(frozen_splits & new_splits):
        seg_new = new_chain[new_chain["split_index"] == split].sort_values("date")
        seg_frozen = frozen_chain[frozen_chain["split_index"] == split].sort_values("date")
        if len(seg_new) != len(seg_frozen):
            report.gate5_diffs.append(
                f"门5 split{split:02d}: 链式净值长度不一致 新={len(seg_new)} vs "
                f"冻结={len(seg_frozen)}"
            )
            continue
        diff = np.abs(seg_new["nav"].to_numpy() - seg_frozen["nav"].to_numpy())
        if float(diff.max(initial=0.0)) > NAV_TOL:
            report.gate5_diffs.append(
                f"门5 split{split:02d}: 链式净值超差 max|Δ|={float(diff.max()):.3e}"
                f"（容差 {NAV_TOL}）"
            )
        # 逐折日净值：新 folds daily/nav（折内未缩放）vs 冻结折段归一
        fold = new_batch.folds.get(split)
        if fold is not None and fold.daily is not None:
            daily_nav = fold.daily.sort_values("trade_date")["nav"].to_numpy()
            frozen_norm = seg_frozen["nav"].to_numpy() / float(seg_frozen["nav"].iloc[0])
            if len(daily_nav) != len(frozen_norm):
                report.gate5_diffs.append(
                    f"门5 split{split:02d}: 逐日净值长度不一致 新={len(daily_nav)} vs "
                    f"冻结={len(frozen_norm)}"
                )
            else:
                ddiff = np.abs(daily_nav - frozen_norm)
                if float(ddiff.max(initial=0.0)) > NAV_TOL:
                    report.gate5_diffs.append(
                        f"门5 split{split:02d}: 逐日净值超差 max|Δ|={float(ddiff.max()):.3e}"
                    )


def compare_replay_vs_b0(
    new_batch_dir: Path,
    frozen_dir: Path,
    *,
    allow_subset: bool = False,
    reference_files: Optional[RawReferenceFiles] = None,
    migration_only_values: Optional[Dict[str, object]] = None,
    shared_expected_values: Optional[Dict[str, Sequence[object]]] = None,
    normalized_shared_keys: Sequence[str] = (),
) -> CompareReport:
    """门 3/4/5 总入口：新 runs 批目录 + B0 冻结批 raw 目录 → 比对报告。

    缺省调用完全保持 P2a 行为。P2b 通过可选参数注入显式参照文件集与预登记
    模式键规则，复用同一指纹、成交、归因和净值比较原语。
    """
    report = CompareReport()
    new_batch = load_runs_batch(Path(new_batch_dir))
    frozen_dir = Path(frozen_dir)
    summary_path = (
        _frozen_glob(frozen_dir, "walk_forward_summary_*.csv")
        if reference_files is None
        else reference_files.summary
    )
    if summary_path is None:
        report.gate3_diffs.append("门3: 冻结侧缺 walk_forward_summary_*.csv")
    else:
        frozen_summary = pd.read_csv(
            summary_path, encoding="utf-8-sig", float_precision="round_trip"
        )
        compare_fingerprint_gate(
            new_batch.summary,
            frozen_summary,
            report,
            allow_subset=allow_subset,
            migration_only_values=migration_only_values,
            shared_expected_values=shared_expected_values,
            normalized_shared_keys=normalized_shared_keys,
        )
    compare_trades_gate(new_batch, frozen_dir, report, reference_files=reference_files)
    compare_nav_gate(
        new_batch,
        frozen_dir,
        report,
        allow_subset=allow_subset,
        reference_files=reference_files,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="验收门 3/4/5：v2 重放 vs B0 冻结比对")
    parser.add_argument("--new-batch", required=True, help="新 runs 批次目录")
    parser.add_argument(
        "--frozen-dir",
        default="data/walk_forward/batches/wf_batch_20260930_171221/raw",
        help="B0 冻结批次 raw 目录",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="显式预检模式：允许新侧折集为冻结折集真子集（默认完整终验要求折集全等）",
    )
    args = parser.parse_args(argv)

    report = compare_replay_vs_b0(
        Path(args.new_batch), Path(args.frozen_dir), allow_subset=args.allow_subset
    )
    for note in report.notes:
        logger.info(f"[登记] {note}")
    for gate, diffs in (
        ("门3 配置指纹", report.gate3_diffs),
        ("门4 成交逐笔", report.gate4_diffs),
        ("门5 净值", report.gate5_diffs),
    ):
        if diffs:
            logger.error(f"{gate} 差异（{len(diffs)} 项）:")
            for d in diffs:
                logger.error(f"  {d}")
    if report.ok:
        logger.info("门 3/4/5 总 PASS")
        return 0
    logger.error("门 3/4/5 总 FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())

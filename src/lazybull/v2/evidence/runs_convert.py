# -*- coding: utf-8 -*-
"""旧 walk-forward 产物 → runs 产物契约（F2）的一次性转换器。

契约依据（`docs/contracts/runs_artifact_contract.md` F2）：
- **数值逐位不动**：只做目录搬迁 / 重命名 / 字段名映射 / 编码规范化；
  禁止任何数值重算（lot_id / daily.csv 为标注性重建，例外已注明）；
- 转换报告：行数校验（源 vs 目标）+ 字段映射表 + **丢弃列清单**（契约 §9 通则 7）
  + 抽样 md5（每文件 ≥3 行）一并产出；
- 文件缺失三态（契约 §9 通则）：trades / summary / chain_nav / batch_meta 必须存在；
  daily 可重建；topk_detail 可缺但不可重建（缺则标 `topk_detail.missing=true`）。

验收：P5a-1 用 3 个已登记历史实验（holdertrade A2 / repurchase / top10fh）
重算与既有报表逐项一致——同时验收转换器与证据机器读入链路。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

# ---- 字段映射（契约 F2 §5 / §6 / §7 / §7.1；源列名 → 目标列名） ----

# 持仓快照：中文表头 → ASCII（契约 §7）
_HOLDINGS_COLS_ZH2EN = {
    "运行标识": "run_id",
    "折序号": "split_index",
    "模型版本": "model_version",
    "日期": "trade_date",
    "股票代码": "ts_code",
    "持仓股数": "shares",
    "持仓市值": "market_value",
    "持仓权重": "weight",
    "组合总值": "total_value",
    "买入 日": "buy_date",  # 源表头有空格（实测）
    "买入日": "buy_date",
    "信号日": "signal_date",
    "持有交易日数": "held_days",
    "到期执行日": "due_date",
    "剩余持有交易日": "remaining_days",
}

# trades：源 stock→ts_code、date→trade_date（契约 §5）
_TRADES_RENAME = {"stock": "ts_code", "date": "trade_date"}

# attribution：源 planned_stock/actual_stock → planned_ts_code/actual_ts_code（契约 §6）
_ATTR_RENAME = {"planned_stock": "planned_ts_code", "actual_stock": "actual_ts_code"}

# summary：KEY_* 前缀大写 → 小写蛇形（契约 §3 信号层口径）
_SUMMARY_KEY_PREFIX = "KEY_"


@dataclass
class ConvertedFile:
    """单文件转换记录。"""

    source: str
    target: str
    rows_in: int
    rows_out: int
    mapping: Dict[str, str] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    sample_md5: str = ""  # 抽样 md5（前 3 行）


@dataclass
class ConvertReport:
    """批次转换报告。"""

    batch_id: str
    source_dir: str
    target_dir: str
    files: List[ConvertedFile] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    meta_notes: Dict[str, str] = field(default_factory=dict)
    row_check_pass: bool = True
    errors: List[str] = field(default_factory=list)


def _sample_md5(df: pd.DataFrame, n: int = 3) -> str:
    head = df.head(n).to_csv(index=False).encode("utf-8")
    return hashlib.md5(head).hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _convert_one(
    src: Path,
    dst: Path,
    rename: Dict[str, str],
    report_files: List[ConvertedFile],
    keep_only: Optional[List[str]] = None,
) -> Optional[ConvertedFile]:
    """单文件转换：读源 → 改名 → 列级映射/丢弃登记 → 写目标（parquet）。"""
    if not src.exists():
        return None
    df = pd.read_csv(src, encoding="utf-8-sig")
    rows_in = len(df)
    mapped = {c: rename.get(c, c) for c in df.columns}
    df = df.rename(columns=rename)
    if keep_only is not None:
        keep = [c for c in keep_only if c in df.columns]
        dropped = [c for c in df.columns if c not in keep]
        df = df[keep]
    else:
        dropped = []
    _write_parquet(df, dst)
    rec = ConvertedFile(
        source=src.name,
        target=str(dst.name),
        rows_in=rows_in,
        rows_out=len(df),
        mapping=mapped,
        dropped_columns=dropped,
        sample_md5=_sample_md5(df),
    )
    report_files.append(rec)
    return rec


def convert_wf_batch(
    source_dir: Path,
    target_root: Path,
    batch_id: str,
) -> ConvertReport:
    """把一个旧 WF 批次目录转换为 runs 契约 schema。

    Args:
        source_dir: 旧批次 raw 目录（含 chain_nav_* / walk_forward_* / data_state_*）
        target_root: runs 根目录（产出落 target_root/<batch_id>/）
        batch_id: 新批次 ID（ASCII）
    """
    source_dir = Path(source_dir)
    out_dir = Path(target_root) / batch_id
    folds_dir = out_dir / "folds"
    report = ConvertReport(
        batch_id=batch_id, source_dir=str(source_dir), target_dir=str(out_dir)
    )

    # ---- 必需文件（缺即批次非法，契约 §9 三态=必须报错） ----
    chain_files = sorted(source_dir.glob("chain_nav_*.csv"))
    summary_files = sorted(source_dir.glob("walk_forward_summary_*.csv"))
    if not chain_files:
        report.missing_required.append("chain_nav")
        report.errors.append("缺 chain_nav_*.csv")
    if not summary_files:
        report.missing_required.append("summary")
        report.errors.append("缺 walk_forward_summary_*.csv")
    if report.errors:
        report.row_check_pass = False
        return report

    # chain_nav：数值逐位不动（契约 §4）
    chain = pd.read_csv(chain_files[-1], encoding="utf-8-sig")
    _write_parquet(chain, out_dir / "chain_nav.parquet")
    report.files.append(
        ConvertedFile(
            source=chain_files[-1].name,
            target="chain_nav.parquet",
            rows_in=len(chain),
            rows_out=len(chain),
            sample_md5=_sample_md5(chain),
        )
    )

    # summary：KEY_* → 小写蛇形（契约 §3）
    summary = pd.read_csv(summary_files[-1], encoding="utf-8-sig")
    summary_rename = {
        c: c.lower() for c in summary.columns if c.startswith(_SUMMARY_KEY_PREFIX)
    }
    summary = summary.rename(columns=summary_rename)
    _write_parquet(summary, out_dir / "summary.parquet")
    report.files.append(
        ConvertedFile(
            source=summary_files[-1].name,
            target="summary.parquet",
            rows_in=len(summary),
            rows_out=len(summary),
            mapping={c: summary_rename.get(c, c) for c in summary.columns},
            dropped_columns=[],
            sample_md5=_sample_md5(summary),
        )
    )

    # ---- 逐折文件 ----
    split_ids = sorted(
        {
            p.stem.split("_split")[-1]
            for p in source_dir.glob("walk_forward_trades_*_split*.csv")
        }
    )
    if not split_ids:
        report.missing_required.append("trades")
        report.errors.append("缺 walk_forward_trades_*_split*.csv")
        report.row_check_pass = False

    n_topk = 0
    for split_id in split_ids:
        fold_dir = folds_dir / f"split{split_id}"
        # trades（契约 §5，含 F2 补 4 列 sell_reason/trigger_type/buy_type/buy_reason 随源带入）
        _convert_one(
            source_dir / f"walk_forward_trades_{chain_files[-1].stem.split('chain_nav_')[-1]}_split{split_id}.csv"
            if len(list(source_dir.glob(f"walk_forward_trades_*_split{split_id}.csv"))) > 1
            else sorted(source_dir.glob(f"walk_forward_trades_*_split{split_id}.csv"))[-1],
            fold_dir / "trades.parquet",
            _TRADES_RENAME,
            report.files,
        )
        # attribution（契约 §6）
        attr_files = sorted(source_dir.glob(f"walk_forward_execution_attribution_*_split{split_id}.csv"))
        if attr_files:
            _convert_one(attr_files[-1], fold_dir / "attribution.parquet", _ATTR_RENAME, report.files)
        # 持仓快照（契约 §7，中文表头映射）
        snap_files = sorted(source_dir.glob(f"walk_forward_持仓快照_*_split{split_id}.csv"))
        if snap_files:
            _convert_one(snap_files[-1], fold_dir / "holdings_snapshot.parquet", _HOLDINGS_COLS_ZH2EN, report.files)
        # topk_detail（契约 §7.1；可缺但不可重建）
        topk_files = sorted(source_dir.glob(f"walk_forward_topk_details_*_split{split_id}.csv"))
        if topk_files:
            n_topk += 1
            _convert_one(topk_files[-1], fold_dir / "topk_detail.parquet", {}, report.files)

    if n_topk == 0 and split_ids:
        # 契约 §7.1：缺失即降级标注（证据机器侧该折不参与信号层尺子）
        report.meta_notes["topk_detail.missing"] = "true"
        logger.warning(f"{batch_id}: 全折缺 topk_detail，标注降级（基线批常见）")

    # ---- batch_meta.json（契约 §2，数据态从 data_state_*.json 带入） ----
    data_state_files = sorted(source_dir.glob("data_state_*.json"))
    data_state: Dict[str, object] = {}
    if data_state_files:
        try:
            data_state = json.loads(data_state_files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.meta_notes["data_state_parse_error"] = str(exc)

    batch_meta = {
        "schema_version": 1,
        "batch_id": batch_id,
        "source": "converted_from_legacy_wf_batch",
        "source_dir": str(source_dir),
        "data_state": data_state,
        "meta_notes": report.meta_notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 行数校验（源 vs 目标）
    for rec in report.files:
        if rec.rows_in != rec.rows_out:
            report.row_check_pass = False
            report.errors.append(f"{rec.source}: 行数不一致 {rec.rows_in}→{rec.rows_out}")

    return report

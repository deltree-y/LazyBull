# -*- coding: utf-8 -*-
"""旧 walk-forward 产物 → runs 产物契约（F2）的一次性转换器。

契约依据（`docs/contracts/runs_artifact_contract.md` F2）：
- **数值逐位不动**：只做目录搬迁 / 重命名 / 字段名映射 / 编码规范化 / 列集对齐
  （多列丢弃、缺列补 NA 并登记 contract_missing；lot_id / daily.csv 为标注性重建，
  例外已注明）；
- 转换报告：行数校验（源 vs 目标）+ 字段映射表 + **丢弃列清单**（契约 §9 通则 7）
  + 缺列登记 + 抽样 md5（每文件 ≥3 行），一并落盘
  `<target_root>/<batch_id>.convert_report.json`（批次目录外，保持 runs 目录契约纯净）；
- 文件缺失三态（契约 §9 通则）：trades / summary / chain_nav / batch_meta 必须存在；
  daily 可重建（无源标 `daily.missing=true`）；topk_detail 可缺但不可重建
  （缺则标 `topk_detail.missing=true`）。

契约列集与配置指纹排除清单分别从 `runs_schema.py` / `fingerprint_keys.py` 导入
（单一来源，禁止重写）。

验收：P5a-1 用 3 个已登记历史实验（holdertrade A2 / repurchase / top10fh）
重算与既有报表逐项一致——同时验收转换器与证据机器读入链路。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.lazybull.v2.evidence.fingerprint_keys import config_fingerprint_keys, fingerprint
from src.lazybull.v2.evidence.runs_schema import (
    ATTR_CONTRACT_COLS,
    SNAP_CONTRACT_COLS,
    TOPK_CONTRACT_COLS,
    TRADES_CONTRACT_COLS,
)

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

#: 已知实验臂开关 → 臂标签（summary 配置列值为真时计入 batch_meta.arms）
_ARM_SWITCHES = {
    "enable_holdertrade_features": "holdertrade",
    "enable_repurchase_features": "repurchase",
    "enable_top10fh_features": "top10fh",
    "enable_top_inst_features": "top_inst",
}


@dataclass
class ConvertedFile:
    """单文件转换记录。"""

    source: str
    target: str
    rows_in: int
    rows_out: int
    mapping: Dict[str, str] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    contract_missing: List[str] = field(default_factory=list)  # 契约缺列（已补 NA，登记不阻断）
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


def _norm_date_key(s: pd.Series) -> pd.Series:
    """日期键规范化（YYYYMMDD 字符串）：兼容 str / int / Timestamp / ISO 日期。"""
    return s.astype(str).str.replace("-", "", regex=False).str[:8]


def _convert_one(
    src: Path,
    dst: Path,
    rename: Dict[str, str],
    report_files: List[ConvertedFile],
    contract_cols: Optional[List[str]] = None,
) -> Optional[ConvertedFile]:
    """单文件转换：读源 → 改名 → 列集对齐（多列丢弃 / 缺列补 NA）→ 写目标（parquet）。

    Args:
        contract_cols: 契约列集（契约 §9.7 / §10）；给出时目标列集合严格等于契约集合，
            多列进 dropped_columns，缺列补 NA 并进 contract_missing 登记。
    """
    if not src.exists():
        return None
    df = pd.read_csv(src, encoding="utf-8-sig")
    rows_in = len(df)
    mapped = {c: rename.get(c, c) for c in df.columns}  # rename 前构造，保留真实映射
    df = df.rename(columns=rename)
    dropped: List[str] = []
    contract_missing: List[str] = []
    if contract_cols is not None:
        dropped = [c for c in df.columns if c not in contract_cols]
        if dropped:
            df = df[[c for c in df.columns if c in contract_cols]]
        contract_missing = [c for c in contract_cols if c not in df.columns]
        for c in contract_missing:
            df[c] = pd.NA  # 缺列补 NA（schema 对齐；源缺事实进 contract_missing）
        df = df[[c for c in contract_cols]]  # 列序对齐契约
    _write_parquet(df, dst)
    rec = ConvertedFile(
        source=src.name,
        target=str(dst.name),
        rows_in=rows_in,
        rows_out=len(df),
        mapping=mapped,
        dropped_columns=dropped,
        contract_missing=contract_missing,
        sample_md5=_sample_md5(df),
    )
    report_files.append(rec)
    return rec


def convert_wf_batch(
    source_dir: Path,
    target_root: Path,
    batch_id: str,
    baseline_ref: Optional[str] = None,
) -> ConvertReport:
    """把一个旧 WF 批次目录转换为 runs 契约 schema。

    Args:
        source_dir: 旧批次 raw 目录（含 chain_nav_* / walk_forward_* / data_state_*）
        target_root: runs 根目录（产出落 target_root/<batch_id>/）
        batch_id: 新批次 ID（ASCII）
        baseline_ref: 对照批次 id（契约 §2：A/B 实验臂必填，基线批为 None）
    """
    source_dir = Path(source_dir)
    out_dir = Path(target_root) / batch_id
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
        _write_report(report, target_root)
        return report

    # chain_nav：数值逐位不动（契约 §4）
    chain = pd.read_csv(chain_files[-1], encoding="utf-8-sig")
    _write_parquet(chain, out_dir / "chain_nav.parquet")
    report.files.append(
        ConvertedFile(
            source=chain_files[-1].name, target="chain_nav.parquet",
            rows_in=len(chain), rows_out=len(chain), sample_md5=_sample_md5(chain),
        )
    )

    summary = _convert_summary_block(summary_files[-1], batch_id, out_dir, report)

    # ---- 逐折文件 ----
    split_ids = sorted(
        {p.stem.split("_split")[-1]
         for p in source_dir.glob("walk_forward_trades_*_split*.csv")}
    )
    if not split_ids:
        report.missing_required.append("trades")
        report.errors.append("缺 walk_forward_trades_*_split*.csv")
        report.row_check_pass = False
        _write_report(report, target_root)
        return report  # 无折则直接返回（契约 §9 三态=必须报错）

    # policy_lambda 台账（契约 §8.1 条件条款）：fold 循环前读取，供 daily 的 λ 列按日 join
    lambda_by_fold = _read_policy_lambda(source_dir, out_dir, report)

    n_topk = 0
    folds_missing_topk: List[str] = []
    for split_id in split_ids:
        has_topk = _convert_fold(source_dir, split_id, out_dir, lambda_by_fold, batch_id, report)
        if has_topk is True:
            n_topk += 1
        elif has_topk is False:
            folds_missing_topk.append(split_id)

    if n_topk == 0 and split_ids:
        # 契约 §7.1：缺失即降级标注（证据机器侧该折不参与信号层尺子）
        report.meta_notes["topk_detail.missing"] = "true"
        report.meta_notes["topk_detail.missing_folds"] = ",".join(folds_missing_topk)
        logger.warning(f"{batch_id}: 全折缺 topk_detail，标注降级（基线批常见）")

    batch_meta = _build_batch_meta(source_dir, batch_id, summary, baseline_ref, report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # 行数校验（源 vs 目标）
    for rec in report.files:
        if rec.rows_in != rec.rows_out:
            report.row_check_pass = False
            report.errors.append(f"{rec.source}: 行数不一致 {rec.rows_in}→{rec.rows_out}")

    _write_report(report, target_root)
    return report


def _convert_summary_block(
    summary_file: Path, batch_id: str, out_dir: Path, report: ConvertReport
) -> pd.DataFrame:
    """summary：KEY_* → 小写蛇形 + 非 ASCII 列进丢弃清单（契约 §3/§9.5）+ batch_id 补列。"""
    summary = pd.read_csv(summary_file, encoding="utf-8-sig")
    # 映射表在 rename 前构造（源列名 → 目标列名，契约 §10 审计可对账）
    summary_mapping = {
        c: (c.lower() if c.startswith(_SUMMARY_KEY_PREFIX) else c) for c in summary.columns
    }
    summary_rename = {k: v for k, v in summary_mapping.items() if k != v}
    # 非 ASCII 列（中文诊断列）进丢弃清单——诊断展示口径，不进裁决 schema
    summary_dropped = [c for c in summary.columns if not str(c).isascii()]
    summary = summary.rename(columns=summary_rename)
    summary = summary[[c for c in summary.columns if str(c).isascii()]].copy()
    # 补 batch_id 标识列（契约 §3）
    summary["batch_id"] = batch_id
    _write_parquet(summary, out_dir / "summary.parquet")
    report.files.append(
        ConvertedFile(
            source=summary_file.name,
            target="summary.parquet",
            rows_in=len(summary),
            rows_out=len(summary),
            mapping=summary_mapping,
            dropped_columns=summary_dropped,
            sample_md5=_sample_md5(summary),
        )
    )
    return summary


def _convert_fold(
    source_dir: Path,
    split_id: str,
    out_dir: Path,
    lambda_by_fold: Dict[int, Dict[str, float]],
    batch_id: str,
    report: ConvertReport,
) -> Optional[bool]:
    """单折转换（trades/attribution/快照/topk/daily + 折级 _meta.json）。

    Returns:
        True/False = 该折有/无 topk_detail；None = 缺 trades（已记 error，跳过本折）。
    """
    fold_dir = out_dir / "folds" / f"split{split_id}"
    fold_meta: Dict[str, object] = {}
    # trades（契约 §9 三态=必须报错；不静默跳过）
    trades_files = sorted(source_dir.glob(f"walk_forward_trades_*_split{split_id}.csv"))
    if not trades_files:
        report.errors.append(f"split{split_id}: 缺 trades（契约三态=必须报错）")
        report.row_check_pass = False
        return None
    _convert_one(
        trades_files[-1], fold_dir / "trades.parquet", _TRADES_RENAME,
        report.files, contract_cols=TRADES_CONTRACT_COLS,
    )
    # lot_id 重建（契约 §5 例外条款）：FIFO + 卖出行自带 buy_date 交叉校验
    trades_df = pd.read_parquet(fold_dir / "trades.parquet")
    trades_df, fifo_violations = rebuild_lot_id(trades_df)
    _write_parquet(trades_df, fold_dir / "trades.parquet")
    fold_meta["trades.lot_reconstructed"] = True
    if fifo_violations:
        fold_meta["lot_id.fifo_violations"] = fifo_violations
        logger.warning(
            f"{batch_id}/split{split_id}: lot_id FIFO 重建与卖出行 buy_date "
            f"不一致 {fifo_violations} 行（引擎非 FIFO 卖出），归属按 FIFO 口径"
        )
    # attribution（契约 §6）
    attr_files = sorted(
        source_dir.glob(f"walk_forward_execution_attribution_*_split{split_id}.csv")
    )
    if attr_files:
        _convert_one(
            attr_files[-1], fold_dir / "attribution.parquet", _ATTR_RENAME,
            report.files, contract_cols=ATTR_CONTRACT_COLS,
        )
    # 持仓快照（契约 §7，中文表头映射）
    snap_files = sorted(source_dir.glob(f"walk_forward_持仓快照_*_split{split_id}.csv"))
    snap_df = None
    if snap_files:
        _convert_one(
            snap_files[-1], fold_dir / "holdings_snapshot.parquet",
            _HOLDINGS_COLS_ZH2EN, report.files, contract_cols=SNAP_CONTRACT_COLS,
        )
        snap_df = pd.read_parquet(fold_dir / "holdings_snapshot.parquet")
    # topk_detail（契约 §7.1；可缺但不可重建——折级降级标注）
    topk_files = sorted(source_dir.glob(f"walk_forward_topk_details_*_split{split_id}.csv"))
    has_topk = bool(topk_files)
    if has_topk:
        _convert_one(
            topk_files[-1], fold_dir / "topk_detail.parquet", {},
            report.files, contract_cols=TOPK_CONTRACT_COLS,
        )
    else:
        fold_meta["topk_detail.missing"] = True
    # daily.csv 重建（契约 §9 三态=可缺+可重建）：从 trades + 快照重建；无源标注
    daily_df = _rebuild_daily(
        trades_df, snap_df, lambda_by_fold.get(int(split_id)) if split_id.isdigit() else None
    )
    if daily_df is not None:
        _write_parquet(daily_df, fold_dir / "daily.parquet")
        fold_meta["daily.daily_reconstructed"] = True
    else:
        fold_meta["daily.missing"] = True  # 可缺+可重建但无源：标注降级（同 topk 口径）
    # 折级 _meta.json（契约 §5.1/§7.1/§9.6）
    if fold_meta:
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "_meta.json").write_text(
            json.dumps(fold_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return has_topk


def _build_batch_meta(
    source_dir: Path,
    batch_id: str,
    summary: pd.DataFrame,
    baseline_ref: Optional[str],
    report: ConvertReport,
) -> Dict[str, object]:
    """batch_meta.json 组装（契约 §2 全字段；指纹 = 附录 A 排除清单后的全键）。"""
    data_state_files = sorted(source_dir.glob("data_state_*.json"))
    data_state: Dict[str, object] = {}
    if data_state_files:
        try:
            data_state = json.loads(data_state_files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.meta_notes["data_state_parse_error"] = str(exc)

    config: Dict[str, object] = {}
    code_state: Dict[str, object] = {}
    arms: List[str] = []
    if not summary.empty:
        row = summary.iloc[-1]
        for k in ("git_commit", "git_dirty"):
            if k in summary.columns:
                code_state[k] = str(row[k]) if pd.notna(row[k]) else None
        fp_keys = config_fingerprint_keys(summary.columns.tolist())
        config = {k: (row[k].item() if hasattr(row[k], "item") else row[k])
                  for k in fp_keys if k in summary.columns and pd.notna(row[k])}
        for switch, arm in _ARM_SWITCHES.items():
            if switch in summary.columns and pd.notna(row[switch]) and bool(row[switch]):
                arms.append(arm)
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": pd.Timestamp.now().isoformat(),
        "host_mode": "backtest",  # WF OOS 回测批
        "source": "converted_from_legacy_wf_batch",
        "source_dir": str(source_dir),
        "code_state": code_state,
        "data_state": data_state,
        "config": config,
        "config_fingerprint": fingerprint(config),
        "baseline_ref": baseline_ref,  # 契约 §2：A/B 实验臂必填，基线批为 null
        "arms": arms,
        "meta_notes": report.meta_notes,
    }


def _write_report(report: ConvertReport, target_root: Path) -> None:
    """转换报告落盘（契约 §10 审计产物；批次目录外，不污染 runs schema 目录）。"""
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    (target_root / f"{report.batch_id}.convert_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _read_policy_lambda(
    source_dir: Path, out_dir: Path, report: ConvertReport
) -> Dict[int, Dict[str, float]]:
    """收编 policy_lambda 台账（契约 §8.1 条件条款），返回 {折号: {YYYYMMDD: multiplier}}。"""
    lambda_by_fold: Dict[int, Dict[str, float]] = {}
    policy_lambda_files = sorted(source_dir.glob("policy_lambda_*.csv"))
    if not policy_lambda_files:
        return lambda_by_fold
    pl = pd.read_csv(policy_lambda_files[-1], encoding="utf-8-sig")
    _write_parquet(pl, out_dir / "policy_lambda.parquet")
    report.files.append(
        ConvertedFile(
            source=policy_lambda_files[-1].name, target="policy_lambda.parquet",
            rows_in=len(pl), rows_out=len(pl), sample_md5=_sample_md5(pl),
        )
    )
    json_files = sorted(source_dir.glob("policy_lambda_*.json"))
    if json_files:
        (out_dir / "policy_lambda.json").write_text(
            json_files[-1].read_text(encoding="utf-8"), encoding="utf-8"
        )
    if {"date", "multiplier", "fold"} <= set(pl.columns):
        pl = pl.copy()
        pl["date_key"] = _norm_date_key(pl["date"])
        for fold_no, grp in pl.groupby("fold"):
            lambda_by_fold[int(fold_no)] = dict(
                zip(grp["date_key"], grp["multiplier"].astype(float))
            )
    else:
        report.meta_notes["policy_lambda.schema_unexpected"] = (
            f"缺 date/multiplier/fold 列: {pl.columns.tolist()[:10]}"
        )
        logger.warning(f"policy_lambda 台账列不符契约 §8.1，λ 不 join 进 daily: {pl.columns.tolist()[:10]}")
    return lambda_by_fold


# ---- lot_id / daily 重建（契约 §5 例外条款 / §9 三态） ----


def consume_lots_fifo(
    lots: List[Dict[str, object]], shares: float, sell_buy_date: object
) -> Tuple[List[str], bool]:
    """FIFO 消耗 lot 栈，返回 (消耗的 lot_id 列表, 是否与卖出行 buy_date 不一致)。"""
    remaining = shares
    consumed: List[str] = []
    first_lot_buy_date = None
    while remaining > 0 and lots:
        head = lots[0]
        if first_lot_buy_date is None:
            first_lot_buy_date = head["buy_date"]
        take = min(head["shares"], remaining)
        consumed.append(head["lot_id"])
        head["shares"] -= take
        remaining -= take
        if head["shares"] <= 0:
            lots.pop(0)
    violation = (
        first_lot_buy_date is not None
        and pd.notna(sell_buy_date)
        and pd.notna(first_lot_buy_date)
        and str(sell_buy_date) != str(first_lot_buy_date)
    )
    return consumed, violation


def rebuild_lot_id(trades: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """按 FIFO 从 trades 重建 lot_id（v2 新增列，契约 §5），并交叉校验卖出行 buy_date。

    规则：每次 buy 开新 lot（lot_id = buy_date_序号）；sell 按最早买入日优先消耗；
    卖出行回填被消耗的 lot_id（多 lot 消耗时用「|」连接）。
    交叉校验：卖出行自带 buy_date（引擎口径的真实买入日）与 FIFO 首消耗 lot 的
    buy_date 不一致即计数（引擎到期/止损/止盈/补位卖出非严格 FIFO）——
    返回 (trades, 不一致行数)，由调用方登记 `_meta.json: lot_id.fifo_violations`。
    """
    if "ts_code" not in trades.columns or "action" not in trades.columns:
        return trades, 0
    trades = trades.copy()
    trades["lot_id"] = ""
    violations = 0
    # 按 ts_code + trade_date 排序，逐股维护 lot 栈
    for ts_code, grp in trades.groupby("ts_code"):
        lots: List[Dict[str, object]] = []  # {lot_id, shares, buy_date}
        for idx, row in grp.sort_values("trade_date").iterrows():
            if row["action"] == "buy":
                lot_id = f"{row.get('buy_date', row['trade_date'])}_{len(lots)}"
                lots.append(
                    {"lot_id": lot_id, "shares": row["shares"], "buy_date": row.get("buy_date")}
                )
                trades.at[idx, "lot_id"] = lot_id
            elif row["action"] == "sell":
                consumed, violation = consume_lots_fifo(lots, row["shares"], row.get("buy_date"))
                violations += int(violation)
                trades.at[idx, "lot_id"] = "|".join(consumed) if consumed else ""
    return trades, violations


# 向后兼容别名（P2a-T7：lot_id 重建提为公开名供 hosts/backtest/runs_writer 复用；
# 旧私有名保留为同一函数对象别名，零行为变化）
_consume_lots_fifo = consume_lots_fifo
_rebuild_lot_id = rebuild_lot_id


def _rebuild_daily(
    trades: Optional[pd.DataFrame],
    snapshot: Optional[pd.DataFrame],
    lambda_series: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]:
    """从 trades + 快照重建 daily.csv（契约 §8）。

    Args:
        lambda_series: 政策层 λ 台账（{YYYYMMDD: multiplier}，来自 policy_lambda）；
            给出时 exposure_lambda 按日 join（台账未覆盖日按未启用口径 1.0），
            无台账恒 1.0（契约 §8：未启用恒 1.0）。
    """
    if snapshot is None:
        return None
    daily = snapshot.groupby("trade_date").agg(
        total_value=("total_value", "last"),
        market_value=("market_value", "sum"),
        n_positions=("ts_code", "nunique"),
    ).reset_index()
    # 现金 = 总值 − 持仓市值
    daily["cash"] = daily["total_value"] - daily["market_value"]
    # 当日买卖计数（从 trades）
    if trades is not None and not trades.empty:
        buys = trades[trades["action"] == "buy"].groupby("trade_date").size()
        sells = trades[trades["action"] == "sell"].groupby("trade_date").size()
        daily["n_buys"] = daily["trade_date"].map(buys).fillna(0).astype(int)
        daily["n_sells"] = daily["trade_date"].map(sells).fillna(0).astype(int)
        daily["turnover_amount"] = daily["trade_date"].map(
            trades.groupby("trade_date")["amount"].sum()
        ).fillna(0.0)
    else:
        daily["n_buys"] = 0
        daily["n_sells"] = 0
        daily["turnover_amount"] = 0.0
    # 折内净值与收益
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily["nav"] = daily["total_value"] / daily["total_value"].iloc[0]
    daily["daily_return"] = daily["nav"].pct_change().fillna(0.0)
    if lambda_series:
        daily["exposure_lambda"] = (
            _norm_date_key(daily["trade_date"]).map(lambda_series).fillna(1.0)
        )
    else:
        daily["exposure_lambda"] = 1.0  # 无政策层台账恒 1.0（契约 §8）
    return daily

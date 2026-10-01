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

# trades 契约列集（契约 §5 + F2 补 4 列；R-06-#7 列集校验）
_TRADES_CONTRACT_COLS = [
    "wf_run_id", "split_index", "model_version", "trade_date", "signal_date",
    "ts_code", "action", "price", "shares", "amount", "cost",
    "buy_date", "buy_price", "buy_pnl_price", "sell_pnl_price",
    "pnl_profit_amount", "pnl_profit_pct", "sell_type", "sell_timing",
    "sell_reason", "trigger_type", "buy_type", "buy_reason",  # F2 补（历史批可缺）
    "lot_id", "tranche_idx",  # v2 新增
]

# attribution 契约列集（契约 §6，含 F2 补头部三列）
_ATTR_CONTRACT_COLS = [
    "wf_run_id", "split_index", "model_version", "signal_date", "ranking_date",
    "execution_date", "execution_stage", "tranche_idx", "planned_ts_code",
    "actual_ts_code", "planned_rank", "actual_rank", "pred_score", "target_weight",
    "status", "reason", "buy_price", "signal_price", "signal_to_buy_return",
]

# topk_detail 契约列集（契约 §7.1）
_TOPK_CONTRACT_COLS = [
    "wf_run_id", "split_index", "test_start", "test_end", "model_version",
    "trade_date", "topk", "rank", "ts_code", "pred_score", "true_return",
    "score_column", "ml_score", "risk_score", "final_score",
]

# 快照契约列集（契约 §7，中文映射后的 ASCII 列）
_SNAP_CONTRACT_COLS = [
    "run_id", "split_index", "model_version", "trade_date", "ts_code",
    "shares", "market_value", "weight", "total_value", "buy_date",
    "signal_date", "held_days", "due_date", "remaining_days",
]


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
    contract_cols: Optional[List[str]] = None,
    strict_contract: bool = False,
) -> Optional[ConvertedFile]:
    """单文件转换：读源 → 改名 → 列级映射/丢弃登记 → 写目标（parquet）。

    Args:
        contract_cols: 契约列集（R-06-#7 列级校验）；给出时校验缺列/多列并登记
        strict_contract: True 时列集合不一致报错；False 时记 dropped/missing 并继续
    """
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
    # 列集合校验（R-06-#7）
    contract_missing: List[str] = []
    contract_extra: List[str] = []
    if contract_cols is not None:
        contract_missing = [c for c in contract_cols if c not in df.columns]
        contract_extra = [c for c in df.columns if c not in contract_cols]
        if strict_contract and (contract_missing or contract_extra):
            raise ValueError(
                f"{src.name}: 列集合与契约不符——缺 {contract_missing}，多 {contract_extra}"
            )
    _write_parquet(df, dst)
    rec = ConvertedFile(
        source=src.name,
        target=str(dst.name),
        rows_in=rows_in,
        rows_out=len(df),
        mapping=mapped,
        dropped_columns=dropped + contract_extra,
        sample_md5=_sample_md5(df),
    )
    rec.__dict__["contract_missing"] = contract_missing  # 登记缺列（不阻断）
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

    # summary：KEY_* → 小写蛇形 + 非 ASCII 列进丢弃清单（契约 §3/§9.5；R-06 修复）
    summary = pd.read_csv(summary_files[-1], encoding="utf-8-sig")
    summary_rename = {
        c: c.lower() for c in summary.columns if c.startswith(_SUMMARY_KEY_PREFIX)
    }
    summary = summary.rename(columns=summary_rename)
    # 非 ASCII 列（中文诊断列）进丢弃清单——诊断展示口径，不进裁决 schema
    non_ascii_cols = [c for c in summary.columns if not str(c).isascii()]
    summary_dropped = list(non_ascii_cols)
    summary = summary[[c for c in summary.columns if str(c).isascii()]]
    # 补 batch_id 标识列（契约 §3；R-06-#10）
    summary["batch_id"] = batch_id
    _write_parquet(summary, out_dir / "summary.parquet")
    report.files.append(
        ConvertedFile(
            source=summary_files[-1].name,
            target="summary.parquet",
            rows_in=len(summary),
            rows_out=len(summary),
            mapping={c: summary_rename.get(c, c) for c in summary.columns},
            dropped_columns=summary_dropped,
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
        return report  # 无折则直接返回（契约 §9 三态=必须报错）

    # ---- 逐折文件 ----
    n_topk = 0
    folds_missing_topk: List[str] = []
    for split_id in split_ids:
        fold_dir = folds_dir / f"split{split_id}"
        fold_meta: Dict[str, object] = {}
        # trades（契约 §9 三态=必须报错；R-06-#9 不再静默跳过）
        trades_files = sorted(source_dir.glob(f"walk_forward_trades_*_split{split_id}.csv"))
        if not trades_files:
            report.errors.append(f"split{split_id}: 缺 trades（契约三态=必须报错）")
            report.row_check_pass = False
            continue
        trades_rec = _convert_one(
            trades_files[-1],
            fold_dir / "trades.parquet",
            _TRADES_RENAME,
            report.files,
            contract_cols=_TRADES_CONTRACT_COLS,
            strict_contract=False,  # 历史批缺 F2 补 4 列属常态（登记不阻断）
        )
        # lot_id 重建（契约 §5 例外条款）：按 FIFO 从 trades 的 buy_date 重建归属批次
        if trades_rec is not None:
            trades_df = pd.read_parquet(fold_dir / "trades.parquet")
            trades_df = _rebuild_lot_id(trades_df)
            _write_parquet(trades_df, fold_dir / "trades.parquet")
            fold_meta["trades.lot_reconstructed"] = True
        # attribution（契约 §6）
        attr_files = sorted(source_dir.glob(f"walk_forward_execution_attribution_*_split{split_id}.csv"))
        if attr_files:
            _convert_one(
                attr_files[-1], fold_dir / "attribution.parquet", _ATTR_RENAME,
                report.files, contract_cols=_ATTR_CONTRACT_COLS,
            )
        # 持仓快照（契约 §7，中文表头映射）
        snap_files = sorted(source_dir.glob(f"walk_forward_持仓快照_*_split{split_id}.csv"))
        if snap_files:
            _convert_one(
                snap_files[-1], fold_dir / "holdings_snapshot.parquet",
                _HOLDINGS_COLS_ZH2EN, report.files, contract_cols=_SNAP_CONTRACT_COLS,
            )
        # topk_detail（契约 §7.1；可缺但不可重建——折级降级标注 R-06-#8）
        topk_files = sorted(source_dir.glob(f"walk_forward_topk_details_*_split{split_id}.csv"))
        if topk_files:
            n_topk += 1
            _convert_one(
                topk_files[-1], fold_dir / "topk_detail.parquet", {},
                report.files, contract_cols=_TOPK_CONTRACT_COLS,
            )
        else:
            fold_meta["topk_detail.missing"] = True
            folds_missing_topk.append(split_id)
        # daily.csv 重建（契约 §9 三态=可缺+可重建；R-06-#1）：从 trades + 快照重建
        daily_df = _rebuild_daily(trades_df if trades_rec else None,
                                   pd.read_parquet(fold_dir / "holdings_snapshot.parquet") if snap_files else None)
        if daily_df is not None:
            _write_parquet(daily_df, fold_dir / "daily.parquet")
            fold_meta["daily.daily_reconstructed"] = True
        # 折级 _meta.json（契约 §5.1/§7.1/§9.6；R-06-#2）
        if fold_meta:
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "_meta.json").write_text(
                json.dumps(fold_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if n_topk == 0 and split_ids:
        # 契约 §7.1：缺失即降级标注（证据机器侧该折不参与信号层尺子）
        report.meta_notes["topk_detail.missing"] = "true"
        report.meta_notes["topk_detail.missing_folds"] = ",".join(folds_missing_topk)
        logger.warning(f"{batch_id}: 全折缺 topk_detail，标注降级（基线批常见）")

    # ---- policy_lambda（契约 §8.1 条件条款；R-06-#4） ----
    policy_lambda_files = sorted(source_dir.glob("policy_lambda_*.csv"))
    if policy_lambda_files:
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

    # ---- batch_meta.json（契约 §2 全字段；R-06-#5） ----
    data_state_files = sorted(source_dir.glob("data_state_*.json"))
    data_state: Dict[str, object] = {}
    if data_state_files:
        try:
            data_state = json.loads(data_state_files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.meta_notes["data_state_parse_error"] = str(exc)

    # 从 summary 提取配置与代码态（契约 §2；指纹 = 附录 A 排除清单后的全键）
    config: Dict[str, object] = {}
    code_state: Dict[str, object] = {}
    arms: List[str] = []
    if not summary.empty:
        row = summary.iloc[-1]
        for k in ("git_commit", "git_dirty"):
            if k in summary.columns:
                code_state[k] = str(row[k]) if pd.notna(row[k]) else None
        # 配置指纹键（全键 − 排除清单；附录 A fail-safe 口径）
        fp_keys = _config_fingerprint_keys(summary.columns.tolist())
        config = {k: (row[k].item() if hasattr(row[k], "item") else row[k])
                  for k in fp_keys if k in summary.columns and pd.notna(row[k])}
        if "enable_repurchase_features" in summary.columns:
            arms.append("repurchase")
    config_fingerprint = _fingerprint(config)

    batch_meta = {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": pd.Timestamp.now().isoformat(),
        "host_mode": "backtest",  # WF OOS 回测批
        "source": "converted_from_legacy_wf_batch",
        "source_dir": str(source_dir),
        "code_state": code_state,
        "data_state": data_state,
        "config": config,
        "config_fingerprint": config_fingerprint,
        "baseline_ref": None,
        "arms": arms,
        "meta_notes": report.meta_notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # 行数校验（源 vs 目标）
    for rec in report.files:
        if rec.rows_in != rec.rows_out:
            report.row_check_pass = False
            report.errors.append(f"{rec.source}: 行数不一致 {rec.rows_in}→{rec.rows_out}")

    return report


# ---- lot_id / daily 重建（契约 §5 例外条款 / §9 三态；R-06-#1/#3） ----


def _rebuild_lot_id(trades: pd.DataFrame) -> pd.DataFrame:
    """按 FIFO 从 trades 重建 lot_id（v2 新增列，契约 §5）。

    规则：每次 buy 开新 lot（lot_id = buy_date_序号）；sell 按最早买入日优先消耗；
    卖出行回填被消耗的 lot_id（多 lot 消耗时用「首 lot」并标注）。
    """
    if "ts_code" not in trades.columns or "action" not in trades.columns:
        return trades
    trades = trades.copy()
    trades["lot_id"] = ""
    # 按 ts_code + trade_date 排序，逐股维护 lot 栈
    for ts_code, grp in trades.groupby("ts_code"):
        lots: List[Dict[str, object]] = []  # {lot_id, shares, buy_date}
        for idx, row in grp.sort_values("trade_date").iterrows():
            if row["action"] == "buy":
                lot_id = f"{row.get('buy_date', row['trade_date'])}_{len(lots)}"
                lots.append({"lot_id": lot_id, "shares": row["shares"], "buy_date": row.get("buy_date")})
                trades.at[idx, "lot_id"] = lot_id
            elif row["action"] == "sell":
                # FIFO 消耗
                remaining = row["shares"]
                consumed: List[str] = []
                while remaining > 0 and lots:
                    head = lots[0]
                    take = min(head["shares"], remaining)
                    consumed.append(head["lot_id"])
                    head["shares"] -= take
                    remaining -= take
                    if head["shares"] <= 0:
                        lots.pop(0)
                trades.at[idx, "lot_id"] = "|".join(consumed) if consumed else ""
    return trades


def _rebuild_daily(
    trades: Optional[pd.DataFrame], snapshot: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """从 trades + 快照重建 daily.csv（契约 §8；R-06-#1）。"""
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
    daily["exposure_lambda"] = 1.0  # 政策层未启用恒 1.0
    return daily


def _fingerprint(config: Dict[str, object]) -> str:
    """配置指纹（附录 A fail-safe：规范化序列化 sha256 短指纹）。"""
    import hashlib

    norm = json.dumps(config, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _config_fingerprint_keys(all_cols: List[str]) -> List[str]:
    """配置指纹键清单（附录 A fail-safe：全键 − 显式排除清单）。"""
    exclude_exact = {
        "wf_run_id", "batch_run_id", "batch_period_label", "registered_at",
        "data_state_id", "git_commit", "git_dirty",
        "data_daily_latest", "data_cs_train_latest", "data_dividend_coverage",
    }
    exclude_prefix = ("bt_total_return", "bt_annual_return", "bt_max_drawdown",
                      "bt_volatility", "bt_sharpe", "bt_calmar", "bt_trading_days",
                      "bt_start", "bt_end", "train_samples", "val_samples",
                      "test_samples", "best_iteration", "key_")
    return [
        c for c in all_cols
        if c not in exclude_exact and not any(c.startswith(p) for p in exclude_prefix)
    ]

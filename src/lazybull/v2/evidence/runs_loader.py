# -*- coding: utf-8 -*-
"""runs 产物契约（F2）读入桥——证据机器只认 runs schema 的通路（R-05 修复）。

契约依据：证据机器只消费 runs schema（`docs/contracts/runs_artifact_contract.md`），
旧 WF 产物由一次性转换器（runs_convert）对齐；本模块是转换产物的唯一读入桥。

读取不变量（契约 §9）在本桥硬校验：
- 必需文件缺失（trades/summary/chain_nav/batch_meta）⇒ 报错；
- trades.action ∈ {buy, sell}、卖出行 buy_date 非空且不晚于 trade_date；
- chain_nav 折内日期严格递增；
- 列集合 = 契约集合（缺列/多列报错）；
- 文件名/字段名全 ASCII（发现中文字段报错，提示先跑转换器）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.compare.fold_subset import RunArtifacts


@dataclass
class RunsFold:
    """单折 runs 产物（契约 schema）。"""

    split_index: int
    trades: Optional[pd.DataFrame] = None
    daily: Optional[pd.DataFrame] = None
    attribution: Optional[pd.DataFrame] = None
    holdings_snapshot: Optional[pd.DataFrame] = None
    topk_detail: Optional[pd.DataFrame] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class RunsBatch:
    """一个 runs 批次（契约 schema 的完整读入）。"""

    batch_id: str
    directory: Path
    batch_meta: Dict[str, object]
    summary: pd.DataFrame
    chain_nav: pd.DataFrame
    folds: Dict[int, RunsFold]
    policy_lambda: Optional[pd.DataFrame] = None
    policy_lambda_meta: Optional[Dict[str, object]] = None

    def to_run_artifacts(self) -> RunArtifacts:
        """转为证据机器复用的 RunArtifacts（chain_nav + summary 窗口）。"""
        windows = self.summary[
            [c for c in ("split_index", "test_start", "test_end") if c in self.summary.columns]
        ].copy()
        return RunArtifacts(
            label=self.batch_id,
            directory=self.directory,
            chain=self.chain_nav,
            windows=windows,
            data_state_id=(self.batch_meta.get("data_state") or {}).get("data_state_id"),
        )


def _check_ascii_columns(df: pd.DataFrame, name: str) -> None:
    """契约 §9.5：发现中文字段报错（提示先跑转换器）。"""
    non_ascii = [c for c in df.columns if not str(c).isascii()]
    if non_ascii:
        raise ValueError(
            f"{name} 含非 ASCII 列 {non_ascii[:5]}——请先跑 runs_convert 转换器对齐 schema"
        )


def _check_trades(trades: pd.DataFrame) -> None:
    """契约 §9.1：action 闭集 + 卖出行 buy_date 校验。"""
    bad_action = ~trades["action"].isin(["buy", "sell"])
    if bad_action.any():
        raise ValueError(f"trades.action 含非法值: {trades.loc[bad_action, 'action'].unique().tolist()}")
    sells = trades[trades["action"] == "sell"]
    bad_sell = sells["buy_date"].isna() | (sells["buy_date"] > sells["trade_date"])
    if bad_sell.any():
        raise ValueError(f"trades 卖出行 buy_date 非法（空或晚于成交日）: {bad_sell.sum()} 行")


def _check_chain_nav(chain: pd.DataFrame) -> None:
    """契约 §9.2：折内日期严格递增（date 可为折内序号 int 或 YYYYMMDD 字符串）。"""
    for split, seg in chain.groupby("split_index"):
        dates = seg["date"]
        # 直接按原始 dtype 判单调（int 序号 / str 日期都适用；astype(str) 会把 int 序号
        # 转成字典序导致 "10"<"2" 误报，实测发现）
        if not dates.is_monotonic_increasing:
            raise ValueError(f"chain_nav 折 {split} 日期非严格递增")


def load_runs_batch(batch_dir: Path) -> RunsBatch:
    """读入一个 runs 批次（契约 schema），并做读取不变量硬校验。"""
    batch_dir = Path(batch_dir)
    meta_path = batch_dir / "batch_meta.json"
    summary_path = batch_dir / "summary.parquet"
    chain_path = batch_dir / "chain_nav.parquet"
    # 必需文件缺失 ⇒ 报错（契约 §9 三态=必须报错）
    for p, name in ((meta_path, "batch_meta.json"), (summary_path, "summary.parquet"),
                    (chain_path, "chain_nav.parquet")):
        if not p.exists():
            raise FileNotFoundError(f"{batch_dir} 缺必需文件 {name}（契约三态=必须报错）")

    batch_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = pd.read_parquet(summary_path)
    chain_nav = pd.read_parquet(chain_path)
    _check_ascii_columns(summary, "summary")
    _check_ascii_columns(chain_nav, "chain_nav")
    _check_chain_nav(chain_nav)

    # 逐折读入
    folds: Dict[int, RunsFold] = {}
    folds_dir = batch_dir / "folds"
    if folds_dir.exists():
        for fold_dir in sorted(folds_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith("split"):
                continue
            try:
                split_idx = int(fold_dir.name.replace("split", ""))
            except ValueError:
                continue
            fold = RunsFold(split_index=split_idx)
            trades_path = fold_dir / "trades.parquet"
            if trades_path.exists():
                fold.trades = pd.read_parquet(trades_path)
                _check_ascii_columns(fold.trades, f"folds/{fold_dir.name}/trades")
                _check_trades(fold.trades)
            for name, attr in (("daily", "daily"), ("attribution", "attribution"),
                               ("holdings_snapshot", "holdings_snapshot"),
                               ("topk_detail", "topk_detail")):
                p = fold_dir / f"{name}.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    _check_ascii_columns(df, f"folds/{fold_dir.name}/{name}")
                    setattr(fold, attr, df)
            meta_file = fold_dir / "_meta.json"
            if meta_file.exists():
                fold.meta = json.loads(meta_file.read_text(encoding="utf-8"))
            folds[split_idx] = fold

    # policy_lambda（条件条款）
    policy_lambda = None
    policy_lambda_meta = None
    pl_path = batch_dir / "policy_lambda.parquet"
    if pl_path.exists():
        policy_lambda = pd.read_parquet(pl_path)
        pl_meta_path = batch_dir / "policy_lambda.json"
        if pl_meta_path.exists():
            policy_lambda_meta = json.loads(pl_meta_path.read_text(encoding="utf-8"))

    return RunsBatch(
        batch_id=batch_meta.get("batch_id", batch_dir.name),
        directory=batch_dir,
        batch_meta=batch_meta,
        summary=summary,
        chain_nav=chain_nav,
        folds=folds,
        policy_lambda=policy_lambda,
        policy_lambda_meta=policy_lambda_meta,
    )

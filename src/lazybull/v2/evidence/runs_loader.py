# -*- coding: utf-8 -*-
"""runs 产物契约（F2）读入桥——证据机器只认 runs schema 的通路（R-05 修复）。

契约依据：证据机器只消费 runs schema（`docs/contracts/runs_artifact_contract.md`），
旧 WF 产物由一次性转换器（runs_convert）对齐；本模块是转换产物的唯一读入桥。

读取不变量（契约 §9）在本桥硬校验（全部实现，列集常量与转换器共用
`runs_schema.py` 单一来源）：
- §9 通则：必需文件缺失（batch_meta/summary/chain_nav/trades）⇒ 报错；
- §9.1 trades.action ∈ {buy, sell}、卖出行 buy_date 非空且不晚于 trade_date；
- §9.2 chain_nav 折内日期严格递增（唯一且单调）；summary.bt_total_return 与
  chain_nav 折内起止重算对账（容差 1e-6）；
- §9.3 batch_meta.config_fingerprint 重算一致；data_state.data_state_id 与
  逐折 summary 列一致；
- §9.4 同日同 (action, ts_code) 不得两行（引擎内应已合并）；
- §9.5 列名全 ASCII（发现中文字段报错，提示先跑转换器）；
- §9.6 折级 _meta.json 重建/降级标注读取；
- §9.7 列集合 = 契约集合（缺列/多列报错；summary 随配置扩展，仅校验必需列子集）；
- §9.8 跨折衔接（后折首行 nav = 前折末行 nav；date 为真实日期时同校验日期）+
  daily/trades 交叉对账（n_buys/n_sells 当日计数一致、折内 total_value 起止比
  与 chain_nav 折内 nav 起止比一致）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.compare.fold_subset import RunArtifacts
from src.lazybull.v2.evidence.fingerprint_keys import fingerprint
from src.lazybull.v2.evidence.runs_schema import (
    CHAIN_NAV_CONTRACT_COLS,
    FOLD_FILE_CONTRACT_COLS,
    SUMMARY_REQUIRED_COLS,
)

#: §9.2 后半对账容差（契约明文 1e-6）
_BT_RETURN_TOL = 1e-6
#: §9.8 跨折 nav 衔接容差
_SEAM_NAV_TOL = 1e-9


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


def _check_column_set(df: pd.DataFrame, contract_cols: List[str], name: str) -> None:
    """契约 §9.7：列集合 = 契约集合（缺列/多列报错）。"""
    missing = [c for c in contract_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in contract_cols]
    if missing or extra:
        raise ValueError(f"{name} 列集合与契约不符——缺 {missing}，多 {extra}")


def _check_trades(trades: pd.DataFrame, name: str) -> None:
    """契约 §9.1（action 闭集 + 卖出行 buy_date）+ §9.4（同日同股同向唯一）。"""
    bad_action = ~trades["action"].isin(["buy", "sell"])
    if bad_action.any():
        raise ValueError(f"{name}: action 含非法值: {trades.loc[bad_action, 'action'].unique().tolist()}")
    sells = trades[trades["action"] == "sell"]
    bad_sell = sells["buy_date"].isna() | (sells["buy_date"] > sells["trade_date"])
    if bad_sell.any():
        raise ValueError(f"{name}: 卖出行 buy_date 非法（空或晚于成交日）: {bad_sell.sum()} 行")
    dup = trades.duplicated(subset=["trade_date", "action", "ts_code"], keep=False)
    if dup.any():
        raise ValueError(
            f"{name}: 同日同 (action, ts_code) 两行 {int(dup.sum())} 行——契约 §9.4 "
            "要求引擎内已合并，出现即属引擎 bug"
        )


def _check_chain_nav(chain: pd.DataFrame, summary: pd.DataFrame) -> None:
    """契约 §9.2：折内日期严格递增 + bt_total_return 与折内 nav 起止对账（1e-6）。"""
    for split, seg in chain.groupby("split_index"):
        dates = seg["date"]
        # 严格递增 = 单调 + 唯一（date 可为折内序号 int 或 YYYYMMDD 字符串；
        # 直接按原始 dtype 判单调，astype(str) 会把 int 序号转成字典序导致误报）
        if not (dates.is_monotonic_increasing and dates.is_unique):
            raise ValueError(f"chain_nav 折 {split} 日期非严格递增（含重复或乱序）")
    # §9.2 后半：summary.bt_total_return 与 chain_nav 折内起止重算对账
    for split, seg in chain.groupby("split_index"):
        seg = seg.sort_values("date")
        nav_ret = float(seg["nav"].iloc[-1]) / float(seg["nav"].iloc[0]) - 1.0
        row = summary[summary["split_index"] == split]
        if row.empty:
            raise ValueError(f"summary 缺折 {split}（chain_nav 存在该折）")
        bt = float(row["bt_total_return"].iloc[0])
        if abs(bt - nav_ret) > _BT_RETURN_TOL:
            raise ValueError(
                f"折 {split}: summary.bt_total_return={bt} 与 chain_nav 折内起止 "
                f"{nav_ret} 对账超差（容差 {_BT_RETURN_TOL}）"
            )


def _check_batch_meta(batch_meta: Dict[str, object], summary: pd.DataFrame) -> None:
    """契约 §9.3：config_fingerprint 重算一致 + data_state_id 与逐折 summary 列一致。"""
    config = batch_meta.get("config")
    expected = batch_meta.get("config_fingerprint")
    if not isinstance(config, dict) or not expected:
        raise ValueError("batch_meta 缺 config / config_fingerprint（契约 §2 必填）")
    actual = fingerprint(config)
    if actual != expected:
        raise ValueError(
            f"config_fingerprint 重算不一致: 登记 {expected} vs 重算 {actual}——"
            "config 被篡改或指纹口径漂移"
        )
    ds_id = (batch_meta.get("data_state") or {}).get("data_state_id")
    if not ds_id:
        raise ValueError("batch_meta.data_state.data_state_id 缺失（契约 §9.3 数据态血缘必需）")
    col = summary["data_state_id"].dropna().unique()
    if len(col) != 1 or str(col[0]) != str(ds_id):
        raise ValueError(
            f"data_state_id 不一致: batch_meta={ds_id} vs summary 列 {col.tolist()}"
        )


def _check_fold_seams(chain: pd.DataFrame) -> None:
    """契约 §9.8 前半：跨折衔接——后折首行 nav = 前折末行 nav（折界重复点语义）。

    date 为真实日期（str）时同时校验后折首日 = 前折末日；折内序号 int 口径下
    date 逐折重置（实测），日期衔接不可校验，仅校验 nav 衔接。
    """
    seams = []
    for split, seg in chain.groupby("split_index"):
        seg = seg.sort_values("date")
        seams.append((split, seg.iloc[0], seg.iloc[-1]))
    for (a, _, last_a), (b, first_b, _) in zip(seams[:-1], seams[1:]):
        if abs(float(last_a["nav"]) - float(first_b["nav"])) > _SEAM_NAV_TOL:
            raise ValueError(
                f"跨折衔接断裂: 折 {a} 末行 nav={last_a['nav']} ≠ 折 {b} 首行 nav={first_b['nav']}"
            )
        if pd.api.types.is_string_dtype(chain["date"]) and str(last_a["date"]) != str(first_b["date"]):
            raise ValueError(
                f"跨折日期衔接断裂: 折 {a} 末日 {last_a['date']} ≠ 折 {b} 首日 {first_b['date']}"
            )


def _check_daily_trades_xref(fold: RunsFold, chain_seg: pd.DataFrame) -> None:
    """契约 §9.8 后半：daily 与 trades 交叉对账（daily 存在时）。"""
    daily = fold.daily
    if daily is None or fold.trades is None:
        return
    name = f"folds/split{fold.split_index}/daily"
    trades = fold.trades
    buys = trades[trades["action"] == "buy"].groupby("trade_date").size()
    sells = trades[trades["action"] == "sell"].groupby("trade_date").size()
    for _, row in daily.iterrows():
        d = row["trade_date"]
        if int(row["n_buys"]) != int(buys.get(d, 0)) or int(row["n_sells"]) != int(sells.get(d, 0)):
            raise ValueError(
                f"{name}: {d} 指令计数与 trades 对不上（daily "
                f"{row['n_buys']}/{row['n_sells']} vs trades {int(buys.get(d, 0))}/{int(sells.get(d, 0))}）"
            )
    # 折内 total_value 起止比与 chain_nav 折内 nav 起止比一致
    daily_sorted = daily.sort_values("trade_date")
    tv_ratio = float(daily_sorted["total_value"].iloc[-1]) / float(daily_sorted["total_value"].iloc[0])
    seg = chain_seg.sort_values("date")
    nav_ratio = float(seg["nav"].iloc[-1]) / float(seg["nav"].iloc[0])
    # 容差 1e-3（相对）：daily 自快照重建，与引擎净值口径存在实测 ~3e-4 稳定系统差
    # （快照金额舍入/费用口径）；1e-3 足以暴露串批次/串折级断裂又容忍口径差
    if abs(tv_ratio - nav_ratio) / abs(nav_ratio) > 1e-3:
        raise ValueError(
            f"{name}: 折内 total_value 起止比 {tv_ratio:.8f} 与 chain_nav nav 起止比 "
            f"{nav_ratio:.8f} 对不上（相对容差 1e-3，口径差实测 ~3e-4）"
        )


def _load_fold_dir(fold_dir: Path) -> Optional[RunsFold]:
    """读入单个折目录（§4.4 提取式拆分，P2a-T1）；非 split 目录或目录名无法解析时跳过。"""
    if not fold_dir.is_dir() or not fold_dir.name.startswith("split"):
        return None
    try:
        split_idx = int(fold_dir.name.replace("split", ""))
    except ValueError:
        return None
    fold = RunsFold(split_index=split_idx)
    trades_path = fold_dir / "trades.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(
            f"{fold_dir} 缺 trades.parquet（契约三态=必须报错）"
        )
    fold.trades = pd.read_parquet(trades_path)
    _check_ascii_columns(fold.trades, f"folds/{fold_dir.name}/trades")
    _check_column_set(fold.trades, FOLD_FILE_CONTRACT_COLS["trades"],
                      f"folds/{fold_dir.name}/trades")
    _check_trades(fold.trades, f"folds/{fold_dir.name}/trades")
    for name in ("daily", "attribution", "holdings_snapshot", "topk_detail"):
        p = fold_dir / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            _check_ascii_columns(df, f"folds/{fold_dir.name}/{name}")
            _check_column_set(df, FOLD_FILE_CONTRACT_COLS[name], f"folds/{fold_dir.name}/{name}")
            setattr(fold, name, df)
    meta_file = fold_dir / "_meta.json"
    if meta_file.exists():
        fold.meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return fold


def load_runs_batch(batch_dir: Path) -> RunsBatch:
    """读入一个 runs 批次（契约 schema），并做契约 §9 读取不变量硬校验。"""
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
    _check_column_set(chain_nav, CHAIN_NAV_CONTRACT_COLS, "chain_nav")
    summary_missing = [c for c in SUMMARY_REQUIRED_COLS if c not in summary.columns]
    if summary_missing:
        raise ValueError(f"summary 缺必需列 {summary_missing}（契约 §3）")
    _check_chain_nav(chain_nav, summary)
    _check_batch_meta(batch_meta, summary)
    _check_fold_seams(chain_nav)

    # 逐折读入（folds/ 目录整体缺失 ⇒ 报错：trades 属契约三态「必须报错」）
    folds: Dict[int, RunsFold] = {}
    folds_dir = batch_dir / "folds"
    if not folds_dir.exists():
        raise FileNotFoundError(f"{batch_dir} 缺 folds/ 目录（trades 契约三态=必须报错）")
    for fold_dir in sorted(folds_dir.iterdir()):
        fold = _load_fold_dir(fold_dir)
        if fold is not None:
            folds[fold.split_index] = fold

    # 折集合一致（summary / chain_nav / folds 目录三方）
    summary_splits = set(summary["split_index"].astype(int).tolist())
    chain_splits = {int(s) for s in chain_nav["split_index"].unique()}
    if summary_splits != chain_splits or summary_splits != set(folds):
        raise ValueError(
            f"折集合不一致: summary={sorted(summary_splits)} chain_nav={sorted(chain_splits)} "
            f"folds={sorted(folds)}"
        )

    # daily/trades 交叉对账（契约 §9.8 后半）
    for split_idx, fold in folds.items():
        _check_daily_trades_xref(fold, chain_nav[chain_nav["split_index"] == split_idx])

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

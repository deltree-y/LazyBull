"""runs 产物契约（F2 schema）的正式写出出口（P2a-T7 全新代码）。

本模块是 T7 验收门 5-8 的**正式出口**（非临时适配器，不推迟 P3）：
把 v2 回测宿主（``hosts/backtest/replay.py``）的逐折结果写为
``docs/contracts/runs_artifact_contract.md`` 的目录结构，并即时用读入桥
``v2/evidence/runs_loader.load_runs_batch`` 自校验（写不出合法产物即失败）。

目录结构（契约 §1，parquet 形态——runs_loader 只认 parquet）::

    runs_root/<batch_id>/
    ├── batch_meta.json            # 契约 §2 全字段
    ├── summary.parquet            # 逐折汇总（配置全键 + bt_* 指标 + 血缘）
    ├── chain_nav.parquet          # 链式净值（复刻旧 chain_nav_splits 串联语义）
    └── folds/splitXX/
        ├── trades.parquet         # 成交账本 + lot_id FIFO 重建（复用证据层实现）
        ├── attribution.parquet    # 执行归因
        ├── holdings_snapshot.parquet
        ├── daily.parquet          # 逐日组合明细（真实数据源直建，非重建）
        └── _meta.json             # 折级标注（lot_reconstructed / fifo_violations /
                                   #   topk_detail.missing / contract_missing）

口径决策登记：
- ``wf_run_id = batch_id``（v2 出口合一：批次即运行，契约 §5 血缘标识）；
- 列集一律从 ``v2/evidence/runs_schema.py`` 导入，禁止重写；
- lot_id FIFO 重建复用证据层公开实现 ``runs_convert.rebuild_lot_id``
  （P2a-T7 由私有名提为公开名，旧私有名保留别名）；
- chain_nav 的 ``date`` = 折内整数序号 0..n（与 B0 冻结件同形态，旧
  ``chain_nav_splits`` 中 nav 曲线 RangeIndex 下的实际行为）；
- topk_detail 不写（B0 无该产物）⇒ 每折 ``_meta.json: topk_detail.missing=true``
  + 批次 meta_notes 登记（同 runs_convert 口径）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from src.lazybull.v2.evidence.fingerprint_keys import config_fingerprint_keys, fingerprint
from src.lazybull.v2.evidence.runs_convert import rebuild_lot_id
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.evidence.runs_schema import (
    ATTR_CONTRACT_COLS,
    CHAIN_NAV_CONTRACT_COLS,
    DAILY_CONTRACT_COLS,
    SNAP_CONTRACT_COLS,
    TRADES_CONTRACT_COLS,
)

#: trades 引擎内部键 → 契约列名（契约 §5）
_TRADES_RENAME = {"date": "trade_date", "stock": "ts_code"}

#: attribution 引擎内部键 → 契约列名（契约 §6）
_ATTR_RENAME = {"planned_stock": "planned_ts_code", "actual_stock": "actual_ts_code"}

#: holdings_snapshot 引擎内部键（v2 table_schema SNAPSHOT_KEYS）→ 契约列名（契约 §7）
_SNAP_RENAME = {
    "date": "trade_date",
    "portfolio_value": "total_value",
    "holding_days": "held_days",
    "planned_exit_date": "due_date",
    "remaining_intervals": "remaining_days",
}

#: 日期类列（统一规范化 YYYYMMDD 字符串，项目日期契约）
_TRADES_DATE_COLS = ("trade_date", "signal_date", "buy_date")
_ATTR_DATE_COLS = ("signal_date", "ranking_date", "execution_date")
_SNAP_DATE_COLS = ("trade_date", "buy_date", "signal_date", "due_date")

#: 结果元素的必填键（缺一即出口输入契约违反）
_RESULT_REQUIRED_KEYS = (
    "split_index",
    "model_version",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "metrics",
    "nav_curve",
    "trades",
    "attribution",
    "holdings_snapshot",
)


def _norm_date_value(value: object) -> object:
    """标量日期规范化（YYYYMMDD 字符串）：兼容 Timestamp / ISO / YYYYMMDD / int。

    空值（None / NaN / NaT）归一为 ``pd.NA``；解析失败直接抛错（出口不静默）。
    """
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    digits = text.replace("-", "")
    if len(digits) >= 8 and digits[:8].isdigit():
        return digits[:8]
    raise ValueError(f"日期值无法规范化为 YYYYMMDD: {value!r}")


def _norm_date_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """逐列规范化日期为 YYYYMMDD 字符串（缺列跳过——缺列由契约对齐统一补 NA）。"""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(_norm_date_value).astype("object")
    return df


def _align_to_contract(
    df: pd.DataFrame,
    contract_cols: List[str],
    name: str,
    fold_meta: Dict[str, object],
    meta_key: str,
) -> pd.DataFrame:
    """列集对齐契约（§9.7）：契约外列报错（禁止静默丢列），缺列补 NA 并登记。"""
    extra = [c for c in df.columns if c not in contract_cols]
    if extra:
        raise ValueError(f"{name} 含契约外列 {extra}——runs 出口禁止静默丢列（契约 §9.7）")
    missing = [c for c in contract_cols if c not in df.columns]
    if missing:
        fold_meta[meta_key] = list(missing)
        df = df.copy()
        for col in missing:
            df[col] = pd.NA
    return df[[c for c in contract_cols]]


def _build_trades(
    result: Dict[str, object], batch_id: str, fold_meta: Dict[str, object]
) -> pd.DataFrame:
    """trades：引擎记录帧 → 契约 schema（改名 + 血缘头三列 + 日期规范化 + lot_id 重建）。"""
    split_index = int(result["split_index"])
    name = f"folds/split{split_index:02d}/trades"
    trades = result["trades"]
    if trades is None or len(trades) == 0:
        # 空折账本合法（契约三态只管文件存在性）；列集 = 契约全列
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in TRADES_CONTRACT_COLS})
        fold_meta["trades.lot_reconstructed"] = True
        return df
    df = trades.copy().rename(columns=_TRADES_RENAME)
    df.insert(0, "wf_run_id", batch_id)  # 口径登记：wf_run_id = batch_id（见模块头）
    df.insert(1, "split_index", split_index)
    df.insert(2, "model_version", result["model_version"])
    df = _norm_date_columns(df, _TRADES_DATE_COLS)
    df = _align_to_contract(df, TRADES_CONTRACT_COLS, name, fold_meta, "trades.contract_missing")
    # lot_id FIFO 重建（复用证据层公开实现；引擎原生 lot 口径落地前一律重建标注）。
    # lot_id 由重建填充，从缺列登记中剔除（重建后必然存在，登记会误导）
    recorded_missing = fold_meta.get("trades.contract_missing")
    if recorded_missing and "lot_id" in recorded_missing:
        recorded_missing = [c for c in recorded_missing if c != "lot_id"]
        if recorded_missing:
            fold_meta["trades.contract_missing"] = recorded_missing
        else:
            del fold_meta["trades.contract_missing"]
    df, fifo_violations = rebuild_lot_id(df)
    fold_meta["trades.lot_reconstructed"] = True
    if fifo_violations:
        fold_meta["lot_id.fifo_violations"] = int(fifo_violations)
        logger.warning(
            f"{batch_id}/split{split_index:02d}: lot_id FIFO 重建与卖出行 buy_date "
            f"不一致 {fifo_violations} 行（引擎非 FIFO 卖出），归属按 FIFO 口径"
        )
    return df


def _build_attribution(
    result: Dict[str, object], batch_id: str, fold_meta: Dict[str, object]
) -> Optional[pd.DataFrame]:
    """attribution：改名 + 血缘头三列 + 日期规范化 + 契约列集对齐（空则不出文件）。"""
    split_index = int(result["split_index"])
    name = f"folds/split{split_index:02d}/attribution"
    attribution = result["attribution"]
    if attribution is None or len(attribution) == 0:
        return None
    df = attribution.copy().rename(columns=_ATTR_RENAME)
    df.insert(0, "wf_run_id", batch_id)
    df.insert(1, "split_index", split_index)
    df.insert(2, "model_version", result["model_version"])
    df = _norm_date_columns(df, _ATTR_DATE_COLS)
    return _align_to_contract(
        df, ATTR_CONTRACT_COLS, name, fold_meta, "attribution.contract_missing"
    )


def _build_snapshot(
    result: Dict[str, object], batch_id: str, fold_meta: Dict[str, object]
) -> Optional[pd.DataFrame]:
    """holdings_snapshot：引擎内部键（SNAPSHOT_KEYS）→ 契约列名映射 + 血缘头三列。"""
    split_index = int(result["split_index"])
    name = f"folds/split{split_index:02d}/holdings_snapshot"
    snapshot = result["holdings_snapshot"]
    if snapshot is None or len(snapshot) == 0:
        return None
    df = snapshot.copy().rename(columns=_SNAP_RENAME)
    df.insert(0, "run_id", batch_id)
    df.insert(1, "split_index", split_index)
    df.insert(2, "model_version", result["model_version"])
    df = _norm_date_columns(df, _SNAP_DATE_COLS)
    return _align_to_contract(
        df, SNAP_CONTRACT_COLS, name, fold_meta, "holdings_snapshot.contract_missing"
    )


def _build_daily(
    nav_curve: pd.DataFrame,
    trades: pd.DataFrame,
    snapshot: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """daily：从引擎净值曲线 + trades + 快照**直接构建**（真实数据源，非重建）。

    列口径（契约 §8）：nav = 引擎折内净值（起点 1.0）；daily_return = nav 环比
    （首日 0）；n_positions = 快照当日持仓行数（快照无持仓日不记行，计 0）；
    n_buys/n_sells/turnover_amount = trades 当日计数与金额和；exposure_lambda
    恒 1.0（政策层退役，契约 §8 未启用口径）。
    """
    daily = nav_curve.rename(
        columns={"date": "trade_date", "portfolio_value": "total_value", "capital": "cash"}
    ).copy()
    daily = _norm_date_columns(daily, ("trade_date",))
    daily["daily_return"] = daily["nav"].pct_change().fillna(0.0)
    if snapshot is not None and len(snapshot) > 0:
        n_positions = snapshot.groupby("trade_date").size()
    else:
        n_positions = pd.Series(dtype=int)
    daily["n_positions"] = daily["trade_date"].map(n_positions).fillna(0).astype(int)
    if len(trades) > 0:
        buys = trades[trades["action"] == "buy"].groupby("trade_date").size()
        sells = trades[trades["action"] == "sell"].groupby("trade_date").size()
        turnover = trades.groupby("trade_date")["amount"].sum()
    else:
        buys = sells = pd.Series(dtype=int)
        turnover = pd.Series(dtype=float)
    daily["n_buys"] = daily["trade_date"].map(buys).fillna(0).astype(int)
    daily["n_sells"] = daily["trade_date"].map(sells).fillna(0).astype(int)
    daily["turnover_amount"] = daily["trade_date"].map(turnover).fillna(0.0)
    daily["exposure_lambda"] = 1.0  # 政策层退役（F8）：未启用恒 1.0
    daily = daily.drop(columns=["return"], errors="ignore")
    return daily[[c for c in DAILY_CONTRACT_COLS]]


def _build_chain_nav(ordered_results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """链式净值（复刻旧 ``ml/walk_forward/reporting.py::chain_nav_splits`` 串联语义）。

    逐折 cumulative 缩放（折界重复点：后折首行 nav = 前折末行 nav）、
    ``date`` = 折内整数序号 0..n（与 B0 冻结件同形态）。
    """
    chained_records: List[Dict[str, object]] = []
    cumulative_nav = 1.0
    for result in ordered_results:
        nav = result["nav_curve"]
        raw = nav["nav"].values
        if len(raw) == 0:
            continue
        scale = cumulative_nav / raw[0] if raw[0] != 0 else 1.0
        scaled = raw * scale
        for index, value in enumerate(scaled):
            chained_records.append(
                {
                    "date": index,
                    "nav": value,
                    "split_index": int(result["split_index"]),
                }
            )
        cumulative_nav = scaled[-1]
    if not chained_records:
        raise ValueError("全部折的净值曲线为空，无法串联 chain_nav（契约三态=必须报错）")
    return pd.DataFrame(chained_records, columns=CHAIN_NAV_CONTRACT_COLS)


def _build_summary(
    ordered_results: Sequence[Dict[str, object]],
    config: Dict[str, object],
    batch_id: str,
    data_state_id: str,
) -> pd.DataFrame:
    """summary.parquet：每行一折 = 配置全键 + bt_* 指标 + 血缘标识列。"""
    rows = []
    for result in ordered_results:
        row = dict(config)
        # 逐折差异配置键（如 cols_live——实取自折模型元数据，见 replay_b0）
        row.update(result.get("config_overrides") or {})
        row.update(result["metrics"])  # bt_* 指标列（bt_start/bt_end 亦在其中）
        row.update(
            {
                "batch_id": batch_id,
                "split_index": int(result["split_index"]),
                "model_version": result["model_version"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "wf_run_id": batch_id,  # 口径登记：wf_run_id = batch_id
                "data_state_id": data_state_id,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_batch_meta(
    batch_id: str,
    summary: pd.DataFrame,
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    baseline_ref: Optional[str],
    arms: Iterable[str],
    source: str,
    meta_notes: Dict[str, str],
) -> Dict[str, object]:
    """batch_meta.json 组装（契约 §2 全字段）。

    config 口径同 ``runs_convert._build_batch_meta``：对 summary 列跑
    ``config_fingerprint_keys`` 后取首行键值（batch_id 等新增键按 fail-safe
    口径一并入指纹，与转换器产物同口径可比）。
    """
    fp_keys = config_fingerprint_keys(summary.columns.tolist())
    row = summary.iloc[0]
    meta_config = {
        k: (row[k].item() if hasattr(row[k], "item") else row[k])
        for k in fp_keys
        if k in summary.columns and pd.notna(row[k])
    }
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": pd.Timestamp.now().isoformat(),
        "host_mode": "backtest",
        "source": source,
        "code_state": code_state,
        "data_state": data_state,
        "config": meta_config,
        "config_fingerprint": fingerprint(meta_config),
        "baseline_ref": baseline_ref,
        "arms": list(arms),
        "meta_notes": meta_notes,
    }


def _validate_results(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """出口输入契约校验 + 按 split_index 升序排序（chain 串联语义依赖折序）。"""
    if not results:
        raise ValueError("results 为空——runs 批次至少需要一折")
    ordered = sorted(results, key=lambda r: int(r["split_index"]))
    for result in ordered:
        missing = [k for k in _RESULT_REQUIRED_KEYS if k not in result]
        if missing:
            raise ValueError(
                f"split {result.get('split_index')} 的结果缺必填键 {missing}"
                "（runs 出口输入契约）"
            )
        nav_curve = result["nav_curve"]
        if nav_curve is None or len(nav_curve) == 0 or "nav" not in nav_curve.columns:
            raise ValueError(
                f"split {result['split_index']} 净值曲线为空（chain_nav 契约三态=必须报错）"
            )
    split_indices = [int(r["split_index"]) for r in ordered]
    if len(set(split_indices)) != len(split_indices):
        raise ValueError(f"split_index 重复: {split_indices}")
    return ordered


def write_runs_batch(
    results: Sequence[Dict[str, object]],
    *,
    batch_id: str,
    runs_root: Path,
    config: Dict[str, object],
    code_state: Dict[str, object],
    data_state: Dict[str, object],
    baseline_ref: Optional[str] = None,
    arms: Iterable[str] = (),
    source: str = "v2_p2a_native_replay",
) -> Path:
    """把一批逐折回测结果写为 runs 契约批次目录，并用读入桥自校验。

    Args:
        results: 逐折结果（键集见模块头/_RESULT_REQUIRED_KEYS；metrics 含 bt_* 列，
            nav_curve/trades/attribution/holdings_snapshot 为引擎原始产物帧）
        batch_id: 批次 ID（ASCII；同时作为 wf_run_id 写入逐折血缘列）
        runs_root: runs 根目录（产出落 ``runs_root/<batch_id>/``）
        config: 批次配置字典（127 指纹键全列；缺值键如 val_rankic_ir 给 NA——
            排除键不影响指纹）
        code_state: ``{git_commit, git_dirty}``（契约 §2）
        data_state: 数据态快照（必须含 ``data_state_id``，契约 §9.3 硬校验）
        baseline_ref: 对照批次 id（基线批为 None）
        arms: 本批次包含的臂标签
        source: 产物来源标识

    Returns:
        批次目录路径（``runs_root/<batch_id>``）

    Raises:
        ValueError: 出口输入契约违反或写后自校验（load_runs_batch 硬校验）失败
    """
    data_state_id = (data_state or {}).get("data_state_id")
    if not data_state_id:
        raise ValueError("data_state 必须含 data_state_id（契约 §9.3 数据态血缘必需）")
    ordered = _validate_results(results)

    out_dir = Path(runs_root) / batch_id
    meta_notes: Dict[str, str] = {}

    # 逐折产物
    for result in ordered:
        split_index = int(result["split_index"])
        fold_dir = out_dir / "folds" / f"split{split_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_meta: Dict[str, object] = {}

        trades_df = _build_trades(result, batch_id, fold_meta)
        trades_df.to_parquet(fold_dir / "trades.parquet", index=False)

        attr_df = _build_attribution(result, batch_id, fold_meta)
        if attr_df is not None:
            attr_df.to_parquet(fold_dir / "attribution.parquet", index=False)

        snap_df = _build_snapshot(result, batch_id, fold_meta)
        if snap_df is not None:
            snap_df.to_parquet(fold_dir / "holdings_snapshot.parquet", index=False)

        daily_df = _build_daily(result["nav_curve"], trades_df, snap_df)
        daily_df.to_parquet(fold_dir / "daily.parquet", index=False)

        # topk_detail 不写（B0 无该产物）⇒ 折级降级标注（契约 §7.1，同 runs_convert 口径）
        fold_meta["topk_detail.missing"] = True
        (fold_dir / "_meta.json").write_text(
            json.dumps(fold_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    # 批次 meta_notes：topk 全折缺失登记（同 runs_convert 口径）
    fold_ids = [f"split{int(r['split_index']):02d}" for r in ordered]
    meta_notes["topk_detail.missing"] = "true"
    meta_notes["topk_detail.missing_folds"] = ",".join(fold_ids)

    # 批次级产物
    chain_nav = _build_chain_nav(ordered)
    chain_nav.to_parquet(out_dir / "chain_nav.parquet", index=False)
    summary = _build_summary(ordered, config, batch_id, data_state_id)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    batch_meta = _build_batch_meta(
        batch_id, summary, code_state, data_state, baseline_ref, arms, source, meta_notes
    )
    (out_dir / "batch_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # 写完即自校验（契约 §9 读取不变量硬校验，抛错即失败）
    load_runs_batch(out_dir)
    logger.info(f"runs 批次已写出并自校验通过: {out_dir}（{len(ordered)} 折）")
    return out_dir

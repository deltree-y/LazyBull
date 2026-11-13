# -*- coding: utf-8 -*-
"""P7 T2.5 分数层：冻结模型只推理、逐折分片、锚批次一致性核验。

口径
====
- 打分复用 P3 已验证工具 ``score_model_fingerprint``（``cs_train_verify`` 源，
  单折试点已证与锚批次 ``pred_score`` 逐位一致）：**不换源、不改实现**；
- 全排名（用于目标层）复用内核 ``MLSignal.generate_ranked``（含 BasicUniverse 候选域
  与成交额/市值/金融股选择过滤），与回测引擎同一实现；
- 价格只用于**目标级估算**（股数整手、市值与资金诊断），不是 T4 的成交价源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.t25.params import (
    ARMS,
    REBALANCE_FREQ,
    STAGGER_TRANCHES,
    TOP_N,
    FoldRow,
    T25Params,
)
from scripts.v2_p7.t25.target_layer import PlannerKernel, select_targets


class SignalLayerError(RuntimeError):
    """分数层身份或覆盖不满足冻结口径。"""


def _signal_selection() -> Mapping[str, object]:
    from scripts.v2_p3.t5_arm_executors import _signal_selection_mapping

    selection = _signal_selection_mapping()
    if int(selection["top_n"]) != TOP_N:
        raise SignalLayerError(f"信号 Top-N 与冻结口径不符：{selection['top_n']}")
    return selection


def _signal_universe_config() -> Mapping[str, object]:
    from scripts.v2_p3.t5_arm_executors import _signal_universe_config_mapping

    return _signal_universe_config_mapping()


def score_fold(
    params: T25Params,
    fold: FoldRow,
    fold_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """单折打分并落盘（预测行 + 信号 top-N），返回身份摘要。

    打分日 = 该折 ``test_start~test_end`` 的全部交易日（缺分区即失败，不缩域）。
    """
    from scripts.v2_p3.t05b_fingerprint import (
        load_main_board_codes,
        score_model_fingerprint,
    )

    fold_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = fold_dir / "scores.parquet"
    if predictions_path.exists():
        predictions_path.unlink()
    topk_path = fold_dir / "signal_topk.parquet"
    if topk_path.exists():
        topk_path.unlink()

    main_board = load_main_board_codes(params.data_root, stock_domain="main")
    result = score_model_fingerprint(
        models_dir=params.data_root / "models" / "stock_selection",
        model_version=int(fold.model_version),
        data_root=params.data_root,
        score_days=list(fold.signal_days),
        main_board_codes=main_board,
        producer_identity={"task": "p7_t2.5", "run_id": run_id, "split_index": fold.split_index},
        persist_predictions=True,
        predictions_path=predictions_path,
        signal_selection=_signal_selection(),
        signal_universe_config=_signal_universe_config(),
    )
    # 合法缩域：打分工具内部的「窄口径合法闭市日」（仅 Stock Connect 闭市，§19.18）
    # 不进打分域；其余拒绝仍由严格日门抛错（strict_days 默认 True，未放宽）。
    closure_days = sorted(str(day) for day in (result.get("excluded_days") or []))
    covered = int(result["n_days"]) + len(closure_days)
    if covered != len(fold.signal_days):
        raise SignalLayerError(
            f"split{fold.split_index:02d} 打分日覆盖不足：{int(result['n_days'])}"
            f"+{len(closure_days)} != {len(fold.signal_days)}"
        )
    topk_rows = int(result.get("signal_topk_rows") or 0)
    empty_days = list(result.get("signal_topk_empty_days") or [])
    if topk_rows <= 0:
        raise SignalLayerError(f"split{fold.split_index:02d} 信号 top-N 为空")
    return {
        "split_index": fold.split_index,
        "model_version": fold.model_version,
        "signal_days": len(fold.signal_days),
        "score_days": int(result["n_days"]),
        "closure_days": closure_days,
        "closure_day_count": len(closure_days),
        "score_rows": int(result["n_rows"]),
        "fingerprint": str(result["fingerprint"]),
        "predictions_file": predictions_path.name,
        "predictions_sha256": str(result.get("predictions_sha256") or ""),
        "signal_topk_file": topk_path.name,
        "signal_topk_sha256": str(result.get("signal_topk_sha256") or ""),
        "signal_topk_rows": topk_rows,
        "signal_topk_empty_days": len(empty_days),
        "source_mode": str(result.get("source_mode") or ""),
        "row_set_semantics": str(result.get("row_set_semantics") or ""),
        "model_identity": dict(params.model_identity[int(fold.model_version)]),
        "per_seed_diagnostic": list(result.get("per_seed_diagnostic") or []),
    }


def anchor_score_consistency(
    params: T25Params, fold: FoldRow, predictions: pd.DataFrame
) -> Dict[str, Any]:
    """锚批次 ``pred_score`` 逐位一致性。

    **口径关键（两处键）**：锚 attribution 的 ``pred_score = candidate_score.get(actual_stock)``
    且候选来自 ``ranking_date`` 当日排名。因此关联键必须是
    ``(ranking_date, actual_ts_code)``：按 ``planned_ts_code`` 关联会把 T+1 停牌顺延链
    误读成名次错位；按 ``signal_date`` 关联会把补位阶段（``execution_stage=completion``，
    ``ranking_date`` 晚于信号日）行误判为不一致。
    """
    path = params.anchor_dir / "folds" / f"split{fold.split_index:02d}" / "attribution.parquet"
    if not path.exists():
        raise SignalLayerError(f"锚批次归因缺件：{path}")
    attribution = pd.read_parquet(path)
    result = compare_anchor_scores(attribution, predictions)
    result["split_index"] = fold.split_index
    return result


def compare_anchor_scores(attribution: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Any]:
    """纯比对：``(ranking_date, actual_ts_code)`` → 逐位相等（便于单元回归）。"""
    required = {"ranking_date", "actual_ts_code", "pred_score", "planned_ts_code"}
    missing = sorted(required - set(attribution.columns))
    if missing:
        raise SignalLayerError(f"锚归因缺列：{missing}")
    executed = attribution.dropna(subset=["actual_ts_code", "pred_score"])
    merged = executed[["ranking_date", "actual_ts_code", "pred_score"]].merge(
        predictions,
        left_on=["ranking_date", "actual_ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
    )
    diff = (merged["pred_score"] - merged["score"]).abs()
    matched = int(merged["score"].notna().sum())
    exact = int((diff <= 1e-12).sum())
    substituted = int(
        (executed["actual_ts_code"].astype(str) != executed["planned_ts_code"].astype(str)).sum()
    )
    return {
        "anchor_rows": int(len(executed)),
        "substituted_rows": substituted,
        "matched_rows": matched,
        "exact_rows": exact,
        "max_abs_diff": None if diff.notna().sum() == 0 else float(diff.max()),
        "all_exact": bool(matched == len(executed) and exact == len(executed)),
        "compare_key": "ranking_date+actual_ts_code",
    }


def anchor_plan_consistency(
    params: T25Params,
    fold: FoldRow,
    ranked_lookup: "RankedProvider",
) -> Dict[str, Any]:
    """按锚批次**折内**排期复算计划集合，逐（日期, 档）比对计划股与名次。

    该复算只用于证明目标层与回测内核同源；正式目标仍按跨折连续账户的
    全局相位日历产出（两者在折首日之外可以不同，属预期）。

    **差异归因**：本复算的持仓是「T+1 全额成交」的计划账，锚的持仓是「实际成交」账
    （T+1 停牌顺延会换成下一名候选），两者在折内会逐步分叉 ⇒ 后续为再平衡日的
    「排除已持仓」集合不同，选股出现差异。因此除逐日严格比对外，另按
    「差异股 ∈ 两侧持仓差集」逐日归因，只有**无法用持仓差解释**的差异才算口径不符。
    """
    from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule

    fold_dir = params.anchor_dir / "folds" / f"split{fold.split_index:02d}"
    attribution = pd.read_parquet(fold_dir / "attribution.parquet")
    planned: Dict[Tuple[str, int], List[str]] = {}
    for (date, tranche), group in attribution.groupby(["signal_date", "tranche_idx"]):
        ordered = group.dropna(subset=["planned_ts_code"]).sort_values("planned_rank")
        planned[(str(date), int(tranche))] = [str(code) for code in ordered["planned_ts_code"]]
    anchor_held = _anchor_holdings_by_date(fold_dir / "holdings_snapshot.parquet")

    days = list(fold.signal_days)
    schedule = compute_tranche_schedule(days, REBALANCE_FREQ, STAGGER_TRANCHES)
    kernel = PlannerKernel(TOP_N)
    book: Dict[str, int] = {}
    pending: Dict[int, List[str]] = {}
    due: Dict[int, List[str]] = {}
    matched = 0
    strict_matched = 0
    compared = 0
    short_rows = 0
    pinned = 0
    fallback = 0
    mismatched: List[Dict[str, Any]] = []
    for pos, day in enumerate(days):
        tranche = schedule.get(day)
        if tranche is not None:
            # 持仓集合钉在锚的逐日快照上：本比对只考察「选择逻辑是否同源」，
            # 不把「计划账 vs 实际成交账（停牌顺延）」的持仓分叉混进来。
            prev_day = days[pos - 1] if pos > 0 else ""
            snapshot = anchor_held.get(prev_day) if prev_day else None
            if snapshot is None:
                held = sorted(book)
                fallback += 1
            else:
                held = sorted(snapshot)
                pinned += 1
            picks, _ = select_targets(kernel, ranked_lookup(day), day, int(tranche), held)
            key = (day, int(tranche))
            if key in planned:
                compared += 1
                mine = [str(code) for code, _ in picks]
                anchor = planned[key]
                # 锚 attribution 可能少行（未成交/未写行的槽位不产出记录），
                # 故按公共前缀比对，并把「锚行数不足」单独登记，不算选股差异。
                common = min(len(mine), len(anchor))
                if mine[:common] == anchor[:common] and len(mine) >= len(anchor):
                    matched += 1
                    if len(mine) == len(anchor):
                        strict_matched += 1
                    else:
                        short_rows += 1
                elif len(mismatched) < 5:
                    mismatched.append(
                        {
                            "signal_date": day,
                            "tranche_idx": int(tranche),
                            "mine": mine,
                            "anchor": anchor,
                        }
                    )
            next_pos = pos + 1
            if next_pos < len(days):
                pending.setdefault(next_pos, []).extend(str(code) for code, _ in picks)
        for code in pending.pop(pos, []):
            book[code] = pos
            due.setdefault(pos + 20, []).append(code)
        for code in due.pop(pos, []):
            book.pop(code, None)
    return {
        "split_index": fold.split_index,
        "anchor_plans": len(planned),
        "compared_plans": compared,
        "matched_plans": matched,
        "strict_matched_plans": strict_matched,
        "anchor_short_row_days": short_rows,
        "unexplained_days": compared - matched,
        "held_pinned_days": pinned,
        "held_fallback_days": fallback,
        "all_matched": bool(compared == matched and compared > 0),
        "compare_rule": "持仓钉锚快照 + 公共前缀（锚 attribution 少行时不算选股差异）",
        "missing_in_anchor_bt_window": len(planned) - compared,
        "mismatch_examples": mismatched,
    }


def _anchor_holdings_by_date(path: Path) -> Dict[str, set]:
    """读取锚批次逐日持仓快照（``trade_date -> 股票集合``）。"""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if "trade_date" not in frame.columns or "ts_code" not in frame.columns:
        return {}
    grouped: Dict[str, set] = {}
    for date, group in frame.groupby(frame["trade_date"].astype(str)):
        grouped[str(date)] = {str(code) for code in group["ts_code"]}
    return grouped


class RankedProvider:
    """逐日全排名提供者（内核 ``MLSignal.generate_ranked``，按日缓存）。"""

    def __init__(self, params: T25Params) -> None:
        from src.lazybull.data.loader import DataLoader
        from src.lazybull.data.storage import Storage
        from src.lazybull.universe import BasicUniverse

        self._params = params
        self._storage = Storage(root_path=str(params.data_root))
        self._loader = DataLoader(self._storage)
        stock_basic = self._loader.load_clean_stock_basic()
        if stock_basic is None or len(stock_basic) == 0:
            raise SignalLayerError("缺少 clean/stock_basic，禁止回退未保护元数据")
        config = _signal_universe_config()
        self._universe = BasicUniverse(
            stock_basic=stock_basic,
            exclude_st=bool(config["exclude_st"]),
            min_list_days=config["min_list_days"],
            markets=list(config["markets"]),
            verbose=False,
        )
        self._selection = _signal_selection()
        self._signals: Dict[int, Any] = {}
        self._cache: Dict[str, List[Tuple[str, float]]] = {}
        self.calls = 0
        self.empty_days: List[str] = []

    def _signal(self, model_version: int) -> Any:
        from scripts.v2_p3.t05b_fingerprint import _configure_signal_selection
        from src.lazybull.v2.core.signal.ml_signal import MLSignal

        signal = self._signals.get(int(model_version))
        if signal is None:
            signal = MLSignal(
                top_n=TOP_N,
                model_version=int(model_version),
                models_dir=str(self._params.data_root / "models" / "stock_selection"),
                verbose=False,
            )
            _configure_signal_selection(signal, self._selection)
            if int(signal.top_n) != TOP_N:
                raise SignalLayerError(f"MLSignal top_n 不符：{signal.top_n}")
            self._signals[int(model_version)] = signal
        return signal

    def __call__(self, day: str) -> List[Tuple[str, float]]:
        cached = self._cache.get(day)
        if cached is not None:
            return cached
        model_version = self._params.model_version_of_day(day)
        signal = self._signal(model_version)
        frame = self._storage.load_cs_train_day(day)
        if frame is None or len(frame) == 0:
            raise SignalLayerError(f"{day} 缺 cs_train 分区，无法生成全排名")
        quote = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if quote is None or len(quote) == 0:
            raise SignalLayerError(f"{day} 缺 clean/daily 行情，无法生成候选域")
        codes = self._universe.get_stocks(pd.Timestamp(day), quote_data=quote)
        ranked = signal.generate_ranked(pd.Timestamp(day), codes, {"features": frame})
        result = [(str(code), float(score)) for code, score in ranked]
        if not result:
            # 内核在该日拒绝预测（特征质量门禁）⇒ 当日无候选、不产生计划；
            # 显式登记为「空排名日」，不静默缩域、不补数据。
            self.empty_days.append(day)
        self._cache[day] = result
        self.calls += 1
        return result

    @property
    def cached_days(self) -> int:
        return len(self._cache)


class PriceProvider:
    """逐日估计价提供者（``clean/daily`` 的 open/close；按日缓存）。"""

    def __init__(self, params: T25Params) -> None:
        from src.lazybull.data.loader import DataLoader
        from src.lazybull.data.storage import Storage

        self._loader = DataLoader(Storage(root_path=str(params.data_root)))
        self._cache: Dict[str, Dict[str, Tuple[float, float]]] = {}

    def __call__(self, day: str) -> Dict[str, Tuple[float, float]]:
        cached = self._cache.get(day)
        if cached is not None:
            return cached
        frame = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if frame is None or len(frame) == 0:
            self._cache[day] = {}
            return {}
        needed = {"ts_code", "open", "close"}
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise SignalLayerError(f"{day} clean/daily 缺列：{missing}")
        quote = {
            str(row.ts_code): (float(row.open or 0.0), float(row.close or 0.0))
            for row in frame.itertuples(index=False)
        }
        self._cache[day] = quote
        return quote

    def load_frame(self, day: str) -> Optional[pd.DataFrame]:
        """当日 ``clean/daily`` 原始帧（供 Kelly 方差历史建表，不做缓存）。"""
        frame = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if frame is None or len(frame) == 0:
            return None
        return frame

    def closes(self, day: str) -> Dict[str, float]:
        """当日收盘价（逐日流式读取，不缓存全量，避免内存膨胀）。"""
        frame = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if frame is None or len(frame) == 0:
            return {}
        if "ts_code" not in frame.columns or "close" not in frame.columns:
            raise SignalLayerError(f"{day} clean/daily 缺 ts_code/close 列")
        return {str(row.ts_code): float(row.close or 0.0) for row in frame.itertuples(index=False)}


def verify_replay(
    params: T25Params,
    run_root: Path,
    splits: Sequence[int],
    days_per_fold: int = 3,
) -> Dict[str, Any]:
    """逐位重放复核：抽 ``splits × days`` 重算分数并与已落盘分片全字段比对。

    比对项：行集（逐行 ``trade_date``/``ts_code``）、行数、逐行分数逐位相等。
    """
    fold_map = params.fold_by_split
    checks: List[Dict[str, Any]] = []
    for split in splits:
        fold = fold_map.get(int(split))
        if fold is None:
            raise SignalLayerError(f"未知折：{split}")
        split_dir = run_root / "folds" / f"split{int(split):02d}"
        stored = pd.read_parquet(split_dir / "scores.parquet")
        sample_days = _sample_days(fold.signal_days, days_per_fold)
        reference = stored[stored["trade_date"].astype(str).isin(sample_days)]
        reference = reference[["trade_date", "ts_code", "score"]].copy()
        reference["trade_date"] = reference["trade_date"].astype(str)
        reference = reference.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        replay = _rescore_frame(params, fold, sample_days)
        same_rows = len(reference) == len(replay)
        same_keys = same_rows and list(
            zip(reference["trade_date"], reference["ts_code"].astype(str))
        ) == list(zip(replay["trade_date"], replay["ts_code"]))
        diff = float((reference["score"] - replay["score"]).abs().max()) if same_keys else None
        checks.append(
            {
                "split_index": fold.split_index,
                "model_version": fold.model_version,
                "days": sample_days,
                "rows": int(len(reference)),
                "rows_match": bool(same_rows),
                "keys_match": bool(same_keys),
                "max_abs_diff": diff,
                "bitwise_identical": bool(same_keys and diff == 0.0),
            }
        )
    return {
        "splits": [int(split) for split in splits],
        "days_per_fold": int(days_per_fold),
        "checks": checks,
        "all_bitwise_identical": all(item["bitwise_identical"] for item in checks),
    }


def _column_mismatch_count(left: pd.Series, right: pd.Series) -> Optional[int]:
    """单列严格比对（dtype 必须一致；整数精确相等；浮点按位表示；其余按值）。

    Returns:
        不一致行数；``None`` 表示 dtype 不同（本身就是不一致）。
    """
    import numpy as np

    if left.dtype != right.dtype:
        return None
    lvalues = left.to_numpy()
    rvalues = right.to_numpy()
    if pd.api.types.is_integer_dtype(left):
        # 整数用精确比较，避免减法/绝对值溢出（int64 极值会被误判为相等）
        return int((lvalues != rvalues).sum())
    if pd.api.types.is_float_dtype(left):
        # 位相等语义：+0.0 与 -0.0、不同尾数、NaN 的位模式都按位比较
        lbits = lvalues.astype(np.float64, copy=False).view(np.uint64)
        rbits = rvalues.astype(np.float64, copy=False).view(np.uint64)
        return int((lbits != rbits).sum())
    if pd.api.types.is_bool_dtype(left):
        return int((lvalues != rvalues).sum())
    ltext = pd.Series(lvalues, dtype="object").fillna("<NA>").astype(str).to_numpy()
    rtext = pd.Series(rvalues, dtype="object").fillna("<NA>").astype(str).to_numpy()
    return int((ltext != rtext).sum())


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> Dict[str, Any]:
    """逐列逐行严格比对两帧（dtype 一致 + 整数精确 + 浮点位相等 + 字符串全等）。"""
    if list(left.columns) != list(right.columns):
        return {
            "columns_match": False,
            "rows_match": False,
            "identical": False,
            "column_diffs": {"<columns>": [list(left.columns), list(right.columns)]},
        }
    if len(left) != len(right):
        return {
            "columns_match": True,
            "rows_match": False,
            "identical": False,
            "column_diffs": {"<rows>": [int(len(left)), int(len(right))]},
        }
    diffs: Dict[str, Any] = {}
    for column in left.columns:
        bad = _column_mismatch_count(
            left[column].reset_index(drop=True), right[column].reset_index(drop=True)
        )
        if bad is None:
            diffs[column] = f"dtype:{left[column].dtype}->{right[column].dtype}"
        elif bad:
            diffs[column] = f"{bad} rows differ"
    return {
        "columns_match": True,
        "rows_match": True,
        "identical": not diffs,
        "rows": int(len(left)),
        "column_diffs": diffs,
    }


def verify_targets_replay(
    params: T25Params,
    arms: Mapping[str, Any],
    ranked_lookup: "RankedProvider",
    price_lookup: Any,
    variance_book: Any = None,
) -> Dict[str, Any]:
    """目标层逐位重放：同进程重算两袖，与本次生成的计划/持仓/逐日帧全字段比对。

    覆盖 ``shares`` / ``buy_price`` / ``due_date`` / ``tranche_idx`` / ``model_version``
    等目标全字段（不只看分数），任一处被篡改即失败。
    """
    from scripts.v2_p7.t25.target_layer import simulate_arm

    checks: List[Dict[str, Any]] = []
    for arm in ARMS:
        again = simulate_arm(
            arm=arm,
            calendar=params.extended_days,
            schedule=params.schedule(arm),
            ranked_lookup=ranked_lookup,
            price_lookup=price_lookup,
            params=params,
            capital=params.sleeve_capital[arm],
            top_n=params.sleeve_top_n[arm],
            variance_book=variance_book,
        )
        current = arms[arm]
        for label, left, right in (
            ("plans", current.plans, again.plans),
            ("holdings", current.holdings, again.holdings),
            ("daily", current.daily, again.daily),
        ):
            result = compare_frames(left, right)
            result.update({"arm": arm, "frame": label})
            checks.append(result)
    return {
        "arms": list(ARMS),
        "frames_checked": len(checks),
        "checks": checks,
        "all_identical": all(item["identical"] for item in checks),
    }


def _sample_days(days: Sequence[str], count: int) -> List[str]:
    """确定性抽样：首日、中点、末日（不足则去重保序）。"""
    if count <= 0:
        raise SignalLayerError("抽样天数必须为正")
    picks = [days[0], days[len(days) // 2], days[-1]][:count]
    unique: List[str] = []
    for day in picks:
        if day not in unique:
            unique.append(day)
    return unique


def _rescore_frame(params: T25Params, fold: FoldRow, days: Sequence[str]) -> pd.DataFrame:
    """按 ``cs_train_verify`` 口径重算指定日的全行分数（与打分核同一实现）。"""
    from scripts.v2_p3.t05b_fingerprint import (
        _configure_signal_selection,
        _filter_tradable_section,
        load_main_board_codes,
    )
    from src.lazybull.data.storage import Storage
    from src.lazybull.v2.core.signal.ml_signal import MLSignal

    storage = Storage(root_path=str(params.data_root))
    signal = MLSignal(
        top_n=TOP_N,
        model_version=int(fold.model_version),
        models_dir=str(params.data_root / "models" / "stock_selection"),
        verbose=False,
    )
    signal._load_model()
    _configure_signal_selection(signal, _signal_selection())
    main_board = load_main_board_codes(params.data_root, stock_domain="main")
    rows: List[Dict[str, Any]] = []
    for day in days:
        frame = storage.load_cs_train_day(day)
        section = _filter_tradable_section(frame[frame["ts_code"].isin(main_board)])
        built = signal._build_model_input(pd.Timestamp(day), section)
        if built is None:
            raise SignalLayerError(f"{day} 重放时模型输入被拒绝")
        features, matrix = built
        scores = signal._predict_scores(matrix, 0, 0, ranked=False)
        for code, score in zip(features["ts_code"].tolist(), scores):
            rows.append({"trade_date": day, "ts_code": str(code), "score": float(score)})
    frame = pd.DataFrame(rows, columns=["trade_date", "ts_code", "score"])
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def fold_timing_summary(timings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """推理耗时与增长曲线（供全窗任务重新申请机器批）。"""
    if not timings:
        return {"folds": [], "total_seconds": 0.0}
    total = float(sum(float(item["score_seconds"]) for item in timings))
    rows = int(sum(int(item["score_rows"]) for item in timings))
    days = int(sum(int(item["score_days"]) for item in timings))
    return {
        "folds": [dict(item) for item in timings],
        "total_seconds": round(total, 1),
        "total_rows": rows,
        "total_days": days,
        "rows_per_second": round(rows / total, 1) if total > 0 else None,
        "seconds_per_day": round(total / days, 3) if days > 0 else None,
    }


__all__ = [
    "PriceProvider",
    "RankedProvider",
    "SignalLayerError",
    "anchor_plan_consistency",
    "anchor_score_consistency",
    "fold_timing_summary",
    "score_fold",
    "verify_replay",
]

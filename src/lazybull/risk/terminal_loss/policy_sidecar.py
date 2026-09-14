"""terminal_loss 政策旁路：持仓打分、风险台账、触发清单与阈值扫描（P2-1）。

阶段定位：**离线回放**——不改回测引擎的决策，也不进回测 loop；只消费回测导出的
中文表头持仓快照（``common.sidecar_schema.SNAPSHOT_COLUMNS_ZH``），按"训练段严格
更早"的规则选折模型给持仓打分，产出三张中文表头的表：

1. 持仓风险台账（一行 = 某持仓的某个剩余持有期）
2. 触发清单（双重条件命中：正确拦截 / 误杀；可选漏报）
3. 阈值扫描（每个 (截面分位阈值, 绝对概率阈值) 组合一行）

硬约束：

- **不得前视**：某日期的打分只能用"该日期时 Train/Val 都已结束"的折模型
  （折的 ES 段未参与训练，可用于评估）；区间外日期明确跳过并告警，不猜测、不外推。
- **标签口径唯一**：事后收益与异常亏损标签一律经 ``labels.build_terminal_loss_labels``
  生成后按 (T, 股票, h) 关联，禁止本模块复制标签公式；每个折用自己的 k 与 h_max。
- **σ 口径唯一**：沿用 ``factors.risk.volatility_factors.compute_sigma_daily_panel``
  （严格日历对齐，停牌缺行不压缩），窗口按交易日历位置前推。
- **截面分位必须在完整同日截面内计算**（先全截面排名、再取持仓），禁止持仓子集内重排。
- **特征列必须自证可得**：模型 ``feature_names`` 中任何不在 cs_train 或本模块派生范围内
  的列（例如 ``full`` 的 ``pct_*``）一律**明确报错**，不允许静默降级。
- **h 必须在模型网格内**：``剩余持有交易日``超出 [1, h_max] 直接报错（禁止外推）。
- 中文表头是用户可见产物的统一约定；内部键只用于代码与 join。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ...common.sidecar_schema import (
    LEDGER_COLUMNS_ZH,
    SCAN_COLUMNS_ZH,
    SNAPSHOT_COLUMNS_ZH,
    TRIGGER_COLUMNS_ZH,
    select_chinese,
)
from ...factors.risk.volatility_factors import compute_sigma_daily_panel
from .dataset import (
    attach_horizon_features,
    load_clean_daily_panels,
    load_trade_calendar,
)
from .labels import TerminalLossLabelConfig, build_terminal_loss_labels

#: 台账内部列（按内部键计算，落盘前统一中文化；顺序即输出列序）
_LEDGER_KEYS: List[str] = [
    "run_id",
    "split_index",
    "fold",
    "date",
    "ts_code",
    "weight",
    "market_value",
    "buy_date",
    "holding_days",
    "planned_exit_date",
    "remaining_intervals",
    "p_loss",
    "p_loss_daypct",
    "day_pool_size",
    "sigma_daily_20",
    "expected_vol_over_horizon",
    "mkt_vol_20",
    "terminal_return",
    "loss_label",
    "label_status",
    "avoidable_loss",
    "last_holding_date",
]

#: 触发清单内部列
_TRIGGER_KEYS: List[str] = [
    "date",
    "ts_code",
    "event_type",
    "p_loss",
    "p_loss_daypct",
    "remaining_intervals",
    "weight",
    "sigma_daily_20",
    "expected_vol_over_horizon",
    "mkt_vol_20",
    "terminal_return",
    "loss_label",
    "avoidable_loss",
]

_LEDGER_MAP: Dict[str, str] = dict(zip(_LEDGER_KEYS, LEDGER_COLUMNS_ZH))
_TRIGGER_MAP: Dict[str, str] = dict(zip(_TRIGGER_KEYS, TRIGGER_COLUMNS_ZH))

#: σ 面板窗口（与标签契约一致）
SIGMA_WINDOW = 20

#: 模型期限网格上限（与 TerminalLossLabelConfig 默认一致）
HORIZON_MAX = 20

#: 标签网格分块交易日数（多年 OOS 上一次生成全网格会达上亿行，内存不可接受；
#: 块尾按 h_max + 1 多切，保证端点落在真实数据上）
LABEL_CHUNK_DAYS = 60


@dataclass(frozen=True)
class FoldModel:
    """一个 terminal_loss 折模型的可用信息（按日期选择用）。"""

    fold: str
    model_path: Path
    val_end: str
    es_start: str
    es_end: str
    h_max: int
    loss_sigma_multiple: float
    feature_names: Tuple[str, ...]


def load_fold_index(risk_root: str, arm_suffix: str) -> List[FoldModel]:
    """扫描折目录并构建按日期选择的模型索引（升序按 val_end）。

    Raises:
        ValueError: 未找到折目录，或折元数据缺 stage_dates / 模型文件缺失
    """
    import json

    root = Path(risk_root)
    meta_paths = sorted(root.glob(f"*{arm_suffix}/terminal_loss_model.json"))
    if not meta_paths:
        raise ValueError(f"{root} 下未找到匹配 *{arm_suffix} 的折元数据，无法选择打分模型")
    folds: List[FoldModel] = []
    for meta_path in meta_paths:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        md = meta.get("metadata") or {}
        stages = md.get("stage_dates") or {}
        if not {"train", "val", "es"} <= set(stages):
            raise ValueError(f"{meta_path} 缺少 stage_dates（train/val/es），无法确定前视边界")
        model_path = meta_path.parent / "terminal_loss_model.joblib"
        if not model_path.exists():
            raise ValueError(f"折 {meta_path.parent.name} 缺少模型文件 {model_path.name}")
        label_config = md.get("label_config") or {}
        folds.append(
            FoldModel(
                fold=meta_path.parent.name[: -len(arm_suffix)],
                model_path=model_path,
                val_end=str(stages["val"][1]),
                es_start=str(stages["es"][0]),
                es_end=str(stages["es"][1]),
                h_max=int(label_config.get("h_max", HORIZON_MAX)),
                loss_sigma_multiple=float(label_config.get("loss_sigma_multiple", 1.0)),
                feature_names=tuple(meta.get("feature_names") or []),
            )
        )
    folds.sort(key=lambda item: item.val_end)
    return folds


def select_fold_for_date(date: str, folds: Sequence[FoldModel]) -> Optional[FoldModel]:
    """按"训练段严格更早"的规则为某交易日选择折模型（无可用折返回 None）。

    取 ``val_end <= date`` 中 val_end 最大的折：该折的 Train/Val 均在 ``date`` 之前
    结束，ES 段未参与训练，因此用它打分无前视。
    """
    candidates = [item for item in folds if item.val_end <= date]
    return candidates[-1] if candidates else None


def _read_snapshot(path: Path) -> pd.DataFrame:
    """读取中文表头持仓快照并规范化主键（缺列明确报错）。"""
    frame = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    english_by_zh = {zh: key for key, zh in SNAPSHOT_COLUMNS_ZH.items()}
    missing = [zh for zh in english_by_zh if zh not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少中文列 {sorted(missing)}；表结构由 sidecar_schema 冻结")
    frame = frame.rename(columns=english_by_zh)
    frame["run_id"] = frame["运行标识"] if "运行标识" in frame.columns else path.stem
    frame["split_index"] = frame["折序号"] if "折序号" in frame.columns else -1
    frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def _calendar_window(
    data_root: str, dates: Sequence[str], forward_extra: int = 0
) -> Tuple[List[str], List[str]]:
    """返回覆盖台账所需的交易日历窗口（含 σ 预热与最远端点）。

    Returns:
        (calendar, dates_in_calendar)：calendar 为窗口内完整 Trading 日历
    """
    all_days = load_trade_calendar(data_root, "19900101", "20991231")
    first_idx = all_days.index(dates[0])
    last_idx = all_days.index(dates[-1])
    start = all_days[max(0, first_idx - SIGMA_WINDOW - 1)]
    end_idx = min(len(all_days) - 1, last_idx + forward_extra)
    calendar = [day for day in all_days if start <= day <= all_days[end_idx]]
    return calendar, list(calendar)


def _sigma_lookup(data_root: str, calendar: Sequence[str]) -> pd.DataFrame:
    """按严格日历口径计算 σ 面板并平铺为 (date, ts_code, sigma_daily_20)。"""
    _, close_panel, _, _ = load_clean_daily_panels(data_root, calendar[0], calendar[-1])
    panel = compute_sigma_daily_panel(
        close_panel.stack().rename("close_adj").reset_index(), list(calendar), window=SIGMA_WINDOW
    )
    stacked = panel.stack().rename("sigma_daily_20").reset_index()
    stacked.columns = ["date", "ts_code", "sigma_daily_20"]
    return stacked


def _read_feature_day(path: Path, needed: Sequence[str]) -> pd.DataFrame:
    """按需列读取 cs_train 分区（缺列明确报错；避免整表 383 列带来无谓 IO）。"""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in needed if c not in available]
    if missing:
        raise ValueError(
            f"{path.name} 缺少打分所需列 {sorted(missing)}；"
            f"（``full`` 的 pct_* 不在 cs_train 内，请使用 core / core_state 特征集）"
        )
    columns = sorted(set(needed) | {"ts_code"})
    frame = pd.read_parquet(path, columns=columns)
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame


def _attach_inputs(
    frame: pd.DataFrame,
    data_root: str,
    feature_root: Path,
    feature_columns: Sequence[str],
    sigma_table: pd.DataFrame,
    context_columns: Sequence[str] = ("mkt_vol_20",),
) -> pd.DataFrame:
    """补齐模型输入列（cs_train 直读 + σ 派生），缺列/缺分区明确报错。"""
    derived = ("remaining_intervals", "sigma_daily_20", "expected_vol_over_horizon")
    direct = [c for c in feature_columns if c not in derived]
    needed = sorted(set(direct) | set(context_columns))
    pieces: List[pd.DataFrame] = []
    for date in sorted(frame["date"].unique()):
        path = feature_root / f"{date}.parquet"
        if not path.exists():
            raise ValueError(f"缺少 cs_train 分区 {path}（台账日期 {date} 需要特征列）")
        day = _read_feature_day(path, needed)
        day = day[["ts_code"] + needed].copy()
        day["date"] = date
        pieces.append(day)
    merged = frame.merge(pd.concat(pieces, ignore_index=True), on=["date", "ts_code"], how="left")
    merged = merged.merge(sigma_table, on=["date", "ts_code"], how="left")
    # 规范派生实现以 ``h`` 为期限键（训练矩阵口径），此处对齐后取回派生列
    merged["h"] = merged["remaining_intervals"].astype(int)
    merged = attach_horizon_features(merged).drop(columns=["h"])
    return merged


def _attach_labels(
    frame: pd.DataFrame,
    data_root: str,
    folds: Sequence[FoldModel],
    calendar: Sequence[str],
    chunk_days: int = LABEL_CHUNK_DAYS,
) -> pd.DataFrame:
    """按折分别调用规范标签实现，并按其 (T, 股票, h) 关联事后结果。

    每个折用自己的 ``loss_sigma_multiple`` / ``h_max``（折间超参可能不同，
    混用会改变标签定义）。**标签网格按交易日分块构建**：全网格一次生成会
    在多年 OOS 上产生上亿行（内存不可接受），分块后只保留台账需要的
    (T, 股票, h) 键，块尾多切 ``h_max + 1`` 日以保证端点落在真实数据上。
    """
    open_panel, close_panel, _, _ = load_clean_daily_panels(data_root, calendar[0], calendar[-1])
    sigma_panel = compute_sigma_daily_panel(
        close_panel.stack().rename("close_adj").reset_index(), list(calendar), window=SIGMA_WINDOW
    )

    keep = ["ts_code", "trade_date", "h", "terminal_return", "loss_label", "label_status"]
    pieces: List[pd.DataFrame] = []
    for fold_name, part in frame.groupby("fold", sort=False):
        fold = next(item for item in folds if item.fold == fold_name)
        config = TerminalLossLabelConfig(
            h_min=1, h_max=fold.h_max, loss_sigma_multiple=fold.loss_sigma_multiple
        )
        wanted = part[["date", "ts_code"]].drop_duplicates()
        collected: List[pd.DataFrame] = []
        for chunk_start in range(0, len(calendar), chunk_days):
            chunk = list(calendar[chunk_start : chunk_start + chunk_days])
            chunk_dates = set(chunk)
            keys = wanted[wanted["date"].isin(chunk_dates)]
            if keys.empty:
                continue
            tail = calendar[
                chunk_start : min(len(calendar), chunk_start + chunk_days + fold.h_max + 1)
            ]
            labels = build_terminal_loss_labels(open_panel.loc[tail], sigma_panel.loc[tail], config)
            needed_keys = set(zip(keys["date"], keys["ts_code"]))
            hit = labels[
                [
                    (date, code) in needed_keys
                    for date, code in zip(labels["trade_date"], labels["ts_code"])
                ]
            ]
            collected.append(hit[keep])
        if not collected:
            part = part.assign(terminal_return=np.nan, loss_label=np.nan, label_status=None)
            pieces.append(part)
            continue
        labels = pd.concat(collected, ignore_index=True)
        merged = part.merge(
            labels,
            left_on=["date", "ts_code", "remaining_intervals"],
            right_on=["trade_date", "ts_code", "h"],
            how="left",
            validate="many_to_one",
        )
        missing = int(merged["label_status"].isna().sum())
        if missing:
            logger.warning(
                f"折 {fold_name}: {missing} 行未能关联到标签"
                f"（T 日无报价或 h 超出该折网格 {fold.h_max}），已保留空值"
            )
        pieces.append(merged.drop(columns=["trade_date", "h"]))
    out = pd.concat(pieces, ignore_index=True)
    out["avoidable_loss"] = -out["terminal_return"]
    return out


def _default_model_loader(path: str) -> Any:
    """加载折模型：固定名别名是 payload dict，必须经 ``TerminalLossModel.load`` 还原。

    （``v{N}_model.joblib`` 才是可直接 ``joblib.load`` 的模型实例；policy sidecar
    统一读固定名别名，保证"读的就是既有工具看到的那份"。）
    """
    import joblib

    from .model import TerminalLossModel

    payload = joblib.load(path)
    if isinstance(payload, TerminalLossModel):
        return payload
    if isinstance(payload, dict):
        return TerminalLossModel.load(path)
    raise ValueError(f"无法识别的模型 artifact 类型 {type(payload).__name__}: {path}")


def score_holdings(
    snapshot_files: Sequence[Path],
    risk_root: str,
    arm_suffix: str,
    data_root: str,
    with_daypct: bool = True,
    model_loader: Optional[Callable[[str], Any]] = None,
    feature_root: Optional[str] = None,
) -> pd.DataFrame:
    """给持仓快照打分并生成风险台账（内部键；落盘经 :func:`write_ledger` 中文化）。

    Args:
        snapshot_files: 中文表头持仓快照 CSV 列表
        risk_root: terminal_loss WF 根目录（选择折模型）
        arm_suffix: 折目录后缀（如 ``_d5_v6m_fscore``）
        data_root: 数据根目录（clean / features）
        with_daypct: 是否计算当日截面分位（需按 (日期, 剩余持有期) 给完整截面打分）
        model_loader: 模型加载器（默认 joblib.load；测试可注入桩实现）
        feature_root: cs_train 分区目录（默认 ``{data_root}/features/cs_train``）

    Raises:
        ValueError: 无可打分行、日期早于所有折、h 超网格、缺分区或缺特征列
    """
    loader = model_loader or _default_model_loader
    folds = load_fold_index(risk_root, arm_suffix)
    feature_dir = Path(feature_root or (Path(data_root) / "features" / "cs_train"))

    frames: List[pd.DataFrame] = []
    for path in snapshot_files:
        frame = _read_snapshot(Path(path))
        frame["remaining_intervals"] = pd.to_numeric(frame["remaining_intervals"], errors="coerce")
        zero_h = int((frame["remaining_intervals"] == 0).sum())
        if zero_h:
            logger.info(f"{path.name}: {zero_h} 行剩余持有期为 0（期末当日，标签恒为 0），已剔除")
        frame = frame[frame["remaining_intervals"].notna()]
        frame = frame[frame["remaining_intervals"] >= 1].copy()
        if not frame.empty:
            frame["remaining_intervals"] = frame["remaining_intervals"].astype(int)
            frames.append(frame)
    if not frames:
        raise ValueError("持仓快照中没有可打分的行（剩余持有交易日全部为空或为 0）")
    ledger = pd.concat(frames, ignore_index=True)

    fold_of_date = {d: select_fold_for_date(d, folds) for d in sorted(ledger["date"].unique())}
    skipped = [d for d, fold in fold_of_date.items() if fold is None]
    if skipped:
        logger.warning(
            f"{len(skipped)} 个日期早于最早的折（{folds[0].val_end}），已跳过："
            f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}"
        )
    ledger["fold"] = ledger["date"].map(lambda d: fold_of_date[d].fold if fold_of_date[d] else None)
    ledger = ledger[ledger["fold"].notna()].reset_index(drop=True)
    if ledger.empty:
        raise ValueError("所有快照日期都早于最早的折模型，无法打分；请先训练覆盖该区间的折")

    h_max = int(ledger["remaining_intervals"].max())
    if h_max > min(HORIZON_MAX, min(item.h_max for item in folds)):
        raise ValueError(
            f"剩余持有交易日最大 {h_max} 超出模型网格（上限 "
            f"{min(HORIZON_MAX, min(item.h_max for item in folds))}）；禁止外推打分"
        )

    calendar, _ = _calendar_window(
        data_root, sorted(ledger["date"].unique()), forward_extra=h_max + 1
    )
    sigma_table = _sigma_lookup(data_root, calendar)

    scored: List[pd.DataFrame] = []
    for fold_name, part in ledger.groupby("fold", sort=False):
        fold = next(item for item in folds if item.fold == fold_name)
        model = loader(str(fold.model_path))
        feature_names = list(getattr(model, "feature_names", None) or fold.feature_names)
        if not feature_names:
            raise ValueError(f"折 {fold_name} 未提供 feature_names，无法构造模型输入")
        inputs = _attach_inputs(
            part.drop(columns=["p_loss"], errors="ignore"),
            data_root,
            feature_dir,
            feature_names,
            sigma_table,
        )
        missing = [c for c in feature_names if c not in inputs.columns]
        if missing:
            raise ValueError(f"折 {fold_name} 缺少模型输入列 {sorted(missing)}")
        nan_share = float(inputs[feature_names].isna().any(axis=1).mean())
        if nan_share:
            logger.info(f"折 {fold_name}: {nan_share:.1%} 行含缺失输入（XGBoost 原生处理 NaN）")
        inputs["p_loss"] = np.asarray(model.predict_proba(inputs[feature_names]), dtype=float)
        scored.append(inputs)
    ledger = pd.concat(scored, ignore_index=True)

    ledger = _attach_labels(ledger, data_root, folds, calendar)

    if with_daypct:
        ledger = _attach_day_percentile(
            ledger, folds, loader, data_root, feature_dir, calendar, sigma_table
        )
    else:
        ledger["p_loss_daypct"] = np.nan
        ledger["day_pool_size"] = np.nan

    keys = ["run_id", "split_index", "buy_date", "ts_code"]
    ledger["last_holding_date"] = ledger.groupby(keys)["date"].transform("max")
    return ledger


def _attach_day_percentile(
    ledger: pd.DataFrame,
    folds: Sequence[FoldModel],
    loader: Callable[[str], Any],
    data_root: str,
    feature_dir: Path,
    calendar: Sequence[str],
    sigma_table: pd.DataFrame,
) -> pd.DataFrame:
    """按 (日期, 剩余持有期) 给**完整当日截面**打分并回填持仓分位。

    契约：先全截面排名、再取持仓行；分母是当日该 h 的全部可打分股票，
    不是持仓子集（禁止在持仓内重排）。
    """
    out = ledger.copy()
    out["p_loss_daypct"] = np.nan
    out["day_pool_size"] = np.nan
    key = ["date", "ts_code", "remaining_intervals"]
    for fold_name, part in out.groupby("fold", sort=False):
        fold = next(item for item in folds if item.fold == fold_name)
        model = loader(str(fold.model_path))
        feature_names = list(getattr(model, "feature_names", None) or fold.feature_names)
        # 回填目标列必须先移除，否则 join 会因同名重叠而报错
        part = part.drop(
            columns=[c for c in ("p_loss_daypct", "day_pool_size") if c in part.columns]
        )
        pools: List[pd.DataFrame] = []
        for date, day_part in part.groupby("date", sort=True):
            day_frame = pd.read_parquet(feature_dir / f"{date}.parquet", columns=["ts_code"])
            horizons = sorted(day_part["remaining_intervals"].unique().tolist())
            sections = []
            for horizon in horizons:
                section = day_frame[["ts_code"]].copy()
                section["date"] = date
                section["remaining_intervals"] = int(horizon)
                sections.append(section)
            pool = pd.concat(sections, ignore_index=True)
            pool = _attach_inputs(
                pool, data_root, feature_dir, feature_names, sigma_table, context_columns=()
            )
            missing = [c for c in feature_names if c not in pool.columns]
            if missing:
                raise ValueError(f"折 {fold_name} 截面打分缺少输入列 {sorted(missing)}")
            pool["p_loss"] = np.asarray(model.predict_proba(pool[feature_names]), dtype=float)
            pool["p_loss_daypct"] = pool.groupby("remaining_intervals")["p_loss"].rank(pct=True)
            pool["day_pool_size"] = pool.groupby("remaining_intervals")["p_loss"].transform("size")
            pools.append(pool[key + ["p_loss_daypct", "day_pool_size"]])
        pool_all = pd.concat(pools, ignore_index=True).drop_duplicates(key).set_index(key)
        joined = part.join(pool_all, on=key)  # 保持左表索引，便于回填
        out.loc[joined.index, "p_loss_daypct"] = joined["p_loss_daypct"].to_numpy()
        out.loc[joined.index, "day_pool_size"] = joined["day_pool_size"].to_numpy()
        missing_pct = int(joined["p_loss_daypct"].isna().sum())
        if missing_pct:
            logger.warning(f"折 {fold_name}: {missing_pct} 行未能取到当日截面分位")
    return out


def build_trigger_list(
    ledger: pd.DataFrame, p_hi: float, p_abs: float, include_missed: bool = True
) -> pd.DataFrame:
    """构建触发清单：双重条件命中（正确拦截 / 误杀）与（可选）漏报事件。"""
    frame = ledger[ledger["loss_label"].notna()].copy()
    frame["triggered"] = (frame["p_loss_daypct"] >= p_hi) & (frame["p_loss"] >= p_abs)
    is_loss = frame["loss_label"] == 1.0
    masks = [
        (frame["triggered"] & is_loss, "正确拦截"),
        (frame["triggered"] & ~is_loss, "误杀"),
    ]
    if include_missed:
        masks.append((~frame["triggered"] & is_loss, "漏报"))
    parts = [_trigger_rows(frame.loc[mask], event_type) for mask, event_type in masks if mask.any()]
    if not parts:
        return pd.DataFrame(columns=_TRIGGER_KEYS)
    return pd.concat(parts, ignore_index=True)


def _trigger_rows(frame: pd.DataFrame, event_type: str) -> pd.DataFrame:
    out = frame[
        [
            "date",
            "ts_code",
            "p_loss",
            "p_loss_daypct",
            "remaining_intervals",
            "weight",
            "sigma_daily_20",
            "expected_vol_over_horizon",
            "mkt_vol_20",
            "terminal_return",
            "loss_label",
            "avoidable_loss",
        ]
    ].copy()
    out.insert(2, "event_type", event_type)
    return out


def scan_thresholds(
    ledger: pd.DataFrame, p_hi_list: Iterable[float], p_abs_list: Iterable[float]
) -> pd.DataFrame:
    """阈值扫描：每个 (截面分位阈值, 绝对概率阈值) 组合统计拦截/误杀/漏报与收益代价。"""
    valid = ledger[ledger["loss_label"].notna()].copy()
    total = len(valid)
    events = int((valid["loss_label"] == 1.0).sum())
    rows: List[Dict[str, float]] = []
    for p_hi in p_hi_list:
        for p_abs in p_abs_list:
            triggered = (valid["p_loss_daypct"] >= p_hi) & (valid["p_loss"] >= p_abs)
            hit = triggered & (valid["loss_label"] == 1.0)
            false_alarm = triggered & ~(valid["loss_label"] == 1.0)
            missed = ~triggered & (valid["loss_label"] == 1.0)
            n_trig = int(triggered.sum())
            rows.append(
                {
                    "截面分位阈值": float(p_hi),
                    "绝对概率阈值": float(p_abs),
                    "触发笔数": n_trig,
                    "触发占比": float(n_trig / total) if total else np.nan,
                    "事件拦截率": float(int(hit.sum()) / events) if events else np.nan,
                    "误杀率": float(int(false_alarm.sum()) / n_trig) if n_trig else np.nan,
                    "误杀且上涨占比": (
                        float((false_alarm & (valid["terminal_return"] > 0)).sum() / n_trig)
                        if n_trig
                        else np.nan
                    ),
                    "漏报率": float(int(missed.sum()) / events) if events else np.nan,
                    "触发样本平均事后收益": (
                        float(valid.loc[triggered, "terminal_return"].mean()) if n_trig else np.nan
                    ),
                    "未触发样本平均事后收益": (
                        float(valid.loc[~triggered, "terminal_return"].mean())
                        if int((~triggered).sum())
                        else np.nan
                    ),
                    "事后可避免损失合计": float(valid.loc[triggered, "avoidable_loss"].sum()),
                }
            )
    return pd.DataFrame(rows, columns=SCAN_COLUMNS_ZH)


def write_ledger(ledger: pd.DataFrame, path: Path) -> None:
    """落盘风险台账（中文表头，utf-8-sig 便于 Excel 直接打开）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    select_chinese(ledger, _LEDGER_MAP, _LEDGER_KEYS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    logger.info(f"风险台账已写入: {path}（{len(ledger)} 行）")


def write_trigger_list(triggers: pd.DataFrame, path: Path) -> None:
    """落盘触发清单（中文表头）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if triggers.empty:
        Path(path).write_text("", encoding="utf-8")
        logger.info(f"触发清单为空（无命中与漏报），已写入空文件: {path}")
        return
    select_chinese(triggers, _TRIGGER_MAP, _TRIGGER_KEYS).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    logger.info(f"触发清单已写入: {path}（{len(triggers)} 行）")


def write_threshold_scan(scan: pd.DataFrame, path: Path) -> None:
    """落盘阈值扫描（中文表头）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    scan.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"阈值扫描已写入: {path}（{len(scan)} 行）")


def summarize_ledger(ledger: pd.DataFrame) -> Dict[str, float]:
    """台账概览（控制台展示用）。"""
    valid = ledger[ledger["loss_label"].notna()]
    return {
        "行数": float(len(ledger)),
        "有效标签行数": float(len(valid)),
        "事件率": float((valid["loss_label"] == 1.0).mean()) if len(valid) else float("nan"),
        "平均风险概率": float(ledger["p_loss"].mean()) if len(ledger) else float("nan"),
        "日期数": float(ledger["date"].nunique()) if len(ledger) else 0.0,
        "持仓记录数": float(ledger.groupby(["date", "ts_code"]).ngroups) if len(ledger) else 0.0,
    }

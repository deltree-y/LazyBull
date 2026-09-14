"""terminal_loss 政策层 E2：条件暴露门控（regime × 组合平均风险）的校准与离线评估。

阶段定位（P2-2 的一阶筛选）：本模块只做**离线**规则校准与评估，不改回测引擎、
不产生任何交易指令。它消费 P2-1 已经落盘的中文表头风险台账
（``LEDGER_COLUMNS_ZH``），把每行"某持仓的某个剩余持有期"聚合为**日级组合面板**，
再按预登记口径拟合阈值并评估三臂：

============  ==========================================================
臂            触发条件
============  ==========================================================
``regime``    仅 regime：``市场波动状态 >= 校准段分位阈值``
``score``     仅模型分数字：``组合平均风险概率 >= 校准段全段分位阈值``
``combined``  E2：``市场波动状态 >= 分位阈值`` 且 ``组合平均风险概率 >= 层内分位阈值``
============  ==========================================================

**为什么必须带 ``regime`` 与 ``score`` 两条对照臂**：2026-09-14 三批复核显示"纯市场波动分层"
与"层内用市场波动排序"均无区分度（事件率比 0.87–1.04），而层内模型分数排序有效（1.77–3.06）；
只报 ``combined`` 无法区分增量来自 regime 还是模型分数。

硬约束
------

- **禁止前视**：阈值只在**校准段**上拟合，评估段必须与之**时间不重叠**（重叠直接报错）；
  逐日判定里的"层内得分分位"是对**校准样本层内分布**的经验分位，不使用评估段信息。
- **口径唯一**：事后收益/异常亏损直接取台账列（台账由 ``policy_sidecar`` 经
  ``labels.build_terminal_loss_labels`` 生成）；本模块不复制标签公式。
- **标签状态过滤显式化**：非 ``valid`` 行（``sigma_unavailable`` / ``endpoint_missing`` /
  ``immature``）必须剔除并**告警计数**，禁止静默按 NaN 参与均值。
- **市场波动状态必须逐日唯一**：同一天出现多个不同取值说明台账口径有问题，直接报错
  （否则 regime 阈值语义不明）。
- **降暴露系数**取值必须落于 (0, 1]；``1.0`` 表示不降暴露（只判定不动仓）。
- **评估口径是"一阶代理"，不是 NAV**：见 :func:`evaluate_gate` 文档；组合级净值对比
  必须由回测引擎的 shadow 模式（P2-3）给出，不得用本模块结论替代。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ...common.sidecar_schema import (
    GATE_CALIBRATION_COLUMNS_ZH,
    GATE_DAILY_COLUMNS_ZH,
    GATE_EVAL_COLUMNS_ZH,
    select_chinese,
)

#: 支持的三臂（顺序即输出顺序）
ARMS: Tuple[str, ...] = ("regime", "score", "combined")

#: 臂的中文展示名（用户可见产物一律中文）
ARM_LABELS: Dict[str, str] = {
    "regime": "对照A纯regime",
    "score": "对照B纯模型分数",
    "combined": "E2(regime×分数)",
}

#: 评估口径（P&L 代理基准）
SCOPES: Tuple[str, ...] = ("全部触发日", "首触日")

#: 评估分组（按 regime 阈值切层；"全部" 为整体）
GROUPS: Tuple[str, ...] = ("全部", "高波动层", "低波动层")

#: 阈值口径：固定校准段阈值 / 滚动分位阈值
THRESHOLD_MODES: Tuple[str, ...] = ("fixed", "rolling")

THRESHOLD_MODE_LABELS: Dict[str, str] = {
    "fixed": "固定校准段",
    "rolling": "滚动分位",
}

#: 台账中文表头 → 本模块内部键（只取所需列；缺列报错）
_LEDGER_COLUMN_MAP: Dict[str, str] = {
    "日期": "date",
    "风险模型折": "fold",
    "股票代码": "ts_code",
    "持仓权重": "weight",
    "风险概率": "p_loss",
    "市场波动状态": "mkt_vol_20",
    "事后实际收益": "terminal_return",
    "是否异常亏损": "loss_label",
    "标签状态": "label_status",
}

#: 校准/逐日/评估表的内部键（顺序即输出列序）
_CALIBRATION_KEYS: List[str] = [
    "arm",
    "calib_start",
    "calib_end",
    "calib_days",
    "regime_quantile",
    "regime_threshold",
    "score_quantile",
    "score_threshold",
    "de_exposure_multiplier",
    "layer_days",
    "calib_trigger_days",
    "calib_trigger_share",
]

_DAILY_KEYS: List[str] = [
    "arm",
    "date",
    "fold",
    "threshold_mode",
    "window_days",
    "regime_threshold",
    "mkt_vol_20",
    "vol_percentile",
    "p_loss_mean",
    "score_threshold",
    "layer_score_percentile",
    "holdings",
    "weight_sum",
    "day_weighted_return",
    "day_mean_return",
    "triggered",
    "first_trigger",
    "exposure_multiplier",
    "in_calibration",
]

_EVAL_KEYS: List[str] = [
    "arm",
    "scope",
    "group",
    "fold",
    "days",
    "trigger_days",
    "trigger_share",
    "first_trigger_days",
    "first_trigger_share",
    "trigger_mean_return",
    "nontrigger_mean_return",
    "return_gap",
    "trigger_loss_day_rate",
    "nontrigger_loss_day_rate",
    "trigger_event_rate",
    "nontrigger_event_rate",
    "gain_term",
    "cost_term",
    "net_daily_delta",
]

#: 同日市场波动状态的允许波动（超过即认为是口径错误）
_REGIME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ExposureGateConfig:
    """E2 暴露门控的单臂配置（单变量直读、单默认值，无多层回退）。

    Args:
        arm: ``regime`` / ``score`` / ``combined``
        regime_quantile: regime 阈值分位（校准段 `市场波动状态` 的分位）
        score_quantile: 得分阈值分位（``score`` 臂取全段分位；``combined`` 臂取层内分位）
        de_exposure_multiplier: 触发日目标暴露系数 λ ∈ (0, 1]
        cost_bps: 单边成本（基点），用于估算降暴露的往返成本（买入+卖出）
        min_layer_days: ``combined`` 臂层内校准日数下限（低于此值拒绝校准）
    """

    arm: str = "combined"
    regime_quantile: float = 2.0 / 3.0
    score_quantile: float = 0.5
    de_exposure_multiplier: float = 0.5
    cost_bps: float = 15.0
    min_layer_days: int = 20

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"未知臂 {self.arm!r}；合法取值 {list(ARMS)}")
        if not 0.0 < self.regime_quantile < 1.0:
            raise ValueError(f"regime_quantile 必须落于 (0, 1)，当前 {self.regime_quantile}")
        if self.arm != "regime" and not 0.0 < self.score_quantile < 1.0:
            raise ValueError(f"score_quantile 必须落于 (0, 1)，当前 {self.score_quantile}")
        if not 0.0 < self.de_exposure_multiplier <= 1.0:
            raise ValueError(
                f"de_exposure_multiplier 必须落于 (0, 1]，当前 {self.de_exposure_multiplier}"
            )
        if self.cost_bps < 0.0:
            raise ValueError(f"cost_bps 不能为负，当前 {self.cost_bps}")
        if self.min_layer_days < 1:
            raise ValueError(f"min_layer_days 至少为 1，当前 {self.min_layer_days}")


@dataclass(frozen=True)
class GateCalibration:
    """一个臂在校准段上拟合出的阈值（冻结后用于评估段，禁止再拟合）。"""

    arm: str
    calib_start: str
    calib_end: str
    calib_days: int
    regime_quantile: Optional[float]
    regime_threshold: Optional[float]
    score_quantile: Optional[float]
    score_threshold: Optional[float]
    de_exposure_multiplier: float
    layer_days: int
    calib_trigger_days: int
    calib_trigger_share: float
    #: 层内校准得分样本（仅用于计算逐日"层内得分分位"，不落盘）
    layer_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def as_row(self) -> Dict[str, object]:
        """转为校准表一行（内部键）。"""
        return {
            "arm": ARM_LABELS[self.arm],
            "calib_start": self.calib_start,
            "calib_end": self.calib_end,
            "calib_days": self.calib_days,
            "regime_quantile": self.regime_quantile,
            "regime_threshold": self.regime_threshold,
            "score_quantile": self.score_quantile,
            "score_threshold": self.score_threshold,
            "de_exposure_multiplier": self.de_exposure_multiplier,
            "layer_days": self.layer_days,
            "calib_trigger_days": self.calib_trigger_days,
            "calib_trigger_share": self.calib_trigger_share,
        }


def read_ledger_frames(paths: Sequence[Path]) -> pd.DataFrame:
    """读取中文表头风险台账并规范为内部键；非 valid 标签行剔除并告警。

    Args:
        paths: 台账 CSV 路径列表（``policy_sidecar`` 落盘产物）

    Returns:
        内部键长表（一行 = 某持仓的某个剩余持有期）

    Raises:
        ValueError: 缺列、文件为空或全部行标签非 valid
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(Path(path), encoding="utf-8-sig")
        missing = [zh for zh in _LEDGER_COLUMN_MAP if zh not in frame.columns]
        if missing:
            raise ValueError(
                f"{Path(path).name} 缺少台账列 {sorted(missing)}；"
                f"表结构由 sidecar_schema 冻结，禁止降级读取"
            )
        frame = frame.rename(columns=dict(_LEDGER_COLUMN_MAP))
        frame = frame[list(_LEDGER_COLUMN_MAP.values())].copy()
        frame["date"] = frame["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        frame["ts_code"] = frame["ts_code"].astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("未提供任何风险台账文件")
    ledger = pd.concat(frames, ignore_index=True)
    if ledger.empty:
        raise ValueError("风险台账为空，无法校准/评估暴露门控")

    status_counts = ledger["label_status"].value_counts(dropna=False).to_dict()
    invalid = ledger["label_status"] != "valid"
    if int(invalid.sum()):
        logger.warning(
            f"剔除 {int(invalid.sum())} 行非 valid 标签（{status_counts}）；"
            f"禁止以 NaN 事后收益参与均值"
        )
    ledger = ledger[~invalid].copy()
    if ledger.empty:
        raise ValueError(f"台账全部行标签非 valid（{status_counts}），无可评估样本")
    return ledger


def build_daily_frame(ledger: pd.DataFrame) -> pd.DataFrame:
    """把持仓级台账聚合为日级组合面板（评估与校准的唯一输入）。

    列：``date`` / ``fold`` / ``mkt_vol_20`` / ``p_loss_mean`` / ``holdings`` /
    ``weight_sum`` / ``day_weighted_return``（按持仓权重加权的事后收益）/
    ``day_mean_return``（等权事后收益）/ ``loss_day`` / ``event_rate``。

    Raises:
        ValueError: 同日 ``市场波动状态`` 不唯一（说明台账口径异常）
    """
    required = ["date", "fold", "mkt_vol_20", "p_loss", "weight", "terminal_return", "loss_label"]
    missing = [c for c in required if c not in ledger.columns]
    if missing:
        raise ValueError(f"台账缺少聚合所需内部列 {sorted(missing)}")

    grouped = ledger.groupby("date", sort=True)
    spread = grouped["mkt_vol_20"].agg(lambda s: float(s.max() - s.min()))
    bad = spread[spread > _REGIME_TOLERANCE]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} 个日期的市场波动状态不唯一（最大跨度 {bad.max():.6f}）；"
            f"regime 阈值语义不明，禁止继续（样例 {list(bad.index[:3])}）"
        )

    weight_sum = grouped["weight"].sum()
    weighted = grouped.apply(
        lambda part: float((part["weight"] * part["terminal_return"]).sum()), include_groups=False
    )
    daily = pd.DataFrame(
        {
            "date": weight_sum.index.astype(str),
            "mkt_vol_20": grouped["mkt_vol_20"].mean().to_numpy(),
            "p_loss_mean": grouped["p_loss"].mean().to_numpy(),
            "holdings": grouped.size().to_numpy(),
            "weight_sum": weight_sum.to_numpy(),
            "day_mean_return": grouped["terminal_return"].mean().to_numpy(),
            "event_rate": grouped["loss_label"].mean().to_numpy(),
        }
    )
    daily["day_weighted_return"] = np.where(
        daily["weight_sum"] > 0, weighted.to_numpy() / daily["weight_sum"].to_numpy(), np.nan
    )
    daily["loss_day"] = daily["day_weighted_return"] < 0
    fold_map = grouped["fold"].agg(lambda s: str(s.iloc[0]))
    daily["fold"] = daily["date"].map(fold_map)
    return daily.sort_values("date").reset_index(drop=True)


def _segment_mask(daily: pd.DataFrame, start: str, end: str) -> pd.Series:
    if start > end:
        raise ValueError(f"区间起点 {start} 晚于终点 {end}")
    return (daily["date"] >= start) & (daily["date"] <= end)


def calibrate_gate(
    daily: pd.DataFrame, config: ExposureGateConfig, calib_start: str, calib_end: str
) -> GateCalibration:
    """在**校准段**上拟合阈值（评估段不得参与；重叠由 :func:`assert_disjoint` 拦截）。

    Raises:
        ValueError: 校准段无样本，或 ``combined`` 臂层内样本不足 ``min_layer_days``
    """
    mask = _segment_mask(daily, calib_start, calib_end)
    segment = daily[mask]
    if segment.empty:
        raise ValueError(f"校准段 {calib_start}~{calib_end} 无交易日，无法拟合阈值")

    regime_threshold = float(segment["mkt_vol_20"].quantile(config.regime_quantile))
    layer = segment[segment["mkt_vol_20"] >= regime_threshold]
    score_threshold: Optional[float] = None
    score_quantile: Optional[float] = None
    layer_scores = np.array([], dtype=float)

    # regime 阈值对所有臂都拟合：它是**分层报告维度**（高/低波动层）；
    # 只有 ``regime`` / ``combined`` 臂把它用作**触发条件**，``score`` 臂不参与触发。
    if config.arm == "score":
        score_quantile = config.score_quantile
        score_threshold = float(segment["p_loss_mean"].quantile(config.score_quantile))
    elif config.arm == "combined":
        if len(layer) < config.min_layer_days:
            raise ValueError(
                f"校准段层内日数 {len(layer)} 低于下限 {config.min_layer_days}；"
                f"层内阈值不可靠，请扩大校准段或降低 regime 分位"
            )
        score_quantile = config.score_quantile
        score_threshold = float(layer["p_loss_mean"].quantile(config.score_quantile))
        layer_scores = layer["p_loss_mean"].to_numpy(dtype=float)

    judged = _judge(segment, config, regime_threshold, score_threshold)
    return GateCalibration(
        arm=config.arm,
        calib_start=calib_start,
        calib_end=calib_end,
        calib_days=int(len(segment)),
        regime_quantile=config.regime_quantile,
        regime_threshold=regime_threshold,
        score_quantile=score_quantile,
        score_threshold=score_threshold,
        de_exposure_multiplier=config.de_exposure_multiplier,
        layer_days=int(len(layer)),
        calib_trigger_days=int(judged["triggered"].sum()),
        calib_trigger_share=float(judged["triggered"].mean()),
        layer_scores=layer_scores,
    )


def assert_disjoint(calib_start: str, calib_end: str, eval_start: str, eval_end: str) -> None:
    """校准段与评估段必须时间不重叠（预登记精神：阈值不得在评估段上再拟合）。"""
    if not (calib_end < eval_start or eval_end < calib_start):
        raise ValueError(
            f"校准段 {calib_start}~{calib_end} 与评估段 {eval_start}~{eval_end} 重叠；"
            f"禁止用评估段信息拟合阈值"
        )


def _judge(
    frame: pd.DataFrame,
    arm: str,
    regime_threshold: pd.Series,
    score_threshold: pd.Series,
) -> pd.DataFrame:
    """按臂给出触发判定（阈值逐日提供，允许固定或滚动口径）。"""
    scores = frame["p_loss_mean"].to_numpy(dtype=float)
    vols = frame["mkt_vol_20"].to_numpy(dtype=float)
    vol_thr = np.asarray(regime_threshold, dtype=float)
    score_thr = np.asarray(score_threshold, dtype=float)
    if arm == "regime":
        triggered = vols >= vol_thr
    elif arm == "score":
        triggered = scores >= score_thr
    else:
        triggered = (vols >= vol_thr) & (scores >= score_thr)
    out = frame.copy()
    out["triggered"] = triggered
    return out


def apply_gate(
    daily: pd.DataFrame,
    calibration: GateCalibration,
    eval_start: str,
    eval_end: str,
) -> pd.DataFrame:
    """在**评估段**上套用冻结阈值，产出逐日判定表（含首触标记与暴露系数）。

    首触 = 触发且前一交易日未触发；**折内**判定（跨折边界按折内首日处理，
    因为跨折更换模型，触发区间不跨折延续）。
    """
    mask = _segment_mask(daily, eval_start, eval_end)
    segment = daily[mask].copy()
    if segment.empty:
        raise ValueError(f"评估段 {eval_start}~{eval_end} 无交易日")
    segment["regime_threshold"] = calibration.regime_threshold
    segment["score_threshold"] = calibration.score_threshold
    judged = _judge(
        segment,
        calibration.arm,
        segment["regime_threshold"],
        segment["score_threshold"],
    )
    prev = judged.groupby("fold", sort=False)["triggered"].shift(1)
    judged["first_trigger"] = judged["triggered"] & ~prev.fillna(False).astype(bool)
    judged["exposure_multiplier"] = np.where(
        judged["triggered"], calibration.de_exposure_multiplier, 1.0
    )
    judged["vol_percentile"] = judged["mkt_vol_20"].rank(pct=True)
    judged["layer_score_percentile"] = np.nan
    if calibration.layer_scores.size:
        reference = np.sort(calibration.layer_scores)
        current = judged["p_loss_mean"].to_numpy(dtype=float)
        judged["layer_score_percentile"] = reference.searchsorted(current, side="right") / len(
            reference
        )
    judged["arm"] = ARM_LABELS[calibration.arm]
    judged["threshold_mode"] = THRESHOLD_MODE_LABELS["fixed"]
    judged["window_days"] = np.nan
    judged["in_calibration"] = False
    return judged.sort_values("date").reset_index(drop=True)


@dataclass(frozen=True)
class RollingGateConfig:
    """E2 滚动分位口径：阈值随模型水平漂移自归一（不依赖固定校准段）。

    对每个评估日 ``t``，只用 **[t−window_days, t−1] 的完全历史**（严格不含当日）计算：

    - ``regime_threshold`` = 窗口内 `市场波动状态` 的 ``regime_quantile`` 分位
    - ``score_threshold``  = **窗口内高波动层**（vol ≥ 当日 regime 阈值）的
      ``组合平均风险概率`` 的 ``score_quantile`` 分位

    动机（R-004）：模型得分水平随 regime 漂移（2025H2 均值 0.193 vs 2024H2 0.138），
    固定绝对阈值会在“模型系统性高估”的时期触发、低估期不触发。

    Args:
        arm: ``combined`` / ``regime`` / ``score``（语义与固定口径一致）
        window_days: 滚动窗口长度（交易日）
        min_window_days: 窗口内最少交易日，不足则该日**不判定**（跳过并计数告警）
        label: 用户可见的臂标签（需体现口径，如 “E2-滚动分位250日”）
    """

    arm: str = "combined"
    window_days: int = 250
    regime_quantile: float = 2.0 / 3.0
    score_quantile: float = 0.5
    de_exposure_multiplier: float = 0.5
    cost_bps: float = 15.0
    min_window_days: int = 60
    label: str = "E2-滚动分位"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"未知臂 {self.arm!r}；合法取值 {list(ARMS)}")
        if self.window_days < 2:
            raise ValueError(f"window_days 至少为 2，当前 {self.window_days}")
        if self.min_window_days < 2 or self.min_window_days > self.window_days:
            raise ValueError(
                f"min_window_days 必须落于 [2, window_days]，当前 {self.min_window_days}"
            )
        if not 0.0 < self.regime_quantile < 1.0:
            raise ValueError(f"regime_quantile 必须落于 (0, 1)，当前 {self.regime_quantile}")
        if self.arm != "regime" and not 0.0 < self.score_quantile < 1.0:
            raise ValueError(f"score_quantile 必须落于 (0, 1)，当前 {self.score_quantile}")
        if not 0.0 < self.de_exposure_multiplier <= 1.0:
            raise ValueError(
                f"de_exposure_multiplier 必须落于 (0, 1]，当前 {self.de_exposure_multiplier}"
            )
        if self.cost_bps < 0.0:
            raise ValueError(f"cost_bps 不能为负，当前 {self.cost_bps}")


def apply_rolling_gate(
    daily: pd.DataFrame,
    config: RollingGateConfig,
    eval_start: str,
    eval_end: str,
) -> Tuple[pd.DataFrame, int]:
    """按滚动分位阈值逐日判定（严格无前视）。

    Returns:
        (judged, skipped_days)：``skipped_days`` 为窗口不足而被跳过的交易日数
        （这些日子不判定、系数保持 1.0，阈值与分位记空值）
    """
    mask = _segment_mask(daily, eval_start, eval_end)
    segment = daily[mask].copy()
    if segment.empty:
        raise ValueError(f"评估段 {eval_start}~{eval_end} 无交易日")
    ordered = daily.sort_values("date").reset_index(drop=True)
    vol_values = ordered["mkt_vol_20"].to_numpy(dtype=float)
    score_values = ordered["p_loss_mean"].to_numpy(dtype=float)
    index_of = {date: idx for idx, date in enumerate(ordered["date"])}

    vol_thr: List[float] = []
    score_thr: List[float] = []
    vol_pct: List[float] = []
    score_pct: List[float] = []
    skipped = 0
    for date in segment["date"]:
        position = index_of[date]
        start = max(0, position - config.window_days)
        window_vol = vol_values[start:position]
        if window_vol.size < config.min_window_days:
            skipped += 1
            vol_thr.append(np.nan)
            score_thr.append(np.nan)
            vol_pct.append(np.nan)
            score_pct.append(np.nan)
            continue
        current_vol_thr = float(np.quantile(window_vol, config.regime_quantile))
        layer_mask = window_vol >= current_vol_thr
        window_score = score_values[start:position][layer_mask]
        if window_score.size == 0:
            skipped += 1
            vol_thr.append(np.nan)
            score_thr.append(np.nan)
            vol_pct.append(np.nan)
            score_pct.append(np.nan)
            continue
        vol_thr.append(current_vol_thr)
        score_thr.append(
            np.nan
            if config.arm == "regime"
            else float(np.quantile(window_score, config.score_quantile))
        )
        vol_pct.append(float((window_vol <= vol_values[position]).mean()))
        score_pct.append(float((window_score <= score_values[position]).mean()))

    segment["regime_threshold"] = vol_thr
    segment["score_threshold"] = score_thr
    judged = _judge(
        segment,
        config.arm,
        pd.Series(vol_thr, index=segment.index),
        pd.Series(score_thr, index=segment.index),
    )
    judged.loc[judged["regime_threshold"].isna(), "triggered"] = False
    prev = judged.groupby("fold", sort=False)["triggered"].shift(1)
    judged["first_trigger"] = judged["triggered"] & ~prev.fillna(False).astype(bool)
    judged["exposure_multiplier"] = np.where(
        judged["triggered"], config.de_exposure_multiplier, 1.0
    )
    judged["vol_percentile"] = vol_pct
    judged["layer_score_percentile"] = score_pct
    judged["arm"] = config.label
    judged["threshold_mode"] = THRESHOLD_MODE_LABELS["rolling"]
    judged["window_days"] = config.window_days
    judged["in_calibration"] = False
    return judged.sort_values("date").reset_index(drop=True), skipped


def evaluate_gate(
    judged: pd.DataFrame,
    de_exposure_multiplier: float,
    cost_bps: float,
) -> pd.DataFrame:
    """评估单臂的"降暴露"效果（**一阶代理**，非 NAV）。

    口径（见 :data:`SCOPES`）：

    - ``全部触发日``：收益项假设避开了触发日**全部**事后收益
      ``−λ·E[收益 × 1{触发日}]``（上界口径，触发持续期越长越乐观）；
    - ``首触日``：收益项只计**首触日**（真正下指令那一天）的事后收益（保守口径）。

    成本项固定按**触发事件数**（首触日数）计一次往返：``λ·2·cost_bps/1e4 × 首触占比``。
    两口径共用同一成本，便于横向比较。

    价值判据（P2-2 预登记）：**降低"平均亏损日"频率与幅度**；不得以 MaxDD/尾部损失为判据
    （三批实测显示层内两半最差单日几乎相同）。

    **量纲警告**：台账事后收益是持有期口径（最长 20 交易日、逐日行重叠），
    ``收益项 × 交易日数`` **不能**读作净值幅度（实测相差一个数量级）；本函数只能读方向与排序。

    Returns:
        一行 = 臂 × 口径 × 分组（``fold`` 固定为 "全部"；按折口径见 :func:`evaluate_gate_by_fold`）
    """
    rows: List[Dict[str, object]] = []
    for scope in SCOPES:
        for group in GROUPS:
            subset = _group_slice(judged, group)
            if subset.empty:
                continue
            rows.append(
                _eval_row(subset, scope, group, "全部", de_exposure_multiplier, cost_bps)
            )
    return pd.DataFrame(rows)


def evaluate_gate_by_fold(
    judged: pd.DataFrame, de_exposure_multiplier: float, cost_bps: float
) -> pd.DataFrame:
    """按折拆开的评估（稳健性口径：单折口径固定为 全部触发日 × 全部持仓）。"""
    rows: List[Dict[str, object]] = []
    for fold, part in judged.groupby("fold", sort=True):
        rows.append(
            _eval_row(part, "全部触发日", "全部", str(fold), de_exposure_multiplier, cost_bps)
        )
    return pd.DataFrame(rows)


def _group_slice(judged: pd.DataFrame, group: str) -> pd.DataFrame:
    """按**逐日** regime 阈值切层（固定口径下该阈值退化为常数）。"""
    if group == "全部":
        return judged
    if "regime_threshold" not in judged.columns:
        raise ValueError(f"判定表缺少 regime_threshold 列，无法按 {group} 切层")
    threshold = judged["regime_threshold"]
    if group == "高波动层":
        return judged[judged["mkt_vol_20"] >= threshold]
    if group == "低波动层":
        return judged[judged["mkt_vol_20"] < threshold]
    raise ValueError(f"未知分组 {group!r}；合法取值 {list(GROUPS)}")


def _eval_row(
    subset: pd.DataFrame,
    scope: str,
    group: str,
    fold: str,
    lam: float,
    cost_bps: float,
) -> Dict[str, object]:
    days = int(len(subset))
    trigger_key = "triggered" if scope == "全部触发日" else "first_trigger"
    trig = subset[subset[trigger_key]]
    nontrig = subset[~subset[trigger_key]]
    ret_trig = float(trig["day_weighted_return"].mean()) if len(trig) else np.nan
    ret_nont = float(nontrig["day_weighted_return"].mean()) if len(nontrig) else np.nan
    share = float(len(trig)) / days
    first_share = float(subset["first_trigger"].mean())
    gain = -lam * (share * ret_trig) if len(trig) else 0.0
    cost = lam * 2.0 * cost_bps / 1e4 * first_share
    return {
        "arm": subset["arm"].iloc[0],
        "scope": scope,
        "group": group,
        "fold": fold,
        "days": days,
        "trigger_days": int(len(trig)),
        "trigger_share": share,
        "first_trigger_days": int(subset["first_trigger"].sum()),
        "first_trigger_share": first_share,
        "trigger_mean_return": ret_trig,
        "nontrigger_mean_return": ret_nont,
        "return_gap": ret_trig - ret_nont if len(trig) and len(nontrig) else np.nan,
        "trigger_loss_day_rate": float(trig["loss_day"].mean()) if len(trig) else np.nan,
        "nontrigger_loss_day_rate": float(nontrig["loss_day"].mean()) if len(nontrig) else np.nan,
        "trigger_event_rate": float(trig["event_rate"].mean()) if len(trig) else np.nan,
        "nontrigger_event_rate": float(nontrig["event_rate"].mean()) if len(nontrig) else np.nan,
        "gain_term": gain,
        "cost_term": cost,
        "net_daily_delta": gain - cost,
    }


def run_exposure_gate(
    ledger: pd.DataFrame,
    configs: Sequence[ExposureGateConfig],
    calib_segment: Tuple[str, str],
    eval_segment: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """校准 + 评估三臂，返回 (校准表, 逐日判定拼接表, (评估表, 逐折评估表))。"""
    calib_start, calib_end = calib_segment
    eval_start, eval_end = eval_segment
    assert_disjoint(calib_start, calib_end, eval_start, eval_end)
    daily = build_daily_frame(ledger)

    calib_rows: List[Dict[str, object]] = []
    daily_frames: List[pd.DataFrame] = []
    eval_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    for config in configs:
        calibration = calibrate_gate(daily, config, calib_start, calib_end)
        judged = apply_gate(daily, calibration, eval_start, eval_end)
        evaluated = evaluate_gate(judged, calibration.de_exposure_multiplier, config.cost_bps)
        calib_rows.append(calibration.as_row())
        daily_frames.append(judged)
        eval_frames.append(evaluated)
        fold_frames.append(
            evaluate_gate_by_fold(judged, calibration.de_exposure_multiplier, config.cost_bps)
        )
        logger.info(
            f"臂 {ARM_LABELS[config.arm]}: 校准 {calibration.calib_days} 日"
            f"（层内 {calibration.layer_days} 日，触发占比 {calibration.calib_trigger_share:.1%}）"
            f" → 评估段 {int(evaluated['days'].iloc[0])} 日"
        )
    calibration_frame = pd.DataFrame(calib_rows)
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    overall = pd.concat(eval_frames, ignore_index=True)
    by_fold = pd.concat(fold_frames, ignore_index=True)
    return calibration_frame, daily_frame, (overall, by_fold)


def run_rolling_gate(
    ledger: pd.DataFrame,
    configs: Sequence[RollingGateConfig],
    eval_segment: Tuple[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """滚动分位口径：无需校准段，逐日自归一阈值，返回结构同 :func:`run_exposure_gate`。

    校准表在此口径下退化为“口径说明行”（记录窗口长度、分位参数、跳过日数），
    使两份产物列结构完全一致、可直接对比。
    """
    eval_start, eval_end = eval_segment
    daily = build_daily_frame(ledger)
    calib_rows: List[Dict[str, object]] = []
    daily_frames: List[pd.DataFrame] = []
    eval_frames: List[pd.DataFrame] = []
    fold_frames: List[pd.DataFrame] = []
    for config in configs:
        judged, skipped = apply_rolling_gate(daily, config, eval_start, eval_end)
        evaluated = evaluate_gate(judged, config.de_exposure_multiplier, config.cost_bps)
        calib_rows.append({
            "arm": config.label,
            "calib_start": "(滚动窗口)",
            "calib_end": "(滚动窗口)",
            "calib_days": config.window_days,
            "regime_quantile": config.regime_quantile,
            "regime_threshold": np.nan,
            "score_quantile": config.score_quantile if config.arm != "regime" else np.nan,
            "score_threshold": np.nan,
            "de_exposure_multiplier": config.de_exposure_multiplier,
            "layer_days": int(judged["triggered"].notna().sum()) - skipped,
            "calib_trigger_days": int(judged["triggered"].sum()),
            "calib_trigger_share": float(judged["triggered"].mean()),
        })
        daily_frames.append(judged)
        eval_frames.append(evaluated)
        fold_frames.append(
            evaluate_gate_by_fold(judged, config.de_exposure_multiplier, config.cost_bps)
        )
        logger.info(
            f"臂 {config.label}: 窗口 {config.window_days} 日，窗口不足跳过 {skipped} 日，"
            f"触发占比 {judged['triggered'].mean():.1%}"
            f" → 评估段 {int(evaluated['days'].iloc[0])} 日"
        )
    calibration_frame = pd.DataFrame(calib_rows)
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    overall = pd.concat(eval_frames, ignore_index=True)
    by_fold = pd.concat(fold_frames, ignore_index=True)
    return calibration_frame, daily_frame, (overall, by_fold)


def to_chinese_tables(
    calibration_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    eval_frames: Tuple[pd.DataFrame, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """把四张内部键表转为中文表头（落盘与打印共用同一实现）。"""
    overall, by_fold = eval_frames
    return {
        "calibration": select_chinese(
            calibration_frame,
            _columns_map(GATE_CALIBRATION_COLUMNS_ZH, _CALIBRATION_KEYS),
            _CALIBRATION_KEYS,
        ),
        "daily": select_chinese(
            daily_frame, _columns_map(GATE_DAILY_COLUMNS_ZH, _DAILY_KEYS), _DAILY_KEYS
        ),
        "eval": select_chinese(
            overall, _columns_map(GATE_EVAL_COLUMNS_ZH, _EVAL_KEYS), _EVAL_KEYS
        ),
        "eval_by_fold": select_chinese(
            by_fold, _columns_map(GATE_EVAL_COLUMNS_ZH, _EVAL_KEYS), _EVAL_KEYS
        ),
    }


def write_gate_outputs(
    out_dir: Path,
    calibration_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    eval_frames: Tuple[pd.DataFrame, pd.DataFrame],
) -> Dict[str, Path]:
    """落盘四张中文表头产物（校准 / 逐日判定 / 评估 / 按折评估）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = to_chinese_tables(calibration_frame, daily_frame, eval_frames)
    paths = {
        "calibration": out_dir / "暴露门控校准.csv",
        "daily": out_dir / "暴露门控逐日判定.csv",
        "eval": out_dir / "暴露门控评估.csv",
        "eval_by_fold": out_dir / "暴露门控评估_按折.csv",
    }
    for key, path in paths.items():
        tables[key].to_csv(path, index=False, encoding="utf-8-sig")
    return paths


def _columns_map(chinese: Sequence[str], keys: Sequence[str]) -> Dict[str, str]:
    """{内部键: 中文列名}；数量不一致直接报错（防止两处漂移）。"""
    if len(chinese) != len(keys):
        raise ValueError(
            f"中文表头与内部键数量不一致（{len(chinese)} vs {len(keys)}）；"
            f"sidecar_schema 与 exposure_gate 必须同步"
        )
    return dict(zip(keys, chinese))

"""分块重采样统计：门禁区间与成对比较（方案第 6 节规则 4/5）

三类用途：

1. **折级门禁重判**：8 折 ``lift`` 的点估计不足以说明"lift 最小值 ≥ 阈值"
   是否稳健——余量薄时结论由超参选择决定。以折为复制单位做 fold-level
   bootstrap，给出"均值 lift"与"最小 lift"的区间及
   ``P(最小 lift ≥ 阈值)``。
2. **逐折区间**：折内 ES 段按连续交易日分块（主块长 40 日、20/60 敏感性）
   做 moving-block bootstrap，给出单折 lift 区间。
3. **成对比较**（第二阶段七组对照）：两策略共用同一日期块抽样，报告指标
   差值的区间——分别看两条独立区间是否重叠不是有效比较（方差大得多）。

块单位固定为**完整交易日截面**：同日跨股的市场共同冲击是相关结构中最
主要的一维（方案第 6 节规则 4），因此重采样按日期块进行，块内全部行整块
进出；重叠块（moving block）用于有限样本下保留块边界附近的时序依赖。

所有随机性来自固定 ``seed``，结果可复现；重采样只覆盖采样不确定性，不含
重新训练与选参不确定性，结论表述不得越界。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss

#: 组合级成对比较的主块长（交易日，方案第 6 节规则 4 预登记的研究起点，
#: 用于多年 OOS 上的策略对比，如第二阶段七组对照）
PRIMARY_BLOCK_DAYS = 40

#: 组合级块长敏感性
BLOCK_DAYS_SENSITIVITY: Sequence[int] = (20, 40, 60)

#: ES 段单折区间的主块长：ES 窗口只有 6 个月（且标签抽样后仅约 40 个交易日），
#: 方案第 6 节的 40 日块是**组合级多年 OOS** 口径，直接照搬到 6 个月 ES 段会
#: 使块数不足（甚至块长 ≥ 交易日数，重采样退化为原样本）。
ES_BLOCK_DAYS = 10

#: ES 段块长敏感性
ES_BLOCK_DAYS_SENSITIVITY: Sequence[int] = (5, 10, 20)


@dataclass(frozen=True)
class BootstrapConfig:
    """重采样配置。

    Attributes:
        block_days: 块长（连续交易日数）
        n_resamples: 重采样次数
        seed: 随机种子（可复现）
        ci: 区间置信水平
    """

    block_days: int = PRIMARY_BLOCK_DAYS
    n_resamples: int = 200
    seed: int = 42
    ci: float = 0.90


def lift(y_true: np.ndarray, p_pred: np.ndarray) -> Optional[float]:
    """lift = PR-AUC / 事件率（与 summarize_terminal_risk_wf 同口径）。"""
    y_true = np.asarray(y_true)
    rate = float(y_true.mean()) if y_true.size else 0.0
    if rate <= 0:
        return None
    return float(average_precision_score(y_true, np.asarray(p_pred, dtype=float))) / rate


_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], Optional[float]]] = {
    "lift": lift,
    "pr_auc": lambda y, p: float(average_precision_score(y, np.asarray(p, dtype=float))),
    "brier": lambda y, p: float(brier_score_loss(y, np.asarray(p, dtype=float))),
    "event_rate": lambda y, p: float(np.asarray(y).mean()),
}


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], Optional[float]]:
    """按名取指标函数（未知名称明确报错，不静默回退）。"""
    if name not in _METRICS:
        raise KeyError(f"未知指标 '{name}'，可用: {sorted(_METRICS)}")
    return _METRICS[name]


#: daynorm 口径分数列名（当日截面百分位；见 ``add_day_percentile_score``）
DAY_NORM_SCORE_COL = "p_loss_daypct"


def add_day_percentile_score(
    df: pd.DataFrame,
    src_col: str = "p_loss",
    dst_col: str = DAY_NORM_SCORE_COL,
) -> pd.DataFrame:
    """追加当日截面百分位分数列（daynorm 口径）。

    用途（v0.110.0 门禁双判据）：原始 ``p_loss`` 的池化 PR-AUC 把两种能力
    混在一起——跨日期水平对齐（regime/校准）与当日截面排序。按 ``trade_date``
    分组取百分位后，池化指标只反映**当日截面排序**，可与 raw 口径并列判定。

    逐日单调变换：不改变当日截面排序，只去掉跨日水平差异。缺失值保持缺失
    （不填 0.5，避免把"无分数"伪装成中性排序）。

    Args:
        df: 含 trade_date 与分数列的评估行
        src_col: 源分数列（默认 ``p_loss``）
        dst_col: 目标列名（默认 ``DAY_NORM_SCORE_COL``）

    Returns:
        副本 DataFrame，含 ``dst_col``

    Raises:
        ValueError: 评估行为空或缺源分数列
    """
    if df is None or df.empty:
        raise ValueError("评估行为空，无法生成当日截面百分位分数")
    if src_col not in df.columns:
        raise ValueError(f"缺少源分数列 {src_col!r}，无法生成 {dst_col}")
    out = df.copy()
    out[dst_col] = out.groupby("trade_date")[src_col].rank(pct=True)
    return out


def _day_slices(df: pd.DataFrame) -> tuple:
    """按交易日切行索引（输入按 trade_date 稳定排序）。

    Returns:
        (dates_sorted, order, uniq_dates, first_positions, day_sizes)
    """
    dates = df["trade_date"].astype(str).to_numpy()
    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]
    uniq, first, counts = np.unique(sorted_dates, return_index=True, return_counts=True)
    return sorted_dates, order, uniq, first, counts


def _resample_indices(
    rng: np.random.Generator,
    uniq: np.ndarray,
    first: np.ndarray,
    counts: np.ndarray,
    order: np.ndarray,
    block_days: int,
) -> np.ndarray:
    """按重叠日期块重采样行索引（块内整截面进出）。

    要求 ``block_days < len(uniq)``：块长等于或超过交易日数时，可能的起点
    只有一个，重采样退化为原样本（区间宽度归一为 0，得出"零不确定性"的
    错误结论）。该约束由调用方（moving_block_metric_ci）显式校验。
    """
    n_days = len(uniq)
    block = int(min(max(block_days, 1), n_days))
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days - block + 1, size=n_blocks)
    pieces = []
    for s in starts:
        for j in range(s, s + block):
            pieces.append(order[first[j] : first[j] + counts[j]])
    return np.concatenate(pieces) if pieces else np.array([], dtype=int)


def moving_block_metric_ci(
    df: pd.DataFrame,
    metric: str = "lift",
    config: Optional[BootstrapConfig] = None,
    score_col: str = "p_loss",
) -> Dict[str, Any]:
    """按日期块重采样给出指标区间（单折/单策略）。

    Args:
        df: 含 trade_date / loss_label 与分数列的评估行
        metric: 指标名（lift / pr_auc / brier / event_rate）
        config: 重采样配置
        score_col: 分数列名。``p_loss`` = raw 口径（跨日水平 + 当日截面
            排序混合）；``DAY_NORM_SCORE_COL`` = daynorm 口径（只反映当日
            截面排序，由 ``add_day_percentile_score`` 生成）。两个口径是
            并列判据（v0.110.0），不得只报其中之一。

    Returns:
        dict：point / ci_low / ci_high / std / n_days / block_days / n_resamples
            / score_col

    Raises:
        ValueError: 块长不小于交易日数（重采样会退化为原样本），或缺分数列
    """
    cfg = config or BootstrapConfig()
    fn = get_metric(metric)
    if df is None or df.empty:
        raise ValueError("评估行为空，无法重采样")
    frame = df.sort_values("trade_date", kind="stable").reset_index(drop=True)
    if score_col not in frame.columns:
        raise ValueError(
            f"评估行缺少分数列 {score_col!r}（列: {sorted(frame.columns)}）；"
            f"daynorm 口径请先用 add_day_percentile_score 生成 {DAY_NORM_SCORE_COL}"
        )
    y = frame["loss_label"].to_numpy()
    p = frame[score_col].to_numpy(dtype=float)
    point = fn(y, p)
    _, order, uniq, first, counts = _day_slices(frame)
    if cfg.block_days >= len(uniq):
        raise ValueError(
            f"块长 {cfg.block_days} ≥ 交易日数 {len(uniq)}：可选的块起点只有一个，"
            f"重采样会退化为原样本（区间宽度恒为 0）。请减小块长"
            f"（ES 段建议 {ES_BLOCK_DAYS} 日，组合级多年 OOS 建议 {PRIMARY_BLOCK_DAYS} 日）"
        )
    rng = np.random.default_rng(cfg.seed)
    samples: List[float] = []
    for _ in range(cfg.n_resamples):
        idx = _resample_indices(rng, uniq, first, counts, order, cfg.block_days)
        value = fn(y[idx], p[idx])
        if value is not None and np.isfinite(value):
            samples.append(float(value))
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError(f"指标 {metric} 重采样全部无效（事件率为 0？），无法给出区间")
    alpha = (1.0 - cfg.ci) / 2.0
    return {
        "metric": metric,
        "score_col": score_col,
        "point": point,
        "ci_low": float(np.quantile(arr, alpha)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_days": int(len(uniq)),
        "block_days": int(cfg.block_days),
        "n_resamples": int(arr.size),
        "ci_level": cfg.ci,
        "seed": cfg.seed,
    }


def moving_block_metric_sensitivity(
    df: pd.DataFrame,
    metric: str = "lift",
    block_days_list: Sequence[int] = ES_BLOCK_DAYS_SENSITIVITY,
    config: Optional[BootstrapConfig] = None,
    score_col: str = "p_loss",
) -> List[Dict[str, Any]]:
    """主块长 + 敏感性块长下的区间；不可行块长跳过并告警。

    默认使用 ES 段块长（见 ``ES_BLOCK_DAYS`` 注释）：组合级多年 OOS 口径
    请显式传 ``BLOCK_DAYS_SENSITIVITY``。块长超过可用交易日数的项会被跳过
    （而非静默缩小块长），全部不可行时明确报错。``score_col`` 透传到
    ``moving_block_metric_ci``（raw/daynorm 双口径，v0.110.0）。
    """
    base = config or BootstrapConfig()
    n_days = df["trade_date"].astype(str).nunique() if df is not None and not df.empty else 0
    out: List[Dict[str, Any]] = []
    skipped: List[int] = []
    for block in block_days_list:
        if int(block) >= n_days:
            skipped.append(int(block))
            continue
        cfg = BootstrapConfig(
            block_days=int(block),
            n_resamples=base.n_resamples,
            seed=base.seed,
            ci=base.ci,
        )
        out.append(
            moving_block_metric_ci(df, metric=metric, config=cfg, score_col=score_col)
        )
    if skipped:
        logger.warning(f"块长 {skipped} 不小于交易日数 {n_days}，已跳过（不得静默缩小块长）")
    if not out:
        raise ValueError(
            f"全部块长 {list(block_days_list)} 对 {n_days} 个交易日不可行："
            f"ES 段请用更小块长（如 {ES_BLOCK_DAYS} 日）"
        )
    return out


def block_paired_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "lift",
    config: Optional[BootstrapConfig] = None,
    score_col: str = "p_loss",
    key_columns: Sequence[str] = ("trade_date", "ts_code", "h"),
) -> Dict[str, Any]:
    """成对比较：两策略在同一日期块抽样下的指标差值区间（方案 5.5）。

    两组必须覆盖同一批评估对象（同名键集合），否则报错——不同股票池的
    差值不是成对设计，方差口径会失真。组合级比较请用 ``PRIMARY_BLOCK_DAYS``
    （多年 OOS），不要在只有几十个交易日的 ES 段使用。

    Args:
        df_a / df_b: 各含 key_columns + loss_label + p_loss
        metric: 指标名
        config: 重采样配置
        key_columns: 配对键（默认 交易日 × 股票 × 期限）

    Returns:
        dict：delta_point / ci_low / ci_high / prob_positive / n_days 等
    """
    cfg = config or BootstrapConfig()
    fn = get_metric(metric)
    keys = list(key_columns)
    a = df_a.sort_values(keys, kind="stable").reset_index(drop=True)
    b = df_b.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(a) != len(b):
        raise ValueError(f"成对比较要求两组行数一致: A={len(a)}, B={len(b)}")
    for col in keys:
        if not a[col].astype(str).equals(b[col].astype(str)):
            raise ValueError(f"成对比较要求 {col} 逐行对齐（同一批评估对象），当前不一致")
    if not np.array_equal(a["loss_label"].to_numpy(), b["loss_label"].to_numpy()):
        raise ValueError("成对比较的 loss_label 必须一致（同一标签、不同预测）")

    for frame_name, frame in (("A", a), ("B", b)):
        if score_col not in frame.columns:
            raise ValueError(f"成对比较的 {frame_name} 缺少分数列 {score_col!r}")
    y = a["loss_label"].to_numpy()
    pa = a[score_col].to_numpy(dtype=float)
    pb = b[score_col].to_numpy(dtype=float)
    delta_point = fn(y, pa) - fn(y, pb) if fn(y, pa) is not None and fn(y, pb) is not None else None
    _, order, uniq, first, counts = _day_slices(a)
    if cfg.block_days >= len(uniq):
        raise ValueError(
            f"块长 {cfg.block_days} ≥ 交易日数 {len(uniq)}：重采样会退化为原样本，"
            f"成对差值区间无意义；组合级比较请用小于交易日数的块长"
        )
    rng = np.random.default_rng(cfg.seed)
    deltas: List[float] = []
    for _ in range(cfg.n_resamples):
        idx = _resample_indices(rng, uniq, first, counts, order, cfg.block_days)
        va, vb = fn(y[idx], pa[idx]), fn(y[idx], pb[idx])
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
            continue
        deltas.append(float(va - vb))
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("成对差值重采样全部无效，无法给出区间")
    alpha = (1.0 - cfg.ci) / 2.0
    return {
        "metric": metric,
        "score_col": score_col,
        "delta_point": delta_point,
        "delta_mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, alpha)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        "prob_positive": float((arr > 0).mean()),
        "n_days": int(len(uniq)),
        "block_days": int(cfg.block_days),
        "n_resamples": int(arr.size),
        "ci_level": cfg.ci,
        "seed": cfg.seed,
    }


# ── 折级门禁（8 折 lift → 区间与通过概率）────────────────────────────


def fold_level_gate(
    fold_lifts: Sequence[float],
    lift_min_threshold: float = 1.1,
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.90,
) -> Dict[str, Any]:
    """以折为复制单位重采样，给出折间统计与均值区间。

    **口径边界（勿误用）**：8 折是 8 个独立制度（ES 段互不重叠），对折
    重采样只能反映"换一批制度会看到什么"。由于重采样只会抽到已有折，
    ``ci_min_resampled`` 的下界恒等于观测到的折最小值——因此它**不能**
    用来判定"最差折是否仍高于阈值"（该分布永远不会低于观测值）。真正
    能判断最差折自身估计误差的是**逐折区间**
    （``moving_block_metric_ci``，需要 ES 逐行预测）。

    本函数因此输出三类可用的量：
    - 折间分布（min / median / max / std）与 ``fold_min_pass_share``：
      有多少折本身越过阈值（分布无关、无需自举）；
    - 均值 lift 的自举区间与 ``prob_mean_pass``：下一个制度期望表现；
    - ``ci_min_resampled``：仅作"最差折在重采样集里会显得多好"的描述。

    Args:
        fold_lifts: 各折 lift（点估计）
        lift_min_threshold: 门禁阈值
        n_resamples / seed / ci: 重采样参数

    Returns:
        dict：含点估计、折间统计、均值区间、``prob_mean_pass`` 等字段
    """
    values = np.asarray([v for v in fold_lifts if v is not None and np.isfinite(v)], dtype=float)
    if values.size == 0:
        raise ValueError("没有有效折 lift，无法判定门禁")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    mins = values[idx].min(axis=1)
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    out = {
        "n_folds": int(values.size),
        "point_min": float(values.min()),
        "point_median": float(np.median(values)),
        "point_mean": float(values.mean()),
        "point_max": float(values.max()),
        "fold_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "point_gate_pass": bool(values.min() >= lift_min_threshold),
        "fold_min_pass_share": float((values >= lift_min_threshold).mean()),
        "ci_mean": [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))],
        "prob_mean_pass": float((means >= lift_min_threshold).mean()),
        "ci_min_resampled": [float(np.quantile(mins, alpha)), float(np.quantile(mins, 1 - alpha))],
        "min_ci_degenerate": True,
        "threshold": float(lift_min_threshold),
        "n_resamples": int(n_resamples),
        "ci_level": ci,
        "seed": seed,
    }
    logger.info(
        f"折级门禁重判: 点估计 min={out['point_min']:.3f} / median="
        f"{out['point_median']:.3f}（原口径 "
        f"{'通过' if out['point_gate_pass'] else '不通过'}），"
        f"达标折 {out['fold_min_pass_share'] * 100:.0f}%，"
        f"均值 lift {int(ci * 100)}% 区间 {out['ci_mean']}，折间 std={out['fold_std']:.3f}"
    )
    return out

"""terminal_loss 覆盖审计（方案 2.4 第 6 条 / 8.1 报告门禁）

回答方案 8.1 的三项报告门禁要求，全部可跨块累计（分块标签构建下块内
无法得到全局比率）：

1. **停牌/无端点样本占比**：各 ``label_status`` 按 h 的占比（含
   ``execution_blocked`` 计数），由 ``LabelCoverageAccumulator`` 提供。
2. **可观测代理的条件事件率差异**：对每个代理因子输出两件事——
   （a）缺失组与 valid 组的代理画像矩（均值/标准差）；
   （b）valid 行在「当日截面分位桶 × 事件率」上的条件事件率。
   桶按**当日截面**分位定义（不是跨日绝对阈值），因此跨块可累计、
   且语义与 pct_* 同源：它回答「代理看起来像缺失组的那些股票，历史
   事件率有多高」，即静默丢弃缺失样本可能带来的偏差方向与量级。
3. **endpoint_delayed 敏感性**：对 ``endpoint_missing`` 行改用 E 之后
   首个有报价开盘价重算标签，与 valid 组事件率对比，量化"直接丢弃
   端点缺失样本"是否系统性低估风险（方案 2.4 第 6 条）。

预登记规则：缺失占比 ≥ ``DELAYED_SENSITIVITY_SHARE_THRESHOLD``（1%）时
**必须**补做 delayed endpoint 敏感性；无论是否触发，占比与代理差异都
必须出现在报告里（禁止静默丢弃）。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .labels import (
    LABEL_STATUS_ENDPOINT_MISSING,
    LABEL_STATUS_IMMATURE,
    LABEL_STATUS_SIGMA_UNAVAILABLE,
    LABEL_STATUS_VALID,
    TerminalLossLabelConfig,
)

#: 审计用代理列（均可由 T 日及以前信息计算，与标签状态无关）
AUDIT_PROXY_COLUMNS: List[str] = [
    "sigma_daily_20",
    "amihud_illiq_20",
    "ret_20",
    "cvar_95_20",
]

#: 预登记的「必须补做 delayed 敏感性」占比阈值（占总标签行）
DELAYED_SENSITIVITY_SHARE_THRESHOLD = 0.01

#: delayed endpoint 搜索窗口（E 之后最多再找多少个交易日）
DELAYED_SEARCH_WINDOW_DAYS = 10

#: 条件事件率的当日截面分位桶数
CONDITIONAL_BUCKETS = 3


@dataclass
class ProxyProfileAccumulator:
    """代理画像与条件事件率的跨块累计器。

    Attributes:
        proxies: 参与审计的代理列名
        n_buckets: 当日截面分位桶数
        moments: {(label_status, proxy): [n, sum, sumsq]}，缺失值不计入
        conditional: {(proxy, bucket): [n_valid, n_event]}，仅 valid 行
    """

    proxies: Sequence[str] = tuple(AUDIT_PROXY_COLUMNS)
    n_buckets: int = CONDITIONAL_BUCKETS
    moments: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    conditional: Dict[Tuple[str, int], List[int]] = field(default_factory=dict)

    def add(
        self, labels_df: pd.DataFrame, proxies_df: Optional[pd.DataFrame] = None
    ) -> "ProxyProfileAccumulator":
        """累计一块标签表的代理统计。

        Args:
            labels_df: 标签瘦表（含 ts_code / trade_date / label_status /
                loss_label；sigma_daily_20 可直接取自同名列）
            proxies_df: 可选的长表（ts_code, trade_date + 代理列），提供
                除 sigma 之外的代理值；为 None 时只用 labels 自带列
        """
        if labels_df is None or labels_df.empty:
            return self
        frame = labels_df
        if proxies_df is not None and not proxies_df.empty:
            frame = frame.merge(proxies_df, on=["ts_code", "trade_date"], how="left")
        for proxy in self.proxies:
            if proxy not in frame.columns:
                continue
            for status, series in frame.groupby("label_status")[proxy]:
                values = series.dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                key = (str(status), proxy)
                slot = self.moments.setdefault(key, [0.0, 0.0, 0.0])
                slot[0] += float(values.size)
                slot[1] += float(values.sum())
                slot[2] += float(np.square(values).sum())

            valid = frame[frame["label_status"] == LABEL_STATUS_VALID]
            if valid.empty:
                continue
            for bucket, stats in _bucket_event_stats(valid, proxy, self.n_buckets).items():
                slot = self.conditional.setdefault((proxy, bucket), [0, 0])
                slot[0] += int(stats[0])
                slot[1] += int(stats[1])
        return self

    def profile_table(self) -> pd.DataFrame:
        """缺失组 vs valid 组的代理画像（矩表）。"""
        rows = []
        for (status, proxy), (n, total, sq) in sorted(self.moments.items()):
            if n <= 0:
                continue
            mean = total / n
            var = max(sq / n - mean * mean, 0.0)
            rows.append(
                {
                    "label_status": status,
                    "proxy": proxy,
                    "n": int(n),
                    "mean": mean,
                    "std": float(np.sqrt(var)),
                }
            )
        return pd.DataFrame(rows, columns=["label_status", "proxy", "n", "mean", "std"])

    def conditional_table(self) -> pd.DataFrame:
        """valid 行的「当日截面分位桶 × 条件事件率」表。

        bucket 为 0..n_buckets-1（当日截面分位由低到高），用于回答
        "代理与缺失组同档的股票，历史事件率是多少"。
        """
        rows = []
        for (proxy, bucket), (n, events) in sorted(self.conditional.items()):
            if n <= 0:
                continue
            rows.append(
                {
                    "proxy": proxy,
                    "bucket": int(bucket),
                    "n_valid": int(n),
                    "n_event": int(events),
                    "event_rate": events / n,
                }
            )
        return pd.DataFrame(
            rows,
            columns=["proxy", "bucket", "n_valid", "n_event", "event_rate"],
        )

    def implied_missing_event_rate(
        self, status: str = LABEL_STATUS_ENDPOINT_MISSING
    ) -> pd.DataFrame:
        """按缺失組的代理画像，推算其"隐含事件率"（偏差下界口径）。

        做法：把代理按分位桶的条件事件率（valid 行估计）与缺失组的
        代理桶分布（缺失组在各桶的占比）加权求和。缺失组自身没有标签，
        因此这是**可观测代理条件下的**估计，不是其真实事件率；它只用于
        判断"直接丢弃这批样本的偏差方向与量级"，不进入任何分数。

        Args:
            status: 目标缺失状态（默认端点缺失）

        Returns:
            每个代理一行：missing_share、valid_event_rate（valid 全体加权率）、
            implied_event_rate（按缺失画像加权）、delta。
        """
        profiles = self.profile_table()
        conditions = self.conditional_table()
        if profiles.empty or conditions.empty:
            return pd.DataFrame(
                columns=[
                    "proxy",
                    "missing_n",
                    "valid_event_rate",
                    "implied_event_rate",
                    "delta",
                ]
            )
        missing = profiles[profiles["label_status"] == status]
        valid_all = profiles[profiles["label_status"] == LABEL_STATUS_VALID]
        rows = []
        for proxy in missing["proxy"].unique():
            cond = conditions[conditions["proxy"] == proxy]
            if cond.empty:
                continue
            base_rate = float(np.average(cond["event_rate"], weights=cond["n_valid"]))
            # 缺失组在该代理的当日截面桶分布：用缺失组均值和全样本桶边界近似
            # —— 这里不做二次分桶（缺失组样本没有可复现的桶边界），改为
            # 用 valid 组桶中心与缺失组均值的距离加权，保持口径可复核。
            centers = _bucket_centers(cond)
            miss_mean = float(missing.loc[missing["proxy"] == proxy, "mean"].iloc[0])
            weights = _soft_bucket_weights(centers, miss_mean)
            implied = float(np.sum(weights * cond["event_rate"].to_numpy()))
            rows.append(
                {
                    "proxy": proxy,
                    "missing_n": int(missing.loc[missing["proxy"] == proxy, "n"].iloc[0]),
                    "valid_event_rate": base_rate,
                    "implied_event_rate": implied,
                    "delta": implied - base_rate,
                }
            )
        _ = valid_all  # valid 画像仅用于报告展示，不参与加权
        return pd.DataFrame(
            rows,
            columns=[
                "proxy",
                "missing_n",
                "valid_event_rate",
                "implied_event_rate",
                "delta",
            ],
        )


def _bucket_event_stats(
    valid_df: pd.DataFrame, proxy: str, n_buckets: int
) -> Dict[int, Tuple[int, int]]:
    """按当日截面分位分桶，统计每桶的 (行数, 事件数)。"""
    frame = valid_df[valid_df[proxy].notna()]
    if frame.empty:
        return {}
    pct = frame.groupby("trade_date")[proxy].rank(pct=True, method="average")
    bucket = np.minimum((pct.to_numpy() * n_buckets).astype(int), n_buckets - 1)
    grouped = frame.assign(_bucket=bucket).groupby("_bucket")["loss_label"]
    return {
        int(b): (int(size), int(events))
        for b, (size, events) in grouped.agg(["size", "sum"]).iterrows()
    }


def _bucket_centers(cond: pd.DataFrame) -> np.ndarray:
    """桶中心（把 0..n-1 桶映射为等距分位中点）。"""
    n = max(int(cond["bucket"].max()) + 1, 1)
    return (np.arange(n) + 0.5) / n


def _soft_bucket_weights(centers: np.ndarray, value: float) -> np.ndarray:
    """按到桶中心的距离做三角核权重（缺失组均值 → 桶权重）。

    缺失组没有可复现的当日截面桶边界（其代理值可与同日 valid 行比较，
    但桶边界须固定），因此用核权重代替硬分桶：越接近某桶中心的桶权重
    越高，权重和为 1，保证隐含事件率始终落在桶事件率区间内。
    """
    if centers.size == 0:
        return np.zeros(0)
    span = 1.0 / centers.size
    dist = np.abs(centers - value)
    weights = np.clip(1.0 - dist / span, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return np.full(centers.size, 1.0 / centers.size)
    return weights / total


# ── endpoint_delayed 敏感性（方案 2.4 第 6 条）──────────────────────


def delayed_endpoint_sensitivity(
    labels_df: pd.DataFrame,
    open_adj_panel: pd.DataFrame,
    label_config: Optional[TerminalLossLabelConfig] = None,
    max_delay_days: int = DELAYED_SEARCH_WINDOW_DAYS,
) -> Dict[str, Any]:
    """对端点缺失样本改用 E 之后首个有报价开盘价重算标签。

    端点缺失（T+1 或 E 停牌无行）与高风险强相关，直接剔除会系统性低估
    训练分布的风险水平。本函数量化该偏差：对这些行按 delayed endpoint
    重算 ``R`` 与 ``Y``，给出可评估行数、事件率与相对 valid 组事件率的
    差值，并单独统计无法评估（T+1 本身缺报价）与超出搜索窗的行数。

    Args:
        labels_df: 标签瘦表（须含 ts_code / trade_date / h / label_status /
            sigma_at_t / planned_exit_date）
        open_adj_panel: 复权开盘价面板（index=交易日历升序，columns=ts_code）
        label_config: 标签配置（k 与 h_max）
        max_delay_days: E 之后最多再搜索的交易日数

    Returns:
        dict：by_h 明细 + total 汇总（含 event_rate 与 unavailable 计数）
    """
    cfg = label_config or TerminalLossLabelConfig()
    empty = {
        "by_h": [],
        "total": {
            "n_endpoint_missing": 0,
            "n_evaluated": 0,
            "n_t1_missing": 0,
            "n_delay_unavailable": 0,
            "n_event": 0,
            "event_rate": None,
            "max_delay_days": int(max_delay_days),
        },
    }
    if labels_df is None or labels_df.empty:
        return empty
    if not {"sigma_at_t", "h", "label_status"}.issubset(labels_df.columns):
        raise ValueError(
            "delayed 敏感性需要 ts_code/trade_date/h/label_status/sigma_at_t 列，"
            f"实际列: {sorted(labels_df.columns)}"
        )

    calendar = np.asarray(open_adj_panel.index)
    pos_of = {d: i for i, d in enumerate(calendar)}
    col_of = {code: i for i, code in enumerate(open_adj_panel.columns)}
    open_np = open_adj_panel.to_numpy(dtype=float)

    target = labels_df[labels_df["label_status"] == LABEL_STATUS_ENDPOINT_MISSING]
    by_h: Dict[int, Dict[str, int]] = {}
    totals = dict(empty["total"])
    totals["n_endpoint_missing"] = int(len(target))

    for row in target.itertuples(index=False):
        h = int(getattr(row, "h"))
        sigma = getattr(row, "sigma_at_t")
        code = getattr(row, "ts_code")
        date = getattr(row, "trade_date")
        if code not in col_of or date not in pos_of or not np.isfinite(sigma):
            continue
        col = col_of[code]
        t_pos = pos_of[date]
        t1_pos = t_pos + 1
        e_pos = t1_pos + h
        slot = by_h.setdefault(
            h,
            {
                "n_endpoint_missing": 0,
                "n_evaluated": 0,
                "n_t1_missing": 0,
                "n_delay_unavailable": 0,
                "n_event": 0,
            },
        )
        slot["n_endpoint_missing"] += 1
        if t1_pos >= len(calendar) or not np.isfinite(open_np[t1_pos, col]):
            slot["n_t1_missing"] += 1
            totals["n_t1_missing"] += 1
            continue
        base = open_np[t1_pos, col]
        delayed_pos = -1
        for p in range(e_pos, min(e_pos + max_delay_days + 1, len(calendar))):
            if np.isfinite(open_np[p, col]):
                delayed_pos = p
                break
        if delayed_pos < 0:
            slot["n_delay_unavailable"] += 1
            totals["n_delay_unavailable"] += 1
            continue
        ret = open_np[delayed_pos, col] / base - 1.0
        threshold = -cfg.loss_sigma_multiple * float(sigma) * np.sqrt(h)
        is_event = ret < threshold
        slot["n_evaluated"] += 1
        slot["n_event"] += int(is_event)
        totals["n_evaluated"] += 1
        totals["n_event"] += int(is_event)

    def _rate(slot: Dict[str, int]) -> Optional[float]:
        return slot["n_event"] / slot["n_evaluated"] if slot["n_evaluated"] else None

    rows = []
    for h in sorted(by_h):
        slot = dict(by_h[h])
        slot["h"] = h
        slot["event_rate"] = _rate(slot)
        rows.append(slot)
    totals["event_rate"] = _rate(totals)
    logger.info(
        f"delayed endpoint 敏感性：端点缺失 {totals['n_endpoint_missing']} 行，"
        f"可评估 {totals['n_evaluated']} 行（事件率 {totals['event_rate']}），"
        f"T+1 缺失 {totals['n_t1_missing']}，超出搜索窗 {totals['n_delay_unavailable']}"
    )
    return {"by_h": rows, "total": totals}


def coverage_audit_required(
    status_share: pd.DataFrame, threshold: float = DELAYED_SENSITIVITY_SHARE_THRESHOLD
) -> Dict[str, Any]:
    """按预登记阈值判定是否必须补做 delayed 敏感性（方案 8.1）。

    缺失占比口径 = (endpoint_missing + sigma_unavailable + immature) 行数
    占总标签行数比例（三者都是"不可标记安全"的不可用样本）。
    """
    if status_share is None or status_share.empty:
        return {
            "share": None,
            "threshold": threshold,
            "required": False,
            "by_status": {},
        }
    total = float(status_share["count"].sum())
    unavailable = {
        LABEL_STATUS_ENDPOINT_MISSING,
        LABEL_STATUS_SIGMA_UNAVAILABLE,
        LABEL_STATUS_IMMATURE,
    }
    rows = status_share[status_share["label_status"].isin(unavailable)]
    share = float(rows["count"].sum()) / total if total else 0.0
    by_status = (
        rows.groupby("label_status")["count"].sum() / total if total else pd.Series(dtype=float)
    )
    return {
        "share": share,
        "threshold": threshold,
        "required": bool(share >= threshold),
        "by_status": {k: float(v) for k, v in by_status.items()},
    }

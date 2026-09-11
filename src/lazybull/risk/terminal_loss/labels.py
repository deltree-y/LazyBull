"""剩余持有期期末异常亏损标签构建

实现 docs/plans/terminal_loss_risk_model_plan.md 第 2 节契约：

- ``R_{i,T,h} = open_adj(E) / open_adj(T+1) - 1``（T+1 与 E 均为全市场交易日历日）
- ``Y_{i,T,h} = 1[R < -k * sigma_daily_20(T) * sqrt(h)]``
- ``h = idx(E) - idx(T+1)``，E 为计划退出日（到期日契约：实际买入日 B 后
  第 rebalance_freq 个交易日开盘；训练侧按 h=h_min..h_max 全网格覆盖）

设计约束：

- P0/P1 只用于事后构造标签，本模块不产生任何预测输入；
- 标签保持固定参考端点，回测执行层的可成交性另行结算（方案 2.4.7）；
- 端点缺失（停牌缺行）、未成熟（E 超出日历末端）、sigma 无效分别落
  label_status，不标记安全、不前填价格；
- E 日跌停等"有报价但不可卖"样本保留标签并单独标记 execution_blocked
  （方案 2.4.6），不因事后跌停剔除高风险样本；
- 输入面板 index 必须是完整交易日历（调用方 reindex），本模块按位置
  推导 T+1 与 E，不自行补日历；
- 仅对 T 日有报价的股票生成行（T 日停牌无特征，训练样本不存在），
  T+1/E/sigma 的缺失由 label_status 表达。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 标签状态常量（互斥，优先级 immature > endpoint_missing > sigma）──────

LABEL_STATUS_VALID = "valid"
LABEL_STATUS_IMMATURE = "immature"  # E 超出日历数据末端
LABEL_STATUS_ENDPOINT_MISSING = "endpoint_missing"  # T+1 或 E 停牌缺行
LABEL_STATUS_SIGMA_UNAVAILABLE = "sigma_unavailable"  # sigma 缺失或零波动

_LABEL_TABLE_COLUMNS = [
    "ts_code",
    "trade_date",
    "h",
    "reference_date",
    "planned_exit_date",
    "label_end_date",
    "terminal_return",
    "loss_label",
    "label_status",
    "execution_blocked",
    "sigma_at_t",
]


@dataclass(frozen=True)
class TerminalLossLabelConfig:
    """期末异常亏损标签配置（方案 3.5，每项单一来源，无多层回退）。

    Attributes:
        task_id: 任务标识，与旧三分类风控任务隔离
        h_min / h_max: 支持的开盘到开盘间隔范围
        sigma_window: sigma_daily_20 的历史日历交易日窗口
        loss_sigma_multiple: 异常亏损界限的波动倍数 k（训练前固定，
            改变 k 即改变任务，须产生新的 task 配置与模型）
    """

    task_id: str = "terminal_vol_scaled_loss"
    h_min: int = 1
    h_max: int = 20
    sigma_window: int = 20
    loss_sigma_multiple: float = 1.0


def build_terminal_loss_labels(
    open_adj_panel: pd.DataFrame,
    sigma_panel: pd.DataFrame,
    config: Optional[TerminalLossLabelConfig] = None,
    limit_down_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """构建多期限瘦标签表（方案第 6 节：ts_code × trade_date × h）。

    Args:
        open_adj_panel: 复权开盘价面板，index=完整交易日历（升序），
            columns=ts_code；停牌缺行必须已为 NaN（不可前填）
        sigma_panel: sigma_daily_20 面板，index/columns 与 open_adj_panel
            完全对齐（来自 compute_sigma_daily_panel，严格日历口径）
        config: 标签配置
        limit_down_panel: 可选，跌停面板（index/columns 对齐，1=跌停）；
            提供时对 valid 行按 E 日跌停标记 execution_blocked

    Returns:
        瘦标签表（见 _LABEL_TABLE_COLUMNS）；非 valid 行的
        terminal_return/loss_label 为 NaN
    """
    cfg = config or TerminalLossLabelConfig()
    if not (1 <= cfg.h_min <= cfg.h_max):
        raise ValueError(f"非法期限范围: h_min={cfg.h_min}, h_max={cfg.h_max}")
    if open_adj_panel.empty:
        return pd.DataFrame(columns=_LABEL_TABLE_COLUMNS)
    if not sigma_panel.index.equals(open_adj_panel.index) or not (
        sigma_panel.columns.equals(open_adj_panel.columns)
    ):
        raise ValueError(
            "sigma_panel 与 open_adj_panel 的 index/columns 必须完全对齐"
            "（同一完整交易日历与股票域），拒绝静默 reindex"
        )

    calendar = np.asarray(open_adj_panel.index)
    ts_codes = np.asarray(open_adj_panel.columns)
    n_dates, n_codes = open_adj_panel.shape

    open_np = open_adj_panel.to_numpy(dtype=float)
    sigma_np = sigma_panel.to_numpy(dtype=float)
    quote = ~np.isnan(open_np)  # 当日有报价（clean/daily 停牌缺行已为 NaN）
    sigma_ok = ~np.isnan(sigma_np) & (sigma_np > 0)  # 零波动视为不可用

    if limit_down_panel is not None:
        limit_np = (
            limit_down_panel.reindex(index=open_adj_panel.index, columns=open_adj_panel.columns)
            .fillna(0)
            .to_numpy()
        )
    else:
        limit_np = np.zeros_like(open_np)

    frames: List[pd.DataFrame] = []
    for h in range(cfg.h_min, cfg.h_max + 1):
        # 端点价格：T+1 与 E = T+1+h 的复权开盘（按日历位置取数）
        t1_open = np.full_like(open_np, np.nan)
        e_open = np.full_like(open_np, np.nan)
        t1_open[:-1] = open_np[1:]
        e_open[: -(1 + h)] = open_np[1 + h :]
        e_limit_down = np.zeros(quote.shape, dtype=bool)
        e_limit_down[: -(1 + h)] = limit_np[1 + h :].astype(bool)

        t1_ok = np.zeros(quote.shape, dtype=bool)
        e_ok = np.zeros(quote.shape, dtype=bool)
        t1_ok[:-1] = quote[1:]
        e_ok[: -(1 + h)] = quote[1 + h :]

        terminal_return = e_open / t1_open - 1.0

        # 状态判定：先生成行（T 日有报价），再按优先级覆盖
        pos = np.arange(n_dates)
        immature_row = (pos + 1 + h) > (n_dates - 1)

        status = np.full(open_np.shape, LABEL_STATUS_VALID, dtype=object)
        endpoint_bad = ~(t1_ok & e_ok)
        status[endpoint_bad] = LABEL_STATUS_ENDPOINT_MISSING
        status[immature_row] = LABEL_STATUS_IMMATURE  # 行级最高优先级
        sigma_bad = (status == LABEL_STATUS_VALID) & ~sigma_ok
        status[sigma_bad] = LABEL_STATUS_SIGMA_UNAVAILABLE

        valid_mask = status == LABEL_STATUS_VALID
        loss_label = np.full(open_np.shape, np.nan)
        loss_label[valid_mask] = terminal_return[valid_mask] < (
            -cfg.loss_sigma_multiple * sigma_np[valid_mask] * np.sqrt(h)
        )
        exec_blocked = e_limit_down & valid_mask

        # 展开为长表：仅保留 T 日有报价的格子
        sel = quote
        grid_pos = np.broadcast_to(pos[:, None], open_np.shape)[sel]
        ref_pos = grid_pos + 1
        exit_pos = grid_pos + 1 + h
        in_range = lambda p: p <= n_dates - 1  # noqa: E731

        frame = pd.DataFrame(
            {
                "ts_code": np.broadcast_to(ts_codes, open_np.shape)[sel],
                "trade_date": np.broadcast_to(calendar[:, None], open_np.shape)[sel],
                "h": h,
                "reference_date": np.where(
                    in_range(ref_pos),
                    calendar[np.clip(ref_pos, 0, n_dates - 1)],
                    None,
                ),
                "planned_exit_date": np.where(
                    in_range(exit_pos),
                    calendar[np.clip(exit_pos, 0, n_dates - 1)],
                    None,
                ),
            }
        )
        frame["label_end_date"] = frame["planned_exit_date"]
        frame["terminal_return"] = np.where(valid_mask, terminal_return, np.nan)[sel]
        frame["loss_label"] = loss_label[sel]
        frame["label_status"] = status[sel]
        frame["execution_blocked"] = exec_blocked[sel]
        frame["sigma_at_t"] = sigma_np[sel]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_LABEL_TABLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[_LABEL_TABLE_COLUMNS]


def summarize_label_coverage(labels_df: pd.DataFrame) -> pd.DataFrame:
    """按 h 与 label_status 汇总覆盖分布（方案 8.1 报告门禁输入）。

    Returns:
        DataFrame：h × label_status 行数透视，valid 组附事件率与
        execution_blocked 计数
    """
    if labels_df is None or labels_df.empty:
        return pd.DataFrame()
    return _coverage_pivot(LabelCoverageAccumulator().add(labels_df))


#: 覆盖长表列（分块累计产出的报告列，含 valid 组事件率与执行受阻计数）
COVERAGE_COLUMNS: List[str] = [
    "h",
    "label_status",
    "count",
    "event_rate",
    "execution_blocked_count",
]


@dataclass
class LabelCoverageAccumulator:
    """分块标签构建的覆盖累计器（跨块求和后再算比率）。

    分块构建无法在块内得到最终覆盖率：均值不可累加，必须先累计
    「valid 行数 / 事件数 / execution_blocked 数」，全部块处理完再一次性
    求比率。原来的单块 ``groupby(...).mean()`` 只能在单帧场景使用，误用于
    分块会得到「块内均值的均值」这一错口径。

    Attributes:
        counts: {(h, label_status): 行数}
        valid_rows / valid_events / execution_blocked: 按 h 累计的 valid 行数、
        事件数与 E 日执行受阻（跌停）数
    """

    counts: Dict[Tuple[int, str], int] = field(default_factory=dict)
    valid_rows: Dict[int, int] = field(default_factory=dict)
    valid_events: Dict[int, int] = field(default_factory=dict)
    execution_blocked: Dict[int, int] = field(default_factory=dict)

    def add(self, labels_df: pd.DataFrame) -> "LabelCoverageAccumulator":
        """累计一块标签表的覆盖计数（原地更新并返回自身，便于链式调用）。"""
        if labels_df is None or labels_df.empty:
            return self
        for (h, status), cnt in labels_df.groupby(["h", "label_status"]).size().items():
            key = (int(h), str(status))
            self.counts[key] = self.counts.get(key, 0) + int(cnt)
        valid = labels_df[labels_df["label_status"] == LABEL_STATUS_VALID]
        if not valid.empty:
            grouped = valid.groupby("h")
            for h, cnt in grouped.size().items():
                self.valid_rows[int(h)] = self.valid_rows.get(int(h), 0) + int(cnt)
            for h, ev in grouped["loss_label"].sum().items():
                self.valid_events[int(h)] = self.valid_events.get(int(h), 0) + int(ev)
            for h, blocked in grouped["execution_blocked"].sum().items():
                self.execution_blocked[int(h)] = self.execution_blocked.get(int(h), 0) + int(
                    blocked
                )
        return self

    @property
    def valid_count(self) -> int:
        """全期限 valid 行数合计（全网格事件率分母）。"""
        return int(sum(self.valid_rows.values()))

    @property
    def valid_event_sum(self) -> int:
        """全期限 valid 事件数合计（全网格事件率分子）。"""
        return int(sum(self.valid_events.values()))

    def to_frame(self) -> pd.DataFrame:
        """覆盖长表（含 valid 组事件率与执行受阻计数）。

        ``event_rate`` / ``execution_blocked_count`` 仅对 valid 行给出，
        其余状态为 NaN（非 valid 行没有标签，不存在事件率）。
        """
        rows = []
        for (h, status), cnt in sorted(self.counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            is_valid = status == LABEL_STATUS_VALID
            n_valid = self.valid_rows.get(h, 0)
            rows.append(
                {
                    "h": h,
                    "label_status": status,
                    "count": cnt,
                    "event_rate": (
                        self.valid_events.get(h, 0) / n_valid if is_valid and n_valid else np.nan
                    ),
                    "execution_blocked_count": (
                        self.execution_blocked.get(h, 0) if is_valid else np.nan
                    ),
                }
            )
        if not rows:
            return pd.DataFrame(columns=COVERAGE_COLUMNS)
        return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)

    def status_share(self) -> pd.DataFrame:
        """按 h 的状态占比表（方案 8.1 报告门禁：停牌/不可用样本占比）。"""
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["h", "label_status", "count", "share"])
        frame = frame.copy()
        frame["share"] = frame["count"] / frame.groupby("h")["count"].transform("sum")
        return frame[["h", "label_status", "count", "share"]]


def _coverage_pivot(acc: LabelCoverageAccumulator) -> pd.DataFrame:
    """把累计器结果转成 h × status 透视（单帧便利入口使用）。"""
    frame = acc.to_frame()
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot(index="h", columns="label_status", values="count").fillna(0).astype(int)
    if LABEL_STATUS_VALID in pivot.columns:
        valid_rows = frame.loc[frame["label_status"] == LABEL_STATUS_VALID].set_index("h")
        pivot["event_rate"] = valid_rows["event_rate"]
        pivot["execution_blocked_count"] = valid_rows["execution_blocked_count"]
    return pivot

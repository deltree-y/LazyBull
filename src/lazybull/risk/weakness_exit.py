"""持仓表现弱势退出模块

纯价格表现评估，零模型依赖。每日用累计收益排名、连续下跌天数、
回撤深度、回升乏力 4 个维度评估持仓，识别"持续垫底+下探+无回升迹象"
的弱势股，提前换出+补位。

设计要点:
- 全部维度仅使用后复权收盘价历史序列计算，不依赖 ML 模型输出
- 组合内比较：判断"垫底"用持仓内部累计收益排名百分位
- 三层门控：评分阈值 + 连续弱势天数 + 最低持有天数
- 可选行业弱势过滤（读取已有 ind_momentum_rank，不跑模型）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class WeaknessExitConfig:
    """表现弱势退出配置"""

    enabled: bool = False
    threshold: float = 0.6  # 弱势评分触发阈值 [0, 1]，越高越易触发
    consecutive_days: int = 3  # 需连续弱势天数
    min_holding_days: int = 5  # 最低持有天数（交易日）
    weights: Tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)
    industry_filter: bool = False  # 是否叠加弱势行业过滤
    industry_bottom_pct: float = 0.3  # 行业底部百分位阈值

    def __post_init__(self):
        """验证配置"""
        if not (0 <= self.threshold <= 1):
            raise ValueError(
                f"trigger_threshold 必须在 [0, 1] 范围内，当前值: {self.threshold}"
            )
        if self.consecutive_days < 1:
            raise ValueError(
                f"consecutive_days 必须 >= 1，当前值: {self.consecutive_days}"
            )
        if self.min_holding_days < 1:
            raise ValueError(
                f"min_holding_days 必须 >= 1，当前值: {self.min_holding_days}"
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            self.weights = tuple(w / total for w in self.weights)
            logger.info(f"弱势退出权重自动归一化: {self.weights}")


class WeaknessExitMonitor:
    """表现弱势退出监控器

    维护每只持仓的连续弱势状态，每日评估 4 维表现评分。

    用法:
        monitor = WeaknessExitMonitor(config)
        score, detail = monitor.evaluate(
            stock="000001.SZ",
            price_series=pnl_price_series,          # 买入至今的后复权收盘价
            all_positions_profit={"000001.SZ": -0.05, "000002.SZ": 0.03},
            holding_days=10,
        )
        consec = monitor.update("000001.SZ", "20260315", score)
        if score >= config.threshold and consec >= config.consecutive_days:
            # 触发退出
    """

    def __init__(self, config: WeaknessExitConfig):
        """初始化弱势退出监控器

        Args:
            config: 弱势退出配置
        """
        self.config = config
        # {stock: {"consecutive_weak_days": int, "last_weak_date": str, "scores": [float]}}
        self._state: Dict[str, dict] = {}

        logger.info(
            f"弱势退出监控器初始化: enabled={config.enabled}, "
            f"threshold={config.threshold}, "
            f"consecutive_days={config.consecutive_days}, "
            f"min_holding_days={config.min_holding_days}, "
            f"weights={config.weights}, "
            f"industry_filter={config.industry_filter}"
        )

    # ── 对外主接口 ────────────────────────────────────────────────

    def evaluate(
        self,
        stock: str,
        price_series: pd.Series,
        all_positions_profit: Dict[str, float],
        holding_days: int,
        industry_rank: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """计算单只持仓的弱势评分

        Args:
            stock: 股票代码
            price_series: 买入以来的后复权收盘价序列 (index=date str, values=price)
            all_positions_profit: 所有持仓的当前累计收益率 {stock: profit_rate}
            holding_days: 当前持有交易日数
            industry_rank: 可选行业动量排名 [0,1]，0=最弱

        Returns:
            (weakness_score, dimension_breakdown)
            - weakness_score: [0, 1]，越高越弱
            - breakdown: 各维度子评分字典
        """
        if holding_days < self.config.min_holding_days:
            pass  # 仍继续评分，仅在下游由调用方判断是否触发退出

        w_pnl, w_streak, w_dd, w_recov = self.config.weights

        # 维度1: 相对收益排名
        rank_pnl = self._calc_pnl_rank(stock, all_positions_profit)
        dim_rank = 1.0 - rank_pnl  # 越垫底越高

        # 维度2: 连续下行
        streak_norm = self._calc_down_streak(price_series, holding_days)
        dim_streak = streak_norm

        # 维度3: 回撤深度
        dd_contribution = self._calc_drawdown(price_series)
        dim_dd = dd_contribution

        # 维度4: 回升乏力
        recov_ratio = self._calc_recovery_ratio(price_series)
        dim_recov = 1.0 - min(recov_ratio / 2.0, 1.0)  # 反弹越弱越高

        weakness = (
            w_pnl * dim_rank
            + w_streak * dim_streak
            + w_dd * dim_dd
            + w_recov * dim_recov
        )
        weakness = float(np.clip(weakness, 0.0, 1.0))

        # 可选行业过滤：弱势行业 + 0.1 加分（更容易触发）
        if (
            self.config.industry_filter
            and industry_rank is not None
            and not np.isnan(industry_rank)
        ):
            if industry_rank < self.config.industry_bottom_pct:
                weakness = min(1.0, weakness + 0.10)

        breakdown = {
            "pnl_rank": dim_rank,
            "down_streak": dim_streak,
            "drawdown": dim_dd,
            "recovery": dim_recov,
            "total": weakness,
        }
        return weakness, breakdown

    def update(self, stock: str, date: str, score: float) -> int:
        """更新连续弱势计数

        Args:
            stock: 股票代码
            date: 评估日期 YYYYMMDD
            score: 当日弱势评分

        Returns:
            当前连续弱势天数（含当日）
        """
        state = self._state.setdefault(
            stock,
            {"consecutive_weak_days": 0, "last_weak_date": "", "scores": []},
        )

        if score >= self.config.threshold:
            state["consecutive_weak_days"] += 1
            state["last_weak_date"] = date
            state["scores"].append(score)
            # 只保留最近 N+5 条历史
            if len(state["scores"]) > self.config.consecutive_days + 5:
                state["scores"] = state["scores"][-(self.config.consecutive_days + 5) :]
        else:
            state["consecutive_weak_days"] = 0
            state["scores"] = []

        return state["consecutive_weak_days"]

    def reset(self, stock: str) -> None:
        """股票卖出后清除状态"""
        self._state.pop(stock, None)

    def get_consecutive_days(self, stock: str) -> int:
        """获取某只持仓的当前连续弱势天数"""
        return self._state.get(stock, {}).get("consecutive_weak_days", 0)

    def get_state(self) -> dict:
        """导出监控器状态（用于持久化）。"""
        return {"_state": dict(self._state)}

    def restore_state(self, saved: dict) -> None:
        """从持久化数据恢复监控器状态。"""
        if not isinstance(saved, dict):
            return
        state_data = saved.get("_state")
        if isinstance(state_data, dict):
            self._state = dict(state_data)
            logger.info(
                f"弱势退出监控器状态已恢复: {len(self._state)} 只股票"
            )

    # ── 维度计算（内部方法）───────────────────────────────────────

    def _calc_pnl_rank(
        self, stock: str, all_positions_profit: Dict[str, float]
    ) -> float:
        """持仓按累计收益率排名的百分位

        Args:
            stock: 当前股票代码
            all_positions_profit: {stock: profit_rate}

        Returns:
            百分位 [0, 1]，1 = 最好（收益最高），0 = 最差
        """
        if len(all_positions_profit) <= 1:
            return 0.5

        # 按收益率升序排列
        sorted_stocks = sorted(all_positions_profit.items(), key=lambda x: x[1])
        rank_idx = next(
            (i for i, (s, _) in enumerate(sorted_stocks) if s == stock), -1
        )
        if rank_idx < 0:
            return 0.5  # 未找到

        return rank_idx / (len(sorted_stocks) - 1)

    @staticmethod
    def _calc_down_streak(price_series: pd.Series, holding_days: int) -> float:
        """连续下跌天数归一化

        Args:
            price_series: 买入以来的后复权收盘价序列
            holding_days: 当前持有天数

        Returns:
            归一化连续下跌天数 [0, 1]
        """
        if price_series is None or len(price_series) < 2:
            return 0.0

        # 计算日收益率
        returns = price_series.pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # 从最近往前数连续负收益天数
        streak = 0
        for ret in reversed(returns.values):
            if ret < 0:
                streak += 1
            else:
                break

        norm = min(float(streak) / max(holding_days, 5), 1.0)
        return norm

    @staticmethod
    def _calc_drawdown(price_series: pd.Series) -> float:
        """从持有期最高点回撤深度

        Args:
            price_series: 买入以来的后复权收盘价序列

        Returns:
            回撤贡献 [0, 1]，0=无回撤（当前即最高），1=完全归零
        """
        if price_series is None or len(price_series) < 2:
            return 0.0

        peak = float(price_series.max())
        current = float(price_series.iloc[-1])

        if peak <= 0 or current <= 0:
            return 0.0

        if current >= peak:
            return 0.0  # 当前即最高点

        drawdown = (current - peak) / peak  # 负值
        return abs(min(drawdown, 0.0))  # 取正值

    @staticmethod
    def _calc_recovery_ratio(price_series: pd.Series) -> float:
        """回升比率: 从最低点以来的反弹幅度 / 从最高点以来的最大回撤

        衡量"跌下去之后有没有涨回来的能力"。

        Args:
            price_series: 买入以来的后复权收盘价序列

        Returns:
            回升比率 [0, ∞)，0=毫无反弹，>=1=完全收复
        """
        if price_series is None or len(price_series) < 3:
            return 1.0  # 持有期太短，默认为正常

        values = price_series.values

        # 找到最高点位置
        peak_idx = int(np.argmax(values))
        peak_price = float(values[peak_idx])

        # 从最高点之后找最低点
        if peak_idx >= len(values) - 1:
            return 1.0  # 一直在涨，无回撤

        post_peak = values[peak_idx:]
        trough_idx_in_post = int(np.argmin(post_peak))
        trough_price = float(post_peak[trough_idx_in_post])
        trough_abs_idx = peak_idx + trough_idx_in_post

        if peak_price <= 0:
            return 1.0

        max_dd = abs((trough_price - peak_price) / peak_price)  # 正数
        if max_dd < 1e-8:
            return 1.0  # 几乎无回撤

        # 从最低点以来的反弹
        if trough_abs_idx >= len(values) - 1:
            return 0.0  # 最低点就是当日或之后，无反弹

        post_trough = values[trough_abs_idx:]
        max_after_trough = float(np.max(post_trough))
        recovery_amount = max_after_trough - trough_price

        if recovery_amount <= 0:
            return 0.0

        recovery_pct = recovery_amount / (trough_price + 1e-8)
        ratio = recovery_pct / (max_dd + 1e-8)

        return float(max(ratio, 0.0))


def create_weakness_exit_config_from_dict(config_dict: dict) -> WeaknessExitConfig:
    """从配置字典创建弱势退出配置对象

    Args:
        config_dict: 配置字典，通常来自 YAML 配置文件或命令行参数

    Returns:
        WeaknessExitConfig 对象
    """
    weights_str = config_dict.get("weakness_exit_weights", "30,25,25,20")
    weights_parts = [float(x.strip()) / 100.0 for x in weights_str.split(",")]
    if len(weights_parts) != 4:
        weights_parts = [0.30, 0.25, 0.25, 0.20]

    return WeaknessExitConfig(
        enabled=config_dict.get("weakness_exit_enabled", False),
        threshold=float(config_dict.get("weakness_exit_threshold", 0.6)),
        consecutive_days=int(config_dict.get("weakness_exit_consecutive_days", 3)),
        min_holding_days=int(config_dict.get("weakness_exit_min_holding_days", 5)),
        weights=tuple(weights_parts),
        industry_filter=config_dict.get("weakness_exit_industry_filter", False),
        industry_bottom_pct=float(
            config_dict.get("weakness_exit_industry_bottom_pct", 0.3)
        ),
    )

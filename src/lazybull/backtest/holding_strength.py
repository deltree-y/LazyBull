"""持仓强势度评分器

用于盈利延续持有决策的多维度评分:将单一"浮盈率阈值"判据升级为综合
ML 分数 / 动量加速 / 技术强度 / 资金筹码 / 回撤距离 5 个维度的强势度评分。

设计要点:
- 所有因子复用 src/lazybull/factors/ 已有计算结果(通过 features_by_date 读取),
  不重复计算
- 每个维度缺失时优雅降级(返回该维度中位分 0.5),不崩溃
- 权重可配置,默认均衡权重
- 每只股票独立评分,避免对全市场做截面统计(降低调用成本)
- 经验阈值 + sigmoid 映射到 [0, 1]

调用方(BacktestEngine._check_and_sell)通过 profit_extension_mode="strength"
激活本评分器。

价格和盈亏数据从 position_info 中读取(由 engine 构造),特征数据通过
engine._get_holding_features_row(date, stock) 这个 hook 获取,基类默认返回
None,engine_ml 覆写后从 features_by_date 读取对应行。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _sigmoid(x: float, slope: float = 1.0) -> float:
    """sigmoid 压缩到 [0, 1],slope 控制陡峭度"""
    z = slope * x
    # clamp 防止 np.exp 溢出产生 RuntimeWarning（numpy 溢出不抛 OverflowError）
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))


def _safe_get(row: Optional[pd.Series], col: str, default: float = np.nan) -> float:
    """从 features 行中安全提取列值,缺失返回 default"""
    if row is None or col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return float(val)


@dataclass
class HoldingStrengthWeights:
    """强势度评分的 5 个维度权重,总和应为 1.0"""

    ml_score: float = 0.30      # ML 模型综合打分
    momentum: float = 0.25      # 动量加速 + 行业 alpha
    technical: float = 0.15     # RSI / MACD / KDJ
    fund_flow: float = 0.15     # 融资净流入 + 筹码集中度变化
    drawdown: float = 0.15      # 当前浮盈率 + ATR 波动调整

    def normalize(self) -> "HoldingStrengthWeights":
        """归一化权重,确保总和为 1.0(容错:用户传入非归一化值时自动处理)"""
        total = self.ml_score + self.momentum + self.technical + self.fund_flow + self.drawdown
        if total <= 0:
            # 退化为均匀权重
            return HoldingStrengthWeights(0.2, 0.2, 0.2, 0.2, 0.2)
        return HoldingStrengthWeights(
            ml_score=self.ml_score / total,
            momentum=self.momentum / total,
            technical=self.technical / total,
            fund_flow=self.fund_flow / total,
            drawdown=self.drawdown / total,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "ml_score": self.ml_score,
            "momentum": self.momentum,
            "technical": self.technical,
            "fund_flow": self.fund_flow,
            "drawdown": self.drawdown,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, float]]) -> "HoldingStrengthWeights":
        if not d:
            return cls()
        return cls(
            ml_score=float(d.get("ml_score", 0.30)),
            momentum=float(d.get("momentum", 0.25)),
            technical=float(d.get("technical", 0.15)),
            fund_flow=float(d.get("fund_flow", 0.15)),
            drawdown=float(d.get("drawdown", 0.15)),
        )


@dataclass
class HoldingStrengthBreakdown:
    """单只持仓的强势度评分分解,用于日志和回撤归因"""

    total: float
    ml_score_dim: float
    momentum_dim: float
    technical_dim: float
    fund_flow_dim: float
    drawdown_dim: float
    profit_rate: float = 0.0        # 当前浮盈率(原 pnl 判据基线,保留用于日志对比)
    ml_score_raw: float = 0.0       # 当前 ML 模型预测分数(如可得)
    ml_score_ref_date: Optional[str] = None  # ML 分数来源日期 str

    def to_log_str(self) -> str:
        return (
            f"total={self.total:.3f} "
            f"[ml={self.ml_score_dim:.2f}(raw={self.ml_score_raw:.3f}) "
            f"mom={self.momentum_dim:.2f} "
            f"tech={self.technical_dim:.2f} "
            f"fund={self.fund_flow_dim:.2f} "
            f"dd={self.drawdown_dim:.2f}] "
            f"pnl={self.profit_rate:.2%}"
        )


class HoldingStrengthScorer:
    """持仓强势度评分器

    用法:
        scorer = HoldingStrengthScorer(engine, weights)
        breakdown = scorer.score(stock="000001.SZ", date=pd.Timestamp("20231001"),
                                  position_info=engine.positions["000001.SZ"],
                                  profit_rate=0.05)
        if breakdown.total >= 0.6:
            # 延续持有

    数据源:
        - ML 分数: engine._last_ranked_candidates(当天调仓日生成的截面分数)
        - 截面特征: engine._get_holding_features_row(date, stock)(子类覆写)
        - 浮盈率: 调用方传入(engine 已计算)
    """

    def __init__(
        self,
        engine,
        weights: Optional[HoldingStrengthWeights] = None,
    ):
        self.engine = engine
        self.weights = (weights or HoldingStrengthWeights()).normalize()

    # ── 对外主接口 ────────────────────────────────────────────────

    def score(
        self,
        stock: str,
        date: pd.Timestamp,
        position_info: Dict,
        profit_rate: float,
    ) -> HoldingStrengthBreakdown:
        """计算一只持仓股票的强势度评分

        Args:
            stock: 股票代码
            date: 当前日期(持有期满判断日)
            position_info: self.engine.positions[stock],用于读取 buy_atr_pct 等
            profit_rate: 当前浮盈率(engine 已用后复权价格计算好)

        Returns:
            HoldingStrengthBreakdown,total 字段为 [0, 1] 的强势度总分
        """
        # 1. 获取截面特征行(engine hook,子类覆写)
        features_row = self._get_features_row(date, stock)

        # 2. 获取当前 ML 分数(从最近一次 ranked_candidates)
        ml_raw, ml_ref_date = self._lookup_ml_score(stock)

        # 3. 5 个维度子评分
        dim_ml = self._score_ml_dim(ml_raw, features_row)
        dim_mom = self._score_momentum_dim(features_row)
        dim_tech = self._score_technical_dim(features_row)
        dim_fund = self._score_fund_flow_dim(features_row)
        dim_dd = self._score_drawdown_dim(profit_rate, position_info, features_row)

        # 4. 加权总分
        w = self.weights
        total = (
            w.ml_score * dim_ml
            + w.momentum * dim_mom
            + w.technical * dim_tech
            + w.fund_flow * dim_fund
            + w.drawdown * dim_dd
        )
        total = float(np.clip(total, 0.0, 1.0))

        return HoldingStrengthBreakdown(
            total=total,
            ml_score_dim=dim_ml,
            momentum_dim=dim_mom,
            technical_dim=dim_tech,
            fund_flow_dim=dim_fund,
            drawdown_dim=dim_dd,
            profit_rate=float(profit_rate),
            ml_score_raw=float(ml_raw) if not np.isnan(ml_raw) else 0.0,
            ml_score_ref_date=ml_ref_date,
        )

    # ── 数据源抽取 ────────────────────────────────────────────────

    def _get_features_row(self, date: pd.Timestamp, stock: str) -> Optional[pd.Series]:
        """通过 engine hook 获取股票在当前日期的截面特征行

        基类 BacktestEngine 返回 None(不使用 ML 特征),
        BacktestEngineML 覆写后从 features_by_date 读取。
        """
        getter = getattr(self.engine, "_get_holding_features_row", None)
        if getter is None:
            return None
        try:
            return getter(date, stock)
        except Exception as exc:
            logger.debug(f"获取持仓特征失败 stock={stock} date={date.date()}: {exc}")
            return None

    def _lookup_ml_score(self, stock: str) -> tuple:
        """从 engine 最近一次 ranked_candidates 查找 ML 分数

        Returns:
            (score, ref_date_str) — 找不到时返回 (nan, None)
        """
        last_cands = getattr(self.engine, "_last_ranked_candidates", None)
        if not last_cands:
            return np.nan, None
        # _last_ranked_candidates 是 [(ts_code, score), ...]
        for ts_code, score in last_cands:
            if ts_code == stock:
                last_date = getattr(self.engine, "_last_signal_date", None)
                ref_str = last_date.strftime("%Y%m%d") if last_date is not None else None
                return float(score), ref_str
        return np.nan, None

    # ── 各维度子评分 ──────────────────────────────────────────────

    def _score_ml_dim(
        self, ml_raw: float, features_row: Optional[pd.Series]
    ) -> float:
        """ML 分数维度:当前分数在最近 ranked_candidates 中的相对位置

        策略:
        - 若能取到当前 ML 分数,用 sigmoid 映射
        - 若取不到(非调仓日或缓存缺失),退化为用 acceleration + alpha_industry
          组合近似的"ML 综合性"分数
        """
        if not np.isnan(ml_raw):
            last_cands = getattr(self.engine, "_last_ranked_candidates", None)
            if last_cands and len(last_cands) > 1:
                # 相对位置:该股票分数相对于 candidates 分布的百分位
                all_scores = np.array([s for _, s in last_cands], dtype=float)
                # 转成百分位(越高越强)
                rank = float((all_scores < ml_raw).sum()) / len(all_scores)
                return float(np.clip(rank, 0.0, 1.0))
            # 没有分布参考,用 sigmoid 近似
            return _sigmoid(ml_raw, slope=5.0)

        # 降级:用 acceleration + alpha_industry 近似
        acc = _safe_get(features_row, "acceleration", 0.0)
        alpha = _safe_get(features_row, "alpha_industry_5", 0.0)
        if np.isnan(acc) and np.isnan(alpha):
            return 0.5  # 中位分
        combined = (acc if not np.isnan(acc) else 0.0) + (alpha if not np.isnan(alpha) else 0.0)
        return _sigmoid(combined, slope=20.0)

    def _score_momentum_dim(self, features_row: Optional[pd.Series]) -> float:
        """动量加速维度:acceleration + 行业 alpha"""
        acc = _safe_get(features_row, "acceleration", np.nan)
        alpha5 = _safe_get(features_row, "alpha_industry_5", np.nan)
        alpha20 = _safe_get(features_row, "alpha_industry_20", np.nan)

        parts = []
        if not np.isnan(acc):
            # acceleration 是 ret_5 - ret_10,正常在 ±5% 量级
            parts.append(_sigmoid(acc, slope=20.0))
        if not np.isnan(alpha5):
            parts.append(_sigmoid(alpha5, slope=30.0))
        if not np.isnan(alpha20):
            parts.append(_sigmoid(alpha20, slope=20.0))

        if not parts:
            return 0.5
        return float(np.mean(parts))

    def _score_technical_dim(self, features_row: Optional[pd.Series]) -> float:
        """技术强度维度:RSI + MACD + KDJ

        评分规则(均映射到 [0, 1]):
        - RSI: 50-70 最佳(0.9), >80 过热(0.3), <30 超卖(0.4)
        - MACD hist: 正值加分
        - KDJ J: 20-80 正常区间加分
        """
        rsi = _safe_get(features_row, "rsi_14", np.nan)
        macd_hist = _safe_get(features_row, "macd_hist", np.nan)
        kdj_j = _safe_get(features_row, "kdj_j", np.nan)

        parts = []
        if not np.isnan(rsi):
            if 50 <= rsi <= 70:
                parts.append(0.9)
            elif 70 < rsi <= 80:
                parts.append(0.6)
            elif rsi > 80:
                parts.append(0.3)
            elif 30 <= rsi < 50:
                parts.append(0.5)
            else:  # rsi < 30
                parts.append(0.4)

        if not np.isnan(macd_hist):
            # macd_hist > 0 多头,< 0 空头
            parts.append(_sigmoid(macd_hist, slope=30.0))

        if not np.isnan(kdj_j):
            if 20 <= kdj_j <= 80:
                parts.append(0.7)
            elif kdj_j > 80:
                parts.append(0.3)  # 超买
            else:
                parts.append(0.5)

        if not parts:
            return 0.5
        return float(np.mean(parts))

    def _score_fund_flow_dim(self, features_row: Optional[pd.Series]) -> float:
        """资金筹码维度:融资净流入 + 筹码集中度变化"""
        margin_ratio = _safe_get(features_row, "margin_net_buy_ratio", np.nan)
        winner_chg5 = _safe_get(features_row, "winner_rate_chg_5", np.nan)

        parts = []
        if not np.isnan(margin_ratio):
            # margin_net_buy_ratio 通常在 ±0.05 量级
            parts.append(_sigmoid(margin_ratio, slope=30.0))
        if not np.isnan(winner_chg5):
            # winner_rate_chg_5 是胜率 5 日变化,正值代表筹码成本压力减小
            parts.append(_sigmoid(winner_chg5, slope=20.0))

        if not parts:
            return 0.5
        return float(np.mean(parts))

    def _score_drawdown_dim(
        self,
        profit_rate: float,
        position_info: Dict,
        features_row: Optional[pd.Series],
    ) -> float:
        """回撤距离维度:当前浮盈率 + ATR 波动调整

        - profit_rate 高 → 评分高(保留原浮盈率判据的直觉)
        - atr_pct 过大 → 波动风险高,评分下修
        """
        # 浮盈率主基线
        pnl_base = _sigmoid(profit_rate, slope=20.0)

        # ATR 调整:从 position_info 或 features 读取
        buy_atr = position_info.get("buy_atr_pct") if position_info else None
        cur_atr = _safe_get(features_row, "atr_pct_14", np.nan)

        atr_penalty = 0.0
        if buy_atr is not None and not np.isnan(cur_atr) and buy_atr > 0:
            # 当前 ATR 明显高于买入时 → 波动放大,扣分
            ratio = cur_atr / buy_atr
            if ratio > 1.5:
                atr_penalty = 0.15
            elif ratio > 1.2:
                atr_penalty = 0.08
        elif not np.isnan(cur_atr):
            # 无买入时 ATR 记录,用绝对阈值:cur_atr > 0.05(日均 5%)扣分
            if cur_atr > 0.05:
                atr_penalty = 0.10

        return float(np.clip(pnl_base - atr_penalty, 0.0, 1.0))

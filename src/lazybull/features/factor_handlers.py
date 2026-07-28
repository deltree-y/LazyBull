"""因子处理器 —— 注册表模式替代 builder.py 中 _calculate_features() 的 11 个内联 if-else。

各 handler 返回 {列名: pd.Series} 字典，由 FactorRegistry 统一 concat 到 features，
消除 17+ 次独立 merge 带来的 DataFrame 内存重分配。

同时每类 handler 封装了对应的 column 常量导入与 NaN/默认值回退逻辑。
"""

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from loguru import logger


def _safe_merge_by_ts_code(
    features: pd.DataFrame,
    data: pd.DataFrame,
    merge_cols: List[str],
    handler_name: str,
) -> pd.DataFrame:
    """按 ts_code 做安全 merge，避免重复键导致错位。"""
    subset_cols = ["ts_code"] + [c for c in merge_cols if c in data.columns and c != "ts_code"]
    subset = data[subset_cols].copy()

    if subset.duplicated(subset=["ts_code"], keep=False).any():
        dup_count = int(subset.duplicated(subset=["ts_code"], keep=False).sum())
        logger.warning(
            f"因子处理器 [{handler_name}] 检测到 {dup_count} 条重复 ts_code，"
            "按 ts_code 保留最后一条"
        )
        subset = subset.drop_duplicates(subset=["ts_code"], keep="last")

    merged = features[["ts_code"]].merge(subset, on="ts_code", how="left")
    if len(merged) != len(features):
        raise ValueError(
            f"处理器 [{handler_name}] merge 后行数异常: {len(merged)} != {len(features)}"
        )
    merged.index = features.index
    return merged


def _get_handler_default_columns(name: str) -> List[str]:
    """处理器失败时用于补齐 schema 的默认列名。"""
    if name == "north_flow":
        from ..factors.north_flow import NORTH_COLS

        return list(NORTH_COLS)
    if name == "lhb":
        from ..factors.lhb import LHB_COLS

        return list(LHB_COLS)
    if name == "consensus":
        from ..factors.consensus import CONS_COLS, CONSENSUS_FRESHNESS_COL

        return list(CONS_COLS) + [CONSENSUS_FRESHNESS_COL]
    if name == "cashflow_quality":
        from ..factors.cashflow_quality import CASHFLOW_COLS, CASHFLOW_FRESHNESS_COL

        return list(CASHFLOW_COLS) + [CASHFLOW_FRESHNESS_COL]
    if name == "consensus_revision":
        from ..factors.consensus_revision import (
            CONSENSUS_REVISION_COLS,
            CONSENSUS_REVISION_FRESHNESS_COL,
        )

        return list(CONSENSUS_REVISION_COLS) + [CONSENSUS_REVISION_FRESHNESS_COL]
    return []


# ── Handler 协议 ───────────────────────────────────────────────

class FactorHandler(Protocol):
    """因子处理器协议：接收 features 和当日数据，返回 {列名: Series} 字典。"""

    def apply(
        self,
        features: pd.DataFrame,
        data: Any,
        trade_date: str,
        current_data: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        ...


# ── 各因子 Handler 实现 ────────────────────────────────────────


class FundamentalFactorHandler:
    """基本面因子（fina_indicator 季度前向填充）。"""

    def apply(
        self,
        features: pd.DataFrame,
        data: Optional[pd.DataFrame],
        trade_date: str,
        current_data: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "fundamental")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class MarginFactorHandler:
    """融资融券因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "margin")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class HolderFactorHandler:
    """股东人数因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "holder")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class EarningsFactorHandler:
    """业绩预告/快报因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "earnings")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class CyqPerfFactorHandler:
    """筹码胜率因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        result = {}

        if "weight_avg" in data.columns and "close_adj" in features.columns:
            cyq = data.copy()
            cyq_with_close = cyq.merge(
                features[["ts_code", "close_adj"]], on="ts_code", how="left"
            )
            cyq_with_close["weight_avg_bias"] = np.where(
                cyq_with_close["weight_avg"] > 1e-6,
                (cyq_with_close["close_adj"] - cyq_with_close["weight_avg"])
                / cyq_with_close["weight_avg"],
                np.nan,
            )
            from ..factors.cyq_perf import CYQ_PERF_COLS

            merge_cols = [c for c in CYQ_PERF_COLS if c in cyq_with_close.columns]
            merged = _safe_merge_by_ts_code(features, cyq_with_close, merge_cols, "cyq_perf")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            merge_cols = [
                c
                for c in data.columns
                if c != "ts_code" and c != "weight_avg"
            ]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "cyq_perf")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        return result


class ExpressFactorHandler:
    """业绩快报因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "express")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class FundPortfolioFactorHandler:
    """基金持仓因子。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        if data is None or len(data) == 0:
            return {}
        merge_cols = [c for c in data.columns if c != "ts_code"]
        merged = _safe_merge_by_ts_code(features, data, merge_cols, "fund_portfolio")
        return {col: merged[col] for col in merge_cols if col in merged.columns}


class NorthFlowFactorHandler:
    """北向资金因子（市场级广播）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.north_flow import NORTH_COLS

        result = {}
        if data is not None:
            if isinstance(data, dict) and len(data) > 0:
                for col, val in data.items():
                    result[col] = val
            else:
                for col in NORTH_COLS:
                    result[col] = float("nan")
        return result


class LhbFactorHandler:
    """龙虎榜因子（个股级稀疏，未上榜填 0）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.lhb import LHB_COLS

        result = {}
        if data is not None:
            if len(data) > 0:
                merge_cols = [c for c in data.columns if c != "ts_code"]
                merged = _safe_merge_by_ts_code(features, data, merge_cols, "lhb")
                for col in merge_cols:
                    if col in merged.columns:
                        result[col] = merged[col].fillna(0.0)
                for col in LHB_COLS:
                    if col in result:
                        result[col] = result[col].fillna(0.0)
            else:
                for col in LHB_COLS:
                    result[col] = 0.0
        return result


class ConsensusFactorHandler:
    """一致预期因子（个股级，覆盖度低）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.consensus import CONS_COLS, CONSENSUS_FRESHNESS_COL

        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "consensus")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            for col in CONS_COLS:
                result[col] = float("nan")
        result[CONSENSUS_FRESHNESS_COL] = result.get(CONSENSUS_FRESHNESS_COL, float("nan"))
        return result


class CashflowQualityFactorHandler:
    """现金流质量因子（个股级，季度前向填充）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.cashflow_quality import CASHFLOW_COLS, CASHFLOW_FRESHNESS_COL

        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "cashflow_quality")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
            # 计算 fcf_yield
            if "fcf" in result and "total_mv" in features.columns:
                fcf = result.get("fcf")
                total_mv = features["total_mv"]
                if fcf is not None:
                    result["fcf_yield"] = pd.Series(
                        np.where(total_mv > 1e-6, fcf.values / total_mv.values, np.nan),
                        index=features.index,
                    )
        else:
            for col in CASHFLOW_COLS:
                result[col] = float("nan")
            result[CASHFLOW_FRESHNESS_COL] = float("nan")
        return result


class ConsensusRevisionFactorHandler:
    """一致预期修正因子（基于 report_rc 时序构建）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.consensus_revision import (
            CONSENSUS_REVISION_COLS,
            CONSENSUS_REVISION_FRESHNESS_COL,
        )

        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "consensus_revision")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            for col in CONSENSUS_REVISION_COLS:
                result[col] = float("nan")
            result[CONSENSUS_REVISION_FRESHNESS_COL] = float("nan")
        return result


# ── 注册表 ─────────────────────────────────────────────────────


class FactorRegistry:
    """因子处理器注册表：按注册顺序执行所有 handler，合并列数据后一次性写入 features。"""

    def __init__(self) -> None:
        self._handlers: List[tuple] = []

    def register(self, name: str, handler: FactorHandler, data_getter, enabled=True) -> None:
        """注册一个因子处理器。

        Args:
            name: 因子名称（用于日志）
            handler: FactorHandler 实例
            data_getter: callable(ctx) -> Optional[data]，从上下文获取当日数据
            enabled: 是否启用。支持 bool 或 callable(ctx)->bool
        """
        self._handlers.append((name, handler, data_getter, enabled))

    def apply_all(
        self,
        features: pd.DataFrame,
        ctx: "FeatureContext",  # noqa: F821
        current_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """执行所有已启用的 handler，将所有返回的列一次性合并到 features。

        设计：handler 不各自 merge，只返回列数据 dict → 最后统一 concat 一次。
        """
        all_columns: Dict[str, pd.Series] = {}
        for name, handler, data_getter, enabled in self._handlers:
            enabled_flag = enabled(ctx) if callable(enabled) else bool(enabled)
            if not enabled_flag:
                continue
            data = data_getter(ctx)
            if data is None:
                continue
            try:
                cols = handler.apply(features, data, ctx.trade_date, current_data)
                for col_name, series in cols.items():
                    all_columns[col_name] = series
            except Exception as e:
                logger.error(f"因子处理器 [{name}] 失败，回退 NaN 占位: {e}")
                for col in _get_handler_default_columns(name):
                    if col not in all_columns:
                        all_columns[col] = pd.Series(np.nan, index=features.index)

        if all_columns:
            new_df = pd.DataFrame(all_columns, index=features.index)
            features = pd.concat([features, new_df], axis=1)

        return features


# ── 工厂函数：构建标准注册表 ───────────────────────────────────


def create_factor_registry() -> FactorRegistry:
    """创建包含所有标准因子处理器的注册表。"""
    registry = FactorRegistry()

    registry.register(
        "fundamental",
        FundamentalFactorHandler(),
        lambda ctx: ctx.fundamental_data,
    )
    registry.register(
        "margin",
        MarginFactorHandler(),
        lambda ctx: ctx.margin_data,
    )
    registry.register(
        "holder",
        HolderFactorHandler(),
        lambda ctx: ctx.holder_data,
    )
    registry.register(
        "earnings",
        EarningsFactorHandler(),
        lambda ctx: ctx.earnings_data,
    )
    registry.register(
        "cyq_perf",
        CyqPerfFactorHandler(),
        lambda ctx: ctx.cyq_perf_data,
    )
    registry.register(
        "express",
        ExpressFactorHandler(),
        lambda ctx: ctx.express_data,
    )
    registry.register(
        "fund_portfolio",
        FundPortfolioFactorHandler(),
        lambda ctx: ctx.fund_portfolio_data,
    )
    registry.register(
        "north_flow",
        NorthFlowFactorHandler(),
        lambda ctx: ctx.north_flow_data,
        enabled=lambda ctx: ctx.north_flow_data is not None,
    )
    registry.register(
        "lhb",
        LhbFactorHandler(),
        lambda ctx: ctx.lhb_data,
        enabled=lambda ctx: ctx.lhb_data is not None,
    )
    registry.register(
        "consensus",
        ConsensusFactorHandler(),
        lambda ctx: ctx.consensus_data,
    )
    registry.register(
        "cashflow_quality",
        CashflowQualityFactorHandler(),
        lambda ctx: ctx.cashflow_data,
    )
    registry.register(
        "consensus_revision",
        ConsensusRevisionFactorHandler(),
        lambda ctx: ctx.consensus_revision_data,
    )

    return registry

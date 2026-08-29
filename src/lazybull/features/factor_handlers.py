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
    if name == "dividend_policy":
        from ..factors.dividend import (
            DIVIDEND_COLS,
            DIVIDEND_FRESHNESS_COL,
            DIVIDEND_HIST_MISSING_COL,
        )

        return list(DIVIDEND_COLS) + [DIVIDEND_FRESHNESS_COL, DIVIDEND_HIST_MISSING_COL]
    if name == "pledge":
        from .handlers_announcement import PLEDGE_COLS

        return list(PLEDGE_COLS)
    if name == "share_float":
        from .handlers_announcement import SHARE_FLOAT_COLS

        return list(SHARE_FLOAT_COLS)
    if name == "block_trade":
        from .handlers_announcement import BLOCK_TRADE_COLS

        return list(BLOCK_TRADE_COLS)
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
    ) -> Dict[str, pd.Series]: ...


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

        # weight_avg 为未复权成本价，偏离度必须与未复权收盘价（current_data.close）
        # 同口径计算；误用后复权 close_adj 会混入历史分红送转，使因子退化为
        # adj_factor 代理（≈上市年限×分红送转强度），与设计语义完全背离。
        if "weight_avg" in data.columns and "close" in current_data.columns:
            cyq = data.copy()
            cyq_with_close = cyq.merge(current_data[["ts_code", "close"]], on="ts_code", how="left")
            cyq_with_close["weight_avg_bias"] = np.where(
                cyq_with_close["weight_avg"] > 1e-6,
                (cyq_with_close["close"] - cyq_with_close["weight_avg"])
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
            merge_cols = [c for c in data.columns if c != "ts_code" and c != "weight_avg"]
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
    """现金流质量因子（个股级，季度前向填充，TTM 口径）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.cashflow_quality import (
            _CLIP_FCF_YIELD,
            _MIN_ABS_TOTAL_MV_YUAN,
            CASHFLOW_COLS,
            CASHFLOW_FRESHNESS_COL,
            CASHFLOW_QUALITY_SCHEMA_VERSION,
            CASHFLOW_QUALITY_VERSION_COL,
        )

        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [c for c in data.columns if c != "ts_code"]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "cashflow_quality")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
            # 计算 fcf_yield：FCF(TTM) / 总市值，总市值分母设经济尺度下限并裁剪极端值
            if "fcf" in result and "total_mv" in features.columns:
                fcf = result.get("fcf")
                total_mv = features["total_mv"]
                if fcf is not None:
                    total_mv_yuan = total_mv * 10000.0
                    result["fcf_yield"] = pd.Series(
                        np.where(
                            total_mv_yuan.abs() >= _MIN_ABS_TOTAL_MV_YUAN,
                            np.clip(
                                fcf.values / total_mv_yuan.values,
                                _CLIP_FCF_YIELD[0],
                                _CLIP_FCF_YIELD[1],
                            ),
                            np.nan,
                        ),
                        index=features.index,
                    )
            else:
                result["fcf_yield"] = float("nan")
        else:
            for col in CASHFLOW_COLS:
                result[col] = float("nan")
            result[CASHFLOW_FRESHNESS_COL] = float("nan")

        # 哨兵代表构建管线语义版本而非数据有无：对当日全截面恒写当前版本号
        # （不经过 merge，无数据的股票也写版本号），训练入口据此拦截旧语义分区。
        result[CASHFLOW_QUALITY_VERSION_COL] = pd.Series(
            CASHFLOW_QUALITY_SCHEMA_VERSION, index=features.index
        )
        return result


class ConsensusRevisionFactorHandler:
    """一致预期修正因子（基于 report_rc 时序构建）。"""

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.consensus_revision import (
            CONSENSUS_REVISION_COLS,
            CONSENSUS_REVISION_FRESHNESS_COL,
            CONSENSUS_REVISION_SCHEMA_VERSION,
            CONSENSUS_REVISION_VERSION_COL,
        )

        result = {}
        if data is not None and len(data) > 0:
            merge_cols = [
                c for c in data.columns if c not in ("ts_code", CONSENSUS_REVISION_VERSION_COL)
            ]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "consensus_revision")
            for col in merge_cols:
                if col in merged.columns:
                    result[col] = merged[col]
        else:
            for col in CONSENSUS_REVISION_COLS:
                result[col] = float("nan")
            result[CONSENSUS_REVISION_FRESHNESS_COL] = float("nan")

        # 哨兵代表构建管线版本而非数据有无：对当日全截面恒写当前版本号
        # （不经过 merge，无数据的股票也写 2）。这样新构建分区的哨兵列恒等于
        # 当前版本且无 NaN，训练入口才能可靠拦截混入的旧语义分区。
        result[CONSENSUS_REVISION_VERSION_COL] = pd.Series(
            CONSENSUS_REVISION_SCHEMA_VERSION, index=features.index
        )
        return result


class DividendPolicyFactorHandler:
    """分红政策质量因子（个股级，事件表 PIT 前向填充）。

    缺失语义展开（lookup 仅含"有可见分红事件"或"已公告未除息"的股票行）：
      - 上市满 365 自然日：从未分红 → continuity=0/yield=0/recent=0/hist_missing=1；
        其余状态因子（stability/growth/payout/freshness）为 NaN，days_to_ex=31；
      - 上市不足 365 自然日：全部 NaN。
    yield_hist_12m 由近 12 月累计每股现金分红除以当日未复权收盘价计算。
    """

    def apply(self, features, data, trade_date, current_data) -> Dict[str, pd.Series]:
        from ..factors.dividend import (
            _CASH_12M_ADJ_COL,
            _NO_UPCOMING_EX_DAYS,
            DIVIDEND_COLS,
            DIVIDEND_FRESHNESS_COL,
            DIVIDEND_HIST_MISSING_COL,
            DIVIDEND_POLICY_SCHEMA_VERSION,
            DIVIDEND_POLICY_VERSION_COL,
        )

        result: Dict[str, pd.Series] = {}
        list_days = (
            features["list_days"]
            if "list_days" in features.columns
            else pd.Series(np.nan, index=features.index)
        )
        matured = list_days >= 365

        merged = None
        if data is not None and len(data) > 0:
            merge_cols = [
                c for c in data.columns if c != "ts_code" and c != DIVIDEND_POLICY_VERSION_COL
            ]
            merged = _safe_merge_by_ts_code(features, data, merge_cols, "dividend_policy")

        # ── 事件/状态因子列 ──
        for col in DIVIDEND_COLS:
            if col == "dividend_yield_hist_12m":
                continue  # 单独计算
            if merged is not None and col in merged.columns:
                result[col] = merged[col]
            else:
                result[col] = pd.Series(np.nan, index=features.index)
        if merged is not None and DIVIDEND_FRESHNESS_COL in merged.columns:
            result[DIVIDEND_FRESHNESS_COL] = merged[DIVIDEND_FRESHNESS_COL]
        else:
            result[DIVIDEND_FRESHNESS_COL] = pd.Series(np.nan, index=features.index)

        # ── 未命中行语义展开 ──
        if merged is not None and DIVIDEND_HIST_MISSING_COL in merged.columns:
            has_history = merged[DIVIDEND_HIST_MISSING_COL].eq(0.0)
        else:
            has_history = pd.Series(False, index=features.index)

        # continuity：从未发生除息（上市成熟）→ 0；pre-ex 公告不得改变状态编码
        continuity = result["dividend_continuity_5y"]
        result["dividend_continuity_5y"] = continuity.where(
            ~(matured & ~has_history & continuity.isna()), 0.0
        )
        # yield_hist_12m：近 12 月每股现金累计 / 未复权收盘价
        close_map = None
        if "close" in current_data.columns and "ts_code" in current_data.columns:
            close_map = (
                current_data[["ts_code", "close"]]
                .drop_duplicates(subset=["ts_code"], keep="last")
                .set_index("ts_code")["close"]
            )
            close_series = features["ts_code"].map(close_map).astype(float)
        else:
            close_series = pd.Series(np.nan, index=features.index)
        if merged is not None and _CASH_12M_ADJ_COL in merged.columns:
            cash_12m = merged[_CASH_12M_ADJ_COL]
            yield_series = np.where(close_series > 1e-6, cash_12m / close_series, np.nan)
            yield_series = pd.Series(yield_series, index=features.index)
        else:
            yield_series = pd.Series(np.nan, index=features.index)
        result["dividend_yield_hist_12m"] = yield_series.where(
            ~(matured & ~has_history & yield_series.isna()), 0.0
        )

        # days_to_ex：0~30 为自然日距离；成熟股票无窗口内事件显式编码为 31
        days_to_ex = result["dividend_days_to_ex_date"]
        result["dividend_days_to_ex_date"] = days_to_ex.where(
            ~(matured & days_to_ex.isna()), float(_NO_UPCOMING_EX_DAYS)
        )

        # recent_imp_ann_10d：无公告 → 0
        recent = result["dividend_recent_imp_ann_10d"]
        result["dividend_recent_imp_ann_10d"] = recent.where(
            ~(matured & ~has_history & recent.isna()), 0.0
        )

        # hist_missing：已有 ex_date 历史=0；成熟且尚无落地历史=1；未成熟=NaN
        if merged is not None and DIVIDEND_HIST_MISSING_COL in merged.columns:
            hist = merged[DIVIDEND_HIST_MISSING_COL]
        else:
            hist = pd.Series(np.nan, index=features.index)
        hist = hist.where(hist.notna(), np.where(matured, 1.0, np.nan))
        result[DIVIDEND_HIST_MISSING_COL] = hist

        # 哨兵：对当日全截面恒写当前版本号（含无数据股票），训练入口据此拦截旧语义分区
        result[DIVIDEND_POLICY_VERSION_COL] = pd.Series(
            DIVIDEND_POLICY_SCHEMA_VERSION, index=features.index
        )
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
    registry.register(
        "dividend_policy",
        DividendPolicyFactorHandler(),
        lambda ctx: ctx.dividend_data,
    )

    # 风控公告类（质押/解禁/大宗，PIT 日频截面原始列）
    from .handlers_announcement import (
        BlockTradeFactorHandler,
        PledgeFactorHandler,
        ShareFloatFactorHandler,
    )

    registry.register(
        "pledge",
        PledgeFactorHandler(),
        lambda ctx: ctx.pledge_data,
    )
    registry.register(
        "share_float",
        ShareFloatFactorHandler(),
        lambda ctx: ctx.share_float_data,
    )
    registry.register(
        "block_trade",
        BlockTradeFactorHandler(),
        lambda ctx: ctx.block_trade_data,
    )

    return registry

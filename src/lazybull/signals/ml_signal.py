"""ML 信号生成模块

基于训练好的机器学习模型生成交易信号
使用排序选股 Top N 方式
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.config import get_models_root
from ..ml import ModelRegistry
from ..ml.train_core import (
    DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS,
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
    apply_event_freshness_decay,
)
from .base import Signal


class MLSignal(Signal):
    """ML 信号生成器

    基于机器学习模型预测，选择预测收益最高的 Top N 股票
    """

    # 申万一级行业：银行(801780)、非银金融(801790，含保险/券商)
    _FINANCIAL_SW_L1_CODES = {"801780", "801790"}

    # 数值质量门禁：缺失率告警阈值（超过视为异常数据链路，记录 ERROR 级日志）
    _MISSING_RATE_WARN_THRESHOLD = 0.5

    def __init__(
        self,
        top_n: int = 20,
        model_version: Optional[int] = None,
        models_dir: Optional[str] = None,
        min_amount_ma20: float = 50000.0,
        min_total_mv: float = 500000.0,
        max_total_mv: float = 15000000.0,
        exclude_financial: bool = True,
        verbose: bool = True,
    ):
        """初始化 ML 信号

        Args:
            top_n: 选择 Top N 只股票
            model_version: 模型版本号，None 表示使用最新版本
            models_dir: 模型目录
            min_amount_ma20: 20日均成交额下限（千元），默认50000（=5000万元）
            min_total_mv: 总市值下限（万元），默认500000（=50亿元）
            max_total_mv: 总市值上限（万元），默认15000000（=1500亿元）
            exclude_financial: 是否剔除金融股（银行/非银金融），默认True
            verbose: 是否输出详细日志，默认True
        """
        super().__init__("ml_signal")
        self.top_n = top_n
        self.model_version = model_version
        self.models_dir = models_dir or get_models_root()

        self.min_amount_ma20 = min_amount_ma20
        self.min_total_mv = min_total_mv
        self.max_total_mv = max_total_mv
        self.exclude_financial = exclude_financial
        self.verbose = verbose
        # 延迟加载模型
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.registry = None  # 复用 registry 实例
        # 缓存最近一次排序候选列表（供持仓强势度评分查询）
        self._last_ranked_candidates: List[tuple] = []

        logger.info(
            f"ML 信号初始化: top_n={top_n}, model_version={model_version}, "
            f"min_amount_ma20={min_amount_ma20:.0f}千元, "
            f"total_mv=[{min_total_mv/10000:.0f}亿,{max_total_mv/10000:.0f}亿], "
            f"exclude_financial={exclude_financial}"
        )

    def update_model_version(self, new_version: int) -> None:
        """切换到新模型版本（walk-forward 跨 split 复用时调用）。

        Args:
            new_version: 新模型版本号
        """
        if self.model_version == new_version and self.model is not None:
            return  # 版本未变，无需切换
        old_version = self.model_version
        self.model_version = new_version
        # 重置模型缓存，下次 generate 时触发延迟加载
        self.model = None
        self.metadata = None
        self.feature_columns = None
        logger.info(f"MLSignal 切换模型: v{old_version} → v{new_version}")

    def _load_model(self) -> None:
        """加载模型（延迟加载）"""
        if self.model is None:
            if self.model_version is None:
                raise RuntimeError(
                    "MLSignal.model_version 为 None，请在 generate 前调用 update_model_version()"
                )
            logger.info("开始加载ML模型...")
            self.registry = ModelRegistry(models_dir=self.models_dir)
            # 严格检查：拒绝旧模型
            self.model, self.metadata = self.registry.load_model(
                version=self.model_version, strict_version_check=True
            )
            self.feature_columns = self.metadata["feature_columns"]
            logger.info(
                f"模型已加载: {self.metadata['version_str']}, "
                f"特征数={self.metadata['feature_count']}"
            )

    def _apply_selection_filters(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, int]:
        """选股阶段过滤（实盘/回测共用）

        规则（均使用特征文件中的原始列，z-score 归一化不影响这些列）：
        - 20日均成交额 >= min_amount_ma20（千元，默认50000=5000万）
        - 总市值 in [min_total_mv, max_total_mv]（万元，默认50亿~1500亿）
        - 剔除金融股（申万一级 银行801780/非银金融801790）

        Args:
            features_df: 特征DataFrame

        Returns:
            过滤后的DataFrame、过滤前数量、过滤后数量
        """
        if len(features_df) == 0:
            return features_df, 0, 0

        before = len(features_df)
        mask = pd.Series(True, index=features_df.index)

        # ── 成交额：amount_ma20 原始列（千元）────────────────────────────
        if "amount_ma20" in features_df.columns:
            mask &= features_df["amount_ma20"].fillna(0) >= self.min_amount_ma20
        else:
            logger.warning("选股过滤-成交额: amount_ma20 列不存在，跳过")

        # ── 市值：total_mv 原始列（万元）────────────────────────────────
        if "total_mv" in features_df.columns:
            mask &= features_df["total_mv"].between(self.min_total_mv, self.max_total_mv)
        else:
            logger.warning("选股过滤-市值: total_mv 列不存在，跳过")

        # ── 金融股：sw_l1_code────────────────────────────────────────────
        if self.exclude_financial:
            if "sw_l1_code" not in features_df.columns:
                logger.warning(
                    "选股过滤-金融股: sw_l1_code 列不存在（申万行业数据未加载），跳过此规则"
                )
            else:
                fin_mask = features_df["sw_l1_code"].isin(self._FINANCIAL_SW_L1_CODES)
                mask &= ~fin_mask

        result = features_df[mask].copy()
        return result, before, len(result)

    def _check_feature_quality(self, X: pd.DataFrame) -> bool:
        """检查模型输入特征数值质量，返回是否可继续预测。

        仅对明确的数据完整性失效做硬拒绝：
        - 全空：缺失率 100%（该特征当日完全无数据，无法提供）

        以下情况记录 WARNING 级聚合警告但不阻断（可能是合法状态，不能一票否决整日预测）：
        - 全零：如当日全部股票 is_loss=0、均未上龙虎榜、均不分红等
        - 截面常量：市场环境特征（mkt_*）本就是单日常量广播到全部股票，
          训练期逐日变化，截面内唯一值=1 不代表数据失效
        - 高缺失率（> _MISSING_RATE_WARN_THRESHOLD）：可能是局部数据链路问题

        已知市场级广播列（mkt_*/north_*，单日常量广播到全部股票）的常量/全零
        为设计状态，不产生警告，避免每天为多个广播列制造噪音日志。

        训练侧（prepare.py）已按"整个训练期"判定高缺失/全空/常数并移除，
        推理侧不应以截面分布一票否决；本门禁只拦截最明确的"全空"失效。
        """
        if X is None or len(X) == 0:
            logger.error("特征质量检查：输入为空，拒绝预测")
            return False

        # 市场级广播列前缀：单日常量广播到全部股票，常量/全零为设计状态，不警告
        broadcast_prefixes = ("mkt_", "north_")

        reject_cols = []
        warn_cols = []
        for col in X.columns:
            series = X[col]
            missing_ratio = float(series.isna().mean())
            non_null = series.dropna()
            if missing_ratio >= 1.0 or len(non_null) == 0:
                reject_cols.append((col, "全空"))
            elif missing_ratio > self._MISSING_RATE_WARN_THRESHOLD:
                warn_cols.append(
                    (
                        col,
                        f"缺失率 {missing_ratio:.1%} 超过阈值 "
                        f"{self._MISSING_RATE_WARN_THRESHOLD:.0%}",
                    )
                )
            elif (non_null == 0).all():
                if not col.startswith(broadcast_prefixes):
                    warn_cols.append(
                        (col, "全零（可能为合法状态，如全部不分红/未上榜/未亏损）")
                    )
            elif len(non_null) >= 2 and non_null.nunique() <= 1:
                if not col.startswith(broadcast_prefixes):
                    warn_cols.append((col, "截面常量（无区分度）"))

        for col, reason in reject_cols:
            logger.error(f"特征质量检查未通过: {col} 为{reason}，拒绝本次预测")
        if warn_cols:
            detail = "; ".join(f"{col} {reason}" for col, reason in warn_cols)
            logger.warning(f"特征质量检查警告（不阻断预测）: {detail}")

        return not reject_cols

    def _log_prediction_pipeline_summary(
        self, before_count: int, after_count: int, ranked: bool = False
    ) -> None:
        """将选股过滤与模型预测入口压缩为单行日志。"""
        if ranked:
            return
        universe_text = (
            f"{before_count}→{after_count}" if before_count != after_count else str(after_count)
        )
        logger.info(f"选股/预测: {universe_text}, 特征{len(self.feature_columns)}")

    def _apply_serving_event_decay(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """推理侧按模型训练参数复现事件型 freshness 指数衰减（消除 train/serve skew）。

        训练侧在 state_keep_event_decay 策略下对事件型因子按 freshness 衰减
        并从特征列移除 freshness 列；推理侧必须复现同一衰减，否则模型会在
        未衰减的分布上预测（旧公告以原值全额进入模型）。

        Args:
            features_df: 当日特征 DataFrame（原地修改值列并返回）。

        Returns:
            衰减后的 DataFrame。
        """
        train_params = self.metadata.get("train_params", {}) if self.metadata else {}
        strategy = train_params.get(
            "freshness_strategy", FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
        )
        if strategy != FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY:
            return features_df
        half_life_days = float(
            train_params.get(
                "event_freshness_half_life_days", DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS
            )
        )
        event_freshness_cols = [
            c for c in EVENT_FRESHNESS_TO_VALUE_COLUMNS if c in features_df.columns
        ]
        if not event_freshness_cols:
            return features_df
        features_df, decay_stats = apply_event_freshness_decay(
            features_df,
            event_freshness_cols=event_freshness_cols,
            half_life_days=half_life_days,
        )
        if decay_stats:
            logger.debug(
                f"推理侧事件 freshness 衰减: half_life={half_life_days:.0f}天, "
                + ", ".join(sorted(decay_stats.keys()))
            )
        return features_df

    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成 ML 信号

        Args:
            date: 当前日期
            universe: 股票池（股票代码列表）
            data: 数据字典，应包含 "features" 键，值为当日特征 DataFrame

        Returns:
            信号字典，{股票代码: 权重}
        """
        # 加载模型
        self._load_model()

        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            return {}

        features_df = data["features"]

        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return {}

        # 过滤股票池（布尔索引已创建新对象，无需 .copy()）
        features_df = features_df[features_df["ts_code"].isin(universe)]

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return {}

        # 应用选股过滤（成交额/市值/金融股）
        features_df, filter_before_count, filter_after_count = self._apply_selection_filters(
            features_df
        )

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return {}

        # 特征列一致性检查 (缺失列以 NaN 自动补齐, 由 XGBoost/LightGBM 原生 NaN 处理)
        required_features = set(self.metadata.get("feature_columns", []))
        missing = required_features - set(features_df.columns)
        if missing:
            for col in missing:
                features_df[col] = np.nan
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 推理侧复现训练时的事件型 freshness 衰减（train/serve 一致）
        features_df = self._apply_serving_event_decay(features_df)

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return {}

        # 数值质量门禁：仅拒绝全空列（数据完全缺失），全零/常量/高缺失仅警告不阻断
        if not self._check_feature_quality(X):
            logger.error(f"{date.date()} 特征数值质量异常，跳过预测")
            return {}

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get("train_params", {}).get("task", "regression")
        self._log_prediction_pipeline_summary(
            before_count=filter_before_count,
            after_count=filter_after_count,
            ranked=False,
        )

        if task == "classification" and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)

        features_df["ml_score"] = predictions
        score_column = "ml_score"

        # 按预测分数排序，选择 Top N
        features_df = features_df.sort_values(score_column, ascending=False)
        top_stocks = features_df.head(self.top_n)

        if len(top_stocks) == 0:
            logger.warning(f"{date.date()} 没有有效的预测结果")
            return {}

        # 输出原始 ml_score（供引擎层 _normalize_signals 统一做权重分配）
        # 正分数原样输出；全为负/零时回退到等权（避免引擎层收到无意义负值）
        positive_stocks = top_stocks[top_stocks[score_column] > 0]
        if len(positive_stocks) == 0:
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks["ts_code"].tolist()}
        else:
            scores = positive_stocks[score_column].values
            stocks = positive_stocks["ts_code"].values
            signals = dict(zip(stocks, scores.tolist()))

        logger.debug(
            f"ML 信号生成完成: {date.date()}, 选择 {len(signals)} 只股票, "
            f"平均预测分数={top_stocks[score_column].mean():.6f}"
        )

        return signals

    def generate_ranked(self, date: pd.Timestamp, universe: List[str], data: Dict) -> List[tuple]:
        """生成排序后的候选股票列表（支持回填）

        返回所有候选股票的完整排序列表，而不仅仅是 top N。
        这样可以在 top N 中有不可交易股票时从后续候选中回填。

        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典，应包含 "features" 键

        Returns:
            排序后的 (股票代码, 预测分数) 元组列表，按分数降序排列
        """
        # 加载模型
        self._load_model()

        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            return []

        features_df = data["features"]

        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return []

        # 过滤股票池（布尔索引已创建新对象，无需 .copy()）
        features_df = features_df[features_df["ts_code"].isin(universe)]

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return []

        # 应用选股过滤（成交额/市值/金融股）
        features_df, filter_before_count, filter_after_count = self._apply_selection_filters(
            features_df
        )

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return []

        # 特征列一致性检查 (缺失列以 NaN 自动补齐, 由 XGBoost/LightGBM 原生 NaN 处理)
        required_features = set(self.metadata.get("feature_columns", []))
        missing = required_features - set(features_df.columns)
        if missing:
            for col in missing:
                features_df[col] = np.nan
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 推理侧复现训练时的事件型 freshness 衰减（train/serve 一致）
        features_df = self._apply_serving_event_decay(features_df)

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return []

        # 数值质量门禁：仅拒绝全空列（数据完全缺失），全零/常量/高缺失仅警告不阻断
        if not self._check_feature_quality(X):
            logger.error(f"{date.date()} 特征数值质量异常，跳过预测")
            return []

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get("train_params", {}).get("task", "regression")
        self._log_prediction_pipeline_summary(
            before_count=filter_before_count,
            after_count=filter_after_count,
            ranked=True,
        )

        if task == "classification" and hasattr(self.model, "predict_proba"):
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)

        features_df["ml_score"] = predictions
        score_column = "ml_score"

        # 按预测分数排序，返回所有候选
        features_df = features_df.sort_values(score_column, ascending=False)

        # 返回 (股票代码, 分数) 元组列表
        ranked = list(zip(features_df["ts_code"].tolist(), features_df[score_column].tolist()))

        # 缓存本次完整排序候选列表，供持仓强势度评分（_score_holding_strength）查询 ml_raw_score
        self._last_ranked_candidates = ranked

        return ranked

    def generate_with_features(
        self, date: pd.Timestamp, universe: List[str], features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """使用提供的特征数据生成信号（便捷方法）

        Args:
            date: 当前日期
            universe: 股票池
            features_df: 特征 DataFrame

        Returns:
            信号字典
        """
        data = {"features": features_df}
        return self.generate(date, universe, data)

    def get_model_info(self) -> Dict:
        """获取模型信息

        Returns:
            模型元数据字典
        """
        self._load_model()
        return self.metadata

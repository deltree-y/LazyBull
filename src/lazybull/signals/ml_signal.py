"""ML 信号生成模块

基于训练好的机器学习模型生成交易信号
使用排序选股 Top N 方式
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..ml import ModelRegistry
from .base import Signal


class MLSignal(Signal):
    """ML 信号生成器

    基于机器学习模型预测，选择预测收益最高的 Top N 股票
    """

    # 申万一级行业：银行(801780)、非银金融(801790，含保险/券商)
    _FINANCIAL_SW_L1_CODES = {"801780", "801790"}

    def __init__(
        self,
        top_n: int = 20,
        model_version: Optional[int] = None,
        models_dir: str = "./data/models",
        weight_method: str = "equal",
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
            weight_method: 权重分配方法，"equal" 表示等权，"score" 表示按预测分数加权
            min_amount_ma20: 20日均成交额下限（千元），默认50000（=5000万元）
            min_total_mv: 总市值下限（万元），默认500000（=50亿元）
            max_total_mv: 总市值上限（万元），默认15000000（=1500亿元）
            exclude_financial: 是否剔除金融股（银行/非银金融），默认True
            verbose: 是否输出详细日志，默认True
        """
        super().__init__("ml_signal")
        self.top_n = top_n
        self.model_version = model_version
        self.models_dir = models_dir
        self.weight_method = weight_method
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

        logger.info(
            f"ML 信号初始化: top_n={top_n}, model_version={model_version}, "
            f"weight_method={weight_method}, "
            f"min_amount_ma20={min_amount_ma20:.0f}千元, "
            f"total_mv=[{min_total_mv/10000:.0f}亿,{max_total_mv/10000:.0f}亿], "
            f"exclude_financial={exclude_financial}"
        )

    def _load_model(self) -> None:
        """加载模型（延迟加载）"""
        if self.model is None:
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

    def _apply_selection_filters(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """选股阶段过滤（实盘/回测共用）

        规则（均使用特征文件中的原始列，z-score 归一化不影响这些列）：
        - 20日均成交额 >= min_amount_ma20（千元，默认50000=5000万）
        - 总市值 in [min_total_mv, max_total_mv]（万元，默认50亿~1500亿）
        - 剔除金融股（申万一级 银行801780/非银金融801790）

        Args:
            features_df: 特征DataFrame

        Returns:
            过滤后的DataFrame
        """
        if len(features_df) == 0:
            return features_df

        before = len(features_df)
        mask = pd.Series(True, index=features_df.index)

        # ── 成交额：amount_ma20 原始列（千元）────────────────────────────
        if "amount_ma20" in features_df.columns:
            amount_low = (features_df["amount_ma20"].fillna(0) < self.min_amount_ma20).sum()
            if amount_low > 0:
                #logger.info(
                #    f"  选股过滤-成交额: 剔除 amount_ma20 < {self.min_amount_ma20:.0f}千元"
                #    f"（={self.min_amount_ma20 / 10:.0f}万元）的 {amount_low} 只"
                # )
                pass
            mask &= features_df["amount_ma20"].fillna(0) >= self.min_amount_ma20
        else:
            logger.warning("选股过滤-成交额: amount_ma20 列不存在，跳过")

        # ── 市值：total_mv 原始列（万元）────────────────────────────────
        if "total_mv" in features_df.columns:
            mv_low = (features_df["total_mv"] < self.min_total_mv).sum()
            mv_high = (features_df["total_mv"] > self.max_total_mv).sum()
            if mv_low + mv_high > 0:
                #logger.info(
                #   f"  选股过滤-市值: 剔除 <{self.min_total_mv / 10000:.0f}亿 {mv_low}只, "
                #   f">{self.max_total_mv / 10000:.0f}亿 {mv_high}只"
                #)
                pass
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
                fin_count = fin_mask.sum()
                #if fin_count > 0:
                #    logger.info(f"  选股过滤-金融股: 剔除银行/非银金融 {fin_count} 只")
                mask &= ~fin_mask

        result = features_df[mask].copy()
        if (before - len(result)) > 0:
            logger.info(f"  选股过滤合计: {before} → {len(result)}（剔除 {before - len(result)} 只）")
        return result

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
            logger.info(
                f"data columns: {data['daily'].columns.tolist() if 'daily' in data else 'N/A'}"
            )
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
        features_df = self._apply_selection_filters(features_df)

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return {}

        # 特征列一致性检查
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return {}

        # XGB/LGB 原生支持 NaN，不做 fillna，保留缺失值信息

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        model_type = self.metadata.get("model_type", "unknown")
        task = self.metadata.get("train_params", {}).get("task", "regression")
        logger.info(f"开始模型预测: {len(X)} 只股票, {len(self.feature_columns)} 个特征")

        if task == "classification" and hasattr(self.model, "predict_proba"):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == "classification":
                logger.warning(
                    f"模型声明为 classification，但无 predict_proba 方法，回退到 predict"
                )

        features_df["ml_score"] = predictions

        # 按预测分数排序，选择 Top N
        features_df = features_df.sort_values("ml_score", ascending=False)
        top_stocks = features_df.head(self.top_n)
        logger.info(
            "  TOP预测概率抽样: {}".format(
                features_df[["ts_code", "ml_score"]]
                .head(3)
                .to_string(index=False)
                .replace("\n", " | ")
            )
        )

        if len(top_stocks) == 0:
            logger.warning(f"{date.date()} 没有有效的预测结果")
            return {}

        # 分配权重
        if self.weight_method == "equal":
            # 等权
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks["ts_code"].tolist()}
        elif self.weight_method == "score":
            # 按预测分数加权 — 先过滤负分股票，避免占位但权重为0
            positive_stocks = top_stocks[top_stocks["ml_score"] > 0]
            if len(positive_stocks) == 0:
                # 所有分数都是负数或零，回退到等权
                weight = 1.0 / len(top_stocks)
                signals = {stock: weight for stock in top_stocks["ts_code"].tolist()}
            else:
                # 归一化正分数为权重（使用向量化操作）
                scores = positive_stocks["ml_score"].values
                stocks = positive_stocks["ts_code"].values
                weights = scores / scores.sum()

                signals = dict(zip(stocks, weights))
        else:
            raise ValueError(f"不支持的权重方法: {self.weight_method}")

        logger.debug(
            f"ML 信号生成完成: {date.date()}, 选择 {len(signals)} 只股票, "
            f"平均预测分数={top_stocks['ml_score'].mean():.6f}"
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
        features_df = self._apply_selection_filters(features_df)

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return []

        # 特征列一致性检查
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return []

        # XGB/LGB 原生支持 NaN，不做 fillna

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get("train_params", {}).get("task", "regression")
        logger.info(f"  开始模型预测(ranked): {len(X)} 只股票, {len(self.feature_columns)} 个特征")

        if task == "classification" and hasattr(self.model, "predict_proba"):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == "classification":
                logger.warning(
                    f"模型声明为 classification，但无 predict_proba 方法，回退到 predict"
                )

        features_df["ml_score"] = predictions

        # 按预测分数排序，返回所有候选
        features_df = features_df.sort_values("ml_score", ascending=False)
        if False:
            logger.info(
                "  TOP预测概率抽样: {}".format(
                    features_df[["ts_code", "ml_score"]]
                    .head(3)
                    .to_string(index=False)
                    .replace("\n", " | ")
                )
            )

        # 返回 (股票代码, 分数) 元组列表
        ranked = list(zip(features_df["ts_code"].tolist(), features_df["ml_score"].tolist()))
        if False:
            logger.info(
                f"  ML排序候选生成: {date.date()}, "  # 候选数 {len(ranked)}, "
                f"平均预测分数[{features_df['ml_score'].mean():.3f}], "
                f"最高/最低[{features_df['ml_score'].max():.3f}/{features_df['ml_score'].min():.3f}]"
            )

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


class EnsembleMLSignal(MLSignal):
    """双模型集成信号生成器

    加载两个模型（如 XGBoost + LightGBM），各自预测后对截面排名取加权均值，
    再按综合排名选 Top N。其余逻辑（过滤、权重分配）复用父类 MLSignal。
    """

    def __init__(
        self,
        model_version_a: int,
        model_version_b: int,
        ensemble_weight_a: float = 0.5,
        top_n: int = 20,
        models_dir: str = "./data/models",
        weight_method: str = "equal",
        min_amount_ma20: float = 50000.0,
        min_total_mv: float = 500000.0,
        max_total_mv: float = 15000000.0,
        exclude_financial: bool = True,
        verbose: bool = True,
    ):
        """初始化双模型集成信号

        Args:
            model_version_a: 模型A版本号（如 XGBoost）
            model_version_b: 模型B版本号（如 LightGBM）
            ensemble_weight_a: 模型A的排名权重，模型B权重为 1 - ensemble_weight_a，默认 0.5
            其余参数同 MLSignal
        """
        # 用 model_version_a 作为主模型初始化父类
        super().__init__(
            top_n=top_n,
            model_version=model_version_a,
            models_dir=models_dir,
            weight_method=weight_method,
            min_amount_ma20=min_amount_ma20,
            min_total_mv=min_total_mv,
            max_total_mv=max_total_mv,
            exclude_financial=exclude_financial,
            verbose=verbose,
        )
        self.model_version_b = model_version_b
        self.ensemble_weight_a = ensemble_weight_a
        self.ensemble_weight_b = 1.0 - ensemble_weight_a

        # 模型B的延迟加载
        self.model_b = None
        self.metadata_b = None
        self.feature_columns_b = None

        logger.info(
            f"Ensemble 信号初始化: model_a=v{model_version_a}, model_b=v{model_version_b}, "
            f"weight_a={ensemble_weight_a:.2f}, weight_b={self.ensemble_weight_b:.2f}"
        )

    def _load_model_b(self) -> None:
        """加载模型B（延迟加载）"""
        if self.model_b is None:
            if self.registry is None:
                from ..ml import ModelRegistry

                self.registry = ModelRegistry(models_dir=self.models_dir)
            self.model_b, self.metadata_b = self.registry.load_model(
                version=self.model_version_b, strict_version_check=True
            )
            self.feature_columns_b = self.metadata_b["feature_columns"]
            logger.info(
                f"模型B已加载: {self.metadata_b['version_str']}, "
                f"特征数={self.metadata_b['feature_count']}"
            )

    def _predict_scores(self, X: pd.DataFrame, model, metadata) -> np.ndarray:
        """单模型预测，返回分数数组"""
        task = metadata.get("train_params", {}).get("task", "regression")
        if task == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    def _ensemble_predict(self, features_df: pd.DataFrame):
        """双模型预测 → 排名融合 + 加权原始分数

        Args:
            features_df: 过滤后的特征 DataFrame（已包含 ts_code）

        Returns:
            tuple: (avg_rank, blended_score)
                - avg_rank: pd.Series, 综合排名（值越小排名越靠前），用于选股排序
                - blended_score: pd.Series, 加权原始分数（正数，有区分度），用于 score 加权
        """
        # 模型A预测（XGB/LGB 原生支持 NaN，不做 fillna）
        X_a = features_df[self.feature_columns]
        scores_a = self._predict_scores(X_a, self.model, self.metadata)

        # 模型B预测（特征列可能不同）
        X_b = features_df[self.feature_columns_b]
        scores_b = self._predict_scores(X_b, self.model_b, self.metadata_b)

        s_a = pd.Series(scores_a, index=features_df.index)
        s_b = pd.Series(scores_b, index=features_df.index)

        # 排名融合（用于选股排序）
        rank_a = s_a.rank(ascending=False)
        rank_b = s_b.rank(ascending=False)
        avg_rank = rank_a * self.ensemble_weight_a + rank_b * self.ensemble_weight_b

        # 加权原始分数（用于 score 加权分配权重）
        # 先将两组分数各自 min-max 归一化到 [0,1]，再加权平均，保证正数且有区分度
        def _minmax(s):
            smin, smax = s.min(), s.max()
            if smax - smin < 1e-12:
                return pd.Series(0.5, index=s.index)
            return (s - smin) / (smax - smin)

        norm_a = _minmax(s_a)
        norm_b = _minmax(s_b)
        blended_score = norm_a * self.ensemble_weight_a + norm_b * self.ensemble_weight_b

        return avg_rank, blended_score

    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成 Ensemble 信号"""
        # 加载两个模型
        self._load_model()
        self._load_model_b()

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
            return {}

        # 应用选股过滤
        features_df = self._apply_selection_filters(features_df)
        if len(features_df) == 0:
            return {}

        # 特征一致性检查（两个模型都检查）
        available_features = features_df.columns.tolist()
        self.registry.check_feature_consistency(self.metadata, available_features)
        self.registry.check_feature_consistency(self.metadata_b, available_features)

        # 集成预测
        avg_rank, blended_score = self._ensemble_predict(features_df)

        # 按综合排名选 Top N（排名值越小越好）
        features_df["ensemble_rank"] = avg_rank.values
        features_df["blended_score"] = blended_score.values
        features_df = features_df.sort_values("ensemble_rank", ascending=True)
        top_stocks = features_df.head(self.top_n)

        if self.verbose:
            logger.info(
                f"  Ensemble TOP抽样: {top_stocks[['ts_code', 'ensemble_rank', 'blended_score']].head(3).to_string(index=False).replace(chr(10), ' | ')}"
            )

        if len(top_stocks) == 0:
            return {}

        # 分配权重
        if self.weight_method == "equal":
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks["ts_code"].tolist()}
        elif self.weight_method == "score":
            # score 模式下用归一化加权分数作为权重
            scores = top_stocks["blended_score"].values
            total_score = scores.sum()
            if total_score > 0:
                weights = scores / total_score
            else:
                weights = np.full(len(scores), 1.0 / len(scores))
            signals = dict(zip(top_stocks["ts_code"].tolist(), weights))
        else:
            raise ValueError(f"不支持的权重方法: {self.weight_method}")

        return signals

    def generate_ranked(self, date: pd.Timestamp, universe: List[str], data: Dict) -> List[tuple]:
        """生成排序后的候选股票列表（集成版本）"""
        # 加载两个模型
        self._load_model()
        self._load_model_b()

        if "features" not in data:
            return []

        features_df = data["features"]
        if features_df is None or len(features_df) == 0:
            return []

        features_df = features_df[features_df["ts_code"].isin(universe)]
        if len(features_df) == 0:
            return []

        features_df = self._apply_selection_filters(features_df)
        if len(features_df) == 0:
            return []

        available_features = features_df.columns.tolist()
        self.registry.check_feature_consistency(self.metadata, available_features)
        self.registry.check_feature_consistency(self.metadata_b, available_features)

        # 集成预测
        avg_rank, blended_score = self._ensemble_predict(features_df)
        features_df["ensemble_rank"] = avg_rank.values
        features_df["blended_score"] = blended_score.values
        features_df = features_df.sort_values("ensemble_rank", ascending=True)

        if self.verbose:
            logger.info(
                f"  Ensemble排序候选生成: {date.date()}, "
                f"#候选数 {len(features_df)}, "
                f"blended_score[{features_df['blended_score'].max():.3f}/{features_df['blended_score'].min():.3f}]"
            )

        # 返回 (股票代码, blended_score) — 归一化加权分数，正数且有区分度
        # 按 ensemble_rank 排序（已排好），score 用于下游 score 加权
        ranked = list(zip(features_df["ts_code"].tolist(), features_df["blended_score"].tolist()))
        return ranked

    def get_model_info(self) -> Dict:
        """获取集成模型信息"""
        self._load_model()
        self._load_model_b()
        info = self.metadata.copy()
        info["ensemble"] = True
        info["model_a_version"] = self.metadata["version_str"]
        info["model_b_version"] = self.metadata_b["version_str"]
        info["ensemble_weight_a"] = self.ensemble_weight_a
        info["ensemble_weight_b"] = self.ensemble_weight_b
        return info

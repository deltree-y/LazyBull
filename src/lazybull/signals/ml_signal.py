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
    _FINANCIAL_SW_L1_CODES = {'801780', '801790'}

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
            self.registry = ModelRegistry(models_dir=self.models_dir)
            # 严格检查：拒绝旧模型
            self.model, self.metadata = self.registry.load_model(
                version=self.model_version,
                strict_version_check=True
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
        if 'amount_ma20' in features_df.columns:
            amount_low = (features_df['amount_ma20'].fillna(0) < self.min_amount_ma20).sum()
            if amount_low > 0:
                logger.info(
                    f"选股过滤-成交额: 剔除 amount_ma20 < {self.min_amount_ma20:.0f}千元"
                    f"（={self.min_amount_ma20 / 100:.0f}万元）的 {amount_low} 只"
                )
            mask &= features_df['amount_ma20'].fillna(0) >= self.min_amount_ma20
        else:
            logger.warning("选股过滤-成交额: amount_ma20 列不存在，跳过")

        # ── 市值：total_mv 原始列（万元）────────────────────────────────
        if 'total_mv' in features_df.columns:
            mv_low = (features_df['total_mv'] < self.min_total_mv).sum()
            mv_high = (features_df['total_mv'] > self.max_total_mv).sum()
            if mv_low + mv_high > 0:
                logger.info(
                    f"选股过滤-市值: 剔除 <{self.min_total_mv / 10000:.0f}亿 {mv_low}只, "
                    f">{self.max_total_mv / 10000:.0f}亿 {mv_high}只"
                )
            mask &= features_df['total_mv'].between(self.min_total_mv, self.max_total_mv)
        else:
            logger.warning("选股过滤-市值: total_mv 列不存在，跳过")

        # ── 金融股：sw_l1_code────────────────────────────────────────────
        if self.exclude_financial:
            if 'sw_l1_code' not in features_df.columns:
                logger.warning(
                    "选股过滤-金融股: sw_l1_code 列不存在（申万行业数据未加载），跳过此规则"
                )
            else:
                fin_mask = features_df['sw_l1_code'].isin(self._FINANCIAL_SW_L1_CODES)
                fin_count = fin_mask.sum()
                if fin_count > 0:
                    logger.info(f"选股过滤-金融股: 剔除银行/非银金融 {fin_count} 只")
                mask &= ~fin_mask

        result = features_df[mask].copy()
        if self.verbose and (before - len(result)) > 0:
            logger.info(f"选股过滤合计: {before} → {len(result)}（剔除 {before - len(result)} 只）")
        return result
    
    def generate(
        self,
        date: pd.Timestamp,
        universe: List[str],
        data: Dict
    ) -> Dict[str, float]:
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
            logger.info(f"data columns: {data['daily'].columns.tolist() if 'daily' in data else 'N/A'}")
            return {}
        
        features_df = data["features"]
        
        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return {}
        
        # 过滤股票池
        features_df = features_df[features_df['ts_code'].isin(universe)].copy()
        
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
        
        # 准备特征
        try:
            X = features_df[self.feature_columns].copy()
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return {}

        # 检查高 NaN 比例特征（NaN >30% 时 fillna(0) 可能引入偏差）
        nan_rates = X.isna().mean()
        high_nan_cols = nan_rates[nan_rates > 0.3]
        if len(high_nan_cols) > 0:
            logger.warning(
                f"预测特征中以下列 NaN 比例 >30%（fillna(0) 可能使预测偏离训练分布）: "
                f"{high_nan_cols.round(3).to_dict()}"
            )
        X = X.fillna(0)
        
        # 预测（classification 模型使用 predict_proba 获取正类概率）
        model_type = self.metadata.get('model_type', 'unknown')
        task = self.metadata.get('train_params', {}).get('task', 'regression')
        
        if task == 'classification' and hasattr(self.model, 'predict_proba'):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == 'classification':
                logger.warning(f"模型声明为 classification，但无 predict_proba 方法，回退到 predict")
        
        features_df['ml_score'] = predictions
        
        # 按预测分数排序，选择 Top N
        features_df = features_df.sort_values('ml_score', ascending=False)
        top_stocks = features_df.head(self.top_n)
        logger.info("  TOP预测概率抽样: {}".format(features_df[['ts_code', 'ml_score']].head(3).to_string(index=False).replace('\n', ' | ')))        
        
        if len(top_stocks) == 0:
            logger.warning(f"{date.date()} 没有有效的预测结果")
            return {}
        
        # 分配权重
        if self.weight_method == "equal":
            # 等权
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks['ts_code'].tolist()}
        elif self.weight_method == "score":
            # 按预测分数加权
            total_score = top_stocks['ml_score'].sum()
            if total_score <= 0:
                # 如果所有分数都是负数或零，回退到等权
                weight = 1.0 / len(top_stocks)
                signals = {stock: weight for stock in top_stocks['ts_code'].tolist()}
            else:
                # 归一化分数为权重（使用向量化操作）
                scores = top_stocks['ml_score'].values
                stocks = top_stocks['ts_code'].values
                weights = np.maximum(0, scores) / total_score
                
                # 重新归一化确保权重和为 1
                total_weight = weights.sum()
                if total_weight > 0:
                    weights = weights / total_weight
                
                signals = dict(zip(stocks, weights))
        else:
            raise ValueError(f"不支持的权重方法: {self.weight_method}")
        
        logger.debug(
            f"ML 信号生成完成: {date.date()}, 选择 {len(signals)} 只股票, "
            f"平均预测分数={top_stocks['ml_score'].mean():.6f}"
        )
        
        return signals
    
    def generate_ranked(
        self,
        date: pd.Timestamp,
        universe: List[str],
        data: Dict
    ) -> List[tuple]:
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
        
        # 过滤股票池
        features_df = features_df[features_df['ts_code'].isin(universe)].copy()
        
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
        
        # 准备特征
        try:
            X = features_df[self.feature_columns].copy()
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return []

        nan_rates = X.isna().mean()
        high_nan_cols = nan_rates[nan_rates > 0.3]
        if len(high_nan_cols) > 0:
            logger.warning(
                f"选股特征中以下列 NaN 比例 >30%（fillna(0) 可能使预测偏离训练分布）: "
                f"{high_nan_cols.round(3).to_dict()}"
            )
        X = X.fillna(0)

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get('train_params', {}).get('task', 'regression')
        
        if task == 'classification' and hasattr(self.model, 'predict_proba'):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == 'classification':
                logger.warning(f"模型声明为 classification，但无 predict_proba 方法，回退到 predict")
        
        features_df['ml_score'] = predictions
        
        # 按预测分数排序，返回所有候选
        features_df = features_df.sort_values('ml_score', ascending=False)
        logger.info("  TOP预测概率抽样: {}".format(features_df[['ts_code', 'ml_score']].head(3).to_string(index=False).replace('\n', ' | ')))        

        # 返回 (股票代码, 分数) 元组列表
        ranked = list(zip(features_df['ts_code'].tolist(), features_df['ml_score'].tolist()))
        
        logger.info(
            f"  ML排序候选生成: {date.date()}, "#候选数 {len(ranked)}, "
            f"平均预测分数[{features_df['ml_score'].mean():.3f}], "
            f"最高/最低[{features_df['ml_score'].max():.3f}/{features_df['ml_score'].min():.3f}]"
        )
        
        return ranked
    
    def generate_with_features(
        self,
        date: pd.Timestamp,
        universe: List[str],
        features_df: pd.DataFrame
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

"""多偏移/多子集集成模型包装器

将多个同构模型（如不同训练窗口的 XGBoost）包装为单一模型对象，
对外提供与 XGBRegressor 相同的 predict() 接口，使 MLSignal/ModelRegistry
等下游代码无需任何修改。

典型用法：
    # 多偏移集成（同特征不同窗口）
    models = [xgb_model_0, xgb_model_1, xgb_model_2]
    ensemble = EnsembleModel(models)
    scores = ensemble.predict(X)  # 3个子模型预测取平均

    # 多子集集成（不同特征子集）
    sub_models = [model_momentum, model_fundamental, model_capital]
    sub_features = [momentum_cols, fundamental_cols, capital_cols]
    subset_ensemble = SubsetEnsembleModel(sub_models, sub_features)
    scores = subset_ensemble.predict(X_df)  # 各子模型用各自特征列预测后加权
"""

from typing import Dict, List, Optional

import numpy as np


class EnsembleModel:
    """多偏移集成包装器 — 对子模型预测取平均

    支持 joblib.dump/load 序列化（子模型为可 pickle 的 XGBoost/LightGBM 模型）。
    """

    def __init__(self, models: List):
        """初始化集成模型

        Args:
            models: 子模型列表，每个模型须实现 predict(X) 方法
        """
        if not models:
            raise ValueError("EnsembleModel 至少需要1个子模型")
        self.models = models
        self.n_models = len(models)

    def predict(self, X) -> np.ndarray:
        """所有子模型预测取平均

        Args:
            X: 特征矩阵（DataFrame 或 ndarray）

        Returns:
            平均预测分数数组
        """
        predictions = [m.predict(X) for m in self.models]
        return np.mean(predictions, axis=0)

    def __repr__(self) -> str:
        return f"EnsembleModel(n_models={self.n_models})"


class SubsetEnsembleModel:
    """多特征子集集成包装器 — 每个子模型使用不同特征子集

    对下游透明：predict(X) 接收完整特征 DataFrame，内部自动为每个子模型
    选取对应的特征列。注册模型时 feature_columns 取所有子集的并集。

    支持 joblib.dump/load 序列化。
    """

    def __init__(
        self,
        sub_models: List,
        sub_feature_columns: List[List[str]],
        sub_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        """初始化子集集成模型

        Args:
            sub_models: 子模型列表，每个模型须实现 predict(X) 方法
            sub_feature_columns: 每个子模型对应的特征列名列表
            sub_names: 子模型名称（可选，用于日志），如 ["momentum", "fundamental", "capital"]
            weights: 子模型权重（可选，默认等权），权重之和不要求为1（内部归一化）
        """
        if not sub_models:
            raise ValueError("SubsetEnsembleModel 至少需要1个子模型")
        if len(sub_models) != len(sub_feature_columns):
            raise ValueError("sub_models 与 sub_feature_columns 长度不一致")
        self.sub_models = sub_models
        self.sub_feature_columns = sub_feature_columns
        self.sub_names = sub_names or [f"subset_{i}" for i in range(len(sub_models))]
        self.n_models = len(sub_models)
        # 归一化权重
        if weights is not None:
            w_sum = sum(weights)
            self.weights = [w / w_sum for w in weights]
        else:
            self.weights = [1.0 / self.n_models] * self.n_models
        # 所有子集的特征并集（用于模型注册）
        all_cols = set()
        for cols in sub_feature_columns:
            all_cols.update(cols)
        self.all_feature_columns = sorted(all_cols)

    def predict(self, X) -> np.ndarray:
        """各子模型用各自特征列预测后加权平均

        Args:
            X: 完整特征 DataFrame（必须是 DataFrame，内含所有子集列）

        Returns:
            加权平均预测分数数组
        """
        predictions = []
        for model, cols in zip(self.sub_models, self.sub_feature_columns):
            X_sub = X[cols]
            predictions.append(model.predict(X_sub))
        return np.average(predictions, axis=0, weights=self.weights)

    def __repr__(self) -> str:
        names = ", ".join(self.sub_names)
        return f"SubsetEnsembleModel(n_models={self.n_models}, subsets=[{names}])"

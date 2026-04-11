"""多偏移集成模型包装器

将多个同构模型（如不同训练窗口的 XGBoost）包装为单一模型对象，
对外提供与 XGBRegressor 相同的 predict() 接口，使 MLSignal/ModelRegistry
等下游代码无需任何修改。

典型用法：
    # 多偏移集成（同特征不同窗口）
    models = [xgb_model_0, xgb_model_1, xgb_model_2]
    ensemble = EnsembleModel(models)
    scores = ensemble.predict(X)  # 3个子模型预测取平均
"""

from typing import List

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

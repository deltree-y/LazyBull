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

from typing import List, Optional

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


class TreeLimitedModel:
    """对基础模型施加树数上限的轻量包装器。

    用于 walk-forward 在训练完成后对一组候选树数做后验验证评估，
    最终以统一接口继续参与验证、回测与模型注册。
    """

    def __init__(self, base_model, tree_limit: int, max_trees: Optional[int] = None):
        if tree_limit <= 0:
            raise ValueError(f"tree_limit 必须为正整数，当前值: {tree_limit}")
        self.base_model = base_model
        self.tree_limit = int(tree_limit)
        self.max_trees = self._resolve_max_trees(base_model, max_trees)

    @staticmethod
    def _to_positive_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    @staticmethod
    def _rebuild_base_model_from_legacy_state(state):
        # 兼容早期错误序列化：TreeLimitedModel 的 state 被写成了底层模型 state。
        objective = str(state.get("objective", "")).lower()

        if "_Booster" in state or "booster" in state:
            try:
                import xgboost as xgb

                if objective.startswith("binary:") or objective.startswith("multi:"):
                    model = xgb.XGBClassifier()
                else:
                    model = xgb.XGBRegressor()
                return model
            except Exception:
                return None

        if "booster_" in state:
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor

                if objective.startswith("binary") or objective.startswith("multiclass"):
                    model = LGBMClassifier()
                else:
                    model = LGBMRegressor()
                return model
            except Exception:
                return None

        return None

    @staticmethod
    def _resolve_max_trees(base_model, max_trees: Optional[int]) -> Optional[int]:
        if max_trees is not None:
            return int(max_trees)

        # XGBoost: 实际可用树数应以 booster 已训练轮数为准，
        # 不能直接使用 n_estimators（可能远大于实际训练轮数）
        if hasattr(base_model, "get_booster"):
            try:
                booster = base_model.get_booster()
                if hasattr(booster, "num_boosted_rounds"):
                    rounds = int(booster.num_boosted_rounds())
                    if rounds > 0:
                        return rounds
            except Exception:
                pass

        # LightGBM: booster_.current_iteration() 返回当前有效迭代轮数
        if hasattr(base_model, "booster_"):
            try:
                rounds = int(base_model.booster_.current_iteration())
                if rounds > 0:
                    return rounds
            except Exception:
                pass

        for attr in ("n_estimators", "n_estimators_"):
            value = getattr(base_model, attr, None)
            if value is None:
                continue
            try:
                resolved = int(value)
            except (TypeError, ValueError):
                continue
            if resolved > 0:
                return resolved
        return None

    def _effective_tree_limit(self) -> Optional[int]:
        tree_limit = self._to_positive_int(getattr(self, "tree_limit", None))
        max_trees = self._to_positive_int(getattr(self, "max_trees", None))

        if tree_limit is None:
            return max_trees
        if max_trees is None:
            return tree_limit
        return min(tree_limit, max_trees)

    def predict(self, X) -> np.ndarray:
        tree_limit = self._effective_tree_limit()
        base_model = self.__dict__.get("base_model")
        if base_model is None:
            raise AttributeError("base_model")

        if tree_limit is None:
            return base_model.predict(X)

        if hasattr(base_model, "get_booster"):
            return base_model.predict(X, iteration_range=(0, tree_limit))
        if hasattr(base_model, "booster_"):
            return base_model.predict(X, num_iteration=tree_limit)
        return base_model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        tree_limit = self._effective_tree_limit()
        base_model = self.__dict__.get("base_model")
        if base_model is None:
            raise AttributeError("base_model")

        if tree_limit is None:
            return base_model.predict_proba(X)

        if hasattr(base_model, "get_booster"):
            return base_model.predict_proba(X, iteration_range=(0, tree_limit))
        if hasattr(base_model, "booster_"):
            return base_model.predict_proba(X, num_iteration=tree_limit)
        return base_model.predict_proba(X)

    def __getattr__(self, item):
        # 反序列化早期对象尚未恢复完整 __dict__，
        # 这里不能直接访问 self.base_model，否则可能触发递归。
        if item == "base_model":
            raise AttributeError(item)

        # 向后兼容：旧版本持久化对象可能没有 tree_limit/max_trees 字段。
        if item in {"tree_limit", "max_trees"}:
            return None

        base_model = self.__dict__.get("base_model")
        if base_model is None:
            raise AttributeError(item)
        return getattr(base_model, item)

    def __setstate__(self, state):
        self.__dict__.update(state)

        if "base_model" not in self.__dict__ or self.__dict__.get("base_model") is None:
            rebuilt = self._rebuild_base_model_from_legacy_state(self.__dict__)
            if rebuilt is not None:
                legacy_state = {
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in {"base_model", "tree_limit", "max_trees"}
                }
                rebuilt.__dict__.update(legacy_state)
                self.base_model = rebuilt

        base_model = self.__dict__.get("base_model")
        resolved_max_trees = self._to_positive_int(self.__dict__.get("max_trees"))
        if resolved_max_trees is None and base_model is not None:
            resolved_max_trees = self._resolve_max_trees(base_model, None)
        self.max_trees = resolved_max_trees

        # 向后兼容：旧版本未持久化 tree_limit 时，加载后自动补齐。
        resolved_tree_limit = self._to_positive_int(self.__dict__.get("tree_limit"))
        if resolved_tree_limit is None:
            resolved_tree_limit = resolved_max_trees
        self.tree_limit = resolved_tree_limit

    def __getstate__(self):
        return {
            "base_model": self.__dict__.get("base_model"),
            "tree_limit": self.__dict__.get("tree_limit"),
            "max_trees": self.__dict__.get("max_trees"),
        }

    def __repr__(self) -> str:
        base_model = self.__dict__.get("base_model")
        tree_limit = self.__dict__.get("tree_limit")
        max_trees = self.__dict__.get("max_trees")
        return (
            f"TreeLimitedModel(tree_limit={tree_limit}, max_trees={max_trees}, "
            f"base_model={base_model!r})"
        )

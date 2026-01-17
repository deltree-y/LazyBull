"""模型版本管理模块

负责管理训练的机器学习模型版本，包括：
- 模型版本号自动递增
- 模型元数据记录
- 模型保存和加载
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from loguru import logger


class ModelRegistry:
    """模型注册表
    
    管理模型版本和元数据，自动维护版本号递增
    """
    
    def __init__(self, models_dir: str = "./data/models"):
        """初始化模型注册表
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"模型注册表初始化完成: {self.models_dir}")
    
    def _load_registry(self) -> Dict:
        """加载注册表文件
        
        Returns:
            注册表字典
        """
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"models": [], "next_version": 1}
    
    def _save_registry(self) -> None:
        """保存注册表到文件"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
        logger.debug(f"注册表已保存: {self.registry_file}")
    
    def get_next_version(self) -> int:
        """获取下一个可用版本号
        
        Returns:
            版本号
        """
        return self.registry.get("next_version", 1)
    
    def register_model(
        self,
        model,
        model_type: str,
        train_start_date: str,
        train_end_date: str,
        feature_columns: List[str],
        label_column: str,
        n_samples: int,
        train_params: Dict,
        performance_metrics: Optional[Dict] = None
    ) -> int:
        """注册新模型
        
        Args:
            model: 训练好的模型对象
            model_type: 模型类型（如 "xgboost"）
            train_start_date: 训练开始日期
            train_end_date: 训练结束日期
            feature_columns: 特征列名列表
            label_column: 标签列名
            n_samples: 训练样本数
            train_params: 训练超参数
            performance_metrics: 性能指标（可选）
            
        Returns:
            模型版本号
        """
        version = self.get_next_version()
        version_str = f"v{version}"
        
        # 保存模型文件
        model_file = self.models_dir / f"{version_str}_model.joblib"
        joblib.dump(model, model_file)
        logger.info(f"模型已保存: {model_file}")
        
        # 保存特征列表
        features_file = self.models_dir / f"{version_str}_features.json"
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(feature_columns, f, ensure_ascii=False, indent=2)
        
        # 记录元数据
        metadata = {
            "version": version,
            "version_str": version_str,
            "model_type": model_type,
            "model_file": str(model_file.name),
            "features_file": str(features_file.name),
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "feature_count": len(feature_columns),
            "label_column": label_column,
            "n_samples": n_samples,
            "train_params": train_params,
            "performance_metrics": performance_metrics or {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 更新注册表
        self.registry["models"].append(metadata)
        self.registry["next_version"] = version + 1
        self._save_registry()
        
        logger.info(
            f"模型已注册: {version_str}, 类型={model_type}, "
            f"训练区间={train_start_date}至{train_end_date}, "
            f"特征数={len(feature_columns)}, 样本数={n_samples}"
        )
        
        return version
    
    def load_model(self, version: Optional[int] = None) -> tuple:
        """加载模型
        
        Args:
            version: 模型版本号，None表示加载最新版本
            
        Returns:
            (model, metadata) 元组
        """
        if not self.registry["models"]:
            raise ValueError("没有已注册的模型。请先使用 train_ml_model.py 训练模型。")
        
        if version is None:
            # 加载最新版本
            metadata = self.registry["models"][-1]
        else:
            # 加载指定版本
            metadata = None
            for m in self.registry["models"]:
                if m["version"] == version:
                    metadata = m
                    break
            
            if metadata is None:
                available_versions = [m["version"] for m in self.registry["models"]]
                raise ValueError(
                    f"未找到版本 {version} 的模型。"
                    f"可用版本: {available_versions}"
                )
        
        # 加载模型文件
        model_file = self.models_dir / metadata["model_file"]
        model = joblib.load(model_file)
        
        # 加载特征列表
        features_file = self.models_dir / metadata["features_file"]
        with open(features_file, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
        
        metadata["feature_columns"] = feature_columns
        
        logger.info(
            f"模型已加载: {metadata['version_str']}, "
            f"训练区间={metadata['train_start_date']}至{metadata['train_end_date']}"
        )
        
        return model, metadata
    
    def list_models(self) -> List[Dict]:
        """列出所有已注册的模型
        
        Returns:
            模型元数据列表
        """
        return self.registry["models"]
    
    def get_latest_version(self) -> Optional[int]:
        """获取最新模型版本号
        
        Returns:
            最新版本号，如果没有模型则返回 None
        """
        if not self.registry["models"]:
            return None
        return self.registry["models"][-1]["version"]

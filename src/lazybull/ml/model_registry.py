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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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
    
    def load_model(self, version: Optional[int] = None, strict_version_check: bool = True) -> tuple:
        """加载模型
        
        Args:
            version: 模型版本号，None表示加载最新版本
            strict_version_check: 是否严格检查模型版本元数据（默认 True）
            
        Returns:
            (model, metadata) 元组
            
        Raises:
            ValueError: 当 strict_version_check=True 且模型缺少新版本必需元数据时
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
        
        # 严格模式：检查新版本必需的元数据字段
        if strict_version_check:
            required_fields = ['feature_columns', 'train_params', 'model_type']
            missing_fields = []
            
            # 检查是否有 feature_columns（可能在单独文件中）
            features_file = self.models_dir / metadata.get("features_file", "")
            if not features_file.exists():
                missing_fields.append('feature_columns (features_file 不存在)')
            
            # 检查 train_params
            if 'train_params' not in metadata or not metadata['train_params']:
                missing_fields.append('train_params')
            
            # 检查 model_type
            if 'model_type' not in metadata:
                missing_fields.append('model_type')
            
            if missing_fields:
                raise ValueError(
                    f"旧模型（版本 {metadata.get('version_str', 'unknown')}）缺少新版本必需的元数据字段：{', '.join(missing_fields)}。\n"
                    f"这些字段对于特征列一致性检查和模型推理至关重要。\n"
                    f"请重新训练模型以生成包含完整元数据的新版本。"
                )
        
        # 加载模型文件
        model_file = self.models_dir / metadata["model_file"]
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        model = joblib.load(model_file)
        
        # 加载特征列表
        features_file = self.models_dir / metadata["features_file"]
        if not features_file.exists():
            raise FileNotFoundError(f"特征列表文件不存在: {features_file}")
        
        with open(features_file, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
        
        metadata["feature_columns"] = feature_columns
        
        logger.info(
            f"模型已加载: {metadata['version_str']}, "
            f"训练区间={metadata['train_start_date']}至{metadata['train_end_date']}"
        )
        
        return model, metadata
    
    def check_feature_consistency(
        self,
        model_metadata: Dict,
        available_features: List[str]
    ) -> None:
        """检查特征列一致性
        
        检查推理数据是否包含模型训练时使用的所有特征列。
        
        Args:
            model_metadata: 模型元数据（来自 load_model）
            available_features: 推理数据中可用的特征列
            
        Raises:
            ValueError: 当缺少必需的特征列时
        """
        if "feature_columns" not in model_metadata:
            raise ValueError("模型元数据中缺少 feature_columns 信息，无法进行特征一致性检查")
        
        required_features = set(model_metadata["feature_columns"])
        available_features_set = set(available_features)
        
        missing_features = required_features - available_features_set
        
        if missing_features:
            raise ValueError(
                f"推理数据缺少模型训练时使用的特征列（共 {len(missing_features)} 个）：\n"
                f"{sorted(list(missing_features))[:20]}{'...' if len(missing_features) > 20 else ''}\n"
                f"模型训练特征数: {len(required_features)}, 当前数据特征数: {len(available_features_set)}\n"
                f"请确保推理数据包含模型训练时的所有特征列。"
            )
        
        logger.debug(
            f"特征列一致性检查通过：模型需要 {len(required_features)} 个特征，"
            f"数据提供 {len(available_features_set)} 个特征"
        )
    
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

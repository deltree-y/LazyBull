"""模型版本管理模块

负责管理训练的机器学习模型版本，包括：
- 模型版本号自动递增
- 模型元数据记录
- 模型保存和加载
"""

import json
import re
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
from loguru import logger

from ..common.config import get_models_root


@contextmanager
def _suppress_xgboost_pickle_warning():
    """上下文管理器：抑制 XGBoost 跨版本 pickle 反序列化告警。

    XGBoost 在 pickle 加载旧版模型时会从 pickle.py 模块发出 UserWarning，
    无法通过 module="xgboost" 过滤。使用 message 匹配精确屏蔽。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*loading a serialized model.*",
        )
        yield


def _load_xgboost_native(file_path: str, model_type: str):
    """从 XGBoost 原生 JSON/UBJSON 文件加载模型。

    根据 model_type 自动选择 XGBRegressor 或 XGBClassifier。
    """
    import xgboost as xgb

    is_classification = "classification" in (model_type or "").lower()
    if is_classification:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(file_path)
    return model


class ModelRegistry:
    """模型注册表
    
    管理模型版本和元数据，自动维护版本号递增
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """初始化模型注册表
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = Path(models_dir or get_models_root())
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.models_dir / "model_registry.json"
        self.latest_version_file = self.models_dir / "latest_model_version.txt"
        self.registry: Optional[Dict] = None

        logger.info(f"模型注册表初始化完成: {self.models_dir}")

    def _ensure_registry_loaded(self) -> Dict:
        """按需加载完整注册表。"""
        if self.registry is None:
            self.registry = self._load_registry()
        return self.registry

    def _metadata_file(self, version: int) -> Path:
        """返回单模型元数据旁路文件路径。"""
        return self.models_dir / f"v{version}_metadata.json"

    def _load_metadata_sidecar(self, version: int) -> Optional[Dict]:
        """优先读取单模型元数据，避免整包加载大注册表。"""
        metadata_file = self._metadata_file(version)
        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_metadata_sidecar(self, metadata: Dict) -> None:
        """保存单模型元数据旁路文件。"""
        version = metadata.get("version")
        if version is None:
            return

        metadata_file = self._metadata_file(int(version))
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _read_latest_version_file(self) -> Optional[int]:
        """读取最新版本旁路文件。"""
        if not self.latest_version_file.exists():
            return None

        raw_value = self.latest_version_file.read_text(encoding='utf-8').strip()
        if not raw_value:
            return None

        try:
            return int(raw_value)
        except ValueError:
            logger.warning(
                f"最新模型版本旁路文件损坏，忽略: {self.latest_version_file}"
            )
            return None

    def _save_latest_version_file(self, version: int) -> None:
        """保存最新版本旁路文件。"""
        self.latest_version_file.write_text(str(version), encoding='utf-8')

    def _load_next_version_from_registry_tail(self) -> Optional[int]:
        """从注册表尾部快速读取 next_version。"""
        if not self.registry_file.exists():
            return None

        file_size = self.registry_file.stat().st_size
        tail_size = min(file_size, 8192)

        with open(self.registry_file, 'rb') as f:
            if file_size > tail_size:
                f.seek(-tail_size, 2)
            tail_text = f.read().decode('utf-8', errors='ignore')

        match = re.search(r'"next_version"\s*:\s*(\d+)', tail_text)
        if match is None:
            return None

        return int(match.group(1))

    def _extract_metadata_from_registry(self, version: int) -> Optional[Dict]:
        """从大注册表中按版本流式提取单条元数据。"""
        if not self.registry_file.exists():
            return None

        target_line = f'"version": {version},'
        previous_line = ""

        with open(self.registry_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() != target_line:
                    previous_line = line
                    continue

                collected_lines = [previous_line if previous_line.strip() == "{" else "{\n", line]
                brace_balance = 1 + line.count("{") - line.count("}")

                for next_line in f:
                    collected_lines.append(next_line)
                    brace_balance += next_line.count("{") - next_line.count("}")
                    if brace_balance == 0:
                        metadata_text = "".join(collected_lines).rstrip()
                        if metadata_text.endswith(","):
                            metadata_text = metadata_text[:-1]

                        metadata = json.loads(metadata_text)
                        if metadata.get("version") == version:
                            return metadata
                        return None

                break

        return None

    def _load_metadata(self, version: int) -> Optional[Dict]:
        """按版本读取元数据，优先走快速旁路。"""
        metadata = self._load_metadata_sidecar(version)
        if metadata is not None:
            return metadata

        metadata = self._extract_metadata_from_registry(version)
        if metadata is not None:
            self._save_metadata_sidecar(metadata)
            return metadata

        registry = self._ensure_registry_loaded()
        for candidate in registry.get("models", []):
            if candidate.get("version") == version:
                self._save_metadata_sidecar(candidate)
                return candidate

        return None
    
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
        registry = self._ensure_registry_loaded()
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

        latest_version = self.get_latest_version()
        if latest_version is not None:
            self._save_latest_version_file(latest_version)

        logger.debug(f"注册表已保存: {self.registry_file}")
    
    def get_next_version(self) -> int:
        """获取下一个可用版本号
        
        Returns:
            版本号
        """
        registry = self._ensure_registry_loaded()
        return registry.get("next_version", 1)
    
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
        performance_metrics: Optional[Dict] = None,
        risk_penalty_config: Optional[Dict] = None,
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

        # 保存模型文件（优先 XGBoost 原生格式，避免跨版本 pickle 告警）
        native_saved = False
        if hasattr(model, "save_model"):
            try:
                model_file = self.models_dir / f"{version_str}_model.json"
                model.save_model(str(model_file))
                native_saved = True
                logger.info(f"模型已保存（XGBoost 原生格式）: {model_file}")
            except Exception as exc:
                logger.warning(f"XGBoost 原生保存失败，回退到 joblib: {exc}")

        if not native_saved:
            model_file = self.models_dir / f"{version_str}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"模型已保存（joblib）: {model_file}")
            # 清理可能存在的残留 .json 僵尸文件（避免加载时误读旧分类器）
            stale_json = self.models_dir / f"{version_str}_model.json"
            if stale_json.exists():
                stale_json.unlink()
                logger.debug(f"已清理残留文件: {stale_json.name}")
        else:
            # 清理可能存在的残留 .joblib 文件
            stale_joblib = self.models_dir / f"{version_str}_model.joblib"
            if stale_joblib.exists():
                stale_joblib.unlink()
                logger.debug(f"已清理残留文件: {stale_joblib.name}")
        
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
        if risk_penalty_config is not None:
            # 若为条件式坏票模型（v2），先注册分类器
            clf_model = risk_penalty_config.pop("_clf_model", None)
            clf_features = risk_penalty_config.pop("_clf_features", None)
            if clf_model is not None:
                clf_version = version + 1  # 分类器版本 = 主模型+1
                clf_version_str = f"v{clf_version}"
                try:
                    clf_model_file = self.models_dir / f"{clf_version_str}_model.json"
                    clf_model.save_model(str(clf_model_file))
                    clf_metadata = {
                        "version": clf_version,
                        "version_str": clf_version_str,
                        "model_type": "xgboost_classifier",
                        "model_file": str(clf_model_file.name),
                        "features_file": "",
                        "train_start_date": train_start_date,
                        "train_end_date": train_end_date,
                        "feature_count": len(clf_features) if clf_features else 0,
                        "label_column": "bad_pick",
                        "n_samples": 0,
                        "train_params": {"purpose": "bad_pick_classifier"},
                        "performance_metrics": {},
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "risk_penalty_config": risk_penalty_config,
                    }
                    # 保存分类器特征
                    if clf_features:
                        clf_feat_file = self.models_dir / f"{clf_version_str}_features.json"
                        with open(clf_feat_file, 'w', encoding='utf-8') as ff:
                            json.dump(list(clf_features), ff, ensure_ascii=False, indent=2)
                        clf_metadata["features_file"] = str(clf_feat_file.name)
                    self._save_metadata_sidecar(clf_metadata)
                    # 暂存到内存注册表（最终由外层统一落盘，避免两次 _save_registry 导致 next_version 回滚）
                    registry = self._ensure_registry_loaded()
                    registry["models"].append(clf_metadata)
                    if clf_version >= registry.get("next_version", 1):
                        registry["next_version"] = clf_version + 1
                    # 回填 bad_pick_model_version
                    risk_penalty_config["bad_pick_model_version"] = clf_version
                    logger.info(f"坏票分类器已注册: v{clf_version}")
                except Exception as exc:
                    logger.warning(f"坏票分类器注册失败: {exc}")
                    risk_penalty_config["bad_pick_model_version"] = 0
            metadata["risk_penalty_config"] = risk_penalty_config

        self._save_metadata_sidecar(metadata)
        self._save_latest_version_file(version)
        
        # 更新注册表（next_version 取主模型和分类器的最大值，避免版本回滚）
        registry = self._ensure_registry_loaded()
        registry["models"].append(metadata)
        next_ver = max(version + 1, registry.get("next_version", version + 1))
        registry["next_version"] = next_ver
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
        if version is None:
            latest_version = self.get_latest_version()
            if latest_version is None:
                raise ValueError("没有已注册的模型。请先使用 train_ml_model.py 训练模型。")
            version = latest_version

        metadata = self._load_metadata(version)
        if metadata is None:
            available_versions = [m["version"] for m in self.list_models()]
            raise ValueError(
                f"未找到版本 {version} 的模型。"
                f"可用版本: {available_versions}"
            )

        metadata = dict(metadata)
        
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
        # 优先按 metadata 中记录的实际文件名加载，避免被残留的 .json 僵尸文件误导
        model_file = self.models_dir / metadata["model_file"]
        native_file = self.models_dir / f"v{version}_model.json"

        if model_file.exists():
            if model_file.suffix == ".json":
                model = _load_xgboost_native(str(model_file), metadata.get("model_type", ""))
            else:
                with _suppress_xgboost_pickle_warning():
                    model = joblib.load(model_file)
        elif native_file.exists():
            # metadata 中记录的文件缺失，回退到同名 .json（可能是旧格式迁移场景）
            model = _load_xgboost_native(str(native_file), metadata.get("model_type", ""))
        else:
            raise FileNotFoundError(
                f"模型文件不存在: {model_file} (native: {native_file})"
            )
        
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
        registry = self._ensure_registry_loaded()
        return registry["models"]
    
    def get_latest_version(self) -> Optional[int]:
        """获取最新模型版本号
        
        Returns:
            最新版本号，如果没有模型则返回 None
        """
        latest_version = self._read_latest_version_file()
        if latest_version is not None:
            return latest_version

        next_version = self._load_next_version_from_registry_tail()
        if next_version is not None and next_version > 1:
            latest_version = next_version - 1
            self._save_latest_version_file(latest_version)
            return latest_version

        registry = self._ensure_registry_loaded()
        if not registry["models"]:
            return None

        latest_version = registry["models"][-1]["version"]
        self._save_latest_version_file(latest_version)
        return latest_version

"""配置管理模块"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """配置管理类
    
    支持从YAML文件加载配置，并支持环境变量覆盖
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置
        
        Args:
            config_path: 配置文件路径，如不提供则使用默认base.yaml
        """
        self._config: Dict[str, Any] = {}
        
        # 加载环境变量
        load_dotenv()
        
        # 加载配置文件
        if config_path:
            self.load_config(config_path)
        else:
            # 加载默认配置
            default_config = Path(__file__).parent.parent.parent.parent / "configs" / "base.yaml"
            if default_config.exists():
                self.load_config(str(default_config))
    
    def load_config(self, config_path: str) -> None:
        """加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self._config.update(config or {})
    
    def merge_config(self, config_path: str) -> None:
        """合并另一个配置文件（覆盖已有配置）
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self._deep_update(self._config, config or {})
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """深度更新字典
        
        Args:
            base: 基础字典
            update: 更新字典
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'data.root' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项
        
        Args:
            key: 配置键，支持 'data.root' 格式
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """获取环境变量
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        return os.getenv(key, default)
    
    @property
    def all(self) -> Dict[str, Any]:
        """返回所有配置"""
        return self._config.copy()


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def init_config(config_path: str) -> Config:
    """初始化全局配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置实例
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


def normalize_shenwan_level(level: Optional[str], default: str = "l2") -> str:
    """标准化申万行业层级配置。

    Args:
        level: 原始层级值，允许为 None
        default: 当 level 为空时使用的默认值

    Returns:
        标准化后的层级字符串（l1/l2/l3）

    Raises:
        ValueError: 当层级值不在支持范围内时抛出
    """
    normalized = str(level or default).strip().lower()
    if normalized not in {"l1", "l2", "l3"}:
        raise ValueError(
            f"shenwan_level 仅支持 'l1'、'l2'、'l3'，当前值: {level}"
        )
    return normalized


def get_shenwan_level(default: str = "l2") -> str:
    """从项目级配置获取申万行业主口径层级。"""
    config = get_config()
    return normalize_shenwan_level(config.get("industry.shenwan_level", default), default=default)


def get_data_root(default: str = "./data") -> str:
    """从项目级配置获取数据根目录。"""
    config = get_config()
    return str(config.get("data.root", default))


def get_data_path(name: str, default: Optional[str] = None) -> str:
    """从项目级配置获取数据子目录。"""
    config = get_config()
    root = Path(get_data_root())
    fallback = default or str(root / name)
    configured = config.get(f"data.{name}")
    if configured is None:
        return fallback

    normalized = str(configured).replace("\\", "/")
    default_templates = {f"./data/{name}", f"data/{name}"}
    if normalized in default_templates:
        return str(root / name)
    return str(configured)


def get_paper_root(default: Optional[str] = None) -> str:
    """获取纸面交易数据目录，默认派生自 data.root/paper。"""
    fallback = default or str(Path(get_data_root()) / "paper")
    return str(get_config().get("data.paper", fallback))


def get_models_root(default: Optional[str] = None) -> str:
    """获取模型目录，默认派生自 data.root/models。"""
    return default or str(Path(get_data_root()) / "models")


def get_reports_root(default: Optional[str] = None) -> str:
    """获取报告目录，优先读取 data.reports。"""
    return get_data_path("reports", default=default)


def get_tushare_settings() -> Dict[str, Any]:
    """获取 TuShare 默认配置。

    Returns:
        max_retries / retry_delay / rate_limit : 基础限频参数
        download_concurrency : 下载脚本的并发线程数 (1=串行)
        rate_limit_error_keywords : 识别"限流"异常的错误关键字
        retry_rate_limit_sleep : 命中限流关键字时的长等秒数
    """
    config = get_config()
    kw = config.get(
        "tushare.rate_limit_error_keywords",
        ["每分钟", "访问", "频次", "rate", "limit", "频率", "429", "超过"],
    )
    return {
        "max_retries": int(config.get("tushare.max_retries", 3)),
        "retry_delay": float(config.get("tushare.retry_delay", 1.0)),
        "rate_limit": int(config.get("tushare.rate_limit", 500)),
        "download_concurrency": int(config.get("tushare.download_concurrency", 1)),
        "rate_limit_error_keywords": [str(k).lower() for k in kw],
        "retry_rate_limit_sleep": float(config.get("tushare.retry_rate_limit_sleep", 15.0)),
    }


def get_cost_settings() -> Dict[str, float]:
    """获取交易成本默认配置。"""
    config = get_config()
    return {
        "commission_rate": float(config.get("costs.commission_rate", 0.0001954)),
        "min_commission": float(config.get("costs.min_commission", 5.0)),
        "stamp_tax": float(config.get("costs.stamp_tax", 0.0005)),
        "slippage": float(config.get("costs.slippage", 0.0005)),
    }

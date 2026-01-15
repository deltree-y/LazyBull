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

"""日志工具模块"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None
) -> None:
    """配置日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径，None则只输出到控制台
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_string: 日志格式字符串
    """
    # 移除默认handler
    logger.remove()
    
    # 默认格式
    if format_string is None:
        format_string = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            #"<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            #"<cyan>{name}</cyan> - "
            "<level>{message}</level>"
        )
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True
    )
    
    # 添加文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    logger.info(f"日志系统初始化完成，级别: {log_level}")


def get_logger(name: str):
    """获取logger实例
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return logger.bind(name=name)

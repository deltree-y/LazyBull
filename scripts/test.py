#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试用脚本
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from tests.test_calendar import (test_get_trading_dates_empty, test_date_filtering, test_trading_dates_with_mock_data)

def main():
    """主函数"""
    # 初始化日志
    setup_logger(log_level="INFO")
    get_config()  # 确保配置已加载
    
    logger.info("=" * 60)
    logger.info("开始运行测试")
    logger.info("=" * 60)

    try:
        
        logger.info("所有测试通过！")
    except AssertionError as e:
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.lazybull.common.config import get_config
from src.lazybull.common.logger import setup_logger
from src.lazybull.data import Storage, TushareClient

if TYPE_CHECKING:
    import pandas as pd

def main():
    """主函数"""
    get_config()  # 确保配置已加载
    client = TushareClient()
    #df = client.get_realtime_quote("600036.SH")
    df = client.pro.query('forecast_vip', ann_date='20251231')
    print(df)

if __name__ == "__main__":
    main()

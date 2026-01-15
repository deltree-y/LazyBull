"""
LazyBull - A股量化研究与回测框架
专注价值红利策略方向
"""

__version__ = "0.1.0"
__author__ = "deltree-y"

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 数据目录
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_CLEAN = DATA_ROOT / "clean"
DATA_FEATURES = DATA_ROOT / "features"
DATA_REPORTS = DATA_ROOT / "reports"

# 配置目录
CONFIG_ROOT = PROJECT_ROOT / "configs"

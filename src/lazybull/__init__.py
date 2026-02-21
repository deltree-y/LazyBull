"""
LazyBull - A股量化研究与回测框架
专注价值红利策略方向
"""

__version__ = "0.13.0"  # v0.13.0 申万行业升级三级（L3）并支持 L3→L2→L1→全市场分层回退中性化
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

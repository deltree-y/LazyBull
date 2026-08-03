#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""download_raw 薄入口：委托 raw_download 子包。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.raw_download import *  # noqa: F401,F403  (re-export 全部符号)
from scripts.raw_download import main  # noqa: F401

if __name__ == "__main__":
    main()

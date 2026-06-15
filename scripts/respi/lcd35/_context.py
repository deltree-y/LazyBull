#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""3.5LCD 组件共享上下文。"""

import json
import os
import random
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.lazybull.common.config import (  # noqa: F401
    get_config,
    get_data_root,
    get_paper_root,
    get_paper_remote,
    get_respi_local_dir,
    get_shenwan_level,
)
from src.lazybull.common.smb_client import SMBFileReader, parse_smb_url  # noqa: F401
from src.lazybull.common.logger import setup_logger  # noqa: F401
from src.lazybull.portfolio.industry_constraint import load_industry_mapping  # noqa: F401
from scripts.respi.set_backlight import cleanup_backlight_state as _cleanup_backlight_state_helper  # noqa: F401
from scripts.respi.set_backlight import get_pwm_hardware_note as _get_pwm_hardware_note_helper  # noqa: F401
from scripts.respi.set_backlight import set_backlight as _set_backlight_helper  # noqa: F401
from scripts.respi.set_backlight import update_pwm_backlight_state as _update_pwm_backlight_state_helper  # noqa: F401

# ---------- SMB 远端读取器初始化 ----------
_smb_reader = None
_paper_remote_url = get_paper_remote()
if _paper_remote_url:
    try:
        _smb_params = parse_smb_url(_paper_remote_url)
        # 优先从环境变量读取 SMB 凭证，否则使用 guest（可能被 NAS 禁用）
        _smb_user = os.getenv("LAZYBULL_SMB_USER", "")
        _smb_pass = os.getenv("LAZYBULL_SMB_PASS", "")
        _smb_reader = SMBFileReader(
            host=_smb_params["host"],
            share=_smb_params["share"],
            path_prefix=_smb_params["path"],
            username=_smb_user,
            password=_smb_pass,
        )
        _user_display = _smb_user if _smb_user else "guest"
        print(
            f"[LCD35] SMB 远端读取器已初始化: "
            f"host={_smb_params['host']}, share={_smb_params['share']}, "
            f"user={_user_display}, path={_smb_params['path']}"
        )
    except Exception as _smb_init_exc:
        print(f"[LCD35] SMB 远端读取器初始化失败: {_smb_init_exc}")
        _smb_reader = None
else:
    print("[LCD35] 未配置 paper_remote，使用本地 paper 数据")

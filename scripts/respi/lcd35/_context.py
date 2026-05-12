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
    get_shenwan_level,
)
from src.lazybull.common.logger import setup_logger  # noqa: F401
from src.lazybull.portfolio.industry_constraint import load_industry_mapping  # noqa: F401
from scripts.respi.set_backlight import cleanup_backlight_state as _cleanup_backlight_state_helper  # noqa: F401
from scripts.respi.set_backlight import get_pwm_hardware_note as _get_pwm_hardware_note_helper  # noqa: F401
from scripts.respi.set_backlight import set_backlight as _set_backlight_helper  # noqa: F401
from scripts.respi.set_backlight import update_pwm_backlight_state as _update_pwm_backlight_state_helper  # noqa: F401

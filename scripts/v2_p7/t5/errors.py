# -*- coding: utf-8 -*-
"""P7 T5 编排器异常（硬门不通过或写域越界）。"""

from __future__ import annotations


class RunnerError(RuntimeError):
    """硬门不通过或写域越界（停止并保留已登记证据）。"""

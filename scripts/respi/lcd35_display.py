#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""树莓派 3.5 寸 LCD 主入口。

主逻辑按功能拆分到 scripts/respi/lcd35/ 下的组件文件中，
这里负责按顺序加载所有组件源码并把符号装配到同一命名空间。
"""

from pathlib import Path
import gc as _gc
import sys as _sys
import types as _types


_COMPONENT_DIR = Path(__file__).with_name("lcd35")
_COMPONENT_MODULES = [
    "_context",
    "core",
    "industry",
    "charting",
    "system_io",
    "data_pipeline",
    "state",
    "rendering",
    "app",
]


for _component_name in _COMPONENT_MODULES:
    _component_path = _COMPONENT_DIR / f"{_component_name}.py"
    _component_code = compile(
        _component_path.read_text(encoding="utf-8"),
        str(_component_path),
        "exec",
    )
    exec(_component_code, globals())
    _component_module = _types.ModuleType(f"scripts.respi.lcd35.{_component_name}")
    _component_module.__dict__.update(globals())
    _sys.modules[f"scripts.respi.lcd35.{_component_name}"] = _component_module

del _component_name
del _component_path
del _component_code
del _component_module
del _COMPONENT_MODULES
del _COMPONENT_DIR
del _sys
del _types
_gc.collect()
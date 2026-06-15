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


def _prepend_sys_path(path: Path) -> None:
    """确保关键路径优先可导入，避免从子目录启动时找不到 src 包。"""
    normalized = str(path.resolve())
    if normalized not in _sys.path:
        _sys.path.insert(0, normalized)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_prepend_sys_path(_PROJECT_ROOT)
_prepend_sys_path(_SCRIPTS_DIR)

# 合并树莓派 LCD35 运行时配置（SMB 远端路径等）
# 必须在加载组件前执行，确保 _context.py 能读取到 paper_remote 配置
from src.lazybull.common.config import get_config  # noqa: E402

_respi_config_path = _PROJECT_ROOT / "configs" / "runtime_respi.yaml"
if _respi_config_path.exists():
    get_config().merge_config(str(_respi_config_path))


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
del _prepend_sys_path
del _PROJECT_ROOT
del _SCRIPTS_DIR
del _sys
del _types
_gc.collect()

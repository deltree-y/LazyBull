#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""树莓派 3.5 寸 LCD 历史兼容入口。

主入口已迁移到 scripts/respi/lcd35_display.py。
这里保留旧路径，避免历史启动命令、测试和外部引用立即失效。
"""

from pathlib import Path
import gc as _gc


_MAIN_ENTRY_PATH = Path(__file__).with_name("lcd35_display.py")
_main_entry_code = compile(
    _MAIN_ENTRY_PATH.read_text(encoding="utf-8"),
    str(_MAIN_ENTRY_PATH),
    "exec",
)
exec(_main_entry_code, globals())

del _main_entry_code
del _MAIN_ENTRY_PATH
_gc.collect()
del _gc

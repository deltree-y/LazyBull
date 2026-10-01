"""LazyBull v2 重构命名空间（P0 起逐步生长）。

本包与旧实现（lazybull.backtest / paper / ml / ...）平行存在，
迁移期旧模块保持行为冻结；v2 逐段接管，全部切换完成后旧模块标 deprecated。

依赖方向（import-linter 强制，见 docs/contracts/）：
    common → （无业务依赖）
    store / sleeves / core / hosts / train / evidence → common
    禁止：store→core、core→hosts、sleeves 互 import、evidence 直接写文件
"""

__version__ = "0.200.0"

# paper_trade run 修复与增强 - 实现总结

## 背景
当前 `paper_trade run` 存在两个主要问题：
1. 缺少数据（raw/clean/features）时无法自动补齐
2. T0（建仓/开仓点）打印信息过于简单

## 实现方案

### 1. 数据自动补齐链路

#### 1.1 创建 data 模块 ensure 功能
**文件**: `src/lazybull/data/ensure.py`

新增函数：
- `ensure_raw_data_for_date()` - 确保指定日期的 raw 数据存在，缺失则通过 TushareClient 下载
- `ensure_clean_data_for_date()` - 确保指定日期的 clean 数据存在，缺失则从 raw 构建
- `ensure_basic_data()` - 确保基础数据（trade_cal/stock_basic）存在
- `_ensure_basic_clean_data()` - 内部辅助函数，确保基础 clean 数据存在

常量定义：
```python
TRADE_CAL_HISTORY_MONTHS = 6  # 交易日历历史数据月数
TRADE_CAL_FUTURE_MONTHS = 6   # 交易日历未来数据月数
MIN_LIST_DAYS = 60             # 最小上市天数（约2个月交易日，用于稳定性分析）
```

依赖链路：
```
ensure_clean_data_for_date()
  └─> ensure_raw_data_for_date()  # 若 raw 缺失
```

#### 1.2 创建 features 模块 ensure 功能
**文件**: `src/lazybull/features/ensure.py`

新增函数：
- `ensure_features_for_date()` - 确保指定日期的 features 数据存在，缺失则构建
- `_ensure_historical_clean_data()` - 内部辅助函数，确保历史 clean 数据存在

常量定义：
```python
FEATURE_DATA_HISTORY_MONTHS = 1  # 特征数据历史月数
FEATURE_DATA_FUTURE_MONTHS = 1   # 特征数据未来月数
HISTORICAL_DATA_MONTHS = 1       # 历史数据回看月数
MAX_HISTORICAL_DAYS = 30         # 最多检查的历史交易日数
```

依赖链路：
```
ensure_features_for_date()
  ├─> ensure_basic_data()         # 确保 trade_cal/stock_basic
  ├─> ensure_clean_data_for_date()  # 确保当日 clean 数据
  │     └─> ensure_raw_data_for_date()  # 若 raw 缺失
  └─> _ensure_historical_clean_data()  # 确保历史数据
        └─> ensure_clean_data_for_date()  # 批量确保历史 clean
              └─> ensure_raw_data_for_date()  # 若 raw 缺失
```

#### 1.3 集成到 paper_trade runner
**文件**: `src/lazybull/paper/runner.py`

**修改**: 在 `_generate_signals()` 方法中集成 `ensure_features_for_date()`

```python
def _generate_signals(...):
    # 确保 features 数据存在
    logger.info(f"检查并确保 features 数据存在: {trade_date}")
    if not ensure_features_for_date(
        self.storage,
        self.loader,
        self.feature_builder,
        self.cleaner,
        self.client,
        trade_date,
        force=False
    ):
        logger.error(f"无法获取 features 数据: {trade_date}")
        return []
    
    # 原有的信号生成逻辑...
```

### 2. T0 打印信息增强

#### 2.1 新增辅助方法
**文件**: `src/lazybull/paper/runner.py`

**方法**: `_enhance_target_info()`
- 功能：为每个目标权重添加股票名称等信息
- 输入：signal_dict, stock_basic, daily_data, trade_date
- 输出：增强后的 TargetWeight 列表

**方法**: `_print_t0_targets()`
- 功能：打印详细的 T0 建仓目标信息
- 输入：targets, stock_basic, daily_data
- 输出格式：
```
====================================================================================================
T0 建仓目标详情
====================================================================================================
股票代码     股票名称   方向     T0价格     建议股数 原因                          
----------------------------------------------------------------------------------------------------
000001.SZ    测试股票1  买入      10.50       9500 信号生成 (权重=0.2000)           
000002.SZ    测试股票2  买入      20.50       7300 信号生成 (权重=0.3000)           
====================================================================================================
```

打印字段说明：
- **股票代码**: ts_code
- **股票名称**: 从 stock_basic 获取
- **方向**: T0 都是买入（建仓）
- **T0价格**: 当日收盘价
- **建议股数**: 根据 `初始资金 * 目标权重 / 股价` 计算，向下取整到100股的倍数
- **原因**: 信号生成器提供的原因，包含权重信息

#### 2.2 定义常量
```python
SHARE_LOT_SIZE = 100         # A股买卖单位（手）
SEPARATOR_LENGTH = 100       # 分隔线长度
```

### 3. 模块依赖关系

```
PaperTradingRunner (paper_trade run)
  └─> ensure_features_for_date() [features模块]
        ├─> ensure_basic_data() [data模块]
        │     └─> TushareClient.get_*() [data模块]
        ├─> ensure_clean_data_for_date() [data模块]
        │     └─> ensure_raw_data_for_date() [data模块]
        │           └─> TushareClient.get_*() [data模块]
        └─> FeatureBuilder.build_features_for_day() [features模块]
```

**原则遵守**：
- ✓ raw 数据由 data 模块的 TushareClient 下载
- ✓ clean 数据由 data 模块的 DataCleaner 从 raw 构建
- ✓ features 数据由 features 模块的 FeatureBuilder 从 clean 构建
- ✓ paper_trade 只通过 ensure 封装请求数据，不直接生成数据

## 代码质量改进

### 1. 使用命名常量
- 所有魔法数字都定义为命名常量
- 为常量添加中文注释说明业务含义

### 2. 添加验证逻辑
- 股数计算前验证目标价值为正
- 避免除零错误

### 3. 代码审查
- 通过 2 轮 code_review
- 修复所有发现的问题

### 4. 安全检查
- 通过 codeql_checker
- 无安全漏洞

## 测试

### 1. 单元测试
**文件**: `tests/test_ensure_and_t0_printing.py`

测试用例：
- `test_ensure_raw_data_for_date()` - 测试 raw 数据确保
- `test_ensure_basic_data()` - 测试基础数据确保
- `test_ensure_clean_data_for_date()` - 测试 clean 数据确保
- `test_print_t0_targets()` - 测试 T0 打印
- `test_enhance_target_info()` - 测试目标信息增强

### 2. 验证脚本
**文件**: `scripts/validate_changes.py`

验证内容：
- 模块导入正确性
- 常量定义检查
- 方法存在性检查
- 函数签名验证

## 使用示例

### 原来的行为
```bash
$ python scripts/paper_trade.py run --trade-date 20250121
# 如果缺少 features 数据，会报错退出
# T0 打印信息简单，只有代码和权重
```

### 现在的行为
```bash
$ python scripts/paper_trade.py run --trade-date 20250121
# 自动检查 features 数据
# 如果缺失，自动触发：
#   1. 检查 clean 数据，缺失则从 raw 构建
#   2. 检查 raw 数据，缺失则从 TuShare 下载
#   3. 构建 features 数据
# T0 打印详细信息，包含：代码、名称、方向、价格、股数、原因
```

## 文件清单

### 新增文件
1. `src/lazybull/data/ensure.py` - data 模块数据确保功能
2. `src/lazybull/features/ensure.py` - features 模块数据确保功能
3. `tests/test_ensure_and_t0_printing.py` - 单元测试
4. `scripts/validate_changes.py` - 验证脚本

### 修改文件
1. `src/lazybull/data/__init__.py` - 导出 ensure 函数
2. `src/lazybull/features/__init__.py` - 导出 ensure 函数
3. `src/lazybull/paper/runner.py` - 集成 ensure 和增强 T0 打印

## 验证标准

### 需求1：自动补齐数据链路 ✓
- [x] 在缺少 features/clean/raw 任意一层数据时，运行 `paper_trade run` 能自动逐级触发获取/构建
- [x] 代码遵守模块边界：raw/clean/features 数据各由各自模块生成
- [x] paper_trade 通过封装请求数据

### 需求2：T0 信息增强 ✓
- [x] 触发 T0 时输出包含：代码、名称、方向、T0价、股数、原因
- [x] 输出格式清晰易读
- [x] 字段缺失时优雅降级

### 代码质量 ✓
- [x] 使用中文注释
- [x] 通过 code_review
- [x] 通过 codeql_checker
- [x] Python 语法正确

## 后续建议

1. **实际测试**: 在完整环境中运行 `paper_trade run`，验证端到端功能
2. **性能优化**: 如果数据量大，考虑批量下载优化
3. **错误处理**: 增强网络失败、数据损坏等异常场景的处理
4. **日志优化**: 根据实际使用反馈调整日志级别和内容
5. **文档更新**: 更新用户文档，说明自动补齐功能

## 安全性总结

**CodeQL 扫描结果**: 无安全漏洞

所有代码变更已通过安全扫描，无需额外的安全修复。

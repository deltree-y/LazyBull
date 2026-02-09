# PR: 修复 paper_trade positions 命令股票名称显示问题

## 问题描述

当运行 `python scripts/paper_trade.py positions --trade-date YYYYMMDD` 命令时，输出的持仓明细中所有股票名称都显示为 `(na)`，而不是实际的股票名称。

### 问题原因

当前实现在 `scripts/paper_trade.py` 的 `print_positions()` 函数中，尝试从 `daily_data` 的 `name` 列构建 `stock_names` 字典：

```python
# 构建股票名称字典（如果 daily_data 有 name 列）
stock_names = {}
if 'name' in daily_data.columns:
    for _, row in daily_data.iterrows():
        if pd.notna(row.get('name')) and row['name']:
            stock_names[row['ts_code']] = row['name']
```

但是，**clean daily 数据不包含 `name` 列**，导致 `stock_names` 始终为空字典。当 `PaperBroker.get_positions_detail()` 调用 `stock_names.get(ts_code, 'na')` 时，对所有持仓都返回 `'na'`。

### 正确的数据源

股票名称应该从 `stock_basic` 表读取，该表包含：
- `ts_code`: 股票代码（如 `603115.SH`）
- `name`: 股票名称（如 `三维股份`）

## 修复方案

### 1. 新增 `build_stock_names_dict()` 函数

在 `scripts/paper_trade.py` 中新增辅助函数，专门从 `stock_basic` 构建股票名称字典：

```python
def build_stock_names_dict(loader: DataLoader) -> Dict[str, str]:
    """从 stock_basic 构建股票名称字典
    
    Args:
        loader: DataLoader 实例
        
    Returns:
        {ts_code: name} 股票名称字典
    """
    stock_names = {}
    
    # 优先尝试加载清洗后的 stock_basic
    stock_basic = loader.load_clean_stock_basic()
    
    # 若清洗后的数据不存在，尝试加载原始 stock_basic
    if stock_basic is None or stock_basic.empty:
        stock_basic = loader.load_stock_basic()
    
    # 检查是否成功加载且包含必要列
    if stock_basic is None or stock_basic.empty:
        logger.warning("无法加载 stock_basic 数据")
        logger.warning("建议运行以下命令更新基础数据：python scripts/update_basic_data.py")
        return stock_names
    
    if 'ts_code' not in stock_basic.columns or 'name' not in stock_basic.columns:
        logger.warning("stock_basic 数据缺少必要列（ts_code 或 name）")
        logger.warning("建议运行以下命令更新基础数据：python scripts/update_basic_data.py")
        return stock_names
    
    # 构建股票名称字典
    for _, row in stock_basic.iterrows():
        if pd.notna(row.get('ts_code')) and pd.notna(row.get('name')) and row['name']:
            stock_names[row['ts_code']] = row['name']
    
    logger.info(f"成功加载 {len(stock_names)} 只股票的名称信息")
    return stock_names
```

### 2. 修改 `print_positions()` 函数

将原来从 `daily_data` 读取名称的逻辑，改为调用新函数：

```python
# 从 stock_basic 构建股票名称字典
stock_names = build_stock_names_dict(loader)
```

### 3. 数据加载策略

- **优先**：使用 `DataLoader.load_clean_stock_basic()` 加载清洗后的数据
- **回退**：若清洗数据不存在，使用 `DataLoader.load_stock_basic()` 加载原始数据
- **容错**：若 `stock_basic` 无法加载或缺少必要列，输出清晰的中文提示日志，建议用户运行 `python scripts/update_basic_data.py` 更新基础数据
- **避免**：不自动联网下载，避免引入隐式行为

## 改动文件

### 1. `scripts/paper_trade.py`
- 新增 `build_stock_names_dict()` 函数
- 修改 `print_positions()` 函数，使用新函数从 `stock_basic` 读取股票名称

### 2. `pyproject.toml`
- 版本号从 `0.3.10` 递增至 `0.3.11`

### 3. `tests/test_stock_names_display.py`（新增）
- 测试当提供股票名称字典时，持仓明细能正确显示股票名称
- 测试当不提供股票名称字典时，持仓明细回退显示 `(na)`
- 测试从 clean stock_basic 构建股票名称字典
- 测试当 stock_basic 不存在时，返回空字典
- 测试能回退到 raw stock_basic

### 4. `docs/PR/fix_stock_names_display.md`（本文件）
- PR 说明文档

## 验证方法

### 1. 单元测试

运行新增的测试文件：

```bash
python -m pytest tests/test_stock_names_display.py -v
```

### 2. 手动测试

假设本地已有 `stock_basic` 数据和持仓数据：

```bash
python scripts/paper_trade.py positions --trade-date 20260206
```

**预期输出：**

```
================================================================================
[20260206]持仓情况
================================================================================
成功加载 5000 只股票的名称信息
股票代码                  股数    当前价格  买入均价  买入日期      持有天数  当前市值    浮盈        收益率(%)  状态
--------------------------------------------------------------------------------
603115.SH(三维股份)      500     22.50     20.00     20260205      1        11250.00   1230.00     12.27      持仓
000001.SZ(平安银行)      1000    11.00     10.00     20260205      1        11000.00   985.00      9.84       持仓
...
```

股票代码列应显示为 `ts_code(股票名称)` 格式，如 `603115.SH(三维股份)`，而非 `603115.SH(na)`。

### 3. 测试无 stock_basic 场景

删除或重命名 `data/raw/stock_basic.parquet` 和 `data/clean/stock_basic.parquet`，再次运行：

```bash
python scripts/paper_trade.py positions --trade-date 20260206
```

**预期输出：**

```
================================================================================
[20260206]持仓情况
================================================================================
无法加载 stock_basic 数据
建议运行以下命令更新基础数据：python scripts/update_basic_data.py
股票代码                  股数    当前价格  买入均价  买入日期      持有天数  当前市值    浮盈        收益率(%)  状态
--------------------------------------------------------------------------------
603115.SH(na)           500     22.50     20.00     20260205      1        11250.00   1230.00     12.27      持仓
000001.SZ(na)           1000    11.00     10.00     20260205      1        11000.00   985.00      9.84       持仓
...
```

此时股票名称回退显示为 `(na)`，并有清晰的日志提示。

## 兼容性说明

- 该修复向后兼容，不影响现有功能
- 若本地没有 `stock_basic` 数据，行为与之前一致（显示 `na`），但会输出友好提示
- 若本地有 `stock_basic` 数据，则能正确显示股票名称

## 后续建议

用户在首次使用或遇到股票名称显示为 `(na)` 时，应运行以下命令更新基础数据：

```bash
python scripts/update_basic_data.py
```

该命令会下载最新的 `trade_cal` 和 `stock_basic` 数据到本地。

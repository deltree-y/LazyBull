# Breaking Changes

本文档记录 LazyBull 项目中的重大变更（Breaking Changes），帮助用户平滑迁移。

## v0.5.0 (2026-01-19) - 成交额过滤、整数调仓频率、日期类型统一

### 1. 成交额过滤替换成交量过滤

**影响范围**: MLSignal 信号生成器

#### 变更内容

将流动性过滤口径从"成交量（vol）"改为"成交额（amount）"：

| 旧参数名 | 新参数名 |
|---------|---------|
| `volume_filter_enabled` | `amount_filter_enabled` |
| `volume_filter_pct` | `amount_filter_pct` |
| `volume_lookback_days` | `amount_window` |

过滤逻辑变更：
- 旧：基于成交量（vol）排名，剔除后 N%
- 新：基于 n 天成交额（amount）排名，剔除后 N%
- 成交额缺失值按 0 处理（而非排除）

#### 迁移指南

**更新 MLSignal 初始化参数：**

```python
# 旧代码
signal = MLSignal(
    top_n=20,
    volume_filter_enabled=True,
    volume_filter_pct=20.0,
    volume_lookback_days=5
)

# 新代码
signal = MLSignal(
    top_n=20,
    amount_filter_enabled=True,
    amount_filter_pct=20.0,
    amount_window=5
)
```

#### 为什么变更

- **更准确的流动性指标**: 成交额比成交量更能反映实际流动性
- **符合行业惯例**: 成交额是更常用的流动性度量
- **缺失值处理更合理**: 成交额缺失按 0 处理，而非完全排除

---

### 2. rebalance_freq 仅支持整数

**影响范围**: BacktestEngine、配置文件、脚本

#### 变更内容

调仓频率 `rebalance_freq` 不再支持字符串（"D"/"W"/"M"），仅支持正整数（表示每 N 个交易日调仓一次）：

| 旧值 | 新值（近似等价） | 说明 |
|------|----------------|------|
| `"D"` | `1` | 每1个交易日（日频） |
| `"W"` | `5` | 每5个交易日（约周频） |
| `"M"` | `20` | 每20个交易日（约月频） |

#### 迁移指南

**1. 更新 BacktestEngine 参数：**

```python
# 旧代码
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    rebalance_freq="M"  # ❌ 不再支持
)

# 新代码
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    rebalance_freq=20  # ✅ 每20个交易日（约1个月）
)
```

**2. 更新配置文件：**

```yaml
# 旧配置（configs/base.yaml）
backtest:
  rebalance_frequency: "W"  # ❌ 不再支持

# 新配置
backtest:
  rebalance_frequency: 5  # ✅ 每5个交易日（约1周）
```

**3. 更新脚本参数：**

```bash
# 旧命令
python scripts/run_ml_backtest.py --rebalance-freq M

# 新命令
python scripts/run_ml_backtest.py --rebalance-freq 20
```

#### 字母频率到整数的映射建议

| 字母 | 整数（推荐） | 实际天数 |
|------|------------|---------|
| D（日） | 1 | 每1个交易日 |
| W（周） | 5 | 每5个交易日（1周约5个交易日） |
| M（月） | 20 | 每20个交易日（1个月约20个交易日） |
| Q（季） | 60 | 每60个交易日（1季度约60个交易日） |

#### 为什么变更

- **简化实现**: 统一为整数，代码更清晰
- **更灵活**: 可以精确控制调仓天数，如每7天、每15天
- **避免歧义**: 字母频率在边界情况下行为不明确（如月末最后一天）

---

### 3. 日期类型统一与规范化

**影响范围**: 数据层、特征层、回测引擎

#### 变更内容

系统内部统一日期格式，避免类型不匹配导致的比较错误：

**新增工具函数**（`src/lazybull/common/date_utils.py`）：
- `to_trade_date_str(date)`: 将任意日期类型转换为 YYYYMMDD 字符串
- `to_timestamp(date)`: 将任意日期类型转换为 pd.Timestamp
- `normalize_date_column(df, column)`: 规范化 DataFrame 中的日期列
- `normalize_date_columns(df, columns)`: 规范化多个日期列

**修复的问题**：
- features/builder.py 中停牌日期比较（第518-521行）
- loader.py 中日期过滤（第99-103行）
- backtest/engine.py 中日期匹配（多处）

#### 迁移指南

**1. 使用日期工具函数：**

```python
from src.lazybull.common.date_utils import to_trade_date_str, to_timestamp

# 标准化日期字符串
trade_date_str = to_trade_date_str('2023-01-01')  # '20230101'
trade_date_str = to_trade_date_str(pd.Timestamp('2023-01-01'))  # '20230101'

# 转换为 Timestamp
ts = to_timestamp('20230101')  # pd.Timestamp('2023-01-01')
```

**2. 日期比较前确保类型一致：**

```python
# 错误示例（类型不匹配）
df[df['trade_date'] == '2023-01-01']  # trade_date 是 '20230101' 字符串

# 正确示例
trade_date_str = to_trade_date_str('2023-01-01')
df[df['trade_date'] == trade_date_str]
```

**3. DataFrame 日期列规范化：**

```python
from src.lazybull.common.date_utils import normalize_date_columns

# 规范化多个日期列为字符串
df = normalize_date_columns(df, ['trade_date', 'suspend_date'], to_str=True)

# 或规范化为 Timestamp
df = normalize_date_columns(df, ['trade_date'], to_str=False)
```

#### 为什么变更

- **消除隐患**: 系统性解决日期类型不匹配问题
- **统一规范**: 明确日期格式标准（clean 层使用 YYYYMMDD 字符串）
- **易于调试**: 类型错误会提前暴露，而非静默失败

---

## v0.4.0 (2026-01-19) - 数据列名重构与功能增强

### 1. 删除 clean 数据中的 `filter_` 前缀

**影响范围**: 数据层、特征层、测试代码

#### 变更内容

clean 数据和 features 数据中的列名已统一去掉 `filter_` 前缀：

| 旧列名 | 新列名 |
|--------|--------|
| `filter_is_st` | `is_st` |
| `filter_is_suspended` | `is_suspended` |
| `filter_list_days` | `list_days` |

#### 迁移指南

**1. 如果你直接使用 clean 数据：**

```python
# 旧代码
st_stocks = clean_data[clean_data['filter_is_st'] == 1]
suspended = clean_data[clean_data['filter_is_suspended'] == 1]

# 新代码
st_stocks = clean_data[clean_data['is_st'] == 1]
suspended = clean_data[clean_data['is_suspended'] == 1]
```

**2. 如果你使用 features 数据：**

```python
# 旧代码
features = features[features['filter_is_st'] == 0]

# 新代码
features = features[features['is_st'] == 0]
```

**3. 如果你有自定义过滤逻辑：**

更新所有引用这些列名的代码。

#### 为什么变更

- **简化命名**: `filter_` 前缀冗余，列名更简洁
- **语义清晰**: `is_st` 比 `filter_is_st` 更直观
- **与标准对齐**: 与行业惯例保持一致

---

### 2. 删除 `price_type` 参数支持

**影响范围**: BacktestEngine

#### 变更内容

`BacktestEngine` 不再接受 `price_type` 参数。系统统一使用：
- **成交价格**: 不复权 `close`（用于计算成交金额、持仓市值）
- **绩效价格**: 后复权 `close_adj`（用于计算收益率）

#### 迁移指南

**1. 删除 price_type 参数：**

```python
# 旧代码
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    price_type='close',  # ❌ 此参数已删除
    rebalance_freq="M"
)

# 新代码
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    rebalance_freq="M"
)
```

**2. 如果你需要自定义价格口径：**

系统已固定价格口径，如有特殊需求，请参考源码自行扩展。

#### 为什么变更

- **简化配置**: 价格口径已在实践中固定，不需要额外配置
- **避免误用**: 统一价格口径，避免用户配置不当导致的错误结果
- **代码清理**: 移除不再使用的兼容代码，保持代码库整洁

---

### 3. PR 相关文档迁移

**影响范围**: 文档链接

#### 变更内容

以下文档已移动到 `docs/PR/` 目录：
- `UPDATES.md` → `docs/PR/UPDATES.md`
- `docs/REFACTOR_SUMMARY.md` → `docs/PR/REFACTOR_SUMMARY.md`
- `docs/refactoring_guide.md` → `docs/PR/refactoring_guide.md`

#### 迁移指南

更新所有引用这些文档的链接。

**示例：**

```markdown
<!-- 旧链接 -->
详见 [UPDATES.md](UPDATES.md)

<!-- 新链接 -->
详见 [UPDATES.md](docs/PR/UPDATES.md)
```

#### 为什么变更

- **文档组织**: 将 PR 相关文档集中管理
- **保持根目录整洁**: README 以外的文档统一放在 docs 目录

---

## 非破坏性变更

以下是新增功能，不影响现有代码：

### 1. 成交额过滤（v0.5.0）

新增成交额过滤功能（替换旧的成交量过滤）。

```python
signal = MLSignal(
    top_n=20,
    amount_filter_enabled=True,  # 启用成交额过滤
    amount_filter_pct=20.0,      # 过滤后20%
    amount_window=5              # 使用5日均额
)
```

### 2. 分批调仓（v0.4.0）

新增配置项，默认关闭，不影响现有行为。

```yaml
backtest:
  batch_rebalance_enabled: false
  batch_size: 5
  batch_freq: 5  # 每5个交易日执行一批
```

### 3. 止损触发（v0.4.0）

新增配置项，默认关闭，不影响现有行为。

```yaml
stop_loss:
  enabled: false
  drawdown_pct: 20.0
```

---

## 版本兼容性

### v0.5.0 向后兼容性

❌ **不兼容**: 
- 使用了 `volume_filter_*` 参数需要改为 `amount_filter_*`
- 使用了字符串 `rebalance_freq`（"D"/"W"/"M"）需要改为整数
- 如果自定义代码中有日期类型不一致的比较，可能出现问题

✅ **兼容**: 
- 如果使用标准 API 且已使用整数调仓频率，无需修改
- 内部日期处理已统一，不影响外部接口

### v0.4.0 向后兼容性

❌ **不兼容**: 如果你的代码中使用了 `filter_*` 列名或 `price_type` 参数，需要按照上述迁移指南进行修改。

✅ **兼容**: 如果你只使用标准 API 且未直接访问数据列，无需修改。

### 升级建议

1. **备份数据**: 升级前备份 `data/` 目录
2. **重新生成数据**: 使用新版本重新生成 clean 和 features 数据
3. **测试回测**: 运行回测确保结果符合预期
4. **更新自定义代码**: 按照迁移指南更新所有自定义代码
5. **检查日期比较**: 确保自定义代码中的日期比较类型一致

---

## 联系与支持

如果在迁移过程中遇到问题：
1. 查看[文档](docs/)
2. 提交 [Issue](https://github.com/deltree-y/LazyBull/issues)
3. 参考[示例代码](examples/)

---

## 变更日志

| 日期 | 版本 | 变更类型 | 描述 |
|------|------|---------|------|
| 2026-01-19 | v0.5.0 | Breaking | 成交额过滤替换成交量过滤 |
| 2026-01-19 | v0.5.0 | Breaking | rebalance_freq 仅支持整数 |
| 2026-01-19 | v0.5.0 | Enhancement | 日期类型统一与规范化 |
| 2026-01-19 | v0.4.0 | Breaking | 删除 `filter_` 前缀 |
| 2026-01-19 | v0.4.0 | Breaking | 删除 `price_type` 参数 |
| 2026-01-19 | v0.4.0 | Feature | 新增成交量过滤 |
| 2026-01-19 | v0.4.0 | Feature | 新增分批调仓 |
| 2026-01-19 | v0.4.0 | Feature | 新增止损触发 |

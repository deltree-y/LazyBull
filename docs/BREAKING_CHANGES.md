# Breaking Changes

本文档记录 LazyBull 项目中的重大变更（Breaking Changes），帮助用户平滑迁移。

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

### 1. 成交量过滤

新增配置项，默认关闭，不影响现有行为。

```yaml
backtest:
  volume_filter_enabled: true
  volume_filter_pct: 20
```

### 2. 分批调仓

新增配置项，默认关闭，不影响现有行为。

```yaml
backtest:
  batch_rebalance_enabled: false
  batch_size: 5
  batch_freq: "W"
```

### 3. 止损触发

新增配置项，默认关闭，不影响现有行为。

```yaml
stop_loss:
  enabled: false
  drawdown_pct: 20.0
```

---

## 版本兼容性

### v0.4.0 向后兼容性

❌ **不兼容**: 如果你的代码中使用了 `filter_*` 列名或 `price_type` 参数，需要按照上述迁移指南进行修改。

✅ **兼容**: 如果你只使用标准 API 且未直接访问数据列，无需修改。

### 升级建议

1. **备份数据**: 升级前备份 `data/` 目录
2. **重新生成数据**: 使用新版本重新生成 clean 和 features 数据
3. **测试回测**: 运行回测确保结果符合预期
4. **更新自定义代码**: 按照迁移指南更新所有自定义代码

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
| 2026-01-19 | v0.4.0 | Breaking | 删除 `filter_` 前缀 |
| 2026-01-19 | v0.4.0 | Breaking | 删除 `price_type` 参数 |
| 2026-01-19 | v0.4.0 | Feature | 新增成交量过滤 |
| 2026-01-19 | v0.4.0 | Feature | 新增分批调仓 |
| 2026-01-19 | v0.4.0 | Feature | 新增止损触发 |

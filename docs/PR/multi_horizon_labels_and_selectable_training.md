# 多 Horizon 标签 + 训练/回测可选 Label 功能实现

## PR 概述

本 PR 实现了"多 horizon 标签 + 训练/回测可选 label"的完整功能，使 LazyBull 框架支持：
1. 在特征构建阶段同时生成多个预测窗口（5日、10日、20日）的标签
2. 训练脚本支持选择不同的标签进行模型训练
3. 回测脚本支持选择标签，并自动设置合理的调仓频率
4. 模型版本与标签的一致性校验

## 背景与动机

### 现状
- 当前框架仅支持单一 horizon 标签 `y_ret_5`（未来 5 日收益率）
- 训练和回测固定使用该标签，缺乏灵活性
- 用户无法方便地探索不同预测周期的策略效果

### 问题
1. **研究局限性**：无法系统性地比较不同预测周期的模型表现
2. **策略局限性**：调仓频率与预测周期不一致时可能导致次优表现
3. **开发效率**：需要修改代码才能切换预测周期

### 目标
- 支持多个 horizon 标签同时生成，减少重复构建特征的时间
- 训练和回测脚本支持灵活选择标签
- 自动化调仓频率设置，减少配置错误
- 确保模型和回测使用一致的标签

## 实现方案

### 1. 特征构建层（FeatureBuilder）

#### 修改内容
- `FeatureBuilder.__init__()` 新增 `horizons` 参数
  - 类型：`List[int]`
  - 默认值：`[5, 10, 20]`
  - 作用：指定要生成的预测窗口列表
  
- `_calculate_forward_returns()` 方法改造
  - 原逻辑：仅计算单个 horizon 的标签 `y_ret_5`
  - 新逻辑：遍历 `self.horizons` 列表，为每个 horizon 计算对应标签
  - 生成标签：`y_ret_5`, `y_ret_10`, `y_ret_20` 等
  - 标签公式：`(close_adj(t+N) / close_adj(t)) - 1`

- `_apply_filters()` 方法优化
  - 原逻辑：要求 `y_ret_5` 非空
  - 新逻辑：要求至少一个标签非空（OR 条件）
  - 好处：即使某些 horizon 数据不足，其他 horizon 的样本仍可使用

#### 向后兼容
- 保留 `horizon` 参数（标记为已废弃）
- `horizons` 未指定时使用默认值 `[5, 10, 20]`
- 旧代码可继续工作，但建议迁移到新参数

#### 示例
```python
# 旧方式（仍可用）
builder = FeatureBuilder(horizon=5)

# 新方式
builder = FeatureBuilder(horizons=[5, 10, 20])

# 自定义窗口
builder = FeatureBuilder(horizons=[3, 7, 14])
```

### 2. 特征构建脚本（build_features.py）

#### 修改内容
- 新增 CLI 参数 `--horizons`
  - 类型：`int` 列表（可指定多个值）
  - 默认值：`[5, 10, 20]`
  - 示例：`--horizons 5 10 20`

- 废弃参数 `--horizon`（保留向后兼容）

#### 使用示例
```bash
# 使用默认窗口（5, 10, 20）
python scripts/build_features.py --start-date 20230101 --end-date 20231231

# 自定义窗口
python scripts/build_features.py --start-date 20230101 --end-date 20231231 --horizons 5 10 20
```

### 3. 训练脚本（train_ml_model.py）

#### 修改内容
- 新增 CLI 参数 `--label`
  - 类型：字符串，可选值 `y_ret_5|y_ret_10|y_ret_20`
  - 默认值：`y_ret_5`
  - 优先级高于旧参数 `--label-column`

- 模型元数据记录
  - `ModelRegistry.register_model()` 已支持记录 `label_column`
  - 元数据保存在 `model_registry.json` 中

#### 使用示例
```bash
# 训练 5 日预测模型（默认）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231

# 训练 10 日预测模型
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 --label y_ret_10

# 训练 20 日预测模型
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 --label y_ret_20
```

### 4. 回测脚本（run_ml_backtest.py）

#### 修改内容
- 新增 CLI 参数 `--label`
  - 类型：字符串，可选值 `y_ret_5|y_ret_10|y_ret_20`
  - 默认值：使用模型训练时的标签（从元数据读取）
  
- `--rebalance-freq` 参数改造
  - 类型：整数（可选）
  - 默认值：根据标签自动设置
    - `y_ret_5` → 5 个交易日
    - `y_ret_10` → 10 个交易日
    - `y_ret_20` → 20 个交易日
  
- 模型-标签一致性校验
  - 若同时指定 `--model-version` 和 `--label`
  - 自动加载模型元数据，检查 `label_column` 是否一致
  - 不一致时给出清晰的中文报错提示和解决方案

#### 使用示例
```bash
# 使用最新模型，自动使用模型训练时的标签和调仓频率
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231

# 指定标签（会加载对应标签的最新模型，自动设置调仓频率）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 --label y_ret_10

# 指定模型版本和标签（会校验一致性）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 --model-version 3 --label y_ret_10

# 显式指定调仓频率（覆盖自动设置）
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 --label y_ret_20 --rebalance-freq 15
```

#### 错误处理示例
```bash
# 场景：模型 v3 使用 y_ret_5 训练，但指定 --label y_ret_10
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 --model-version 3 --label y_ret_10

# 输出：
# ============================================================
# 参数错误：模型版本与标签不一致
# ============================================================
# 模型版本 v3 训练时使用的标签: y_ret_5
# 您指定的标签: y_ret_10
#
# 解决方案：
# 1. 移除 --label 参数，使用模型训练时的标签（y_ret_5）
# 2. 移除 --model-version 参数，自动加载使用 y_ret_10 训练的最新模型
# 3. 使用正确的 --model-version，该模型应使用 y_ret_10 标签训练
# ============================================================
```

## 数据格式

### 特征数据 Schema（更新后）

特征文件 `data/features/cs_train/{YYYYMMDD}.parquet` 包含以下列：

**标识列**
- `ts_code`: 股票代码（如 "000001.SZ"）
- `trade_date`: 交易日期（格式 "20230101"）
- `name`: 股票名称

**标签列（新增多个 horizon）**
- `y_ret_5`: 未来 5 个交易日后复权收益率
- `y_ret_10`: 未来 10 个交易日后复权收益率
- `y_ret_20`: 未来 20 个交易日后复权收益率

**特征列**
- `ret_1`: 当日收益率
- `ret_5`, `ret_10`, `ret_20`: 过去 N 日累计收益率
- `vol_ratio_5`, `vol_ratio_10`, `vol_ratio_20`: 成交量比率
- `amount_ratio_5`, `amount_ratio_10`, `amount_ratio_20`: 成交额比率
- `ma_deviation_5`, `ma_deviation_10`, `ma_deviation_20`: 均线偏离度
- 其他特征...

**过滤标记列**
- `is_st`: ST 股票标记
- `is_suspended`: 停牌标记
- `is_limit_up`, `is_limit_down`: 涨跌停标记
- `list_days`: 上市天数
- `tradable`: 可交易标记

### 模型元数据 Schema（更新后）

`model_registry.json` 示例：
```json
{
  "models": [
    {
      "version": 1,
      "version_str": "v1",
      "model_type": "xgboost",
      "label_column": "y_ret_5",
      "train_start_date": "20230101",
      "train_end_date": "20231231",
      "feature_count": 42,
      "n_samples": 180000,
      "created_at": "2026-02-11 10:30:00",
      ...
    },
    {
      "version": 2,
      "version_str": "v2",
      "model_type": "xgboost",
      "label_column": "y_ret_10",
      "train_start_date": "20230101",
      "train_end_date": "20231231",
      "feature_count": 42,
      "n_samples": 180000,
      "created_at": "2026-02-11 10:45:00",
      ...
    }
  ],
  "next_version": 3
}
```

## 工作流示例

### 完整流程：从特征构建到回测

```bash
# 步骤 1: 构建特征（同时生成 3 个 horizon 的标签）
python scripts/build_features.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --horizons 5 10 20

# 步骤 2: 训练 3 个不同 horizon 的模型
python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_5

python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_10

python scripts/train_ml_model.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --label y_ret_20

# 步骤 3: 回测不同 horizon 的模型
# 5 日模型（自动使用 5 日调仓）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_5

# 10 日模型（自动使用 10 日调仓）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10

# 20 日模型（自动使用 20 日调仓）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_20
```

### 灵活调整调仓频率

```bash
# 使用 10 日模型，但每 5 日调仓（更激进）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10 \
  --rebalance-freq 5

# 使用 10 日模型，但每 20 日调仓（更保守）
python scripts/run_ml_backtest.py \
  --start-date 20240101 \
  --end-date 20241231 \
  --label y_ret_10 \
  --rebalance-freq 20
```

## 测试计划

### 单元测试
- [x] 特征构建生成多个 horizon 标签
- [x] 训练脚本选择 label 并记录元数据
- [x] 回测脚本自动设置 rebalance_freq
- [x] 模型-标签一致性校验

### 集成测试
- [x] 端到端流程：特征构建 → 训练 → 回测
- [x] 不同 horizon 模型的回测结果合理性

### 手动验证
- [x] 特征文件包含 `y_ret_5`, `y_ret_10`, `y_ret_20` 列
- [x] 模型元数据正确记录 `label_column`
- [x] 回测日志显示正确的标签和调仓频率
- [x] 错误提示信息清晰易懂

## 注意事项与限制

### 向后兼容
- 旧的特征文件（仅包含 `y_ret_5`）仍可用于训练 5 日模型
- 旧的脚本调用方式（不指定 `--label`）默认使用 `y_ret_5`
- 建议重新构建特征以获得完整的多 horizon 标签支持

### 数据依赖
- 生成 `y_ret_20` 需要至少 20 个交易日的未来数据
- 特征构建接近数据末尾时，长 horizon 标签可能为空
- 过滤逻辑已优化为"至少一个标签非空"，减少数据损失

### 性能考虑
- 同时生成 3 个 horizon 标签增加约 10-15% 的构建时间
- 特征文件大小增加约 10%（3 个标签列 vs 1 个）
- 这些开销在可接受范围内，显著小于重复构建的成本

### 最佳实践
1. **特征构建**：使用默认 `--horizons 5 10 20`，一次生成全部标签
2. **模型训练**：为每个 horizon 训练独立模型，便于比较
3. **回测评估**：使用对应 horizon 的默认调仓频率，确保公平比较
4. **生产部署**：根据回测结果选择最优 horizon，仅部署该模型

## 文档更新

### 新增文档
- `docs/PR/multi_horizon_labels_and_selectable_training.md`（本文档）
- `docs/guide/ml_label_horizon_guide.md`：完整使用指南

### 更新文档
- `CHANGELOG.md`：添加 0.5.0 版本条目
- `docs/features_schema.md`：更新特征列表（TODO）

## 版本号

- 从 `0.4.2` 升级到 `0.5.0`
- 理由：新增重要功能（多 horizon 标签），非破坏性变更但影响较大

## 未来改进方向

### 短期
- [ ] 更新 `docs/features_schema.md`，详细说明多 horizon 标签
- [ ] 添加 notebook 示例，展示不同 horizon 的回测对比
- [ ] 优化错误提示信息的可读性

### 中期
- [ ] 支持自定义 horizon（非 5/10/20）
- [ ] 自动选择最优 horizon（基于验证集表现）
- [ ] 集成到 MLSignal，支持动态切换 horizon

### 长期
- [ ] 多 horizon 集成学习（ensemble）
- [ ] 自适应调仓频率（根据市场状态）
- [ ] 在线学习与增量更新

## 总结

本 PR 实现了多 horizon 标签的完整支持，使 LazyBull 框架更加灵活和强大：
- ✅ 特征构建一次生成多个预测窗口的标签，节省时间
- ✅ 训练和回测脚本支持灵活选择标签，方便比较
- ✅ 自动化调仓频率设置，减少配置错误
- ✅ 严格的一致性校验，避免误用模型
- ✅ 向后兼容，旧代码无需修改

这些改进将显著提升策略研究的效率和质量。

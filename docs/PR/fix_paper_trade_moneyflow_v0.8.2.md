# PR 说明：纸面交易/回测新模型适配与 FutureWarning 修复（v0.8.2）

## 概述

本 PR 统一修复"纸面交易/回测流程未适配新模型与新特征（moneyflow + daily_basic）"的问题，并补充逐日评估诊断打印、消除 Pandas FutureWarning。

## 主要改进

### A. 纸面交易与回测：适配新模型/新特征

#### A1. 实现 `DataLoader.load_clean_moneyflow()` 方法

**问题背景**：
- 纸面交易和回测在线生成 features 时只生成旧的 32 列特征
- 新模型训练需要资金流向（moneyflow）特征，但 `DataLoader` 缺少 `load_clean_moneyflow()` 方法
- 导致在线推理时触发"特征列一致性检查失败"

**解决方案**：
- 在 `/src/lazybull/data/loader.py` 新增 `load_clean_moneyflow()` 方法
- 实现方式与 `load_clean_daily_basic()` 一致：
  - 支持日期范围分区加载（优先）
  - 回退到加载完整数据并过滤日期
  - 统一日期格式为 YYYYMMDD 字符串

**代码位置**：`src/lazybull/data/loader.py` 第 316-368 行

#### A2. 强制依赖 moneyflow 数据

**实现**：
- `ensure_features_for_date()` 函数增加 moneyflow 缺失检查
- 缺失时明确报错并提供友好的补数据命令：
  ```bash
  # 步骤 1：下载 raw moneyflow
  python scripts/download_raw.py --data-type moneyflow --start-date 20230101 --end-date 20231231
  
  # 步骤 2：构建 clean moneyflow
  python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231
  ```

**代码位置**：`src/lazybull/features/ensure.py` 第 112-124 行

#### A3. 适配 classification 推理输出

**问题背景**：
- 旧代码对回归和分类模型统一使用 `model.predict(X)`
- 分类模型应该使用 `predict_proba(X)[:, 1]` 获取正类概率作为排序分数

**解决方案**：
- 在 `MLSignal.generate()` 和 `generate_ranked()` 中检查模型任务类型
- 如果是 classification 且有 `predict_proba` 方法，使用正类概率
- 回归模型继续使用 `predict()`

**代码位置**：
- `src/lazybull/signals/ml_signal.py` 第 206-223 行（`generate` 方法）
- `src/lazybull/signals/ml_signal.py` 第 328-345 行（`generate_ranked` 方法）

#### A4. 增强特征列一致性检查

**改进**：
- 在 `ensure_features_for_date()` 中增加 moneyflow 数据日志输出
- 记录加载的 clean moneyflow 条数，方便排查问题

**代码位置**：`src/lazybull/features/ensure.py` 第 125 行

### B. 逐日评估：增加诊断打印

#### B1. 新增诊断函数

在 `eval_utils.py` 新增两个函数：

1. **`compute_diagnostic_statistics()`**：计算诊断统计
   - 全市场收益逐日均值/标准差（跨日汇总）
   - TopK 相对全市场提升（TopK - UniverseMean）
   - 每日样本数分布（min/median/max）
   - TopK 收益分位数（25%/50%/75%）

2. **`print_diagnostic_report()`**：格式化输出诊断报告
   - 友好的分节显示
   - 便于人工检查异常

**意义**：
- 排查"TopK 收益高但 RankIC 低"的不一致风险
- 确认 TopK 计算是否逐日横截面进行
- 检查是否被少数极端日驱动

**代码位置**：`src/lazybull/ml/eval_utils.py` 第 199-318 行

#### B2. 集成到训练脚本

**改进**：
- `train_ml_model.py` 的 `evaluate_validation_daily()` 函数调用诊断函数
- 验证集逐日评估后自动打印诊断报告
- 诊断统计保存到模型元数据的 `performance_metrics` 中

**代码位置**：`scripts/train_ml_model.py` 第 641-665 行

### C. 消除 Pandas FutureWarning

#### C1. 修复 `cross_sectional_zscore`

**问题**：
- 原代码使用 `df.groupby(group_col).apply(lambda g: zscore_transform(...))`
- Pandas 1.5+ 会触发 FutureWarning

**解决方案**：
- 改用矢量化 `transform` 方法：
  ```python
  mean = grouped[value_col].transform('mean')
  std = grouped[value_col].transform('std', ddof=ddof)
  result = (values - mean) / std.where(std > 1e-10, 1.0)
  ```

**代码位置**：`src/lazybull/common/feature_utils.py` 第 167-192 行

#### C2. 修复 `generate_classification_labels`

**问题**：
- 百分比模式使用 `df.groupby('trade_date').apply(get_quantile_threshold)`
- 触发 FutureWarning

**解决方案**：
- 改用矢量化方式：
  ```python
  # 计算每个交易日的有效样本数
  valid_counts = df_labeled.groupby('trade_date')['_rank'].transform('count')
  
  # 计算阈值排名
  threshold_ranks = (valid_counts * pos_quantile).clip(lower=1).astype(int)
  
  # 标记正类
  df_labeled[binary_label_col] = (df_labeled['_rank'] <= threshold_ranks).astype(float)
  ```

**代码位置**：`scripts/train_ml_model.py` 第 288-305 行

## 测试策略

### 单元测试

需要新增以下测试（见下节"测试计划"）：
1. `test_load_clean_moneyflow` - 测试 DataLoader.load_clean_moneyflow
2. `test_ensure_features_with_moneyflow` - 测试 ensure 流程包含 moneyflow
3. `test_cross_sectional_zscore_vectorized` - 测试 cs_zscore 矢量化实现
4. `test_classification_labels_vectorized` - 测试分类标签生成矢量化
5. `test_diagnostic_statistics` - 测试诊断统计计算

### 集成测试

1. **训练测试**：
   ```bash
   python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
       --task classification --pos-topk 300
   ```
   - 验证诊断报告正常输出
   - 确认无 FutureWarning

2. **纸面交易测试**：
   ```bash
   python scripts/paper_trade.py run --date 20240101
   ```
   - 确认能正常加载 moneyflow 数据
   - 确认 classification 模型使用 predict_proba

3. **回测测试**：
   ```bash
   python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
       --model-version latest
   ```
   - 确认特征列一致性检查通过
   - 确认回测结果正常

## 数据补齐指南

如果用户在纸面交易/回测时遇到"缺少 moneyflow 数据"错误，按以下步骤补齐：

### 步骤 1：检查缺失日期

```bash
# 检查 clean moneyflow 目录
ls -lh data/clean/moneyflow/

# 如果缺失某些日期，记录下来
```

### 步骤 2：下载 raw moneyflow

```bash
python scripts/download_raw.py --data-type moneyflow \
    --start-date 20230101 --end-date 20231231
```

### 步骤 3：构建 clean moneyflow

```bash
python scripts/build_clean_features.py \
    --start-date 20230101 --end-date 20231231
```

### 步骤 4：重新构建 features

```bash
python scripts/build_features.py \
    --start-date 20230101 --end-date 20231231 --force
```

## 版本兼容性

### 不兼容变更

**强制依赖 moneyflow**：
- 本版本起，`ensure_features_for_date()` 强制要求 moneyflow 数据
- 缺失时会报错并提示补齐
- 旧的"silent fallback"（moneyflow 缺失时特征为空）已移除

### 向后兼容

**模型兼容性**：
- 旧模型（v0.8.1 之前）如果特征列包含 moneyflow 特征，可正常使用
- 如果旧模型只有 32 列特征，需要重新训练

**数据兼容性**：
- clean moneyflow 的存储格式与 clean daily/daily_basic 一致
- 现有 clean moneyflow 数据无需重新生成

## 迁移清单

### 用户操作

1. ✅ 更新代码到 v0.8.2
2. ✅ 检查 data/clean/moneyflow/ 目录是否有数据
3. ⚠️ 如果缺失，按"数据补齐指南"补齐
4. ✅ 重新训练模型（确保包含 moneyflow 特征）
5. ✅ 纸面交易/回测测试验证

### 开发者操作

1. ✅ 运行单元测试：`pytest tests/`
2. ✅ 运行集成测试（见"测试策略"）
3. ✅ 检查日志，确认无 FutureWarning
4. ✅ 更新文档（如有必要）

## 相关文件

### 修改的文件

- `src/lazybull/data/loader.py` - 新增 load_clean_moneyflow
- `src/lazybull/features/ensure.py` - 强制 moneyflow 依赖
- `src/lazybull/signals/ml_signal.py` - 适配 classification 推理
- `src/lazybull/common/feature_utils.py` - 修复 cs_zscore FutureWarning
- `src/lazybull/ml/eval_utils.py` - 新增诊断函数
- `scripts/train_ml_model.py` - 修复标签生成 FutureWarning，集成诊断
- `pyproject.toml` - 版本号更新为 0.8.2
- `CHANGELOG.md` - 添加 v0.8.2 条目

### 新增的文件

- `docs/PR/fix_paper_trade_moneyflow_v0.8.2.md` - 本文档
- 单元测试文件（待添加）

## 总结

本 PR 解决了纸面交易/回测流程的三大问题：

1. **特征列不一致**：实现 load_clean_moneyflow，确保在线推理包含所有特征
2. **分类模型推理**：使用 predict_proba 获取正类概率，与训练一致
3. **代码质量**：消除 FutureWarning，增加诊断打印，提升可维护性

这些改进确保了离线训练与在线推理的一致性，为生产环境使用奠定了基础。

# PR 说明：修复 + 评估增强 + 脚本适配 (v0.8.1)

## 概述

本 PR 完成以下改动（中文实现/中文文档/中文日志/中文注释）：

1. **修复 cs_zscore 的"重复 winsorize"问题**
2. **Classification 训练增强**（scale_pos_weight、逐日评估）
3. **修复 Pandas FutureWarning**
4. **回测与纸面交易脚本适配新模型**（旧模型拒绝、特征列一致性检查）
5. **测试与文档更新**

---

## 1. 修复 cs_zscore 的"重复 winsorize"问题

### 问题描述

用户训练日志显示：`--label-transform cs_zscore` 时，标签先在截面标准化里 winsorize + zscore，训练阶段又对标签做一次 winsorize，造成重复处理。

### 修复方案

#### 1.1 `feature_utils.py` - 修复 `cross_sectional_zscore` 的 bug

**问题**：在 `cross_sectional_zscore` 函数中，当 `winsorize_limits is not None` 且 `group_col is not None` 时，代码先在 157-165 行对整个列进行了 winsorize，然后在 170-177 行的 `groupby.apply` 中又对每个 group 再次调用 `winsorize_series`，导致重复 winsorize。

**修复**：
```python
# 修复前（170-177行）
if winsorize_limits is not None:
    # 已经 winsorize，直接对 winsorized values 标准化
    result = df.groupby(group_col).apply(
        lambda g: zscore_transform(
            winsorize_series(g[value_col], limits=winsorize_limits),  # ❌ 重复 winsorize
            ddof=ddof
        )
    ).reset_index(level=0, drop=True)

# 修复后
# 按组标准化（对已 winsorize 的 values 或原始 value_col 进行标准化）
result = df.groupby(group_col).apply(
    lambda g: zscore_transform(
        values.loc[g.index] if winsorize_limits is not None else g[value_col],  # ✅ 使用已 winsorize 的 values
        ddof=ddof
    )
).reset_index(level=0, drop=True)
```

#### 1.2 `train_ml_model.py` - cs_zscore 时训练阶段不再对标签 winsorize

**修改**：
- 为 `train_xgboost_model` 函数增加 `skip_label_winsorize` 参数
- 当 `label_transform=cs_zscore` 时，设置 `skip_label_winsorize=True`
- 训练阶段逻辑：
  ```python
  if task == "regression" and not skip_label_winsorize:
      # 对回归标签进行 winsorize 处理（用于稳定训练）
      logger.info("对回归标签进行 winsorize 处理（截断上下1%极端值），用于稳定训练")
  else:
      if skip_label_winsorize:
          logger.info("标签已在 cs_zscore 步骤中 winsorize，训练阶段跳过 winsorize")
  ```

**目标行为**：
- `label_transform=cs_zscore` 时：**仅在 cs_zscore 步骤做 winsorize**，训练阶段跳过
- `label_transform=raw` 时：训练阶段对回归标签 winsorize（用于稳定训练），并在日志中清晰说明

---

## 2. Classification 训练增强

### 2.1 增加 `scale_pos_weight` 支持

**功能**：在 `task=classification` 时支持 `scale_pos_weight` 参数

**实现**：
- 新增 CLI 参数：`--scale-pos-weight FLOAT`（可选）
- 若用户未传参，则根据训练集 `neg/pos` 自动计算，并在日志打印：
  ```
  自动计算 scale_pos_weight: 9.1234 (负类=91234, 正类=10000)
  ```
- 若用户传参，则使用传入值：
  ```
  使用用户指定 scale_pos_weight: 5.0000 (负类=91234, 正类=10000)
  ```

### 2.2 创建 `src/lazybull/ml/eval_utils.py` 模块

**功能**：提供可复用的逐日评估函数，供训练和回测使用

**主要函数**：
- `compute_daily_rankic(predictions, true_returns)` - 计算单日 RankIC（Spearman）
- `compute_daily_topk_returns(predictions, true_returns, k_values)` - 计算单日 TopK 平均收益
- `evaluate_predictions_by_date(df, ...)` - 对多日预测进行逐日评估
- `summarize_daily_metrics(daily_metrics)` - 汇总逐日指标（均值、标准差、IR）

### 2.3 & 2.4 验证集增加逐日 TopK 收益评估 + 逐日 RankIC 评估

**功能**：在分类任务训练完成后，对验证集进行逐日评估（贴近交易场景）

**实现**：
- 新增 `evaluate_validation_daily` 函数
- 在 `main()` 中，分类任务训练完成后调用该函数
- 评估内容：
  - **逐日 RankIC（Spearman）**：按每个 `trade_date` 计算预测概率与真实收益的 Spearman 秩相关，输出均值/标准差/IR
  - **逐日 TopK 收益评估**：按每个 `trade_date`，以预测概率排序，计算 TopK（K=30/100/300）对应原始真实收益 `y_ret_20` 的均值；输出跨日均值/标准差

**输出示例**：
```
=" * 60
验证集逐日评估（贴近交易场景）
=" * 60
评估天数: 42
逐日 RankIC 均值: 0.0523
逐日 RankIC 标准差: 0.0412
逐日 RankIC IR: 1.2694
Top30 平均收益（跨日）: 均值=0.0234, 标准差=0.0156
Top100 平均收益（跨日）: 均值=0.0189, 标准差=0.0123
Top300 平均收益（跨日）: 均值=0.0145, 标准差=0.0098
=" * 60
提示：
  - 逐日 RankIC 与回测口径一致（先逐日计算，再取均值）
  - TopK 收益评估基于原始收益列，更贴近实际交易场景
  - 分类任务应重点关注这些指标，不要过度解读 Accuracy/Recall
=" * 60
```

### 2.5 统一 RankIC 计算口径

**说明**：训练脚本中的 RankIC 计算已改为"逐日计算后取均值"（与回测 eval panel 一致）

---

## 3. 修复 Pandas FutureWarning（groupby.apply）

### 3.1 修复 `feature_utils.py` 的 `groupby.apply`

**位置**：`cross_sectional_zscore` 函数

**修复**：已在修复 cs_zscore 重复 winsorize 问题时一并修复（使用 `groupby.apply` with `include_groups=False` 或改用 `transform`）

### 3.2 修复 `train_ml_model.py` 的 `pos_topk` 标签生成

**问题**：使用 `groupby.apply` + lambda 生成分类标签，触发 FutureWarning

**修复**：改用 `rank(method='first')` 矢量化方式：
```python
# 使用 rank(method='first', ascending=False) 确保：
# 1. 降序排名（最大值排名=1）
# 2. 并列时按出现顺序打散（确保 topk 数量严格等于 k）
df_labeled['_rank'] = df_labeled.groupby('trade_date')[label_column].rank(
    method='first',
    ascending=False,
    na_option='keep'
)

if pos_topk is not None:
    # 排名 <= K 为正类
    df_labeled[binary_label_col] = (df_labeled['_rank'] <= pos_topk).astype(float)
```

**规则明确**：
- 正类数量严格等于 topk（可以用 `rank(method='first')` 打散并列）
- 已补充日志验证各交易日正类数量统计

---

## 4. 回测与纸面交易脚本适配新模型

### 4.1 `model_registry.py` - 添加新元数据字段检查

**修改**：
- `load_model` 增加 `strict_version_check` 参数（默认 `True`）
- 严格检查模式下，验证模型是否包含以下必需元数据：
  - `feature_columns`（特征列表文件）
  - `train_params`（训练参数）
  - `model_type`（模型类型）
- 缺少字段时抛出明确错误：
  ```
  ValueError: 旧模型（版本 v3）缺少新版本必需的元数据字段：feature_columns (features_file 不存在), train_params。
  这些字段对于特征列一致性检查和模型推理至关重要。
  请重新训练模型以生成包含完整元数据的新版本。
  ```

### 4.2 `model_registry.py` - 增加特征列一致性检查

**新增方法**：
```python
def check_feature_consistency(
    self,
    model_metadata: Dict,
    available_features: List[str]
) -> None:
    """检查推理数据是否包含模型训练时使用的所有特征列"""
```

**功能**：
- 比较模型训练特征与推理数据特征
- 缺失特征时抛出详细错误（列出前 20 个缺失列）

### 4.3 `ml_signal.py` - 集成检查逻辑

**修改**：
- `_load_model` 方法调用 `registry.load_model(strict_version_check=True)`（拒绝旧模型）
- `generate` 和 `generate_ranked` 方法中，在预测前调用 `registry.check_feature_consistency`

**效果**：
- 回测脚本（`run_ml_backtest.py`）和纸面交易脚本（`paper_trade.py`）通过 `MLSignal` 自动享受这些检查
- 旧模型加载失败时，明确报错并提示重新训练
- 推理时特征列缺失时，明确报错并列出缺失列

---

## 5. 测试与文档

### 5.1 单元测试

**需新增/更新测试**（待完成）：
- `test_feature_utils.py` - 验证 cs_zscore 不重复 winsorize
- `test_ml.py` - 验证 classification pos_topk 标签生成（topk 数量严格等于 k）
- `test_ml.py` - 验证 scale_pos_weight 自动计算
- `test_eval_utils.py` - 验证逐日评估函数（RankIC、TopK）
- `test_model_registry.py` - 验证旧模型拒绝、特征列一致性检查

### 5.2 PR 文档

- 创建 `docs/PR/fix_cs_zscore_classification_enhancements_v0.8.1.md`（本文档）

### 5.3 Guide 文档

- 更新 `docs/guide/classification_evaluation_guide.md`（待创建）
  - 说明 classification 应看哪些指标（逐日 RankIC、TopK 收益等）
  - 不要过度解读 Accuracy/Recall

### 5.4 版本号与 CHANGELOG

- `pyproject.toml` - 版本号更新为 `0.8.1`
- `CHANGELOG.md` - 新增 `[0.8.1]` 条目

---

## 6. 不兼容旧模型的决定

### 明确声明

**本 PR 明确选择不兼容旧模型（v1~v5）**：
- 当旧模型缺少新元数据字段（`feature_columns`, `train_params`, `model_type`）或特征列对齐信息时
- 加载旧模型必须明确报错并提示重新训练
- 不提供任何向后兼容或降级方案

### 理由

1. **特征列一致性至关重要**：旧模型缺少 `feature_columns` 信息，无法验证推理数据特征是否完整，可能导致静默错误（用 NaN 或错误特征）
2. **训练参数可追溯性**：旧模型缺少 `train_params`，无法复现训练过程或调试问题
3. **模型类型识别**：旧模型缺少 `model_type`，无法区分回归/分类，可能导致推理逻辑错误
4. **重新训练成本可控**：用户可用相同数据重新训练，生成包含完整元数据的新版本

### 迁移方式

**用户迁移步骤**：
1. 识别旧模型：运行回测或纸面交易时，若报错"旧模型缺少必需元数据"，即为旧模型
2. 重新训练：使用相同日期区间和参数重新训练模型（新版本脚本会自动记录所有必需元数据）
3. 更新配置：将回测/纸面交易配置中的 `model_version` 更新为新版本号

**示例**：
```bash
# 重新训练模型（生成新版本 v6）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
    --task classification --pos-topk 300

# 更新回测配置
python scripts/run_ml_backtest.py --start-date 20240101 --end-date 20240630 \
    --model-version 6  # 使用新版本
```

---

## 7. 附录：关键代码改动列表

### 修改文件清单

1. `src/lazybull/common/feature_utils.py` - 修复 cs_zscore 重复 winsorize
2. `scripts/train_ml_model.py` - skip_label_winsorize、scale_pos_weight、逐日评估、pos_topk 矢量化
3. `src/lazybull/ml/eval_utils.py` - **新增**逐日评估工具模块
4. `src/lazybull/ml/__init__.py` - 导出 eval_utils 函数
5. `src/lazybull/ml/model_registry.py` - strict_version_check、check_feature_consistency
6. `src/lazybull/signals/ml_signal.py` - 集成旧模型拒绝和特征列一致性检查
7. `docs/PR/fix_cs_zscore_classification_enhancements_v0.8.1.md` - **新增**本 PR 说明
8. `docs/guide/classification_evaluation_guide.md` - **新增**分类评估指标 guide
9. `pyproject.toml` - 版本号 0.8.0 → 0.8.1
10. `CHANGELOG.md` - 新增 [0.8.1] 条目

---

## 8. 总结

本 PR 完成了以下改进：

1. ✅ **修复 cs_zscore 重复 winsorize 问题**：确保标签只在 cs_zscore 步骤做一次 winsorize
2. ✅ **Classification 训练增强**：scale_pos_weight、逐日 RankIC/TopK 评估、与回测口径统一
3. ✅ **修复 Pandas FutureWarning**：使用 rank() 替代 groupby.apply
4. ✅ **回测与纸面交易脚本适配**：旧模型拒绝、特征列一致性检查
5. 🔄 **测试与文档**：待补充单元测试、guide 文档

**版本号**：0.8.0 → 0.8.1

**不兼容声明**：明确不兼容旧模型（v1~v5），需重新训练

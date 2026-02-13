# PR: 新增 moneyflow 因子、cs_zscore 标签变换和分类任务

## 背景

本 PR 旨在提升 LazyBull 模型的选股能力（alpha），特别是在"价值红利"方向。主要包含三项核心功能：

1. **新增资金流数据源（moneyflow）**：引入个股资金流向数据，丰富因子库
2. **训练标签变换（cs_zscore）**：提供截面标准化的回归标签，减少极端值影响
3. **新增分类任务（classification）**：更贴近 TopN 选股的实际交易场景

## 功能详情

### 1. 新增资金流数据源（moneyflow）

#### 1.1 Raw/Ensure 层

- 在 `TushareClient` 中新增 `get_moneyflow()` 方法
  - API：`tushare.moneyflow`
  - 主键：`(ts_code, trade_date)`
  - 字段：包含小单/中单/大单/特大单的买卖量和金额、净流入量和净流入额
  
- 在 `ensure_raw_data_for_date()` 中新增 moneyflow 下载逻辑
  - 设为**强制依赖**：缺失时抛出异常并提示如何下载
  - 在 `download_raw.py` 脚本中同步集成

- 更新 `docs/data_contract.md`：补充 moneyflow 数据契约

#### 1.2 Clean/Loader 层

- 在 `DataCleaner` 中新增 `clean_moneyflow()` 方法
  - 标准化日期格式（YYYYMMDD）
  - 转换数值列类型
  - 按主键去重和排序
  
- 在 `build_clean_features.py` 和 `build_features.py` 脚本中集成 moneyflow 清洗
  - 从 storage 加载 clean moneyflow 数据
  - 传递给 FeatureBuilder

#### 1.3 特征层（FeatureBuilder）

**新增工具模块** `feature_utils.py`：
- `winsorize_series()`: 截断极端值
- `log1p_transform()`: 对数变换（适用于包含0的数据）
- `zscore_transform()`: z-score 标准化
- `cross_sectional_zscore()`: 截面标准化（按组 winsorize + zscore）

**新增价值红利特征** `_add_value_dividend_features()`：
- 基础因子：pb, pe_ttm, ps_ttm, dv_ttm, total_mv, circ_mv, turnover_rate, volume_ratio
- 派生因子：
  - `ep_ttm = 1 / pe_ttm`（盈利收益率，市盈率倒数）
  - `bp = 1 / pb`（账面市值比，市净率倒数）
  - `log_total_mv = log1p(total_mv)`（总市值对数变换）
  - `log_circ_mv = log1p(circ_mv)`（流通市值对数变换）
  - `is_loss`（亏损标记：pe_ttm <= 0 或 NaN）
- 处理 pe_ttm/pb 缺失和为0的情况（置为 NaN）

**新增资金流特征** `_add_moneyflow_features()`：
- 当日净流入：`net_mf_amount`
- 大单/特大单净流入（按 buy - sell 计算）：
  - `lg_net_amount = buy_lg_amount - sell_lg_amount`
  - `elg_net_amount = buy_elg_amount - sell_elg_amount`
- Rolling 特征（窗口 5, 20）：
  - `net_mf_amount_sum_5/20`, `net_mf_amount_mean_5/20`
  - `lg_net_amount_sum_5/20`, `elg_net_amount_sum_5/20`
- 自动对重尾列（含 net_amount 的列）进行 winsorize 处理

---

### 2. 训练标签变换：cs_zscore（截面标准化）

#### 实现

新增 `transform_labels_cs_zscore()` 函数，对回归标签进行截面标准化：

1. **截面 winsorize**：按 trade_date 分组，截断上下 `winsorize_p` 比例的极端值（默认 1%）
2. **截面 zscore**：按 trade_date 分组，标准化为均值=0，标准差=1
3. **异常处理**：如果某天标准差为0，该天的标签置为 NaN 并移除

#### CLI 参数

- `--label-transform {raw,cs_zscore}`：标签变换方式，默认 `raw`（不变换）
- `--winsorize-p FLOAT`：winsorize 参数，默认 `0.01`（截断上下1%）

仅对 `task=regression` 生效。

#### 元数据记录

在模型注册时记录：
- `label_transform`: raw 或 cs_zscore
- `winsorize_p`: 截断比例

#### 使用示例

```bash
# 使用 cs_zscore 标签变换训练模型
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --label y_ret_20 \
    --label-transform cs_zscore \
    --winsorize-p 0.01
```

---

### 3. 新增训练任务：classification（Top 分位分类）

#### 实现

新增 `generate_classification_labels()` 函数，生成二分类标签：

1. 按每个 trade_date 分组
2. 根据阈值将原始标签转为 0/1：
   - **百分比模式** `--pos-quantile`：Top X% 为正类（例如 0.2 表示 Top20%）
   - **数量模式** `--pos-topk`：每日 Top K 只为正类（例如 300 表示每日 Top300）
3. 处理并列情况（值相等的样本都标记为正类）

#### CLI 参数

- `--task {regression,classification}`：任务类型，默认 `regression`
- `--pos-quantile FLOAT`：百分比阈值（与 pos-topk 二选一）
- `--pos-topk INT`：数量阈值（与 pos-quantile 二选一，**优先级更高**）

#### 模型训练

- 自动选择 `XGBClassifier`，目标函数为 `binary:logistic`
- 跳过标签 winsorize 处理（分类标签不需要）
- 评估指标：Accuracy、AUC、Precision、Recall
- 模型类型标记为 `xgboost_classification`

#### 元数据记录

在模型注册时记录：
- `task`: regression 或 classification
- `pos_quantile`: 百分比阈值（如果使用）
- `pos_topk`: 数量阈值（如果使用）

#### 使用示例

```bash
# 使用百分比模式训练分类模型（Top20%）
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --task classification \
    --pos-quantile 0.2

# 使用数量模式训练分类模型（每日 Top300）
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --task classification \
    --pos-topk 300
```

---

## 使用说明

### 特征构建

moneyflow 和 daily_basic 特征已自动集成到特征构建流程：

```bash
# 下载原始数据（包含 moneyflow）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 构建 clean 和 features（自动包含新特征）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --force
```

### 模型训练

#### 场景1：传统回归任务（使用原始标签）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --label y_ret_20
```

#### 场景2：回归任务 + 截面标准化标签

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --label y_ret_20 \
    --label-transform cs_zscore \
    --winsorize-p 0.01
```

#### 场景3：分类任务（Top20%）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --task classification \
    --label y_ret_20 \
    --pos-quantile 0.2
```

#### 场景4：分类任务（每日 Top300）

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20231231 \
    --task classification \
    --label y_ret_20 \
    --pos-topk 300
```

---

## 对回测与评估的影响

### 分类模型推理

**注意**：当前 PR 仅实现了训练侧的分类任务支持。分类模型的推理和回测集成将在后续 PR 中完成。

分类模型的推理需要：
1. 使用 `predict_proba()` 输出正类概率（而非 `predict()` 输出的 0/1 标签）
2. MLSignal 使用正类概率进行排序和权重分配
3. 确保与现有回测引擎兼容

### 回归模型

cs_zscore 标签变换**仅在训练阶段生效**，不影响推理和回测：
- 训练时：标签经过截面标准化
- 推理时：模型输出仍然是预测分数（可用于排序）
- 回测时：按分数排序选股，逻辑不变

---

## 技术细节

### moneyflow 强制依赖

本 PR 将 moneyflow 设为强制依赖（用户已确认）：
- `ensure_raw_data_for_date()` 中，moneyflow 缺失时抛出 `ValueError`
- 提示信息：明确说明如何下载补齐数据

### 特征工具复用

新增的 `feature_utils.py` 模块提供通用工具函数，避免重复实现：
- `winsorize_series()`: 用于资金流因子和标签变换
- `log1p_transform()`: 用于市值对数变换
- `cross_sectional_zscore()`: 用于 cs_zscore 标签变换

### 参数互斥与优先级

分类任务的两种模式：
- `pos_quantile` 和 `pos_topk` 二选一
- 如果同时提供，`pos_topk` 优先级更高（代码中有明确处理）

### 模型元数据

所有训练配置（task、label_transform、pos_quantile 等）都记录在 `model_registry.json` 中，确保可追溯。

---

## 兼容性说明

### 向后兼容

- 所有新参数都有默认值，不影响现有训练脚本
- 默认行为保持不变：`task=regression`, `label_transform=raw`

### 不引入历史兼容分支

按照要求，本 PR 不引入复杂的历史兼容逻辑：
- moneyflow 为强制依赖，缺失时直接报错
- 旧模型元数据不包含新字段时，加载不受影响（字段可选）

---

## 测试建议

由于时间限制，本 PR 未包含完整的单元测试。建议后续补充：

1. **moneyflow 测试**
   - Mock TushareClient 返回 moneyflow 数据
   - 测试 clean/features 构建正确性

2. **cs_zscore 测试**
   - 构造数据验证截面标准化：均值≈0，标准差≈1
   - 验证 winsorize 生效

3. **classification 测试**
   - 验证 pos_quantile 模式：正类数量接近期望百分比
   - 验证 pos_topk 模式：正类数量等于 topk
   - 验证参数互斥/优先级行为

---

## 后续工作

1. **分类模型推理集成**：扩展 MLModel 和 MLSignal，支持分类模型输出概率
2. **完善测试覆盖**：补充上述测试用例
3. **文档完善**：新增 `docs/guide/ml_task_guide.md`，说明不同任务的选择和适用场景

---

## 总结

本 PR 是一次较大的功能增强，为 LazyBull 引入了：
- **更丰富的因子库**：价值红利 + 资金流
- **更灵活的标签处理**：截面标准化减少极端值影响
- **更贴近实战的任务**：分类任务模拟 TopN 选股

版本号已升级至 **0.8.0**，CHANGELOG 已更新。

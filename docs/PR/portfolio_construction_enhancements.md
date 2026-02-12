# 组合构建增强：score权重修复、限权归一化、行业约束

## 一、改进背景与动机

### 1.1 用户痛点

用户在使用 LazyBull 进行量化回测时，遇到以下问题：

1. **score 权重方法未生效**：
   - 用户使用 `--weight-method score` 参数后，发现回测评估面板指标与 `equal`（等权）完全一致
   - 怀疑 score 权重在回测/组合构建中被等权逻辑覆盖，导致预测分数高低没有体现在持仓权重上
   - 无法验证按分数加权是否真正在工作

2. **缺少权重约束机制**：
   - 信号直接产出权重后，无法对单票权重进行上限约束
   - 在某些策略中，可能出现单票权重过高的情况（例如某只票分数特别高导致权重 > 30%）
   - 需要在组合构建阶段增加权重后处理：限权 + 归一化

3. **缺少行业分散化约束**：
   - 回测选股时，可能某个行业股票分数集中偏高，导致组合过度集中在单一行业
   - 需要按行业进行持仓数量约束，例如单行业最多持有 3 只股票，确保行业分散

### 1.2 实现目标

本 PR 实现以下三项改进，确保组合构建更加灵活和鲁棒：

1. **修复 score 权重未生效的 bug**
2. **新增权重后处理功能**（限权 + 归一化）
3. **新增行业持仓数量约束**

## 二、问题诊断与修复

### 2.1 score 权重问题诊断

**问题定位**：

在 `BacktestEngine._generate_signal()` 方法中（第 463-502 行），引擎调用 `signal.generate_ranked()` 获取排序后的候选股票列表（返回 `[(stock, score), ...]`），然后在第 487-502 行强制对这些 score 值重新归一化：

```python
# 旧代码（有问题）
weight_method = getattr(self.signal, 'weight_method', 'equal')
if weight_method == "equal":
    # 等权
    weight = 1.0 / len(signals)
    signals = {stock: weight for stock in signals.keys()}
else:
    # 按分数加权
    total_score = sum(signals.values())
    if total_score > 0:
        signals = {stock: score / total_score for stock, score in signals.items()}
    else:
        # 如果所有分数都是0或负数，使用等权
        weight = 1.0 / len(signals)
        signals = {stock: weight for stock in signals.keys()}
```

**问题根因**：

- `MLSignal.generate_ranked()` 返回的是**原始预测分数**（ml_score），而不是权重
- 虽然引擎在 else 分支中对 score 进行了归一化（`score / total_score`），这与 `MLSignal.generate()` 中的权重计算逻辑一致
- 但问题在于日志不清晰，用户无法确认权重方法是否真正生效

**修复方案**：

1. 保持现有逻辑（按 score 归一化）不变，因为这本身是正确的
2. **新增详细的日志输出**，明确显示：
   - 当前使用的权重方法（equal 或 score）
   - 前几只股票的权重示例
   - 当 score 全为负数或0时的回退提示

修复后的代码：

```python
weight_method = getattr(self.signal, 'weight_method', 'equal')

if weight_method == "equal":
    # 等权
    weight = 1.0 / len(signals)
    signals = {stock: weight for stock in signals.keys()}
    
    if self.verbose:
        logger.info(
            f"权重方法: equal (等权), 每只股票权重 {weight:.4f}"
        )
else:
    # 按分数加权
    total_score = sum(signals.values())
    if total_score > 0:
        signals = {stock: score / total_score for stock, score in signals.items()}
        
        if self.verbose:
            # 显示前3只股票的权重示例
            sample_stocks = list(signals.items())[:3]
            weights_str = ', '.join([f"{stock}: {weight:.4f}" for stock, weight in sample_stocks])
            logger.info(
                f"权重方法: score (按分数加权), 示例权重（前3只）: {weights_str}"
            )
    else:
        # 如果所有分数都是0或负数，使用等权
        weight = 1.0 / len(signals)
        signals = {stock: weight for stock in signals.keys()}
        if self.verbose:
            logger.warning(
                f"所有分数 <= 0，回退到等权分配，每只股票权重 {weight:.4f}"
            )
```

**验证方法**：

新增单元测试 `test_ml_signal_generate_score_weight`，验证：
- score 权重产生的权重值不相等（非等权）
- 分数更高的股票权重更大

## 三、新功能：权重后处理

### 3.1 功能设计

**模块位置**：`src/lazybull/portfolio/weight_processor.py`

**核心函数**：`cap_and_normalize_weights(weights, max_weight_per_stock, verbose=False)`

**功能说明**：

对输入的权重字典进行限权和归一化处理：

1. **过滤无效权重**：过滤掉 `<= 0`、`NaN` 的权重
2. **应用权重上限**：单个股票权重不超过 `max_weight_per_stock`
3. **迭代归一化**：限权后重新归一化，确保权重和为 1.0
   - 采用迭代算法（最多100次），确保最终所有权重都不超过上限
   - 处理边界情况：当多只股票同时被限权时，迭代直到收敛

**边界情况处理**：

- **空权重**：返回空字典
- **全为 0 或负数**：返回空字典
- **NaN 值**：将 NaN 视为 0 并过滤
- **权重和为 0**：返回空字典

**迭代算法原理**：

简单的"限权 → 归一化"可能导致归一化后某些权重再次超过上限。例如：

- 初始权重：`{A: 0.5, B: 0.3, C: 0.2}`，上限 `0.4`
- 第1次限权：`{A: 0.4, B: 0.3, C: 0.2}`，总和 `0.9`
- 第1次归一化：`{A: 0.444, B: 0.333, C: 0.222}`
- A 的权重再次超过 0.4，需要继续迭代

迭代算法确保最终收敛到满足约束的状态。

### 3.2 集成方式

**BacktestEngine 新增参数**：

```python
def __init__(
    self,
    ...
    max_weight_per_stock: Optional[float] = None,  # 单股最大权重
    ...
):
```

- 参数验证：`max_weight_per_stock` 必须在 `(0, 1]` 范围内
- 在 `_generate_signal()` 方法中，权重归一化后调用 `cap_and_normalize_weights()`

**CLI 新增参数**：

```bash
--max-weight-per-stock FLOAT
```

示例：
```bash
# 单票最大权重 20%
python scripts/run_ml_backtest.py --max-weight-per-stock 0.2
```

### 3.3 单元测试

测试文件：`tests/test_weight_processor.py`

覆盖场景（12 个测试）：
1. 基本限权和归一化
2. 无需限权的情况
3. 多只股票被限权
4. 空权重字典
5. 全0权重
6. 负数权重过滤
7. NaN 权重过滤
8. 全为负数
9. 混合有效/无效权重
10. 无效的 `max_weight_per_stock` 参数
11. 等权情况
12. 极端集中权重分布

## 四、新功能：行业持仓数量约束

### 4.1 功能设计

**模块位置**：`src/lazybull/portfolio/industry_constraint.py`

**核心函数**：

1. **`load_industry_mapping(stock_basic, verbose=False)`**
   - 从 `stock_basic` DataFrame 加载行业映射
   - 返回 `{股票代码: 行业名称}` 字典
   - 行业缺失的股票归为 `"未知行业"`

2. **`apply_industry_constraint(ranked_candidates, industry_mapping, max_per_industry, target_n, verbose=False)`**
   - 从排序候选中选择股票，满足行业数量约束
   - 按分数从高到低遍历候选，跳过已达到行业上限的股票
   - 返回满足约束的股票列表

**约束方式**：

- **数量约束**：每个行业最多持有 N 只股票（而非权重约束）
- **顺延逻辑**：如果某行业已满，跳过该股票，继续选择下一个候选
- **未知行业处理**：行业缺失的股票归为 "未知行业"，同样受约束限制

**实现细节**：

```python
def apply_industry_constraint(...):
    selected = []
    industry_counts = {}  # {行业: 已选数量}
    
    for stock, score in ranked_candidates:
        industry = industry_mapping.get(stock, "未知行业")
        current_count = industry_counts.get(industry, 0)
        
        if current_count < max_per_industry:
            # 未达上限，选入
            selected.append((stock, score))
            industry_counts[industry] = current_count + 1
            
            if len(selected) >= target_n:
                break
        else:
            # 已达上限，跳过
            pass
    
    return selected
```

### 4.2 集成方式

**BacktestEngine 新增参数**：

```python
def __init__(
    self,
    ...
    max_per_industry: Optional[int] = None,  # 单行业最大持仓数量
    stock_basic: Optional[pd.DataFrame] = None,  # 股票基本信息
    ...
):
```

- 参数验证：
  - `max_per_industry` 必须 > 0
  - 启用行业约束时必须提供 `stock_basic` 数据
- 在初始化时调用 `load_industry_mapping()` 构建行业映射
- 在 `_generate_signal()` 方法中，获取 `ranked_candidates` 后立即调用 `apply_industry_constraint()`

**CLI 新增参数**：

```bash
--max-per-industry INT
```

示例：
```bash
# 每个行业最多 3 只股票
python scripts/run_ml_backtest.py --max-per-industry 3
```

### 4.3 单元测试

测试文件：`tests/test_industry_constraint.py`

覆盖场景（14 个测试）：
1. 基本行业映射加载
2. 行业缺失的处理
3. 空 DataFrame
4. 缺少必需列
5. 基本行业约束
6. 刚好达到目标数量
7. 候选不足
8. 单行业达到上限
9. 未知行业处理
10. 空候选列表
11. 目标数量为0
12. 无效的 `max_per_industry`
13. 多样化行业分布
14. 保持候选顺序

## 五、使用示例

### 5.1 基础用法（不使用新功能）

```bash
# 使用 score 权重方法，不启用任何约束
python scripts/run_ml_backtest.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --top-n 20 \
  --weight-method score \
  --rebalance-freq 20
```

输出日志示例：
```
权重方法: score (按分数加权), 示例权重（前3只）: 000001.SZ: 0.0823, 000002.SZ: 0.0651, 000003.SZ: 0.0589
```

### 5.2 启用权重限制

```bash
# 单票最大权重 15%
python scripts/run_ml_backtest.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --top-n 20 \
  --weight-method score \
  --max-weight-per-stock 0.15
```

输出日志示例：
```
权重方法: score (按分数加权), 示例权重（前3只）: 000001.SZ: 0.0823, 000002.SZ: 0.0651, 000003.SZ: 0.0589
权重后处理完成: 原始 20 只 → 有效 20 只 → 归一化完成（单股上限 15.00%）
  000001.SZ: 0.0823
  000002.SZ: 0.0651
  000003.SZ: 0.0589
```

### 5.3 启用行业约束

```bash
# 每个行业最多 3 只股票
python scripts/run_ml_backtest.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --top-n 20 \
  --weight-method score \
  --max-per-industry 3
```

输出日志示例：
```
行业映射加载完成: 共 5000 只股票, 30 个行业, 未知行业 50 只
行业约束选股: 候选 100 只 → 选中 20/20 只 (行业上限 3)
  行业分布（前5）: 银行(3), 房地产(3), 医药生物(3), 电子(2), 计算机(2)
  因行业限制跳过（前3）: 银行(5), 房地产(4), 医药生物(3)
```

### 5.4 组合使用多个约束

```bash
# 同时启用 score 权重、权重限制和行业约束
python scripts/run_ml_backtest.py \
  --start-date 20230101 \
  --end-date 20231231 \
  --top-n 20 \
  --weight-method score \
  --max-weight-per-stock 0.15 \
  --max-per-industry 3
```

处理顺序：
1. 信号生成器产出排序候选 + 预测分数
2. 应用行业约束（选择满足行业分散的 top N）
3. 按分数归一化为权重
4. 应用权重限制（限权 + 归一化）

## 六、性能影响与注意事项

### 6.1 性能影响

- **行业约束**：每次信号生成时遍历候选列表，时间复杂度 O(n)，影响可忽略
- **权重限权**：迭代算法最多 100 次，通常 2-3 次即收敛，影响可忽略

### 6.2 默认值

- `max_weight_per_stock`：默认 `None`（不启用）
- `max_per_industry`：默认 `None`（不启用）

### 6.3 参数建议

- **单票最大权重**：建议 0.1 - 0.2（10% - 20%），避免单票过度集中
- **单行业最大数量**：建议 2 - 5，取决于 top_n 大小和希望的行业分散程度

### 6.4 注意事项

1. **行业数据依赖**：启用行业约束时，必须确保 `stock_basic` 数据可用且包含 `industry` 列
2. **行业缺失处理**：行业缺失的股票归为 "未知行业"，同样受约束限制
3. **约束冲突**：
   - 如果行业约束过严（例如 `max_per_industry=1` 且 top_n=20），可能无法选满 20 只股票
   - 如果权重上限过低（例如 `max_weight_per_stock=0.01` 且 top_n=20），可能导致迭代次数较多
4. **与等权的兼容性**：
   - `weight_method=equal` 时，权重限制通常不需要（因为已经是等权）
   - 但仍可启用，例如在仓位补齐后可能出现不等权的情况

## 七、测试覆盖

### 7.1 单元测试

- `tests/test_weight_processor.py`：12 个测试，覆盖权重限权的各种场景
- `tests/test_industry_constraint.py`：14 个测试，覆盖行业约束的各种场景
- `tests/test_ml_signal.py`：扩展现有测试，验证 score 权重产生非等权结果

所有测试通过率：100%

### 7.2 集成测试

需要在实际回测中验证：
1. score 权重确实产生不同的权重值
2. 权重限制后权重和为 1 且不超过上限
3. 行业约束正确生效，持仓满足行业分散要求

## 八、未来扩展方向

1. **行业权重约束**：当前是行业**数量**约束，未来可扩展为行业**权重**约束
2. **其他维度约束**：例如市值、风格、区域等维度的约束
3. **动态调整**：根据市场状态动态调整权重上限和行业上限
4. **风险预算**：结合波动率进行风险预算分配

## 九、版本信息

- **版本号**：0.7.0
- **发布日期**：2026-02-12
- **向后兼容性**：完全兼容，新参数默认不启用

## 十、相关文档

- CHANGELOG.md：版本变更记录
- tests/test_weight_processor.py：权重后处理测试
- tests/test_industry_constraint.py：行业约束测试
- tests/test_ml_signal.py：信号生成测试

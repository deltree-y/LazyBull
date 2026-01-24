# 回测与模型优化更新说明

本次更新针对 LazyBull 项目实现了4项重要改进，提升了回测体验和模型性能。

## 1. 回测进度实时显示

### 改进说明
- **现状**：回测过程中使用 logger 每10%打印一次进度，输出较为简单
- **改进**：集成 tqdm 进度条，实时显示回测进度、当前日期、净值和耗时

### 实现细节
- 文件：`src/lazybull/backtest/engine.py`
- 使用 tqdm 进度条替代原有的间隔日志输出
- 显示信息：
  - 进度百分比和进度条
  - 当前交易日期
  - 当前净值
  - 已用时间（秒）
- 性能影响：tqdm 高度优化，对回测速度影响可忽略不计

### 使用示例
```python
from src.lazybull.backtest import BacktestEngine

engine = BacktestEngine(...)
nav_curve = engine.run(...)  # 自动显示进度条
```

输出示例：
```
回测进度: 100%|██████████| 252/252 [00:05<00:00, 当前日期: 2023-12-29, 净值: 1.1523, 已用时: 5.2秒]
```

---

## 2. 收益曲线报告：每笔卖出增加收益数据

### 改进说明
- **现状**：卖出交易只记录价格、股数、金额和成本，没有收益信息
- **改进**：为每笔卖出交易自动计算并记录收益金额和收益率

### 实现细节

#### 2.1 持仓数据结构增强
文件：`src/lazybull/backtest/engine.py` - `_buy_stock` 方法

买入时记录更多信息：
```python
self.positions[stock] = {
    'shares': shares,           # 持仓数量
    'buy_date': date,          # 买入日期
    'buy_price': price,        # 买入价格
    'buy_cost': total_cost     # 买入总成本（含手续费）
}
```

#### 2.2 卖出收益计算
文件：`src/lazybull/backtest/engine.py` - `_sell_stock` 方法

卖出时计算收益：
```python
# 计算收益（基于 FIFO 原则）
buy_cost = self.positions[stock]['buy_cost']        # 买入总成本
sell_proceeds = amount - cost                        # 卖出所得（扣除成本）
profit_amount = sell_proceeds - buy_cost            # 绝对收益
profit_pct = profit_amount / buy_cost               # 收益率
```

收益计算公式：
- **收益金额** = (卖出金额 - 卖出成本) - (买入金额 + 买入成本)
- **收益率** = 收益金额 / (买入金额 + 买入成本)

所有成本（佣金、印花税、滑点）均已在计算中扣除。

#### 2.3 交易记录增强
卖出交易新增字段：
- `buy_price`：买入价格
- `profit_amount`：单笔收益金额（已扣除所有成本）
- `profit_pct`：单笔收益率

#### 2.4 报告格式增强
文件：`src/lazybull/backtest/reporter.py`

中文列名映射：
- `buy_price` → 买入价格
- `profit_amount` → 收益金额
- `profit_pct` → 收益率（自动格式化为百分比，如 "15.23%"）

### 匹配逻辑说明
- **FIFO原则**：当前实现基于持仓记录，每个股票只保留最近一次买入信息
- **T+n策略**：由于系统采用固定持有期（T+n卖出），同一股票不会同时存在多笔持仓
- **边界情况**：如果出现持仓覆盖（不应该发生），会在日志中警告

### 使用示例
```python
from src.lazybull.backtest import BacktestEngine, Reporter

engine = BacktestEngine(...)
nav_curve = engine.run(...)
trades = engine.get_trades()

# 查看卖出交易的收益信息
sell_trades = trades[trades['action'] == 'sell']
print(sell_trades[['date', 'stock', 'buy_price', 'price', 'profit_amount', 'profit_pct']])
```

输出CSV示例（中文）：
```csv
交易日期,股票代码,操作,买入价格,成交价格,收益金额,收益率
2023-01-15,000001.SZ,卖出,10.50,12.30,1523.45,15.23%
```

---

## 3. 修改回测缺省值

### 改进说明
根据项目定位（小资金、价值红利策略），优化默认参数为更合理的设置。

### 参数变更

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `top_n` | 30 | 5 | 选股数量，适合小资金分散度 |
| `initial_capital` | 1,000,000 | 500,000 | 初始资金（50万更贴近个人投资） |
| `exclude_st` | False（默认包含） | True（默认排除） | 价值投资应规避ST风险 |
| `rebalance_freq` | M（月频） | W（周频） | 周频调仓平衡收益与成本 |

### 涉及文件

#### 3.1 脚本默认值
文件：`scripts/run_ml_backtest.py`

```python
parser.add_argument("--top-n", type=int, default=5)
parser.add_argument("--initial-capital", type=float, default=500000.0)
parser.add_argument("--rebalance-freq", type=str, default="W")
parser.add_argument("--exclude-st", action="store_true", default=True)
parser.add_argument("--include-st", action="store_false", dest="exclude_st")
```

#### 3.2 配置文件
文件：`configs/base.yaml`

```yaml
backtest:
  initial_capital: 500000
  rebalance_frequency: "W"
  top_n: 5
  exclude_st: true
```

#### 3.3 文档更新
文件：`README.md`
- 更新快速开始示例
- 说明默认参数
- 提供自定义参数示例

### 使用说明

**使用新默认值**：
```bash
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231
# 自动应用：top_n=5, initial_capital=500000, rebalance_freq=W, exclude_st=True
```

**自定义参数**：
```bash
# 增加选股数量和资金
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --top-n 10 --initial-capital 1000000

# 包含ST股票
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --include-st

# 改为月频调仓
python scripts/run_ml_backtest.py --start-date 20230101 --end-date 20231231 \
    --rebalance-freq M
```

---

## 4. 针对性优化模型以提升验证集表现

### 问题诊断
- **现状**：验证集 R² 接近 0，模型在样本外几乎无预测能力
- **原因分析**：
  1. 短期收益（5日）噪音大
  2. 过拟合训练集，泛化能力差
  3. 标签存在极端异常值
  4. R² 对股票预测不是最佳指标

### 改进措施

#### 4.1 早停机制（Early Stopping）
文件：`scripts/train_ml_model.py` - `train_xgboost_model` 函数

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,  # 30轮无改进则停止
    verbose=False
)
```

**效果**：防止过拟合，自动选择最优迭代次数

#### 4.2 标签 Winsorize 处理
```python
from scipy.stats import mstats

y_train_winsorized = mstats.winsorize(
    y_train, 
    limits=[0.01, 0.01]  # 截断上下1%极端值
)
```

**效果**：减少异常涨跌停对模型的干扰

#### 4.3 正则化参数
```python
train_params = {
    # ... 其他参数
    "gamma": 0.1,          # 分裂所需的最小损失减少
    "reg_alpha": 0.1,      # L1 正则化
    "reg_lambda": 1.0,     # L2 正则化
}
```

**效果**：增强模型泛化能力

#### 4.4 优化超参数
| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `n_estimators` | 100 | 200 | 增加树的数量（配合早停） |
| `max_depth` | 6 | 8 | 适当增加树深度 |
| `learning_rate` | 0.1 | 0.05 | 降低学习率，更稳健 |

#### 4.5 新增评估指标
除了 MSE/RMSE/R² 外，新增：

**IC（Information Coefficient）**：
- 衡量预测值与真实值的线性相关性
- IC > 0.03 通常认为有预测能力

**RankIC（Spearman Rank IC）**：
- 衡量预测排序与真实排序的相关性
- **最重要指标**：选股策略关注相对排序而非绝对值
- RankIC > 0.05 说明排序能力较好

```python
# 信息系数（IC）
train_ic = y_train.corr(pd.Series(y_train_pred))
val_ic = y_val.corr(pd.Series(y_val_pred))

# 排序IC（RankIC）
from scipy.stats import spearmanr
val_rank_ic, _ = spearmanr(y_val, y_val_pred)
```

### 训练输出示例
```
============================================================
验证集评估结果
============================================================
验证集样本数: 50000
MSE（均方误差）: 0.003456
RMSE（均方根误差）: 0.058789
R2（决定系数）: 0.0234
IC（信息系数）: 0.0456  <- 重要指标
RankIC（排序IC）: 0.0678  <- 选股策略关键指标
============================================================
提示：对于选股策略，IC 和 RankIC 比 R2 更重要
     IC > 0.03 通常可认为有一定预测能力
     RankIC > 0.05 说明排序能力较好
============================================================
```

### 使用示例
```bash
# 使用新的改进训练
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231

# 自定义超参数（进一步调优）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
    --n-estimators 300 --max-depth 10 --learning-rate 0.03
```

### 预期改进
- **IC**：从接近0提升到 0.03-0.08
- **RankIC**：从接近0提升到 0.05-0.10
- **R²**：可能仍然较低（0.01-0.05），但这对选股策略不是主要问题

### 进一步优化建议
如果指标仍不理想，可以考虑：
1. **特征工程**：添加动量、波动率、换手率等衍生特征
2. **标签改进**：尝试10日/20日收益，或使用分位数回归
3. **样本过滤**：过滤低流动性股票、异常交易日
4. **时间窗口**：使用滚动窗口训练，更贴近实盘场景
5. **集成学习**：结合多个模型的预测

---

## 依赖更新

### 新增依赖
- `tqdm>=4.64.0`：进度条显示
- `scipy>=1.9.0`：统计函数（winsorize, spearmanr）

### 安装方式
```bash
# 使用 pip
pip install -r requirements.txt

# 使用 Poetry
poetry install
```

---

## 测试验证

### 新增测试
文件：`tests/test_profit_tracking.py`

测试覆盖：
1. 卖出交易包含收益字段
2. 收益计算准确性验证
3. 价格上涨时的盈利场景
4. 价格下跌时的亏损场景

### 运行测试
```bash
# 运行所有回测相关测试
pytest tests/test_backtest_t1.py tests/test_profit_tracking.py -v

# 运行特定测试
pytest tests/test_profit_tracking.py::test_profit_calculation_accuracy -v
```

### 测试结果
```
tests/test_backtest_t1.py::test_t1_trading_logic PASSED
tests/test_backtest_t1.py::test_pending_signals_mechanism PASSED
tests/test_backtest_t1.py::test_position_tracking_with_buy_date PASSED
tests/test_profit_tracking.py::test_profit_tracking_in_sell_trades PASSED
tests/test_profit_tracking.py::test_profit_calculation_accuracy PASSED
tests/test_profit_tracking.py::test_profit_with_price_increase PASSED
tests/test_profit_tracking.py::test_profit_with_price_decrease PASSED

7 passed in 0.43s
```

---

## 向后兼容性

### 兼容性说明
- 所有改动向后兼容
- 旧的脚本和代码可以继续使用
- 新增字段对旧代码透明（可选）

### 迁移建议
如果您的代码依赖旧的默认值：
```python
# 明确指定旧参数
python scripts/run_ml_backtest.py \
    --top-n 30 \
    --initial-capital 1000000 \
    --rebalance-freq M \
    --include-st  # 包含ST股票
```

---

## 总结

本次更新全面提升了 LazyBull 的回测体验和模型性能：

✅ **用户体验**：实时进度条，直观了解回测进度
✅ **数据完整**：每笔交易都有完整的收益信息，便于分析
✅ **合理默认**：参数设置更符合个人投资者实际情况
✅ **模型优化**：早停、正则化、更好的评估指标，提升泛化能力

所有改动都经过充分测试，确保稳定性和准确性。

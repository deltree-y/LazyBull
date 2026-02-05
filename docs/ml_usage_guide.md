# ML 信号生成与回测使用指南

本文档介绍如何使用 LazyBull 的机器学习模块进行模型训练和回测。

## 目录

1. [前置准备](#前置准备)
2. [模型训练](#模型训练)
3. [运行回测](#运行回测)
4. [查看结果](#查看结果)
5. [高级用法](#高级用法)

## 前置准备

### 1. 安装依赖

```bash
# 使用 pip 安装
pip install xgboost scikit-learn joblib

# 或使用 poetry
poetry add xgboost scikit-learn joblib
```

### 2. 准备特征数据

在训练模型之前，需要先构建特征数据：

```bash
# 构建指定日期区间的特征
python scripts/build_features.py --start-date 20230101 --end-date 20231231
```

特征数据将保存在 `data/features/cs_train/` 目录下，按交易日分区存储。

## 模型训练

### 基础训练

使用默认参数训练 XGBoost 模型：

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231
```

**训练过程：**
1. 自动读取指定日期区间的特征数据
2. 使用全量特征列训练 XGBoost 回归模型
3. 标签为 `y_ret_5`（未来 5 个交易日的收益率）
4. 自动过滤不可交易样本（ST、停牌、涨跌停等）
5. 模型保存到 `data/models/` 目录
6. 版本号自动递增（v1, v2, v3...）
7. 元数据记录到 `model_registry.json`

**输出示例：**
```
2024-01-17 15:30:00.000 | INFO     | 加载特征数据: 20230101 至 20231231
2024-01-17 15:30:05.000 | INFO     | 成功加载 250000 条样本
2024-01-17 15:30:06.000 | INFO     | 特征列数量: 45
2024-01-17 15:30:06.000 | INFO     | 过滤后样本数: 180000 / 250000
2024-01-17 15:30:10.000 | INFO     | 开始训练 XGBoost 模型...
2024-01-17 15:32:00.000 | INFO     | 模型训练完成
2024-01-17 15:32:00.000 | INFO     | 训练集性能: MSE=0.001234, RMSE=0.035128, R2=0.4567
2024-01-17 15:32:01.000 | INFO     | 模型已保存: data/models/v1_model.joblib
2024-01-17 15:32:01.000 | INFO     | 模型已注册: v1, 类型=xgboost, 训练区间=20230101至20231231
```

### 自定义超参数

可以自定义 XGBoost 的超参数：

```bash
python scripts/train_ml_model.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --n-estimators 200 \
    --max-depth 5 \
    --learning-rate 0.05 \
    --subsample 0.8 \
    --colsample-bytree 0.8
```

**可用参数：**
- `--n-estimators`: 树的数量（默认 100）
- `--max-depth`: 树的最大深度（默认 6）
- `--learning-rate`: 学习率（默认 0.1）
- `--subsample`: 样本采样比例（默认 0.8）
- `--colsample-bytree`: 特征采样比例（默认 0.8）
- `--random-state`: 随机种子（默认 42）

### 查看已训练模型

训练完成后，可以查看模型文件：

```bash
# 查看模型目录
ls -lh data/models/

# 输出示例：
# v1_model.joblib         # 模型文件
# v1_features.json        # 特征列表
# v2_model.joblib
# v2_features.json
# model_registry.json     # 模型注册表
```

查看模型注册表：

```bash
cat data/models/model_registry.json
```

**注册表格式：**
```json
{
  "models": [
    {
      "version": 1,
      "version_str": "v1",
      "model_type": "xgboost",
      "model_file": "v1_model.joblib",
      "features_file": "v1_features.json",
      "train_start_date": "20230101",
      "train_end_date": "20231231",
      "feature_count": 45,
      "label_column": "y_ret_5",
      "n_samples": 180000,
      "train_params": {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
      },
      "performance_metrics": {
        "mse": 0.001234,
        "rmse": 0.035128,
        "r2": 0.4567
      },
      "created_at": "2024-01-17 15:32:01"
    }
  ],
  "next_version": 2
}
```

## 运行回测

### 基础回测

使用最新训练的模型运行回测：

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231
```

**回测过程：**
1. 自动加载最新版本的模型
2. 读取回测期间的特征数据和价格数据
3. 每个调仓日使用模型预测，选择 Top N 股票
4. 等权配置或按预测分数加权
5. 计算组合净值和收益指标
6. 生成回测报告

**输出示例：**
```
2024-01-17 15:35:00.000 | INFO     | 模型已加载: v1, 训练区间=20230101至20231231
2024-01-17 15:35:00.000 | INFO     | 特征数: 45
2024-01-17 15:35:05.000 | INFO     | 开始运行 ML 信号回测...
2024-01-17 15:35:30.000 | INFO     | 回测完成: 共 252 个交易日, 12 笔交易
============================================================
回测报告摘要
============================================================
总收益率      : 18.45%
年化收益率    : 18.75%
最大回撤      : -6.23%
波动率        : 15.30%
夏普比率      : 1.03
交易次数      : 12
总交易成本    : 8234.56元
回测天数      : 252
起始净值      : 1.0000
结束净值      : 1.1845
============================================================
```

### 指定模型版本

可以指定使用特定版本的模型：

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --model-version 1
```

### 自定义 Top N

选择不同数量的股票：

```bash
# 选择 Top 50 只股票
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 50

# 选择 Top 20 只股票
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 20
```

### 调整调仓频率

支持日度、周度、月度调仓：

```bash
# 月度调仓（默认）
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --rebalance-freq M

# 周度调仓
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --rebalance-freq W

# 日度调仓
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --rebalance-freq D
```

### 按预测分数加权

除了等权配置，还可以按预测分数加权：

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --weight-method score
```

- `equal`（默认）：Top N 股票等权配置
- `score`：按预测分数加权，预测收益越高权重越大

## 查看结果

回测完成后，报告文件保存在 `data/reports/` 目录：

```bash
ls -lh data/reports/

# 输出：
# ml_backtest_nav.csv              # 净值曲线
# ml_backtest_trades.csv           # 交易记录（本次回测）
# ml_backtest_trades_runs.csv      # 交易记录累加文件（所有历史回测）
# ml_backtest_stats.txt            # 统计指标
# backtest_runs.csv                # 回测运行记录
```

### 净值曲线

```bash
head data/reports/ml_backtest_nav.csv
```

**格式：**
```csv
date,portfolio_value,capital,market_value,nav,return
2023-01-03,1000000.00,0.00,1000000.00,1.0000,0.0000
2023-01-04,1005000.00,50000.00,955000.00,1.0050,0.0050
2023-01-05,1012000.00,50000.00,962000.00,1.0120,0.0120
...
```

### 交易记录

回测系统生成两个交易记录文件：

#### 1. 本次回测交易记录（ml_backtest_trades.csv）

每次回测会**覆盖**此文件，只保留最近一次回测的交易明细：

```bash
head data/reports/ml_backtest_trades.csv
```

**格式：**
```csv
date,stock,action,price,shares,amount,cost
2023-01-03,000001.SZ,buy,15.30,6500,99450.00,29.84
2023-01-03,000002.SZ,buy,28.50,3500,99750.00,29.93
2023-02-01,000001.SZ,sell,16.20,6500,105300.00,105.30
...
```

#### 2. 累加历史交易记录（ml_backtest_trades_runs.csv）

每次回测会**追加**交易记录到此文件，保留所有历史回测的交易明细，并标注每次回测的核心参数：

```bash
head data/reports/ml_backtest_trades_runs.csv
```

**格式示例：**
```csv
回测ID,回测时间,开始日期,结束日期,模型版本,TopN,权重方法,调仓频率,初始资金,卖出时机,交易日期,股票代码,操作,成交价格,成交股数,成交金额,交易成本,买入价格,收益金额,收益率
20240117153201_a1b2c3d4,2024-01-17 15:32:01,20230101,20231231,最新版本,5,equal,10,500000.0,open,2023-01-03,000001.SZ,买入,15.30,6500,99450.00,29.84,,,
20240117153201_a1b2c3d4,2024-01-17 15:32:01,20230101,20231231,最新版本,5,equal,10,500000.0,open,2023-01-03,000002.SZ,买入,28.50,3500,99750.00,29.93,,,
20240117153201_a1b2c3d4,2024-01-17 15:32:01,20230101,20231231,最新版本,5,equal,10,500000.0,open,2023-02-01,000001.SZ,卖出,16.20,6500,105300.00,105.30,15.30,5850.00,5.88%
...
```

**累加文件的核心参数列说明：**

| 列名 | 说明 | 示例 |
|------|------|------|
| 回测ID | 唯一标识本次回测（时间戳+参数hash） | 20240117153201_a1b2c3d4 |
| 回测时间 | 回测执行时间 | 2024-01-17 15:32:01 |
| 开始日期 | 回测开始日期 | 20230101 |
| 结束日期 | 回测结束日期 | 20231231 |
| 模型版本 | 使用的模型版本号（未指定时显示"最新版本"） | 最新版本 或 1 |
| TopN | 选择的股票数量 | 5 |
| 权重方法 | 权重分配方法 | equal 或 score |
| 调仓频率 | 调仓频率（交易日数） | 10 |
| 初始资金 | 初始资金 | 500000.0 |
| 卖出时机 | 卖出时机 | open 或 close |

**收益计算说明：**

累加文件中的"收益金额"和"收益率"列仅在**卖出**记录中有值，买入记录中为空。

计算规则：
- **买入成本** = 买入金额 + 买入时交易成本
- **收益金额** = 卖出金额 - 买入成本 - 卖出时交易成本
- **收益率** = (收益金额 / 买入成本) × 100%（保留2位小数）

**FIFO 配对规则：**

当同一只股票多次买卖时，采用 **FIFO（先进先出）** 配对规则：
- 第一次卖出匹配第一次买入
- 第二次卖出匹配第二次买入
- 以此类推

示例：
```
买入1: 000001.SZ @ 10.00元, 金额10000, 成本30
买入2: 000001.SZ @ 11.00元, 金额11000, 成本33
卖出1: 000001.SZ @ 12.00元, 金额12000, 成本60  → 匹配买入1，收益 = 12000 - (10000+30) - 60 = 1910
卖出2: 000001.SZ @ 12.50元, 金额12500, 成本62.5 → 匹配买入2，收益 = 12500 - (11000+33) - 62.5 = 1404.5
```

**用途说明：**

- `ml_backtest_trades.csv`：适合查看和分析单次回测的交易明细
- `ml_backtest_trades_runs.csv`：适合：
  - 对比不同参数配置的交易差异
  - 分析参数对交易行为的影响
  - 追溯历史回测的完整交易记录
  - 进行跨回测的交易统计分析
  - 评估每笔交易的实际盈亏情况

### 统计指标

```bash
cat data/reports/ml_backtest_stats.txt
```

## 高级用法

### Python API 使用

除了命令行脚本，也可以在 Python 代码中使用：

```python
from src.lazybull.signals import MLSignal
from src.lazybull.ml import ModelRegistry

# 1. 查看已训练的模型
registry = ModelRegistry(models_dir="./data/models")
models = registry.list_models()
for model in models:
    print(f"版本: {model['version_str']}, "
          f"样本数: {model['n_samples']}, "
          f"R2: {model['performance_metrics']['r2']}")

# 2. 创建 ML 信号
signal = MLSignal(
    top_n=30,
    model_version=None,  # None 表示使用最新版本
    models_dir="./data/models",
    weight_method="equal"
)

# 3. 获取模型信息
model_info = signal.get_model_info()
print(f"使用模型: {model_info['version_str']}")
print(f"特征数: {model_info['feature_count']}")

# 4. 生成信号
import pandas as pd
date = pd.Timestamp("2023-06-15")
universe = ["000001.SZ", "000002.SZ", "600000.SH", ...]
features_df = ...  # 加载当日特征数据

signals = signal.generate_with_features(date, universe, features_df)
print(f"选择股票: {list(signals.keys())}")
print(f"权重: {list(signals.values())}")
```

### 批量训练和对比

可以训练多个模型并对比效果：

```bash
# 训练模型1：默认参数
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20230630

# 训练模型2：增加树的数量
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20230630 \
    --n-estimators 200

# 训练模型3：减小学习率
python scripts/train_ml_model.py \
    --start-date 20230101 --end-date 20230630 \
    --n-estimators 200 --learning-rate 0.05

# 对比回测效果
for version in 1 2 3; do
    python scripts/run_ml_backtest.py \
        --start-date 20230701 --end-date 20231231 \
        --model-version $version \
        --output-name "ml_backtest_v${version}"
done
```

## 注意事项

1. **数据要求**：
   - 确保特征数据已构建完成
   - 训练区间和回测区间应该分开（避免前视偏差）
   - 特征数据需要包含 `y_ret_5` 标签列

2. **模型选择**：
   - 训练集 R2 不宜过高（可能过拟合）
   - 建议在验证集上评估模型效果
   - 定期重新训练模型以适应市场变化

3. **回测假设**：
   - 使用收盘价成交
   - 包含交易成本（佣金、印花税、滑点）
   - 不考虑涨跌停、停牌等限制（特征数据中已过滤）

4. **性能优化**：
   - 特征数据按日分区存储，加载速度较快
   - 模型使用 joblib 序列化，加载速度快
   - 大规模回测建议使用分布式计算

## 故障排查

### 问题1：没有特征数据

**错误信息：**
```
ValueError: 指定日期区间内没有特征数据
```

**解决方法：**
```bash
# 先构建特征数据
python scripts/build_features.py --start-date 20230101 --end-date 20231231
```

### 问题2：特征列不匹配

**错误信息：**
```
KeyError: 特征列缺失: 'feature_name'
```

**解决方法：**
- 确保使用相同的特征构建流程
- 使用模型训练时的相同特征列

### 问题3：没有已注册的模型

**错误信息：**
```
ValueError: 没有已注册的模型
```

**解决方法：**
```bash
# 先训练模型
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231
```

## 更多信息

- 项目主页：https://github.com/deltree-y/LazyBull
- 文档目录：`docs/`
- 问题反馈：https://github.com/deltree-y/LazyBull/issues

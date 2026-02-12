# 统一评估面板（CSV输出）功能实现

## PR 概述

本 PR 在 `scripts/run_ml_backtest.py` 中实现了"统一评估面板"功能，用于评估 ML 选股信号的预测质量，并输出 CSV 文件便于横向对比不同实验配置。

## 背景与目标

### 问题
- 当前 `run_ml_backtest.py` 能输出回测报告（净值曲线、交易记录、统计文本等）
- 但缺少对 ML 选股信号本身"预测质量"的统一评估与可横向对比的 CSV 面板
- 需要能够快速对比不同标签 horizon（y_ret_5/10/20）、调仓频率、模型版本、TopN、权重方法等实验配置的预测效果

### 解决方案
在回测脚本中增加评估面板导出能力：
- 按日评估 MLSignal 的截面排序质量
- 将结果输出为 CSV 文件
- 支持多种评估指标：RankIC、TopK 收益、分组收益、多空收益等

## 功能特性

### 1. 新增命令行参数

```bash
--export-eval           # 是否导出评估面板（默认开启）
--no-export-eval        # 禁用评估面板导出
--eval-groups N         # 分组数量（默认 10）
--eval-topk K           # TopK 指标的 K（默认使用 --top-n）
```

### 2. 评估口径

**真实收益标签来源**：
- 直接使用 features 文件（`data/features/cs_train/YYYYMMDD.parquet`）中的 label 列
- 支持 `y_ret_5`、`y_ret_10`、`y_ret_20` 三种 horizon

**分组方式**：
- 每天按预测分数排序后"等数量分组"（而不是分位数阈值）
- 默认 10 组
- 前几组可能多 1 个样本（如果总数不能被组数整除）

### 3. 输出文件

评估面板会生成三个 CSV 文件（存放在 `data/reports/` 目录）：

#### 3.1 日度评估 CSV（`{output_name}_eval_daily.csv`）

每行代表一个交易日的评估指标：

| 字段 | 说明 |
|------|------|
| 交易日期 | 交易日期（YYYYMMDD） |
| 样本数 | 参与评估的股票数量 |
| RankIC | Spearman 秩相关系数（预测分数 vs 真实收益） |
| TopK平均收益 | 预测分数最高的 K 只股票的平均真实收益 |
| Top组平均收益 | 第1组（预测分数最高组）的平均真实收益 |
| Bottom组平均收益 | 第N组（预测分数最低组）的平均真实收益 |
| 多空收益 | Top组 - Bottom组 的收益差 |

**用途**：
- 观察每日的预测质量波动
- 识别预测失效的时间段
- 计算时序统计指标（如 IC 均值、IR 等）

#### 3.2 分组收益明细 CSV（`{output_name}_eval_groups.csv`）

每行代表一个交易日×组号的组合：

| 字段 | 说明 |
|------|------|
| 交易日期 | 交易日期（YYYYMMDD） |
| 组号 | 分组编号（1=最高分，N=最低分） |
| 组内股票数 | 该组包含的股票数量 |
| 组内平均真实收益 | 该组的平均真实收益 |
| 组内平均预测分数 | 该组的平均预测分数 |

**用途**：
- 分析预测分数的单调性（分数越高，收益是否越高）
- 绘制分组收益曲线
- 检查是否存在"反转"效应（低分组反而收益高）

#### 3.3 汇总 CSV（`{output_name}_eval_summary.csv`）

每行代表一次回测运行的汇总指标（支持多次运行累加）：

| 字段 | 说明 |
|------|------|
| 回测时间 | 回测执行时间 |
| 开始日期 | 回测开始日期 |
| 结束日期 | 回测结束日期 |
| 标签列 | 使用的标签（y_ret_5/y_ret_10/y_ret_20） |
| 模型版本 | 模型版本号（或"最新版本"） |
| TopN | 选股数量 |
| 权重方法 | 权重分配方法（equal/score） |
| 调仓频率 | 调仓频率（交易日数） |
| 初始资金 | 初始资金 |
| 卖出时机 | 卖出时机（open/close） |
| 分组数 | 评估分组数量 |
| TopK | TopK 指标的 K 值 |
| 评估天数 | 实际评估的交易日数量 |
| RankIC均值 | RankIC 的平均值 |
| RankIC标准差 | RankIC 的标准差 |
| RankIC_IR | Information Ratio（IC均值/IC标准差） |
| TopK平均收益 | TopK 收益的平均值 |
| 多空平均收益 | 多空收益的平均值 |

**用途**：
- 横向对比不同配置的实验结果
- 快速筛选最优参数组合
- 生成实验对比表格

## 实现细节

### 核心函数

#### 1. `equal_count_grouping(scores, n_groups=10)`
按预测分数等数量分组：
- 输入：预测分数 Series
- 输出：分组标签 Series（1=最高分组，n_groups=最低分组）
- 逻辑：先按分数降序排序，然后将样本平均分配到 n_groups 个组

#### 2. `evaluate_daily(date, signal, universe, features_df, label_column, n_groups, topk)`
评估单日的 ML 信号质量：
- 使用 `MLSignal.generate_ranked()` 得到排序候选
- 将候选与当日 features_df 的 label 列对齐
- 计算 RankIC（Spearman 相关）
- 等数量分组并计算每组的平均真实收益
- 返回日度指标和分组明细

#### 3. `export_evaluation_panel(...)`
导出评估面板 CSV：
- 遍历所有交易日
- 调用 `evaluate_daily()` 进行评估
- 汇总所有日度指标和分组明细
- 写入三个 CSV 文件（日度、分组、汇总）
- 输出使用 `utf-8-sig` 编码（Excel 兼容）

### 代码复用

- **复用 `MLSignal.generate_ranked()`**：获取排序候选
- **复用 `Storage.load_cs_train_day()`**：加载特征数据
- **复用 `_append_dict_to_csv()`**：CSV 追加写入
- **复用现有股票池和交易日列表**：无需重新加载

### 兼容性

- **不影响现有回测流程**：评估面板是可选功能，默认开启
- **可通过 `--no-export-eval` 禁用**：不需要评估时可以关闭
- **与现有报告共存**：Reporter 输出不变，评估 CSV 作为新增输出

## 使用示例

### 基本用法（使用默认参数）

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 30
```

默认会生成：
- `data/reports/ml_backtest_eval_daily.csv`
- `data/reports/ml_backtest_eval_groups.csv`
- `data/reports/ml_backtest_eval_summary.csv`

### 自定义评估参数

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --top-n 30 \
    --eval-groups 20 \
    --eval-topk 50 \
    --output-name my_backtest
```

会生成：
- `data/reports/my_backtest_eval_daily.csv`（使用 20 组）
- `data/reports/my_backtest_eval_groups.csv`（使用 20 组）
- `data/reports/my_backtest_eval_summary.csv`（TopK=50）

### 禁用评估面板

```bash
python scripts/run_ml_backtest.py \
    --start-date 20230101 \
    --end-date 20231231 \
    --no-export-eval
```

不会生成评估 CSV 文件。

### 对比不同 horizon

```bash
# 实验1：y_ret_5
python scripts/run_ml_backtest.py \
    --start-date 20230101 --end-date 20231231 \
    --label y_ret_5 --rebalance-freq 5 \
    --output-name exp_horizon_5

# 实验2：y_ret_10
python scripts/run_ml_backtest.py \
    --start-date 20230101 --end-date 20231231 \
    --label y_ret_10 --rebalance-freq 10 \
    --output-name exp_horizon_10

# 实验3：y_ret_20
python scripts/run_ml_backtest.py \
    --start-date 20230101 --end-date 20231231 \
    --label y_ret_20 --rebalance-freq 20 \
    --output-name exp_horizon_20
```

然后查看 `data/reports/*_eval_summary.csv` 对比三个实验的 RankIC、IR、多空收益等指标。

## 测试

新增测试文件：`tests/test_eval_panel.py`

测试覆盖：
- ✅ 等数量分组切分正确（总数/组大小分布）
- ✅ RankIC 在简单可控数据上结果正确（完美正相关=1.0，负相关=-1.0）
- ✅ 日度评估函数能正确计算指标
- ✅ CSV 文件能被生成且包含预期列

运行测试：
```bash
pytest tests/test_eval_panel.py -v
```

## 版本更新

- 版本号从 `0.5.0` 更新到 `0.6.0`（功能增强，小版本升级）
- 更新 `pyproject.toml`
- 更新 `CHANGELOG.md`

## 文档

- ✅ `docs/PR/unified_eval_panel_csv_output.md`（本文档）
- ✅ `docs/guide/ml_eval_panel_guide.md`（使用指南）
- ✅ `CHANGELOG.md` 更新

## 注意事项

1. **CSV 编码**：所有 CSV 文件使用 `utf-8-sig` 编码，Excel 可直接打开
2. **缺失标签处理**：样本的 label 缺失时会被过滤，不参与评估
3. **评估开销**：评估过程会遍历所有交易日，对于长周期回测可能需要一些时间
4. **汇总 CSV 累加**：`*_eval_summary.csv` 支持多次运行累加，方便批量实验对比

## 未来扩展

可选的未来增强方向：
- 支持更多评估指标（如分位数收益、胜率、最大连续亏损等）
- 支持自定义分组边界（如按分位数阈值分组）
- 支持行业中性化后的评估
- 支持可视化输出（如分组收益曲线图、IC 时序图等）

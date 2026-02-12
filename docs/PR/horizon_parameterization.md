# PaperTradingRunner horizon 参数化

## 概述

本 PR 实现了 `PaperTradingRunner` 的 `horizon` 参数化，使纸面交易的特征构建周期可以通过配置灵活指定，不再固定为 5 天。

## 背景

在之前的实现中，`PaperTradingRunner` 的 `FeatureBuilder` 硬编码 `horizon=5`，导致以下问题：

1. **灵活性不足**：无法根据实际需求切换到不同的预测周期（如 10 天、20 天）
2. **训练回测不一致**：如果训练模型使用 `horizon=10` 或 `horizon=20` 生成的标签，但纸面交易仍使用 `horizon=5` 构建特征，会导致特征构建口径不一致
3. **调试困难**：当特征不匹配时难以快速定位问题

### 原有代码

```python
# src/lazybull/paper/runner.py (line 76)
self.feature_builder = FeatureBuilder(min_list_days=60, horizon=5, require_label=False)
```

## 解决方案

### 1. PaperTradingRunner 参数化

在 `PaperTradingRunner.__init__` 增加 `horizon` 参数：

```python
def __init__(
    self,
    signal: Optional[Signal] = None,
    initial_capital: float = 500000.0,
    data_root: str = "./data",
    paper_root: str = "./data/paper",
    weight_method: str = "equal",
    horizon: int = 5,  # 新增参数
    verbose: bool = True,
):
    """初始化运行器
    
    Args:
        signal: 信号生成器（可选）
        initial_capital: 初始资金
        data_root: 数据根目录
        paper_root: 纸面交易数据目录
        weight_method: 权重分配方法，"equal"表示等权，"score"表示按分数加权
        horizon: 特征构建的预测周期（天数），用于生成 y_ret_N 特征，默认 5
        verbose: 是否输出详细日志
    """
    # ...
    self.feature_builder = FeatureBuilder(
        min_list_days=60, 
        horizon=horizon,  # 使用参数
        require_label=False
    )
    self.horizon = horizon  # 保存供其他地方使用
```

**关键变更**：
- 添加 `horizon: int = 5` 参数，默认值为 5，保持向后兼容
- 将 `horizon` 传递给 `FeatureBuilder`
- 保存 `self.horizon` 属性供后续使用

### 2. 配置支持

在 `scripts/paper_trade.py` 的配置系统中增加 `horizon` 支持：

#### 2.1 配置命令（config）

添加 `--horizon` 命令行参数：

```python
config_parser.add_argument(
    '--horizon',
    type=int,
    default=5,
    help='特征构建的预测周期（天数），用于生成 y_ret_N 特征（默认：5）'
)
```

#### 2.2 配置保存

在 `run_config` 函数中保存 `horizon` 到配置文件：

```python
config = {
    # ... 其他配置项
    'horizon': args.horizon,
    # ...
}
```

#### 2.3 配置读取与使用

在 `run_main` 函数中读取配置并传递给 `PaperTradingRunner`：

```python
# 设置默认 horizon，如果配置中不存在
if 'horizon' not in config:
    config['horizon'] = 5

logger.info("使用配置：")
logger.info(f"  特征预测周期（horizon）: {config['horizon']} 天")

# 创建运行器时传入 horizon
runner = PaperTradingRunner(
    initial_capital=config['initial_capital'],
    weight_method=config['weight_method'],
    horizon=config['horizon'],  # 传入 horizon
)
```

### 3. 调用点更新

#### 调用点 1：`scripts/paper_trade.py` - `run_main` 函数（主要执行流程）

**修改前**：
```python
runner = PaperTradingRunner(
    initial_capital=config['initial_capital'],
    weight_method=config['weight_method'],
)
```

**修改后**：
```python
runner = PaperTradingRunner(
    initial_capital=config['initial_capital'],
    weight_method=config['weight_method'],
    horizon=config['horizon'],  # 从配置读取
)
```

#### 调用点 2：`scripts/paper_trade.py` - `print_positions` 函数（查看持仓）

**保持不变**：
```python
runner = PaperTradingRunner(verbose=False)
```

该调用点仅用于读取持仓信息，不涉及特征构建，使用默认 `horizon=5` 即可。

#### 其他调用点：测试文件

测试文件中的 `PaperTradingRunner` 实例化保持不变，均使用默认 `horizon=5`：
- `tests/test_ensure_and_t0_printing.py`
- `tests/test_replenishment_no_sell.py`
- `tests/test_equal_weight_lot_constraint.py`

## 使用方法

### 1. 配置 horizon

使用 `config` 子命令设置 horizon：

```bash
# 设置 horizon 为 10 天
python scripts/paper_trade.py config \
    --buy-price close \
    --sell-price close \
    --top-n 5 \
    --initial-capital 500000 \
    --rebalance-freq 5 \
    --weight-method equal \
    --horizon 10
```

### 2. 运行纸面交易

配置保存后，运行纸面交易时会自动使用配置的 horizon：

```bash
python scripts/paper_trade.py run --trade-date 20260212
```

日志中会输出当前使用的 horizon：

```
使用配置：
  买入价格类型: close
  卖出价格类型: close
  持仓数: 5
  调仓频率: 5 个交易日
  权重方法: equal
  特征预测周期（horizon）: 10 天
  止损开关: False
  ECT开关: False
```

### 3. 查看持仓

查看持仓命令不受影响：

```bash
python scripts/paper_trade.py positions --trade-date 20260212
```

## 影响范围

### 代码变更

1. **src/lazybull/paper/runner.py**
   - `__init__` 方法增加 `horizon` 参数（1 行）
   - 更新 `FeatureBuilder` 初始化，使用参数化的 `horizon`（1 行）
   - 保存 `self.horizon` 属性（1 行）

2. **scripts/paper_trade.py**
   - `run_config` 函数：配置字典增加 `horizon` 项（1 行）
   - `run_main` 函数：添加默认值处理和日志输出（4 行）
   - `run_main` 函数：`PaperTradingRunner` 实例化增加 `horizon` 参数（1 行）
   - argparse 配置：增加 `--horizon` 参数（6 行）

3. **pyproject.toml**
   - 版本号从 0.6.0 升级到 0.6.1

### 向后兼容性

**完全向后兼容**：
- `horizon` 参数默认值为 5，与原有硬编码值一致
- 未配置 `horizon` 时自动使用默认值 5
- 测试文件无需修改，自动使用默认值
- 旧的配置文件（不含 `horizon` 字段）仍可正常使用

### 用户体验改进

1. **灵活性提升**：用户可根据训练模型的 horizon 设置纸面交易的 horizon
2. **日志透明**：运行时明确显示当前使用的 horizon 值，便于问题排查
3. **配置统一**：horizon 与其他参数（如 top_n、rebalance_freq）统一通过配置管理

## 测试验证

### 1. 功能测试

- **测试场景 1**：使用默认 horizon=5
  ```bash
  python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5
  # 未指定 --horizon，使用默认值 5
  ```

- **测试场景 2**：指定 horizon=10
  ```bash
  python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5 --horizon 10
  # 显式指定 horizon=10
  ```

- **测试场景 3**：指定 horizon=20
  ```bash
  python scripts/paper_trade.py config --buy-price close --sell-price close --top-n 5 --horizon 20
  # 显式指定 horizon=20
  ```

### 2. 兼容性测试

- 所有现有测试用例保持通过（使用默认 horizon=5）
- 旧的配置文件（不含 horizon 字段）可正常加载并使用默认值

## 注意事项

### 1. horizon 与训练模型一致

**重要**：纸面交易的 `horizon` 应与训练模型时使用的 `horizon` 保持一致。

例如：
- 如果模型使用 `y_ret_10` 标签训练（horizon=10），则纸面交易应设置 `--horizon 10`
- 如果模型使用 `y_ret_20` 标签训练（horizon=20），则纸面交易应设置 `--horizon 20`

**不一致的后果**：
- 特征构建逻辑不同，可能导致模型输入特征与训练时不匹配
- 预测效果可能下降

### 2. require_label=False

纸面交易的 `FeatureBuilder` 始终使用 `require_label=False`，因为：
- 纸面交易是实时的，没有未来数据无法生成标签（y_ret_N）
- 仅用于生成特征，不需要生成标签列

### 3. 现有配置兼容

如果配置文件中没有 `horizon` 字段：
- 自动使用默认值 5
- 不需要重新运行 `config` 命令（除非希望显式设置 horizon）

## 相关文件

### 配置文件

配置持久化在 `./data/paper/config.yaml`（由 `PaperStorage` 管理）：

```yaml
buy_price: close
sell_price: close
top_n: 5
initial_capital: 500000
rebalance_freq: 5
weight_method: equal
horizon: 10  # 新增字段
# ... 其他配置项
```

### 基础配置

项目基础配置 `configs/base.yaml` 不需要修改（该文件主要用于回测等场景）。

## 版本信息

- **版本号**：0.6.0 → 0.6.1
- **发布日期**：2026-02-12
- **向后兼容**：是
- **破坏性变更**：无

## 后续改进建议

1. **配置验证**：可以在配置时验证 horizon 的有效范围（如 1-252 交易日）
2. **模型 horizon 自动同步**：如果使用 ML 模型，可考虑从模型元数据中自动读取 horizon
3. **文档完善**：在用户指南中增加 horizon 选择建议（如何根据交易策略选择 horizon）
4. **回测对齐**：考虑在回测配置中也支持多 horizon，与纸面交易保持一致

## 参考资料

- 相关 Issue：horizon 写死导致训练回测不一致
- 相关文档：`docs/paper_trading_guide.md`
- 测试文件：`tests/test_multi_horizon_labels.py`（多 horizon 标签测试）

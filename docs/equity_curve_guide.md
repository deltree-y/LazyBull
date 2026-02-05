# 权益曲线交易（ECT）功能实现文档

## 概述

权益曲线交易（Equity Curve Trading, ECT）是一种基于账户盈亏曲线的仓位/风险管理策略。本功能在 v0.3.5 版本中实现，同时支持纸面交易和回测。

## 核心概念

### 1. 回撤分档控制

根据净值回撤程度，自动降低仓位：

- **回撤阈值**：例如 [5%, 10%, 15%, 20%]
- **对应仓位系数**：例如 [0.8, 0.6, 0.4, 0.2]
- **逻辑**：回撤越大，仓位越低

示例：
- 回撤 < 5%：满仓（系数 1.0）
- 回撤 5%-10%：降至 80%（系数 0.8）
- 回撤 10%-15%：降至 60%（系数 0.6）
- 回撤 15%-20%：降至 40%（系数 0.4）
- 回撤 > 20%：降至 20%（系数 0.2）

### 2. 均线趋势过滤

基于净值短期/长期均线判断趋势：

- **短期均线 > 长期均线**：趋势向上，允许持仓（系数 1.0）
- **短期均线 < 长期均线**：趋势向下，降低仓位（系数 0.5）

默认窗口：短期 5 天，长期 20 天

### 3. 逐步恢复机制

风险解除后，仓位不立即满仓，而是阶梯式回升：

**Gradual 模式（默认）**：
- 每个调仓周期增加固定步长（默认 0.1）
- 支持恢复前等待周期（默认 1 个周期）
- 避免过早增仓

**Immediate 模式**：
- 立即恢复到目标仓位
- 适合快速反应的策略

### 4. 组合决策

最终仓位系数 = min(回撤系数, 均线系数)

采用更保守的策略，两个条件都满足才允许较高仓位。

## 实现架构

### 模块结构

```
src/lazybull/risk/
├── __init__.py           # 导出 ECT 相关类
├── equity_curve.py       # ECT 核心实现
└── stop_loss.py          # 止损功能（已有）

scripts/
├── paper_trade.py        # 纸面交易脚本（已集成 ECT）
└── run_ml_backtest.py    # 回测脚本（已集成 ECT）

src/lazybull/backtest/
└── engine.py             # 回测引擎（已集成 ECT）
```

### 核心类

#### EquityCurveConfig

配置类，包含所有 ECT 参数：

```python
@dataclass
class EquityCurveConfig:
    enabled: bool = False  # 是否启用
    
    # 回撤控制
    drawdown_thresholds: List[float] = [5.0, 10.0, 15.0, 20.0]
    exposure_levels: List[float] = [0.8, 0.6, 0.4, 0.2]
    
    # 均线趋势
    ma_short_window: int = 5
    ma_long_window: int = 20
    ma_exposure_on: float = 1.0
    ma_exposure_off: float = 0.5
    
    # 恢复策略
    recovery_mode: str = "gradual"
    recovery_step: float = 0.1
    recovery_delay_periods: int = 1
    
    # 限制
    min_exposure: float = 0.0
    max_exposure: float = 1.0
```

#### EquityCurveMonitor

监控器类，负责计算仓位系数：

```python
class EquityCurveMonitor:
    def calculate_exposure(
        self,
        nav_history: pd.Series,
        current_date: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        计算当前仓位系数
        
        Args:
            nav_history: 历史 NAV 序列
            current_date: 当前日期（用于日志）
            
        Returns:
            (exposure_multiplier, reason)
            - exposure_multiplier: 0.0-1.0 的仓位系数
            - reason: 计算原因的中文描述
        """
```

### 集成点

#### 纸面交易

在 `scripts/paper_trade.py` 的 T0 调仓点集成：

1. 从 `data/paper/nav/nav.parquet` 加载历史 NAV
2. 创建 EquityCurveMonitor 并计算 exposure
3. 将 exposure 应用到目标权重：`target_weight *= exposure`
4. 重新保存调整后的目标权重
5. 输出 ECT 信息到日志

#### 回测

在 `src/lazybull/backtest/engine.py` 的买入执行点集成：

1. 在 `_execute_pending_buys` 方法中
2. 从 `portfolio_values` 构建历史 NAV 序列
3. 创建 EquityCurveMonitor 并计算 exposure
4. 将 exposure 应用到所有信号权重：`signals = {s: w * exposure for s, w in signals.items()}`
5. 输出 ECT 信息到日志（如果 verbose=True）

## 使用指南

### 纸面交易

#### 1. 配置 ECT 参数

```bash
python scripts/paper_trade.py config \
  --buy-price close --sell-price close \
  --top-n 5 --initial-capital 500000 --rebalance-freq 5 \
  --equity-curve-enabled \
  --equity-curve-drawdown-thresholds 5.0 10.0 15.0 \
  --equity-curve-exposure-levels 0.8 0.6 0.4 \
  --equity-curve-ma-short 5 \
  --equity-curve-ma-long 20 \
  --equity-curve-recovery-mode gradual \
  --equity-curve-recovery-step 0.1
```

#### 2. 运行纸面交易

```bash
python scripts/paper_trade.py run --trade-date 20260205
```

ECT 会自动生效，日志输出示例：

```
计算 ECT 仓位系数
ECT 计算结果: [20260205] ECT: 回撤 12.50% (触发), 均线趋势向下, 系数=0.40
ECT 仓位系数: 0.40
应用 ECT 系数 0.40 到目标权重
已将 ECT 系数应用到 5 个目标权重

【ECT 仓位管理】
仓位系数: 0.40
计算原因: [20260205] ECT: 回撤 12.50% (触发), 均线趋势向下, 系数=0.40
```

### 回测

#### 1. 命令行运行

```bash
python scripts/run_ml_backtest.py \
  --start-date 20230101 --end-date 20231231 \
  --top-n 5 --rebalance-freq 5 \
  --equity-curve-enabled \
  --equity-curve-drawdown-thresholds 5.0 10.0 15.0 \
  --equity-curve-exposure-levels 0.8 0.6 0.4 \
  --equity-curve-ma-short 5 \
  --equity-curve-ma-long 20
```

#### 2. Python API

```python
from src.lazybull.risk.equity_curve import EquityCurveConfig
from src.lazybull.backtest import BacktestEngine

# 创建 ECT 配置
ect_config = EquityCurveConfig(
    enabled=True,
    drawdown_thresholds=[5.0, 10.0, 15.0, 20.0],
    exposure_levels=[0.8, 0.6, 0.4, 0.2],
    ma_short_window=5,
    ma_long_window=20,
    recovery_mode='gradual',
    recovery_step=0.1
)

# 创建回测引擎
engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=1000000,
    rebalance_freq=5,
    equity_curve_config=ect_config,
    verbose=True
)

# 运行回测
nav_curve = engine.run(
    start_date=pd.Timestamp('20230101'),
    end_date=pd.Timestamp('20231231'),
    trading_dates=trading_dates,
    price_data=price_data
)
```

## 参数调优建议

### 回撤阈值

- **保守型**：[3%, 5%, 8%, 10%]，系数 [0.8, 0.6, 0.4, 0.2]
- **中性型**：[5%, 10%, 15%, 20%]，系数 [0.8, 0.6, 0.4, 0.2]（默认）
- **激进型**：[10%, 15%, 20%, 25%]，系数 [0.8, 0.6, 0.4, 0.2]

### 均线窗口

- **短周期**：短期 3 天，长期 10 天（快速反应）
- **中周期**：短期 5 天，长期 20 天（默认）
- **长周期**：短期 10 天，长期 60 天（平滑趋势）

### 恢复策略

- **快速恢复**：recovery_step=0.2，recovery_delay_periods=0
- **中速恢复**：recovery_step=0.1，recovery_delay_periods=1（默认）
- **慢速恢复**：recovery_step=0.05，recovery_delay_periods=2

## 测试

运行单元测试：

```bash
python -m pytest tests/test_equity_curve.py -v
```

测试覆盖：
- 配置创建和验证
- 回撤计算
- 均线趋势计算
- 恢复机制（gradual/immediate）
- 边界情况（禁用、空历史等）

## 注意事项

1. **数据要求**：需要足够的历史 NAV 数据（至少 ma_long_window 个数据点）
2. **首次运行**：纸面交易首次运行时，NAV 历史为空，ECT 不生效
3. **恢复速度**：恢复太快可能导致过早增仓，建议根据策略特点调整
4. **组合使用**：可与止损功能同时使用，提供多层次风险控制
5. **向后兼容**：默认 ECT 禁用，不影响现有策略

## 版本信息

- **版本**：v0.3.5
- **发布日期**：2026-02-05
- **作者**：deltree-y

## 相关文档

- [README.md](../README.md) - 项目主文档
- [止损功能文档](stop_loss_guide.md) - 止损功能说明（如果存在）
- [API 参考](api_reference.md) - API 文档（如果存在）

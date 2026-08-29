# 分批调仓功能说明

## 概述

分批调仓功能允许将总 `TopN` 持仓分散到多个时间点逐步完成，降低单一调仓时点风险。
`top_n` 始终表示组合最终总持仓数，不会乘以批次数。

例如 `top_n=20`、`stagger_tranches=4` 时，每批建立 5 个槽位，每批预算为组合资产的
25%，四批完成后目标仍是 20 只股票。`top_n=30`、`stagger_tranches=4` 时，批次槽位为
`8/8/7/7`，预算按槽位比例分配并合计为 100%。

**回测引擎和纸面交易共用同一套分批调仓核心逻辑**（`trading/stagger.py`），包括：
- 排期计算（`compute_tranche_schedule` / `build_tranche_schedule_from_anchor`）
- 槽位拆分（`get_tranche_target_count`）
- 预算比例（`get_tranche_capital_fraction`）

## 功能说明

### 使用场景

当目标持仓为 20 只股票时，可以配置为每周只调整其中 5 只，分 4 周完成一次完整调仓。

### 优势

1. **降低冲击成本**: 避免一次性大量交易对价格的影响
2. **平滑执行**: 分散交易时间，降低执行风险
3. **灵活性**: 可根据市场情况动态调整
4. **适应小资金**: 对于资金量较小的账户，可以减少单次交易数量

## 配置说明

walk-forward OOS 回测通过命令行配置：

```powershell
python scripts/walk_forward.py --oos-backtest --bt-top-n 20 --bt-rebalance-freq 20 --stagger-tranches 4
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `bt_top_n` | int | 30 | 组合最终总持仓数 |
| `bt_rebalance_freq` | int | 标签推断 | 每个批次自身的完整持有周期（交易日） |
| `stagger_tranches` | int | 1 | 批次数；1 表示不分批 |

## Python 使用示例

```python
from src.lazybull.backtest import BacktestEngine

engine = BacktestEngine(
    universe=universe,
    signal=signal,
    initial_capital=500000,
   rebalance_freq=20,
   stagger_tranches=4,
)
```

## 分批调仓流程

1. 将 `top_n` 拆成 K 个槽位组，余数优先分配给较早批次。
2. K 个批次在一个 `rebalance_freq` 周期内按比例均匀取整排期。例如 20 日分 3 批时，
   信号偏移为 `0/7/13`，循环间隔为 `7/6/7`。
3. 每批从当日最新排名中选择本批槽位数，排除当前已持仓股票。
4. 每批按“本批槽位数 / 总 TopN”分配预算，并在 T+1 执行。
5. 各批持仓按自己的实际买入日计算持有期，到期后独立滚动更新。
6. T+1 首选买入失败时，按 T0 原始排名在同一天继续顺延；当天仍未成交的槽位才进入后续补位队列。

## 边界情况处理

- `top_n` 不能整除 K：按实际槽位比例分配预算，所有批次合计仍为 100%。
- `K` 必须同时满足 `1 <= K <= top_n` 和 `K <= rebalance_freq`，避免空批次或同日排期覆盖。
- 配置单股上限时必须满足 `max_weight_per_stock * top_n >= 1`；上限按本批资金比例换算为批内权重，最终仍按全组合口径约束。
- 候选不足或不可交易：T+1 先按原排名同日顺位替补，仅剩余缺口进入跨日仓位补齐。
- 买入失败不会改变卖出规则；停牌或跌停导致的卖出失败继续延期，且不会提前释放持仓槽位。
- 回测区间起点：组合需要经过 K 个批次才逐步达到目标仓位，属于分批策略本身的建仓期。

## 性能考虑

### 交易成本

分批调仓会：
- **增加交易次数**: 原本一次调仓变为多次
- **可能降低冲击成本**: 避免一次性大量交易
- **总成本**: 需要根据实际情况权衡

### 建议

1. **对于大资金**：分批调仓可以明显降低冲击成本
2. **对于小资金**：交易成本增加可能抵消分批的优势
3. **流动性好的股票**：分批效果不明显
4. **流动性差的股票**：分批调仓更有价值

## 纸面交易配置

纸面交易通过 `config.yaml` 的 `stagger_tranches` 字段启用分批调仓：

```yaml
# data/paper/config.yaml
top_n: 20
rebalance_freq: 20
stagger_tranches: 4  # 分4批调仓，1=不分批
```

或通过命令行设置：

```powershell
python scripts/paper_trade.py config --top-n 20 --rebalance-freq 20 --stagger-tranches 4
```

### 纸面交易与回测的差异

| 维度 | 回测引擎 | 纸面交易 |
|------|---------|---------|
| 排期锚定 | 回测区间起点 | `tranche_anchor_date`（批次0调仓日） |
| 空仓提前调仓 | tranche_idx=0 全量建仓 | tranche_idx=0 全量建仓（一致） |
| 拖尾提前调仓 | tranche_idx=0 全量建仓 | tranche_idx=0 全量建仓（一致） |
| 持有期计算 | 按各批实际买入日独立计算 | 按各批实际买入日独立计算（一致） |
| 资金分配 | 组合总资产 × 槽位比例 | 组合总资产 × 槽位比例（一致） |
| 买入失败 | T1 按 T0 排名同日顺延 | T1 按 T0 排名同日顺延（一致） |
| 行业/单股上限 | 计入已有持仓，按全组合权重约束 | 相同（一致） |

### 状态持久化

分批模式下，`rebalance_state.json` 额外维护：
- `tranche_anchor_date`: 批次0锚定日，用于推算后续各批次排期
- `last_rebalance_date`: 最近一次 T0 的实际执行日
- `last_scheduled_rebalance_date`: 最近一次已履行的计划日，用于识别漏批
- `stagger_tranches`: 批次数

若纸面任务错过计划日，下一次运行会先补最早未履行批次；一天只履行一个批次，后续漏批在之后的交易日依次追赶。修改 `rebalance_freq` 或 `stagger_tranches` 后，当前成功 T0 会作为新周期的批次0和 anchor。手工 truncate 会从最后保留的 T0 运行记录恢复完整排期状态。

## 注意事项

- `rebalance_freq` 是每个批次自身的持有周期，不是相邻批次间隔。
- 相邻批次间隔由引擎按 `rebalance_freq / stagger_tranches` 自动排期。
- 免训练扫描必须同时保留 `stagger_tranches=1` 基线，并使用相同模型版本和 split 边界。
- 2026-08-01 之前生成的分批结果受"仅首批实际买入"问题影响，不应用于策略判断。

## 相关文档

- [回测假设说明](backtest_assumptions.md)
- [交易成本配置](../configs/base.yaml)
- [调仓频率设置](README.md#调仓频率增强)

# terminal_loss 模型家族处置清单（P3-T5 处置执行，2026-10-23）

> **裁决依据**：P1.5 政策层迁移前裁决 = **退役**（F8，2026-10-01 生效，
> `docs/reports/p15_policy_layer_adjudication_20261001.md` §5，台账 `H-P1.5-adjudication-conclusion-20261001`）。
> **处置口径**：Q1-A（2026-10-20 用户裁决，`docs/plans/v2/p3_implementation_plan_20261020.md` §13.1-Q1 / §6）。
> **契约依据**：`docs/contracts/v2_architecture_plan.md` §4.7（功能退役契约）/ §4.3 :493 行 + :502 尾注 /
> §8-P3 行（:736）/ §9（旧链路观察期）。
> **处置评审状态**：本清单随 **T5 独立评审（用户发起）** 评审；评审通过前不执行任何清单外动作。
> **本期执行动作边界（Q1-A）**：**不动现役引用、不改默认臂**（e2online_r 继续服务旧链路日常运行至 v2 切换日）、
> **无物理删除、不打 git tag**；代码与测试文件**保留至 P4 观察期结束、禁止新增投入**。
> **台账**：退役档案条目 = `data/ledger/hypotheses.jsonl`（`H-p3-t5-terminal-loss-family-disposal-20261023`）；
> 风险登记复审注记 = `docs/terminal_loss_risk_register.md`（R-001 / R-004 / R-006 / R-007，F8R-03 联动）。

## 1. 处置总口径

本清单覆盖的每一项资产，处置口径统一为：

| 项 | 口径 |
|---|---|
| **处置** | **登记退役（封存至 v2 切换日）**——评审登记 + 退役档案（本清单）+ 台账 terminated 条目 + 封存，按 §4.7 执行 |
| **删除时点** | **随 §9 旧链路观察期后统一执行**——切换日只执行「默认链路移除 + git tag 留档 + 退役档案」；物理删除不早于观察期结束（F8R-05 时序衔接：保活期回滚须能复现切换前默认臂） |
| **依据** | **F8（P1.5 退役裁决）+ Q1-A（P3-T0.5a 处置裁决）+ P1.5 裁决报告 §5（连带执行项）** |

切换日动作（§4.7 退役清单，**非本期执行**）：默认链路移除 + git tag 留档 + 退役档案登记收口。

## 2. 家族代码（`src/lazybull/risk/terminal_loss/`，12 个 .py，实测合计 6,136 行）

行数为 2026-10-23 实测（`wc -l`）。每文件处置 = **登记退役（封存至切换日）｜删除时点 = §9 观察期后统一执行｜依据 = F8 + Q1-A + P1.5 §5**。

| 文件 | 行数（实测） | 职责 | 处置 |
|---|---|---|---|
| `__init__.py` | 200 | 家族入口与公共导出（terminal_vol_scaled_loss 定义、特征列集） | 登记退役（封存至切换日）｜§9 观察期后删除｜F8 + Q1-A + P1.5 §5 |
| `artifacts.py` | 240 | 产物落盘：ModelRegistry 版本化注册 + 固定名别名（注册机制实现） | 同上 |
| `block_stats.py` | 511 | 分块重采样统计：门禁区间与成对比较（`paired_day_mean_ci` 等） | 同上（**有家族外消费方**，见 §6 依赖登记） |
| `coverage_audit.py` | 409 | 覆盖审计（标签覆盖率报告门禁） | 同上 |
| `dataset.py` | 665 | 数据集构建：多期限瘦表、特征矩阵、时间分割与隔离 | 同上 |
| `exposure_gate.py` | 900 | 政策层 E2 条件暴露门控（regime × 组合平均风险）校准与离线评估 | 同上 |
| `exposure_online.py` | 923 | 政策层在线暴露系数 λ_t 随持仓实时计算（P2-5；现役默认臂运行路径） | 同上（**现役引用存续至切换日**，本期不动） |
| `labels.py` | 314 | 剩余持有期期末异常亏损标签构建 | 同上 |
| `model.py` | 183 | 模型加载、输入契约与批量预测 | 同上 |
| `mother_section.py` | 627 | `pct_*` 完整母截面重建 | 同上 |
| `policy_sidecar.py` | 634 | 政策旁路：持仓打分、风险台账、触发清单与阈值扫描（P2-1） | 同上 |
| `train.py` | 530 | 二分类训练与概率质量评估 | 同上 |
| **合计** | **6,136** | — | — |

## 3. 注册表与折模型产物（数据资产）

处置同上：登记退役（封存至切换日）｜§9 观察期后统一删除｜F8 + Q1-A + P1.5 §5。**本期不动其内容**。

### 3.1 平铺注册表 `data/models/terminal_loss/`（实测 5 个固定名产物，248K）

| 产物 | 说明 |
|---|---|
| `terminal_loss_model.joblib` | 固定名模型别名（payload dict 形态） |
| `terminal_loss_model.json` | 固定名模型元数据别名 |
| `terminal_loss_report.json` | 训练报告 |
| `calibration_by_h_sigma.csv` | 分期限 σ 校准表 |
| `label_coverage.csv` | 标签覆盖率表 |

- **注册机制（实测确认）**：`risk/terminal_loss/artifacts.py` 经 **ModelRegistry 版本化注册**
  （`save_versioned_artifacts`，每次训练新增 `v{N}` 永不覆盖历史），固定名文件为**额外别名**
  （供 `summarize_terminal_risk_wf.py` 等工具按固定名读取，可被覆盖、不作唯一副本）；
  model_type = `xgboost_terminal_loss`。
- 本目录为三平级模型家族目录之一（`stock_selection` / `risk` / `terminal_loss`）；随家族退役整目录封存。

### 3.2 折模型散落（WF 折目录，实测确认）

| 目录 | 规模（实测） | 内容 |
|---|---|---|
| `data/walk_forward/terminal_risk_wf/` | 123 个目录条目，**1.3G** | 旧 8 折日历半年集（2022H2 起）+ 调参臂折目录（`_lr*` / `_s*` / `_d4/d5` 等）+ `archives/`（门禁区间证据） |
| `data/walk_forward/terminal_risk_wf_oos14/` | 14 个 OOS 折目录 + `paper_warmup/` + summary/tuning CSV，**179M** | 对齐选股 OOS 的 14 折集（`OOS00_201811`~`OOS13_202506`，现役默认臂 `policy_model_root` 指向此目录） |

- 折模型未走集中注册表，散落 WF 批次目录（每折目录内含版本化 `v{N}_model.joblib` 等）；
  处置随家族一并封存；物理删除前须先确认 §6 所列消费方已摘除。

## 4. 训练 / 分析链路脚本（12 个，逐个实测存在）

处置同上：登记退役（封存至切换日）｜§9 观察期后统一删除｜F8 + Q1-A + P1.5 §5。
匹配处数 = 2026-10-23 grep 实测（pattern `terminal_loss|terminal_risk`）。

### 4.1 家族专用链路（9 个，随家族处置）

| 脚本 | 匹配处数 | 用途 |
|---|---|---|
| `scripts/train_terminal_risk_model.py` | 13 | 家族训练入口（import 家族 4 处） |
| `scripts/batch/batch_terminal_risk_wf.ps1` | 9 | 家族滚动训练批入口 |
| `scripts/calibrate_exposure_gate.py` | 5 | E2 门控校准（import 家族 3 处） |
| `scripts/analyze_terminal_risk_gate.py` | 13 | 门禁分析（import 家族 2 处） |
| `scripts/summarize_terminal_risk_wf.py` | 10 | 折产物汇总（按固定名别名读注册表） |
| `scripts/analyze_drawdown_sweep.py` | 2 | P2-4 回撤侧总扫描分析入口 |
| `scripts/analyze_policy_sidecar.py` | 7 | P2-1 旁路报告分析（import 家族 1 处） |
| `scripts/prepare_paper_exposure_warmup.py` | 4 | 纸面暴露预热（import 家族 3 处） |
| `scripts/batch/batch_policy_sidecar.ps1` | 4 | 旁路批入口 |

### 4.2 主链路 / 通用工具中的引用（3 个，**不随家族处置**，仅登记引用面）

| 脚本 | 匹配处数 | 性质 |
|---|---|---|
| `scripts/batch/batch_walk_forward.ps1` | 6 | **主链路批脚本**：默认臂 e2online_r 定义与 `policy_model_root` 默认（现役引用，本期不动；切换日随默认链路移除） |
| `scripts/compare/signal_metrics.py` | 2 | 通用比较工具：**真实 import** `block_stats.paired_day_mean_ci`（见 §6 依赖登记） |
| `scripts/compare/drawdown_sweep.py` | 2 | P2-4 对照分析（docstring 引用） |

## 5. 引用面实测（2026-10-23；pattern `terminal_loss|terminal_risk`，排除 `__pycache__`）

### 5.1 src 引用面：**39 文件 / 85 匹配行**（其中家族自身 12 文件 / 46 行；家族外 27 文件 / 39 行）

家族外 27 文件按性质分类：

**A. 真实 import（3 文件 / 3 处）——物理删除的前置摘除点**

| 文件 | 引用 | 类别 |
|---|---|---|
| `src/lazybull/ml/walk_forward/runner.py:22` | `from ...risk.terminal_loss.exposure_online import ...` | 主链路（WF OOS 编排，现役默认臂路径） |
| `src/lazybull/paper/exposure_policy.py:48` | `from ..risk.terminal_loss.exposure_online import ...` | 主链路（纸面政策接线，现役默认臂路径） |
| `src/lazybull/v2/evidence/signal_ruler.py:34` | `from src.lazybull.risk.terminal_loss.block_stats import ...` | **v2 证据机器复用统计工具**（见 §6 依赖登记） |

**B. 主链路编排 / 配置引用（注释、参数与路径级，14 文件）**

- 回测侧：`backtest/engine.py`、`backtest/run_loop.py`（P2-1 快照旁路注释）、
  `backtest/exposure_override.py`（3 处，政策买入缩放）、`backtest/exposure_trim.py`（P2-3 主动减仓）、
  `backtest/exposure_replenish.py`（P2-4 对称回补）、`backtest/holdings_snapshot.py`（P2-1 快照动机注释）；
- WF / 训练侧：`ml/walk_forward/backtest.py`、`ml/walk_forward/cli.py`（`--policy-model-root` 参数）；
- 纸面侧：`paper/runtime.py`（P2-5 接线注释）、`paper/storage/config.py`（配置展示文案）；
- 配置侧：`common/config.py`（2 处，模型目录层级注释）、`common/trading_config.py`（2 处，暴露政策参数区）。

**C. 工具 / 兼容层文档级引用（docstring/注释，4 文件）**

- `common/sidecar_schema.py`、`common/xgboost_compat.py`（2 处，加载通路说明）、
  `factors/risk/volatility_factors.py`（2 处，mkt_vol_20 尺度归属注释）、`ml/train_core/eval.py`（group 口径注释）。

**D. v2 复制件 / 新内核（9 文件 / 12 行；P2a~P5a 迁移复制带入）**

- `v2/common/trading_config.py`（2）、`v2/common/xgboost_compat.py`（2）、
  `v2/core/accounting/holdings_snapshot.py`、`v2/core/accounting/state.py`、`v2/core/execution/run_loop.py`、
  `v2/hosts/backtest/replay.py`（均为 P2-1 旁路注释/配置）、`v2/train/core/eval.py`（注释）、
  `v2/train/walk_forward/orchestrator.py`（注释）；
- 例外：`v2/evidence/signal_ruler.py`（2 行）含**真实 import**（已列 A 类）。

### 5.2 tests 引用面：**19 个 test_*.py + 1 个 fixture**

- 家族专测（10）：`test_terminal_loss_{artifacts, block_stats, coverage_audit, dataset, gate_analysis,
  labels, mother_section, rank_early_stop, script, train_model}.py`；
- 政策层 / 旁路链路测试（7）：`test_exposure_gate.py`、`test_exposure_online.py`、
  `test_backtest_exposure_override.py`、`test_exposure_budget_discount_replenish.py`、`test_policy_sidecar.py`、
  `test_summarize_terminal_risk_wf.py`、`test_paper_config_cleanup.py`；
- 复用 / 配置引用（2）：`test_signal_metrics.py`（import `block_stats.paired_day_mean_ci`）、
  `test_config.py`（目录层级注释断言）；
- fixture（1）：`tests/fixtures/inuse_closure_baseline_20261006.json`（12 处字样，P2a 在用闭包基线）。

### 5.3 引用面口径更正注记

规划 `docs/plans/v2/p3_implementation_plan_20261020.md` §6 旧口径「35 文件 81 处代码引用 / 11 个测试文件」
为规划期（2026-10-20 前）统计，**已过时**：其后 P2a/P2b/P5a 的 v2 复制件带入 9 个新引用文件，
测试面亦随 exposure_online / 政策层测试扩张。**以本清单 2026-10-23 实测为准**：
src 39 文件 / 85 行（家族外 27 文件 / 39 行，其中真实 import 仅 3 处）、tests 19 个 test_*.py + 1 fixture、
scripts 12 个（家族专用 9 + 主链路/通用 3）。

## 6. 存续口径与依赖登记（Q1-A 边界）

1. **代码与测试文件保留至 P4 观察期结束、禁止新增投入**：§2/§4.1/§5.2 所列代码与测试文件在观察期内
   不删除、不重构、不新增功能；仅崩溃级修复（行为冻结纪律，§9.1 基线稳定条款）。
2. **不动现役引用、不改默认臂**：§5.1-A/B 中现役默认臂（e2online_r）运行路径
   （`ml/walk_forward/runner.py`、`paper/exposure_policy.py`、`scripts/batch/batch_walk_forward.ps1` 默认臂定义）
   本期一律不动；切换日才执行默认链路移除。
3. **物理删除前置依赖（删除时点行动项，登记）**：
   - `v2/evidence/signal_ruler.py:34` 与 `scripts/compare/signal_metrics.py:32` 真实 import
     `block_stats.paired_day_mean_ci`——家族删除前须先把该统计函数迁移 / 复制到 v2 侧
     （证据机器自建）或一并删除其消费点；
   - `tests/fixtures/inuse_closure_baseline_20261006.json` 为 P2a 在用闭包基线，含家族字样，
     删除家族时须同步重估该基线。
4. **退役 ≠ 永别（可逆性，沿 F8）**：§4.7 全流程保留 git tag 留档与本档案；
   重开条件见 §7。

## 7. 重开条件（沿 F8 台账登记，二者满足其一方可重新预登记）

1. P5b 回撤型功效标定产出后，回撤域配对读数在**频率语义**下可检出（非单次读数）；
2. 出现消除链式依赖的新形态（如 λ 由物化市场状态库驱动而非在线模型推理），按 §4.8
   新组件标准重新立项（独立可验收性检查必答）。

## 8. 依据链与关联文档

- 裁决报告：`docs/reports/p15_policy_layer_adjudication_20261001.md`（§5 连带执行项 = 本清单的立项来源；
  F8R-03 风险登记联动义务 → 已落 `docs/terminal_loss_risk_register.md` 四条复审注记）；
- 处置裁决：`docs/plans/v2/p3_implementation_plan_20261020.md` §6（评审设计）/ §13.1-Q1（Q1-A 裁决）；
- 契约：`docs/contracts/v2_architecture_plan.md` §4.3（:493 行 / :502 尾注，本次执行登记已同步）/
  §4.7 / §8-P3 行（:736）/ §9；
- 风险登记：`docs/terminal_loss_risk_register.md`（R-001 / R-004 / R-006 / R-007 复审注记，2026-10-23）；
- 台账：`data/ledger/hypotheses.jsonl`（`H-p3-t5-terminal-loss-family-disposal-20261023`，terminated）；
- 历史结论参照：各家族 WF A/B 终结报告（`docs/reports/{holdertrade, repurchase, top10fh, top_inst,
  downside_penalty}_wf_ab_result.md`）。

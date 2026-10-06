# LazyBull — AI 编码代理指南

> 本文件面向对项目一无所知的 AI 编码代理。项目为中文优先：注释、文档、提交信息、与用户交流均使用中文。
> **权威契约库是 `CLAUDE.md`**（尤其第 5 节"核心设计模式"中的数十条硬契约）——本文件只作导航与摘要，改动前必须阅读 `CLAUDE.md`，任何实现与契约冲突时以 `CLAUDE.md` 为准。

## 1. 项目概览

- **名称**：LazyBull（当前版本见 `pyproject.toml`，如 v0.204.1）。
- **定位**：A 股量化研究、训练、回测、纸面交易一体化框架，专注**价值红利策略**。
- **语言/运行时**：Python 3.12，包管理用 Poetry（`pyproject.toml`），也可用 pip + `requirements.txt`。
- **数据源**：TuShare Pro 为主（辅以 akshare、efinance）；Token 经 `.env` 的 `TS_TOKEN` 注入（python-dotenv 加载）。
- **交易边界前提**：只做 A 股**现货、非杠杆**交易。禁止引入期权、期货、融资融券、杠杆 ETF 等衍生品；不考虑打新等制度性随机收益。所有方案与收益/回撤评估都必须在此前提下进行。

## 2. 技术栈

| 类别 | 库 |
|---|---|
| ML | XGBoost ^3.4、scikit-learn ^1.6、scipy ^1.13、joblib |
| 数据 | pandas ^3.0、numpy ^2.0、pyarrow ^21（Parquet 分区存储） |
| 数据源 | tushare、akshare、efinance |
| 配置/日志 | PyYAML、python-dotenv、loguru |
| 其他 | tqdm、dingtalk-stream（钉钉机器人）、psutil、gpiod（Linux 树莓派显示，平台标记依赖） |
| 开发 | pytest ^7.2、pytest-cov、black、flake8、isort |

## 3. 目录结构与模块划分

```
src/lazybull/           # 全部源代码（Poetry src 布局）
├── common/             # 配置（config.py，唯一配置入口）、日志、成本模型、TradingConfig、get_logs_dir()
├── data/               # TuShare 客户端、分区存储 storage.py、懒加载 loader.py、清洗 cleaner.py、
│                       # build_clean.py 批量清洗、ensure.py 自动补齐、按年分区 raw 数据集
│                       #（dividend/holdertrade/repurchase/top10_floatholders/top_inst）
├── factors/            # 因子库（技术/量价/行业/基本面/另类/现金流/分红/北向/龙虎榜等）；
│   └── risk/           # 风控模型专用因子子包（@register_risk_factor 注册）
├── features/           # 特征工程（builder.py，120 日预热）、pipeline.py 批量流水线、ensure.py
├── quality/            # 数据质量扫描与 HTML 看板
├── ml/                 # 训练与注册
│   ├── train_core/     # prepare/eval/labels/weights/freshness/xgb/lgb
│   ├── walk_forward/   # 滚动训练 + OOS 回测 + 链式指标 chain_metrics.py + 数据态血缘 data_state.py
│   └── model_registry.py  # 模型版本注册（ModelRegistry，含旁路 metadata）
├── backtest/           # 回测引擎（engine.py / engine_ml.py）、买卖执行拆分、报告、
│                       # 暴露政策模块（exposure_override/trim/replenish）、持仓快照（只读旁路）
├── trading/            # 回测/纸面共享决策核心（买入计划、卖出规则 sell_rules.py、分批排期 stagger.py）
├── signals/            # MLSignal / EnsembleSignal，Top-N 排序与门控，downside_penalty.py（默认关）
├── risk/               # 止损止盈、PositionRiskModel、terminal_loss/（期末异常亏损）、
│                       # per_stock_exit/（已判定终止，不实施引擎接入）
├── portfolio/          # 组合与行业约束、权重处理
├── paper/              # 纸面交易：账户、券商、存储、runtime、reporting、performance.py（年化唯一实现）
├── universe/           # 选股域（domains.py 为域定义单一来源）
├── v2/                 # v2 契约区实现（store/ P1 数据底座、common/protocols 协议层、evidence/ 证据机器通路）
├── drv/                # 树莓派 LCD 驱动
└── live/               # 实盘接口预留（TODO）

scripts/                # 薄入口脚本 + 子包（compare/、factor_health/、model_audit/、raw_download/、
                        # check/、batch/、respi/、ana/、v2_p1/（P1 数据底座构建/对账/验收） 等）
tests/                  # pytest 测试（180+ 个 test_*.py 文件）
configs/                # base.yaml（默认唯一自动加载）+ 因子排除清单 JSON + runtime_*.yaml（手工覆盖示例）
data/                   # 数据与产物：raw/ clean/ features/（含 panel 热区/panel_archive 冷区）labels/ models/ paper/ reports/ walk_forward/ ledger/ frozen_reference/v2_p1/（P1 验收基准）
docs/                   # 文档；按类别归档（contracts / data / experiments / guide / glossary / PR / reports + 本地 plans / review），
                        # 顶层仅 3 份治理文档（roadmap / BREAKING_CHANGES / terminal_loss_risk_register）；规范见 CLAUDE.md §5
logs/                   # 临时日志唯一去处（gitignore）
temp/                   # 临时脚本/中间产物唯一去处（gitignore）
examples/               # 示例
```

## 4. 数据架构与核心链路

### 4.1 三层数据架构（统一 Parquet 分区存储）

- **raw/**：TuShare 原始数据。`trade_cal`/`stock_basic` 为单文件；行情类按日分区 `YYYY-MM-DD.parquet`；公告类（dividend / stk_holdertrade / repurchase / top10_floatholders / top_inst）按**年分区** `YYYY-12-31.parquet`。
- **clean/**：去重（主键 ts_code+trade_date）、复权价格（`*_adj`）、可交易标记（`tradable`/`is_st`/`is_suspended`/`is_limit_up`/`is_limit_down`）。
- **features/**：`cs_train/`（含标签 y_ret_5/10/20）与 `cs_infer/`（无标签），按交易日分区 `YYYYMMDD.parquet`。两层 schema 必须一致。
- **统一日期契约**：各层日期字段统一为 YYYYMMDD 字符串。

### 4.2 离线训练与回测链路

1. `scripts/download_raw.py` → raw 分区
2. `scripts/build_clean_features.py --horizon 20`（或 `--horizons 5 10 20`）→ clean + cs_train
3. `scripts/train_ml_model.py`（单次）或 `scripts/walk_forward.py`（滚动训练 + OOS 回测）→ XGBoost 模型经 ModelRegistry 版本化注册到 `data/models/stock_selection/`
4. MLSignal（Top-N + 门控）→ 回测引擎 → 报告

模型家族三平级目录：`data/models/stock_selection/`、`data/models/risk/`、`data/models/terminal_loss/`（各自独立 registry；terminal_loss 任何模式都必须经 ModelRegistry 版本化注册）。

### 4.3 纸面交易链路

- `scripts/paper_trade.py`（t0/t1/real/config/positions 子命令）：T0 收盘后生成信号，T1 执行。
- 数据自动补齐：`features/ensure.py` 链式 ensure_basic → ensure_clean → ensure_raw，自动补齐 daily/adj_factor/suspend/因子数据。
- 账户展示契约：起始日解析与自然日复合年化**只允许**在 `paper/performance.py` 实现，禁止脚本复制公式。
- 钉钉机器人 `scripts/bot_service.py` 复用同一执行链路。

## 5. 构建与常用命令

### 5.1 安装

```bash
poetry install          # 推荐
# 或
pip install -r requirements.txt
# 配置 TuShare Token：cp .env.example .env 并填入 TS_TOKEN
```

### 5.2 数据与训练

```bash
# 下载 raw（存在即跳过；--force 强制重下）
python scripts/download_raw.py --start-date 20230101 --end-date 20231231

# 构建 clean + features（--horizon / --horizons 互斥必填其一）
python scripts/build_clean_features.py --start-date 20230101 --end-date 20231231 --horizon 20

# 训练（随机种子固定 42，保证可复现）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231

# 滚动训练（OOS 折回测已并入 walk_forward）
python scripts/walk_forward.py --split-count 14 --final-date 20260105

# 纸面交易
python scripts/paper_trade.py t0 --trade-date 20260121 --top-n 5
python scripts/paper_trade.py t1 --trade-date 20260122
python scripts/paper_trade.py real

# 数据质量看板 / 因子体检 / 列集审计
python scripts/quality_dashboard.py
python scripts/analyze_factor_health.py
python scripts/audit_model_columns.py --last 40
```

### 5.3 批量任务（PowerShell，位于 `scripts/batch/`）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_terminal_risk_wf.ps1
```

## 6. 测试

```bash
pytest                                   # 全量（testpaths=tests，-v --tb=short）
pytest tests/test_features.py -v         # 单文件
pytest tests/ --cov=src/lazybull         # 覆盖率（目标 > 80%）
```

- 测试数据隔离：fixture（`tests/conftest.py`）提供 mock 配置与数据，**测试不得依赖真实配置与真实生产数据**，不得污染工作区。
- 新增或修改功能必须同步新增或修改测试，提交前跑通相关测试。

## 7. 开发规范

### 7.1 代码风格

- PEP 8；行宽 100（`black --line-length=100`）；import 用 `isort --profile=black --line-length=100`；`flake8` 检查。
- 强制类型提示（Type Hints）；注释和文档统一中文。
- 错误处理：捕获具体异常，**禁止空 except**；缺失数据优雅降级（返回 NaN）；错误信息必须含上下文。
- 日志统一用 loguru，按模块绑定 logger；临时日志一律落 `logs/`（唯一入口 `common/config.py::get_logs_dir()`）。

### 7.2 版本与文档同步（改动后必做）

1. 代码改动必须同步更新：**CHANGELOG.md**（版本变更唯一去处）、**README.md**（只承载介绍与使用文档，**禁止混入 changelog**）、**pyproject.toml**（版本号）。
2. 版本号规则：功能完全落地前只递增小版本号（0.127.0 → 0.127.1，分阶段功能各 Phase 算同一功能）；完全落地后才递增大版本号；Bugfix/优化/重构递增小版本号。
3. 若实现与 `CLAUDE.md` 契约不一致，必须同步更新 `CLAUDE.md`。

### 7.3 临时文件规范（项目共识）

- 一切临时脚本/中间产物放 `temp/`，临时日志放 `logs/`（均已 gitignore）；命名建议 `temp/<用途>_<日期>.<ext>`、`logs/<用途>_<日期>.log`。
- 任务收尾主动删除临时文件；需要保留的结论必须写进正式文档（CHANGELOG/docs/契约），不得只留在 temp/ 或 logs/。
- 正式产物写各自归档目录（`data/models/**`、`data/walk_forward/**`、`data/reports/**`），不得写进 `temp/`。

### 7.4 新增因子归属

- 通用选股因子 → `src/lazybull/factors/`；风控模型专用因子 → `factors/risk/` 子包（`@register_risk_factor` 注册）。
- 新增功能尽量对应新增文件，不持续累积大文件。

## 8. 关键架构契约与禁忌（节选，完整见 CLAUDE.md §5）

- **T+1 交易**：T 日信号，T+1 收盘买入（默认），T+n 卖出；到期日判定唯一来源 `trading/sell_rules.py::is_holding_period_exit_due`，回测与纸面共用，禁止任何一侧重算阈值。
- **涨跌停/停牌标记只能在 cleaner 层生成**，features 层只复用不重算。
- **运行时派生家族**（holdertrade / repurchase / top10_floatholders / availability markers）：列**不写入 cs_train/cs_infer 分区**，由训练/OOS 评估/OOS 回测/纸面四侧按同一实现运行时派生；启用开关而未传派生表必须报错，禁止静默补 NaN。
- **TuShare 分页契约**：接口普遍有单页行数上限且**超限不报错**——所有拉取必须翻页读到空 + 全字段去重 + 完整性断言；禁止以"单次调用成功"作为数据完整依据。
- **默认关闭逐位一致**：暴露政策、下行风险惩罚、回补等可选机制，默认关闭时成交与净值必须与基线**逐位一致**。
- **实验裁决纪律**：因子裁剪/新增候选只是实验输入，采纳必须走 walk-forward A/B（预登记判据 + 噪声带）；链式净值 ΔMaxDD 为主判据；已终结的因子家族（holdertrade / repurchase / top10_floatholders / top_inst / downside_penalty 等，见 CLAUDE.md §5 与 `docs/*_wf_ab_result.md`）**不再开新臂、禁止消融位搜索**，重开需新信息且重新预登记。
- **pct_\* 母截面**：分母必须来自 `clean/daily` 重建的标签过滤前完整同日截面（`build_mother_section`），禁止用 cs_train 当分母，禁止在持仓子集内重排。
- **旁路只读**：持仓快照、执行归因等旁路产物不得写回持仓状态、不参与买卖判断；中文表头产物的列名唯一来源 `common/sidecar_schema.py`。
- **数据态血缘**：walk-forward 每次运行采集 git 版本 + 数据水位（`data_state_{wf_run_id}.json`）；同一对比表混入多数据态必须告警，配置差异只在同一数据态内比较。
- **文档归档**：`docs/` 顶层禁止散文件（仅 roadmap / BREAKING_CHANGES / terminal_loss_risk_register 三份治理文档）；新文档按类别归入子目录（contracts / data / experiments / guide / glossary / PR / reports；plans / review 为本地区）；**移动文档必须全仓同步引用**（含 CHANGELOG 历史条目、代码注释、CLAUDE.md / copilot-instructions 等本地文件），以终端全量扫描验证。
- **检视意见文档**统一归档 `docs/review/`（append-only）；复杂度 hook = `scripts/check/check_complexity.py`（棘轮基线只缩不扩）。
- **不要**：破坏 cs_train 与 cs_infer 的 schema 一致性；绕过 T0/T1 指令链路；用局部重建覆盖生产 cs_train 分区；把 changelog 写进 README。

## 9. 安全注意事项

- 敏感信息只放 `.env`（TS_TOKEN、LAZYBULL_SMB_USER/LAZYBULL_SMB_PASS 等），经 python-dotenv 加载；**不得提交、不得打印、不得硬编码**。
- SMB 远端读取（树莓派/NAS）走系统 `smbclient` 命令，无 Python SMB 库依赖。
- 本项目仅供量化研究学习，不构成投资建议；历史回测不代表未来。

## 10. 文档导航

- `CLAUDE.md` — **代理权威契约库**（目录职责、数据流、全部设计契约、开发规范、命令清单）。
- `README.md` — 项目介绍与完整使用文档（各工具命令、口径说明）。
- `CHANGELOG.md` — 版本变更唯一记录。
- `docs/data/data_contract.md` / `docs/guide/backtest_assumptions.md` / `docs/data/features_schema.md` — 数据契约、回测假设、特征标签定义。
- `docs/` — 按类别归档：`contracts/`（v2 契约区）、`data/`（数据契约与审计）、`experiments/`（实验协议）、`guide/`（使用指南）、`reports/`（结果报告）、`PR/`（历史记录）；`plans/`、`review/` 为本地区（gitignore）；顶层仅 roadmap / BREAKING_CHANGES / terminal_loss_risk_register。
- `docs/terminal_loss_risk_register.md` — terminal_loss 风险登记（事实/根因/影响/缓解/复审）。

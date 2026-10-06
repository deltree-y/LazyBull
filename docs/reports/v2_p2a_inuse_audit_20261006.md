# v2 P2a 前置：在用全域对账报告（import 闭包 + 运行时调用点）

> 日期：2026-10-06　任务依据：方案 `docs/contracts/v2_architecture_plan.md` §4.3（v1.11，P0 评审 1-B）
> 「以 `scripts/batch/batch_walk_forward.ps1` + `scripts/paper_trade.py` 两条主链路的 import 闭包 +
> 运行时调用点为在用全域做逐项对账」；「凡"在用但未列入清单"的组件视同遗漏，P2 开工前必须清零」。
> 性质：**只读诊断**，未修改任何生产代码 / 契约 / 配置。

## 1. 采集方法与证据

| 环节 | 方法 | 结果 |
|---|---|---|
| 静态 import 闭包 | ast 递归解析全部 import 语句（含函数体 / 条件块内惰性 import），从三入口展开：`scripts/walk_forward.py`（ps1 主调用）+ `scripts/compare_walk_forward.py`（ps1 收尾）+ `scripts/paper_trade.py` | **196** 个 `src.lazybull.*` 模块 + 13 个 `scripts.*` 模块 |
| 运行时调用点（动态加载） | 全仓扫描 `importlib` / `__import__` 动态 import 机制 | 唯一命中 `factors/risk/factor_registry.py:82`，importlib 加载 5 个风控因子模块，其中 **4 个不在静态闭包** |
| 钉钉链路附带核对 | 同法采集 `scripts/bot_service.py` 闭包 | 163 模块 ⊂ 主链路 196（差集 = 不经 WF 训练侧，符合预期）⇒ **通过**，钉钉链路不引入闭包外模块 |
| 采集脚本与原始产物 | **正式底稿（R4 评审后工具化）**：`scripts/check/check_inuse_closure.py`（可复跑，退出码 = 守恒校验结果；含 `--json-out` 结构化输出），测试 `tests/test_check_inuse_closure.py` 6 项（合成树机制单测 + 真实仓棘轮锚定 196/200/49 + 底稿 fixture 逐集合比对）；结构化底稿 `tests/fixtures/inuse_closure_baseline_20261006.json`（进版本控制；`data/reports/` 整体 gitignore 仅本地归档，故对齐 P1 列族 fixture 先例落 tests/fixtures） | PASS |

**在用全域 = 196（静态闭包）+ 4（factor_registry 动态加载）= 200 模块**；全仓 lazybull 模块 249 个。

工程注记（供 P2a 采集复用）：① 带 BOM 的 `__init__.py` 需 `utf-8-sig` 解析（tushare_client / builder / ensure 三处）；② Windows 文件系统大小写不敏感，模块名→路径映射必须精确大小写匹配（实测 `data.Storage` 误匹配 `storage.py`）；③ 相对导入层级计算需区分普通模块与 `__init__`（实测 `from . import account` 在包 init 内被错砍一层）。

## 2. 在用全域逐包归属（笼统映射覆盖检查）

方案 §4.3 笼统映射：`signals/trading/portfolio/backtest 引擎核心 → core/*`；`paper 决策 → core、宿主 → hosts/paper`；`ml/* → train/* + sleeves/value`；`factors/features → store 构建器 + sleeves 特征集`；`factors/、universe/ 沿用`。

| 包（在用模块数，机器计数，`__init__` 归包） | 归属判定 | 备注 |
|---|---|---|
| common（15） | common ✓ | 明细见 §5：signal_factory / backtest_runtime / smb_client / sidecar_schema / suspend_calendar / trade_status / xgboost_compat 等建议显式登记 |
| data（22） | store ✓ | loader / storage / cleaner / 6 个 raw 家族 / ensure / tushare_client 全家，无例外 |
| factors（39，静态 34 + 动态 4 + init 1） | factors ✓ | 纯计算本体；4 个动态加载模块（downside / liquidity / announcement / derived_factors）经 factor_registry 注册机制进链路 |
| features（28） | store ✓ | builder×7 / ensure×13 / labels / market_state / neutralization / handlers 等；双 market_state 问题见 §5-B6 |
| ml（29） | train + sleeves/value ✓ | walk_forward×12 / train_core×12 / ensemble / eval_utils / model_registry / run_logger；lgb 算法位见 §5-B7 |
| backtest（14） | core + hosts/backtest ✓ | engine / engine_ml / run_loop / `*_execution`×4 / pending_execution → core/execution+decision；reporter + reporting 双报告模块见 §5-B6；`exposure_*`×3 + holdings_snapshot 见 §3 |
| paper（26） | core + hosts/paper ✓ | runner×8 / broker×5 / storage×6 / account / runtime / reporting / performance / models / exposure_policy |
| signals（5） | core/signal + sleeves/value ✓ | base / ml_signal / ensemble_signal；downside_penalty = A5 退役（§3） |
| trading（4） | common/rules + core/decision ✓ | sell_rules / stagger → common/rules（组件清单）；buy_plan / sizing → core/decision |
| portfolio（3） | core/decision ✓ | industry_constraint（组件清单）+ weight_processor |
| risk（10 在用） | 组件清单逐项 | stop_loss / stop_loss_checker / precompute + terminal_loss 在线链 7 模块；**position_risk / label_builder 不在闭包**（§4） |
| universe（3） | universe ✓ | base / domains |
| 顶层根（1） | — | `lazybull/__init__.py` |

**数量守恒（R1 评审修正后，机器数）**：静态闭包 196 + 动态加载 4 = **在用 200**；包级计数合计 = 200 ✓；全仓 249 = 在用 200 + 不在用 49 ✓（§6）。初版报告 §2 包级数字为手写目测、多处失准（加总 223 ≠ 196），已按重跑机器数全量重写；总量 196/200 与逐模块归属判定不受影响（归属核对基于完整模块清单而非包级小计）。

## 3. §4.3 组件迁移清单逐项核对（9 行）

| 组件 | 清单标注 | 闭包实测 | 判定 |
|---|---|---|---|
| PositionRiskModel（pct_*） | 在用 → 迁移原样 core/decision | **不在闭包**（详见 §4） | ⚠ **状态修正候选** |
| terminal_loss 政策层（e2online_r） | 退役（F8），不迁移 | exposure_online / exposure_gate / exposure_trim / exposure_replenish / exposure_override / paper/exposure_policy + terminal_loss.{dataset,labels,model,train,policy_sidecar} 共 11 模块在闭包（默认臂至切换日在用） | ✅ 一致：在用但按 F8 于切换日退役，符合「生效时点 = v2 切换日、迁移期默认臂不变」 |
| per_stock_exit | 已终止（R-005），不迁移 | 不在闭包 | ✅ 一致 |
| 低波惩罚 A5（downside_penalty） | 默认关，不迁移退役档案 | signals/downside_penalty 在闭包（挂载、λ 默认 0） | ✅ 一致：代码在链路但默认关；退役 = 不迁 |
| 止损 / 止盈规则 | 双层归属 common/rules + core/decision | risk/stop_loss + risk/stop_loss_checker 在闭包 | ✅ 一致 |
| 分批调仓排期 stagger | common/rules（anchor 单点化） | trading/stagger 在闭包 | ✅ 一致 |
| 到期规则 sell_rules | common/rules | trading/sell_rules 在闭包 | ✅ 一致 |
| 行业约束 | core/decision（四件硬约束之一） | portfolio/industry_constraint 在闭包 | ✅ 一致 |
| 持仓快照 holdings_snapshot | 内核旁路 | backtest/holdings_snapshot 在闭包（WF OOS 显式开启） | ✅ 一致 |

## 4. 核心发现：PositionRiskModel 状态修正（需用户裁决）

**事实链**（全部静态证据，可复核）：

1. `src/lazybull/risk/position_risk.py` 与 `risk/label_builder.py` **不在两条主链路 import 闭包内**（196+4 均无）；
2. 全仓唯一引用方 = `scripts/train_position_risk_model.py`（离线训练入口）；
3. 推理 / OOS 回测 / 纸面侧**没有任何**加载 `data/models/risk` 注册表的代码（grep `models/risk` / `get_risk_models_root` / `risk_model` 于 signals / backtest / paper / common.config 均无命中）；历史产物 `data/models/risk/v1_*` 存在但无运行时消费方；
4. pct_* 风控因子列经 `risk/precompute.py`（在闭包，由 features/builder 调度）预计算进特征——**链路上跑的是因子，不是该模型**。

**判定**：PositionRiskModel 现状 = 「有离线训练入口 + 有历史产物 + 无主链路消费方」，与 terminal_loss 家族 F8 补登的「在用但无消费方」同类（程度更轻：其因子链 pct_* 仍在生产）。§4.3 清单「在用 → 迁移原样 core/decision」的**前提已不成立**。

**选项**（提交用户裁决，本报告不代决）：
- **A（建议）**：状态改注「在用但无消费方」——迁移清单行保留但标注"仅当 P2a~P4 期间恢复消费方才迁移 core/decision；否则随 §4.7 退役评审处置"；对齐 terminal_loss 家族先例（评审前禁止新增投入）；
- **B**：维持原行不变（接受"离线研究资产也迁 core/decision"）；
- **C**：立即改判退役档案。

**裁决结果（2026-10-06，用户）**：**不迁移**（接近选项 A 的处置形态）——§4.3 清单行已按「不迁移 + 本体随 §4.7 退役评审处置 + 评审前视为在用但无消费方、禁止新增投入」修订，登记于方案 F10（草案）。

**保留链与退役链的准确切分（R3 评审修正，初版表述有误）**：初版把保留链称为"pct_* 因子预计算"是**错误表述**——两者是不同对象：
- **仍迁移（生产在用）**：`risk/precompute.py` 批量预计算通道输出 **22 个基础风险因子**（A 类下行风险 8 / B 类波动结构 6 等，与 downside / volatility / liquidity 注册名一致；A5 终结裁决引用的 `downside_vol_20` 即出自此链），因子本体在 `factors/risk` 注册模块——该链随 store 迁移；
- **随模型不迁移（停用模型专用能力）**：`pct_*` 截面百分位特征为 PositionRiskModel 专用派生——推理侧 `position_risk.py::_ensure_pct_columns`、训练侧 `train_position_risk_model.py::_add_pct_features`，**不在 precompute 输出中**；连同 `factors/risk/position_features.py`（无引用孤儿，§6）一并随 §4.7 退役评审处置。

## 5. 遗漏清零判定

### 5.A 清零结论

**在用全域 200 模块中，不存在「笼统映射与组件清单双盲区」的模块**——每个在用模块均可由 §4.3 笼统映射（目录级）或组件清单（风控家族级）之一覆盖。**方案口径的"遗漏 = 0"，P2a 可开工**（PositionRiskModel 状态修正属清单行内状态问题，非遗漏）。

### 5.B 建议显式登记项（笼统映射可覆盖、但属易漏点；登记防 P2a 分解时漏搬）

| # | 模块 / 形态 | 建议归属 | 理由 |
|---|---|---|---|
| B1 | common/signal_factory.py（单模型/ensemble 信号创建工厂） | core/signal | 三端（WF / 纸面 / 钉钉）共用信号装配；协议 §2 SignalProtocol 消费端 |
| B2 | common/backtest_runtime.py（OOS 回测工厂） | hosts/backtest | WF OOS 入口经此构造引擎；CLAUDE.md 已注记"纸面 serving 模式未接线"，但 **WF 侧在用**（被 ml/walk_forward/backtest 引用） |
| B3 | common/smb_client.py（SMB 缓存） | hosts/paper | 外部设备缓存适配（180 秒有效、亮屏同步契约），非内核语义；被 paper/storage 引用 |
| B4 | common/sidecar_schema.py（快照列名 / 中文表头唯一来源） | **拆分归属（R2 评审修正，初版"随政策层退役"登记有误）** | 实测引用方**三方共享**：① `backtest/holdings_snapshot.py:29`（SNAPSHOT_KEYS，组件清单胜者·内核旁路）；② `ml/walk_forward/reporting.py:10`（SNAPSHOT_COLUMNS_ZH + to_chinese，WF 报告链）；③ terminal_loss 政策链三处（exposure_gate / exposure_online / policy_sidecar，F8 退役范围）。处置：**快照列名 / 中文表头与 to_chinese 转换能力迁入 v2 公共表结构模块（唯一来源）**，快照与报告调用方改指新址，一致性验收后随退役时序删除旧模块与政策专用键；不得以"下游依赖"为由保留退役实现 |
| B5 | risk/precompute.py（批量预计算入口，输出 22 个基础风险因子） | store 构建调度 | 与 §4 保留/退役切分对齐（R3 精确化）：**批量预计算入口**（precompute，22 列）与**注册表计算入口**（factor_registry 动态加载 5 模块）分开登记；因子本体在 factors/risk ✓（归属契约已有）；调度器从 features/builder 迁 store 时需带上。`pct_*` 派生（模型专用）**不在本链**，不构成 store 迁移要求 |
| B6 | 三组"多实现并存"形态 | P2a 分解时单一来源化 | ① 双 market_state（factors/ 与 features/ 各一，D-11 缓存锚定先例）；② 双报告模块 backtest/reporter.py 与 reporting.py；③ 延迟订单三形态（backtest/pending_execution 队列 + paper/storage/queue 队列 + execution/pending_order 共享数据类）——协议 §11 只登记了前两处 |
| B7 | ml/train_core/lgb.py（LightGBM 可选算法位） | train | `training_core.py:255` 按 `algorithm=="lightgbm"` 分支可达（默认 xgboost）；train 迁移需带上或预登记剔除（技术栈仅声明 XGBoost） |
| B8 | common/suspend_calendar.py / trade_status.py / xgboost_compat.py / feature_utils.py | core/execution 支撑 / common | 停牌日历、交易状态判定、XGB 兼容层、特征工具——均被引擎 / 构建链直接引用，笼统映射归 core/common，列名防漏 |

### 5.C 附带发现（不改代码，仅登记）

- **CLAUDE.md §5「前置信号门控：legacy/composite/disabled 三模式」描述已过时**：全仓无该门控实现，`signal_gate` 家族键仅存在于 `paper/storage/config.py` 的退役键清单（v0.90.2 删除死接口）。下次 CLAUDE.md 修订时同步。

## 6. 不在用全域的 49 模块归类（反向核对）

**口径声明（R1 评审修正）**：本节口径 = **不在用全域**（全仓 249 − 在用 200 = 49）。初版标题"53"为**静态闭包差**（249 − 196），混用口径未声明——53 中含 4 个动态加载模块（downside / liquidity / announcement / derived_factors，属在用），不在本节。分类小计 24+7+14+1+2+1 = **49** ✓。

| 类别 | 模块 | 判定 |
|---|---|---|
| v2 新代码（24） | v2 根 + v2.store×9 / v2.evidence×10 / v2.common×3 + 各层 `__init__` | 自身即 v2 资产（P1/P5a 交付），不属旧链路对账范围 |
| 纯 `__init__` 聚合包（7） | common / execution / factors.risk / ml.walk_forward / risk / risk.terminal_loss / trading | 子模块在用即包在用，非遗漏 |
| 离线工具 / 研究入口（14） | data.build_clean、features.pipeline、features.parallel（build_clean_features.py + v2 影子构建引用）；backtest.eval_panel（**全仓无引用方，孤儿模块**）；quality×4（质量看板）；risk.position_risk、risk.label_builder（§4）；risk.terminal_loss.{artifacts,block_stats,coverage_audit,mother_section}（研究链路） | 均有明确非主链路用途；eval_panel 建议随 P2a 前清理评审处置 |
| 模型家族孤儿（1） | factors.risk.position_features | **初版漏分类（R1 评审发现）**：全仓唯一引用 = `factors/risk/__init__.py` docstring 提及，无代码 import；归属 PositionRiskModel 家族（docstring 定位"持仓上下文特征"），随 §4 裁决不迁移、随 §4.7 退役评审处置 |
| 独立设备（2） | drv.mini_led.{device,fonts} | 树莓派驱动，§4.3 明示独立于内核 |
| 预留（1） | live | P8 端口预留 |

**terminal_loss 训练侧四模块（artifacts / block_stats / coverage_audit / mother_section）不在主链路**，与其「随 P3 段按 §4.7 评审处置」（F8 补登）一致 ✅。

## 7. 方案 §4.3 组件表 ↔ 协议 §11 对照表交叉核对

| 语义 | §4.3 组件表 | §11 协议表 | 处置 |
|---|---|---|---|
| 到期判定 / stagger / 锚定 / 止损 / 止盈 | ✓ | ✓ | 双侧齐 |
| 行业约束 | ✓（core/decision） | 无 | **有意不镜像**：§11 收录"引擎语义"，行业约束是组合约束，用途不同——登记即可 |
| 延迟订单队列 | 无显式行（笼统映射覆盖） | ✓（登记"×2"，实测 3 形态含共享数据类） | §5-B6③ 已补注第三形态 |
| 估值回退链 / Kelly 包装 / 信号门控 | 无显式行 | 估值回退链 ✓ / Kelly ✓ / 门控无 | 前两项笼统映射覆盖（v1.11 已知）；门控已退役（§5-C），双侧无行是**正确终态** |
| holdings_snapshot / 政策层 / 集成 / WF 编排 / TradingConfig / 成本 / data_state | ✓ 或语义等价行 | ✓ | 双侧齐 |

两表差异均为「用途不同的有意不镜像」或「已知笼统映射覆盖」，**无需要回写契约的缺口**。

## 8. 结论

1. **清零达标**：在用全域（200 模块，守恒 249 = 200 + 49）全部有归属；P2a 满足方案 §4.3 的开工前提；
2. **一项状态修正已裁决**：PositionRiskModel 不迁移（§4，用户 2026-10-06）；
3. **八项建议显式登记**（§5-B）供 P2a 分解时防漏搬，不构成契约修改义务；
4. **可复核底稿**：`scripts/check/check_inuse_closure.py` + 测试 5 项 + JSON 底稿（§1 表），P2a 迁移期间可重跑监控在用清单漂移。

## 9. 首轮评审处置记录（R1~R4，2026-10-06）

| 意见 | 评估 | 处置 |
|---|---|---|
| R1（高）数量不守恒（223 / 53 vs 49 / 小计 45） | **成立**：初版 §2 包级数字为手写目测、多处失准（加总 223 ≠ 196）；§6 标题 53 是静态闭包差口径（含 4 个动态模块）与"不在用"口径混用未声明；v2 分类 22 实为 24、离线 13 实为 14；**position_features 全漏分类**。总量 196/200 与逐模块归属判定（基于完整清单而非包级小计）不受影响 | §2 表格按机器数全量重写并加守恒声明；§6 改"不在用全域 49"+ 口径声明 + 小计修正 + position_features 补归类；守恒经正式脚本断言（测试棘轮锚定） |
| R2（中）sidecar_schema 非政策专用、退役后下游替换未安排 | **成立**：实测引用三方共享（holdings_snapshot / WF reporting / terminal_loss 政策链），初版 B4"随政策层退役"登记错误 | B4 改拆分归属：快照列名 / 中文表头 / to_chinese 迁 v2 公共表结构模块唯一来源，调用方改指新址、一致性验收后随退役时序删旧；不以"下游依赖"保留退役实现 |
| R3（中）保留链描述与实现不符（pct_* vs 22 因子） | **成立**：precompute 输出 = 22 个基础风险因子，pct_* 是模型专用派生（position_risk.py / 训练脚本），初版把两者混称"pct_* 因子预计算" | §4 增"保留链与退役链准确切分"；契约 §4.3 行同步修正（pct_* 派生 + position_features 孤儿随模型不迁移）；B5 拆分登记批量 / 注册表两入口 |
| R4（中）完成声明缺可复核正式底稿 | **成立**（初版采集脚本按 temp/ 一次性诊断处置即删） | 采集脚本工具化为 `scripts/check/check_inuse_closure.py`（argparse / 守恒退出码 / JSON 输出）+ 测试 5 项 + 底稿 JSON 落 `data/reports/` |

**机器时间**：全程静态分析，未运行动态链路、未触碰生产数据。

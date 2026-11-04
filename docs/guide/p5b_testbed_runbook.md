# P5b 试车场 T6 历史端到端演练运行手册

> 状态：永久文档（H7 交付件之一）。适用版本 ≥ 0.209.6。
> 证据类别：`evidence_class=calibration_only`、`scope=historical_testbed`、
> `virtual_clock=true`、`not_forward_evidence=true`、`not_sla_evidence=true`。
> **闸门判定归用户**：本手册与演练产物只提供作者侧检查结果，不构成生产授权或前向证据。

---

## 1. 目的与非目标

**目的**：用历史虚拟时钟验证「多臂运行器 → 配对账本 → 裁判 → 报告」**全链路**是否成立，并在
已知方向上取三条 H6 单条数值（收益 −5δ、DD-I +12pp、DD-W −12pp 各 1 条），只检查**点估计方向与
链路**（不产生检出率/MDE）。

**非目标**（明确不做，违反即视为越界）：

1. 不运行真实前向影子、不等待真实日历、不产生候选前向分数或真实 SLA；
2. 不注册真实候选、不写生产账户/券商、不写正式 `runs` 或 `LedgerEntry`；
3. 不写 `data/paper`、`data/runs`、`data/models`（生产零写入是 H1 的硬条件）；
4. 不把机械 A/A（同路径克隆）当作统计 A/A；统计 A/A 只来自 T4/T5 的独立同分布路径；
5. 不产生检出率/MDE/显著性；频率类读数一律只引用 T4/T5 冻结产物；
6. 不修改 T4/T5 的仪器身份（manifest / thresholds / holdout / power / verdict / report 及其字节）。

**五臂**（角色名在 manifest 冻结，运行中不得改名）：

| 臂 | 角色 | 效应 |
|---|---|---|
| `base` | `aa_clone` | 无（生产冻结模型/配置的 fresh_cash 克隆、基准臂） |
| `mech_aa` | `aa_clone` | 无（与 base 完全同模型/配置/输入，仅验逐值一致） |
| `income_minus_5d` | `synthetic_effect` | 日度 log 位移 `x' = x − 5δ`（δ = ln(1.05)/252） |
| `dd_improve_12pp` | `synthetic_effect` | 冻结崩盘窗内负收益 `x' = (1−λ)x`（λ 由 T5 求解器语义求得） |
| `dd_worsen_12pp` | `synthetic_effect` | 冻结崩盘窗内负收益 `x' = (1+ζ)x`（ζ 同上）；**目标 = `c^W_anchor − 2pp`（≈ −33.11pp）**——T5 冻结 `c^W_anchor=-31.1094pp` 深于 −12pp，故 −12pp 的恶化结构上无法触发否决事件（2026-11-03 用户裁决修正预登记目标，理由随 manifest 冻结） |

---

## 2. 前置条件（数据根 / 模型 / 冻结件 / 版本 / 哈希）

1. **数据根**：仓库 `data/`（只读）。虚拟窗口只能落在 `row_domain=structural` 段内
   （边界 `structural_from=20260703`），且必须晚于模型训练截止日；
2. **模型/配置**：`data/paper/config.yaml`（生产冻结配置，`engine=v2`）与其冻结模型版本
   （当前 `model_version=24022`，`train_end_date=20260105`）；
3. **上游冻结件**（哈希不符即拒绝运行）：T4 收益仪器 manifest + 收益临界值、T5 回撤仪器
   manifest + 回撤临界值 + 回撤功效产物；
4. **T6 冻结件**：`docs/plans/v2/p5b_t6_rehearsal_manifest_<日期>.json`（正式件 `lineage.sequence=21`，
   campaign 内副本必须与之逐字节一致；正文的 campaign 与 sequence 由 `verify_runbook` 机器校验）；
5. **版本**：`pyproject.toml` 与 `CHANGELOG.md` 首条版本一致，且 `README.md` 引用本手册；
6. **机器**：单日族级实测约 20 s（五臂 × 生产 structural 窗口）；整批预计 > 20 分钟时**必须先
   取得用户授权**（见 §3 第 4 步与 §4 退出码 4）。

**当前冻结件哈希（稳定成员，可独立复算）**：

<!-- rehearsal-artifacts -->
| 产物 | sha256 |
|---|---|
| `rehearsal_manifest.json` | `700a58bbee7263a2e730d44b55c3858e1151bb1750ccfe8002b36c7efe3ad564` |
| `t4_manifest` | `2942d604719a3ef8e686bc0dc138812660be57114558a6da6d9c736f64059416` |
| `t4_thresholds` | `400c6306bb85f19fc3af528c8537f18c5d0a14f215e832fca068905b3ebdc808` |
| `t4_holdout` | `7189a17e0106614d1736b4ae3102327e0c427753f992672ad151e6db4a408030` |
| `t5_manifest` | `4dcd09b2ef2e789ea617e15a338e6bf1cbc3c09bdd2d9738e7396b6be8916b27` |
| `t5_power` | `4f5360f6fc152a572f433c4a85753dda9b80b485ec2242182bea954fda0a273e` |
| `t5_thresholds` | `945cb3190634dc9b5a0ec17c4173e247dff87c9812c795a4e24202f0bcae305b` |
<!-- /rehearsal-artifacts -->

> 该哈希块只收**稳定成员**（T6 manifest + 上游五链）。报告与判定表的哈希记录在报告本体与判定表
> `hash_chain` 内（三环互链），因此手册不会因每次重跑报告而失效。

---

## 3. 命令与参数逐条示例

所有命令在仓库根执行；正式演练根 `<campaign>` = `p5b_t6_rehearsal_20261104_r21`（`lineage.sequence=21`）。
更早的 `p5b_t6_rehearsal_20261102`/*_r2*…*_r11*/*_r12*…*_r16* 为**被取代** run（格式化整改、
首轮/复审整改过程中冻结），保留作谱系证据，**不参与任何读数**。
**默认 dry-run / 只读**：`--freeze` 与 `--report` 需 `--apply` 才写盘。

> 重冻（仅当校准器身份必须变更时）：`--freeze --apply --manifest-suffix _rN --lineage-sequence N
> --supersede-manifest-sha <前身 manifest> --supersede-calibrator-sha <前身校准器>
> --supersede-reason "<原因>"`，并**换新 campaign_id**（旧 campaign 的窗口/身份不可复用）。

### 3.1 冻结 manifest（盲态预登记）

```bash
python scripts/v2_p5b/rehearsal.py --freeze --campaign-id <campaign>            # dry-run：只算不写
python scripts/v2_p5b/rehearsal.py --freeze --campaign-id <campaign> --apply    # 冻结（独占创建）
```

冻结时执行：解析全部可用 structural 交易日 → 复算上游五链哈希 → 在冻结 B0 锚窗（500 样本）上
按 T5 求解器语义求 λ/ζ（101 点扫描 + ≤80 次细化 + 0.05pp 容差 + 最小干预）→ 生成并封存 manifest。
**冻结后不得按读数调整任何参数**。

### 3.2 登记 campaign + 五臂 + D0 fresh

```bash
python scripts/v2_p5b/rehearsal.py --register --campaign-id <campaign>
```

一次完成：`register_campaign`（含窗口连续性校验与数据态/代码态身份）→ 五个 `register_arm`
（生产扁平配置 + 显式冻结模型版本）→ `initialize_d0_fresh`（五臂同一空现金起点，逐臂盘上观测）
→ 把仓库冻结件复制为 `<campaign>/reports/rehearsal_manifest.json` 并逐字节核对
→ 采集生产根 `before` 快照写入 `production_before.json`（供报告阶段跨阶段比对）。
**D0 只能初始化一次**；重复注册或无注册臂初始化一律拒绝。

### 3.3 单日墙钟探针（先测后跑）

```bash
python scripts/v2_p5b/rehearsal.py --run --campaign-id <campaign> --days 1 --json
```

输出含 `per_day_seconds`、`projection`（单日实测 × **本次待跑日数**）与 `exceeds_threshold`；
单日探针本身不产生 `projection`（无可外推基线，`projected_seconds=null`）。
该步只推进 1 个虚拟日，**不产生收益样本**（D0 只作估值点）。

### 3.4 整批运行（>20 分钟须先授权）

```bash
# 未取得授权时：若 projected_seconds > 1200，命令以退出码 4 拒绝运行（不会缩量）
python scripts/v2_p5b/rehearsal.py --run --campaign-id <campaign> --authorization <授权记录.json>
```

授权记录 JSON 字段（缺一即拒绝）：

```json
{
  "authorized_at": "2026-11-02T22:00:00",
  "authorized_by": "user",
  "quote": "原话摘要",
  "basis": "单日实测 X s × N 日",
  "scope": {"campaign_id": "<campaign>", "max_virtual_days": N}
}
```

### 3.5 账本读取（只读，供审计/复算）

```python
from src.lazybull.v2.hosts.testbed.layout import layout_for
from src.lazybull.v2.hosts.testbed.ledger_repair import query_shards
from src.lazybull.v2.hosts.testbed.referee.ledger_source import paired_returns_from_ledger

layout = layout_for("<campaign 根>")
query_shards(layout, "base", start=None, end=None)          # 区间 + 连续性 + 链 digest
paired_returns_from_ledger(layout, "income_minus_5d", "base")  # 配对 d_t
```

`query_shards` 对 INVALIDATED 区间 fail-closed（`allow_invalidated=True` 仅审计旁路使用）。

### 3.6 读数 + 报告 + 判定表

```bash
python scripts/v2_p5b/rehearsal.py --report --campaign-id <campaign>          # dry-run：只算摘要
python scripts/v2_p5b/rehearsal.py --report --campaign-id <campaign> --apply  # 写报告与判定表
```

报告阶段**只读账本**：机械 A/A 从分片逐值复算；配对收益走 T3 只读查询门；裁判阈值只从 T4/T5
冻结产物读取（不重训、不改值）；H6 三条分别判定并给出证据字段路径。

### 3.7 校验（只读）

```bash
python scripts/v2_p5b/rehearsal.py --check --campaign-id <campaign> --json
```

校验项：仓库冻结件与 campaign 副本逐字节一致、上游五链哈希、机械 A/A 复算、判定表存在性与
**「无闸门通过字段」**、`h6_claim`/`h7_claim`/`gate_owner` 取值。

---

## 4. 退出码语义

| 退出码 | 含义 | 处置 |
|---:|---|---|
| 0 | 命令成功；`--report`/`--check` 下指**作者侧检查全部 pass** | 记录产物哈希 |
| 1 | 校验/前置/写域错误（manifest 不符、上游哈希不符、未注册、生产根命中、campaign 已存在） | 按报错修正后重跑；**不得**改阈值或跳步 |
| 2 | argparse 用法错误（缺子命令、参数互斥冲突） | 修正命令 |
| 3 | 作者侧检查存在 `fail` 或 `not_executed` | 如实登记；**不得**写成闸门通过 |
| 4 | 单批预计 > 20 分钟且无授权记录（设计 §9） | 先报预计耗时与依据，取得用户授权后加 `--authorization` 重跑 |

> 退出码 3 是**合法结局**：H6 的 `not_constructible`/`not_executed` 必须如实保留，
> 不得为了拿 0 而改报目标值、截断档位或删除臂。

---

## 5. 产物清单与哈希

产物根：`data/paper_shadow/p5b_testbed/<campaign_id>/`（**禁止**写入 T4/T5 的 campaign 目录）。

| 产物 | 位置 | 说明 |
|---|---|---|
| `rehearsal_manifest.json` | `<campaign>/reports/` | T6 冻结件副本（与仓库冻结件逐字节一致） |
| `rehearsal_report.json` | `<campaign>/reports/` | §7.2 报告（封存，确定性，无墙钟） |
| `rehearsal_report.md` | `<campaign>/reports/` | 报告可读版（含产物哈希与限制声明） |
| `rehearsal_verdict.json` | `<campaign>/reports/` | 判定表（H1/H6/H7 作者侧检查 + 三环哈希） |
| `machine_time_run.json` | `<campaign>/reports/` | 运行墙钟（逐日）、外推与授权哈希（**不入身份**） |
| `machine_time.json` | `<campaign>/reports/` | 报告阶段墙钟（**不入身份**） |
| `production_before.json` | `<campaign>/reports/` | register 阶段生产根快照（跨阶段零写入比对；生成时缺该文件 ⇒ 报告判「仅报告阶段」并要求 H1-6 未执行） |
| `production_before.json` | `<campaign>/reports/` | register 阶段的生产根快照（`data/paper`/`data/runs`/`data/models` 的文件数、字节数、路径摘要、最大 mtime）；报告阶段的 `after` 与之比对 ⇒ 生产零写入覆盖 **register → report** 全窗口 |

账本与事务结构（只读证据，不在 `reports/`）：`ledger/<arm>/<date>.json`（分片）、
`journal/<date>.json`（族级事务）、`_index.json`（唯一发布点）、`audit/`（链/审计）、
`failed/`（失败现场）、`staging/`（暂存，终态后回收）。

复算某产物哈希：

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('<产物路径>').read_bytes()).hexdigest())"
```

---

## 6. 验证步骤（独立复算）

### 6.1 独立复算机械 A/A（不信任运行期字段）

```python
from src.lazybull.v2.hosts.testbed.layout import layout_for
from src.lazybull.v2.hosts.testbed.rehearsal.aa import mechanical_aa

result = mechanical_aa(layout_for("<campaign 根>"), arm_base="base", arm_clone="mech_aa")
assert result["identical"] is True
assert result["compared_days"] > 0 and result["compared_fields"] == 6 * result["compared_days"]
```

比对字段固定为 `virtual_date / generation / cash / positions / fills / nav`（无成交日同样比对现金与
持仓），逐字段用规范化 JSON 哈希比较（无容差放宽）。**负例自证**：把任一日任一字段改动 1e-12，
`identical` 必须变为 `False`。

### 6.2 独立复算一条 H6（以收益臂为例）

```python
from src.lazybull.v2.hosts.testbed.layout import layout_for
from src.lazybull.v2.hosts.testbed.rehearsal.injection import income_daily_shift
from src.lazybull.v2.hosts.testbed.referee.ledger_source import paired_returns_from_ledger

sample = paired_returns_from_ledger(layout_for("<campaign 根>"), "income_minus_5d", "base")
mean = sum(sample["returns"]) / len(sample["returns"])
assert abs(mean - income_daily_shift()) <= 1e-9          # 方向为负且幅度 = −5δ
```

要点：位移臂是**确定性克隆 + 常数位移** ⇒ 配对路径方差为零，冻结 block 统计量在该样本上
**不可识别**；此时报告把 `z_point_estimate` 置空并标 `identifiable=false`，
**不得**把浮点抖动放大出的数值伪影当作 Z 点估计。

### 6.3 复算回撤臂的可构造性与锚读数

```python
from src.lazybull.v2.hosts.testbed.rehearsal.injection import arm_effect_plan
import json

doc = json.loads(open("<campaign>/reports/rehearsal_manifest.json", encoding="utf-8").read())
effects = arm_effect_plan(doc, doc["window"]["return_days"])
print(effects["dd_improve_12pp"].status, effects["dd_improve_12pp"].required_return_days)
```

`status=not_constructible` 表示锚 look（`T_anchor=500`）或冻结窗终点超出演练路径收益样本数；
此时**不得**改用更短锚或截断窗，只能如实登记。

### 6.4 三环互链与判定边界

```bash
python scripts/v2_p5b/rehearsal.py --check --campaign-id <campaign> --json
```

检查 `manifest_sha256` 与仓库冻结件一致、`upstream_links_verified=5`、`aa_identical=true`、
`forbidden_gate_fields=[]`、`h6_claim=h7_claim=evaluated_not_declared_pass`。

---

## 7. 回滚、清理与常见故障处置

### 7.1 回滚

1. **未发布 attempt 中断**：运行器启动即调用 `recover_campaign`——未发布 attempt 走
   `ABORTING→ABORTED`（新根移入 `failed/`，从可验证父代备份还原），同日以新的
   `family_txn_id/attempt_id` 从父代重算；**已发布代只做幂等补齐，永不重跑交易**；
2. **已发布分片缺失/清单或 index 落后**：`scripts/v2_p5b/ledger_audit.py --repair`（默认 dry-run）
   按 journal 幂等重建；**先隔离未发布残片**再推进；
3. **整批重跑**：禁止在同一 campaign 上重跑（分片/产物独占）。需要重跑时**换新 campaign_id**
   （并重冻 manifest，`lineage.sequence` 递增、登记前身哈希与原因）；
4. **回滚演练产物**：直接删除该 campaign 根（试车场根内，不涉生产根），或整体移除
   `data/paper_shadow/p5b_testbed/<campaign_id>/`；**不得**删除 T4/T5 campaign 目录。

### 7.2 清理

- `staging/` 每次运行收尾按 journal 终态回收（`prune_staging`）；手工清理前先确认 attempt 已终态；
- `temp/`（临时脚本/中间产物）与 `logs/`（临时日志）为 gitignore 目录，任务收尾应清空；
- `_index.json` / `manifests` / `audit` 由 journal 确定性重建，**不要手工编辑**。

### 7.3 常见故障与处置

| 现象 | 根因 | 处置 |
|---|---|---|
| `CampaignBusyError`（campaign 已被持有） | 另一运行器持锁 `.run_attempt`，或同主机已死持有者 | 确认无活跃运行后人工删除锁文件；**禁止**并发运行 |
| `LedgerInvalidatedError` | 区间含 INVALIDATED 终态（不可重建缺口） | 人工核对现场；该臂该日不得作为任何统计输入 |
| `LedgerRepairRequiredError` | 账本内残留未发布 attempt 分片 | 先 `ledger_audit.py --repair` 隔离，再推进 |
| `not_constructible`（DD 臂） | 锚 500 或冻结窗终点超出演练路径收益样本 | 如实登记；**不得**截断窗、换短锚或改报目标 |
| `not_executed`（H6 条目） | 配对样本不可得 / 锚 look 超出路径（不可构造） | 如实登记；不得写成 pass。**注意**：Z 点估计在确定性位移臂上必然不可识别（退化方差），它只作 `z_identifiable=false` 披露，**不构成** `not_executed` 的理由 |
| `RehearsalHostError`（估值行 0/多条） | 宿主当日未提交或状态快照损坏 | 检查 `failed/` 与 journal；按 §7.1 回滚该日 |
| 冻结/注册被拒（`已有 manifest`/`campaign 根已存在`） | 独占创建保护 | 换新 campaign_id 或重冻（新 sequence） |
| `--run` 退出码 4 | 单批预计 > 20 分钟且未授权 | 先报预算与依据，取得授权后加 `--authorization` |
| 上游哈希不符 | T4/T5 冻结件被改写/替换 | **停止**：禁止继续运行；先查明仪器身份变更 |

---

## 8. H1~H7 闸门对照表与限制声明

### 8.1 闸门对照表（作者侧检查 vs 用户判定）

| 门 | 条件（设计 v4 §8） | 本手册对应证据 | 判定权 |
|---|---|---|---|
| H0 治理 | 用户批准范围 + 评审闭合 + 契约批准 | 设计 v4 / F22 / F5 | 用户 |
| H1 机械 | ≥10 个 structural 虚拟日；多臂 A/A 逐值一致；首次驱动成功或合法 `already_run` 且恰一条完整估值；无生产写 | §6.1 复算 A/A、`_index.json` 已发布代、分片 `valuation.row_count`、§3.7 校验 | 用户 |
| H2 状态与恢复 | 部分/全部 roots × 未发布/已发布状态均有确定动作；已发布不重复入账 | T2 闭包 + `failed/`、journal 状态机 | 用户 |
| H3 账本 | 族级事务 ID/父代/发布点唯一；分片+manifest+index 一致 | `audit/shard_chain.json`、`audit/ledger_audit.json` | 用户 |
| H4 收益裁判 | 逐单元 Wilson 上界 ≤6%；功效与 MDE 完整 | T4 holdout/power（本演练**只引用**） | 用户 |
| H5 回撤裁判 | `DD-W`/`DD-I` 分别通过 A/A、注入与 MDE；方向一致 | T5 holdout/power/verdict（本演练**只引用**） | 用户 |
| H6 端到端 | 历史模拟 A/A 链路 + 已知效应方向检查 | `rehearsal_report.json` → `h6.items[*]`、`rehearsal_verdict.json` → `author_side_checks.h6` | 用户 |
| H7 文档 | 校准 manifest 哈希、运行手册、版本、回归、限制声明收口 | 本手册 + `rehearsal_verdict.json` → `author_side_checks.h7` | 用户 |

> 任何门的**通过与否都不由本手册宣布**；作者侧检查状态只有三种：`pass` / `fail` / `not_executed`。

### 8.2 限制声明（与 manifest `limitations` 逐条同源，共 10 条）

1. `HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE`：历史虚拟时钟演练，不是前向影子，不产生候选前向
   结论或生产建议；
2. 历史模拟只能证明「在**已观测依赖结构**下的链路与方向」，不能证明**未来真实分布不变**，也不能
   自动获得生产裁决权；
3. T4/T5 的训练与 holdout **共享同一对真实源**（`source_run_id` 相同），只以互斥日期块与独立
   realization/seed 隔离；真实源只有两批，不构成独立真实换种子源对验证；
4. 收益裁判的 **MA(H−1)** 合成只覆盖持有期重叠；**经验分位**是已测场景的有限样本结果，
   不是普适尾概率保证，不覆盖未知 regime、流动性冲击或未来结构变化；
5. 回撤臂的崩盘窗是 B0 最后 500 个收益样本上的**序号区间**（无自然日），演练路径只能按样本序号
   对齐复用：索引对齐不等于同一段历史；
6. 确定性位移臂（clone + 常数位移）的配对路径方差为零，收益裁判的 block 统计量在该样本上
   **不可识别**；不得把退化方差下的数值伪影当作 Z 点估计；
7. 机械 A/A（同路径克隆）只证明运行与账本逐值一致，**不得**充当统计 A/A；
8. `DD-I` 与 `DD-W` 分别判定，任一方向不可执行即该方向无结论；不得互相抵扣，不得用「未触发恶化
   否决」冒充改善检出；
9. 本演练**不产生检出率/MDE/显著性**；频率类读数一律只引用 T4/T5 冻结产物。

10. `INJECTED_ARM_VALUATION_DECOMPOSITION`：注入臂只保证 NAV 读数——`portfolio_value`/
    `market_value` 被读侧覆写后**不再**与持仓快照分解自洽（`position_value` 仍为真实市值）；
    该不一致只存在于注入臂的旁路读数，不进入成交/现金/持仓/日期。

# stk_holdertrade（股东增减持）Phase 0 PIT 审计

> 日期：2026-09-17 ｜ 结论：**PIT 合格，可进 Phase 1**（附 4 条硬工程约束）
> 路线图见 `CLAUDE.md` 数据层方向与 `/memories`（holdertrade → repurchase → disclosure_date）。
> 本审计只读 API，不写 raw、不建分区；审计脚本为一次性（已删除），结论即本节。

## 一、接口画像

| 项 | 实测 |
|---|---|
| 接口名 | `stk_holdertrade`（积分门槛已满足，8000 分账号可调） |
| 返回字段 | `ts_code, ann_date, holder_name, holder_type, in_de, change_vol, change_ratio, after_share, after_ratio, avg_price, total_share` |
| **缺失字段** | **无 `begin_date` / `close_date`** —— 没有"变动期间"，只能按**公告日**使用（对事件型因子足够：公告日当天事件信息完整） |
| 单次上限 | **单页 3000 行**（per-request，不是数据上限）；超出时**不报错**，默认按 `ann_date` 降序返回**最新 3000 行** |
| 分页 | `limit` / `offset` **有效且能取全**：实测 2024H1 分页合计 4559 行（去重 4478）与**逐月拼装逐行一致**；2019 全年分页到取空 = 19571 行；offset 21000 仍有效，无总量硬上限。**跨页会重复行**（同日行被切到两页，81/4559 ≈ 1.8%）⇒ 分页后必须去重 |
| 单股查询 | `ts_code` 不带日期可拉**全历史**（600519.SH：16 行，覆盖 2010~2025，7 个年份） |
| 历史可回溯 | 2018 / 2021 / 2024 / 2026 窗口均可拉取 ✓ |

## 二、字段完整度（4 个披露季窗口合并 8486 行）

| 字段 | 缺失率 | 说明 |
|---|---|---|
| `ann_date` | **0.00%**（且 100% 为合法 8 位日期） | PIT 锚点可用 |
| `change_vol` / `change_ratio` / `holder_name` / `holder_type` / `in_de` | **0.00%** | 事件核心字段齐全 |
| `total_share` | 7.51% | 并非每笔都披露总股本 |
| `after_share` | 16.72% | 变动后持股数 |
| `after_ratio` | 20.43% | 变动后持股比例 |
| `avg_price` | 38.49% | 均价缺失可解释（大宗/集中竞价未必披露） |

取值分布：`holder_type` = C(公司) 4180 / G(高管) 2216 / P(个人) 2090；
`in_de` = DE(减持) 6313（74%）/ IN(增持) 2173（26%）。

## 三、键唯一性与去重语义（重要）

**该接口没有自然唯一键**（合并样本 8486 行）：

| 候选键 | 重复行数 | 占比 |
|---|---|---|
| `ts_code + ann_date` | 5691 | 67.1% |
| `ts_code + ann_date + holder_type + in_de` | 5457 | 64.3% |
| `ts_code + ann_date + holder_name` | 4657 | 54.9% |
| `ts_code + ann_date + holder_name + in_de` | 4564 | 53.8% |
| `ts_code + ann_date + holder_name + in_de + change_vol` | 2550 | 30.0% |
| **全字段完全相同** | **2094** | **24.7%** |

含义：同一公告日、同一股东可能有多笔明细行（分笔披露），且**存在 1/4 的整行重复**。
⇒ Phase 1/2 必须**按全字段去重**，并且**不得假设任何弱键唯一**；因子聚合按 `(ts_code, ann_date)`
汇总（增持/减持分别求和）才是有语义的口径。

## 四、覆盖密度（稀疏性）

- 2024 披露季（03-01~04-30）：517 只股票 / 1516 行 / 50 个公告日；
  **每日行数 min/中位/max = 3 / 24 / 191**，**每日股票数中位 = 12 只**（全市场同期已上市 5607 只）。
- 历史密度更高：2019 年任一单季窗口即触及 3000 行上限。
- ⇒ 属**事件型稀疏数据**：日度覆盖 <1%，因子必须走「状态保留 + `freshness_days` + 公共指数衰减」
  （沿 pledge/dividend 既有契约），且**稀疏列会被训练入口 0.6 缺失率门禁删除**——Phase 2 需要
  显式处理（分列开关或对标记列豁免），不得让整族因子被静默丢弃。

## 五、PIT 严格性验证（决定性证据）

| 检查 | 结果 |
|---|---|
| 窗口边界一致性（未截断）：窄窗 20240401~0430（564 行）vs 宽窗 20240301~0531 的 4 月切片（564 行） | **逐值 / 逐行完全一致 ✓** |
| 截断后的"不一致"是否是数据漂移 | **否**：是 3000 行封顶返回最新行造成的假象（1~2 月被丢弃） |
| 是否存在"当前值快照"型字段 | 未发现：`change_vol/change_ratio/after_*/avg_price` 均为事件时点值 |

⇒ 接口对同一 `ann_date` 区间返回确定内容，按 `ann_date` 因子化**不泄露未来信息**。

## 六、Phase 1 硬约束（审计产出，必须落到实现与测试）

1. **必须分页读满**：`limit=3000` + `offset` 递增直到返回空（**触顶页 =3000 时必须继续翻页**，不得当作完整数据）；
   分页后**全字段去重**（跨页重复 ≈1.8%，源内重复 24.7%）；测试断言用「分页合计 == 逐日期窗口独立拉取合计」锁定完整性。
2. **全字段去重**（24.7% 整行重复），去重后仍保留 `(ts_code, ann_date)` 多行聚合语义；
   禁止引入弱键唯一性假设。
3. **PIT 锚点 = `ann_date`**：分区/水位按公告日推进（沿 `dividend` 的按年分区 + `report_rc` 的增量水位模式）；
   单股回补可走 `ts_code` 全历史查询。
4. **稀疏性**：默认关闭开关；因子构建需 `freshness_days` + 指数衰减；训练入口缺失率门禁需显式对待。

## 七、判定

**PIT 合格 → 进入 Phase 1（下载 + 水位 + loader + ensure 挂点 + 测试）。**
Phase 1 完成后按既有协议走 Phase 2（因子构建）/ Phase 3（离线体检，含对 pledge/holder/forecast 的增量）
/ Phase 4（先过 v0.122.0 **信号层尺子**，再决定是否跑全量净值 A/B）。

预期管理（沿既有先例）：新数据集被否的概率不低（参照 dividend / consensus_revision），
因此 Phase 1 采用**薄接入、默认关、可回退**。

## 八、Phase 1 落地（2026-09-17 完成）

**落盘布局（决策）**：`data/raw/stk_holdertrade/YYYY-12-31.parquet`——**按 `ann_date` 年分区**。
选年分区而非整体单文件/半年/季度：① 全年实测 1.9 万行（2019）→ 年文件 <2MB，按季/半年分区过碎；
② 当前年度续传 + 合并只动 1 个分区；③ 与 dividend / report_rc 年分区语义一致，复用 `Storage` 既有能力。

| 组件 | 位置 |
|---|---|
| 客户端 getter | `tushare_client/alt.py::get_stk_holdertrade`（docstring 标注分页限制） |
| raw 核心 | `src/lazybull/data/holdertrade_raw.py`（下载/去重/年分区/水位/加载） |
| 薄包装 | `scripts/raw_download/holdertrade.py` + `--download stk_holdertrade`（已入 `ALT_DATASETS`） |
| loader | `DataLoader.load_stk_holdertrade()` |
| 纸面 ensure | `features/ensure/downloads.py::_try_download_stk_holdertrade` + `factor_load.py` 挂点（**仅维持 raw 新鲜**，失败仅告警） |
| 测试 | `tests/test_holdertrade_raw.py`（9 项，含**分页取全 + 跨页重叠行**） |

**冒烟证据（真实 API）**：2024-03-01~04-30 拉取 **1476 行**（去重后；Phase 0 审计 raw 计数 1516，
差值为整行重复）✓；分区仅 `2024-12-31` ✓；水位 `20240430` ✓；二次调用零请求（幂等）✓。

**项目共识**：本次教训（单页上限 + 超限不报错 + 跨页重复）已写入 `CLAUDE.md` /
`.github/copilot-instructions.md` 的《TuShare 分页读取契约》与《stk_holdertrade 数据集契约》。

## 九、生产全历史回补（2026-09-17）

`python scripts/download_raw.py --start-date 20100101 --end-date 20260917 --download stk_holdertrade`
→ **201 个月窗口、39 秒、退出码 0、无错误**；落盘 **17 个年分区**（2010-12-31 ~ 2026-12-31），
年文件 91KB~760KB（最大 2020 年），**全库 176,762 行**（已全字段去重），水位 `20260917`。

| 年 | 行数 | 年 | 行数 |
|---|---|---|---|
| 2010 | 1,795 | 2019 | 19,477 |
| 2011 | 2,274 | 2020 | **20,540**（峰值） |
| 2012 | 2,835 | 2021 | 18,921 |
| 2013 | 4,288 | 2022 | 15,339 |
| 2014 | 5,564 | 2023 | 13,057 |
| 2015 | 10,636 | 2024 | 9,873 |
| 2016 | 8,068 | 2025 | 13,679 |
| 2017 | 8,305 | 2026（至 09-17） | 7,939 |
| 2018 | 14,172 | | |

全库字段缺失率（相对 Phase 0 的 4 窗口样本，长历史下更低）：
`change_ratio` 0.02%、`after_share` 12.3%、`after_ratio` 16.1%、`total_share` 10.4%、`avg_price` 28.5%；
`ts_code/ann_date/holder_name/holder_type/in_de/change_vol` 均为 0%。
分布：`in_de` DE 132,093(74.7%) / IN 44,669(25.3%)；`holder_type` C 94,466 / G 48,765 / P 33,531。

`DataLoader().load_stk_holdertrade()` 端到端读回 176,762 行 ✓；后续每日由纸面 ensure 自动增量续传。

# 因子去重 / 口径交换 A-B 实验预登记（v0.119.0，2026-09-16）

> 目的：用**单变量** walk-forward 对照实验，回答因子体检给出的两个最大结构性问题——
> ① 剔掉「同簇非代表」的 49 列是否会改变 OOS 表现；② 去重口径（保留 plain vs 保留 `_sz` 市值中性化版）
> 是否才是真正起作用的那一维。**实验前冻结判据，禁止事后改判据或做多臂搜索。**

## 一、前置冻结（实验开始时的状态）

| 项 | 值 |
|---|---|
| 代码态 | git `dcd902b`（工作区干净） |
| 数据态 | `cs_train` 最新分区 `20260702`；`raw/daily` 2026-07-31 |
| 标签 | `neu_y_ret_20`（`--neutral-label-blend-weight 0`） |
| 折定义 | `--split-count 14 --final-date 20260105 --train-window-years 6 --test-window-months 6 --val-ratio 0.2` |
| 折窗口 | 折0 20181126~20190524 … 折13 20250612~20260105（与 2026-09 各批次一致） |
| 生产模型版本 | `latest_model_version.txt` = `24052`（实验期间**每个臂跑完即还原**，避免实验模型被纸面交易取用） |

## 二、臂定义（单变量，逐臂只动 exclude 清单）

| 臂 | `--factor-prune --factor-exclude-file` | 列数 | 清单来源 |
|---|---|---|---|
| 基线 | 不启用（全 154 列） | 154 | — |
| A 去重 | `configs/factor_exclude_health_dedup_v1.json` | 105 | 因子体检同簇非代表（43 簇，保留 \|ic_ir\| 最高者） |
| D 口径交换 | `configs/factor_exclude_health_plain_v1.json` | 105 | 同一簇划分，代表改为优先保留非 `_sz` 口径（28 对换边） |

要点：A 与 D **列数相同、簇划分相同**，差别只在孪生对保留哪一侧（plain vs `_sz`）；
两臂均为 49 列裁剪，裁剪清单**不包含** 17 个市场级截面常数（北向 13 + 市场环境 4），
因此 `--enable-north-features` 等开关在三个臂中含义一致。

## 三、冻结配置（命令模板）

除 `--factor-prune/--factor-exclude-file` 外，三个臂**完全一致**：

```
python scripts/walk_forward.py --algorithm xgboost --split-count 14 --final-date 20260105 \
  --train-window-years 6 --test-window-months 6 --val-ratio 0.2 --label neu_y_ret_20 \
  --neutral-label-blend-weight 0 --task regression --label-transform cs_zscore --objective mse \
  --n-estimators 500 --max-depth 5 --num-leaves 63 --learning-rate 0.03 --subsample 0.8 \
  --colsample-bytree 0.3 --min-child-weight 200 --reg-alpha 0.03 --reg-lambda 7 --gamma 0.5 \
  --rank-weight-topk 120 --rank-weight 100 --rank-weight-topk-weight-mode linear_decay \
  --early-stopping-rounds 50 --early-stopping-metric rank_ic_daily --min-best-iteration 30 \
  --time-decay-half-life 0 --freshness-strategy state_keep_event_decay \
  --event-freshness-half-life-days 120 \
  --enable-fundamental-features --enable-alt-features --enable-margin-features \
  --enable-cyq-features --enable-fund-features --enable-express-features \
  --enable-north-features --enable-lhb-features --enable-consensus-features \
  --enable-enhanced-features --enable-cashflow-quality-features \
  --ensemble-seeds 42,61,82 --ensemble-seed-keep-top-ratio 1 --ensemble-seed-keep-min-models 3 \
  --stagger-tranches 2 --position-sizing kelly --kelly-vol-window 60 --kelly-max-leverage 0.2 \
  --oos-backtest --oos-backtest-months 0 --bt-top-n 20 --bt-initial-capital 1000000 \
  --bt-sell-timing open --bt-min-list-days 365 --bt-max-weight-per-stock 0.15 \
  --no-deploy-train --data-root ./data \
  --batch-run-id <id> --batch-period-label <arm> --wf-summary-csv <batch_dir>/raw/walk_forward_summary_<arm>_0001.csv
```

说明：`--stagger-tranches 2 --bt-top-n 20` 为用户确认的当前最优配置；`--no-deploy-train` 仅省去部署模型
（不参与 OOS 指标），且实验结束后统一把 `latest_model_version.txt` 还原为 `24052`。
置信区间与判据由 `scripts/compare_wf_fold_subset.py`（全 14 折，无 `--splits`）计算。

## 四、预登记判据（实验前确认，2026-09-16）

对每个臂 A、D 分别与基线比较，全部指标取自**同一 14 折链式口径**：

1. **ΔCAGR > 0** 且其**折级自举 95% 区间下限 > 0**（噪声带：对 14 个折的逐折收益差做 1000 次有放回重采样）；
2. **Δ夏普 > 0** 且其折级自举 95% 区间下限 > 0；
3. **不得牺牲最大回撤**：ΔMaxDD ≥ 0（Δ = 臂 − 基线，正号表示回撤更浅）；
4. **逐折一致性**：逐折收益差的方向与总差同向的折数 ≥ **10/14**。

判定：
- 三条数值判据 + 逐折一致性**全部满足** → 记为「通过」（可进入下一阶段：扩大样本/换数据态复跑确认）；
- 任一不满足 → 记为「不通过」，**不得**用其他口径、其他子集或换判据翻案；
- 结果无论正负都写入 `CHANGELOG.md` 与本文件「结果」节，并按需登记进风险登记文档。

判读矩阵（预登记，避免事后编故事）：

| A 结果 | D 结果 | 结论 |
|---|---|---|
| 通过 | 通过 | 去重本身有效，且与孪生口径无关 |
| 不通过 | 通过 | 有效的是「口径」（size 暴露方向），而非去冗余 |
| 通过 | 不通过 | plain 口径更优，`_sz` 中性化损害本策略（价值红利天然含 size 暴露） |
| 不通过 | 不通过 | 保持全特征，去重/换口径在 OOS 上无收益 |

## 五、运行与产物

- 三个臂各写一个批次目录：`data/walk_forward/batches/factor_ab_<时间戳>_{base,A,D}/raw/`；
- 终端全量日志：`logs/factor_ab_<时间戳>.log`；
- 汇总对比产物：`data/reports/wf_fold_subset/factor_ab_<时间戳>/`（`折子集对比.csv`、`逐折对比.csv`、`判据结论.csv`、`折子集对比说明.md`）；
- 自举与判定：`scripts/compare_wf_fold_subset.py --bootstrap 1000`（默认即开）会输出「判据结论.csv」——
  逐臂给出 ΔCAGR/ΔMaxDD/Δ夏普 的点估计 + 折级自举 95% 区间 + 逐折同向数 + 四条判据的通过与否；
  阈值固定为「ΔCAGR、Δ夏普 区间下限 > 0」「ΔMaxDD ≥ 0」「逐折同向 ≥ ceil(0.7 × 折数)」。
- 已知口径说明：训练流水线会在裁剪清单之外**按折自动剔除**高缺失（<0.6）与常数截面列
  （例：早期折会剔掉 north_turnover 系列与 margin/cyq 部分列），因此每折实际入模列数 ≤ 158；
  三个臂面对同一剔除逻辑，A/D 与基线的差异仅来自 49 列裁剪清单本身。
- 串联执行（**禁止并行**，当前机器内存已在边缘），预计 3 × 1.5~2.0 小时 ≈ 4.5~6 小时。

## 六、结果（2026-09-16 完成）

运行：驱动 `temp/run_factor_ablation_20260916.ps1`，启动 07:11:57，三臂串行（base 1h57m / A 1h18m / D 1h20m，
均退出码 0）；批次目录 `data/walk_forward/batches/factor_ab_20260916_071157_{base,A_dedup,D_plain}/raw/`；
产物 `data/reports/wf_fold_subset/factor_ab_20260916_071157/`；日志 `logs/factor_ab_20260916_071157.log`；
实验期间 `latest_model_version.txt` 已按臂还原为 `24052`。

### 6.1 主结果（全 14 折链式）

| 运行 | 全周期CAGR | 全周期最大回撤 | 全周期夏普 | ΔCAGR | ΔMaxDD | Δ夏普 | 折收益为正 | 逐折同向 |
|---|---|---|---|---|---|---|---|---|
| 基线（154 列） | 0.2216 | −0.2400 | 0.9066 | — | — | — | 11/14 | — |
| A 去重 | 0.1637 | −0.3215 | 0.6788 | **−0.0579** | **−0.0815** | **−0.2278** | 9/14 | 12/14 |
| D 口径交换 | 0.1637 | −0.3215 | 0.6788 | −0.0579 | −0.0815 | −0.2278 | 9/14 | 12/14 |

折级自举 1000 次（种子 42）：ΔCAGR 95% 区间 [−0.1512, +0.0294]、ΔMaxDD [−0.1616, +0.0294]、
Δ夏普 [−0.6040, +0.1176]；逐折差值均值 −2.44pp（中位 −2.13pp，仅 5/14 折变好）。

### 6.2 判定（按预登记§四）

**不通过**：ΔCAGR > 0 不满足（−0.0579，区间下限 −0.1512 < 0）、Δ夏普不满足（−0.2278，下限 −0.6040 < 0）、
ΔMaxDD ≥ 0 不满足（回撤加深 8.15pp）；仅逐折同向（12/14 ≥ 10）成立。
→ 按预登记：**不采纳因子去重裁剪，保持全特征**。
自举区间跨零说明在 14 折样本下差异未达 95% 显著，但**没有任何一维变好、回撤明确变差**，
且中位逐折差为负，结论方向明确（不采纳）。

### 6.3 两个必须登记的发现

1. **D 臂在现行裁剪语义下是无效对照（A ≡ D，逐位相同）**：`--factor-exclude-file` 的裁剪在
   `src/lazybull/ml/train_core/prepare.py` 中会**成对联动删除** `zscore_x` 与 `zscore_x_sz`
   （“避免 schema 不一致”的既有设计），因此“保留 plain / 保留 `_sz`”两种清单在效果上等价。
   ⇒ **size（市值中性化）暴露这一维本轮无法回答**，需换实验载体（特征集开关或改 cs_train 列集合），
   属新工作、需重新预登记。
2. **体检去重清单与联动语义冲突**：体检按 `|ic_ir|` 保留代表、剔除非代表；但剔除非代表的 plain 列会
   **连带删除它的 `_sz` 孪生（往往就是那个代表）**。A 清单 49 列实际删了 **82 列**（33 列联动），
   其中估值/质量/现金流家族的 `zscore_*` 几乎整族消失（基线与臂的折内入模列数：140~154 → 58~72）。
   ⇒ 本轮实质上测的是“整族删除”，而非“去重”；这也是变差的最可能原因。
   后续若要做真正的冗余消融，必须先取得“可只删一侧”的裁剪语义。

### 6.4 后续动作

- 生产因子裁剪保持关闭（`factor_prune=$false`，与本次结论一致）；`configs/factor_exclude_health_*.json`
  保留为实验输入，**不作为生产裁剪依据**。
- B（弱信息+未用 16 列）与 C（并集）臂未跑：A 的失败主因是联动删族，B 只涉 16 列且多为非 zscore 列，
  价值有限；若要跑，需先预登记并单独判定。
- 真正待解的问题变为：① 如何在“只删一侧”的语义下重测去重；② size 暴露的归因实验。

## 七、B0 噪声带对照（2026-09-16，后续实验的判据基准）

设计：与基线**完全同配置**，仅把集成种子由 `42,61,82` 换为 `43,62,83`（单臂 1h55m）。
产物 `data/reports/wf_fold_subset/noise_band_b0_20260916_123214/`。

| 指标 | 噪声臂（仅换种子） | 82 列裁剪臂（对照） |
|---|---|---|
| ΔCAGR | **−3.62pp**，自举区间 [−7.71, +0.36] | −5.79pp，[−15.12, +2.94] |
| ΔMaxDD | **−0.59pp**，[−6.91, +4.30] | −8.15pp，[−16.16, +2.94] |
| Δ夏普 | **−0.144**，[−0.304, +0.012] | −0.228，[−0.604, +0.118] |
| 逐折差值 SD | 3.53pp | 8.13pp |

**判据分级（后续实验强制）**：
1. **主判据 = ΔMaxDD**（噪声带实测仅 −0.59pp）；裁剪臂 −8.15pp **超出噪声区间** → “回撤变差”是唯一站得住的结论。
2. **辅判据 = ΔCAGR / Δ夏普**：噪声带 3.6pp / 0.14，可检出下限 **5~8pp**；裁剪臂 −5.79pp **落在噪声区间内，不可区分**。
3. 逐折同向数保留作一致性检查（噪声臂自身就 10/14，故门槛 70% 对单臂对比偏宽松，需与主判据合看）。

**分辨率成本规律**：MDE ∝ 1/√n（折数）或 1/√k（种子组数平均）——
28 折仅换 1.41×（成本 2×）、3 组种子平均换 1.73×（成本 3×），**性价比低**。
因此：**预期效应 < 3pp 的列级候选不得用 WF 追**；要测请先做低成本、低噪声的前置指标
（覆盖率/可得域、换手、分布偏移、MaxDD）。

**对 A-B 实验结论的修正读法**：判定仍为“不采纳裁剪”（回撤变差已足以否决），
但**理由收窄为“回撤显著变差”**；CAGR/夏普的差异不足以作为证据。

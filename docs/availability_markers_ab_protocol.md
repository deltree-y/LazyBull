# 可用性标记因子 A/B 实验协议（预登记）

> 目的：验证「把**结构性缺失**显式化为 `has_*` 标记」是否能改善 OOS 净值判据。
> 预登记时间：2026-09-16（实验前冻结命令与判据，结果无论正负都写入本文件与 CHANGELOG）。

## 一、假设与依据

诊断 v2（`data/reports/factor_diagnosis/20260916/`）结论：一致预期（覆盖 44~63%）、
两融（**标的资格** 2013 年 22% → 2026 年 71%）、基金持仓、业绩快报的缺失是**结构性的**
（资格名单 / 覆盖 / 披露口径），且缺口基本只是**规模代理**（有值-缺失的规模分位差 +0.75~0.90，
标签中位差≈0）。因此：

- **H1**：把「是否可得」显式喂给模型，可以减少模型用家族因子残余幅度去代理「资格」这一伪信号；
- **H0**：该信息对 OOS 无增量（或纯噪声），或效应小于噪声带（列级 0~1pp）。

## 二、冻结配置

对照臂：`data/walk_forward/batches/factor_ab_20260916_071157_base/raw/`（154 列基线，
与 `docs/factor_pruning_ab_protocol.md` 同一次运行）；
实验臂：`data/walk_forward/batches/avail_markers_20260916_v2/raw/`。

除 `--enable-availability-markers` 外**完全一致**（与 A-B 实验同一命令模板）：

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
  --enable-availability-markers \
  --ensemble-seeds 42,61,82 --ensemble-seed-keep-top-ratio 1 --ensemble-seed-keep-min-models 3 \
  --stagger-tranches 2 --position-sizing kelly --kelly-vol-window 60 --kelly-max-leverage 0.2 \
  --oos-backtest --oos-backtest-months 0 --bt-top-n 20 --bt-initial-capital 1000000 \
  --bt-sell-timing open --bt-min-list-days 365 --bt-max-weight-per-stock 0.15 \
  --no-deploy-train --data-root ./data
```

标记（4 个，运行时派生，不写回特征分区）：
`has_cons_coverage` / `has_margin_balance` / `has_fund_holding` / `has_express_data`。

## 三、预登记判据（与因子裁剪 A-B 实验同口径）

全部指标取自**同一 14 折链式口径**（`scripts/compare_wf_fold_subset.py`）：

1. **ΔCAGR > 0** 且折级自举 95% 区间下限 > 0；
2. **Δ夏普 > 0** 且折级自举 95% 区间下限 > 0；
3. **不得牺牲最大回撤**：ΔMaxDD ≥ 0（正号 = 回撤更浅）；
4. **逐折一致性**：逐折收益差与总差同向折数 ≥ **10/14**。

判定：四条全满足 → 「通过」；任一不满足 → 「不通过」，不得换口径/换子集翻案。
**排期约束**：本次实验**不新增/删除任何特征分区**，因此不存在数据态漂移；
但工作区带未提交改动（git 标记不同），对比时使用 `--allow-state-mismatch` 并在结论中登记该例外。

## 四、已知执行事件（必须随结论一起报告）

1. **首轮 14 折全灭（v1，`avail_markers_20260916`）**：`execute_split_training` 的 OOS 测试集由
   `load_features_data` 单独加载、不经 `prepare_training_data`，因此不含 `has_*` 列 →
   `X_test_features = df_test_eval[feature_columns]` 抛 `KeyError`（14/14 失败，0 个成功切分）。
   修复：OOS 评估侧（与 MLSignal 同语义）在切片前调用
   `ensure_availability_markers(df_test_eval, feature_columns)`，并对缺列给出显式报错；
   回归测试 `tests/test_availability_markers.py::test_oos_eval_frame_derivation_matches_training`。
2. **单折冒烟（`avail_markers_smoke_20260916`，1 seed）**：退出码 0，训练侧与 OOS 侧均派生 4 个标记 ✓。
3. v2 为正式实验臂；v1 不计入结果。

## 五、结果（2026-09-16，v2 运行 14/14 折成功，约 1h58m）

批次：`data/walk_forward/batches/avail_markers_20260916_v2/raw/`（`wf_20260916_184356_6d3ca6d7`）；
对比产物：`data/reports/wf_fold_subset/avail_markers_20260916/`（含 `判据结论.csv`）；
执行例外：**无**。两臂 `data_state_id` 均为 `8d33174f`（数据水位与代码标记一致），
未新增/删除任何特征分区；首次对比预先带了 `--allow-state-mismatch`，随后**在不开该标志下复算**
（`avail_markers_20260916_strict/`）仍退出码 0、结论一致，因此该例外实际未被使用。

| 指标（全周期链式） | 基线 | markers 臂 | Δ |
|---|---|---|---|
| CAGR | 0.2216 | 0.1632 | **−5.84pp**（自举 95% [−10.36, −1.78]） |
| 最大回撤 | −0.2400 | −0.2612 | **−2.12pp**（回撤更深，[−7.65, +2.01]） |
| 夏普 | 0.9066 | 0.6756 | **−0.231**（[−0.421, −0.073]） |
| 折收益为正 | 11/14 | 8/14 | — |
| 逐折同向 | — | 11/14 | ≥10 ✓ |

**判定：不通过**（预登记四条判据过 1 条：CAGR ✗ / 夏普 ✗ / 回撤 ✗ / 同向 ✓）。

### 机理核查（结论的解释力边界，必须随判定一起报告）

1. **标记几乎未被模型使用**：v24123（本臂末折集成模型，158 列 = 154 + 4 标记）的 gain 份额为
   `has_margin_balance` 0.616%（9 次分裂）、`has_cons_coverage` 0.093%（2 次）、
   `has_express_data` 0.071%（1 次）、`has_fund_holding` **0.000%（0 次）**，合计 ≈0.78%。
2. **信号层复核（2026-09-17 补做，见下节）**：本臂在**日频配对口径**下同样轻微为负
   （Top30 Δ平均持有期收益 −10.89 bps，95% 区间 [−24.89, −0.02]，相对 −9.5%），
   与净值口径方向一致；而「仅换种子」臂的同一读数为 −1.34 bps [−11.24, +7.85]（中心≈0）
   ⇒ 本臂的负向读数**不是纯净值噪声**，但幅度仅与尺子噪声带（±10 bps）相当，属**边缘可判定**。
3. **扰动带解释**：给 `colsample_bytree=0.3` 的模型加 4 列（+2.5% 列数）会改变每棵树的列抽样流与
   早停轨迹——本仓库已实测同类现象（`core_state` 加 2 列即让早停停点 628→99 棵，
   逐折 lift 变化 0.03~0.11）。逐折差异也**有正有负**（fold5 +2.17pp、fold6 +1.95pp，最差 fold3 −9.49pp）。
4. **与列裁剪实验对照**：裁剪 A 臂的日频读数 −28.45 bps（−24.9%）**明显超出**噪声带，
   而本臂 −10.89 bps（−9.5%）**贴着**噪声带 ⇒ 「删掉有信息的列」代价远大于「加进几乎无用的列」，
   两者共同说明列集改动本身带系统性成本（`colsample` 稀释 / 抽样流改变）。
5. 因此本结论登记为「**不采纳 + 边缘可判定，且机理归因于列集扰动而非可得性信息本身**」，
   **不得**当作「可得性信息无用」的证据。

### 日频信号级复核（v0.122.0 尺子，2026-09-17）

产物：`data/reports/wf_signal_compare/20260917_ledger/`（14 折 × 1725 个交易日，同日配对 + 40 日分块自举）。
基线水平：Top20/Top30 平均持有期收益 ≈ 114 bps，命中率 ≈ 50%；噪声带（换种子臂）= ±10 bps。

| 臂 | Top30 Δ平均持有期收益 | 95% 区间 | 相对基线 | 判定 |
|---|---|---|---|---|
| 换种子 B0（噪声带） | −1.34 bps | [−11.24, +7.85] | −1.2% | 中心≈0 |
| 裁剪 A | −28.45 bps | [−57.29, +0.41] | −24.9% | 明显超出噪声带 |
| **本臂 markers** | **−10.89 bps** | **[−24.89, −0.02]** | **−9.5%** | 贴噪声带边缘 |

### 后续可行方向（仅登记，不启动）

- 给树模型加 0/1 缺失标记本就不是表达"可得性"的最佳载体（树可用 NaN 分支自行学缺失方向，
  实测 gain≈0 与之吻合）；若要真正利用资格/覆盖信息，正确载体是**改样本域或标签**：
  ① 在「有覆盖子域」内单独训练/评估并做两域对照；② 分域校准（把可得性作为分组变量而非特征列）。
- 任何此类实验必须**预登记判据**，且先过 v0.122.0 的信号层尺子（相对降幅 ≥9% 才值得跑全量净值 A/B）。

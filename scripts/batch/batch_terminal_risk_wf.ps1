# batch_terminal_risk_wf.ps1
# 期末异常亏损风险模型滚动 Walk-forward 批量脚本（第一阶段研究型 WF）
#
# 每折：滚动 Train（由 $train_window_years_list 决定年数）+ Val（早停段，由
# $val_months 决定）+ ES（6 个月，互不重叠）→ 独立概率质量报告；全部完成后
# 自动运行 summarize_terminal_risk_wf.py 拼接
# summary、按 _d*/_lr*/_w*/*_s* 消融后缀 × 折 meta 超参签名分组输出调参分
# （tuning_score = 0.5×lift几何均值 + 0.5×组内lift最小值，绝对量纲跨 batch 可比）
# 并按组判定门禁（组内 lift 最小值 >= 阈值才建议进入第二阶段）；随后与
# tuning_history.csv 台账按签名聚合自动比较历史、醒目打印历史最优超参与当次
# 是否刷新（--no-history 跳过台账追加）。
#
# 注意：随机种子与训练窗口年数都是**签名维度**（v0.108.5 起）：多种子/多窗口
# 的折不会被并进同一组，因此不能靠多跑几个种子去“凑”一个好看的最小组内最小
# lift；集成（多种子平均预测）必须作为单独方案显式评估。
#
# v0.109.0 协议变更（必须登记）：早停只用 Val 段，ES 段只用于评估/门禁。
# 旧批次的门禁 lift 算在早停选择段上（乐观偏差），其目录名不含 _v{N}m 后缀、
# 签名带 valm=0；新旧不可混组比较（分组已按签名隔离，勿手工并组）。
#
# 每折训练使用 --fixed-name（研究型折产物）：注意它自 v0.108.0 起是**附加**
# 固定名别名，模型本体始终经 ModelRegistry 版本化注册（v{N} 永不覆盖），
# 因此同一折目录反复训练不会丢历史模型，汇总工具读固定名别名不受影响。
#
# 本脚本不含 C 段（校准）/ V 段（阈值）/ OOS 组合回测——这些依赖第二阶段的
# 持仓快照与 policy，完整 WF 待二阶段后补齐（方案 5.2/5.5）。
#
# 示例启动：
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_terminal_risk_wf.ps1

# ============================================================
#  参数配置区
# ============================================================

$data_root   = "data"
$wf_root     = "data\walk_forward\terminal_risk_wf"   # 各折输出父目录
$device      = "cuda"     # cuda | cpu（GPU 不稳定时可切 cpu）
$run_summary = $true      # 全部完成后运行汇总工具

# ── 折定义（ES 段 6 个月互不重叠，覆盖 2022H2..2026H1）────────────
# 三段由配置派生（日期为自然日边界，训练脚本按交易日历过滤，并按
# label_end_date < 下一段起点 做多期限标签隔离）：
#   ValStart   = EsStart - $val_months 月
#   ValEnd     = EsStart - 1 天
#   TrainEnd   = ValStart - 1 天
#   TrainStart = ValStart - N 年（N = $train_window_years_list）
# 即 Train 不再直接贴到 ES 起点：中间一段留给早停（Val），ES 保持纯评估。
# Label      : 目录名与汇总标识
# EsStart/EsEnd : ES 评估段自然日边界
# Selected   : $false 跳过该折（结果保留不删）
$folds = @(
    [PSCustomObject]@{ Label = "2022H2"; EsStart = "20220701"; EsEnd = "20221231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2023H1"; EsStart = "20230101"; EsEnd = "20230630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2023H2"; EsStart = "20230701"; EsEnd = "20231231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2024H1"; EsStart = "20240101"; EsEnd = "20240630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2024H2"; EsStart = "20240701"; EsEnd = "20241231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2025H1"; EsStart = "20250101"; EsEnd = "20250630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2025H2"; EsStart = "20250701"; EsEnd = "20251231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2026H1"; EsStart = "20260101"; EsEnd = "20260630"; Selected = $true  }
)

# ── 标签配置（训练前冻结；改 k 即改任务，勿在 OOS 上调）──────────
$k              = 1.0
$h_max          = 20
$sigma_window   = 20

# ── 训练超参（消融位：数组即多组实验，Label 依次追加 _d*/_lr*/_w*/*_s*）──
$max_depth_list     = @(5)      # 例：@(2, 3) 做深度消融（后缀 _d*）
$learning_rate_list = @(0.04)   # 例：@(0.03, 0.05) 做学习率消融（后缀 _lr*）
# 早停段（Val）月数（消融位）：早停只用这一段，ES 段只用于评估/门禁
# （v0.109.0 协议）。TrainEnd 由 EsStart-1 天前移到 ValStart-1 天；该值
# **恒入**目录名（_v{N}m，含单值），避免与旧协议（valm=0）同名折目录互相
# 覆盖；签名维度为 `valm=`。例：@(3, 6, 12) 做早停段长度消融（看早停点
# 决策稳定性），默认 6（半年）。
$val_months_list = @(6)

# 训练窗口年数（消融位；TrainStart = ValStart - N 年）：
# 例：@(3, 5) 做窗口消融（后缀 _w{N}y）。窗口起点早于数据可用起点（cs_train
# 首个分区）时该组合直接判失败并跳过，不静默截短窗口；
# 数据自 2012-01 起，最长 10 年窗口仍然可行。
# v0.108.8 登记：单折配对实验（2023H1，every_n_days=1）显示 7 年窗口
# （lift 1.386 / 10 日块下限 1.207）优于 5 年（1.243 / 1.110），故默认取 7 年。
$train_window_years_list = @(5)
# 随机种子（消融位；多种子用于量化种子方差）：
# 例：@(42, 7, 2024)（后缀 _s*）。注意种子是签名维度，多种子不会被混入同组。
$random_state_list  = @(42)
$n_estimators       = 1000       # 树数量上限（配合早停）
$early_stopping_rounds = 50     # Val 段（早停段）早停轮数
$subsample          = 0.8
$colsample_bytree   = 0.8
$reg_lambda         = 1.0
# 早停指标（消融位，仅对 binary 臂生效）：logloss（概率校准口径）| rank_ic_daily
# （逐日截面 Spearman 均值，与门禁 lift 同向，复用 ml/train_core/eval.py）。
# 两种口径是不同签名（`em=`），禁止混组比较。例：@("logloss", "rank_ic_daily")
# 做口径消融（后缀 _em*，仅多值时追加，保持 baseline 目录名稳定）。
# 注：排序臂（rank_pairwise）的早停恒为 auc（自实现池化 AUC 回调，不用
# XGBoost 内置 ranking auc 的 O(n²) 成对展开），不受本列表影响。
$eval_metric_list   = @("logloss")
# 特征集（消融位）：full（冻结 33 列）| core（波动状态 10 列）| core_state
# （波动状态 + 风格/状态轴 12 列，v0.113.0）| ic_admit（训练段单变量 |IC|
# 降序 top-k）。非 full 值时目录追加 _fs* 后缀（单值也追加，避免覆盖基线
# 目录，与 _v{N}m 同约定）；特征集是签名维度（`fs=`），禁止混组比较。
# 例：@("core") 或 @("full", "ic_admit")
# 注：core_state 已于 2026-09-14 实测否定（8 折三判据点估计 7/8 折下降、
# 2024H1 区间缺口从 0.0006 扩到 0.0444）→ 默认已回滚到 core，不再启用；
# 也勿靠加列去追 2024H1 的 0.000x 缺口（小于流程自身扰动带）。
# 详见 CHANGELOG 0.113.0 与 plan 实施状态补充。
$feature_set_list   = @("core")
$ic_top_k           = 8          # ic_admit 入选列数（必选列另计）
# 训练目标（消融位）：binary（binary:logistic，输出即概率）| rank_pairwise
# （rank:pairwise + qid=trade_date，只学当日截面排序，再用 Val 段 isotonic
# 映射回概率）。非 binary 时目录追加 _obj* 后缀；排序臂的早停指标恒为
# auc（自实现回调池化 AUC，与门禁 auc_lift 同向且基准率不变；不用 XGBoost
# 内置 ranking auc——大 query group 下 O(n²) 成对展开会超 int32 上限必然
# 崩溃；em=ndcg 已实测在 Val 上极早饱和 → 欠训练），并恒加 _em* 后缀，
# 避免与 em=ndcg 旧产物落到同一折目录。
# 只跑排序臂时把本行改为 @("rank_pairwise")（否则 binary 臂会重跑一遍）。
# 当前配置：只跑 binary 臂（core_state 特征集首测，单变量原则）；排
# 序臂的 v0.112.2 修复后重训排在后面，避免两个变量同时动导致归因不清。
$objective_list     = @("binary")
# 注：min_child_weight / scale_pos_weight 未透传——两者为正则尺度策略 A 的
# 设计不变量（min_child_weight 与样本权重 1/网格大小绑定，scale_pos_weight
# 会破坏自然事件率口径）。

# ── 预登记抽样（v0.108.8 变更：every_n_days 3 → 1，必须登记且不得按 OOS 回调）──
# every_n_days=1：矩阵不再按日期等距抽样（训练行数 ×3，ES 评估网格从 1/3 日
# 加密为全部交易日）。动机：门禁判据是逐折 moving-block 区间下限 > 1.1，评估
# 网格只有 1/3 日会让区间宽度虚大、且不同训练窗口的臂采样相位不同（不可配对
# 比较）。单折实测（2023H1）：10 日块下限 1.063（end=3）→ 1.110（end=1），
# 训练行 263 万 → 788 万，折耗时 6 → 8 分钟。
# 注意：该变更同时改变训练数据量，跨 end 值的对比属不同签名（`end=`），禁止混组。
$h_per_group      = 2        # 每期限组内抽样 h 数（保持 2 不变）
$every_n_days     = 1        # v0.108.8 登记变更（原 3）
$chunk_days       = 50

# ============================================================
#  执行区（一般无需修改）
# ============================================================

$failed = @()
$selected_folds = @($folds | Where-Object { $_.Selected })
# 臂总数：排序目标恒为单条 auc 臂（自实现池化 AUC 回调，忽略 eval_metric
# 消融），binary 用 eval_metric 数量；目标维度另乘特征集个数
$metricArmCount = 0
foreach ($obj in $objective_list) {
    $metricArmCount += if ($obj -eq "rank_pairwise") { 1 } else { $eval_metric_list.Count }
}
$total = $selected_folds.Count * $max_depth_list.Count * $learning_rate_list.Count *
    $train_window_years_list.Count * $val_months_list.Count * $feature_set_list.Count *
    $metricArmCount * $random_state_list.Count
$done = 0

# 数据可用起点：cs_train 首个分区（训练窗口越界时判失败，不静默截短窗口）
$cs_train_dir = Join-Path $data_root "features\cs_train"
$data_floor = (Get-ChildItem $cs_train_dir -Filter *.parquet -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -First 1).BaseName
if (-not $data_floor) { throw "未找到 cs_train 分区（$cs_train_dir）：请先构建特征" }
Write-Host "数据可用起点: $data_floor" -ForegroundColor DarkGray

# 目标维度组合（特征集 × 训练目标；后缀 _fs* / _obj*，非默认值恒追加）
$goalCombos = @()
foreach ($featureSet in $feature_set_list) {
    foreach ($objective in $objective_list) {
        $fsSuffix = if ($featureSet -ne "full") { "_fs$featureSet" } else { "" }
        $objSuffix = if ($objective -ne "binary") { "_obj$objective" } else { "" }
        $goalCombos += [PSCustomObject]@{
            FeatureSet = $featureSet
            Objective  = $objective
            Suffix     = "$fsSuffix$objSuffix"
        }
    }
}

# 训练窗口 × 早停段月数 × 目标维度 × 早停指标 的组合展开（单层循环，避免更深嵌套）
$armCombos = @()
foreach ($years in $train_window_years_list) {
    foreach ($valMonths in $val_months_list) {
        foreach ($goal in $goalCombos) {
            # 排序臂早停由目标决定（auc：自实现池化 AUC 回调，与门禁 auc_lift
            # 同向；不用 XGBoost 内置 ranking auc 的 O(n²) 成对展开）：忽略
            # $eval_metric_list 消融，恒为单条臂；且恒加 _em* 后缀——签名不同
            # （em=ndcg 的旧产物）不得混组，也不得静默覆写同一目录
            $metricList = if ($goal.Objective -eq "rank_pairwise") { @("auc") } else { $eval_metric_list }
            $metricMulti = $metricList.Count -gt 1
            $forceMetricSuffix = $goal.Objective -eq "rank_pairwise"
            foreach ($evalMetric in $metricList) {
                $windowSuffix = if ($train_window_years_list.Count -gt 1) { "_w${years}y" } else { "" }
                # 早停段长度恒入后缀（含单值）：旧协议产物无 _v{N}m，避免目录互相覆盖
                $valSuffix = "_v${valMonths}m"
                $metricSuffix = if ($metricMulti -or $forceMetricSuffix) {
                    "_em" + ($evalMetric -replace "_daily", "")
                } else { "" }
                $armCombos += [PSCustomObject]@{
                    Years      = $years
                    ValMonths  = $valMonths
                    EvalMetric = $evalMetric
                    FeatureSet = $goal.FeatureSet
                    Objective  = $goal.Objective
                    Suffix     = "$windowSuffix$valSuffix$metricSuffix$($goal.Suffix)"
                }
            }
        }
    }
}

foreach ($depth in $max_depth_list) {
    $depthSuffix = if ($max_depth_list.Count -gt 1) { "_d$depth" } else { "" }
    foreach ($lr in $learning_rate_list) {
        $lrSuffix = if ($learning_rate_list.Count -gt 1) { "_lr$lr" } else { "" }
        foreach ($combo in $armCombos) {
            $years = $combo.Years
            $valMonths = $combo.ValMonths
            $evalMetric = $combo.EvalMetric
            $featureSet = $combo.FeatureSet
            $objective = $combo.Objective
            foreach ($seed in $random_state_list) {
                $seedSuffix = if ($random_state_list.Count -gt 1) { "_s$seed" } else { "" }
                $suffix = "$depthSuffix$lrSuffix$($combo.Suffix)$seedSuffix"
                foreach ($fold in $selected_folds) {
                    $esStartDate = [datetime]::ParseExact($fold.EsStart, "yyyyMMdd", $null)
                    $valStartDate = $esStartDate.AddMonths(-$valMonths)
                    $valStart = $valStartDate.ToString("yyyyMMdd")
                    $valEnd = $esStartDate.AddDays(-1).ToString("yyyyMMdd")
                    $trainStart = $valStartDate.AddYears(-$years).ToString("yyyyMMdd")
                    $trainEnd = $valStartDate.AddDays(-1).ToString("yyyyMMdd")
                    $done++
                    if ($valStart -lt $data_floor) {
                        Write-Host "折 $($fold.Label)$suffix 早停段起点 $valStart 早于数据起点 $data_floor：跳过（不静默截短窗口）" -ForegroundColor Red
                        $failed += "$($fold.Label)$suffix(窗口越界)"
                        continue
                    }
                    if ($trainStart -lt $data_floor) {
                        Write-Host "折 $($fold.Label)$suffix 训练窗口起点 $trainStart 早于数据起点 $data_floor：跳过（不静默截短窗口）" -ForegroundColor Red
                        $failed += "$($fold.Label)$suffix(窗口越界)"
                        continue
                    }
                    $out_dir = Join-Path $wf_root "$($fold.Label)$suffix"
                    Write-Host ""
                    Write-Host "==== [$done/$total] terminal_loss WF 折 $($fold.Label)$suffix (depth=$depth, lr=$lr, 窗口=${years}年, 早停段=${valMonths}月, em=$evalMetric, fs=$featureSet, obj=$objective, seed=$seed) ====" -ForegroundColor Cyan
                    Write-Host "      Train [$trainStart,$trainEnd]  Val(早停) [$valStart,$valEnd]  ES(评估) [$($fold.EsStart),$($fold.EsEnd)]"

                    $pythonCmd = "py .\scripts\train_terminal_risk_model.py" +
                        " --data-root $data_root" +
                        " --output-dir $out_dir" +
                        " --start-date $trainStart" +
                        " --end-date $($fold.EsEnd)" +
                        " --train-start $trainStart" +
                        " --train-end $trainEnd" +
                        " --val-start $valStart" +
                        " --val-end $valEnd" +
                        " --es-start $($fold.EsStart)" +
                        " --es-end $($fold.EsEnd)" +
                        " --k $k --h-max $h_max --sigma-window $sigma_window" +
                        " --max-depth $depth --learning-rate $lr --n-estimators $n_estimators" +
                        " --early-stopping-rounds $early_stopping_rounds" +
                        " --subsample $subsample --colsample-bytree $colsample_bytree --reg-lambda $reg_lambda" +
                        " --random-state $seed" +
                        " --eval-metric $evalMetric" +
                        " --feature-set $featureSet --ic-top-k $ic_top_k --objective $objective" +
                        " --h-per-group $h_per_group --every-n-days $every_n_days --chunk-days $chunk_days" +
                        " --device $device" +
                        " --fixed-name"

                    # 注意：训练脚本读取 [TrainStart, EsEnd] 之后 h_max+1 个交易日的端点数据，
                    # 最后一折（2026H1）的 E 最远落在 2026-07-末，数据末端须覆盖。
                    Invoke-Expression $pythonCmd
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "折 $($fold.Label)$suffix 训练失败（exit=$LASTEXITCODE），跳过继续" -ForegroundColor Red
                        $failed += "$($fold.Label)$suffix"
                    }
                }
            }
        }
    }
}

Write-Host ""
if ($run_summary) {
    Write-Host "==== 汇总 summary 与跨折门禁 ====" -ForegroundColor Cyan
    py .\scripts\summarize_terminal_risk_wf.py --wf-root $wf_root
}

if ($failed.Count -gt 0) {
    Write-Host "失败折: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "terminal_loss 滚动 WF 全部完成: $wf_root" -ForegroundColor Green

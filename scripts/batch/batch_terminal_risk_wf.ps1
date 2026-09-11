# batch_terminal_risk_wf.ps1
# 期末异常亏损风险模型滚动 Walk-forward 批量脚本（第一阶段研究型 WF）
#
# 每折：滚动 Train（由 $train_window_years_list 决定年数）+ ES（6 个月，互不重叠）
# → 独立概率质量报告；全部完成后自动运行 summarize_terminal_risk_wf.py 拼接
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
# Train 段由 $train_window_years_list 派生：TrainStart = EsStart - N 年，
# TrainEnd = EsStart - 1 天（N=3 与首轮基线完全一致）；日期为自然日边界，
# 训练脚本内部按交易日历过滤。
# Label      : 目录名与汇总标识
# EsStart/EsEnd : ES 段自然日边界
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
# 训练窗口年数（消融位；TrainStart = EsStart - N 年）：
# 例：@(3, 5) 做窗口消融（后缀 _w{N}y）。窗口起点早于数据可用起点（cs_train
# 首个分区）时该组合直接判失败并跳过，不静默截短窗口；
# 数据自 2012-01 起，最长 10 年窗口仍然可行。
$train_window_years_list = @(5)
# 随机种子（消融位；多种子用于量化种子方差）：
# 例：@(42, 7, 2024)（后缀 _s*）。注意种子是签名维度，多种子不会被混入同组。
$random_state_list  = @(42,7,2024)
$n_estimators       = 1000       # 树数量上限（配合早停）
$early_stopping_rounds = 50     # ES 段 logloss 早停轮数
$subsample          = 0.8
$colsample_bytree   = 0.8
$reg_lambda         = 1.0
# 注：min_child_weight / scale_pos_weight / eval_metric 未透传——前两者为
# 正则尺度策略 A 的设计不变量（min_child_weight 与样本权重 1/网格大小绑定，
# scale_pos_weight 会破坏自然事件率口径），早停指标契约固定 logloss。

# ── 预登记抽样（与首轮一致；变更必须登记，勿按 OOS 结果回调）───────
$h_per_group      = 2
$every_n_days     = 3
$chunk_days       = 50

# ============================================================
#  执行区（一般无需修改）
# ============================================================

$failed = @()
$selected_folds = @($folds | Where-Object { $_.Selected })
$total = $selected_folds.Count * $max_depth_list.Count * $learning_rate_list.Count *
    $train_window_years_list.Count * $random_state_list.Count
$done = 0

# 数据可用起点：cs_train 首个分区（训练窗口越界时判失败，不静默截短窗口）
$cs_train_dir = Join-Path $data_root "features\cs_train"
$data_floor = (Get-ChildItem $cs_train_dir -Filter *.parquet -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -First 1).BaseName
if (-not $data_floor) { throw "未找到 cs_train 分区（$cs_train_dir）：请先构建特征" }
Write-Host "数据可用起点: $data_floor" -ForegroundColor DarkGray

foreach ($depth in $max_depth_list) {
    $depthSuffix = if ($max_depth_list.Count -gt 1) { "_d$depth" } else { "" }
    foreach ($lr in $learning_rate_list) {
        $lrSuffix = if ($learning_rate_list.Count -gt 1) { "_lr$lr" } else { "" }
        foreach ($years in $train_window_years_list) {
            $windowSuffix = if ($train_window_years_list.Count -gt 1) { "_w${years}y" } else { "" }
            foreach ($seed in $random_state_list) {
                $seedSuffix = if ($random_state_list.Count -gt 1) { "_s$seed" } else { "" }
                $suffix = "$depthSuffix$lrSuffix$windowSuffix$seedSuffix"
                foreach ($fold in $selected_folds) {
                    $esStartDate = [datetime]::ParseExact($fold.EsStart, "yyyyMMdd", $null)
                    $trainStart = $esStartDate.AddYears(-$years).ToString("yyyyMMdd")
                    $trainEnd = $esStartDate.AddDays(-1).ToString("yyyyMMdd")
                    $done++
                    if ($trainStart -lt $data_floor) {
                        Write-Host "折 $($fold.Label)$suffix 训练窗口起点 $trainStart 早于数据起点 $data_floor：跳过（不静默截短窗口）" -ForegroundColor Red
                        $failed += "$($fold.Label)$suffix(窗口越界)"
                        continue
                    }
                    $out_dir = Join-Path $wf_root "$($fold.Label)$suffix"
                    Write-Host ""
                    Write-Host "==== [$done/$total] terminal_loss WF 折 $($fold.Label)$suffix (depth=$depth, lr=$lr, 窗口=${years}年, seed=$seed) ====" -ForegroundColor Cyan
                    Write-Host "      Train [$trainStart,$trainEnd]  ES [$($fold.EsStart),$($fold.EsEnd)]"

                    $pythonCmd = "py .\scripts\train_terminal_risk_model.py" +
                        " --data-root $data_root" +
                        " --output-dir $out_dir" +
                        " --start-date $trainStart" +
                        " --end-date $($fold.EsEnd)" +
                        " --train-start $trainStart" +
                        " --train-end $trainEnd" +
                        " --es-start $($fold.EsStart)" +
                        " --es-end $($fold.EsEnd)" +
                        " --k $k --h-max $h_max --sigma-window $sigma_window" +
                        " --max-depth $depth --learning-rate $lr --n-estimators $n_estimators" +
                        " --early-stopping-rounds $early_stopping_rounds" +
                        " --subsample $subsample --colsample-bytree $colsample_bytree --reg-lambda $reg_lambda" +
                        " --random-state $seed" +
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

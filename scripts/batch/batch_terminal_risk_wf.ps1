# batch_terminal_risk_wf.ps1
# 期末异常亏损风险模型滚动 Walk-forward 批量脚本（第一阶段研究型 WF）
#
# 每折：滚动 Train（约 3 年）+ ES（6 个月，互不重叠）→ 独立概率质量报告；
# 全部完成后自动运行 summarize_terminal_risk_wf.py 拼接 summary 并打印跨折门禁
# （lift = ES PR-AUC / ES 事件率，lift 最小值 >= 阈值才建议进入第二阶段）。
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
# TrainStart/TrainEnd = ES 起点前推约 3 年（滚动窗）；
# 日期为自然日边界，训练脚本内部按交易日历过滤。
# Label                : 目录名与汇总标识
# TrainStart/TrainEnd  : Train 段自然日边界
# EsStart/EsEnd        : ES 段自然日边界
# Selected             : $false 跳过该折（结果保留不删）
$folds = @(
    [PSCustomObject]@{ Label = "2022H2"; TrainStart = "20190701"; TrainEnd = "20220630"; EsStart = "20220701"; EsEnd = "20221231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2023H1"; TrainStart = "20200101"; TrainEnd = "20221231"; EsStart = "20230101"; EsEnd = "20230630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2023H2"; TrainStart = "20200701"; TrainEnd = "20230630"; EsStart = "20230701"; EsEnd = "20231231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2024H1"; TrainStart = "20210101"; TrainEnd = "20231231"; EsStart = "20240101"; EsEnd = "20240630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2024H2"; TrainStart = "20210701"; TrainEnd = "20240630"; EsStart = "20240701"; EsEnd = "20241231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2025H1"; TrainStart = "20220101"; TrainEnd = "20241231"; EsStart = "20250101"; EsEnd = "20250630"; Selected = $true  }
    [PSCustomObject]@{ Label = "2025H2"; TrainStart = "20220701"; TrainEnd = "20250630"; EsStart = "20250701"; EsEnd = "20251231"; Selected = $true  }
    [PSCustomObject]@{ Label = "2026H1"; TrainStart = "20230101"; TrainEnd = "20251231"; EsStart = "20260101"; EsEnd = "20260630"; Selected = $true  }
)

# ── 标签配置（训练前冻结；改 k 即改任务，勿在 OOS 上调）──────────
$k              = 1.0
$h_max          = 20
$sigma_window   = 20

# ── 训练超参（消融位：数组即多组实验，Label 会追加后缀）────────────
$max_depth_list   = @(3)      # 例：@(2, 3) 做深度消融
$n_estimators     = 500
$random_state     = 42

# ── 预登记抽样（与首轮一致；变更必须登记，勿按 OOS 结果回调）───────
$h_per_group      = 2
$every_n_days     = 3
$chunk_days       = 50

# ============================================================
#  执行区（一般无需修改）
# ============================================================

$failed = @()
$total = ($folds | Where-Object { $_.Selected }).Count * $max_depth_list.Count
$done = 0

foreach ($depth in $max_depth_list) {
    $suffix = if ($max_depth_list.Count -gt 1) { "_d$depth" } else { "" }
    foreach ($fold in $folds) {
        if (-not $fold.Selected) { continue }
        $done++
        $out_dir = Join-Path $wf_root "$($fold.Label)$suffix"
        Write-Host ""
        Write-Host "==== [$done/$total] terminal_loss WF 折 $($fold.Label)$suffix (depth=$depth) ====" -ForegroundColor Cyan
        Write-Host "      Train [$($fold.TrainStart),$($fold.TrainEnd)]  ES [$($fold.EsStart),$($fold.EsEnd)]"

        $pythonCmd = "py .\scripts\train_terminal_risk_model.py" +
            " --data-root $data_root" +
            " --output-dir $out_dir" +
            " --start-date $($fold.TrainStart)" +
            " --end-date $($fold.EsEnd)" +
            " --train-start $($fold.TrainStart)" +
            " --train-end $($fold.TrainEnd)" +
            " --es-start $($fold.EsStart)" +
            " --es-end $($fold.EsEnd)" +
            " --k $k --h-max $h_max --sigma-window $sigma_window" +
            " --max-depth $depth --n-estimators $n_estimators --random-state $random_state" +
            " --h-per-group $h_per_group --every-n-days $every_n_days --chunk-days $chunk_days" +
            " --device $device"

        # 注意：训练脚本读取 [TrainStart, EsEnd] 之后 h_max+1 个交易日的端点数据，
        # 最后一折（2026H1）的 E 最远落在 2026-07-末，数据末端须覆盖。
        Invoke-Expression $pythonCmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "折 $($fold.Label)$suffix 训练失败（exit=$LASTEXITCODE），跳过继续" -ForegroundColor Red
            $failed += "$($fold.Label)$suffix"
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

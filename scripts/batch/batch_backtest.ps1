# batch_backtest.ps1
# ML 回测批量参数扫描脚本
#
# 用法：
#   在 "参数配置区" 里，把想对比的参数设置为多个值（数组），固定参数保留单个值。
#   脚本会自动遍历所有参数组合，每组调用一次 run_ml_backtest.py。
#
# 示例启动：
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch_backtest.ps1

# ============================================================
#  参数配置区（修改这里控制实验组合）
# ============================================================

# ── 回测日期范围 ────────────────────────────────────────────
$startDate               = Get-Date "2025-07-01"
$endDate                 = Get-Date "2025-07-10"
$backtest_end_date       = "20251231"         # 每组回测的结束日期

# ── 模型与信号 ──────────────────────────────────────────────
$mv_list                 = @(3774)#(3299)            # 模型版本（模型A）
$model_version_b         = $null#3275             # 第二个模型版本号（集成模式），$null 表示不启用
$ensemble_weight_a       = 0.5                # 集成时模型A的排名权重（模型B = 1 - 该值）

# ── 选股与调仓 ──────────────────────────────────────────────
$top_n_list              = @(30)              # 持股数量
$freq_list               = @(20)              # 调仓频率（交易日天数）
$sell_timing             = "open"             # 卖出时机：open | close
$position_sizing         = "score"            # 仓位管理：equal | score | kelly | half_kelly

# ── 组合约束 ────────────────────────────────────────────────
$max_weight_per_stock    = 0.04               # 单票最大权重，$null 表示不限
$max_per_industry        = 3                  # 单行业最大持仓数，$null 表示不限

# ── 止损配置 ────────────────────────────────────────────────
$stop_loss_enabled       = $true              # $true 启用 | $false 禁用
$stop_loss_drawdown_pct  = 30                 # 回撤止损阈值（%）

# ── ECT（权益曲线交易）配置 ─────────────────────────────────
$equity_curve_enabled    = $false             # $true 启用 | $false 禁用

# ── 输出 ────────────────────────────────────────────────────
$output_name             = "dual"       # 报告输出名称

# ── 全部完成后是否倒计时关机 ────────────────────────────────
$shutdown_on_complete    = $false
$shutdown_timeout_sec    = 600

# ============================================================
#  以下为执行逻辑（通常不需修改）
# ============================================================

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count   = 0
$failed  = 0

# 计算工作日数
$workingDays = 0
$tempDate = $startDate
while ($tempDate -le $endDate) {
    if ($tempDate.DayOfWeek -ne "Saturday" -and $tempDate.DayOfWeek -ne "Sunday") { $workingDays++ }
    $tempDate = $tempDate.AddDays(1)
}
$totalTasks = $workingDays * $freq_list.Length * $top_n_list.Length * $mv_list.Length

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  ML 回测批量实验" -ForegroundColor Cyan
Write-Host "  回测起止    : $($startDate.ToString('yyyyMMdd')) ~ $($endDate.ToString('yyyyMMdd'))" -ForegroundColor Cyan
Write-Host "  总任务数    : $totalTasks" -ForegroundColor Cyan
if ($model_version_b) {
    Write-Host "  集成模式    : 模型B=$model_version_b, 权重A=$ensemble_weight_a" -ForegroundColor Cyan
}
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$currentDate = $startDate
while ($currentDate -le $endDate) {
    $dateStr = $currentDate.ToString("yyyyMMdd")
    if ($currentDate.DayOfWeek -ne "Saturday" -and $currentDate.DayOfWeek -ne "Sunday") {
        foreach ($freq in $freq_list) {
        foreach ($topn in $top_n_list) {
        foreach ($mv in $mv_list) {

            $count++

            # 构建命令参数
            $pythonCmd = "py scripts\run_ml_backtest.py" +
                         " --start-date $dateStr" +
                         " --end-date $backtest_end_date" +
                         " --model-version $mv" +
                         " --top-n $topn" +
                         " --rebalance-freq $freq" +
                         " --sell-timing $sell_timing" +
                         " --position-sizing $position_sizing" +
                         " --output-name $output_name"

            # 集成模型
            if ($model_version_b) {
                $pythonCmd += " --model-version-b $model_version_b"
                $pythonCmd += " --ensemble-weight-a $ensemble_weight_a"
            }

            # 组合约束
            if ($max_weight_per_stock) {
                $pythonCmd += " --max-weight-per-stock $max_weight_per_stock"
            }
            if ($max_per_industry) {
                $pythonCmd += " --max-per-industry $max_per_industry"
            }

            # 止损
            if ($stop_loss_enabled) {
                $pythonCmd += " --stop-loss-enabled"
                $pythonCmd += " --stop-loss-drawdown-pct $stop_loss_drawdown_pct"
            }

            # ECT
            if ($equity_curve_enabled) {
                $pythonCmd += " --equity-curve-enabled"
            }

            Write-Host ""
            Write-Host "[任务 $count / $totalTasks]" -ForegroundColor Green
            Write-Host $pythonCmd -ForegroundColor Gray
            Write-Host ""

            Invoke-Expression $pythonCmd
            $exitCode = $LASTEXITCODE

            if ($exitCode -ne 0) {
                $failed++
                Write-Host "[警告] 任务 $count 异常退出（exit code: $exitCode）" -ForegroundColor Red
            }

            # 进度与 ETA
            $percent    = [Math]::Round(($count / $totalTasks) * 100, 1)
            $elapsedMs  = $totalTimer.ElapsedMilliseconds
            $avgMs      = $elapsedMs / $count
            $remainMs   = $avgMs * ($totalTasks - $count)
            $eta        = [TimeSpan]::FromMilliseconds($remainMs)
            $etaTime    = (Get-Date).AddMilliseconds($remainMs)

            Write-Host "--------------------------------------------------------" -ForegroundColor DarkCyan
            Write-Host "进度    : $percent% ($count / $totalTasks，失败 $failed 个)" -ForegroundColor White
            Write-Host "已耗时  : $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor White
            Write-Host "预计还需: $($eta.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
            Write-Host "预计完成: $($etaTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Magenta

        }}}
    }
    $currentDate = $currentDate.AddDays(1)
}

# ── 全部完成 ──────────────────────────────────────────────────
$totalTimer.Stop()
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  全部 $totalTasks 个实验已完成（失败 $failed 个）" -ForegroundColor Magenta
Write-Host "  总耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta

# ── 倒计时关机（可选）────────────────────────────────────────
if ($shutdown_on_complete) {
    while ($Host.UI.RawUI.KeyAvailable) { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") }
    Write-Host @"

================================================
任务已完成。系统将在 $shutdown_timeout_sec 秒后自动关机。
[取消方式]：直接关闭本窗口，或按一次 Ctrl+C。
[注意]：请勿用鼠标点击窗口内部以免脚本暂停。
================================================
"@ -ForegroundColor Yellow
    timeout.exe /t $shutdown_timeout_sec /nobreak
    Write-Host "`n[!] 倒计时结束，正在关机..." -ForegroundColor Red
    Stop-Computer -Force
}

# 0. 定义参数
$startDate = Get-Date "2025-07-01"
$endDate = Get-Date "2025-07-31"
$freq_list = 18, 19, 20, 30
$top_n_list = 30
$mv_list = 15
$para_list = @('--stop-loss-enabled','')

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count = 0

# 重新计算总任务数
$workingDays = 0
$tempDate = $startDate
while ($tempDate -le $endDate) {
    if ($tempDate.DayOfWeek -ne "Saturday" -and $tempDate.DayOfWeek -ne "Sunday") { $workingDays++ }
    $tempDate = $tempDate.AddDays(1)
}
$totalTasks = $workingDays * $freq_list.Length * $top_n_list.Length * $mv_list.Length


# 1. 检查管理员权限（可选但建议）
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "请以管理员身份运行此脚本，否则关机指令可能失效。"
}

# 2. 循环开始
$currentDate = $startDate 
while ($currentDate -le $endDate) {
    $dateStr = $currentDate.ToString("yyyyMMdd")
    
    if ($currentDate.DayOfWeek -ne "Saturday" -and $currentDate.DayOfWeek -ne "Sunday") {
        foreach ($para in $para_list) {
            foreach ($freq in $freq_list) {
                foreach ($topn in $top_n_list) {
                    foreach ($mv in $mv_list) {
                        $count++
                        
                        # 执行 Python 命令
                        py scripts\run_ml_backtest.py --start-date $dateStr --end-date 20251231 --rebalance-freq $freq --sell-timing open --top-n $topn --model-version $mv $para
                        
                        # 预测逻辑
                        $percent = [Math]::Round(($count / $totalTasks) * 100, 2)
                        $elapsedMs = $totalTimer.ElapsedMilliseconds
                        $avgMsPerTask = $elapsedMs / $count
                        $remainingTasks = $totalTasks - $count
                        $remainingMs = $avgMsPerTask * $remainingTasks
                        
                        $eta = [TimeSpan]::FromMilliseconds($remainingMs)
                        $completionTime = (Get-Date).AddMilliseconds($remainingMs)

                        Write-Host "`n--------------------------------------------------" -ForegroundColor Cyan
                        Write-Host "进度: $percent% ($count/$totalTasks)" -ForegroundColor White
                        Write-Host "正在执行: 日期=$dateStr, Freq=$freq, TopN=$topn, MV=$mv" -ForegroundColor Green
                        Write-Host "已耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))"
                        Write-Host "还剩余: $($eta.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
                        Write-Host "预计将完成于: $($completionTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Magenta
                    }
                }
            }
        }
    }
    $currentDate = $currentDate.AddDays(1)
}

$totalTimer.Stop()
Write-Host "`n全部任务已完成！总耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta

# 3. 关机倒计时逻辑
$timeout = 600 
# 3.1. 强制清空所有之前的按键干扰
while ($Host.UI.RawUI.KeyAvailable) { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") }

# 3.2. 强制刷新输出（解决不显示问题）
$msg = @"
================================================
任务已完成。系统将在 10 分钟后自动关机。
[取消方式]：直接关闭本窗口，或按一次 Ctrl+C。
[注意]：请勿用鼠标点击窗口内部以免脚本暂停。
================================================
"@
Write-Host $msg -ForegroundColor Yellow

# 3.3. 使用 Windows 自带的倒计时工具 (timeout.exe)
# 这是最稳妥的，因为它自带倒计时显示，且不受 PowerShell 缓冲区影响
# 倒计时
Write-Host "正在进入系统倒计时..."
timeout.exe /t $timeout /nobreak

# 3.4. 倒计时结束后执行
Write-Host "`n[!] 倒计时结束，正在强制关机..." -ForegroundColor Red
Stop-Computer -Force
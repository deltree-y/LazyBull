# 1. 定义参数
$startDate = Get-Date "2025-07-01"
$endDate = Get-Date "2025-07-31"
$freq_list = 8,9,10,11,12,13,14,15,16,17,18,19,20
$top_n_list = 30
$mv_list = 15

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count = 0
$totalTasks = $freq_list.Length * $top_n_list.Length * $mv_list.Length * ((($endDate - $startDate).Days + 1) - ((($endDate - $startDate).Days + 1) / 7) * 2) # 刚才计算的总数

# 2. 循环开始
while ($startDate -le $endDate) {
    $dateStr = $startDate.ToString("yyyyMMdd")
    
    # 判断工作日
    if ($startDate.DayOfWeek -ne "Saturday" -and $startDate.DayOfWeek -ne "Sunday") {
        
        foreach ($freq in $freq_list) {
            foreach ($topn in $top_n_list) {
                foreach ($mv in $mv_list) {
                    $count++
                    $percent = [Math]::Round(($count / $totalTasks) * 100, 2)
                    
                    Write-Host "`n--------------------------------------------------" -ForegroundColor Cyan
                    Write-Host "进度: $percent% ($count/$totalTasks)" -ForegroundColor White
                    Write-Host "正在执行: 日期=$dateStr, Freq=$freq, TopN=$topn, MV=$mv" -ForegroundColor Green
                    Write-Host "已耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))"
                    
                    # 执行 Python 命令
                    py scripts\run_ml_backtest.py --start-date $dateStr --end-date 20251231 --rebalance-freq $freq --sell-timing open --top-n $topn --model-version $mv --stop-loss-enabled
                }
            }
        }
    }
    $startDate = $startDate.AddDays(1)
}

$totalTimer.Stop()
Write-Host "`n全部任务已完成！总耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta

# 如果需要跑完自动关机，请删掉下面这一行开头的 # 号
Stop-Computer -Force
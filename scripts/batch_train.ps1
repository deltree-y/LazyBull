# 0. 定义参数
$startDate_list = @(Get-Date "2021-07-01")
$endDate_list = @(Get-Date "2025-06-30")
$label_list = @("y_ret_20")#, "y_ret_10", "y_ret_20")    #标签
$topk_list = @(500)            #训练集正值数量
$max_depth_list = @(16)           #树的最大深度
$learning_rate_list = @(0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)     #学习率

$n_estimators = 2000

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count = 0

# 重新计算总任务数
$totalTasks = $startDate_list.Length * $endDate_list.Length * $label_list.Length * $max_depth_list.Length * $learning_rate_list.Length


# 1. 检查管理员权限（可选但建议）
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "请以管理员身份运行此脚本，否则关机指令可能失效。"
}

# 2. 循环开始
foreach ($startDate in $startDate_list) {
    foreach ($endDate in $endDate_list) {
        foreach ($label in $label_list) {
            foreach ($topk in $topk_list) {
                foreach ($max_depth in $max_depth_list) {
                    foreach ($learning_rate in $learning_rate_list) {
                        $count++

                        # 1. 构造命令字符串（使用变量存储，方便复用）
                        $startDateStr = $startDate.ToString("yyyyMMdd")
                        $endDateStr = $endDate.ToString("yyyyMMdd")

                        $pythonCmd = "py .\scripts\train_ml_model.py " +
                                    "--start-date $startDateStr --end-date $endDateStr " +
                                    "--label $label " +
                                    "--task classification --pos-topk $topk " +
                                    "--n-estimators $n_estimators " +
                                    "--max-depth $max_depth " +
                                    "--learning-rate $learning_rate"

                        # 2. 打印命令到控制台（带颜色标注，方便肉眼识别）
                        Write-Host "`n[执行指令]:" -ForegroundColor Green
                        Write-Host $pythonCmd -ForegroundColor Gray

                        # 3. 实际执行命令
                        # 使用 Invoke-Expression 执行字符串命令，或者直接调用
                        Invoke-Expression $pythonCmd

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
                        Write-Host "已耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))"
                        Write-Host "还剩余: $($eta.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
                        Write-Host "预计将完成于: $($completionTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Magenta
                    }
                }
            }
        }
    }
}


# 3. 关机倒计时逻辑
$totalTimer.Stop()
Write-Host "`n全部任务已完成！总耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta

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
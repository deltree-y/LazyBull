# batch_train_position_risk.ps1
# 持仓风控模型批量参数扫描脚本
#
# 用法：
#   在 "参数配置区" 里，把想对比的参数设置为多个值（数组），固定参数保留单个值。
#   脚本会自动遍历所有参数组合，每组调用一次 train_position_risk_model.py。
#
# 示例启动：
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_train_position_risk.ps1

# ============================================================
#  参数配置区（修改这里控制实验组合）
# ============================================================

# ── 训练时间段配置（支持多组）─────────────────────────────
# Label       : 时间段标签，仅用于日志/汇总展示
# StartDate   : 训练数据起始日期（yyyyMMdd）
# EndDate     : 训练数据结束日期（yyyyMMdd）
# SplitCount  : Walk-forward 切分数量
$risk_period_configs = @(
    [PSCustomObject]@{
        Label      = "default"
        StartDate  = "20200101"
        EndDate    = "20231231"
        SplitCount = 6
    }
)

# ── 标签与持有期 ──────────────────────────────────────────
$horizon_list            = @(10)             # 持有期天数（标签窗口）
$label_column_list       = @("y_ret_10")     # 前向收益率列名（与 horizon 对应）

# ── 模型超参（想对比的参数放多个值，其余放单个值）──────────
$n_estimators_list       = @(200)            # 树数量上限（配合早停）
$max_depth_list          = @(4)              # 树最大深度，可多值扫描如 @(3, 4, 5, 6)
$learning_rate_list      = @(0.03)           # 学习率，可多值扫描如 @(0.01, 0.03, 0.05)
$subsample_list          = @(0.7)            # 行采样率，可多值扫描如 @(0.6, 0.7, 0.8)
$colsample_bytree_list   = @(0.6)            # 列采样率，可多值扫描如 @(0.4, 0.6, 0.8)
$reg_lambda_list         = @(1.0)            # L2 正则，可多值扫描如 @(0.5, 1.0, 2.0)
$early_stopping_rounds_list = @(30)          # 早停轮数，可多值扫描如 @(20, 30, 50)

# ── 随机种子 ──────────────────────────────────────────────
$random_state_list       = @(42)             # 随机种子，可多值扫描如 @(42, 61, 82)

# ── 守卫条件 ──────────────────────────────────────────────
$min_f1_list             = @(0.35)           # 最低 F1(macro) 阈值，可多值扫描如 @(0.30, 0.35, 0.40)

# ── Monitor 参数 ──────────────────────────────────────────
$proba_threshold_list    = @(0.6)            # REDUCE 触发提前退出的最低概率，可多值扫描如 @(0.5, 0.6, 0.7)

# ── 路径 ──────────────────────────────────────────────────
$data_root               = "./data"

# ── 全部完成后是否倒计时关机 ──────────────────────────────
$shutdown_on_complete    = $false
$shutdown_timeout_sec    = 600

# ============================================================
#  以下为执行逻辑（通常不需修改）
# ============================================================

$batch_run_id = "risk_batch_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$batch_output_root = Join-Path $data_root ("models\risk\batches\{0}" -f $batch_run_id)
New-Item -ItemType Directory -Path $batch_output_root -Force | Out-Null

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count      = 0
$failed     = 0

# 计算总任务数（各列表长度的笛卡尔积）
$totalTasks = $risk_period_configs.Length *
              $horizon_list.Length *
              $label_column_list.Length *
              $n_estimators_list.Length *
              $max_depth_list.Length *
              $learning_rate_list.Length *
              $subsample_list.Length *
              $colsample_bytree_list.Length *
              $reg_lambda_list.Length *
              $early_stopping_rounds_list.Length *
              $random_state_list.Length *
              $min_f1_list.Length *
              $proba_threshold_list.Length

$periodSummary = ($risk_period_configs | ForEach-Object {
    "{0}:{1}~{2}, split={3}" -f $_.Label, $_.StartDate, $_.EndDate, $_.SplitCount
}) -join "; "

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  持仓风控模型 批量训练实验" -ForegroundColor Cyan
Write-Host "  批次ID     : $batch_run_id" -ForegroundColor Cyan
Write-Host "  时间段数   : $($risk_period_configs.Count)" -ForegroundColor Cyan
Write-Host "  时间段列表 : $periodSummary" -ForegroundColor Cyan
Write-Host "  总任务数   : $totalTasks" -ForegroundColor Cyan
Write-Host "  数据目录   : $data_root" -ForegroundColor Cyan
Write-Host "  批次目录   : $batch_output_root" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($riskPeriod in $risk_period_configs) {
foreach ($horizon in $horizon_list) {
foreach ($label_column in $label_column_list) {
foreach ($n_estimators in $n_estimators_list) {
foreach ($max_depth in $max_depth_list) {
foreach ($learning_rate in $learning_rate_list) {
foreach ($subsample in $subsample_list) {
foreach ($colsample_bytree in $colsample_bytree_list) {
foreach ($reg_lambda in $reg_lambda_list) {
foreach ($early_stopping_rounds in $early_stopping_rounds_list) {
foreach ($random_state in $random_state_list) {
foreach ($min_f1 in $min_f1_list) {
foreach ($proba_threshold in $proba_threshold_list) {

    $count++
    $period_label = $riskPeriod.Label
    $start_date = $riskPeriod.StartDate
    $end_date = $riskPeriod.EndDate
    $split_count = $riskPeriod.SplitCount

    # 构建命令字符串
    $pythonCmd = "py .\scripts\train_position_risk_model.py" +
                 " --data-root $data_root" +
                 " --start-date $start_date" +
                 " --end-date $end_date" +
                 " --horizon $horizon" +
                 " --label-column $label_column" +
                 " --split-count $split_count" +
                 " --min-f1 $min_f1" +
                 " --max-depth $max_depth" +
                 " --learning-rate $learning_rate" +
                 " --n-estimators $n_estimators" +
                 " --subsample $subsample" +
                 " --colsample-bytree $colsample_bytree" +
                 " --reg-lambda $reg_lambda" +
                 " --early-stopping-rounds $early_stopping_rounds" +
                 " --random-state $random_state" +
                 " --proba-threshold $proba_threshold"

    Write-Host ""
    Write-Host "[任务 $count / $totalTasks][时间段 $period_label][split=$split_count, $start_date ~ $end_date]" -ForegroundColor Green
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

}}}}}}}}}}}}}  # end foreach（时间段+参数组合循环）

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

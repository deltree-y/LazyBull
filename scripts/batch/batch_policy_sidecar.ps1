# batch_policy_sidecar.ps1
# terminal_loss 政策旁路批量打分（P2-1）：指定 WF batch → 持仓快照打分 → 中文表产物
#
# 前置：该 batch 必须已导出持仓快照（`walk_forward_持仓快照_*.csv`，v0.114.0 起；
# batch 由 scripts/batch/batch_walk_forward.ps1 生成到 data/walk_forward/batches/<batch>/raw）。
# 旧 batch 没有快照时：重跑一次 batch_walk_forward.ps1，或复用已注册模型只跑 OOS 回测：
#   py .\scripts\walk_forward.py --skip-training --start-model-version <N> \
#      --wf-summary-csv <batch>\raw\walk_forward_summary_<label>_<idx>.csv ...
#
# 用法：
#   # 列出可选 batch（含快照数），不执行打分
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_policy_sidecar.ps1 -ListBatches
#   # 对最新 batch 打分（默认参数）
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_policy_sidecar.ps1
#   # 指定 batch 与阈值网格、多个特征集
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_policy_sidecar.ps1 `
#       -Batch wf_batch_20260914_081944 -Arms _d5_v6m_fscore `
#       -PHi 0.90,0.95,0.99 -PAbs 0.05,0.10,0.15,0.20
#
# 产物（每个 arm 一份，全部中文表头）：
#   <batch>\policy_sidecar<arm>\风险台账.csv
#   <batch>\policy_sidecar<arm>\触发清单.csv
#   <batch>\policy_sidecar<arm>\阈值扫描.csv
# 运行日志：logs\policy_sidecar_<batch>_<时间戳>.log（logs/ 已 gitignore）

param(
    [string]$Batch = "latest",              # latest | batch 目录名 | 完整路径
    [string[]]$Arms = @("_d5_v6m_fscore"),  # terminal_loss 折目录后缀（可多值）
    [double[]]$PHi = @(0.90, 0.95, 0.99),   # 截面分位阈值网格（触发清单用第一个值）
    [double[]]$PAbs = @(0.05, 0.10, 0.15, 0.20),  # 绝对概率阈值网格
    [switch]$NoDayPct,                      # 跳过当日截面分位（阈值扫描随之不可用）
    [switch]$NoMissed,                      # 触发清单不包含漏报事件
    [switch]$ListBatches,                   # 仅列出可选 batch 后退出
    [string]$DataRoot = "./data",
    [string]$RiskRoot = "",                 # 默认 <data>\walk_forward\terminal_risk_wf
    [string]$OutDir = ""                    # 默认 <batch>\policy_sidecar<arm>
)

$ErrorActionPreference = "Continue"   # 原生命令的 stderr（loguru 日志）不应视为终止性错误
$data_root = $DataRoot.Trim('"')
$batches_root = Join-Path $data_root "walk_forward\batches"
if (-not (Test-Path $batches_root)) {
    throw "未找到 batch 根目录 $batches_root：请先用 scripts\batch\batch_walk_forward.ps1 跑一次 WF"
}
$risk_root = if ($RiskRoot) { $RiskRoot } else { Join-Path $data_root "walk_forward\terminal_risk_wf" }

function Get-BatchInfo {
    param([string]$Path)
    $snapshots = @(Get-ChildItem $Path -Recurse -File -Filter "walk_forward_持仓快照_*.csv" `
            -ErrorAction SilentlyContinue)
    $splits = @(Get-ChildItem $Path -Recurse -File -Filter "walk_forward_summary_*.csv" `
            -ErrorAction SilentlyContinue)
    [PSCustomObject]@{
        Name       = (Split-Path $Path -Leaf)
        Path       = $Path
        SnapshotCount = $snapshots.Count
        SnapshotFiles = $snapshots
        SplitCount = $splits.Count
        Modified   = (Get-Item $Path).LastWriteTime
    }
}

$allBatches = @(Get-ChildItem $batches_root -Directory | Sort-Object Name -Descending |
        ForEach-Object { Get-BatchInfo $_.FullName })

if ($ListBatches) {
    Write-Host "可选 batch（$batches_root）：" -ForegroundColor Cyan
    $allBatches | ForEach-Object {
        $mark = if ($_.SnapshotCount -gt 0) { "可打分" } else { "无持仓快照" }
        "{0}  生成={1:yyyy-MM-dd HH:mm}  summary文件={2,-3} 持仓快照={3,-3} {4}" -f `
            $_.Name, $_.Modified, $_.SplitCount, $_.SnapshotCount, $mark
    }
    return
}

# ── 解析目标 batch ──────────────────────────────────────────────
if (-not $Batch -or $Batch -eq "latest") {
    $target = $allBatches | Where-Object { $_.SnapshotCount -gt 0 } | Select-Object -First 1
    if (-not $target) { $target = $allBatches | Select-Object -First 1 }
    Write-Host "未指定 -Batch：自动选择最新可打分 batch $($target.Name)" -ForegroundColor DarkGray
}
elseif (Test-Path $Batch) {
    $target = Get-BatchInfo (Resolve-Path $Batch).Path
}
else {
    $candidate = Join-Path $batches_root $Batch
    if (-not (Test-Path $candidate)) {
        $names = ($allBatches | Select-Object -First 8 -ExpandProperty Name) -join ", "
        throw "batch 不存在：$Batch。可用（最近 8 个）：$names；或用 -ListBatches 查看全部"
    }
    $target = Get-BatchInfo $candidate
}

if ($target.SnapshotCount -eq 0) {
    throw @"
batch $($target.Name) 没有持仓快照（v0.114.0 之前的批次不产出该文件）。
解决：1) 用 scripts\batch\batch_walk_forward.ps1 重跑（自动导出快照）；
      2) 或复用已注册模型只跑 OOS 回测：
         py .\scripts\walk_forward.py --skip-training --start-model-version <N> --wf-summary-csv $($target.Path)\raw\walk_forward_summary_dummy_0001.csv
"@
}

Write-Host ("=" * 78) -ForegroundColor DarkGray
Write-Host "政策旁路打分（P2-1）" -ForegroundColor Green
Write-Host ("  batch      : {0}（{1:yyyy-MM-dd HH:mm}）" -f $target.Name, $target.Modified)
Write-Host ("  持仓快照   : {0} 个" -f $target.SnapshotCount)
Write-Host ("  风险模型根 : {0}" -f $risk_root)
Write-Host ("  特征集     : {0}" -f ($Arms -join ", "))
Write-Host ("  阈值网格   : P_hi={0} | P_abs={1}" -f ($PHi -join "/"), ($PAbs -join "/"))
Write-Host ("  截面分位   : {0}" -f $(if ($NoDayPct) { "跳过" } else { "计算" }))
Write-Host ("=" * 78) -ForegroundColor DarkGray

$repo_root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$logs_dir = Join-Path $repo_root "logs"
if (-not (Test-Path $logs_dir)) { New-Item -ItemType Directory -Force -Path $logs_dir | Out-Null }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$results = @()

foreach ($arm in $Arms) {
    $armTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $out_dir = if ($OutDir) { Join-Path $OutDir $arm.TrimStart("_") } else {
        Join-Path $target.Path ("policy_sidecar{0}" -f $arm)
    }
    $log_file = Join-Path $logs_dir ("policy_sidecar_{0}{1}_{2}.log" -f $target.Name, $arm, $timestamp)

    Write-Host "`n[打分] arm=$arm" -ForegroundColor Cyan
    Write-Host "  输出: $out_dir" -ForegroundColor DarkGray
    Write-Host "  日志: $log_file" -ForegroundColor DarkGray

    $pyArgs = @(".\scripts\analyze_policy_sidecar.py", "--snapshot") +
        @($target.SnapshotFiles | ForEach-Object { $_.FullName }) +
        @("--data-root", $data_root, "--risk-root", $risk_root, "--arm", $arm, "--out-dir", $out_dir,
          "--p-hi") + @($PHi | ForEach-Object { $_.ToString() }) +
        @("--p-abs") + @($PAbs | ForEach-Object { $_.ToString() })
    if ($NoDayPct) { $pyArgs += "--no-daypct" }
    if ($NoMissed) { $pyArgs += "--no-missed" }

    & py @pyArgs 2>&1 | Tee-Object -FilePath $log_file
    $exitCode = $LASTEXITCODE
    $armTimer.Stop()

    $ledger_path = Join-Path $out_dir "风险台账.csv"
    $trigger_path = Join-Path $out_dir "触发清单.csv"
    $scan_path = Join-Path $out_dir "阈值扫描.csv"
    $results += [PSCustomObject]@{
        Arm       = $arm
        ExitCode  = $exitCode
        Elapsed   = $armTimer.Elapsed.TotalSeconds
        Ledger    = if (Test-Path $ledger_path) { (Import-Csv $ledger_path | Measure-Object).Count } else { -1 }
        Trigger   = if (Test-Path $trigger_path) { (Import-Csv $trigger_path | Measure-Object).Count } else { -1 }
        OutDir    = $out_dir
        Log       = $log_file
    }
}

Write-Host "`n$("=" * 78)" -ForegroundColor DarkGray
Write-Host "汇总" -ForegroundColor Green
$results | ForEach-Object {
    $state = if ($_.ExitCode -eq 0) { "成功" } else { "失败(exit=$($_.ExitCode))" }
    "{0,-26} {1,-8} 台账={2,-6} 触发清单={3,-6} 耗时={4:N1}s" -f `
        $_.Arm, $state, $_.Ledger, $_.Trigger, $_.Elapsed
}
Write-Host "产物目录：" -ForegroundColor DarkGray
$results | ForEach-Object { Write-Host "  $($_.OutDir)" -ForegroundColor DarkGray }

# 打印每个 arm 的阈值扫描（行数少，直接展示便于对比）
foreach ($item in $results) {
    $scan_path = Join-Path $item.OutDir "阈值扫描.csv"
    if (Test-Path $scan_path) {
        Write-Host "`n[阈值扫描] $($item.Arm)" -ForegroundColor Cyan
        Import-Csv $scan_path | Format-Table -AutoSize
    }
}

$totalTimer.Stop()
Write-Host ("全部完成，总耗时 {0:N1} 秒" -f $totalTimer.Elapsed.TotalSeconds) -ForegroundColor Green
if (($results | Where-Object { $_.ExitCode -ne 0 }).Count -gt 0) { exit 1 }

# batch_paper_trade.ps1
#
# 用法：
# powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1 -StartDate 20260325 -EndDate 20260430 -ModelVersion 19321

param(
    [Parameter(Mandatory = $true)]
    [string]$StartDate,

    [Parameter(Mandatory = $true)]
    [string]$EndDate,

    [Parameter(Mandatory = $true)]
    [int]$ModelVersion,

    [Parameter(Mandatory = $false)]
    [bool]$ResetT0BeforeRun = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DateFromText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DateText,

        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    if ($DateText -notmatch "^\d{8}$") {
        throw "$FieldName 格式错误，应为 YYYYMMDD，例如 20260131"
    }

    try {
        return [datetime]::ParseExact($DateText, "yyyyMMdd", $null)
    }
    catch {
        throw "$FieldName 非法: $DateText"
    }
}

function Get-LastTradeDate {
    $statePath = ".\data\paper\state\last_trade_date.json"
    if (-not (Test-Path $statePath)) {
        return $null
    }

    try {
        $raw = Get-Content -Path $statePath -Raw -Encoding UTF8
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }
        $obj = $raw | ConvertFrom-Json
        return [string]$obj.last_trade_date
    }
    catch {
        throw "读取 last_trade_date 失败: $($_.Exception.Message)"
    }
}

function Resolve-NextOpenTradeDate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AfterDate
    )

    $pyCode = @'
import pandas as pd
import sys

after_date = sys.argv[1]
trade_cal = pd.read_parquet("./data/clean/trade_cal.parquet")
open_dates = (
    trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"]
    .astype(str)
    .sort_values()
    .tolist()
)
future = [d for d in open_dates if d > after_date]
print(future[0] if future else '')
'@

    try {
        $nextDate = (& py -c $pyCode $AfterDate | Out-String).Trim()
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "py -c 解析失败，exit code=$exitCode"
        }
        if ([string]::IsNullOrWhiteSpace($nextDate)) {
            return $null
        }
        return $nextDate
    }
    catch {
        throw "解析下一交易日失败: $($_.Exception.Message)"
    }
}

function Invoke-PaperTradeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host "[$Description] py $($Arguments -join ' ')" -ForegroundColor Gray
    & py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description 执行失败，exit code=$LASTEXITCODE"
    }
}

if (-not (Test-Path ".\scripts\paper_trade.py")) {
    throw "未找到 scripts\paper_trade.py，请在项目根目录执行本脚本。"
}

$start = Get-DateFromText -DateText $StartDate -FieldName "StartDate"
$end = Get-DateFromText -DateText $EndDate -FieldName "EndDate"

if ($start -gt $end) {
    throw "StartDate 不能晚于 EndDate"
}

$startStr = $start.ToString("yyyyMMdd")
$endStr = $end.ToString("yyyyMMdd")

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  纸面交易批量执行（按日期范围）" -ForegroundColor Cyan
Write-Host "  开始日期   : $startStr" -ForegroundColor Cyan
Write-Host "  结束日期   : $endStr" -ForegroundColor Cyan
Write-Host "  模型编号   : $ModelVersion" -ForegroundColor Cyan
Write-Host "  reset-t0   : $ResetT0BeforeRun" -ForegroundColor Cyan
Write-Host "  执行模式   : 首日指定日期，后续 trade-date=next" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 先写入模型编号到 data/paper/config.yaml（由 paper_trade.py config 统一渲染）
Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "config", "--model-version", "$ModelVersion") -Description "写入模型配置"

if ($ResetT0BeforeRun) {
    Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "adjust", "reset-t0") -Description "清理历史状态(reset-t0)"
}

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$runCount = 0
$currentRequest = $startStr

while ($true) {
    $runCount++

    Write-Host ""
    Write-Host "[任务 $runCount] 请求 trade-date: $currentRequest" -ForegroundColor Green

    Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "run", "--trade-date", $currentRequest) -Description "执行 paper_trade run"

    $lastTradeDate = Get-LastTradeDate
    if ([string]::IsNullOrWhiteSpace($lastTradeDate)) {
        throw "执行后未读取到 last_trade_date，无法继续推进 next"
    }

    Write-Host "已执行交易日: $lastTradeDate" -ForegroundColor White

    if ($lastTradeDate -ge $endStr) {
        Write-Host "到达结束日期，停止执行。" -ForegroundColor Yellow
        break
    }

    $nextOpenDate = Resolve-NextOpenTradeDate -AfterDate $lastTradeDate
    if ([string]::IsNullOrWhiteSpace($nextOpenDate)) {
        Write-Host "后续无可用交易日，停止执行。" -ForegroundColor Yellow
        break
    }

    if ($nextOpenDate -gt $endStr) {
        Write-Host "下一交易日($nextOpenDate)超过结束日期($endStr)，停止执行。" -ForegroundColor Yellow
        break
    }

    $currentRequest = "next"
}

$totalTimer.Stop()
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  执行完成" -ForegroundColor Magenta
Write-Host "  总执行轮次 : $runCount" -ForegroundColor Magenta
Write-Host "  总耗时     : $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta

exit 0

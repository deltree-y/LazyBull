# batch_paper_trade.ps1
#
# 用法：
# 1) 直接使用脚本内默认值（直接修改下方 param 默认值）
#    powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1
# 2) 运行时覆盖默认值
#    powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1 -StartDate 20260325 -EndDate 20260430 -ModelVersions 19321,19322
# 3) 只整合已有汇总，不重跑交易
#    powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1 -AppendSummaryOnly

param(
    [Parameter(Mandatory = $false)]
    [string]$StartDate = "20251231",

    [Parameter(Mandatory = $false)]
    [string]$EndDate = "20260708",

    [Parameter(Mandatory = $false)]
    [int[]]$ModelVersions = @(21142, 21157),

    [Parameter(Mandatory = $false)]
    [bool]$ResetT0BeforeRun = $true,

    [Parameter(Mandatory = $false)]
    [switch]$AppendSummaryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedStartDate = $StartDate
$resolvedEndDate = $EndDate
$resolvedModelVersions = $ModelVersions
$resolvedResetT0BeforeRun = $ResetT0BeforeRun
$resolvedAppendSummaryOnly = $AppendSummaryOnly.IsPresent

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
        $tmpPyPath = Join-Path $env:TEMP ("lazybull_next_trade_date_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
        Set-Content -Path $tmpPyPath -Value $pyCode -Encoding UTF8

        $nextDate = (& py $tmpPyPath $AfterDate | Out-String).Trim()
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "py 脚本解析失败，exit code=$exitCode"
        }
        if ([string]::IsNullOrWhiteSpace($nextDate)) {
            return $null
        }
        return $nextDate
    }
    catch {
        throw "解析下一交易日失败: $($_.Exception.Message)"
    }
    finally {
        if ($tmpPyPath -and (Test-Path $tmpPyPath)) {
            Remove-Item -Path $tmpPyPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Update-BatchSummaryCsv {
    param(
        [Parameter(Mandatory = $true)]
        [array]$Rows,

        [Parameter(Mandatory = $true)]
        [string]$CsvPath
    )

    $existingRows = @()
    if (Test-Path $CsvPath) {
        try {
            $existingRows = @(Import-Csv -Path $CsvPath)
        }
        catch {
            throw "读取已有汇总文件失败: $($_.Exception.Message)"
        }
    }

    $orderedColumns = @(
        '模型编号',
        '计划开始',
        '计划结束',
        '最终交易日',
        '总资产',
        '总收益率',
        '年化收益率',
        '执行轮次',
        '耗时',
        '完成时间',
        '状态',
        '错误'
    )

    $combinedRows = @()
    $seenKeys = @{}

    foreach ($row in $Rows) {
        $key = "$($row.模型编号)|$($row.计划开始)|$($row.计划结束)"
        if (-not $seenKeys.ContainsKey($key)) {
            $seenKeys[$key] = $true
            $combinedRows += $row
        }
    }

    foreach ($row in $existingRows) {
        $key = "$($row.模型编号)|$($row.计划开始)|$($row.计划结束)"
        if (-not $seenKeys.ContainsKey($key)) {
            $seenKeys[$key] = $true
            $combinedRows += $row
        }
    }

    $combinedRows |
        Select-Object $orderedColumns |
        Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8
}

function Read-BatchSummaryCsv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CsvPath
    )

    if (-not (Test-Path $CsvPath)) {
        return @()
    }

    try {
        return @(Import-Csv -Path $CsvPath)
    }
    catch {
        throw "读取汇总文件失败: $($_.Exception.Message)"
    }
}

function Get-PaperMetrics {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TradeDate
    )

    $pyCode = @'
import json, sys, yaml
from datetime import datetime
from pathlib import Path

import pandas as pd

trade_date = sys.argv[1]

# 1. 读取 config.yaml（嵌套结构，需解析 paper_trade 子段）
with open("./data/paper/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
pt = config.get("paper_trade", {})
initial_capital = float(pt.get("initial_capital", 0))
account_start_date = str(config.get("account_start_date", "") or "")

# 2. 若无 account_start_date，回退到 nav.parquet 最早日期（与 broker 一致）
if not account_start_date:
    nav_path = Path("./data/paper/nav/nav.parquet")
    if nav_path.exists():
        nav_df = pd.read_parquet(nav_path)
        if len(nav_df) > 0:
            account_start_date = str(nav_df["trade_date"].iloc[0])

# 3. 读取 account.json
with open("./data/paper/state/account.json", "r", encoding="utf-8") as f:
    account = json.load(f)
cash = float(account.get("cash", 0))
positions = account.get("positions", {})

# 4. 读取当日收盘价
date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
daily_path = Path(f"./data/clean/daily/{date_str}.parquet")
prices = {}
if daily_path.exists():
    df = pd.read_parquet(daily_path)
    for _, row in df.iterrows():
        price = row.get("close")
        if pd.notna(price) and float(price) > 0:
            prices[row["ts_code"]] = float(price)

# 5. 持仓市值（停牌回退买入价，与 broker 一致）
position_value = 0.0
for ts_code, pos in positions.items():
    shares = int(pos.get("shares", 0))
    buy_price = float(pos.get("buy_price", 0))
    price = prices.get(ts_code, 0.0)
    if price <= 0:
        price = buy_price
    position_value += shares * price

total_assets = cash + position_value

# 6. CAGR（与 broker._calculate_annualized_return 一致）
total_return_pct = (total_assets / initial_capital - 1.0) * 100.0 if initial_capital > 0 else 0.0

start_date = account_start_date if account_start_date else trade_date
d0 = datetime.strptime(start_date, "%Y%m%d")
d1 = datetime.strptime(trade_date, "%Y%m%d")
days = (d1 - d0).days
if days > 0 and initial_capital > 0 and total_assets > 0:
    annualized = ((total_assets / initial_capital) ** (365.0 / days) - 1.0) * 100.0
else:
    annualized = None

print(json.dumps({
    "total_assets": round(total_assets, 2),
    "total_return_pct": round(total_return_pct, 2),
    "annualized_return_pct": round(annualized, 2) if annualized is not None else None,
}))
'@

    $tmpPy = Join-Path $env:TEMP "lazybull_metrics_$([Guid]::NewGuid().ToString('N')).py"
    try {
        Set-Content -Path $tmpPy -Value $pyCode -Encoding UTF8
        $raw = & py $tmpPy $TradeDate | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "Python 指标脚本退出码=$LASTEXITCODE, 输出=$raw"
        }
        return ($raw.Trim() | ConvertFrom-Json)
    }
    finally {
        Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
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

$start = Get-DateFromText -DateText $resolvedStartDate -FieldName "StartDate"
$end = Get-DateFromText -DateText $resolvedEndDate -FieldName "EndDate"

if ($start -gt $end) {
    throw "StartDate 不能晚于 EndDate"
}

if (-not $resolvedModelVersions -or $resolvedModelVersions.Count -eq 0) {
    throw "ModelVersions 不能为空"
}

$normalizedModels = @()
foreach ($m in $resolvedModelVersions) {
    if ($m -le 0) {
        throw "模型编号必须为正整数: $m"
    }
    if (-not ($normalizedModels -contains $m)) {
        $normalizedModels += $m
    }
}

$startStr = $start.ToString("yyyyMMdd")
$endStr = $end.ToString("yyyyMMdd")

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  纸面交易批量执行（多模型）" -ForegroundColor Cyan
Write-Host "  开始日期   : $startStr" -ForegroundColor Cyan
Write-Host "  结束日期   : $endStr" -ForegroundColor Cyan
Write-Host "  模型编号   : $($normalizedModels -join ', ')" -ForegroundColor Cyan
Write-Host "  reset-t0   : $resolvedResetT0BeforeRun" -ForegroundColor Cyan
Write-Host "  仅整合汇总 : $resolvedAppendSummaryOnly" -ForegroundColor Cyan
Write-Host "  执行模式   : 首日指定日期，后续 trade-date=next" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".\data\reports")) {
    New-Item -Path ".\data\reports" -ItemType Directory -Force | Out-Null
}

$summaryRows = @()
$completionOrder = 0

if ($resolvedAppendSummaryOnly) {
    $summaryCsvPath = ".\data\reports\paper_trade_batch_summary.csv"
    $summaryRows = @(Read-BatchSummaryCsv -CsvPath $summaryCsvPath)
    if ($summaryRows.Count -eq 0) {
        throw "append-summary-only 模式下未找到已有汇总文件: $summaryCsvPath"
    }

    $summaryRows = @($summaryRows | Sort-Object -Property 完成时间 -Descending)
    Update-BatchSummaryCsv -Rows $summaryRows -CsvPath $summaryCsvPath

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Magenta
    Write-Host "  汇总整合完成（未执行交易）" -ForegroundColor Magenta
    Write-Host "========================================================" -ForegroundColor Magenta
    $summaryRows | Format-Table 模型编号, 计划开始, 最终交易日, 总资产, 年化收益率, 状态 -AutoSize
    Write-Host "汇总文件: $summaryCsvPath" -ForegroundColor Magenta

    exit 0
}

foreach ($modelVersion in $normalizedModels) {
    Write-Host ""
    Write-Host "########################################################" -ForegroundColor DarkCyan
    Write-Host "开始执行模型: $modelVersion" -ForegroundColor DarkCyan
    Write-Host "########################################################" -ForegroundColor DarkCyan

    $modelTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $modelRunCount = 0
    $modelStatus = "成功"
    $modelError = ""
    $finalTradeDate = ""

    try {
        # 先写入模型编号到 data/paper/config.yaml（由 paper_trade.py config 统一渲染）
        Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "config", "--model-version", "$modelVersion") -Description "写入模型配置"

        if ($resolvedResetT0BeforeRun) {
            Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "adjust", "reset-t0") -Description "清理历史状态(reset-t0)"
        }

        $currentRequest = $startStr
        while ($true) {
            $modelRunCount++

            Write-Host ""
            Write-Host "[模型 $modelVersion | 任务 $modelRunCount] 请求 trade-date: $currentRequest" -ForegroundColor Green

            Invoke-PaperTradeCommand -Arguments @("scripts\paper_trade.py", "run", "--trade-date", $currentRequest) -Description "执行 paper_trade run"

            $lastTradeDate = Get-LastTradeDate
            if ([string]::IsNullOrWhiteSpace($lastTradeDate)) {
                throw "执行后未读取到 last_trade_date，无法继续推进 next"
            }

            $finalTradeDate = $lastTradeDate
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
    }
    catch {
        $modelStatus = "失败"
        $modelError = $_.Exception.Message
        Write-Host "模型 $modelVersion 执行失败: $modelError" -ForegroundColor Red
    }

    $modelTimer.Stop()

    # 通过独立 Python 脚本读取 config + account + clean daily，与 broker 口径完全一致
    $totalAssets = $null
    $totalReturnPct = $null
    $annualizedReturnPct = $null

    if ($modelStatus -eq "成功" -and $finalTradeDate) {
        try {
            $metrics = Get-PaperMetrics -TradeDate $finalTradeDate
            $totalAssets = $metrics.total_assets
            $totalReturnPct = $metrics.total_return_pct
            $annualizedReturnPct = $metrics.annualized_return_pct
        }
        catch {
            $modelError = "读取指标失败: $($_.Exception.Message)"
            $modelStatus = "失败"
        }
    }

    $summaryRows += [PSCustomObject]@{
        模型编号 = $modelVersion
        计划开始 = $startStr
        计划结束 = $endStr
        最终交易日 = $finalTradeDate
        总资产 = if ($null -ne $totalAssets) { "{0:N2}" -f $totalAssets } else { "N/A" }
        总收益率 = if ($null -ne $totalReturnPct) { "{0:F2}%" -f $totalReturnPct } else { "N/A" }
        年化收益率 = if ($null -ne $annualizedReturnPct) { "{0:F2}%" -f $annualizedReturnPct } else { "N/A" }
        执行轮次 = $modelRunCount
        耗时 = $modelTimer.Elapsed.ToString('hh\:mm\:ss')
        完成时间 = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        完成序号 = ++$completionOrder
        状态 = $modelStatus
        错误 = $modelError
    }
}

$summaryCsvPath = ".\data\reports\paper_trade_batch_summary.csv"
$summaryRows = @($summaryRows | Sort-Object -Property 完成序号 -Descending)
Update-BatchSummaryCsv -Rows $summaryRows -CsvPath $summaryCsvPath

Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  多模型执行汇总" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
$summaryRows | Format-Table 模型编号, 计划开始, 最终交易日, 总资产, 年化收益率, 状态 -AutoSize
Write-Host "汇总文件: $summaryCsvPath" -ForegroundColor Magenta

exit 0

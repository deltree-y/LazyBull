# batch_paper_trade.ps1
#
# 用法：
# 1) 直接使用脚本内默认值（推荐先在“默认配置区”填写）
#    powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1
# 2) 运行时覆盖默认值
#    powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_paper_trade.ps1 -StartDate 20260325 -EndDate 20260430 -ModelVersions 19321,19322

# =========================
# 默认配置区（可直接编辑）
# =========================
$DefaultStartDate = "20251231"
$DefaultEndDate = "20260708"
$DefaultModelVersions = @(21142,21157)
$DefaultResetT0BeforeRun = $true

param(
    [Parameter(Mandatory = $false)]
    [string]$StartDate,

    [Parameter(Mandatory = $false)]
    [string]$EndDate,

    [Parameter(Mandatory = $false)]
    [int[]]$ModelVersions,

    [Parameter(Mandatory = $false)]
    [bool]$ResetT0BeforeRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedStartDate = if ([string]::IsNullOrWhiteSpace($StartDate)) { $DefaultStartDate } else { $StartDate }
$resolvedEndDate = if ([string]::IsNullOrWhiteSpace($EndDate)) { $DefaultEndDate } else { $EndDate }
$resolvedModelVersions = if ($ModelVersions -and $ModelVersions.Count -gt 0) { $ModelVersions } else { $DefaultModelVersions }
$resolvedResetT0BeforeRun = if ($PSBoundParameters.ContainsKey('ResetT0BeforeRun')) { $ResetT0BeforeRun } else { $DefaultResetT0BeforeRun }

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

function Get-NavSummary {
    $pyCode = @'
import json
from datetime import datetime

import pandas as pd

file_path = "./data/paper/nav/nav.parquet"

try:
    df = pd.read_parquet(file_path)
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"读取 nav.parquet 失败: {exc}"}, ensure_ascii=False))
    raise SystemExit(0)

if df is None or df.empty:
    print(json.dumps({"ok": False, "error": "nav.parquet 为空，无法汇总"}, ensure_ascii=False))
    raise SystemExit(0)

if "trade_date" not in df.columns or "total_value" not in df.columns:
    print(json.dumps({"ok": False, "error": "nav.parquet 缺少 trade_date 或 total_value 字段"}, ensure_ascii=False))
    raise SystemExit(0)

df = df.sort_values("trade_date").reset_index(drop=True)

start_trade_date = str(df.iloc[0]["trade_date"])
end_trade_date = str(df.iloc[-1]["trade_date"])
start_total_value = float(df.iloc[0]["total_value"])
end_total_value = float(df.iloc[-1]["total_value"])

total_return_pct = 0.0
if start_total_value > 0:
    total_return_pct = (end_total_value / start_total_value - 1.0) * 100.0

annualized_return_pct = None
try:
    d0 = datetime.strptime(start_trade_date, "%Y%m%d")
    d1 = datetime.strptime(end_trade_date, "%Y%m%d")
    days = (d1 - d0).days
    if days > 0 and start_total_value > 0 and end_total_value > 0:
        annualized_return_pct = ((end_total_value / start_total_value) ** (365.0 / days) - 1.0) * 100.0
except Exception:
    annualized_return_pct = None

print(
    json.dumps(
        {
            "ok": True,
            "start_trade_date": start_trade_date,
            "end_trade_date": end_trade_date,
            "start_total_value": start_total_value,
            "end_total_value": end_total_value,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return_pct,
        },
        ensure_ascii=False,
    )
)
'@

    try {
        $tmpPyPath = Join-Path $env:TEMP ("lazybull_nav_summary_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
        Set-Content -Path $tmpPyPath -Value $pyCode -Encoding UTF8

        $raw = (& py $tmpPyPath | Out-String).Trim()
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "py 汇总脚本执行失败，exit code=$exitCode"
        }
        if ([string]::IsNullOrWhiteSpace($raw)) {
            throw "py 汇总脚本无输出"
        }

        $obj = $raw | ConvertFrom-Json
        return $obj
    }
    catch {
        throw "读取 NAV 汇总失败: $($_.Exception.Message)"
    }
    finally {
        if ($tmpPyPath -and (Test-Path $tmpPyPath)) {
            Remove-Item -Path $tmpPyPath -Force -ErrorAction SilentlyContinue
        }
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
Write-Host "  执行模式   : 首日指定日期，后续 trade-date=next" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".\data\reports")) {
    New-Item -Path ".\data\reports" -ItemType Directory -Force | Out-Null
}

$summaryRows = @()

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

    $totalAssets = $null
    $totalReturnPct = $null
    $annualizedReturnPct = $null
    $navStartDate = ""
    $navEndDate = ""

    try {
        $navSummary = Get-NavSummary
        if ($navSummary.ok -eq $true) {
            $totalAssets = [double]$navSummary.end_total_value
            $totalReturnPct = [double]$navSummary.total_return_pct
            if ($null -ne $navSummary.annualized_return_pct) {
                $annualizedReturnPct = [double]$navSummary.annualized_return_pct
            }
            $navStartDate = [string]$navSummary.start_trade_date
            $navEndDate = [string]$navSummary.end_trade_date
        }
        else {
            if ([string]::IsNullOrWhiteSpace($modelError)) {
                $modelError = [string]$navSummary.error
            }
            if ($modelStatus -eq "成功") {
                $modelStatus = "失败"
            }
        }
    }
    catch {
        if ([string]::IsNullOrWhiteSpace($modelError)) {
            $modelError = $_.Exception.Message
        }
        if ($modelStatus -eq "成功") {
            $modelStatus = "失败"
        }
    }

    $summaryRows += [PSCustomObject]@{
        模型编号 = $modelVersion
        计划开始 = $startStr
        计划结束 = $endStr
        NAV开始 = $navStartDate
        NAV结束 = $navEndDate
        最终交易日 = $finalTradeDate
        总资产 = if ($null -ne $totalAssets) { "{0:N2}" -f $totalAssets } else { "N/A" }
        总收益率 = if ($null -ne $totalReturnPct) { "{0:F2}%" -f $totalReturnPct } else { "N/A" }
        年化收益率 = if ($null -ne $annualizedReturnPct) { "{0:F2}%" -f $annualizedReturnPct } else { "N/A" }
        执行轮次 = $modelRunCount
        耗时 = $modelTimer.Elapsed.ToString('hh\:mm\:ss')
        状态 = $modelStatus
        错误 = $modelError
    }
}

$summaryTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryCsvPath = ".\data\reports\paper_trade_batch_summary_$summaryTimestamp.csv"
$summaryRows | Export-Csv -Path $summaryCsvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  多模型执行汇总" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
$summaryRows | Format-Table 模型编号, 计划开始, 计划结束, NAV结束, 总资产, 年化收益率, 状态 -AutoSize
Write-Host "汇总文件: $summaryCsvPath" -ForegroundColor Magenta

exit 0

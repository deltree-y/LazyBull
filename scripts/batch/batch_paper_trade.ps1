# batch_paper_trade.ps1
#
# Chinese text is emitted via Base64 decoding to avoid encoding issues in PowerShell 5.1.

# =========================
# Config
# =========================
$start_date = "20260325" #"20240812"
$end_date = "20260430"#"20250212"#"20240909"
$reset_t0_before_run = $true

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function CN {
    param([Parameter(Mandatory = $true)][string]$Base64Text)
    return [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($Base64Text))
}

function Get-DateFromText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DateText,

        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    if ($DateText -notmatch "^\d{8}$") {
        throw "$FieldName $(CN '5qC85byP6ZSZ6K+v77yM5bqU5Li6IFlZWVlNTURE77yM5L6L5aaCIDIwMjYwMTMx')"
    }

    try {
        return [datetime]::ParseExact($DateText, "yyyyMMdd", $null)
    }
    catch {
        throw "$FieldName $(CN '6Z2e5rOV77ya')$DateText"
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
        throw "$(CN '6K+75Y+WIGxhc3RfdHJhZGVfZGF0ZSDlpLHotKU6IA==')$($_.Exception.Message)"
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
trade_cal = pd.read_parquet('./data/clean/trade_cal.parquet')
open_dates = (
    trade_cal.loc[trade_cal['is_open'] == 1, 'cal_date']
    .astype(str)
    .sort_values()
    .tolist()
)
future = [d for d in open_dates if d > after_date]
print(future[0] if future else '')
'@

    try {
        $nextDate = (& py -c $pyCode $AfterDate | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($nextDate)) {
            return $null
        }
        return $nextDate
    }
    catch {
        throw "$(CN '6Kej5p6Q5LiL5LiA5Lqk5piT5pel5aSx6LSlOiA=')$($_.Exception.Message)"
    }
}

if (-not (Test-Path ".\scripts\paper_trade.py")) {
    throw (CN '5pyq5om+5YiwIHNjcmlwdHNccGFwZXJfdHJhZGUucHnvvIzor7flnKjpobnnm67moLnnm67lvZXmiafooYzmnKzohJrmnKzjgII=')
}

$start = Get-DateFromText -DateText $start_date -FieldName "start_date"
$end = Get-DateFromText -DateText $end_date -FieldName "end_date"

if ($start -gt $end) {
    throw (CN 'c3RhcnRfZGF0ZSDkuI3og73mmZrkuo4gZW5kX2RhdGU=')
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  $(CN '57q46Z2i5Lqk5piT5om56YeP5omn6KGM')" -ForegroundColor Cyan
Write-Host ("  {0}" -f ((CN '6LW35q2i5pel5pyfICAgIDogezB9IH4gezF9') -f $start.ToString('yyyyMMdd'), $end.ToString('yyyyMMdd'))) -ForegroundColor Cyan
Write-Host ("  reset-t0    : {0}" -f $reset_t0_before_run) -ForegroundColor Cyan
Write-Host "  $(CN '5omn6KGM5qih5byPICAgIDog6aaW5pel5oyH5a6a5pel5pyf77yM5ZCO57utIHRyYWRlLWRhdGU9bmV4dA==')" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if ($reset_t0_before_run) {
    $resetCmd = "py scripts\paper_trade.py adjust reset-t0"
    Write-Host ("[{0}] {1}" -f (CN '6aKE5aSE55CG'), (CN '5omn6KGMIHJlc2V0LXQw')) -ForegroundColor Yellow
    Write-Host $resetCmd -ForegroundColor Gray
    Invoke-Expression $resetCmd
    if ($LASTEXITCODE -ne 0) {
        throw ((CN 'cmVzZXQtdDAg5omn6KGM5aSx6LSl77yMZXhpdCBjb2RlPXswfQ==') -f $LASTEXITCODE)
    }
}

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$runCount = 0
$failed = 0
$currentRequest = $start.ToString("yyyyMMdd")
$endStr = $end.ToString("yyyyMMdd")

while ($true) {
    $runCount++
    $pythonCmd = "py scripts\paper_trade.py run --trade-date $currentRequest"

    Write-Host ""
    Write-Host ("[{0}]" -f ((CN '5Lu75YqhIHswfSDor7fmsYIgdHJhZGUtZGF0ZTogezF9') -f $runCount, $currentRequest)) -ForegroundColor Green
    Write-Host $pythonCmd -ForegroundColor Gray

    Invoke-Expression $pythonCmd
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $failed++
        Write-Host ("[{0}] {1}" -f (CN '6K2m5ZGK'), ((CN '6K+35rGCIHswfSDmiafooYzlpLHotKXvvIhleGl0IGNvZGU6IHsxfe+8iQ==') -f $currentRequest, $exitCode)) -ForegroundColor Red
    }

    $lastTradeDate = Get-LastTradeDate
    if ([string]::IsNullOrWhiteSpace($lastTradeDate)) {
        throw (CN '5omn6KGM5ZCO5pyq6K+75Y+W5YiwIGxhc3RfdHJhZGVfZGF0Ze+8jOaXoOazlee7p+e7rSBuZXh0IOaOqOi/m+OAgg==')
    }

    Write-Host ((CN '5bey5omn6KGM5Lqk5piT5pelOiB7MH0=') -f $lastTradeDate) -ForegroundColor White

    if ($lastTradeDate -gt $endStr) {
        Write-Host ("[{0}] {1}" -f (CN '5YGc5q2i'), ((CN '5b2T5YmN5omn6KGM5pelIHswfSDlt7LotoXov4cgZW5kX2RhdGU9ezF944CC') -f $lastTradeDate, $endStr)) -ForegroundColor Yellow
        break
    }

    $nextOpenDate = Resolve-NextOpenTradeDate -AfterDate $lastTradeDate
    if ([string]::IsNullOrWhiteSpace($nextOpenDate)) {
        Write-Host ("[{0}] {1}" -f (CN '5YGc5q2i'), (CN '5Lqk5piT5pel5Y6G5Lit5pyq5om+5Yiw5ZCO57ut5Lqk5piT5pel44CC')) -ForegroundColor Yellow
        break
    }

    if ($nextOpenDate -gt $endStr) {
        Write-Host ("[{0}] {1}" -f (CN '5YGc5q2i'), ((CN '5LiL5LiA5Liq5Lqk5piT5pelIHswfSDotoXov4cgZW5kX2RhdGU9ezF944CC') -f $nextOpenDate, $endStr)) -ForegroundColor Yellow
        break
    }

    $elapsedMs = $totalTimer.ElapsedMilliseconds
    $avgMs = $elapsedMs / $runCount
    Write-Host "--------------------------------------------------------" -ForegroundColor DarkCyan
    Write-Host ((CN '5bey5a6M5oiQ5Lu75Yqh5pWwOiB7MH3vvIjlpLHotKUgezF9IOS4qu+8iQ==') -f $runCount, $failed) -ForegroundColor White
    Write-Host ((CN '5bey6ICX5pe2ICAgIDogezB9') -f $totalTimer.Elapsed.ToString('hh\:mm\:ss')) -ForegroundColor White
    Write-Host ((CN '5bmz5Z2H5Y2V5qyh6ICX5pe2OiB7MH0=') -f [TimeSpan]::FromMilliseconds($avgMs).ToString('hh\:mm\:ss')) -ForegroundColor Yellow

    $currentRequest = "next"
}

$totalTimer.Stop()
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host ("  {0}" -f (CN '5omn6KGM5a6M5oiQ')) -ForegroundColor Magenta
Write-Host ("  {0}" -f ((CN '5oiQ5Yqf5Lu75Yqh5pWwOiB7MH0=') -f ($runCount - $failed))) -ForegroundColor Magenta
Write-Host ("  {0}" -f ((CN '5aSx6LSl5Lu75Yqh5pWwOiB7MH0=') -f $failed)) -ForegroundColor Magenta
Write-Host ("  {0}" -f ((CN '5oC76ICX5pe2ICAgIDogezB9') -f $totalTimer.Elapsed.ToString('hh\:mm\:ss'))) -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta

if ($failed -gt 0) {
    exit 1
}

exit 0

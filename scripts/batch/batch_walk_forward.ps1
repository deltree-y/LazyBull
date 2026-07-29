# batch_walk_forward.ps1
# Walk-forward 批量参数扫描脚本
#
# 用法：
#   在 "参数配置区" 里，把想对比的参数设置为多个值（数组），固定参数保留单个值。
#   脚本会自动遍历所有参数组合，每组调用一次 walk_forward.py，
#   全部完成后自动运行 compare_walk_forward.py 生成对比 Excel。
#
# 示例启动：
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch_walk_forward.ps1

# ============================================================
#  参数配置区（修改这里控制实验组合）
# ============================================================

# ── 跳过训练，仅调参回测（复用已有模型）──────────────────────
# 使用场景：模型已训练完毕，只想调整回测参数（止盈/止损/仓位等）时，跳过耗时的训练步骤
$skip_training           = $false   # $true 启用 | $false 禁用

# ── Walk-forward 时间段配置（支持多组）───────────────────────
# Label                : 时间段标签，仅用于日志/汇总展示
# SplitCount           : 训练切分数量
# FinalDate            : 最终日期（启用部署训练时=部署训练数据最后一天；禁用部署训练时=最后split测试结束日）
# ContinueDays         : 连续执行天数；>1 时会从 FinalDate 起按自然日逐日向后推进展开，并自动顺延到最近后一交易日
# StartModelVersion    : skip-training 模式下该时间段首个 split 对应模型版本号
# SelectedSplits       : 可选 split 下标列表（如 @(0,4,5,7,9)）；@() 或不填表示训练该时间段全部 split
$wf_period_configs = @(
    [PSCustomObject]@{
        Label = "0105"
        SplitCount = 14
        FinalDate = "20260105"# 20251231
        ContinueDays = 1
        StartModelVersion = 19206#18968#(0.035)#19206#(0.03)#19220#(0.04)
        #SelectedSplits = @(0,4,5,7,8,9,10,12,13)
        SelectedSplits = @()
    }
    #[PSCustomObject]@{
    #    Label = "0109"
    #    SplitCount = 14
    #    FinalDate = "20260109" # 20260209
    #    ContinueDays = 1
    #    StartModelVersion = 19234#19057#(0.035)#19234#(0.03)#19249#(0.04)
    #    SelectedSplits = @()
    #}
    #[PSCustomObject]@{
    #    Label = "0116"
    #    SplitCount = 14
    #    FinalDate = "20260116" # 20260324
    #    ContinueDays = 1
    #    StartModelVersion = 19264#19147#(0.035)#19264#(0.03)#19279#(0.04)
    #}
    #[PSCustomObject]@{
    #    Label = "0123"
    #    SplitCount = 14
    #    FinalDate = "20260123"
    #   ContinueDays = 1
    #    StartModelVersion = 15034
    #}
    #[PSCustomObject]@{
    #    Label = "0130"
    #    SplitCount = 14
    #    FinalDate = "20260130"
    #    ContinueDays = 1
    #    StartModelVersion = 15034
    #}
)

# ── Walk-forward 窗口配置 ─────────────────────────────────────
$step_list               = @("semiannual")    #此参数并不会改变切分,无用处 monthly | quarterly | semiannual | yearly
$train_window_years_list = @(6)             # 训练窗口年数
$test_window_months_list = @(6)             # 测试窗口月数（建议与标签持仓周期接近）
$val_ratio_list          = @(0.2)           # 训练数据内部验证集比例，可改为 @(0.1, 0.15, 0.2) 扫描

# ── 标签与任务 ────────────────────────────────────────────────
$algorithm_list          = @("xgboost")        # xgboost | lightgbm（训练算法）
$label_list              = @("neu_y_ret_20")#,"neu_y_ret_20")      # skip-training 默认只保留单标签，避免对同一组旧模型重复回测
$task_list               = @("regression")     # regression | classification
$label_transform_list    = @("cs_zscore")      # raw | cs_zscore（仅 regression 有效）

# ── 模型超参（想对比的参数放多个值，其余放单个值）──────────────
$n_estimators_list       = @(5000)      #. 树数量上限（配合早停，可多值扫描，如 @(500, 1000, 2000)）
$max_depth_list          = @(5)         #. XGB推荐9, LGB推荐5
$learning_rate_list      = @(0.03)      #0.009. XGB推荐0.005, LGB推荐0.005
$min_child_weight_list   = @(200)       #. XGB推荐150, LGB推荐200
$colsample_bytree_list   = @(0.3)       #. XGB/LGB均推荐0.3

$subsample_list          = @(0.8)       #. XGB推荐0.8, LGB推荐0.7
$reg_alpha_list          = @(0.05)      #. XGB推荐0.05, LGB推荐0.1
$reg_lambda_list         = @(7)         #. XGB推荐5.0, LGB推荐5.0
$gamma_list              = @(0.5)       #. 映射LGB min_split_gain。XGB推荐0.5, LGB推荐1.0
$num_leaves_list         = @(63)        #  仅LightGBM有效，XGBoost忽略。LGB推荐63

# ── 目标函数 ─────────────────────────────────────────────────
$objective_list          = @("mse")  # mse | lambdarank（排序学习，直接优化股票排序）
# ── 早停配置 ───────────────────────────────────────────────────
$early_stopping_rounds_list = @(50)    # 早停轮数，设为 0 则禁用早停（固定 n_estimators 棵树），可多值扫描如 @(100, 300, 500)
$early_stopping_metric   = "rank_ic"       # 早停指标：auto（mae/auc）| rank_ic（Spearman，尺度无关更稳定）


# ── rank-weight 配置（固定，不参与组合扫描）─────────────────────
$rank_weight_enabled     = $true   # $true 启用 | $false 禁用
$rank_weight_topk_list   = @(120)         #50
$rank_weight_list        = @(100)         #3
$rank_weight_topk_weight_mode = "linear_decay"   # linear_decay | flat

# ── 时间衰减权重 ──────────────────────────────────────────────
$time_decay_half_life_list = @(0)      # 半衰期（年）。0=禁用，1.0=1年前权重0.5，2.0=2年前权重0.5

# ── best_iteration 自适应候选重训 ─────────────────────────────
$adaptive_best_iter_retrain = $false  # $true 启用：低迭代/撞上限 split 自动重训候选，并按 Top30中位数(70%)+RankIC IR(30%) 加权打分择优
$adaptive_low_iter_max_retries = 1  # low_iter（best_iter<=50）随机种子重试上限

# ── 多种子 bagging（每个split用多个随机种子各训一个子模型取平均，降训练随机方差）─
$ensemble_seeds            = "42"#,61,82"#,29,23"#42,61,82,100,200"#,300"#,400,500,600,700,800,900,1000"#,220,719"     # 逗号分隔种子如 "42,1,2,3,4"；空=单种子（用 --random-state），与多偏移可叠加
$ensemble_seed_keep_top_ratio = 1  # 多种子筛选保留比例（0~1）
$ensemble_seed_keep_min_models = 3    # 多种子筛选最少保留模型数 

###  以下为因子选择
# ── 基本面因子（需先运行 download_raw.py --download fina_indicator）───
$enable_fundamental      = $true  # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤增加约2%

# ── 另类数据因子（股东人数、业绩预告等）(0310添加)──────────────────
$enable_alt              = $true  # $true 启用 | $false 禁用
# 0711关闭后得分大幅下降, 属于关键因子, 必须打开

# ── 融资融券因子（通过 margin_detail 接口下载）────────────────────
$enable_margin           = $true  # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤基本不变 
# 0610:打开后CAGR大幅下降
# 0711关闭后得分小降,但rank_ic最高,暂时保持打开

# ── 筹码胜率因子（需5000+积分，需先下载 cyq_perf）─────────────────
$enable_cyq              = $true # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤基本不变
# 0711 关掉后分数大幅下降, 属于关键因子, 必须打开

# ── 基金持仓因子（需5000+积分，需先下载 fund_portfolio）──────────
$enable_fund             = $true  # $true 启用 | $false 禁用
#改为false似乎可以提升少量收益并减少少量回撤, 并提升稳定效果
# 0711关闭后会有下降, 需要保持打开

# ── 业绩快报因子（需5000+积分，需先下载 express）─────────────────
$enable_express          = $true  # $true 启用 | $false 禁用
# 0711关闭后分数大幅下降, 需要保持打开

# ── 北向资金因子（moneyflow_hsgt 市场级广播, 2000+积分）───────────
$enable_north            = $false  # $true 启用 | $false 禁用
#实测:打开后CAGR下降约6%, 回撤上升8%
# 0711 似乎无影响, 那就保持关闭

# ── 龙虎榜因子（top_list 个股级, 2000+积分）──────────────────────
$enable_lhb              = $true  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约4%, 回撤提升约5%
# 0711关闭后微降,先保持打开

# ── 一致预期因子（report_rc 研报滚动聚合, 8000积分）──────────────
$enable_consensus        = $false  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约2%, 回撤无明显变化
# 0711关闭后大幅提升分数

# ── 一致预期修正因子（0512基于已有 report_rc 构建时序修正信号，无需额外下载）─
# 0711该因子有问题, 打开的话会导致训练异常bug
$enable_consensus_revision = $false  # $true 启用 | $false 禁用（实验性因子）

# ── 现金流质量因子（0512需 cashflow 接口，2000 积分，需先下载 cashflow 数据）─
$enable_cashflow_quality   = $true  # $true 启用 | $false 禁用（实验性因子）
# 0711关闭后得分下降

### 以下为训练功能选择
# ── 特征稳定性筛选（移除跨时期IC方向不一致的特征, 0326引入）──────────────
$feature_stability_filter = $false  # $true 启用 | $false 禁用（实验验证效果不佳）
# 0711关闭后得分上升

# ── 因子精简（基于IC分析排除低效因子, 0606引入）────────────────────────
$factor_prune             = $true  # $true 启用 | $false 禁用（需先运行 generate_factor_exclude_list.py）

# ── freshness 处理策略（P2-C）────────────────────────────────────────────
$freshness_strategy             = "state_keep_event_decay"  # state_keep_event_decay | drop_all
$event_freshness_half_life_days = 150                         # 事件型因子衰减半衰期（天）,修改此值无需重新build features

# ── 多偏移集成（每个split训练3个偏移模型取平均，消除边界敏感性, 0326引入）─
$ensemble_offsets          = 0      # 偏移月数（0=禁用, 1=±1个月→3模型）

# 0408引入
# ── 因子增强（开盘强度/日内波动结构/委托不平衡）───────
$enable_enhanced           = $true # $true 启用 | $false 禁用
# 0429关闭后CAGR下降约3%, 回撤保持不变

# ── 部署模型训练（walk-forward完成后自动训练部署模型）──────────
$deploy_train            = $true   # $true 启用 | $false 禁用

### 以下为回测功能选择
# ── 分批调仓（将资金分K份错开调仓，降低时点风险）────────────
$stagger_tranches_list   = @(1)    # 1=不分批, 4=分4批（等效每rebalance_freq/4天调仓1/4仓位）

# ── OOS 回测（每个 split 训练后运行真实组合回测）──────────────
$oos_backtest            = $true            # $true 启用 | $false 禁用
# 以下基础参数仅在 $oos_backtest = $true 时透传给 walk_forward.py
$oos_backtest_months     = 0                # 回测时长（月），0 = 自动对齐 test_window_months

$bt_top_n_list           = @(20)            # 回测持仓 Top N
$bt_rebalance_freq_list  = @($null)            # 调仓频率（可多值扫描；@($null) 表示从标签自动推断）
$bt_initial_capital      = 1000000          # 回测初始资金（默认：100万）
$bt_sell_timing_list     = @("open")        # 卖出时机：open | close
$bt_exclude_st           = $true            # $true 排除 ST | $false 不排除
$bt_min_list_days_list   = @(365)           # 最少上市天数
# 以下组合约束也仅在 $oos_backtest = $true 时生效
$bt_max_weight_per_stock_list = @(0.15)     # 单股最大权重，$null = 不限制，如 @(0.15, 0.20)
$bt_max_per_industry_list = @($null)        # 单行业最大持仓数，$null = 不限制，如 @(2, 3)

# ── OOS 仓位管理模式（仅在 $oos_backtest = $true 时参与回测）──────
# equal：等权 | score：按分数比例 | kelly：凯利公式 | half_kelly：半凯利（更稳健）
# 仅当 mode 为 kelly / half_kelly 时，Kelly 参数才会真正生效
$position_sizing_list             = @('equal')#, 'score', 'kelly', 'half_kelly') # equal | score | kelly | half_kelly
$kelly_vol_window_list           = @(60)      # Kelly 波动率窗口（交易日，可多值如 @(40, 60, 120)）
$kelly_max_leverage_list          = @(0.2)    # Kelly 单股仓位上限（可多值，如 @(0.15, 0.25)）

# ── 空仓/持有期拖尾提前调仓（独立开关）────
$enable_early_rebalance_on_empty_list = @($true)  # 可多值如 @($false, $true)

# ── OOS 止损（总开关）────────────────────────────────────────
# 0426这里应为true
$bt_stop_loss_enabled                 = $false   # $true 启用 | $false 禁用
# 以下参数仅在 $bt_stop_loss_enabled = $true 时生效
$bt_stop_loss_drawdown_pct_list       = @(20) # 回撤止损阈值（%）
$bt_stop_loss_consecutive_limit_down_list = @(2) # 连续跌停止损天数

# ── 路径 ─────────────────────────────────────────────────────
$data_root               = "./data"

# ── 运行后自动汇总对比（强烈建议保持 $true）──────────────────
$run_compare_after       = $true

# ── 全部完成后是否倒计时关机 ──────────────────────────────────
$shutdown_on_complete    = $false
$shutdown_timeout_sec    = 600

# ============================================================
#  以下为执行逻辑（通常不需修改）
# ============================================================

$effective_label_list = $label_list
if ($skip_training -and $label_list.Length -gt 1) {
    Write-Host "[提示] skip-training 模式下标签不会切换模型，仅保留首个标签避免重复任务。" -ForegroundColor Yellow
    $effective_label_list = @($label_list[0])
}

function Get-NextOrSameTradeDate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Date,
        [Parameter(Mandatory = $true)]
        [string[]]$TradeDates
    )

    foreach ($tradeDate in $TradeDates) {
        if ($tradeDate -ge $Date) {
            return $tradeDate
        }
    }

    throw "日期 $Date 晚于交易日历可用范围，无法顺延到有效交易日"
}

$normalized_wf_period_configs = @()
$wfPeriodIndex = 0
if (-not (Test-Path $data_root)) {
    throw "数据目录不存在: $data_root"
}

$resolvedDataRoot = (Resolve-Path $data_root).Path
$tradeCalLoaderScript = @"
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.lazybull.data import DataLoader, Storage

storage = Storage(root_path=r'''$resolvedDataRoot''')
loader = DataLoader(storage)
trade_cal = loader.load_clean_trade_cal()
if trade_cal is None:
    trade_cal = loader.load_trade_cal()

if trade_cal is None or len(trade_cal) == 0:
    raise SystemExit("无法通过 DataLoader 读取交易日历")

required_cols = {"cal_date", "is_open"}
if not required_cols.issubset(trade_cal.columns):
    missing = ", ".join(sorted(required_cols.difference(trade_cal.columns)))
    raise SystemExit(f"交易日历缺少必要列: {missing}")

cal_dates = trade_cal["cal_date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
is_open = trade_cal["is_open"].astype(str)
open_trade_dates = sorted(cal_dates[is_open == "1"].dropna().unique().tolist())
for trade_date in open_trade_dates:
    print(trade_date)
"@

$openTradeDates = @($tradeCalLoaderScript | py -)

if ($openTradeDates.Count -eq 0) {
    throw "交易日历中不存在开市日: $resolvedDataRoot"
}

foreach ($wfPeriod in $wf_period_configs) {
    $wfPeriodIndex++
    $seenAlignedFinalDates = [System.Collections.Generic.HashSet[string]]::new()
    $periodLabel = if ($wfPeriod.PSObject.Properties.Name -contains 'Label' -and -not [string]::IsNullOrWhiteSpace([string]$wfPeriod.Label)) {
        [string]$wfPeriod.Label
    } else {
        "period_$wfPeriodIndex"
    }
    $periodSplitCount = [int]$wfPeriod.SplitCount
    $periodFinalDate = [string]$wfPeriod.FinalDate
    $periodContinueDays = if (
        $wfPeriod.PSObject.Properties.Name -contains 'ContinueDays' -and
        -not [string]::IsNullOrWhiteSpace([string]$wfPeriod.ContinueDays)
    ) {
        [int]$wfPeriod.ContinueDays
    } else {
        1
    }
    $periodStartModelVersion = $wfPeriod.StartModelVersion
    $periodSelectedSplits = @()
    if ($wfPeriod.PSObject.Properties.Name -contains 'SelectedSplits' -and $null -ne $wfPeriod.SelectedSplits) {
        if ($wfPeriod.SelectedSplits -is [string]) {
            $splitTokens = $wfPeriod.SelectedSplits -split "[ ,;]+"
            foreach ($token in $splitTokens) {
                if ([string]::IsNullOrWhiteSpace($token)) {
                    continue
                }
                $splitIndex = [int]$token
                if ($splitIndex -lt 0) {
                    throw "wf_period_configs[$($wfPeriodIndex - 1)] 的 SelectedSplits 仅支持非负整数，收到: $splitIndex"
                }
                if ($periodSelectedSplits -notcontains $splitIndex) {
                    $periodSelectedSplits += $splitIndex
                }
            }
        } else {
            foreach ($splitIndexValue in $wfPeriod.SelectedSplits) {
                if ($null -eq $splitIndexValue -or [string]::IsNullOrWhiteSpace([string]$splitIndexValue)) {
                    continue
                }
                $splitIndex = [int]$splitIndexValue
                if ($splitIndex -lt 0) {
                    throw "wf_period_configs[$($wfPeriodIndex - 1)] 的 SelectedSplits 仅支持非负整数，收到: $splitIndex"
                }
                if ($periodSelectedSplits -notcontains $splitIndex) {
                    $periodSelectedSplits += $splitIndex
                }
            }
        }
    }

    if ($periodSplitCount -le 0 -or [string]::IsNullOrWhiteSpace($periodFinalDate)) {
        throw "wf_period_configs[$($wfPeriodIndex - 1)] 缺少有效的 SplitCount 或 FinalDate"
    }
    if ($periodContinueDays -le 0) {
        throw "wf_period_configs[$($wfPeriodIndex - 1)] 的 ContinueDays 必须大于 0"
    }
    if ($skip_training -and $null -eq $periodStartModelVersion) {
        throw "skip-training 模式要求每个时间段都设置 StartModelVersion，缺失时间段: $periodLabel"
    }

    for ($continueOffset = 0; $continueOffset -lt $periodContinueDays; $continueOffset++) {
        $candidateFinalDate = (
            [datetime]::ParseExact($periodFinalDate, 'yyyyMMdd', $null)
        ).AddDays($continueOffset).ToString('yyyyMMdd')
        $alignedFinalDate = Get-NextOrSameTradeDate -Date $candidateFinalDate -TradeDates $openTradeDates
        if (-not $seenAlignedFinalDates.Add($alignedFinalDate)) {
            continue
        }

        $normalized_wf_period_configs += [PSCustomObject]@{
            Label = $periodLabel
            SplitCount = $periodSplitCount
            FinalDate = $alignedFinalDate
            ContinueDays = $periodContinueDays
            ContinueOffset = $continueOffset
            StartModelVersion = $periodStartModelVersion
            SelectedSplits = $periodSelectedSplits
        }
    }
}

if ($normalized_wf_period_configs.Count -eq 0) {
    throw "wf_period_configs 不能为空"
}

$batch_run_id = "wf_batch_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$batch_output_root = Join-Path $data_root ("walk_forward\batches\{0}" -f $batch_run_id)
$batch_raw_dir = Join-Path $batch_output_root "raw"
$batch_compare_output = Join-Path $data_root "walk_forward\wf_comparison.xlsx"
New-Item -ItemType Directory -Path $batch_raw_dir -Force | Out-Null

$periodSummary = ($normalized_wf_period_configs | ForEach-Object {
    $selectedSplitsLabel = if ($_.SelectedSplits -and $_.SelectedSplits.Count -gt 0) {
        " | selected_splits=$($_.SelectedSplits -join ',')"
    } else {
        ""
    }
    if ($_.ContinueDays -gt 1) {
        "{0}:split={1}, final={2}, day={3}/{4}{5}" -f $_.Label, $_.SplitCount, $_.FinalDate, ($_.ContinueOffset + 1), $_.ContinueDays, $selectedSplitsLabel
    } else {
        "{0}:split={1}, final={2}{3}" -f $_.Label, $_.SplitCount, $_.FinalDate, $selectedSplitsLabel
    }
}) -join "; "

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count      = 0
$failed     = 0

# 计算总任务数（各列表长度的笛卡尔积）
$totalTasks = $normalized_wf_period_configs.Length *
              $algorithm_list.Length *
              $n_estimators_list.Length *
              $early_stopping_rounds_list.Length *
              $step_list.Length *
              $train_window_years_list.Length *
              $test_window_months_list.Length *
              $val_ratio_list.Length *
              $effective_label_list.Length *
              $task_list.Length *
              $label_transform_list.Length *
              $objective_list.Length *
              $max_depth_list.Length *
              $num_leaves_list.Length *
              $learning_rate_list.Length *
              $subsample_list.Length *
              $colsample_bytree_list.Length *
              $min_child_weight_list.Length *
              $reg_alpha_list.Length *
              $reg_lambda_list.Length *
              $gamma_list.Length *
              $time_decay_half_life_list.Length *
              $rank_weight_topk_list.Length *
              $rank_weight_list.Length *
              $bt_top_n_list.Length *
              $bt_rebalance_freq_list.Length *
              $bt_sell_timing_list.Length *
              $bt_min_list_days_list.Length *
              $bt_max_weight_per_stock_list.Length *
              $bt_max_per_industry_list.Length *
              $bt_stop_loss_drawdown_pct_list.Length *
              $bt_stop_loss_consecutive_limit_down_list.Length *
              $stagger_tranches_list.Length *
              $enable_early_rebalance_on_empty_list.Length *
              $position_sizing_list.Length *
              $kelly_vol_window_list.Length *
              $kelly_max_leverage_list.Length

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Walk-forward 批量实验" -ForegroundColor Cyan
Write-Host "  批次ID     : $batch_run_id" -ForegroundColor Cyan
Write-Host "  时间段数   : $($normalized_wf_period_configs.Count)" -ForegroundColor Cyan
Write-Host "  时间段列表 : $periodSummary" -ForegroundColor Cyan
Write-Host "  总任务数   : $totalTasks" -ForegroundColor Cyan
Write-Host "  数据目录   : $data_root" -ForegroundColor Cyan
Write-Host "  批次目录   : $batch_output_root" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($wfPeriod in $normalized_wf_period_configs) {
foreach ($algorithm in $algorithm_list) {
foreach ($step in $step_list) {
foreach ($train_window_years in $train_window_years_list) {
foreach ($test_window_months in $test_window_months_list) {
foreach ($val_ratio in $val_ratio_list) {
foreach ($label in $effective_label_list) {
foreach ($task in $task_list) {
foreach ($label_transform in $label_transform_list) {
foreach ($objective in $objective_list) {
foreach ($n_estimators in $n_estimators_list) {
foreach ($early_stopping_rounds in $early_stopping_rounds_list) {
foreach ($max_depth in $max_depth_list) {
foreach ($num_leaves in $num_leaves_list) {
foreach ($learning_rate in $learning_rate_list) {
foreach ($subsample in $subsample_list) {
foreach ($colsample_bytree in $colsample_bytree_list) {
foreach ($min_child_weight in $min_child_weight_list) {
foreach ($reg_alpha in $reg_alpha_list) {
foreach ($reg_lambda in $reg_lambda_list) {
foreach ($gamma in $gamma_list) {
foreach ($time_decay_half_life in $time_decay_half_life_list) {
foreach ($rank_weight_topk in $rank_weight_topk_list) {
foreach ($rank_weight in $rank_weight_list) {
foreach ($bt_top_n in $bt_top_n_list) {
foreach ($bt_rebalance_freq in $bt_rebalance_freq_list) {
foreach ($bt_sell_timing in $bt_sell_timing_list) {
foreach ($bt_min_list_days in $bt_min_list_days_list) {
foreach ($bt_max_weight_per_stock in $bt_max_weight_per_stock_list) {
foreach ($bt_max_per_industry in $bt_max_per_industry_list) {
foreach ($bt_stop_loss_drawdown_pct in $bt_stop_loss_drawdown_pct_list) {
foreach ($bt_stop_loss_consecutive_limit_down in $bt_stop_loss_consecutive_limit_down_list) {
foreach ($stagger_tranches in $stagger_tranches_list) {
foreach ($enable_early_rebalance_on_empty in $enable_early_rebalance_on_empty_list) {
foreach ($position_sizing in $position_sizing_list) {
foreach ($kelly_vol_window in $kelly_vol_window_list) {
foreach ($kelly_max_leverage in $kelly_max_leverage_list) {

    $count++
    $split_count = $wfPeriod.SplitCount
    $final_date = $wfPeriod.FinalDate
    $period_label = $wfPeriod.Label
    $continue_days = $wfPeriod.ContinueDays
    $continue_offset = $wfPeriod.ContinueOffset
    $batch_period_label = if ($continue_days -gt 1) {
        "{0}_{1}" -f $period_label, $final_date
    } else {
        $period_label
    }
    $start_model_version = $wfPeriod.StartModelVersion
    $selected_splits = @($wfPeriod.SelectedSplits)
    $summary_csv_path = Join-Path $batch_raw_dir ("walk_forward_summary_{0}_{1:D4}.csv" -f $period_label, $count)

    # 构建命令字符串
    $pythonCmd = "py .\scripts\walk_forward.py" +
                 " --algorithm $algorithm" +
                 " --split-count $split_count" +
                 " --final-date $final_date" +
                 " --step $step" +
                 " --train-window-years $train_window_years" +
                 " --test-window-months $test_window_months" +
                 " --val-ratio $val_ratio" +
                 " --label $label" +
                 " --task $task" +
                 " --label-transform $label_transform" +
                 " --objective $objective" +
                 " --n-estimators $n_estimators" +
                 " --max-depth $max_depth" +
                 " --num-leaves $num_leaves" +
                 " --learning-rate $learning_rate" +
                 " --subsample $subsample" +
                 " --colsample-bytree $colsample_bytree" +
                 " --min-child-weight $min_child_weight" +
                 " --reg-alpha $reg_alpha" +
                 " --reg-lambda $reg_lambda" +
                 " --gamma $gamma" +
                 " --rank-weight-topk $rank_weight_topk" +
                 " --rank-weight $rank_weight" +
                 " --rank-weight-topk-weight-mode $rank_weight_topk_weight_mode" +
                 " --data-root $data_root" +
                 " --early-stopping-rounds $early_stopping_rounds" +
                 " --early-stopping-metric $early_stopping_metric" +
                 " --time-decay-half-life $time_decay_half_life" +
                 " --freshness-strategy $freshness_strategy" +
                 " --event-freshness-half-life-days $event_freshness_half_life_days" +
                 " --batch-run-id $batch_run_id" +
                 " --batch-period-label $batch_period_label" +
                 " --wf-summary-csv `"$summary_csv_path`""

    if ($null -ne $selected_splits -and $selected_splits.Count -gt 0) {
        $pythonCmd += " --selected-split-indices $($selected_splits -join ' ')"
    }

    if (-not $rank_weight_enabled) {
        $pythonCmd += " --no-rank-weight"
    }

    if ($enable_fundamental) {
        $pythonCmd += " --enable-fundamental-features"
    }

    if ($enable_alt) {
        $pythonCmd += " --enable-alt-features"
    }

    if ($enable_margin) {
        $pythonCmd += " --enable-margin-features"
    }

    if ($enable_cyq) {
        $pythonCmd += " --enable-cyq-features"
    }

    if ($enable_fund) {
        $pythonCmd += " --enable-fund-features"
    }

    if ($enable_express) {
        $pythonCmd += " --enable-express-features"
    }

    if ($enable_north) {
        $pythonCmd += " --enable-north-features"
    }

    if ($enable_lhb) {
        $pythonCmd += " --enable-lhb-features"
    }

    if ($enable_consensus) {
        $pythonCmd += " --enable-consensus-features"
    }

    if ($feature_stability_filter) {
        $pythonCmd += " --feature-stability-filter"
    }

    if ($factor_prune) {
        $pythonCmd += " --factor-prune"
    }

    if ($ensemble_offsets -gt 0) {
        $pythonCmd += " --ensemble-offsets $ensemble_offsets"
    }

    if ($ensemble_seeds -ne "") {
        $pythonCmd += " --ensemble-seeds $ensemble_seeds" +
                      " --ensemble-seed-keep-top-ratio $ensemble_seed_keep_top_ratio" +
                      " --ensemble-seed-keep-min-models $ensemble_seed_keep_min_models"
    }

    if ($adaptive_best_iter_retrain) {
        $pythonCmd += " --adaptive-best-iter-retrain" +
                      " --adaptive-low-iter-max-retries $adaptive_low_iter_max_retries"
    }

    if ($enable_enhanced) {
        $pythonCmd += " --enable-enhanced-features"
    }

    if ($enable_cashflow_quality) {
        $pythonCmd += " --enable-cashflow-quality-features"
    }

    if ($enable_consensus_revision) {
        $pythonCmd += " --enable-consensus-revision-features"
    }

    if (-not $enable_early_rebalance_on_empty) {
        $pythonCmd += " --no-early-rebalance-on-empty"
    }

    if ($stagger_tranches -gt 1) {
        $pythonCmd += " --stagger-tranches $stagger_tranches"
    }

    if ($oos_backtest) {
        $pythonCmd += " --oos-backtest --oos-backtest-months $oos_backtest_months --bt-top-n $bt_top_n --bt-initial-capital $bt_initial_capital --bt-sell-timing $bt_sell_timing --bt-min-list-days $bt_min_list_days"
        if ($null -ne $bt_rebalance_freq) {
            $pythonCmd += " --bt-rebalance-freq $bt_rebalance_freq"
        }
        if (-not $bt_exclude_st) {
            $pythonCmd += " --bt-no-exclude-st"
        }
        if ($null -ne $bt_max_weight_per_stock) {
            $pythonCmd += " --bt-max-weight-per-stock $bt_max_weight_per_stock"
        }
        if ($null -ne $bt_max_per_industry) {
            $pythonCmd += " --bt-max-per-industry $bt_max_per_industry"
        }
        if ($bt_stop_loss_enabled) {
            $pythonCmd += " --bt-stop-loss-enabled" +
                          " --bt-stop-loss-drawdown-pct $bt_stop_loss_drawdown_pct" +
                          " --bt-stop-loss-consecutive-limit-down $bt_stop_loss_consecutive_limit_down"
        }
    } else {
        $pythonCmd += " --no-oos-backtest"
    }

    if ($position_sizing -ne 'equal') {
        $pythonCmd += " --position-sizing $position_sizing"
        if ($position_sizing -eq 'kelly' -or $position_sizing -eq 'half_kelly') {
            $pythonCmd += " --kelly-vol-window $kelly_vol_window --kelly-max-leverage $kelly_max_leverage"
        }
    }

    if (-not $deploy_train) {
        $pythonCmd += " --no-deploy-train"
    }

    if ($skip_training) {
        $pythonCmd += " --skip-training"
        if ($null -ne $start_model_version) {
            $pythonCmd += " --start-model-version $start_model_version"
        }
    }

    Write-Host ""
    if ($continue_days -gt 1) {
        Write-Host "[任务 $count / $totalTasks][时间段 $period_label][split=$split_count, final=$final_date][day=$($continue_offset + 1)/$continue_days]" -ForegroundColor Green
    } else {
        Write-Host "[任务 $count / $totalTasks][时间段 $period_label][split=$split_count, final=$final_date]" -ForegroundColor Green
    }
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

}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  #  end foreach（时间段+参数组合循环）

# ── 全部完成 ──────────────────────────────────────────────────
$totalTimer.Stop()
Write-Host ""
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host "  全部 $totalTasks 个实验已完成（失败 $failed 个）" -ForegroundColor Magenta
Write-Host "  总耗时: $($totalTimer.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta

# ── 自动汇总对比 ──────────────────────────────────────────────
if ($run_compare_after) {
    Write-Host ""
    Write-Host "[汇总对比] 正在汇总本批次结果（含跨时间段稳定性）..." -ForegroundColor Green
    py .\scripts\compare_walk_forward.py --raw-dir "$batch_raw_dir" --output "$batch_compare_output"
    Write-Host "[汇总对比] 完成，输出: $batch_compare_output" -ForegroundColor Green

    Write-Host "[汇总对比] 正在刷新 raw / batches 两份总表 ..." -ForegroundColor Green
    py .\scripts\compare_walk_forward.py --data-root $data_root
    Write-Host "[汇总对比] 完成，输出: $data_root\walk_forward\wf_comparison_raw.xlsx / wf_comparison_batches.xlsx" -ForegroundColor Green
}

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

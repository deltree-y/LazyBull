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

# ── Walk-forward 时间范围（固定，两端通常不需要多组）───────────
$wf_start_date           = "20130101"   #20130101   #20130224
$wf_end_date             = "20251231"   #20251231   #20260224

# ── Walk-forward 窗口配置 ─────────────────────────────────────
$step_list               = @("semiannual")   # monthly | quarterly | semiannual
$train_window_years_list = @(6)             # 训练窗口年数
$test_window_months_list = @(6)             # 测试窗口月数（建议与标签持仓周期接近）

# ── 标签与任务 ────────────────────────────────────────────────
$algorithm_list          = @("xgboost")        # xgboost | lightgbm（训练算法）
$label_list              = @("neu_y_ret_20")   # neu_y_ret_5 | neu_y_ret_10 | neu_y_ret_20 | y_ret_5 | y_ret_10 | y_ret_20
$task_list               = @("regression")     # regression | classification
$label_transform_list    = @("cs_zscore")      # raw | cs_zscore（仅 regression 有效）

# ── 模型超参（想对比的参数放多个值，其余放单个值）──────────────
$n_estimators            = 2000       # 固定：树数量上限（配合早停，不需要多组）
$max_depth_list          = @(4)         # XGB推荐9, LGB推荐5
$num_leaves_list         = @(63)        # 仅LightGBM有效，XGBoost忽略。LGB推荐63
$learning_rate_list      = @(0.011)      # XGB推荐0.005, LGB推荐0.005
$subsample_list          = @(0.8)       # XGB推荐0.8, LGB推荐0.7
$colsample_bytree_list   = @(0.3)       # XGB/LGB均推荐0.3
$min_child_weight_list   = @(150)       # XGB推荐150, LGB推荐200
$reg_alpha_list          = @(0.05)       # XGB推荐0.05, LGB推荐0.1
$reg_lambda_list         = @(1.0)       # XGB推荐1.0, LGB推荐5.0
$gamma_list              = @(0.5)       # 映射LGB min_split_gain。XGB推荐0.5, LGB推荐1.0

# ── 早停配置 ───────────────────────────────────────────────────
$early_stopping_rounds   = 500       # 早停轮数，设为 0 则禁用早停（固定 n_estimators 棵树）
$early_stopping_metric   = "auto" # 早停指标：auto（mae/auc）| rank_ic（Spearman，尺度无关更稳定）

# ── rank-weight 配置（固定，不参与组合扫描）─────────────────────
$rank_weight_enabled     = $true   # $true 启用 | $false 禁用
$rank_weight_topk_list   = @(50)
$rank_weight_list        = @(3)

# ── 时间衰减权重 ──────────────────────────────────────────────
$time_decay_half_life    = 0         # 半衰期（年）。0=禁用，1.0=1年前权重0.5，2.0=2年前权重0.5

# ── 目标函数 ─────────────────────────────────────────────────
$objective_list          = @("mse")  # mse | lambdarank（排序学习，直接优化股票排序）

# ── 基本面因子（需先运行 download_raw.py --download fina_indicator）───
$enable_fundamental      = $true  # $true 启用 | $false 禁用

# ── 另类数据因子（股东人数、业绩预告等）(0310添加)──────────────────
$enable_alt              = $true  # $true 启用 | $false 禁用

# ── 融资融券因子（通过 margin_detail 接口下载）────────────────────
$enable_margin           = $true  # $true 启用 | $false 禁用

# ── 筹码胜率因子（需5000+积分，需先下载 cyq_perf）─────────────────
$enable_cyq              = $true  # $true 启用 | $false 禁用

# ── 基金持仓因子（需5000+积分，需先下载 fund_portfolio）──────────
$enable_fund             = $true  # $true 启用 | $false 禁用

# ── 业绩快报因子（需5000+积分，需先下载 express）─────────────────
$enable_express          = $true  # $true 启用 | $false 禁用

# ── 特征稳定性筛选（移除跨时期IC方向不一致的特征, 0326引入）──────────────
$feature_stability_filter = $false  # $true 启用 | $false 禁用（实验验证效果不佳）

# ── 多偏移集成（每个split训练3个偏移模型取平均，消除边界敏感性, 0326引入）─
$ensemble_offsets          = 1      # 偏移月数（0=禁用, 1=±1个月→3模型）

# ── 部署模型训练（walk-forward完成后自动训练部署模型）──────────
$deploy_train            = $false   # $true 启用 | $false 禁用

# ── OOS 回测（每个 split 训练后运行真实组合回测）──────────────
$oos_backtest            = $true   # $true 启用 | $false 禁用
$oos_backtest_months     = 0       # 回测时长（月），0 = 自动对齐 test_window_months
$bt_top_n_list           = @(20)   # 回测持仓 Top N
$bt_rebalance_freq       = $null   # 调仓频率（$null 表示从标签自动推断）
$bt_weight_method        = "equal" # 权重方法：equal（等权）| score（按预测分数加权）

# ── 行业动量过滤（剔除弱势行业股票，自动补位）──────────────────
$industry_momentum_filter     = $false  # $true 启用 | $false 禁用
$industry_momentum_bottom_pct = 0.5     # 剔除排名后 X% 的行业（0~1），默认 0.2

# ── 市场择时仓位管理 ─────────────────────────────────────────
$market_regime                = $false       # $true 启用 | $false 禁用
$market_regime_mode_list      = @("combined")  # binary | vol_target | trend | combined
$market_regime_bear_threshold_list = @(-0.03)  # binary 模式：mkt_ret_avg_20 低于此值判定为熊市
$market_regime_bear_exposure  = 0.3            # binary 模式：熊市仓位系数（0~1）
$market_regime_vol_target_list = @(0.2)       # vol_target/combined 模式：年化波动率目标
$market_regime_trend_threshold = 1.0           # trend/combined 模式：mkt_ma_trend 低于此值降仓
$market_regime_min_exposure    = 0.18           # 非 binary 模式：最低仓位下限
$market_regime_combine_method  = "min"         # combined 模式组合方式：min | multiply
$market_regime_trend_guard     = $true        # combined 模式趋势保护：上行趋势跳过 vol 降仓
$market_regime_drawdown_guard  = $true        # 回撤保护：已大幅下跌时停止降仓，避免底部踏空
$market_regime_drawdown_threshold = -0.08     # 回撤保护阈值：mkt_drawdown_20 低于此值停止降仓

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

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count      = 0
$failed     = 0

# 计算总任务数（各列表长度的笛卡尔积）
$totalTasks = $algorithm_list.Length *
              $step_list.Length *
              $train_window_years_list.Length *
              $test_window_months_list.Length *
              $label_list.Length *
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
              $rank_weight_topk_list.Length *
              $rank_weight_list.Length *
              $market_regime_bear_threshold_list.Length *
              $market_regime_mode_list.Length *
              $market_regime_vol_target_list.Length *
              $bt_top_n_list.Length

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Walk-forward 批量实验" -ForegroundColor Cyan
Write-Host "  WF 区间    : $wf_start_date ~ $wf_end_date" -ForegroundColor Cyan
Write-Host "  总任务数   : $totalTasks" -ForegroundColor Cyan
Write-Host "  数据目录   : $data_root" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($algorithm in $algorithm_list) {
foreach ($step in $step_list) {
foreach ($train_window_years in $train_window_years_list) {
foreach ($test_window_months in $test_window_months_list) {
foreach ($label in $label_list) {
foreach ($task in $task_list) {
foreach ($label_transform in $label_transform_list) {
foreach ($objective in $objective_list) {
foreach ($max_depth in $max_depth_list) {
foreach ($num_leaves in $num_leaves_list) {
foreach ($learning_rate in $learning_rate_list) {
foreach ($subsample in $subsample_list) {
foreach ($colsample_bytree in $colsample_bytree_list) {
foreach ($min_child_weight in $min_child_weight_list) {
foreach ($reg_alpha in $reg_alpha_list) {
foreach ($reg_lambda in $reg_lambda_list) {
foreach ($gamma in $gamma_list) {
foreach ($rank_weight_topk in $rank_weight_topk_list) {
foreach ($rank_weight in $rank_weight_list) {
foreach ($market_regime_bear_threshold in $market_regime_bear_threshold_list) {
foreach ($market_regime_mode in $market_regime_mode_list) {
foreach ($market_regime_vol_target in $market_regime_vol_target_list) {
foreach ($bt_top_n in $bt_top_n_list) {

    $count++

    # 构建命令字符串
    $pythonCmd = "py .\scripts\walk_forward.py" +
                 " --algorithm $algorithm" +
                 " --wf-start-date $wf_start_date" +
                 " --wf-end-date $wf_end_date" +
                 " --step $step" +
                 " --train-window-years $train_window_years" +
                 " --test-window-months $test_window_months" +
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
                 " --data-root $data_root" +
                 " --early-stopping-rounds $early_stopping_rounds" +
                 " --early-stopping-metric $early_stopping_metric" +
                 " --time-decay-half-life $time_decay_half_life"

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

    if ($feature_stability_filter) {
        $pythonCmd += " --feature-stability-filter"
    }

    if ($ensemble_offsets -gt 0) {
        $pythonCmd += " --ensemble-offsets $ensemble_offsets"
    }

    if ($market_regime) {
        $pythonCmd += " --market-regime" +
                      " --market-regime-mode $market_regime_mode" +
                      " --market-regime-bear-threshold $market_regime_bear_threshold" +
                      " --market-regime-bear-exposure $market_regime_bear_exposure" +
                      " --market-regime-vol-target $market_regime_vol_target" +
                      " --market-regime-trend-threshold $market_regime_trend_threshold" +
                      " --market-regime-min-exposure $market_regime_min_exposure" +
                      " --market-regime-combine-method $market_regime_combine_method"
        if (-not $market_regime_trend_guard) {
            $pythonCmd += " --no-market-regime-trend-guard"
        }
        if ($market_regime_drawdown_guard) {
            $pythonCmd += " --market-regime-drawdown-guard --market-regime-drawdown-threshold $market_regime_drawdown_threshold"
        } else {
            $pythonCmd += " --no-market-regime-drawdown-guard"
        }
    }

    if ($oos_backtest) {
        $pythonCmd += " --oos-backtest --oos-backtest-months $oos_backtest_months --bt-top-n $bt_top_n --bt-weight-method $bt_weight_method"
        if ($null -ne $bt_rebalance_freq) {
            $pythonCmd += " --bt-rebalance-freq $bt_rebalance_freq"
        }
    } else {
        $pythonCmd += " --no-oos-backtest"
    }

    if ($industry_momentum_filter) {
        $pythonCmd += " --industry-momentum-filter --industry-momentum-bottom-pct $industry_momentum_bottom_pct"
    }

    if (-not $deploy_train) {
        $pythonCmd += " --no-deploy-train"
    }

    Write-Host ""
    Write-Host "[任务 $count / $totalTasks]" -ForegroundColor Green
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

}}}}}}}}}}}}}}}}}}}}}}}  # end foreach（23 层）

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
    Write-Host "[汇总对比] 正在运行 compare_walk_forward.py ..." -ForegroundColor Green
    py .\scripts\compare_walk_forward.py --data-root $data_root
    Write-Host "[汇总对比] 完成，输出: $data_root\walk_forward\wf_comparison.xlsx" -ForegroundColor Green
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

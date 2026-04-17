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
$wf_start_date           = "20130209"   #20130101   #20130224
$wf_end_date             = "20260209"   #20251231   #20260224

# ── Walk-forward 窗口配置 ─────────────────────────────────────
$step_list               = @("semiannual")   # monthly | quarterly | semiannual
$train_window_years_list = @(6)             # 训练窗口年数
$test_window_months_list = @(6)             # 测试窗口月数（建议与标签持仓周期接近）
$val_ratio_list          = @(0.1)           # 训练数据内部验证集比例，可改为 @(0.1, 0.15, 0.2) 扫描

# ── 标签与任务 ────────────────────────────────────────────────
$algorithm_list          = @("xgboost")        # xgboost | lightgbm（训练算法）
$label_list              = @("neu_y_ret_20")#,"neu_y_ret_20")      # skip-training 默认只保留单标签，避免对同一组旧模型重复回测
$task_list               = @("regression")     # regression | classification
$label_transform_list    = @("cs_zscore")      # raw | cs_zscore（仅 regression 有效）

# ── 模型超参（想对比的参数放多个值，其余放单个值）──────────────
$n_estimators_list       = @(1000)      #. 树数量上限（配合早停，可多值扫描，如 @(500, 1000, 2000)）
$max_depth_list          = @(4)         #. XGB推荐9, LGB推荐5
$num_leaves_list         = @(63)        # 仅LightGBM有效，XGBoost忽略。LGB推荐63
$learning_rate_list      = @(0.012)     #. XGB推荐0.005, LGB推荐0.005
$subsample_list          = @(0.8)       #. XGB推荐0.8, LGB推荐0.7
$colsample_bytree_list   = @(0.3)       #. XGB/LGB均推荐0.3
$min_child_weight_list   = @(175)       #. XGB推荐150, LGB推荐200
$reg_alpha_list          = @(0.05)      #. XGB推荐0.05, LGB推荐0.1
$reg_lambda_list         = @(5)         #. XGB推荐1.0, LGB推荐5.0
$gamma_list              = @(0)         #. 映射LGB min_split_gain。XGB推荐0.5, LGB推荐1.0

# ── 早停配置 ───────────────────────────────────────────────────
$early_stopping_rounds_list = @(650)    # 早停轮数，设为 0 则禁用早停（固定 n_estimators 棵树），可多值扫描如 @(100, 300, 500)
$early_stopping_metric   = "auto"       # 早停指标：auto（mae/auc）| rank_ic（Spearman，尺度无关更稳定）

# ── rank-weight 配置（固定，不参与组合扫描）─────────────────────
$rank_weight_enabled     = $true   # $true 启用 | $false 禁用
$rank_weight_topk_list   = @(50)
$rank_weight_list        = @(3)

# ── 时间衰减权重 ──────────────────────────────────────────────
$time_decay_half_life    = 0         # 半衰期（年）。0=禁用，1.0=1年前权重0.5，2.0=2年前权重0.5

# ── 目标函数 ─────────────────────────────────────────────────
$objective_list          = @("mse")  # mse | lambdarank（排序学习，直接优化股票排序）

###  以下为因子选择
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


### 以下为训练功能选择
# ── 特征稳定性筛选（移除跨时期IC方向不一致的特征, 0326引入）──────────────
$feature_stability_filter = $false  # $true 启用 | $false 禁用（实验验证效果不佳）

# ── 多偏移集成（每个split训练3个偏移模型取平均，消除边界敏感性, 0326引入）─
$ensemble_offsets          = 0      # 偏移月数（0=禁用, 1=±1个月→3模型）

# 0408引入
# ── 因子增强（开盘强度/日内波动结构/委托不平衡）───────
$enable_enhanced           = $true # $true 启用 | $false 禁用

# ── 部署模型训练（walk-forward完成后自动训练部署模型）──────────
$deploy_train            = $false   # $true 启用 | $false 禁用

# ── 跳过训练，仅调参回测（复用已有模型）──────────────────────
# 使用场景：模型已训练完毕，只想调整回测参数（止盈/止损/仓位等）时，跳过耗时的训练步骤
# start_model_version：第一个 split 对应的模型版本号，后续 split 依次 +1
# 例如：已有模型 v10~v24（共15个split），设 $start_model_version = 10
$skip_training           = $false   # $true 启用 | $false 禁用
#$start_model_version     = 10816    # 0224
$start_model_version     = 10830    # 0101
#$start_model_version     = 10802    # 0209
                                   #d3(0101):7969/9430(no enh)/9416(enh)
                                   #d3(0209):8165/9446(no enh)/9461(enh)/9601(ofst+1)
                                   #d2:8137

### 以下为回测功能选择
# ── 分批调仓（将资金分K份错开调仓，降低时点风险）────────────
$stagger_tranches_list   = @(1)    # 1=不分批, 4=分4批（等效每rebalance_freq/4天调仓1/4仓位）

# ── OOS 回测（每个 split 训练后运行真实组合回测）──────────────
$oos_backtest            = $true            # $true 启用 | $false 禁用
$oos_backtest_months     = 0                # 回测时长（月），0 = 自动对齐 test_window_months
$bt_top_n_list           = @(22)            # 回测持仓 Top N
$bt_rebalance_freq       = $null            # 调仓频率（$null 表示从标签自动推断）
$bt_weight_method        = "score"          # 权重方法："equal"（等权）| "score"（按预测分数加权）
$bt_initial_capital      = 1000000          # 回测初始资金（默认：100万）
$bt_sell_timing_list     = @("open")        # 卖出时机：open | close
$bt_exclude_st           = $true            # $true 排除 ST | $false 不排除
$bt_min_list_days_list   = @(365)           # 最少上市天数
$bt_max_weight_per_stock_list = @(0.15)     # 单股最大权重，$null=不限制，如 @(0.15, 0.20)
$bt_max_per_industry_list = @($null)        # 单行业最大持仓数，$null=不限制，如 @(2, 3)

# ── OOS 信号入口门控 v2（替代旧置信度门控，0406引入）────────────
$signal_gate_mode = "composite"                 # "legacy" 旧公式 | "composite" 新公式(成本+百分位) | "disabled" 关闭
$signal_gate_cost_multiplier_list = @(0.3)      # composite: 门控严格度扫描
$signal_gate_round_trip_cost = 0.003            # composite: 往返交易成本估算（佣金+印花税+滑点，仅原始收益模式使用）
$signal_gate_percentile_warmup = 5              # composite: 百分位归一化预热期（调仓次数）
$signal_gate_quality_enabled = $true            # 滚动模型质量监控: $true 启用 | $false 禁用
$signal_gate_quality_window_list = @(3)         # 滚动质量回看调仓周期数
$signal_gate_quality_threshold_list = @(0.4)    # 滚动质量最低 hit rate
$signal_gate_quality_halflife = 4               # 滚动质量 EWM 半衰期

# ── OOS 动态 Top-N（按置信度调整选股数量, 0406引入）─────────────────────────────
$signal_gate_dynamic_topn = $false              # $true 启用 | $false 禁用
$signal_gate_topn_high_multiplier = 0.6         # 高置信度缩减系数（<1，集中持股，如 top_n=17 → 10只）
$signal_gate_topn_low_multiplier  = 1.5         # 中低置信度扩大系数（>1，分散持股，如 top_n=17 → 25只）

# ── OOS 换手率约束（持仓保留奖励, 0407引入）─────────────────────────────────────
$holding_bonus_enabled = $false                 # $true 启用 | $false 禁用（对已持仓股票给予分数加成，降低换手）
$holding_bonus_sigma   = 0.5                    # 奖励幅度（截面分数标准差的倍数，0.3~1.0）

# ── OOS 旧版置信度门控（signal_gate_mode="legacy" 时生效）────────────
$signal_confidence_gate_enabled = $false  # $true 启用 | $false 禁用（仅 legacy 模式）
$signal_confidence_gate_top_k_list = @(20)
$signal_confidence_gate_threshold_sets = @( "0.01 0.02 0.10" )
$signal_confidence_gate_exposure_sets =  @( "0.10 0.99 1.00" )

# ── OOS 止损（关闭时请保持各阈值列表为单值，避免重复任务）─────────
$bt_stop_loss_enabled                 = $true  # $true 启用 | $false 禁用
$bt_stop_loss_drawdown_pct_list       = @(30.0) # 回撤止损阈值（%）
$bt_stop_loss_trailing_enabled        = $false  # $true 启用移动止损 | $false 禁用
$bt_stop_loss_trailing_pct_list       = @(15.0) # 移动止损阈值（%）
$bt_stop_loss_consecutive_limit_down_list = @(2) # 连续跌停止损天数

# ── OOS ECT 权益曲线交易（关闭时请保持相关列表为单值）────────────
$bt_equity_curve_enabled                  = $false  # $true 启用 | $false 禁用
$bt_equity_curve_drawdown_thresholds      = @(5.0, 10.0, 15.0, 20.0)
$bt_equity_curve_exposure_levels          = @(0.8, 0.6, 0.4, 0.2)
$bt_equity_curve_ma_short_list            = @(5)
$bt_equity_curve_ma_long_list             = @(20)
$bt_equity_curve_recovery_mode_list       = @("gradual") # gradual | immediate
$bt_equity_curve_recovery_step_list       = @(0.25)
$bt_equity_curve_recovery_delay_periods_list = @(0)

# ── 行业动量过滤（剔除弱势行业股票，自动补位）──────────────────
$industry_momentum_filter     = $false  # $true 启用 | $false 禁用
$industry_momentum_bottom_pct = 0.5     # 剔除排名后 X% 的行业（0~1），默认 0.2

# ── 行业轮动加权（按行业动量排名对候选分数做乘性调整）─────────────
$industry_rotation_enhanced       = $false     # $true 启用 | $false 禁用（独立于上方硬过滤）
$industry_rotation_alpha_list     = @(0.3)#1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)     # 加权强度（可多值，如 @(0.1, 0.3, 0.5)）

# ── 仓位管理模式（Kelly / 半 Kelly）──────────────────────────────
$position_sizing_list             = @('half_kelly')#, 'score', 'kelly', 'half_kelly') # equal | score | kelly | half_kelly
$kelly_vol_window                 = 60         # Kelly 波动率窗口（交易日）
$kelly_max_leverage_list          = @(0.15)    # Kelly 单股仓位上限（可多值，如 @(0.15, 0.25)）

# ── 市场择时仓位管理 ─────────────────────────────────────────
$market_regime                = $false       # $true 启用 | $false 禁用
$market_regime_mode_list      = @("vol_target")  # binary | vol_target | trend | combined
$market_regime_bear_threshold_list = @(-0.03)   # binary 模式：mkt_ret_avg_20 低于此值判定为熊市
$market_regime_bear_exposure  = 0.3             # binary 模式：熊市仓位系数（0~1）
$market_regime_vol_target_list = @(0.20)   # 默认只围绕均衡区间做更窄扫描
$market_regime_trend_threshold = 1.0            # trend/combined 模式：mkt_ma_trend 低于此值降仓
$market_regime_min_exposure    = 0.2            # 非 binary 模式：最低仓位下限
$market_regime_combine_method  = "min"          # combined 模式组合方式：min | multiply
$market_regime_trend_guard     = $true          # combined 模式趋势保护：上行趋势跳过 vol 降仓
$market_regime_drawdown_guard  = $false         # 回撤保护：已大幅下跌时停止降仓，避免底部踏空
$market_regime_drawdown_threshold = -0.08       # 回撤保护阈值：mkt_drawdown_20 低于此值停止降仓

# ── MA250 长周期硬条件（系统性熊市保护）─────────────────────────
$market_regime_ma250_hard_stop = $true      # $true 启用 | $false 禁用
$market_regime_ma250_threshold = 1          # 触发阈值（大盘收益曲线/MA250 < 此值触发）
$market_regime_ma250_exposure  = 0.9        # 触发后的仓位系数（0.0=完全空仓）
$ma250_atr_scaling             = $true      # $true 启用 ATR 动态仓位缩放（仓位=base×MA(ATR,250)/CurrentATR）

# ── 盈亏动态持仓（提高换仓效率）──────────────────────────────────
$enable_profit_based_holding      = $true       # $true 启用 | $false 禁用
$early_exit_loss_threshold_list   = @(-0.07)    #-0.07 # 亏损提前换出阈值（可多值，如 @(-0.03, -0.05, -0.08)）
$early_exit_holding_ratio_list    = @(0.5)      #0.6  # 亏损提前换出最早触发时点（占持有期比例，可多值，如 @(0.3, 0.5, 0.7)）
$profit_extension_threshold_list  = @(0.1)      #0.1   # 盈利延续持有阈值（pnl模式，可多值，如 @(0.03, 0.05, 0.10)）
$profit_extension_days_list       = @(2)        # baseline 对齐当前最佳防守型 run

# 0411新增strength
# ── 盈利延续判据模式(新) ──
#   pnl=单一浮盈率(兼容原行为) | strength=5维度强势度评分 | disabled=关闭延续
$profit_extension_mode_list              = @('strength')     # 可多值如 @('pnl','strength')
$profit_extension_strength_threshold_list = @(0.75)      # strength 模式延续阈值 [0,1]

# ── ATR 动态阈值与仓位缩放（需先构建含 atr_14 的特征）──────────────
$use_atr_for_early_exit           = $false   # $true 启用 ATR 动态止损阈值（需同时开启 $enable_profit_based_holding）
$atr_multiplier_list              = @(2.8)   # baseline 对齐当前最佳防守型 run（仅启用 ATR 止损时生效）

# 0412新增strength_veto
# ── 亏损提前换出二次确认（strength_veto 门控）──────────────────────
#   disabled=原硬卖(默认) | strength_veto=触发后用强势度评分二次确认,评分高时否决卖出(缓刑)
$early_exit_mode_list                        = @('strength_veto')   # 可多值如 @('disabled','strength_veto')
$early_exit_strength_protect_threshold_list   = @(0.1)        # strength_veto 保护阈值 [0,1]
$early_exit_max_reprieves_list               = @(1)            # 单只股票最多缓刑次数

# ── 整体持仓止盈（整体浮盈达到目标后清仓并补位）──────────────────
$take_profit_threshold_list   = @(0.3)      # 可多值，$null=禁用，如 @($null, 0.15, 0.20)
$take_profit_refill           = $false      # $true=整体止盈后自动补位买入

# ── 空仓/持有期拖尾提前调仓 ────────────────────────────────────
# $true  = 启用：当持仓全部清零或 cycle_day>=holding_period 且仍有残留盈利延续持仓时，
#          尝试提前触发新一轮 T0 流程（拖尾场景下需"残留仓位+新目标仓位<=100%"方可入队）
# $false = 禁用：严格等待下一个预定调仓日
$enable_early_rebalance_on_empty = $true

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

# 预生成门控扫描配置，避免关闭门控时无意义地展开笛卡尔积
$signal_confidence_gate_scan_configs = @(
    [PSCustomObject]@{
        TopK = $null
        Thresholds = $null
        Exposures = $null
    }
)
if ($signal_confidence_gate_enabled) {
    $signal_confidence_gate_scan_configs = @()
    foreach ($gateTopK in $signal_confidence_gate_top_k_list) {
        foreach ($gateThresholds in $signal_confidence_gate_threshold_sets) {
            foreach ($gateExposures in $signal_confidence_gate_exposure_sets) {
                $signal_confidence_gate_scan_configs += [PSCustomObject]@{
                    TopK = $gateTopK
                    Thresholds = $gateThresholds
                    Exposures = $gateExposures
                }
            }
        }
    }
}

$totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$count      = 0
$failed     = 0

# 计算总任务数（各列表长度的笛卡尔积）
$totalTasks = $algorithm_list.Length *
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
              $rank_weight_topk_list.Length *
              $rank_weight_list.Length *
              $market_regime_bear_threshold_list.Length *
              $market_regime_mode_list.Length *
              $market_regime_vol_target_list.Length *
              $bt_top_n_list.Length *
              $signal_gate_cost_multiplier_list.Length *
              $signal_gate_quality_window_list.Length *
              $signal_gate_quality_threshold_list.Length *
              $signal_confidence_gate_scan_configs.Length *
              $bt_sell_timing_list.Length *
              $bt_min_list_days_list.Length *
              $bt_max_weight_per_stock_list.Length *
              $bt_max_per_industry_list.Length *
              $bt_stop_loss_drawdown_pct_list.Length *
              $bt_stop_loss_trailing_pct_list.Length *
              $bt_stop_loss_consecutive_limit_down_list.Length *
              $bt_equity_curve_ma_short_list.Length *
              $bt_equity_curve_ma_long_list.Length *
              $bt_equity_curve_recovery_mode_list.Length *
              $bt_equity_curve_recovery_step_list.Length *
              $bt_equity_curve_recovery_delay_periods_list.Length *
              $stagger_tranches_list.Length *
              $early_exit_loss_threshold_list.Length *
              $early_exit_holding_ratio_list.Length *
              $profit_extension_threshold_list.Length *
              $profit_extension_days_list.Length *
              $profit_extension_mode_list.Length *
              $profit_extension_strength_threshold_list.Length *
              $atr_multiplier_list.Length *
              $early_exit_mode_list.Length *
              $early_exit_strength_protect_threshold_list.Length *
              $early_exit_max_reprieves_list.Length *
              $take_profit_threshold_list.Length *
              $industry_rotation_alpha_list.Length *
              $position_sizing_list.Length *
              $kelly_max_leverage_list.Length

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
foreach ($rank_weight_topk in $rank_weight_topk_list) {
foreach ($rank_weight in $rank_weight_list) {
foreach ($market_regime_bear_threshold in $market_regime_bear_threshold_list) {
foreach ($market_regime_mode in $market_regime_mode_list) {
foreach ($market_regime_vol_target in $market_regime_vol_target_list) {
foreach ($bt_top_n in $bt_top_n_list) {
foreach ($signal_gate_cost_multiplier in $signal_gate_cost_multiplier_list) {
foreach ($signal_gate_quality_window in $signal_gate_quality_window_list) {
foreach ($signal_gate_quality_threshold in $signal_gate_quality_threshold_list) {
foreach ($signal_confidence_gate_config in $signal_confidence_gate_scan_configs) {
foreach ($bt_sell_timing in $bt_sell_timing_list) {
foreach ($bt_min_list_days in $bt_min_list_days_list) {
foreach ($bt_max_weight_per_stock in $bt_max_weight_per_stock_list) {
foreach ($bt_max_per_industry in $bt_max_per_industry_list) {
foreach ($bt_stop_loss_drawdown_pct in $bt_stop_loss_drawdown_pct_list) {
foreach ($bt_stop_loss_trailing_pct in $bt_stop_loss_trailing_pct_list) {
foreach ($bt_stop_loss_consecutive_limit_down in $bt_stop_loss_consecutive_limit_down_list) {
foreach ($bt_equity_curve_ma_short in $bt_equity_curve_ma_short_list) {
foreach ($bt_equity_curve_ma_long in $bt_equity_curve_ma_long_list) {
foreach ($bt_equity_curve_recovery_mode in $bt_equity_curve_recovery_mode_list) {
foreach ($bt_equity_curve_recovery_step in $bt_equity_curve_recovery_step_list) {
foreach ($bt_equity_curve_recovery_delay_periods in $bt_equity_curve_recovery_delay_periods_list) {
foreach ($stagger_tranches in $stagger_tranches_list) {
foreach ($early_exit_loss_threshold in $early_exit_loss_threshold_list) {
foreach ($early_exit_holding_ratio in $early_exit_holding_ratio_list) {
foreach ($profit_extension_threshold in $profit_extension_threshold_list) {
foreach ($profit_extension_days in $profit_extension_days_list) {
foreach ($profit_extension_mode in $profit_extension_mode_list) {
foreach ($profit_extension_strength_threshold in $profit_extension_strength_threshold_list) {
foreach ($atr_multiplier in $atr_multiplier_list) {
foreach ($early_exit_mode in $early_exit_mode_list) {
foreach ($early_exit_strength_protect_threshold in $early_exit_strength_protect_threshold_list) {
foreach ($early_exit_max_reprieves in $early_exit_max_reprieves_list) {
foreach ($take_profit_threshold in $take_profit_threshold_list) {
foreach ($industry_rotation_alpha in $industry_rotation_alpha_list) {
foreach ($position_sizing in $position_sizing_list) {
foreach ($kelly_max_leverage in $kelly_max_leverage_list) {

    $count++

    # 构建命令字符串
    $pythonCmd = "py .\scripts\walk_forward.py" +
                 " --algorithm $algorithm" +
                 " --wf-start-date $wf_start_date" +
                 " --wf-end-date $wf_end_date" +
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

    if ($enable_enhanced) {
        $pythonCmd += " --enable-enhanced-features"
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

    if ($market_regime_ma250_hard_stop) {
        $pythonCmd += " --market-regime-ma250-hard-stop" +
                      " --market-regime-ma250-threshold $market_regime_ma250_threshold" +
                      " --market-regime-ma250-exposure $market_regime_ma250_exposure"
        if ($ma250_atr_scaling) {
            $pythonCmd += " --ma250-atr-scaling"
        }
    }

    if ($enable_profit_based_holding) {
        $pythonCmd += " --enable-profit-based-holding" +
                      " --early-exit-loss-threshold $early_exit_loss_threshold" +
                      " --early-exit-holding-ratio $early_exit_holding_ratio" +
                      " --profit-extension-threshold $profit_extension_threshold" +
                      " --profit-extension-days $profit_extension_days" +
                      " --profit-extension-mode $profit_extension_mode" +
                      " --profit-extension-strength-threshold $profit_extension_strength_threshold"
    }

    if ($use_atr_for_early_exit) {
        $pythonCmd += " --use-atr-for-early-exit --atr-multiplier $atr_multiplier"
    }

    if ($early_exit_mode -ne 'disabled') {
        $pythonCmd += " --early-exit-mode $early_exit_mode" +
                      " --early-exit-strength-protect-threshold $early_exit_strength_protect_threshold" +
                      " --early-exit-max-reprieves $early_exit_max_reprieves"
    }

    if ($null -ne $take_profit_threshold) {
        $pythonCmd += " --take-profit-threshold $take_profit_threshold"
        if (-not $take_profit_refill) {
            $pythonCmd += " --no-take-profit-refill"
        }
    }

    if (-not $enable_early_rebalance_on_empty) {
        $pythonCmd += " --no-early-rebalance-on-empty"
    }

    if ($stagger_tranches -gt 1) {
        $pythonCmd += " --stagger-tranches $stagger_tranches"
    }

    if ($oos_backtest) {
        $pythonCmd += " --oos-backtest --oos-backtest-months $oos_backtest_months --bt-top-n $bt_top_n --bt-weight-method $bt_weight_method --bt-initial-capital $bt_initial_capital --bt-sell-timing $bt_sell_timing --bt-min-list-days $bt_min_list_days"
        # 信号入口门控 v2
        $pythonCmd += " --signal-gate-mode $signal_gate_mode"
        if ($signal_gate_mode -eq "composite") {
            $pythonCmd += " --signal-gate-cost-multiplier $signal_gate_cost_multiplier" +
                          " --signal-gate-round-trip-cost $signal_gate_round_trip_cost" +
                          " --signal-gate-percentile-warmup $signal_gate_percentile_warmup"
        }
        if ($signal_gate_mode -eq "legacy" -and $signal_confidence_gate_enabled) {
            $pythonCmd += " --signal-confidence-gate-enabled" +
                          " --signal-confidence-gate-top-k $($signal_confidence_gate_config.TopK)" +
                          " --signal-confidence-gate-thresholds $($signal_confidence_gate_config.Thresholds)" +
                          " --signal-confidence-gate-exposure-levels $($signal_confidence_gate_config.Exposures)"
        }
        if ($signal_gate_quality_enabled) {
            $pythonCmd += " --signal-gate-quality-enabled" +
                          " --signal-gate-quality-window $signal_gate_quality_window" +
                          " --signal-gate-quality-threshold $signal_gate_quality_threshold" +
                          " --signal-gate-quality-halflife $signal_gate_quality_halflife"
        }
        if ($signal_gate_dynamic_topn) {
            $pythonCmd += " --signal-gate-dynamic-topn" +
                          " --signal-gate-topn-high-multiplier $signal_gate_topn_high_multiplier" +
                          " --signal-gate-topn-low-multiplier $signal_gate_topn_low_multiplier"
        }
        if ($holding_bonus_enabled) {
            $pythonCmd += " --holding-bonus-enabled" +
                          " --holding-bonus-sigma $holding_bonus_sigma"
        }
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
            if ($bt_stop_loss_trailing_enabled) {
                $pythonCmd += " --bt-stop-loss-trailing-enabled --bt-stop-loss-trailing-pct $bt_stop_loss_trailing_pct"
            }
        }
        if ($bt_equity_curve_enabled) {
            $btEquityCurveDrawdownArgs = $bt_equity_curve_drawdown_thresholds -join " "
            $btEquityCurveExposureArgs = $bt_equity_curve_exposure_levels -join " "
            $pythonCmd += " --bt-equity-curve-enabled" +
                          " --bt-equity-curve-drawdown-thresholds $btEquityCurveDrawdownArgs" +
                          " --bt-equity-curve-exposure-levels $btEquityCurveExposureArgs" +
                          " --bt-equity-curve-ma-short $bt_equity_curve_ma_short" +
                          " --bt-equity-curve-ma-long $bt_equity_curve_ma_long" +
                          " --bt-equity-curve-recovery-mode $bt_equity_curve_recovery_mode" +
                          " --bt-equity-curve-recovery-step $bt_equity_curve_recovery_step" +
                          " --bt-equity-curve-recovery-delay-periods $bt_equity_curve_recovery_delay_periods"
        }
    } else {
        $pythonCmd += " --no-oos-backtest"
    }

    if ($industry_momentum_filter) {
        $pythonCmd += " --industry-momentum-filter --industry-momentum-bottom-pct $industry_momentum_bottom_pct"
    }

    if ($industry_rotation_enhanced) {
        $pythonCmd += " --industry-rotation-enhanced --industry-rotation-alpha $industry_rotation_alpha"
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

}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  # end foreach（参数组合循环）

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

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
$step_list               = @("semiannual")   # monthly | quarterly | semiannual
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
$learning_rate_list      = @(0.03, 0.05)      #0.009. XGB推荐0.005, LGB推荐0.005
$min_child_weight_list   = @(250)       #. XGB推荐150, LGB推荐200
$colsample_bytree_list   = @(0.3)       #. XGB/LGB均推荐0.3

$subsample_list          = @(0.8)       #. XGB推荐0.8, LGB推荐0.7
$reg_alpha_list          = @(0.05)      #. XGB推荐0.05, LGB推荐0.1
$reg_lambda_list         = @(7)         #. XGB推荐5.0, LGB推荐5.0
$gamma_list              = @(0.5)       #. 映射LGB min_split_gain。XGB推荐0.5, LGB推荐1.0
$num_leaves_list         = @(63)        #  仅LightGBM有效，XGBoost忽略。LGB推荐63

# ── 目标函数 ─────────────────────────────────────────────────
$objective_list          = @("mse")  # mse | lambdarank（排序学习，直接优化股票排序）
# ── 早停配置 ───────────────────────────────────────────────────
$early_stopping_rounds_list = @(500)    # 早停轮数，设为 0 则禁用早停（固定 n_estimators 棵树），可多值扫描如 @(100, 300, 500)
$early_stopping_metric   = "rank_ic"       # 早停指标：auto（mae/auc）| rank_ic（Spearman，尺度无关更稳定）


# ── rank-weight 配置（固定，不参与组合扫描）─────────────────────
$rank_weight_enabled     = $true   # $true 启用 | $false 禁用
$rank_weight_topk_list   = @(200)         #50
$rank_weight_list        = @(15)         #3
$rank_weight_topk_weight_mode = "linear_decay"   # linear_decay | flat

# ── 时间衰减权重 ──────────────────────────────────────────────
$time_decay_half_life_list = @(0)      # 半衰期（年）。0=禁用，1.0=1年前权重0.5，2.0=2年前权重0.5

# ── best_iteration 自适应候选重训 ─────────────────────────────
$adaptive_best_iter_retrain = $false  # $true 启用：低迭代/撞上限 split 自动重训候选，并按 Top30中位数(70%)+RankIC IR(30%) 加权打分择优
$adaptive_low_iter_max_retries = 1  # low_iter（best_iter<=50）随机种子重试上限

# ── 多种子 bagging（每个split用多个随机种子各训一个子模型取平均，降训练随机方差）─
$ensemble_seeds            = "1100"#,2200"#,3300"#,4400,5500,6600,7700,8800,9900,10000"#,220,719"     # 逗号分隔种子如 "42,1,2,3,4"；空=单种子（用 --random-state），与多偏移可叠加
$ensemble_seed_keep_top_ratio = 0.50  # 多种子筛选保留比例（0~1）
$ensemble_seed_keep_min_models = 1    # 多种子筛选最少保留模型数 

# ── 候选树数后验选优（训练完成后按逐日验证指标重选最终树数）─────────
$posterior_tree_selection_mode = "disabled"   # disabled | grid
$posterior_tree_candidates     = ""           # 逗号分隔，如 "16,32,64,128"；空=使用内置网格
$posterior_tree_selection_metric = "topk_median"  # topk_median | topk_mean | topk_lift | topk_hit_rate | topk_excess_hit_rate | rankic_ir | rankic_mean
$posterior_tree_selection_topk   = 20             # 后验选优使用的 TopK，默认20（可改30切回旧口径）

###  以下为因子选择
# ── 基本面因子（需先运行 download_raw.py --download fina_indicator）───
$enable_fundamental      = $true  # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤增加约2%

# ── 另类数据因子（股东人数、业绩预告等）(0310添加)──────────────────
$enable_alt              = $true  # $true 启用 | $false 禁用

# ── 融资融券因子（通过 margin_detail 接口下载）────────────────────
$enable_margin           = $true  # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤基本不变 
# 0610:打开后CAGR大幅下降

# ── 筹码胜率因子（需5000+积分，需先下载 cyq_perf）─────────────────
$enable_cyq              = $true  # $true 启用 | $false 禁用
# 0428:关闭后CAGR下降约3%, 回撤基本不变

# ── 基金持仓因子（需5000+积分，需先下载 fund_portfolio）──────────
$enable_fund             = $false  # $true 启用 | $false 禁用
#改为false似乎可以提升少量收益并减少少量回撤, 并提升稳定效果

# ── 业绩快报因子（需5000+积分，需先下载 express）─────────────────
$enable_express          = $true  # $true 启用 | $false 禁用

# ── 北向资金因子（moneyflow_hsgt 市场级广播, 2000+积分）───────────
$enable_north            = $false  # $true 启用 | $false 禁用
#实测:打开后CAGR下降约6%, 回撤上升8%

# ── 龙虎榜因子（top_list 个股级, 2000+积分）──────────────────────
$enable_lhb              = $true  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约4%, 回撤提升约5%

# ── 一致预期因子（report_rc 研报滚动聚合, 8000积分）──────────────
$enable_consensus        = $true  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约2%, 回撤无明显变化

# ── 一致预期修正因子（0512基于已有 report_rc 构建时序修正信号，无需额外下载）─
$enable_consensus_revision = $false  # $true 启用 | $false 禁用（实验性因子）

# ── 现金流质量因子（0512需 cashflow 接口，2000 积分，需先下载 cashflow 数据）─
$enable_cashflow_quality   = $false  # $true 启用 | $false 禁用（实验性因子）

### 以下为训练功能选择
# ── 特征稳定性筛选（移除跨时期IC方向不一致的特征, 0326引入）──────────────
$feature_stability_filter = $false  # $true 启用 | $false 禁用（实验验证效果不佳）

# ── 因子精简（基于IC分析排除低效因子, 0606引入）────────────────────────
$factor_prune             = $false  # $true 启用 | $false 禁用（需先运行 generate_factor_exclude_list.py）

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

# ── OOS 信号入口门控 v2（替代旧置信度门控，0406引入）────────────
# 0426这里应为composite
$signal_gate_mode_list = @("disabled")         # "legacy" 旧公式 | "composite" 新公式(成本+百分位) | "disabled" 关闭（可多值扫描）
# 以下 3 个参数仅在 $signal_gate_mode = "composite" 时生效
# 0601 回撤降低4%, 收益降低4%
$signal_gate_cost_multiplier_list = @(0.8)      #0.3 composite: 门控严格度扫描
$signal_gate_round_trip_cost = 0.003            # composite: 往返交易成本估算（佣金+印花税+滑点，仅原始收益模式使用）
$signal_gate_percentile_warmup = 3              # composite: 百分位归一化预热期（调仓次数）

# 滚动模型质量监控子开关：仅在开启时才使用以下质量参数
# 0601:没用
$signal_gate_quality_enabled = $false            # $true 启用 | $false 禁用
$signal_gate_quality_window_list = @(2)         #2 滚动质量回看调仓周期数
$signal_gate_quality_threshold_list = @(0.2)    #0.5 滚动质量最低 hit rate
$signal_gate_quality_halflife = 3               #4 滚动质量 EWM 半衰期

# 动态 Top-N 子开关：仅在开启时按置信度缩放持仓数量
# 0601: 降低回撤2%, 降低收益3%
$signal_gate_dynamic_topn = $false              # $true 启用 | $false 禁用
$signal_gate_topn_high_multiplier = 0.8         # 高置信度缩减系数（<1，集中持股，如 top_n=17 → 10只）
$signal_gate_topn_low_multiplier  = 1.2         # 中低置信度扩大系数（>1，分散持股，如 top_n=17 → 25只）

# 持仓保留奖励子开关：仅在开启时对已持仓股票给予分数加成，降低换手
$holding_bonus_enabled = $false                 # $true 启用 | $false 禁用
$holding_bonus_sigma   = 0.25                    # 奖励幅度（截面分数标准差的倍数，0.3~1.0）

# 旧版置信度门控子开关：仅在 $signal_gate_mode = "legacy" 且开关为 $true 时生效
$signal_confidence_gate_enabled = $false
$signal_confidence_gate_top_k_list = @(20)
$signal_confidence_gate_threshold_sets = @( "0.01 0.02 0.10" )
$signal_confidence_gate_exposure_sets =  @( "0.10 0.99 1.00" )

# ── 盈亏动态持仓（总开关，控制提前换出与到期延续）────────────────
# 0601这里应为true
$enable_profit_based_holding_list = @($false)    # $true 启用 | $false 禁用（可多值扫描，如 @($false, $true)）
# 以下所有参数仅在 $enable_profit_based_holding = $true 时生效

# 1) 亏损提前换出基础阈值：持有达到 early_exit_holding_ratio × 持有期后，
#    若收益率 <= early_exit_loss_threshold，则提前换出
#    注意：这两个阈值在 $enable_profit_based_holding = $true 时始终生效，
#    不受 $early_exit_mode_list = @('disabled') 影响；disabled 表示原硬卖。
$early_exit_holding_ratio_list    = @(0.8)        # 最早触发时点（占持有期比例，可多值，如 @(0.3, 0.5, 0.7)）
$early_exit_loss_threshold_list   = @(-0.15)    # 亏损提前换出阈值（可多值，如 @(-0.03, -0.05, -0.08)）
# 0429这个似乎不开比较好, early_exit_holding_ratio设置为1就是关闭

# 2) 亏损提前换出二次确认：仅在 early_exit 条件已触发后再做一次强势度否决
#    disabled      = 原硬卖
#    strength_veto = 评分高于保护阈值时否决卖出（缓刑）
$early_exit_mode_list                        = @('disabled')   # 可多值如 @('disabled', 'strength_veto')
# 仅在 $early_exit_mode_list 扫到 'strength_veto' 时生效
$early_exit_strength_protect_threshold_list   = @(0.1)         # strength_veto 保护阈值 [0,1]
$early_exit_max_reprieves_list               = @(3)            # 单只股票最多缓刑次数

# 3) ATR 动态亏损阈值：在亏损提前换出分支中，用 ATR 替代固定亏损阈值
#    需同时满足 $enable_profit_based_holding = $true 且 $use_atr_for_early_exit = $true
$use_atr_for_early_exit           = $false   # $true 启用 | $false 禁用
$atr_multiplier_list              = @(2.5)   # ATR 倍数（仅启用 ATR 止损时生效）
# 0430效果都不好, 收益和回测都变差

# 4) 时间止损：持仓超过指定天数后仍未达到最低盈利要求，视为"死钱"触发提前换出0512
#    需同时满足 $enable_profit_based_holding = $true 且 $time_stop_loss_enabled = $true
$time_stop_loss_enabled        = $false   # $true 启用 | $false 禁用
# 以下参数仅在 $time_stop_loss_enabled = $true 时生效
$time_stop_loss_days_list      = @(5)    # 触发时间止损的最低持有天数（交易日）
$time_stop_loss_profit_ratio_list = @(-0.02) # 时间止损利润阈值（当前盈亏低于此值触发，如-0.02=-2%）

# 5) 盈利延续持有：仅在持有期满后进入该分支
#    pnl      = 浮盈率判据（兼容原行为）
#    strength = 5 维强势度评分判据
#    disabled = 持有期满直接卖出，不做延续
$profit_extension_mode_list = @('strength')     # 可多值如 @('pnl', 'strength', 'disabled')
# 在 $profit_extension_mode_list 扫到 "pnl" 或 "strength" 时生效
$profit_extension_days_list       = @(20)        # 额外延续天数（交易日）

# 仅在 $profit_extension_mode_list 扫到 "pnl" 时生效
$profit_extension_threshold_list  = @(0.04)      # 盈利延续阈值（浮盈率，可多值，如 @(0.03, 0.05, 0.10)）
# 0430 0.04/20:43%(33%)年化,-31%(-31)回撤

# 仅在 $profit_extension_mode_list 扫到 "strength" 时生效
$profit_extension_strength_threshold_list = @(0.56) # strength 模式延续阈值 [0,1]
# 0503 0.56/20:49%(38%),-25%(-28%); 0.6/20:45%(29%)年化,-28%(-32%)回撤

# ── 整体持仓止盈（独立于 $enable_profit_based_holding）──────────────
# 0426这里应为0.15
$take_profit_threshold_list   = @($null)      # 可多值；$null = 禁用，如 @($null, 0.15, 0.20)
# 仅在 $take_profit_threshold_list 不为 $null 时，$take_profit_refill 才有意义
$take_profit_refill           = $false       # $true = 整体止盈后自动补位买入

# ── 空仓/持有期拖尾提前调仓（独立开关，常与盈利延续/整体止盈联动）────
# $true  = 启用：当持仓全部清零或 cycle_day >= holding_period 且仍有残留盈利延续持仓时，
#          尝试提前触发新一轮 T0 流程（拖尾场景下需"残留仓位+新目标仓位<=100%"方可入队）
# $false = 禁用：严格等待下一个预定调仓日
# 0426这里应为true
$enable_early_rebalance_on_empty_list = @($true)  # 可多值如 @($false, $true)

# ── MA250 长周期硬条件（系统性熊市保护）─────────────────────────
$market_regime_ma250_hard_stop = $false      # $true 启用 | $false 禁用
# 以下参数仅在 $market_regime_ma250_hard_stop = $true 时生效
$market_regime_ma250_threshold_list = @(1)   # 触发阈值（大盘收益曲线 / MA250 < 此值触发）
$market_regime_ma250_exposure_list  = @(0.9) # 触发后的仓位系数（0.0 = 完全空仓）
$ma250_atr_scaling             = $false       # $true 启用 ATR 动态仓位缩放（仓位 = base × MA(ATR,250) / CurrentATR）

# ── OOS 止损（总开关）────────────────────────────────────────
# 0426这里应为true
$bt_stop_loss_enabled                 = $false   # $true 启用 | $false 禁用
# 以下参数仅在 $bt_stop_loss_enabled = $true 时生效
$bt_stop_loss_drawdown_pct_list       = @(20) # 回撤止损阈值（%）
$bt_stop_loss_consecutive_limit_down_list = @(2) # 连续跌停止损天数
$bt_stop_loss_trailing_enabled        = $false  # 移动止损子开关：$true 启用 | $false 禁用
# 仅在 $bt_stop_loss_enabled = $true 且 $bt_stop_loss_trailing_enabled = $true 时生效
$bt_stop_loss_trailing_pct_list       = @(15.0) # 移动止损阈值（%）

# ── OOS 表现弱势退出（总开关）───────────────────────────────
$bt_weakness_exit_enabled                = $false  # $true 启用 | $false 禁用
# 以下参数仅在 $bt_weakness_exit_enabled = $true 时生效
$bt_weakness_exit_threshold_list         = @(0.52)     # 弱势评分触发阈值 [0,1]
$bt_weakness_exit_consecutive_days_list  = @(5)       #5 需连续弱势天数
$bt_weakness_exit_min_holding_days_list  = @(4)       #4 最低持有天数
#$bt_weakness_exit_weights                = "30,25,25,20"  # 4维度权重: 排名,连跌,回撤,回升
$bt_weakness_exit_weights                = "30,25,25,20"  # 4维度权重: 排名,连跌,回撤,回升
$bt_weakness_exit_industry_filter        = $false     # 弱势行业过滤
$bt_weakness_exit_industry_bottom_pct_list = @(0.3)   # 行业底部阈值

# ── OOS ECT 权益曲线交易（总开关）────────────────────────────
$bt_equity_curve_enabled                  = $false  # $true 启用 | $false 禁用
# 以下参数仅在 $bt_equity_curve_enabled = $true 时生效
$bt_equity_curve_drawdown_thresholds      = @(5.0, 10.0, 15.0, 20.0)
$bt_equity_curve_exposure_levels          = @(0.8, 0.6, 0.4, 0.2)
$bt_equity_curve_ma_short_list            = @(5)
$bt_equity_curve_ma_long_list             = @(20)
$bt_equity_curve_recovery_mode_list       = @("gradual") # gradual | immediate
$bt_equity_curve_recovery_step_list       = @(0.25)
$bt_equity_curve_recovery_delay_periods_list = @(0)

# ── 行业动量过滤（总开关）────────────────────────────────────
$industry_momentum_filter     = $false  # $true 启用 | $false 禁用
# 仅在 $industry_momentum_filter = $true 时生效
$industry_momentum_bottom_pct = 0.5     # 剔除排名后 X% 的行业（0~1），默认 0.2

# ── 行业轮动加权（总开关）────────────────────────────────────
$industry_rotation_enhanced       = $false     # $true 启用 | $false 禁用（独立于上方硬过滤）
# 仅在 $industry_rotation_enhanced = $true 时生效
$industry_rotation_alpha_list     = @(0.3)#1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)     # 加权强度（可多值，如 @(0.1, 0.3, 0.5)）

# ── 市场择时仓位管理（总开关）────────────────────────────────
$market_regime                = $false            # $true 启用 | $false 禁用
# 以下参数仅在 $market_regime = $true 时生效
$market_regime_mode_list      = @("vol_target")  # binary | vol_target | trend | combined
# 仅在 mode = binary 时，bear_threshold / bear_exposure 真正决定熊市降仓
$market_regime_bear_threshold_list = @(-0.03)    # mkt_ret_avg_20 低于此值判定为熊市
$market_regime_bear_exposure  = 0.3              # binary 模式熊市仓位系数（0~1）
# vol_target / combined 模式会使用 vol_target；trend / combined 模式会使用 trend_threshold
$market_regime_vol_target_list = @(0.20)         # 默认只围绕均衡区间做更窄扫描
$market_regime_trend_threshold = 1.0             # trend / combined 模式：mkt_ma_trend 低于此值降仓
$market_regime_min_exposure    = 0.2             # 非 binary 模式：最低仓位下限
$market_regime_combine_method  = "min"          # combined 模式组合方式：min | multiply
$market_regime_trend_guard     = $true           # combined 模式趋势保护：上行趋势跳过 vol 降仓
$market_regime_drawdown_guard  = $false          # 回撤保护子开关：已大幅下跌时停止降仓，避免底部踏空
# 仅在 $market_regime_drawdown_guard = $true 时生效
$market_regime_drawdown_threshold = -0.08        # mkt_drawdown_20 低于此值停止降仓

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
              $market_regime_bear_threshold_list.Length *
              $market_regime_mode_list.Length *
              $market_regime_vol_target_list.Length *
              $market_regime_ma250_threshold_list.Length *
              $market_regime_ma250_exposure_list.Length *
              $bt_top_n_list.Length *
              $bt_rebalance_freq_list.Length *
              $signal_gate_mode_list.Length *
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
              $bt_weakness_exit_threshold_list.Length *
              $bt_weakness_exit_consecutive_days_list.Length *
              $bt_weakness_exit_min_holding_days_list.Length *
              $bt_weakness_exit_industry_bottom_pct_list.Length *
              $bt_equity_curve_ma_short_list.Length *
              $bt_equity_curve_ma_long_list.Length *
              $bt_equity_curve_recovery_mode_list.Length *
              $bt_equity_curve_recovery_step_list.Length *
              $bt_equity_curve_recovery_delay_periods_list.Length *
              $stagger_tranches_list.Length *
              $enable_profit_based_holding_list.Length *
              $early_exit_loss_threshold_list.Length *
              $early_exit_holding_ratio_list.Length *
              $profit_extension_threshold_list.Length *
              $profit_extension_days_list.Length *
              $profit_extension_mode_list.Length *
              $profit_extension_strength_threshold_list.Length *
              $atr_multiplier_list.Length *
              $time_stop_loss_days_list.Length *
              $time_stop_loss_profit_ratio_list.Length *
              $early_exit_mode_list.Length *
              $early_exit_strength_protect_threshold_list.Length *
              $early_exit_max_reprieves_list.Length *
              $take_profit_threshold_list.Length *
              $enable_early_rebalance_on_empty_list.Length *
              $industry_rotation_alpha_list.Length *
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
foreach ($market_regime_bear_threshold in $market_regime_bear_threshold_list) {
foreach ($market_regime_mode in $market_regime_mode_list) {
foreach ($market_regime_vol_target in $market_regime_vol_target_list) {
foreach ($market_regime_ma250_threshold in $market_regime_ma250_threshold_list) {
foreach ($market_regime_ma250_exposure in $market_regime_ma250_exposure_list) {
foreach ($bt_top_n in $bt_top_n_list) {
foreach ($bt_rebalance_freq in $bt_rebalance_freq_list) {
foreach ($signal_gate_mode in $signal_gate_mode_list) {
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
foreach ($bt_weakness_exit_threshold in $bt_weakness_exit_threshold_list) {
foreach ($bt_weakness_exit_consecutive_days in $bt_weakness_exit_consecutive_days_list) {
foreach ($bt_weakness_exit_min_holding_days in $bt_weakness_exit_min_holding_days_list) {
foreach ($bt_weakness_exit_industry_bottom_pct in $bt_weakness_exit_industry_bottom_pct_list) {
foreach ($bt_equity_curve_ma_short in $bt_equity_curve_ma_short_list) {
foreach ($bt_equity_curve_ma_long in $bt_equity_curve_ma_long_list) {
foreach ($bt_equity_curve_recovery_mode in $bt_equity_curve_recovery_mode_list) {
foreach ($bt_equity_curve_recovery_step in $bt_equity_curve_recovery_step_list) {
foreach ($bt_equity_curve_recovery_delay_periods in $bt_equity_curve_recovery_delay_periods_list) {
foreach ($stagger_tranches in $stagger_tranches_list) {
foreach ($enable_profit_based_holding in $enable_profit_based_holding_list) {
foreach ($early_exit_loss_threshold in $early_exit_loss_threshold_list) {
foreach ($early_exit_holding_ratio in $early_exit_holding_ratio_list) {
foreach ($profit_extension_threshold in $profit_extension_threshold_list) {
foreach ($profit_extension_days in $profit_extension_days_list) {
foreach ($profit_extension_mode in $profit_extension_mode_list) {
foreach ($profit_extension_strength_threshold in $profit_extension_strength_threshold_list) {
foreach ($atr_multiplier in $atr_multiplier_list) {
foreach ($time_stop_loss_days in $time_stop_loss_days_list) {
foreach ($time_stop_loss_profit_ratio in $time_stop_loss_profit_ratio_list) {
foreach ($early_exit_mode in $early_exit_mode_list) {
foreach ($early_exit_strength_protect_threshold in $early_exit_strength_protect_threshold_list) {
foreach ($early_exit_max_reprieves in $early_exit_max_reprieves_list) {
foreach ($take_profit_threshold in $take_profit_threshold_list) {
foreach ($enable_early_rebalance_on_empty in $enable_early_rebalance_on_empty_list) {
foreach ($industry_rotation_alpha in $industry_rotation_alpha_list) {
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
                 " --posterior-tree-selection-mode $posterior_tree_selection_mode" +
                 " --posterior-tree-selection-metric $posterior_tree_selection_metric" +
                 " --posterior-tree-selection-topk $posterior_tree_selection_topk" +
                 " --time-decay-half-life $time_decay_half_life" +
                 " --batch-run-id $batch_run_id" +
                 " --batch-period-label $batch_period_label" +
                 " --wf-summary-csv `"$summary_csv_path`""

    if ($posterior_tree_candidates -ne "") {
        $pythonCmd += " --posterior-tree-candidates $posterior_tree_candidates"
    }

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
                      " --early-exit-mode $early_exit_mode" +
                      " --profit-extension-threshold $profit_extension_threshold" +
                      " --profit-extension-days $profit_extension_days" +
                      " --profit-extension-mode $profit_extension_mode" +
                      " --profit-extension-strength-threshold $profit_extension_strength_threshold"
    }

    if ($use_atr_for_early_exit) {
        $pythonCmd += " --use-atr-for-early-exit --atr-multiplier $atr_multiplier"
    }

    if ($time_stop_loss_enabled) {
        $pythonCmd += " --time-stop-loss-enabled" +
                      " --time-stop-loss-days $time_stop_loss_days" +
                      " --time-stop-loss-profit-ratio $time_stop_loss_profit_ratio"
    } else {
        $pythonCmd += " --no-time-stop-loss"
    }

    if ($enable_profit_based_holding -and $early_exit_mode -ne 'disabled') {
        $pythonCmd += " --early-exit-strength-protect-threshold $early_exit_strength_protect_threshold" +
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
        $pythonCmd += " --oos-backtest --oos-backtest-months $oos_backtest_months --bt-top-n $bt_top_n --bt-initial-capital $bt_initial_capital --bt-sell-timing $bt_sell_timing --bt-min-list-days $bt_min_list_days"
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
        if ($bt_weakness_exit_enabled) {
            $pythonCmd += " --bt-weakness-exit-enabled" +
                          " --bt-weakness-exit-threshold $bt_weakness_exit_threshold" +
                          " --bt-weakness-exit-consecutive-days $bt_weakness_exit_consecutive_days" +
                          " --bt-weakness-exit-min-holding-days $bt_weakness_exit_min_holding_days" +
                          " --bt-weakness-exit-weights $bt_weakness_exit_weights"
            if ($bt_weakness_exit_industry_filter) {
                $pythonCmd += " --bt-weakness-exit-industry-filter" +
                              " --bt-weakness-exit-industry-bottom-pct $bt_weakness_exit_industry_bottom_pct"
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

}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  # end foreach（时间段+参数组合循环）

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

# batch_walk_forward.ps1
# Walk-forward 批量参数扫描脚本
#
# 用法：
#   在 "参数配置区" 里，把想对比的参数设置为多个值（数组），固定参数保留单个值。
#   脚本会自动遍历所有参数组合，每组调用一次 walk_forward.py，
#   全部完成后自动运行 compare_walk_forward.py 生成对比 Excel。
#
# 示例启动：
#   powershell -ExecutionPolicy Bypass -File .\scripts\batch\batch_walk_forward.ps1
# 暴露政策（默认启用，配在下方“参数配置区”的 $exposure_arm_list）：
#   默认臂 e2online_r = E2-滚动250、λ=0.5、在线现算、**对称回补开**、不裁剪覆盖
#   （2026-09-20 转正：回补默认开 + 覆盖起点 20181126，登记见 docs/terminal_loss_risk_register.md R-007 §6/§7）；
#   · 命令行一键切回基线（无政策）：-NoExposure
#   · 命令行临时换策略：-ExposurePolicy "arm=combined,..." -SkipCompare
#     （注意：命令行给任一暴露参数会**整体替换**臂清单 ⇒ 如需保留回补请同时加 -ExposureReplenish）
#   · 跳过 OOS00 弱折（R-006）的覆盖起点：-PolicyCoverageStart 20190527
#   · 与 R-004 §9 同覆盖：-PolicyCoverageStart 20240102
#   · 折模型源 = 与选股 OOS 对齐的 14 折集（2026-09-20 起默认）

param(
    # 本块全部是【命令行覆盖参数】：留空 = 用下方“参数配置区”的默认值（生产取值一律在配置区，不在这里）
    # 暴露门控系数表（两列 CSV：日期, 暴露系数）；留空 = 不用表（是否启用政策由配置区臂清单决定）
    [string]$ExposureTable = "",
    # 在线暴露政策（P2-5，`k=v,k=v`，如 arm=combined,mode=rolling,window=250,regime_q=0.6667,score_q=0.5,lambda=0.5）
    # 与 -ExposureTable 互斥；给出后 λ_t 随持仓逐日现算（不读预导出表）
    [string]$ExposurePolicy = "",
    # terminal_loss 折模型根目录（在线政策必需；留空 = 用配置区默认）
    [string]$PolicyModelRoot = "",
    # 折目录后缀（在线政策必需；如 _d5_v6m_fscore）
    [string]$PolicyArmSuffix = "",
    # 政策覆盖起点（YYYYMMDD；留空 = 用配置区默认 "20181126" = 不裁剪全 14 折）
    [string]$PolicyCoverageStart = "",
    # 暴露门控对称回补（P2-4；必须与 -ExposurePolicy 或 -ExposureTable 同传，否则直接报错）
    # 默认臂 e2online_r 已开启回补；本参数仅在“命令行覆盖臂清单”时需要显式给出
    [switch]$ExposureReplenish,
    # 建仓折扣回补（A3 v2；必须与 -ExposureReplenish 同用）：λ<1 信号日的买入预算折扣
    # 按实际成交额入回补释放额，λ 恢复后由既有回补机制买回（默认关，逐位一致）
    [switch]$ExposureBudgetDiscount,
    # 减仓/回补共用容差（组合总值比例，如 0.06）；<=0 = 用引擎默认 3%
    [double]$TrimTolerance = 0,
    # 回撤侧对照：启用止损（与暴露门控合并成一次回撤侧总扫描）
    [switch]$StopLoss,
    # 止损回撤阈值（%，仅在 -StopLoss 时生效）
    [int]$StopLossDrawdownPct = 20,
    # 一键切回基线臂（不启用暴露政策/系数表；与暴露类参数互斥）
    [switch]$NoExposure,
    # 影子多臂实验省时开关：加上 -SkipCompare 则不跑运行后自动汇总对比
    [switch]$SkipCompare
)
# ============================================================
#  参数配置区（修改这里控制实验组合）
# ============================================================

# ── 跳过训练，仅调参回测（复用已有模型）──────────────────────
# 使用场景：模型已训练完毕，只想调整回测参数（止盈/止损/仓位等）时，跳过耗时的训练步骤
$skip_training           = $true   # $true 启用 | $false 禁用

# ── 政策层（暴露门控）配置 ─────────────────────────────
# 模型来源（在线现算必需）：**与选股 OOS 14 折对齐的折集**（2026-09-20 起为默认；
# 旧 8 折日历半年集仍在 data\walk_forward\terminal_risk_wf，不再被本脚本默认使用）
$policy_model_root       = if ($PolicyModelRoot -ne "") { $PolicyModelRoot } else { "data\walk_forward\terminal_risk_wf_oos14" }
$policy_arm_suffix       = if ($PolicyArmSuffix -ne "") { $PolicyArmSuffix } else { "_v6m_fscore" }
# 覆盖起点：默认 "20181126" = **不裁剪**（覆盖全 14 折；= 回补转正臂 e2online_r 的实测配置）。
#   若需跳过最早的 OOS00 弱折（lift 1.077 < 1.1，见 R-006）⇒ 显式填 "20190527"；
#   若需与 R-004 §9 总扫描同覆盖 ⇒ 填 "20240102"。
#   语义 = **只抑制动作、不抑制阈值历史**（阈值窗口不冷启动）。
$policy_coverage_start   = if ($PolicyCoverageStart -ne "") { $PolicyCoverageStart } else { "20181126" }

# ── 暴露臂清单（每个元素一个臂；Policy 与 Table 互斥，同时给出直接报错）──
# Name      : 臂名（拼进 batch-period-label，便于汇总/产物分辨）
# Policy    : 在线政策 spec（空 = 不用；需 $policy_model_root / $policy_arm_suffix）
# Table     : 系数表 CSV（空 = 不用；仅用于回放/缓存，正确性绑定一份持仓路径）
# Replenish : 对称回补（P2-4）；Tolerance : 减仓/回补容差（0 = 引擎默认 3%）
# StopLoss  : 该臂启用止损
#
# 默认臂 e2online_r = R-004 §9 登记基准参数（E2-滚动250、λ=0.5、在线现算、不裁剪覆盖）
#   **+ 对称回补开（2026-09-20 转正，登记 docs/terminal_loss_risk_register.md R-007 §6/§7）**：
#   · λ∈{0.3,0.5,0.7} 与 q_regime=0.50 五个臂中，基准组为预登记判据通过且逐折 MaxDD 4/4 者；
#   · λ=0.3 数值更大（ΔMaxDD +1.41pp / Δ收益 +23.2pp）但已登记为“滞留行为收益驱动的放大”，
#     **不得读作更优参数** ⇒ 不默认；
#   · 回补转正依据 = **语义自洽**（λ 回满后应回满仓；现金拖累是 P2-4 已登记缺陷）
#     + 单臂实测方向一致（R vs A：ΔCAGR +1.20pp / ΔMaxDD +1.63pp，逐折年化改善 8/14；
#     窗口分解：16 个恢复窗口中 11 个正贡献）；
#   · **可外推性声明（必须随结论报告）**：回补的收益增量为“方向偏正、量级不可承诺”
#     （单窗口 ±3pp 量级事件会再现，如 split2 2020-03 的 −3.42pp）；**回撤改善不可外推**
#     （全链 −20.25%→−18.62% 由 split6 单折驱动）；不得把本次读数当作预期收益；
#   · StopLoss 对照已登记不通过（ΔMaxDD −0.14pp、逐折 3/4）⇒ 默认关；
#   · 在线口径复核：崩盘月贡献逐值可复现（2024-01/02），主判据 ΔMaxDD 对 λ 扰动敏感
#     ⇒ 默认值用于**日常回测一致性**，不得当作已证实的回撤改善
#     （见 docs/terminal_loss_policy_online_result.md）。
#   · 需要“无回补”对照（2026-09-20 前旧 e2online 口径）时：把本臂 Replenish 改 $false
#     或克隆一行 Name="e2online"；需要纯基线时换成下方 neutral 臂。
$exposure_arm_list = @(
    [PSCustomObject]@{
        Name = "e2online_r"; Policy = "arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5"
        Table = ""; Replenish = $true; Tolerance = 0; StopLoss = $false
    }
    # 需要基线对照（不启用政策、与历史基线逐位一致）时，换成下面这一臂：
    #[PSCustomObject]@{ Name = "neutral"; Policy = ""; Table = ""; Replenish = $false; Tolerance = 0; StopLoss = $false }
    # 需要“无回补”政策对照（旧 e2online 口径）时，改用下面这一臂：
    #[PSCustomObject]@{ Name = "e2online"; Policy = "arm=combined,mode=rolling,window=250,regime_q=0.75,score_q=0.5,lambda=0.5"; Table = ""; Replenish = $false; Tolerance = 0; StopLoss = $false }
)
# 命令行覆盖：给出任一暴露参数 ⇒ 忽略上面清单，按命令行构造单臂（不叠加）
# -NoExposure = 强制基线臂（与暴露类参数互斥）
if ($NoExposure -and (
    $ExposureTable -ne "" -or $ExposurePolicy -ne "" -or [bool]$ExposureReplenish -or
    [bool]$ExposureBudgetDiscount -or $TrimTolerance -gt 0
)) {
    throw "-NoExposure 与 -ExposurePolicy / -ExposureTable / -ExposureReplenish / -ExposureBudgetDiscount / -TrimTolerance 互斥"
}
# 建仓折扣回补必须与对称回补同用（折扣记账服务于回补；引擎层同校验，此处提前失败）
if ([bool]$ExposureBudgetDiscount -and -not [bool]$ExposureReplenish) {
    throw "-ExposureBudgetDiscount 必须与 -ExposureReplenish 同用（单开会记了没人回补）"
}
$cli_arm_specified = (
    $ExposureTable -ne "" -or $ExposurePolicy -ne "" -or [bool]$ExposureReplenish -or
    [bool]$ExposureBudgetDiscount -or
    $TrimTolerance -gt 0 -or [bool]$StopLoss -or [bool]$NoExposure
)
if ($cli_arm_specified) {
    $cli_policy    = if ($NoExposure) { "" } else { $ExposurePolicy }
    $cli_table     = if ($NoExposure) { "" } else { $ExposureTable }
    $cli_replenish = if ($NoExposure) { $false } else { [bool]$ExposureReplenish }
    $cli_budget    = if ($NoExposure) { $false } else { [bool]$ExposureBudgetDiscount }
    $cli_tolerance = if ($NoExposure) { 0 } else { $TrimTolerance }
    $cli_arm_name  = if ($NoExposure) { "neutral" } else { "cli" }
    $exposure_arm_list = @(
        [PSCustomObject]@{
            Name = $cli_arm_name; Policy = $cli_policy; Table = $cli_table
            Replenish = $cli_replenish; BudgetDiscount = $cli_budget
            Tolerance = $cli_tolerance; StopLoss = [bool]$StopLoss
        }
    )
}
foreach ($arm in $exposure_arm_list) {
    if ($arm.Policy -ne "" -and $arm.Table -ne "") {
        throw "暴露臂 $($arm.Name): Policy 与 Table 互斥（单变量直读）"
    }
    if (($arm.Replenish -or $arm.Tolerance -gt 0) -and $arm.Policy -eq "" -and $arm.Table -eq "") {
        throw "暴露臂 $($arm.Name): 回补/容差需要同时给出 -ExposurePolicy 或 -ExposureTable（无政策源时回补无意义，禁止静默忽略）"
    }
    $arm_has_budget = $arm.PSObject.Properties.Name -contains 'BudgetDiscount' -and [bool]$arm.BudgetDiscount
    if ($arm_has_budget -and -not [bool]$arm.Replenish) {
        throw "暴露臂 $($arm.Name): BudgetDiscount 必须与 Replenish 同用（折扣记账服务于回补）"
    }
}
# 防护：给出模型源/覆盖起点但**没有任何臂启用政策** ⇒ 这些参数不会生效，不得静默忽略
$policy_source_given = ($PolicyModelRoot -ne "" -or $PolicyArmSuffix -ne "" -or $PolicyCoverageStart -ne "")
$any_policy_enabled = @($exposure_arm_list | Where-Object { $_.Policy -ne "" }).Count -gt 0
if ($policy_source_given -and -not $any_policy_enabled) {
    throw ("已给出 -PolicyModelRoot/-PolicyArmSuffix/-PolicyCoverageStart，但当前臂清单没有任何 Policy" +
           "（仅 neutral，或 -NoExposure）⇒ 暴露政策不会启用、上述参数不会生效。" +
           "请同时给出 -ExposurePolicy（spec 字符串），或把配置区的 e2online 臂注释切换放开。")
}

# ── Walk-forward 时间段配置（支持多组）───────────────────────
# Label                : 时间段标签，仅用于日志/汇总展示
# SplitCount           : 训练切分数量
# FinalDate            : 最终日期（启用部署训练时=部署训练数据最后一天；禁用部署训练时=最后split测试结束日）
# ContinueDays         : 连续执行天数；>1 时会从 FinalDate 起按自然日逐日向后推进展开，并自动顺延到最近后一交易日
# StartModelVersion    : skip-training 模式下该时间段首个 split 对应模型版本号
# SelectedSplits       : 可选 split 下标列表（如 @(0,4,5,7,9)）；@() 或不填表示训练该时间段全部 split
$wf_period_configs = @(
    [PSCustomObject]@{
        Label = "0101"
        SplitCount = 14
        FinalDate = "20260105"# 20251231
        ContinueDays = 1
        StartModelVersion = 24008#,23682
        #SelectedSplits = @(0,4,5,7,8,9,10,12,13)
        SelectedSplits = @()
    }
    #[PSCustomObject]@{
    #    Label = "0109"
    #    SplitCount = 14
    #    FinalDate = "20260109" # 20260209
    #    ContinueDays = 1
    #    StartModelVersion = 22641
    #    SelectedSplits = @()
    #}
    #[PSCustomObject]@{
    #    Label = "0116"
    #    SplitCount = 14
    #    FinalDate = "20260116" # 20260324
    #    ContinueDays = 1
    #    StartModelVersion = 22656
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
$train_window_years_list = @(6)             # 训练窗口年数
$test_window_months_list = @(6)             # 测试窗口月数（建议与标签持仓周期接近）
$val_ratio_list          = @(0.2)           # 训练数据内部验证集比例，可改为 @(0.1, 0.15, 0.2) 扫描

# ── 标签与任务 ────────────────────────────────────────────────
$algorithm_list          = @("xgboost")        # xgboost | lightgbm（训练算法）
$label_list              = @("neu_y_ret_20")#,"neu_y_ret_20")      # skip-training 默认只保留单标签，避免对同一组旧模型重复回测
$task_list               = @("regression")     # regression | classification
$label_transform_list    = @("cs_zscore")      # raw | cs_zscore（仅 regression 有效）
$neutral_label_blend_weight_list = @(0)  # 0=纯行业中性标签，1=纯原始收益标签

# ── 模型超参（想对比的参数放多个值，其余放单个值）──────────────
$n_estimators_list       = @(500)      #. 树数量上限（配合早停，可多值扫描，如 @(500, 1000, 2000)）
$max_depth_list          = @(5)         #. XGB推荐9, LGB推荐5
$learning_rate_list      = @(0.03)      #0.009. XGB推荐0.005, LGB推荐0.005
$min_child_weight_list   = @(200)       #. XGB推荐150, LGB推荐200
$colsample_bytree_list   = @(0.3)       #. XGB/LGB均推荐0.3

$subsample_list          = @(0.8)       #. XGB推荐0.8, LGB推荐0.7
$reg_alpha_list          = @(0.03)      #. XGB推荐0.05, LGB推荐0.1
$reg_lambda_list         = @(7)         #. XGB推荐5.0, LGB推荐5.0
$gamma_list              = @(0.5)       #. 映射LGB min_split_gain。XGB推荐0.5, LGB推荐1.0
$num_leaves_list         = @(63)        #  仅LightGBM有效，XGBoost忽略。LGB推荐63

# ── 目标函数 ─────────────────────────────────────────────────
$objective_list          = @("mse")  # mse | lambdarank（排序学习，直接优化股票排序）
# ── 早停配置 ───────────────────────────────────────────────────
$early_stopping_rounds_list = @(50)    # 早停轮数，设为 0 则禁用早停（固定 n_estimators 棵树），可多值扫描如 @(100, 300, 500)
$early_stopping_metric   = "rank_ic_daily"  # 早停指标：auto（mae/auc）| rank_ic（整段Spearman）| rank_ic_daily（逐日截面Spearman均值，与daily_rankic评估口径一致）
$min_best_iteration      = 30          # best_iteration 下限监控，低于该值告警并在 summary 标记（0=禁用）


# ── rank-weight 配置（固定，不参与组合扫描）─────────────────────
$rank_weight_enabled     = $true   # $true 启用 | $false 禁用
$rank_weight_topk_list   = @(120)         # 120
$rank_weight_list        = @(100)         # 100
$rank_weight_topk_weight_mode = "linear_decay"   # linear_decay | flat

# ── 时间衰减权重 ──────────────────────────────────────────────
$time_decay_half_life_list = @(0)      # 半衰期（年）。0=禁用，1.0=1年前权重0.5，2.0=2年前权重0.5

# ── 多种子 bagging（每个split用多个随机种子各训一个子模型取平均，降训练随机方差）─
$ensemble_seeds            = "42,61,82"#,42,61,82"#,29,23"#42,61,82,100,200"#,300"#,400,500,600,700,800,900,1000"#,220,719"     # 逗号分隔种子如 "42,1,2,3,4"；空=单种子（用 --random-state），与多偏移可叠加
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
$enable_north            = $true  # $true 启用 | $false 禁用
# 0711 关闭后无影响
# 注意: factor_prune=$true 时生产排除清单含全部 13 个 north 因子, 本开关实际不参与训练;
# 2024-08-19 起披露口径切换已拆为净买入/成交额两套因子（见 v0.95.11）

# ── 龙虎榜因子（top_list 个股级, 2000+积分）──────────────────────
$enable_lhb              = $true  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约4%, 回撤提升约5%
# 0711关闭后微降,先保持打开

# ── 一致预期因子（report_rc 研报滚动聚合, 8000积分）──────────────
$enable_consensus        = $true  # $true 启用 | $false 禁用
#实测:打开后CAGR提升约2%, 回撤无明显变化
# 0711关闭后大幅提升分数

# ── 一致预期修正因子（0512基于已有 report_rc 构建时序修正信号，无需额外下载）─
$enable_consensus_revision = $false  # $true 启用 | $false 禁用（实验性因子）
# 0902-经多次验证, 此因子带来负向影响且有过拟合现象, 不应再打开实验

# ── 现金流质量因子（0512需 cashflow 接口，2000 积分，需先下载 cashflow 数据）─
$enable_cashflow_quality   = $true  # $true 启用 | $false 禁用（实验性因子）
# 0828关闭后收益及回撤都变得更好, 需要保持关闭. 0902-打开较好

# ── 分红政策质量因子（0829需 dividend 接口，2000 积分，需先下载 dividend 数据）─
$enable_dividend_policy    = $false  # $true 启用 | $false 禁用（实验性因子，默认关）
# 0902-关闭较好

### 以下为训练功能选择
# ── 特征稳定性筛选（移除跨时期IC方向不一致的特征, 0326引入）──────────────
$feature_stability_filter = $false  # $true 启用 | $false 禁用（实验验证效果不佳）
# 0711关闭后得分上升

# ── 因子精简（基于IC分析排除低效因子, 0606引入）────────────────────────
$factor_prune             = $false  # $true 启用 | $false 禁用（需先运行 generate_factor_exclude_list.py）

# ── 因子排除列表（可选，0801引入）────────────────────────────────────
# "" = 使用生产默认清单 data/models/factor_exclude_list.json（共 41 个因子）
# 非空 = 显式清单完全替换生产默认清单（仅排除该清单内因子，不会叠加）
# 注意: 本变量仅在 $factor_prune = $true 时生效；路径必须真实存在,
#       否则训练直接报错终止（不会静默跳过精简）
$factor_exclude_file      = ""#configs/factor_exclude_dividend_yield_only_v1.json"  

# ── freshness 处理策略（P2-C）────────────────────────────────────────────
$freshness_strategy             = "state_keep_event_decay"  # state_keep_event_decay | state_keep_event_no_decay | drop_all
$event_freshness_half_life_days = 120                           # 仅 decay 策略生效；no_decay/drop_all 的汇总中记为空

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
$stagger_tranches_list   = @(2)    # 1=不分批, 4=分4批（等效每rebalance_freq/4天调仓1/4仓位）

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
$position_sizing_list             = @('kelly')#, 'score', 'kelly', 'half_kelly') # equal | score | kelly | half_kelly
$kelly_vol_window_list           = @(60)      #60 Kelly 波动率窗口（交易日，可多值如 @(40, 60, 120)）
$kelly_max_leverage_list          = @(0.2)    #0.2 Kelly 单股仓位上限（可多值，如 @(0.15, 0.25)）

# ── 空仓/持有期拖尾提前调仓（独立开关）────
$enable_early_rebalance_on_empty_list = @($true)  # 可多值如 @($false, $true)

# ── OOS 止损（总开关）─────────────────────────────────────
# 每臂由 $exposure_arm_list.StopLoss 控制（见上方臂清单）；下方参数仅在启用时生效
$bt_stop_loss_drawdown_pct_list       = @($StopLossDrawdownPct) # 回撤止损阈值（%）
$bt_stop_loss_consecutive_limit_down_list = @(2) # 连续跌停止损天数
# ── 政策层（暴露门控 P2-3 / 在线现算 P2-5）──────────────────
# 取值来自上方 $exposure_arm_list（每个元素一个臂）；逐个拼进 python 命令。
# 系数表路径必须 ASCII（PowerShell 组装命令串会转码中文）；导出时务必按交易日历补齐台账缺口。
# ── 路径 ─────────────────────────────────────────────────────
$data_root               = "./data"

# ── 运行后自动汇总对比（强烈建议保持 $true）──────────────────
# 命令行加 -SkipCompare 则跳过（影子实验只关心逐折净值时省时）
$run_compare_after       = -not $SkipCompare

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
              $train_window_years_list.Length *
              $test_window_months_list.Length *
              $val_ratio_list.Length *
              $effective_label_list.Length *
              $neutral_label_blend_weight_list.Length *
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
              $kelly_max_leverage_list.Length *
              $exposure_arm_list.Length

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Walk-forward 批量实验" -ForegroundColor Cyan
Write-Host "  批次ID     : $batch_run_id" -ForegroundColor Cyan
Write-Host "  时间段数   : $($normalized_wf_period_configs.Count)" -ForegroundColor Cyan
Write-Host "  时间段列表 : $periodSummary" -ForegroundColor Cyan
Write-Host "  总任务数   : $totalTasks" -ForegroundColor Cyan
Write-Host "  暴露臂     : $(($exposure_arm_list | ForEach-Object { $_.Name }) -join ', ')" -ForegroundColor Cyan
Write-Host "  风险模型源 : $policy_model_root ($policy_arm_suffix)" -ForegroundColor Cyan
Write-Host "  数据目录   : $data_root" -ForegroundColor Cyan
Write-Host "  批次目录   : $batch_output_root" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($wfPeriod in $normalized_wf_period_configs) {
foreach ($algorithm in $algorithm_list) {
foreach ($train_window_years in $train_window_years_list) {
foreach ($test_window_months in $test_window_months_list) {
foreach ($val_ratio in $val_ratio_list) {
foreach ($label in $effective_label_list) {
foreach ($neutral_label_blend_weight in $neutral_label_blend_weight_list) {
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
foreach ($exposure_arm in $exposure_arm_list) {

    $count++
    $arm_name        = [string]$exposure_arm.Name
    $arm_policy      = [string]$exposure_arm.Policy
    $arm_table       = [string]$exposure_arm.Table
    $arm_replenish   = [bool]$exposure_arm.Replenish
    # 建仓折扣回补（A3 v2）：默认 $false；开启时 λ<1 信号日的买入折扣入回补释放额
    $arm_budget_discount = if ($exposure_arm.PSObject.Properties.Name -contains 'BudgetDiscount') {
        [bool]$exposure_arm.BudgetDiscount
    } else { $false }
    $arm_tolerance   = [double]$exposure_arm.Tolerance
    $arm_stop_loss   = [bool]$exposure_arm.StopLoss
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
    if ($arm_name -ne "" -and $arm_name -ne "neutral" -and $arm_name -ne "cli") {
        $batch_period_label = "{0}_{1}" -f $batch_period_label, $arm_name
    }
    $start_model_version = $wfPeriod.StartModelVersion
    $selected_splits = @($wfPeriod.SelectedSplits)
    $summary_csv_path = Join-Path $batch_raw_dir ("walk_forward_summary_{0}_{1:D4}.csv" -f $period_label, $count)

    # 构建命令字符串
    $pythonCmd = "py .\scripts\walk_forward.py" +
                 " --algorithm $algorithm" +
                 " --split-count $split_count" +
                 " --final-date $final_date" +
                 " --train-window-years $train_window_years" +
                 " --test-window-months $test_window_months" +
                 " --val-ratio $val_ratio" +
                 " --label $label" +
                 " --neutral-label-blend-weight $neutral_label_blend_weight" +
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
                 " --min-best-iteration $min_best_iteration" +
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
        if ($factor_exclude_file -ne "") {
            $pythonCmd += " --factor-exclude-file `"$factor_exclude_file`""
        }
    }

    if ($ensemble_offsets -gt 0) {
        $pythonCmd += " --ensemble-offsets $ensemble_offsets"
    }

    if ($ensemble_seeds -ne "") {
        $pythonCmd += " --ensemble-seeds $ensemble_seeds" +
                      " --ensemble-seed-keep-top-ratio $ensemble_seed_keep_top_ratio" +
                      " --ensemble-seed-keep-min-models $ensemble_seed_keep_min_models"
    }

    if ($enable_enhanced) {
        $pythonCmd += " --enable-enhanced-features"
    }

    if ($enable_cashflow_quality) {
        $pythonCmd += " --enable-cashflow-quality-features"
    }

    if ($enable_dividend_policy) {
        $pythonCmd += " --enable-dividend-policy-features"
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
        if ($arm_stop_loss) {
            $pythonCmd += " --bt-stop-loss-enabled" +
                          " --bt-stop-loss-drawdown-pct $bt_stop_loss_drawdown_pct" +
                          " --bt-stop-loss-consecutive-limit-down $bt_stop_loss_consecutive_limit_down"
        }
        if ($arm_policy -ne "") {
            $pythonCmd += " --exposure-policy `"$arm_policy`"" +
                          " --policy-model-root `"$policy_model_root`"" +
                          " --policy-arm-suffix $policy_arm_suffix"
            if ($policy_coverage_start -ne "") {
                $pythonCmd += " --policy-coverage-start $policy_coverage_start"
            }
        }
        if ($arm_table -ne "") {
            $pythonCmd += " --exposure-table `"$arm_table`""
        }
        if ($arm_policy -ne "" -or $arm_table -ne "") {
            if ($arm_replenish) {
                $pythonCmd += " --exposure-replenish"
            }
            if ($arm_budget_discount) {
                $pythonCmd += " --exposure-budget-discount-replenish"
            }
            if ($arm_tolerance -gt 0) {
                $pythonCmd += " --exposure-trim-tolerance $arm_tolerance"
            }
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
        Write-Host "[任务 $count / $totalTasks][时间段 $period_label][臂 $arm_name][split=$split_count, final=$final_date][day=$($continue_offset + 1)/$continue_days]" -ForegroundColor Green
    } else {
        Write-Host "[任务 $count / $totalTasks][时间段 $period_label][臂 $arm_name][split=$split_count, final=$final_date]" -ForegroundColor Green
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

}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}  #  end foreach（时间段+参数组合+暴露臂）

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






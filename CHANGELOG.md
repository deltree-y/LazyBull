# Changelog

All notable changes to this project will be documented in this file.

## [0.85.4] - 2026-07-23

### Changed

- **walk-forward 风险惩罚强度支持可配调节（lambda scale/grid）**：
  - `scripts/walk_forward.py` 新增参数 `--risk-penalty-lambda-scale` 与 `--risk-penalty-lambda-grid`。
  - 在 split 训练与部署训练路径中，`learn_risk_penalty_config()` 现透传 `lambda_grid`，支持更温和/更激进的惩罚强度探索。
  - 当未显式提供 grid 且 `scale != 1.0` 时，自动按默认网格缩放生成候选，保持原默认行为可兼容。

### Added

- **walk-forward 汇总新增风险惩罚效果诊断列**：
  - `risk_penalty_penalized_ratio`（惩罚覆盖率）
  - `risk_penalty_penalty_mean`（惩罚均值）
  - `risk_penalty_topk_changed_days_ratio`（TopN变更日占比）
  - `risk_penalty_swap_alpha`（替换收益贡献）
- **批量对比报表接入风险惩罚效果指标**：
  - `scripts/compare_walk_forward.py` 已支持上述诊断列的聚合、中文映射与主表展示。
- **单元测试补充**：
  - `tests/test_walk_forward.py` 新增 lambda 网格解析与惩罚效果指标计算测试。

## [0.85.3] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的特征名不一致导致惩罚静默失效**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 在 v2 路径中，现会按分类器 `feature_names_in_` 对 `X_clf` 做列对齐（补缺失列、剔除多余列、重排列顺序）。
  - 修复了分类器训练列不含 `mkt_drawdown_20` 但评估输入包含该列时 `predict_proba` 抛错并被静默回退到 `pred_score` 的问题。
  - v2 预测异常日志补充了 `bad_pick_model_version` 与具体异常信息，便于定位线上回退原因。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“输入含额外市场特征列仍能成功应用惩罚”的场景。

## [0.85.2] - 2026-07-23

### Fixed

- **修复 conditional bad-pick 在 walk-forward 评估侧的分类器兜底加载不可达分支**：
  - `scripts/walk_forward.py` 的 `_apply_risk_penalty_scores()` 原先在 `_clf_model` 缺失时会提前返回，导致后续按 `bad_pick_model_version` 兜底加载分类器的逻辑永远无法执行。
  - 现已调整为：先尝试按模型版本恢复分类器，兜底失败后再跳过惩罚，并输出明确失败原因日志。
  - 新增 `tests/test_walk_forward.py` 回归测试，覆盖“_clf_model 缺失 + bad_pick_model_version 兜底”路径。

## [0.85.1] - 2026-07-23

### Fixed

- **年化收益率统一为 CAGR 公式**：`src/lazybull/paper/broker.py` 的 `_calculate_annualized_return()` 从简单线性年化改为复合年化（CAGR），与批量汇总脚本一致。
- **批量纸面交易汇总结束日期修复**：`scripts/batch/batch_paper_trade.ps1` 的 `Get-NavSummary` 现在使用实际最终交易日（`$finalTradeDate`）而非 `nav.parquet` 末行日期，确保年化收益率计算覆盖完整的实际交易区间。

## [0.85.0] - 2026-07-23

### Changed

- **条件式 Bad-Pick 模型（完全替换线性惩罚）**：
  - src/lazybull/risk/bad_pick.py（新建）：BadPickConfig/RegimeBadPickConfig, detect_market_regime() 三层OR判断, apply_conditional_penalty() 阈值门控扣分。
  - src/lazybull/ml/train_core.py：learn_risk_penalty_config() 重写为训练XGB二分类器+分位数网格校准regime阈值+per-regime二维网格搜索。
  - src/lazybull/signals/ml_signal.py：_apply_risk_penalty() 支持v1/v2双模式，_load_model() 自动加载坏票分类器。
  - src/lazybull/ml/model_registry.py：register_model() 自动注册坏票分类器并回填版本号。
  - scripts/walk_forward.py 和 scripts/train_ml_model.py：日志与评估逻辑同步适配v2。
- **分类器特征从15个扩展到22个**：新增主力资金、动量衰竭、日内形态、极端估值、分析师分歧等维度。
- **所有阈值用分位数定义**：每个walk-forward split独立校准，自动适应市场数据分布进化。

### Added

- 	ests/test_bad_pick_conditional.py：21个测试覆盖配置序列化、regime检测、特征提取、门控逻辑、边界条件。

## [0.84.0] - 2026-07-22

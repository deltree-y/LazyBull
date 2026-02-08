# Changelog

All notable changes to this project will be documented in this file.

## [0.3.5] - 2026-02-08

### Added
- 纸面交易T0等权策略的"一手可买约束 + 顺延补足"功能
  - 在等权模式下（`weight_method=="equal"`），对每只候选股票检查按资金分配是否能买入至少1手（100股）
  - 不足1手的股票将被跳过，并从排序候选中顺延选择下一只
  - 确保最终保存到pending的目标都是可有效购买的股票
  - 添加详细日志：原始候选数、跳过数、最终目标数、跳过示例

### Changed
- `PaperTradingRunner._generate_signals` 方法新增 `buy_price_type` 参数，用于确定一手判断价格
- 复用 `MLSignal.generate_ranked()` 方法获取完整排序候选列表，支持顺延补足

### Technical Details
- 新增 `PaperTradingRunner._generate_equal_weight_with_lot_constraint` 方法实现核心逻辑
- 等权策略下使用T1的买入价格类型（open/close）进行一手可买性判断
- Score加权策略暂不启用此约束，保持原有行为

## [0.3.4] - Previous Version
- 之前版本的功能

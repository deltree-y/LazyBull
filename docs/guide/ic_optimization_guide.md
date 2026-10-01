# 模型 IC 与 RankIC 优化指南

## 目录
- [IC 与 RankIC 基础](#ic-与-rankic-基础)
- [当前问题诊断](#当前问题诊断)
- [优化方向](#优化方向)
- [具体实施建议](#具体实施建议)
- [评估与监控](#评估与监控)

---

## IC 与 RankIC 基础

### 什么是 IC (Information Coefficient)

IC（信息系数）是衡量因子预测能力的核心指标，表示因子值与未来收益率之间的相关性：

```
IC = Corr(因子值, 未来收益率)
```

**解读标准：**
- IC > 0.05: 具有较好的预测能力
- IC > 0.03: 有一定预测能力，可以使用
- IC < 0.03: 预测能力较弱
- IC 接近 0: 基本无预测能力
- IC < 0: 负相关，可能需要反向使用

### 什么是 RankIC

RankIC 是基于排序的相关系数（Spearman 相关系数），相比 IC 更稳健：

```
RankIC = Spearman_Corr(因子排序, 收益率排序)
```

**为什么 RankIC 更重要：**
- 对异常值不敏感
- 更符合选股策略的本质（选择排名靠前的股票）
- 在量化策略中通常比 IC 更稳定

---

## 当前问题诊断

### 问题描述

项目当前的模型 IC 与 RankIC 只有 **0.1 左右**，这表明：

1. **预测能力有限**：0.1 的 IC 处于可接受的边缘水平
2. **信息比率较低**：可能导致策略收益不稳定
3. **有优化空间**：通过系统性改进可以显著提升

### 可能原因

1. **标签定义问题**
   - 未来收益率计算方式不合理
   - 预测时间窗口（horizon）选择不当
   - 未考虑交易成本和滑点

2. **特征质量问题**
   - 特征与收益率相关性弱
   - 特征存在噪音和极端值
   - 缺少行业/市值中性化处理

3. **样本选择问题**
   - 训练集包含不可交易样本（停牌、涨跌停等）
   - 样本分布不均衡（大小盘、不同行业）
   - 时间窗口选择不当（包含特殊市场环境）

4. **模型训练问题**
   - 过拟合或欠拟合
   - 超参数未优化
   - 缺少交叉验证

---

## 优化方向

### 1. 特征工程优化

#### 1.1 去极值 (Winsorize)

极端值会严重影响模型训练，建议处理：

```python
from scipy.stats import mstats

# 方法1：百分位截断（推荐）
def winsorize_features(df, feature_cols, limits=(0.01, 0.01)):
    """
    对特征进行去极值处理
    
    Args:
        df: 特征DataFrame
        feature_cols: 需要处理的特征列
        limits: 截断比例，(lower, upper)，默认截断上下1%
    """
    df_processed = df.copy()
    for col in feature_cols:
        df_processed[col] = mstats.winsorize(df[col], limits=limits)
    return df_processed

# 方法2：MAD（中位数绝对偏差）方法
def mad_winsorize(series, n=3):
    """
    使用 MAD 方法去极值
    
    Args:
        series: 数据序列
        n: 标准差倍数，默认3
    """
    median = series.median()
    mad = (series - median).abs().median()
    upper = median + n * mad * 1.4826
    lower = median - n * mad * 1.4826
    return series.clip(lower, upper)
```

#### 1.2 标准化 (Standardization)

不同量纲的特征需要标准化：

```python
# 方法1：Z-Score 标准化（推荐）
def standardize_features(df, feature_cols):
    """
    对特征进行标准化
    
    标准化后均值为0，标准差为1
    """
    df_processed = df.copy()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df_processed[col] = (df[col] - mean) / std
        else:
            df_processed[col] = 0
    return df_processed

# 方法2：Min-Max 归一化
def normalize_features(df, feature_cols):
    """
    Min-Max 归一化到 [0, 1] 区间
    """
    df_processed = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_processed[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_processed[col] = 0
    return df_processed
```

#### 1.3 行业中性化

消除行业因素的影响，提取纯粹的选股能力：

```python
def neutralize_by_industry(df, feature_cols, industry_col='industry'):
    """
    行业中性化处理
    
    对每个特征，在行业内进行标准化
    
    Args:
        df: 特征DataFrame
        feature_cols: 需要中性化的特征列
        industry_col: 行业分类列名
    """
    df_processed = df.copy()
    
    for col in feature_cols:
        # 按行业分组，计算组内均值和标准差
        grouped = df.groupby(industry_col)[col]
        industry_mean = grouped.transform('mean')
        industry_std = grouped.transform('std')
        
        # 组内标准化
        df_processed[col] = (df[col] - industry_mean) / industry_std.replace(0, 1)
    
    return df_processed
```

#### 1.4 市值中性化

消除市值因素的影响：

```python
def neutralize_by_market_cap(df, feature_cols, market_cap_col='total_mv'):
    """
    市值中性化处理
    
    使用回归法消除市值因素的影响
    
    Args:
        df: 特征DataFrame
        feature_cols: 需要中性化的特征列
        market_cap_col: 市值列名
    """
    from sklearn.linear_model import LinearRegression
    
    df_processed = df.copy()
    
    for col in feature_cols:
        # 准备数据
        X = df[[market_cap_col]].values
        y = df[col].values
        
        # 过滤缺失值
        mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) > 0:
            # 回归
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            
            # 计算残差（即中性化后的值）
            y_pred = model.predict(X)
            df_processed.loc[mask, col] = y[mask] - y_pred[mask]
    
    return df_processed
```

### 2. 标签定义优化

#### 2.1 考虑交易成本的标签

当前标签可能未考虑交易成本，导致模型学习到不可交易的信号：

```python
def calculate_label_with_cost(
    df,
    horizon=5,
    commission_rate=0.0003,
    stamp_tax=0.001,
    slippage=0.001
):
    """
    计算考虑交易成本的收益率标签
    
    Args:
        df: 包含价格数据的DataFrame
        horizon: 预测窗口（交易日）
        commission_rate: 佣金费率（买卖双向）
        stamp_tax: 印花税（仅卖出）
        slippage: 滑点
    """
    # 计算原始收益率
    df['ret_raw'] = df.groupby('ts_code')['close'].pct_change(horizon).shift(-horizon)
    
    # 扣除交易成本
    # 买入成本：佣金 + 滑点
    buy_cost = commission_rate + slippage
    # 卖出成本：佣金 + 印花税 + 滑点
    sell_cost = commission_rate + stamp_tax + slippage
    # 总成本
    total_cost = buy_cost + sell_cost
    
    # 净收益率 = 原始收益率 - 交易成本
    df['y_ret_net'] = df['ret_raw'] - total_cost
    
    return df
```

#### 2.2 分层标签

使用分层标签可以提高 RankIC：

```python
def create_quantile_labels(df, label_col='ret_5', n_quantiles=5):
    """
    创建分位数标签（适合分类模型）
    
    将收益率分成 n 个分位数，标签为 0, 1, ..., n-1
    
    Args:
        df: DataFrame
        label_col: 收益率列名
        n_quantiles: 分位数数量
    """
    df['label_quantile'] = pd.qcut(
        df[label_col],
        q=n_quantiles,
        labels=False,
        duplicates='drop'
    )
    return df
```

### 3. 样本选择优化

#### 3.1 严格的样本过滤

```python
def filter_training_samples(df):
    """
    严格过滤训练样本，只保留可交易的样本
    
    过滤规则：
    1. 排除ST股票
    2. 排除停牌股票
    3. 排除涨跌停股票（T日）
    4. 排除新股（上市不足N天）
    5. 排除成交量异常的股票
    """
    mask = (
        (~df['is_st']) &                    # 非ST
        (~df['suspend']) &                  # 非停牌
        (~df['limit_up']) &                 # 非涨停
        (~df['limit_down']) &               # 非跌停
        (df['list_days'] >= 60) &          # 上市>=60天
        (df['vol'] > df['vol'].quantile(0.1))  # 成交量>10%分位数
    )
    
    return df[mask]
```

#### 3.2 样本均衡

确保不同市值、不同行业的样本均衡：

```python
def balanced_sampling(df, group_col='industry', n_samples_per_group=1000):
    """
    分组均衡采样
    
    Args:
        df: DataFrame
        group_col: 分组列（如行业、市值分组）
        n_samples_per_group: 每组采样数量
    """
    sampled_dfs = []
    for name, group in df.groupby(group_col):
        if len(group) >= n_samples_per_group:
            sampled = group.sample(n=n_samples_per_group, random_state=42)
        else:
            sampled = group
        sampled_dfs.append(sampled)
    
    return pd.concat(sampled_dfs, ignore_index=True)
```

### 4. 模型训练优化

#### 4.1 时间序列交叉验证

避免使用随机拆分，应该使用时间序列交叉验证：

```python
def time_series_cv_split(df, n_splits=5, test_ratio=0.2):
    """
    时间序列交叉验证拆分
    
    使用滚动窗口方式拆分，避免未来信息泄漏
    
    Args:
        df: 按时间排序的DataFrame
        n_splits: 拆分数量
        test_ratio: 测试集比例
    """
    df_sorted = df.sort_values('trade_date')
    total_dates = df_sorted['trade_date'].unique()
    n_dates = len(total_dates)
    test_size = int(n_dates * test_ratio)
    
    splits = []
    for i in range(n_splits):
        # 测试集：最后 test_size 个日期
        test_end_idx = n_dates - i * (test_size // n_splits)
        test_start_idx = test_end_idx - test_size
        
        if test_start_idx < test_size:
            break
        
        train_dates = total_dates[:test_start_idx]
        test_dates = total_dates[test_start_idx:test_end_idx]
        
        train_df = df_sorted[df_sorted['trade_date'].isin(train_dates)]
        test_df = df_sorted[df_sorted['trade_date'].isin(test_dates)]
        
        splits.append((train_df, test_df))
    
    return splits
```

#### 4.2 超参数优化

使用网格搜索或贝叶斯优化寻找最优超参数：

```python
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """
    超参数网格搜索
    """
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0]
    }
    
    best_ic = -1
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        y_pred = model.predict(X_val)
        ic = pd.Series(y_val).corr(pd.Series(y_pred))
        
        if ic > best_ic:
            best_ic = ic
            best_params = params
    
    return best_params, best_ic
```

### 5. 调仓频率优化

调仓频率会影响 IC 的表现：

- **高频调仓**（日频）：
  - 优点：及时捕捉短期信号
  - 缺点：交易成本高，IC 衰减快
  
- **中频调仓**（周频）：
  - **推荐**：平衡收益与成本
  - 适合 horizon=5（一周）的标签
  
- **低频调仓**（月频）：
  - 优点：交易成本低
  - 缺点：可能错过短期机会

建议根据标签的 horizon 选择匹配的调仓频率，并测试不同频率下的实际收益。

---

## 具体实施建议

### 短期优化（1-2周可完成）

1. **特征预处理**
   - [ ] 实现去极值函数（winsorize）
   - [ ] 实现标准化函数（Z-score）
   - [ ] 在训练前自动应用预处理

2. **样本过滤**
   - [ ] 严格过滤不可交易样本
   - [ ] 排除涨跌停股票
   - [ ] 排除成交量异常股票

3. **标签优化**
   - [ ] 考虑交易成本调整标签
   - [ ] 测试不同 horizon 的效果

4. **模型调优**
   - [ ] 使用时间序列交叉验证
   - [ ] 增加正则化参数
   - [ ] 使用早停机制

### 中期优化（2-4周可完成）

1. **行业中性化**
   - [ ] 添加行业分类数据
   - [ ] 实现行业中性化函数
   - [ ] 评估中性化效果

2. **市值中性化**
   - [ ] 实现市值中性化函数
   - [ ] 测试不同中性化方法

3. **超参数优化**
   - [ ] 实现自动超参数搜索
   - [ ] 使用贝叶斯优化加速

4. **特征选择**
   - [ ] 计算特征重要性
   - [ ] 移除低相关性特征
   - [ ] 测试不同特征组合

### 长期优化（1-2月可完成）

1. **高级特征工程**
   - [ ] 构建因子库（动量、反转、价值等）
   - [ ] 添加技术指标特征
   - [ ] 添加情绪指标特征

2. **集成学习**
   - [ ] 训练多个模型
   - [ ] 实现模型融合策略
   - [ ] Stacking/Blending

3. **深度学习**
   - [ ] 尝试 LSTM/GRU 时序模型
   - [ ] 尝试 Transformer 架构
   - [ ] 结合深度学习与树模型

---

## 评估与监控

### 关键指标

#### 1. IC 统计指标

```python
def calculate_ic_metrics(predictions, actuals):
    """
    计算 IC 相关统计指标
    
    Returns:
        dict: 包含 IC, RankIC, IC均值, IC标准差, IR, 胜率等
    """
    ic = predictions.corr(actuals)
    rank_ic = predictions.corr(actuals, method='spearman')
    
    # 按日期计算 IC 序列
    ic_series = []
    for date in predictions.index.unique():
        mask = predictions.index == date
        daily_ic = predictions[mask].corr(actuals[mask])
        ic_series.append(daily_ic)
    
    ic_series = pd.Series(ic_series)
    
    return {
        'IC': ic,
        'RankIC': rank_ic,
        'IC_mean': ic_series.mean(),
        'IC_std': ic_series.std(),
        'IR': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        'IC_win_rate': (ic_series > 0).sum() / len(ic_series),
        'IC_positive_days': (ic_series > 0).sum(),
        'IC_negative_days': (ic_series < 0).sum()
    }
```

#### 2. 分层回测

```python
def layered_backtest(df, prediction_col, return_col, n_layers=5):
    """
    分层回测：将股票按预测值分成 n 层，计算每层的收益
    
    用于诊断模型的单调性和区分能力
    """
    results = []
    
    for date in df['trade_date'].unique():
        date_df = df[df['trade_date'] == date].copy()
        
        # 按预测值分层
        date_df['layer'] = pd.qcut(
            date_df[prediction_col],
            q=n_layers,
            labels=False,
            duplicates='drop'
        )
        
        # 计算每层的平均收益
        layer_returns = date_df.groupby('layer')[return_col].mean()
        
        for layer, ret in layer_returns.items():
            results.append({
                'trade_date': date,
                'layer': layer,
                'return': ret
            })
    
    results_df = pd.DataFrame(results)
    
    # 汇总统计
    summary = results_df.groupby('layer')['return'].agg(['mean', 'std', 'count'])
    summary['sharpe'] = summary['mean'] / summary['std']
    
    return results_df, summary
```

#### 3. IC 时间稳定性

```python
def ic_stability_analysis(df, prediction_col, return_col, window=20):
    """
    分析 IC 的时间稳定性
    
    Args:
        window: 滚动窗口大小（交易日）
    """
    dates = sorted(df['trade_date'].unique())
    
    rolling_ics = []
    for i in range(len(dates) - window + 1):
        window_dates = dates[i:i+window]
        window_df = df[df['trade_date'].isin(window_dates)]
        
        ic = window_df[prediction_col].corr(window_df[return_col])
        rank_ic = window_df[prediction_col].corr(
            window_df[return_col],
            method='spearman'
        )
        
        rolling_ics.append({
            'end_date': window_dates[-1],
            'IC': ic,
            'RankIC': rank_ic
        })
    
    rolling_df = pd.DataFrame(rolling_ics)
    
    # 计算稳定性指标
    stability_metrics = {
        'IC_mean': rolling_df['IC'].mean(),
        'IC_std': rolling_df['IC'].std(),
        'IC_min': rolling_df['IC'].min(),
        'IC_max': rolling_df['IC'].max(),
        'RankIC_mean': rolling_df['RankIC'].mean(),
        'RankIC_std': rolling_df['RankIC'].std()
    }
    
    return rolling_df, stability_metrics
```

### 目标设定

根据项目当前状态（IC ≈ 0.1），建议分阶段目标：

**第一阶段（短期）：**
- IC: 0.1 → 0.15
- RankIC: 0.1 → 0.15
- IR（信息比率）: > 0.5

**第二阶段（中期）：**
- IC: 0.15 → 0.20
- RankIC: 0.15 → 0.20
- IR: > 1.0

**第三阶段（长期）：**
- IC: 0.20 → 0.25+
- RankIC: 0.20 → 0.25+
- IR: > 1.5

### 监控仪表板

建议建立监控系统，跟踪以下指标：

1. **训练指标**
   - 训练集 IC/RankIC
   - 验证集 IC/RankIC
   - 过拟合程度（训练-验证 gap）

2. **回测指标**
   - 累计收益率
   - 最大回撤
   - 夏普比率
   - 卡尔玛比率

3. **稳定性指标**
   - IC 标准差
   - IC 胜率
   - 分层收益单调性

4. **风险指标**
   - 持仓集中度
   - 行业暴露
   - 市值暴露

---

## 总结

IC 与 RankIC 的提升是一个系统工程，需要从多个维度协同优化：

1. **数据质量**：严格的样本过滤 + 预处理
2. **特征工程**：去极值 + 标准化 + 中性化
3. **标签设计**：考虑成本 + 匹配调仓频率
4. **模型训练**：交叉验证 + 超参数优化 + 正则化
5. **持续监控**：建立评估体系 + 跟踪稳定性

建议优先实施**短期优化措施**，这些改进成本低、见效快，可以快速验证效果。在看到改进后，再逐步推进中长期优化。

**关键原则：小步快跑，快速迭代，数据驱动决策。**

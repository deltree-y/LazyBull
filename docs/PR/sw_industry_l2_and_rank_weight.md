# PR 说明：申万二级行业切换 + rank-weight 训练增强

**版本**：v0.12.0  
**日期**：2026-02-20  
**模块**：data/cleaner.py、features/builder.py、ml/train_core.py、scripts/train_ml_model.py

---

## 一、申万行业从一级切换到二级

### 背景

v0.11.0 使用申万**一级**行业（~30个行业）作为中性化分组字段。一级行业粒度过粗，  
导致同一行业内仍有较大的行业效应残差。切换为**二级行业**（~100个子行业）  
可显著提升中性化精度，使 `neu_y_ret_*` 标签更准确反映个股超额。

### 实现变更

#### 1. `src/lazybull/data/cleaner.py`

- `clean_shenwan_industry()` 默认 `level_str` 从 `'l1'` 改为 `'l2'`
- 添加字段映射注释（clean 层输出字段 -> FeatureBuilder 输出字段）：

  | clean 层字段 | level | 含义 | FeatureBuilder 对应字段 |
  |---|---|---|---|
  | sw_code | l2 | 申万二级行业指数代码（如 `110101`） | sw_industry_code |
  | sw_name | l2 | 申万二级行业名称（如 `银行I`） | sw_industry |

#### 2. `src/lazybull/features/builder.py`

**`_merge_shenwan_industry()` 重命名输出字段**：

| 旧字段名 | 新字段名 | 说明 |
|---|---|---|
| sw_name | sw_industry | 申万二级行业名称，用于中性化分组 |
| sw_code | sw_industry_code | 申万二级行业指数代码 |
| industry_id | sw_industry_id | 稳定整数编码（按 sw_industry 排序映射） |

**`_apply_industry_neutralization()` 更新**：
- 检查列由 `sw_name` 改为 `sw_industry`
- `industry_col` 参数传递由 `'sw_name'` 改为 `'sw_industry'`

**`_add_advanced_factors()` 更新**：
- 行业 alpha 计算使用 `industry_col='sw_industry'`（若列存在）

### 字段命名原则

- 统一使用 `sw_industry` / `sw_industry_id`，不使用 `sw_l1` / `sw_l2` 命名
- 行业 level（一级/二级）通过 `DataCleaner.clean_shenwan_industry(level_str=)` 参数控制
- FeatureBuilder 不感知 level，只依赖 clean 层输出的 `sw_code`/`sw_name` 字段

### features 中不再输出 stock_basic.industry

train_core.py 的 `prepare_training_data()` 中 `other_exclude_columns` 已包含 `'industry'`，  
该 tushare 行业字段在特征列中不可见，本次变更维持该设定不变。

---

## 二、训练时 Top30/Bottom30 sample_weight 增强

### 背景

默认情况下，XGBoost 对所有样本等权训练。对于选股任务而言，  
预测精度最重要的是**头部（Top30）和尾部（Bottom30）样本**，  
这些样本直接决定买入/做空标的。通过对这些样本赋予更高权重，  
可以强化模型对极端截面的预测精度。

### 实现变更

#### 1. `src/lazybull/ml/train_core.py`

新增函数 **`build_rank_sample_weights()`**：

```python
def build_rank_sample_weights(
    df_train: pd.DataFrame,
    label_column: str,
    topk: int = 30,
    top_weight: float = 5.0,
    date_col: str = 'trade_date'
) -> np.ndarray:
```

- 按 `trade_date` 分组，对每组标签值排序
- Top K（最大）和 Bottom K（最小）样本权重 = `top_weight`
- 其余样本权重 = 1.0
- **退化处理**：若某日样本数 `n <= 2*topk`，则整日全部赋予 `top_weight`

`train_xgboost_model()` 新增参数 `sample_weight: Optional[np.ndarray] = None`，  
传递给 `model.fit(sample_weight=...)`。

#### 2. `scripts/train_ml_model.py`

新增 CLI 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rank-weight-enabled` | `True`（默认开启） | 启用 Top/Bottom K 权重增强 |
| `--no-rank-weight` | — | 禁用 rank-weight（覆盖 --rank-weight-enabled） |
| `--rank-weight-topk` | `30` | 每日 Top/Bottom K 样本数 |
| `--rank-weight-weight` | `5.0` | Top/Bottom K 样本权重 |

训练参数记录到 `full_train_params`，写入 `ml_train_runs.csv`（字段：  
`rank_weight_enabled`、`rank_weight_topk`、`rank_weight_weight`）。

### 使用示例

```bash
# 默认开启 rank-weight（topk=30，weight=5.0）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231

# 自定义参数
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
    --rank-weight-topk 50 --rank-weight-weight 3.0

# 禁用 rank-weight（恢复等权训练）
python scripts/train_ml_model.py --start-date 20230101 --end-date 20231231 \
    --no-rank-weight
```

---

## 三、测试覆盖

### 新增测试文件

1. **`tests/test_sw_industry_l2.py`**（11 个测试）
   - `_merge_shenwan_industry()` 输出字段正确（sw_industry/sw_industry_code/sw_industry_id）
   - 旧字段（sw_name/sw_code/industry_id）不再出现
   - sw_industry_id 编码稳定
   - 空/异常数据的安全处理
   - `clean_shenwan_industry()` 默认 level_str='l2' 生效

2. **`tests/test_rank_sample_weight.py`**（13 个测试）
   - 单日 Top K / Bottom K 权重正确
   - 加权样本数恰好为 2*topk
   - K 大于样本数时全部加权（退化处理）
   - 多日分组独立（不串权重）
   - NaN 标签、缺失列等鲁棒性测试

---

## 四、版本与文档更新

- `src/lazybull/__init__.py`：`0.11.0` -> `0.12.0`
- `pyproject.toml`：`0.11.0` -> `0.12.0`
- `README.md`：更新当前版本描述
- `CHANGELOG.md`：新增 v0.12.0 条目
- `docs/features_schema.md`：更新行业字段说明（sw_industry* 命名）
- `docs/PR/sw_industry_l2_and_rank_weight.md`：本文档
- `docs/guide/rank_weight_guide.md`：rank-weight 使用与验证指南

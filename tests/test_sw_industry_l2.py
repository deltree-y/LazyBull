"""测试申万二级行业字段切换

验证：
- FeatureBuilder._merge_shenwan_industry() 输出 sw_industry / sw_industry_code / sw_industry_id
- sw_industry_id 编码稳定（相同名称始终映射到相同整数）
- _apply_industry_neutralization() 使用 sw_industry 进行分组中性化
- DataCleaner.clean_shenwan_industry() 默认使用 level_str='l2'
"""

import pandas as pd
import pytest

from src.lazybull.data import DataCleaner
from src.lazybull.features.builder import FeatureBuilder


# ---------------------------------------------------------------------------
# 通用 mock 数据
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_shenwan_industry_clean():
    """模拟 clean 层申万行业分类表（sw_code=二级代码，sw_name=二级名称）"""
    return pd.DataFrame({
        'ts_code': [
            '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ',
            '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ', '000010.SZ',
        ],
        'sw_code': ['110101', '110101', '110101', '110101', '110101',
                    '210101', '210101', '210101', '210101', '210101'],
        'sw_name': ['国有大型银行', '国有大型银行', '国有大型银行', '国有大型银行', '国有大型银行',
                    '化学原料', '化学原料', '化学原料', '化学原料', '化学原料'],
    })


@pytest.fixture
def mock_features_df():
    """模拟特征 DataFrame（合并行业信息之前）"""
    return pd.DataFrame({
        'ts_code': [
            '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ',
            '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ', '000010.SZ',
        ],
        'trade_date': ['20230102'] * 10,
        'ret_1': [0.01, 0.02, -0.01, 0.03, -0.02,
                  0.05, -0.03, 0.01, 0.02, -0.01],
        'tradable': [1] * 10,
    })


# ---------------------------------------------------------------------------
# 1. 测试 _merge_shenwan_industry 输出字段
# ---------------------------------------------------------------------------

class TestMergeShenwanIndustry:
    """测试 FeatureBuilder._merge_shenwan_industry 输出字段"""

    def test_output_columns_renamed(self, mock_features_df, mock_shenwan_industry_clean):
        """合并后应输出 sw_industry / sw_industry_code / sw_industry_id，
        而不再输出旧字段 sw_name / sw_code / industry_id"""
        builder = FeatureBuilder()
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean)

        # 新字段存在
        assert 'sw_industry' in result.columns, "缺少 sw_industry 列"
        assert 'sw_industry_code' in result.columns, "缺少 sw_industry_code 列"
        assert 'sw_industry_id' in result.columns, "缺少 sw_industry_id 列"

        # 旧字段不再存在
        assert 'sw_name' not in result.columns, "旧字段 sw_name 不应出现在输出中"
        assert 'sw_code' not in result.columns, "旧字段 sw_code 不应出现在输出中"
        assert 'industry_id' not in result.columns, "旧字段 industry_id 不应出现在输出中"

    def test_sw_industry_name_values(self, mock_features_df, mock_shenwan_industry_clean):
        """sw_industry 应包含正确的二级行业名称"""
        builder = FeatureBuilder()
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean)

        bank_stocks = result[result['ts_code'].isin(['000001.SZ', '000002.SZ'])]
        assert (bank_stocks['sw_industry'] == '国有大型银行').all(), "银行股的 sw_industry 应为 '国有大型银行'"

        chem_stocks = result[result['ts_code'].isin(['000006.SZ', '000007.SZ'])]
        assert (chem_stocks['sw_industry'] == '化学原料').all(), "化工股的 sw_industry 应为 '化学原料'"

    def test_sw_industry_id_stable(self, mock_features_df, mock_shenwan_industry_clean):
        """sw_industry_id 编码应稳定：相同名称始终映射到相同整数"""
        builder = FeatureBuilder()
        result1 = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean)
        result2 = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean)

        # 两次调用结果相同
        assert (result1['sw_industry_id'] == result2['sw_industry_id']).all()

        # 同一行业内 id 相同
        ids_bank = result1[result1['sw_industry'] == '国有大型银行']['sw_industry_id'].unique()
        ids_chem = result1[result1['sw_industry'] == '化学原料']['sw_industry_id'].unique()
        assert len(ids_bank) == 1, "同一行业内 sw_industry_id 应唯一"
        assert len(ids_chem) == 1, "同一行业内 sw_industry_id 应唯一"

        # 不同行业 id 不同
        assert ids_bank[0] != ids_chem[0], "不同行业的 sw_industry_id 应不同"

    def test_no_shenwan_data_returns_original(self, mock_features_df):
        """当申万行业数据为空时，返回原始 DataFrame，不抛出异常"""
        builder = FeatureBuilder()
        result = builder._merge_shenwan_industry(mock_features_df, pd.DataFrame())
        assert len(result) == len(mock_features_df)
        assert 'sw_industry' not in result.columns

    def test_missing_required_columns_returns_original(self, mock_features_df):
        """当申万行业数据缺少必要字段时，返回原始 DataFrame"""
        builder = FeatureBuilder()
        bad_sw = pd.DataFrame({'ts_code': ['000001.SZ']})  # 只有 ts_code，缺少 sw_code/sw_name
        result = builder._merge_shenwan_industry(mock_features_df, bad_sw)
        assert 'sw_industry' not in result.columns


# ---------------------------------------------------------------------------
# 2. 测试 _apply_industry_neutralization 使用 sw_industry 列
# ---------------------------------------------------------------------------

class TestApplyIndustryNeutralization:
    """测试 _apply_industry_neutralization 基于 sw_industry 分组"""

    def _make_features_with_industry(self):
        """创建包含 sw_industry 列的特征 DataFrame"""
        return pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'trade_date': ['20230102'] * 10,
            'y_ret_20': [0.05, 0.03, -0.01, 0.02, 0.04,
                         0.08, -0.02, 0.01, 0.06, -0.03],
            'ret_5': [0.01, 0.02, -0.01, 0.03, -0.02,
                      0.05, -0.03, 0.01, 0.02, -0.01],
            'sw_industry': ['国有大型银行'] * 5 + ['化学原料'] * 5,
            'tradable': [1] * 10,
        })

    def test_uses_sw_industry_column(self):
        """中性化应基于 sw_industry 列分组，而不是旧的 sw_name"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = self._make_features_with_industry()
        result = builder._apply_industry_neutralization(features)

        # 应生成去均值列
        assert 'neu_y_ret_20' in result.columns, "缺少 neu_y_ret_20 列"
        assert 'neu_ret_5' in result.columns, "缺少 neu_ret_5 列"

    def test_missing_sw_industry_returns_unchanged(self):
        """当 sw_industry 列不存在时，返回原始 DataFrame（不报错）"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20230102'] * 2,
            'y_ret_20': [0.05, -0.03],
            'sw_name': ['国有大型银行', '化学原料'],  # 旧字段名，不应被使用
            'tradable': [1, 1],
        })
        result = builder._apply_industry_neutralization(features)
        # 没有 sw_industry 列，中性化应跳过，neu_y_ret_20 不应出现
        assert 'neu_y_ret_20' not in result.columns

    def test_demean_within_industry(self):
        """验证行业内去均值后，各行业内均值接近0"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5])
        features = self._make_features_with_industry()
        result = builder._apply_industry_neutralization(features)

        if 'neu_y_ret_20' in result.columns:
            for ind in ['国有大型银行', '化学原料']:
                grp = result[result['sw_industry'] == ind]['neu_y_ret_20']
                assert abs(grp.mean()) < 0.01, f"行业 {ind} 去均值后均值应接近0，实际={grp.mean():.4f}"


# ---------------------------------------------------------------------------
# 3. 测试 DataCleaner.clean_shenwan_industry 默认使用 l2
# ---------------------------------------------------------------------------

class TestCleanShenwanIndustryL2:
    """测试 DataCleaner.clean_shenwan_industry 默认为二级行业"""

    def test_default_level_str_is_l2(self):
        """默认调用应使用 level_str='l2'，成分股数据中必须有 l2_code 字段"""
        cleaner = DataCleaner()

        # 模拟包含 l2_code 的成分股数据（二级）
        raw_index_basic = pd.DataFrame({
            'index_code': ['801011'],
            'industry_name': ['国有大型银行'],
        })
        raw_index_members = {
            '801011': pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'l2_code': ['110101', '110101'],  # 二级代码字段
                'in_date': ['20200101', '20200101'],
            })
        }

        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members)

        assert len(result) == 2
        assert set(result.columns) >= {'ts_code', 'sw_code', 'sw_name'}
        assert (result['sw_name'] == '国有大型银行').all()

    def test_l1_fallback_when_no_l2_code(self):
        """当传入不含 l2_code 的数据时，应跳过该行业（logged warning）"""
        cleaner = DataCleaner()

        raw_index_basic = pd.DataFrame({
            'index_code': ['801011'],
            'industry_name': ['银行'],
        })
        raw_index_members = {
            '801011': pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'l1_code': ['110000'],  # 一级代码字段，默认 l2 时应跳过
            })
        }

        # 默认 level_str='l2'，没有 l2_code 的行业会被跳过
        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members)
        assert len(result) == 0  # 被跳过，返回空表

    def test_explicit_level_str_l1(self):
        """显式传 level_str='l1' 时，应能处理 l1_code 字段"""
        cleaner = DataCleaner()

        raw_index_basic = pd.DataFrame({
            'index_code': ['801010'],
            'industry_name': ['银行'],
        })
        raw_index_members = {
            '801010': pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'l1_code': ['110000', '110000'],
                'in_date': ['20200101', '20200101'],
            })
        }

        result = cleaner.clean_shenwan_industry(
            raw_index_basic, raw_index_members, level_str='l1'
        )

        assert len(result) == 2
        assert (result['sw_name'] == '银行').all()

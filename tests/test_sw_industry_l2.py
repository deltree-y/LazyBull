"""测试申万行业主字段默认使用申万二级，并支持按配置切换。

验证：
- FeatureBuilder._merge_shenwan_industry() 在 L2 配置下以 L2 为主字段输出
    sw_industry / sw_industry_code / sw_industry_id，并保留 L2/L3/L1 层级字段
- FeatureBuilder._merge_shenwan_industry() 支持切换到 L1/L3 主字段
- sw_industry_id 编码稳定（相同名称始终映射到相同整数）
- _apply_industry_neutralization() 使用 sw_industry 进行分组中性化
- DataCleaner.clean_shenwan_industry() 默认 level_str='l3'，旧式 level_str='l2' 向后兼容
"""

import src.lazybull.common.config as config_module
import pandas as pd
import pytest

from src.lazybull.common.config import Config
from src.lazybull.data import DataCleaner
from src.lazybull.features.builder import FeatureBuilder


# ---------------------------------------------------------------------------
# 通用 mock 数据（L3 格式）
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_shenwan_industry_clean_l3():
    """模拟 clean 层申万行业分类表（三级行业，含 L1/L2/L3 字段）"""
    return pd.DataFrame({
        'ts_code': [
            '000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ',
            '000006.SZ', '000007.SZ', '000008.SZ', '000009.SZ', '000010.SZ',
        ],
        'sw_l1_code': ['110000'] * 5 + ['220000'] * 5,
        'sw_l1':      ['银行'] * 5 + ['化工'] * 5,
        'sw_l2_code': ['110100'] * 5 + ['220100'] * 5,
        'sw_l2':      ['国有银行'] * 5 + ['化学原料'] * 5,
        'sw_l3_code': ['110101'] * 5 + ['220101'] * 5,
        'sw_l3':      ['国有大型银行'] * 5 + ['基础化学原料'] * 5,
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
# 1. 测试 _merge_shenwan_industry 输出字段（L2 主字段）
# ---------------------------------------------------------------------------

class TestMergeShenwanIndustry:
    """测试 FeatureBuilder._merge_shenwan_industry 输出字段（L2 主字段）"""

    def test_output_columns_renamed(self, mock_features_df, mock_shenwan_industry_clean_l3):
        """合并后应输出 sw_industry / sw_industry_code / sw_industry_id（映射 L2）。"""
        builder = FeatureBuilder(shenwan_level="l2")
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        # L2 主字段存在
        assert 'sw_industry' in result.columns, "缺少 sw_industry 列"
        assert 'sw_industry_code' in result.columns, "缺少 sw_industry_code 列"
        assert 'sw_industry_id' in result.columns, "缺少 sw_industry_id 列"

        # L2 显式字段存在
        assert 'sw_l2' in result.columns, "缺少 sw_l2 列"
        assert 'sw_l2_code' in result.columns, "缺少 sw_l2_code 列"
        assert 'sw_l2_id' in result.columns, "缺少 sw_l2_id 列"

        # L3 细粒度字段保留
        assert 'sw_l3' in result.columns, "缺少 sw_l3 列"
        assert 'sw_l3_code' in result.columns, "缺少 sw_l3_code 列"

        # L1 辅助字段存在
        assert 'sw_l1' in result.columns, "缺少 sw_l1 列"
        assert 'sw_l1_code' in result.columns, "缺少 sw_l1_code 列"
        assert 'sw_l1_id' in result.columns, "缺少 sw_l1_id 列"

        # 旧字段不应出现
        assert 'sw_name' not in result.columns, "旧字段 sw_name 不应出现"
        assert 'sw_code' not in result.columns, "旧字段 sw_code 不应出现"
        assert 'industry_id' not in result.columns, "旧字段 industry_id 不应出现"

    def test_sw_industry_maps_to_l2(self, mock_features_df, mock_shenwan_industry_clean_l3):
        """sw_industry 应包含 L2 行业名称，L3 保留在 sw_l3 中。"""
        builder = FeatureBuilder(shenwan_level="l2")
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        bank_stocks = result[result['ts_code'].isin(['000001.SZ', '000002.SZ'])]
        assert (bank_stocks['sw_industry'] == '国有银行').all(), "银行股 sw_industry 应为 L2 名称"
        assert (bank_stocks['sw_l3'] == '国有大型银行').all(), "银行股 sw_l3 应保留 L3 名称"
        # L1 应为一级行业名
        assert (bank_stocks['sw_l1'] == '银行').all(), "银行股 sw_l1 应为 '银行'"

    @pytest.mark.parametrize(
        ("level", "expected_industry", "expected_code"),
        [
            ("l1", "银行", "110000"),
            ("l2", "国有银行", "110100"),
            ("l3", "国有大型银行", "110101"),
        ],
    )
    def test_sw_industry_can_switch_levels(
        self,
        mock_features_df,
        mock_shenwan_industry_clean_l3,
        level,
        expected_industry,
        expected_code,
    ):
        """显式指定 shenwan_level 时，主行业字段应映射到对应层级。"""
        builder = FeatureBuilder(shenwan_level=level)
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        bank_stock = result[result['ts_code'] == '000001.SZ'].iloc[0]
        assert bank_stock['sw_industry'] == expected_industry
        assert bank_stock['sw_industry_code'] == expected_code

    def test_default_level_reads_project_config(
        self,
        monkeypatch,
        mock_features_df,
        mock_shenwan_industry_clean_l3,
    ):
        """未显式传参时，应读取项目配置中的 industry.shenwan_level。"""
        config = Config()
        config.set("industry.shenwan_level", "l3")
        monkeypatch.setattr(config_module, "_global_config", config)

        builder = FeatureBuilder()
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        bank_stock = result[result['ts_code'] == '000001.SZ'].iloc[0]
        assert bank_stock['sw_industry'] == '国有大型银行'
        assert bank_stock['sw_industry_code'] == '110101'

    def test_explicit_level_overrides_project_config(
        self,
        monkeypatch,
        mock_features_df,
        mock_shenwan_industry_clean_l3,
    ):
        """显式传入 shenwan_level 时，应优先于项目配置。"""
        config = Config()
        config.set("industry.shenwan_level", "l3")
        monkeypatch.setattr(config_module, "_global_config", config)

        builder = FeatureBuilder(shenwan_level="l1")
        result = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        bank_stock = result[result['ts_code'] == '000001.SZ'].iloc[0]
        assert bank_stock['sw_industry'] == '银行'
        assert bank_stock['sw_industry_code'] == '110000'

    def test_sw_industry_id_stable(self, mock_features_df, mock_shenwan_industry_clean_l3):
        """sw_industry_id 编码应稳定：相同 L2 名称始终映射到相同整数。"""
        builder = FeatureBuilder(shenwan_level="l2")
        result1 = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)
        result2 = builder._merge_shenwan_industry(mock_features_df, mock_shenwan_industry_clean_l3)

        assert (result1['sw_industry_id'] == result2['sw_industry_id']).all()

        ids_bank = result1[result1['sw_industry'] == '国有银行']['sw_industry_id'].unique()
        ids_chem = result1[result1['sw_industry'] == '化学原料']['sw_industry_id'].unique()
        assert len(ids_bank) == 1, "同一 L2 行业内 sw_industry_id 应唯一"
        assert len(ids_chem) == 1, "同一 L2 行业内 sw_industry_id 应唯一"
        assert ids_bank[0] != ids_chem[0], "不同 L2 行业的 sw_industry_id 应不同"

    def test_no_shenwan_data_returns_original(self, mock_features_df):
        """当申万行业数据为空时，返回原始 DataFrame，不抛出异常"""
        builder = FeatureBuilder(shenwan_level="l2")
        result = builder._merge_shenwan_industry(mock_features_df, pd.DataFrame())
        assert len(result) == len(mock_features_df)
        assert 'sw_industry' not in result.columns

    def test_missing_required_columns_returns_original(self, mock_features_df):
        """当申万行业数据缺少必要字段时，返回原始 DataFrame"""
        builder = FeatureBuilder(shenwan_level="l2")
        bad_sw = pd.DataFrame({'ts_code': ['000001.SZ']})  # 只有 ts_code
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
            'sw_industry': ['国有大型银行'] * 5 + ['基础化学原料'] * 5,
            'tradable': [1] * 10,
        })

    def test_uses_sw_industry_column(self):
        """中性化应基于 sw_industry 列分组"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5], shenwan_level="l2")
        features = self._make_features_with_industry()
        result = builder._apply_industry_neutralization(features)

        assert 'neu_y_ret_20' in result.columns, "缺少 neu_y_ret_20 列"
        assert 'neu_ret_5' in result.columns, "缺少 neu_ret_5 列"

    def test_missing_sw_industry_returns_unchanged(self):
        """当 sw_industry 列不存在时，返回原始 DataFrame（不报错）"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5], shenwan_level="l2")
        features = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20230102'] * 2,
            'y_ret_20': [0.05, -0.03],
            'sw_name': ['国有大型银行', '基础化学原料'],  # 旧字段名，不应被使用
            'tradable': [1, 1],
        })
        result = builder._apply_industry_neutralization(features)
        assert 'neu_y_ret_20' not in result.columns

    def test_demean_within_industry(self):
        """验证行业内去均值后，各行业内均值接近0"""
        builder = FeatureBuilder(horizons=[20], lookback_windows=[5], shenwan_level="l2")
        features = self._make_features_with_industry()
        result = builder._apply_industry_neutralization(features)

        if 'neu_y_ret_20' in result.columns:
            for ind in ['国有大型银行', '基础化学原料']:
                grp = result[result['sw_industry'] == ind]['neu_y_ret_20']
                assert abs(grp.mean()) < 0.01, f"行业 {ind} 去均值后均值应接近0，实际={grp.mean():.4f}"


# ---------------------------------------------------------------------------
# 3. 测试 DataCleaner.clean_shenwan_industry L3 默认和 L2 向后兼容
# ---------------------------------------------------------------------------

class TestCleanShenwanIndustryL3:
    """测试 DataCleaner.clean_shenwan_industry 默认三级行业（l3）"""

    def test_default_level_str_is_l3(self):
        """默认调用应使用 level_str='l3'，产出 sw_l1/sw_l2/sw_l3 字段"""
        cleaner = DataCleaner()

        raw_index_basic = pd.DataFrame({
            'index_code': ['110101'],
            'industry_name': ['国有大型银行'],
        })
        raw_index_members = {
            '110101': pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'l1_code': ['110000', '110000'],
                'l1_name': ['银行', '银行'],
                'l2_code': ['110100', '110100'],
                'l2_name': ['国有银行', '国有银行'],
                'l3_code': ['110101', '110101'],
                'l3_name': ['国有大型银行', '国有大型银行'],
                'in_date': ['20200101', '20200101'],
            })
        }

        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members)

        assert len(result) == 2
        required_cols = {'ts_code', 'sw_l1_code', 'sw_l1', 'sw_l2_code', 'sw_l2',
                         'sw_l3_code', 'sw_l3'}
        assert required_cols.issubset(set(result.columns)), (
            f"缺少必要字段，现有: {result.columns.tolist()}"
        )
        assert (result['sw_l3'] == '国有大型银行').all()
        assert (result['sw_l1'] == '银行').all()

    def test_l3_filters_out_stocks(self):
        """L3 清洗时应过滤 out_date 非空的历史成员"""
        cleaner = DataCleaner()
        raw_index_basic = pd.DataFrame({'index_code': ['110101'], 'industry_name': ['国有大型银行']})
        raw_index_members = {
            '110101': pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'l1_code': ['110000', '110000'],
                'l1_name': ['银行', '银行'],
                'l2_code': ['110100', '110100'],
                'l2_name': ['国有银行', '国有银行'],
                'l3_code': ['110101', '110101'],
                'l3_name': ['国有大型银行', '国有大型银行'],
                'out_date': [None, '20221231'],  # 第二只已退出
            })
        }
        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members, level_str='l3')
        assert len(result) == 1
        assert result['ts_code'].iloc[0] == '000001.SZ'

    def test_l3_missing_l3_code_skips_industry(self):
        """L3 模式下缺少 l3_code 字段的行业应被跳过"""
        cleaner = DataCleaner()
        raw_index_basic = pd.DataFrame({'index_code': ['110101'], 'industry_name': ['银行']})
        raw_index_members = {
            '110101': pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'l2_code': ['110100'],  # 有 l2_code 但缺少 l3_code
            })
        }
        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members, level_str='l3')
        assert len(result) == 0

    def test_legacy_l2_still_works(self):
        """level_str='l2' 旧式调用仍应正常工作，产出 sw_code/sw_name"""
        cleaner = DataCleaner()
        raw_index_basic = pd.DataFrame({
            'index_code': ['801011'],
            'industry_name': ['国有大型银行'],
        })
        raw_index_members = {
            '801011': pd.DataFrame({
                'ts_code': ['000001.SZ', '000002.SZ'],
                'l2_code': ['110101', '110101'],
                'in_date': ['20200101', '20200101'],
            })
        }
        result = cleaner.clean_shenwan_industry(raw_index_basic, raw_index_members, level_str='l2')
        assert len(result) == 2
        assert set(result.columns) >= {'ts_code', 'sw_code', 'sw_name'}
        assert (result['sw_name'] == '国有大型银行').all()


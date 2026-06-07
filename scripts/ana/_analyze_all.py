"""综合分析因子相关性、冗余、稳定性"""
import pandas as pd, numpy as np

# 1. 冗余分析
print('='*80)
print('  一、因子冗余分析 (factor_redundancy.csv)')
print('='*80)
try:
    rd = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_redundancy.csv')
    n_total = rd['factor'].nunique()
    n_clusters = rd['cluster_id'].nunique()
    n_drop = (~rd['keep']).sum()
    print(f'因子总数: {n_total}, 独立组: {n_clusters}, 可精简: {n_drop}')
    large = rd.groupby('cluster_id').filter(lambda g: len(g) > 1)
    if len(large) > 0:
        print(f'\n多因子冗余组 ({len(large["cluster_id"].unique())}组):')
        for cid in sorted(large['cluster_id'].unique()):
            g = large[large['cluster_id'] == cid]
            parts = []
            for _,r in g.iterrows():
                mark = '★' if r['keep'] else ' '
                parts.append(f'{mark}{r["factor"]}(corr={r["max_corr"]:.2f})')
            print(f'  组{cid}: {", ".join(parts)}')
    else:
        print('无多因子冗余组')
except Exception as e:
    print(f'读取失败: {e}')

# 2. 稳定性
print()
print('='*80)
print('  二、因子稳定性 (factor_ic_stability.csv)')
print('='*80)
try:
    st = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_ic_stability.csv')
    print(f'评估因子数: {len(st)}')
    print('\nTop 15 最稳定:')
    for _,r in st.head(15).iterrows():
        trend = '↑' if r['icir_trend']>0.05 else ('↓' if r['icir_trend']<-0.05 else '→')
        print(f'  {r["factor"]:35s} score={r["stability_score"]:.3f} mean={r["icir_mean"]:.3f} std={r["icir_std"]:.3f} trend={r["icir_trend"]:+.3f}{trend}')
    print('\n衰减因子 (trend < -0.05):')
    decay = st[st['icir_trend'] < -0.05].sort_values('icir_trend')
    if len(decay) > 0:
        for _,r in decay.head(15).iterrows():
            print(f'  {r["factor"]:35s} trend={r["icir_trend"]:+.3f} mean={r["icir_mean"]:.3f} min={r["icir_min"]:.3f}')
    else:
        print('  无显著衰减因子')
    print('\n改善因子 (trend > +0.05):')
    improve = st[st['icir_trend'] > 0.05].sort_values('icir_trend', ascending=False)
    if len(improve) > 0:
        for _,r in improve.head(10).iterrows():
            print(f'  {r["factor"]:35s} trend={r["icir_trend"]:+.3f} mean={r["icir_mean"]:.3f}')
    else:
        print('  无显著改善因子')
except Exception as e:
    print(f'读取失败: {e}')

# 3. 交叉分析：对比 ICIR 和稳定性
print()
print('='*80)
print('  三、ICIR vs 稳定性 综合评估')
print('='*80)
try:
    ic = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_ic_compare.csv')
    ic20 = ic[(ic['label']=='y_ret_20') & (ic['neutralize']=='ind_size')][['factor','ICIR']]
    
    merged = st.merge(ic20, on='factor', how='inner')
    merged = merged.dropna(subset=['ICIR','stability_score'])
    
    # 好因子：高ICIR + 高稳定性
    merged['composite'] = merged['ICIR'].abs() * 0.6 + merged['stability_score'] * 0.4
    merged = merged.sort_values('composite', ascending=False)
    
    print('\n综合最优 Top 20 (ICIR*0.6 + stability*0.4):')
    for _,r in merged.head(20).iterrows():
        print(f'  {r["factor"]:35s} ICIR={r["ICIR"]:.3f} stab={r["stability_score"]:.3f} composite={r["composite"]:.3f}')
    
    # 高ICIR低稳定性（可能是过拟合）
    merged['rank_icir'] = merged['ICIR'].abs().rank(ascending=False)
    merged['rank_stab'] = merged['stability_score'].rank(ascending=False)
    merged['rank_gap'] = merged['rank_icir'] - merged['rank_stab']
    
    sus = merged.nlargest(10, 'rank_gap')
    print('\n高ICIR但低稳定性（疑似过拟合）:')
    for _,r in sus.iterrows():
        print(f'  {r["factor"]:35s} ICIR={r["ICIR"]:.3f} stab={r["stability_score"]:.3f} gap={r["rank_gap"]:.0f}')
except Exception as e:
    print(f'分析失败: {e}')

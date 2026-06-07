"""全量滚动IC稳定性分析"""
import pandas as pd, numpy as np

st = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_ic_stability.csv')
print(f'评估因子数: {len(st)}')
print()

# Top 20
print('=== Top 20 最稳定 ===')
for _,r in st.head(20).iterrows():
    t = '↑' if r['icir_trend']>0.05 else ('↓' if r['icir_trend']<-0.05 else '→')
    print(f'{r["factor"]:35s} score={r["stability_score"]:7.3f} mean={r["icir_mean"]:7.3f} std={r["icir_std"]:6.3f} trend={r["icir_trend"]:+7.3f}{t}')

# 衰减
print(f'\n=== 衰减因子 (trend < -0.08, 共{(st["icir_trend"]<-0.08).sum()}个) ===')
for _,r in st[st['icir_trend'] < -0.08].sort_values('icir_trend').iterrows():
    print(f'{r["factor"]:35s} trend={r["icir_trend"]:+.3f} mean={r["icir_mean"]:.3f} min={r["icir_min"]:.3f}')

# 改善
print(f'\n=== 改善因子 (trend > +0.10, 共{(st["icir_trend"]>0.10).sum()}个) ===')
for _,r in st[st['icir_trend'] > 0.10].sort_values('icir_trend', ascending=False).iterrows():
    print(f'{r["factor"]:35s} trend={r["icir_trend"]:+.3f} mean={r["icir_mean"]:.3f}')

# 交叉ICIR
print('\n=== 综合评估（结合ICIR） ===')
try:
    ic = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_ic_compare.csv')
    ic20 = ic[(ic['label']=='y_ret_20') & (ic['neutralize']=='ind_size')][['factor','ICIR']]
    merged = st.merge(ic20, on='factor', how='inner').dropna(subset=['ICIR'])
    merged['composite'] = merged['ICIR'].abs()*0.5 + merged['stability_score']*0.5
    merged = merged.sort_values('composite', ascending=False)
    
    print('\n综合最优 Top 15 (ICIR*0.5 + stability*0.5):')
    for _,r in merged.head(15).iterrows():
        print(f'{r["factor"]:35s} ICIR={r["ICIR"]:+.3f} stab={r["stability_score"]:+.3f} composite={r["composite"]:+.3f}')
except Exception as e:
    print(f'跳过: {e}')

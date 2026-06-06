"""快速分析 factor_ic_compare.csv"""
import pandas as pd, numpy as np

df = pd.read_csv('d:/my_pro/LazyBull/data/reports/factor_ic_compare.csv')

# ── 总览 ──
print(f"行数: {len(df)}")
print(f"因子数: {df['factor'].nunique()}")
print(f"标签: {sorted(df['label'].unique())}")
print(f"中性化模式: {sorted(df['neutralize'].unique())}")
print()

# ── ICIR Top 40 ──
print("=" * 95)
print("ICIR Top 40")
print("=" * 95)
top = df.nlargest(40, 'ICIR')
for _, r in top.iterrows():
    print(f"{r['factor']:35s} {r['label']:8s} {r['neutralize']:8s}  IC={r['IC_mean']:.4f}  IR={r['ICIR']:.3f}  win={r['IC_win_rate']:.0%}  n={int(r['n_days_valid'])}")

# ── 按 neutralize 分组的最佳因子 ──
print()
print("=" * 95)
print("各中性化模式下 ICIR > 0.5 的因子数（y_ret_20）")
print("=" * 95)
for mode in ['raw', 'industry', 'size', 'ind_size']:
    sub = df[(df['neutralize'] == mode) & (df['label'] == 'y_ret_20')]
    good = sub[sub['ICIR'] > 0.5]
    print(f"  {mode:10s}: {len(good)} 个 (ICIR>0.5), 最强: {good.iloc[0]['factor'] if len(good)>0 else '无'} IR={good.iloc[0]['ICIR']:.3f}" if len(good)>0 else f"  {mode:10s}: 0 个")

# ── 真正的 Alpha（所有4种模式下 ICIR 均 > 0.3 on y_ret_20）──
print()
print("=" * 95)
print("真正 Alpha 因子：4种中性化模式下 y_ret_20 ICIR 均 > 0.3")
print("=" * 95)
pivot = df[df['label'] == 'y_ret_20'].pivot_table(
    index='factor', columns='neutralize', values='ICIR', aggfunc='first'
)
alpha_mask = (pivot['raw'] > 0.3) & (pivot['industry'] > 0.3) & (pivot['size'] > 0.3) & (pivot['ind_size'] > 0.3)
alpha = pivot[alpha_mask].copy()
if len(alpha) > 0:
    alpha['avg'] = alpha[['raw','industry','size','ind_size']].mean(axis=1)
    alpha = alpha.sort_values('avg', ascending=False)
    for idx, row in alpha.iterrows():
        print(f"  {idx:35s}  raw={row['raw']:.3f}  ind={row['industry']:.3f}  size={row['size']:.3f}  both={row['ind_size']:.3f}")
else:
    print("  无因子在所有模式下 ICIR 均 > 0.3")

# ── 行业暴露最大（raw ICIR 高但 industry 中性化后下降最多）──
print()
print("=" * 95)
print("行业暴露因子：raw IR > 0.5，行业中性化后下降 > 0.3（y_ret_20）")
print("=" * 95)
sub20 = df[df['label'] == 'y_ret_20']
for _, r in sub20.iterrows():
    pass
raw_ir = sub20[sub20['neutralize']=='raw'].set_index('factor')['ICIR']
ind_ir = sub20[sub20['neutralize']=='industry'].set_index('factor')['ICIR']
size_ir = sub20[sub20['neutralize']=='size'].set_index('factor')['ICIR']
both_ir = sub20[sub20['neutralize']=='ind_size'].set_index('factor')['ICIR']

common = raw_ir.dropna().index.intersection(ind_ir.dropna().index)
for f in common:
    drop = raw_ir[f] - ind_ir[f]
    if raw_ir[f] > 0.5 and drop > 0.3:
        print(f"  {f:35s}  raw={raw_ir[f]:.3f} → ind={ind_ir[f]:.3f}  (下降 {drop:.3f})")

# ── 市值暴露最大 ──
print()
print("=" * 95)
print("市值暴露因子：raw IR > 0.3，市值中性化后下降 > 0.3（y_ret_20）")
print("=" * 95)
common2 = raw_ir.dropna().index.intersection(size_ir.dropna().index)
for f in common2:
    drop = raw_ir[f] - size_ir[f]
    if raw_ir[f] > 0.3 and drop > 0.3:
        print(f"  {f:35s}  raw={raw_ir[f]:.3f} → size={size_ir[f]:.3f}  (下降 {drop:.3f})")

# ── 被行业/市值掩盖的因子（raw IR 低，但双重中性化后 IR > 0.3）──
print()
print("=" * 95)
print("被掩盖的因子：raw IR < 0.1，但双重中性化后 IR > 0.3（y_ret_20）")
print("=" * 95)
common3 = raw_ir.dropna().index.intersection(both_ir.dropna().index)
for f in common3:
    if abs(raw_ir[f]) < 0.1 and both_ir[f] > 0.3:
        print(f"  {f:35s}  raw={raw_ir[f]:.3f} → both={both_ir[f]:.3f}")

"""全量因子优化分析：相关性 + 稳定性 + ICIR 三维综合"""
import pandas as pd, numpy as np

# ── 加载数据 ──
st = pd.read_csv("d:/my_pro/LazyBull/data/reports/factor_ic_stability.csv")
ic = pd.read_csv("d:/my_pro/LazyBull/data/reports/factor_ic_compare.csv")
rd = pd.read_csv("d:/my_pro/LazyBull/data/reports/factor_redundancy.csv")

ic20 = ic[(ic["label"]=="y_ret_20") & (ic["neutralize"]=="ind_size")][["factor","ICIR","IC_mean","IC_win_rate"]]

# ── 合并 ──
m = st.merge(ic20, on="factor", how="left")
m = m.dropna(subset=["ICIR"])
m["abs_icir"] = m["ICIR"].abs()
m["composite"] = m["abs_icir"] * 0.4 + m["stability_score"] * 0.4 + m["icir_trend"] * 0.2

print(f"因子总数(稳定性报告): {len(m)}")
print()

# ═══════════════════════════════════════
# 1. 综合最优
# ═══════════════════════════════════════
print("=" * 80)
print("  一、综合最优 Top 25 (ICIR*0.4 + stability*0.4 + trend*0.2)")
print("=" * 80)
top = m.nlargest(25, "composite")
for i, (_, r) in enumerate(top.iterrows()):
    t = "↑" if r["icir_trend"]>0.05 else ("↓" if r["icir_trend"]<-0.05 else "→")
    print(f"  {i+1:2d}. {r['factor']:35s} comp={r['composite']:+.3f}  ICIR={r['ICIR']:+.3f}  stab={r['stability_score']:+.3f}  trend={r['icir_trend']:+.3f}{t}")

# ═══════════════════════════════════════
# 2. 衰减严重但ICIR仍高
# ═══════════════════════════════════════
print()
print("=" * 80)
print("  二、高位回落因子（ICIR>0.3 但 trend<-0.15）— 需警惕")
print("=" * 80)
falling = m[(m["abs_icir"] > 0.3) & (m["icir_trend"] < -0.15)].sort_values("icir_trend")
for _, r in falling.iterrows():
    print(f"  {r['factor']:35s} ICIR={r['ICIR']:+.3f}  trend={r['icir_trend']:+.3f}↓  min={r['icir_min']:+.3f}")

# ═══════════════════════════════════════
# 3. 改善中但ICIR仍低
# ═══════════════════════════════════════
print()
print("=" * 80)
print("  三、潜力因子（ICIR<0.2 但 trend>+0.15）— 观察列表")
print("=" * 80)
rising = m[(m["abs_icir"] < 0.2) & (m["icir_trend"] > 0.15)].sort_values("icir_trend", ascending=False)
for _, r in rising.iterrows():
    print(f"  {r['factor']:35s} ICIR={r['ICIR']:+.3f}  trend={r['icir_trend']:+.3f}↑  stab={r['stability_score']:+.3f}")

# ═══════════════════════════════════════
# 4. 冗余组中该保留但被误杀的
# ═══════════════════════════════════════
print()
print("=" * 80)
print("  四、冗余分析检查：被误杀的好因子")
print("=" * 80)
# 找出被排除但composite很高的因子
keepers = set(rd[rd["keep"]]["factor"])
excluded_rd = set(rd[~rd["keep"]]["factor"])
top_excluded = m[m["factor"].isin(excluded_rd)].nlargest(10, "composite")
for _, r in top_excluded.iterrows():
    if r["composite"] > 0.2:
        print(f"  ⚠ {r['factor']:35s} comp={r['composite']:+.3f} ICIR={r['ICIR']:+.3f} — 冗余分析中标记为排除，建议恢复")

# ═══════════════════════════════════════
# 5. 因子分类汇总
# ═══════════════════════════════════════
print()
print("=" * 80)
print("  五、因子分类统计")
print("=" * 80)
cats = {
    "资金流": ["lg_net_amount","net_mf_amount","elg_net_amount","order_imbalance"],
    "价值": ["bp","pb","pe_ttm","dv_ttm","ep_ttm"],
    "动量/反转": ["ret_","neu_ret_","acceleration","alpha_industry"],
    "波动/形态": ["volatility","amplitude","shadow","body_length","atr","bb_","spec_score"],
    "技术指标": ["rsi","kdj","macd","ma_deviation"],
    "基本面": ["roe","roa","profit","or_yoy","netprofit","q_gr","equity","grossprofit","netprofit_margin","debt_to_assets","current_ratio","quick_ratio","assets_turn","inv_turn","int_to_talcap","cf_sales","cf_nm","ocf_to"],
    "另类": ["holder","forecast","express","fund_hold","fund_count","winner_rate","cost_concentration"],
    "开盘/日内": ["opening_strength","intraday_vol"],
    "市值": ["log_total_mv","log_circ_mv","circ_mv","total_mv","size","list_days"],
    "市场环境": ["mkt_","ind_ret_avg","ind_momentum"],
}
for cat, patterns in cats.items():
    matched = m[m["factor"].apply(lambda f: any(p in f for p in patterns))]
    if len(matched) > 0:
        avg_comp = matched["composite"].mean()
        avg_icir = matched["abs_icir"].mean()
        n_pos = (matched["ICIR"] > 0).sum()
        print(f"  {cat:12s}: {len(matched):3d}个  平均ICIR={avg_icir:.3f}  平均composite={avg_comp:.3f}  正向{n_pos}/{len(matched)}")

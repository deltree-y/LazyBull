"""P2a-T3 新旧逐值一致等价测试（core/decision 四模块复制 + D8 拆分 + 归位改指）。

T0 实施规划 v4 §4-T3 验收口径（仿 T1/T4 等价测试风格）：
- 复制件逐值一致：同一合成输入下，旧模块函数与 v2 复制件输出逐值比对
  （任务卡点名 **Kelly / 归一化数值 fixture** ⇒ ``compute_kelly_weights`` /
  ``cap_and_normalize_weights`` 为重点，含锚定断言防两侧同时错）；
- D8 提取式拆分等价：sizing.compute_kelly_weights（旧 mccabe 15）与
  weight_processor.cap_and_normalize_weights（旧 mccabe 17）在旧路径基线
  在册，复制到新路径 = 新增未豁免硬超限 ⇒ 已按「纯函数提取 + 委托」拆分，
  等价性由本文件新旧逐值断言锁定；
- 归位改指锁定（§3.5 登记表 T3 到期义务）：params / weighting /
  accounting.state 对旧 ``trading.sizing`` / ``portfolio`` 的只读过渡依赖
  随 T3 归位去除——AST 扫描（含函数体内延迟 import）+ 函数对象同一性断言。

不依赖真实配置与真实数据：所有输入为合成 fixture；行业约束的
``get_shenwan_level`` / ``normalize_shenwan_level`` 经 monkeypatch 打为
同一合成 stub（范式同 T4 ``synthetic_cost_settings``，R2-T4-01）。
"""

import ast
import dataclasses
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# ── 旧模块（生产在用，迁移期不修改） ──
from src.lazybull.portfolio import industry_constraint as old_ic
from src.lazybull.portfolio import weight_processor as old_wp
from src.lazybull.trading import buy_plan as old_bp
from src.lazybull.trading import sizing as old_sizing

# ── v2 复制件（P2a-T3 交付） ──
from src.lazybull.v2.core.decision import buy_plan as new_bp
from src.lazybull.v2.core.decision import industry_constraint as new_ic
from src.lazybull.v2.core.decision import sizing as new_sizing
from src.lazybull.v2.core.decision import weight_processor as new_wp

_MISSING = dataclasses.MISSING

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def synthetic_shenwan_config(monkeypatch):
    """合成申万行业口径配置（范式同 T4 synthetic_cost_settings）。

    新旧 ``load_industry_mapping`` 在 ``shenwan_level=None`` 时读取共享
    配置单例 ``get_shenwan_level()``。本 fixture 把新旧两侧行业约束模块
    各自绑定的 ``get_shenwan_level`` / ``normalize_shenwan_level`` 打为
    同一合成 stub（默认口径 l2；normalize 大小写归一），pytest 结束自动
    恢复；返回读取计数供锚定断言。
    """
    reads: List[int] = []
    normalize_calls: List[int] = []

    def _get_level_stub() -> str:
        reads.append(1)
        return "l2"

    def _normalize_stub(level: str) -> str:
        normalize_calls.append(1)
        mapping = {"l1": "l1", "l2": "l2", "l3": "l3", "L1": "l1", "L2": "l2", "L3": "l3"}
        return mapping.get(str(level), "l1")

    for mod in (old_ic, new_ic):
        monkeypatch.setattr(mod, "get_shenwan_level", _get_level_stub)
        monkeypatch.setattr(mod, "normalize_shenwan_level", _normalize_stub)
    return {"default_level": "l2", "reads": reads, "normalize_calls": normalize_calls}


def _assert_ordered_items_equal(old_dict, new_dict, what: str = "字典") -> None:
    """键值相等 + **遍历顺序**一致（R2-T3-01 处置：dict == 不比较插入序，
    返回字典倒序漂移必须被拒绝——迁移冻结含遍历顺序）。"""
    assert list(old_dict.items()) == list(new_dict.items()), f"{what} 键序/键值不一致"


def _assert_dataclass_fields_equal(old_cls, new_cls) -> None:
    """dataclass 字段集 + 默认值逐项比对（同 T1/T4 口径）。"""
    old_fields = old_cls.__dataclass_fields__
    new_fields = new_cls.__dataclass_fields__
    assert list(old_fields) == list(new_fields), "字段集或顺序不一致"
    for name in old_fields:
        of, nf = old_fields[name], new_fields[name]
        assert of.type == nf.type, f"字段 {name} 类型不一致"
        if of.default is _MISSING:
            assert nf.default is _MISSING, f"字段 {name} 默认值缺失状态不一致"
        else:
            assert of.default == nf.default, f"字段 {name} 默认值不一致"
        if of.default_factory is _MISSING:  # type: ignore[misc]
            assert nf.default_factory is _MISSING, (  # type: ignore[misc]
                f"字段 {name} default_factory 缺失状态不一致"
            )
        else:
            same_factory = of.default_factory() == nf.default_factory()  # type: ignore[misc]
            assert same_factory, f"字段 {name} default_factory 产物不一致"


class _PassedCounter:
    """inner pytest 运行的通过数记录器（复收证据：复跑非空跑）。"""

    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.outcome == "passed":
            self.passed += 1


# ══════════════════ core/decision/sizing（D8 拆分件重点） ══════════════════


def test_compute_lot_shares_cases():
    """整手股数：正常/零/负/缺省参数逐值一致。"""
    cases = [
        (10000, 10.5, 100),  # int(10000/10.5/100)*100 = 900
        (10000, 10.0, 100),
        (999, 10.0, 100),  # int(99.9/100)=0
        (0, 10.0, 100),
        (-500, 10.0, 100),
        (10000, 0.0, 100),
        (10000, -3.0, 100),
        (10000, 10.0, 0),
        (None, 10.0, 100),  # None 兜底为 0
        (10000, None, 100),
        (12345.678, 8.88, 200),  # 非整手 lot_size
    ]
    for budget, price, lot in cases:
        old_v = old_sizing.compute_lot_shares(budget, price, lot)
        new_v = new_sizing.compute_lot_shares(budget, price, lot)
        assert old_v == new_v, f"compute_lot_shares({budget}, {price}, {lot}) 不一致"
    # 锚定（防两侧同时错）：整手向下取整语义
    assert new_sizing.compute_lot_shares(10000, 10.5, 100) == 900
    assert new_sizing.compute_lot_shares(999, 10.0, 100) == 0


def test_estimate_variance_from_prices_cases():
    """方差估计：正常序列/不足/非有限过滤逐值一致。"""
    # 等比序列 ⇒ 对数收益为常数 ⇒ 方差恒 0
    geo = 100.0 * np.power(1.01, np.arange(15))
    # 带波动序列（确定性构造）
    rng_vals = [10.0, 10.2, 9.8, 10.5, 10.1, 9.9, 10.4, 10.3, 9.7, 10.6, 10.0, 9.8]
    cases = [
        geo,
        np.array(rng_vals),
        np.arange(1, 10, dtype=float),  # 9 点 → None
        np.arange(1, 15, dtype=float),
        np.array([np.nan] * 3 + rng_vals),  # NaN 过滤后 12 点有效
        np.array(rng_vals + [0.0, -5.0]),  # 非正价过滤
        np.array([5.0] * 10),  # 常数价格 → 方差 0
        np.array([]),
    ]
    for prices in cases:
        old_v = old_sizing.estimate_variance_from_prices(prices)
        new_v = new_sizing.estimate_variance_from_prices(prices)
        assert old_v == new_v, f"estimate_variance_from_prices 不一致: {prices}"
    # 锚定：等比序列对数收益为常数（方差 ≈ 0，容浮点 ulp 残差）；9 点返回 None
    assert new_sizing.estimate_variance_from_prices(geo) == pytest.approx(0.0, abs=1e-24)
    assert new_sizing.estimate_variance_from_prices(np.arange(1, 10)) is None
    # 锚定：混合序列与直接 numpy 口径一致
    arr = np.array(rng_vals)
    expected = float(np.var(np.diff(np.log(arr))))
    assert new_sizing.estimate_variance_from_prices(arr) == pytest.approx(expected)


def test_compute_kelly_weights_numeric_fixture():
    """Kelly 数值 fixture（任务卡点名）：5 股全正分数 + 显式方差，逐值一致。

    锚定：f* = score_rank / σ² ⇒ s4（分数第 2 低但方差最小）权重最大、
    s3（方差最大）权重最小；权重和为 1。
    """
    signals = {"s1": 0.9, "s2": 0.7, "s3": 0.5, "s4": 0.3, "s5": 0.1}
    variances = {"s1": 0.04, "s2": 0.01, "s3": 0.09, "s4": 0.0025, "s5": 0.0225}
    calls_old: List[str] = []
    calls_new: List[str] = []

    def variance_fn_old(s):
        calls_old.append(s)
        return variances.get(s)

    def variance_fn_new(s):
        calls_new.append(s)
        return variances.get(s)

    old_w, old_fb = old_sizing.compute_kelly_weights(signals, variance_fn_old)
    new_w, new_fb = new_sizing.compute_kelly_weights(signals, variance_fn_new)
    assert old_w == new_w, "Kelly 权重不一致（D8 拆分引入漂移）"
    _assert_ordered_items_equal(old_w, new_w, "Kelly 权重")
    assert old_fb == new_fb == 0
    # 方差回调调用序列一致（R2-T3-01：锁定 positive_stocks/raw_kelly 构建顺序）
    assert calls_old == calls_new, "方差回调调用序列不一致"
    # 锚定断言（防两侧同时错）
    assert sum(new_w.values()) == pytest.approx(1.0)
    assert max(new_w, key=new_w.get) == "s4"  # 0.4/0.0025 = 160 最大
    assert min(new_w, key=new_w.get) == "s3"  # 0.6/0.09 最小
    assert new_w["s4"] == pytest.approx(160.0 / 280.55555555555554, rel=1e-9)


def test_compute_kelly_weights_empty_and_all_nonpositive():
    """空信号与全非正分数：等权回退逐值一致。"""
    # 空信号
    assert (
        old_sizing.compute_kelly_weights({}, lambda s: 0.01)
        == new_sizing.compute_kelly_weights({}, lambda s: 0.01)
        == ({}, 0)
    )
    # 全非正分数（0 与负）⇒ 等权，fallback=0
    signals = {"a": 0.0, "b": -1.5, "c": -0.001}
    old_w, old_fb = old_sizing.compute_kelly_weights(signals, lambda s: 0.01)
    new_w, new_fb = new_sizing.compute_kelly_weights(signals, lambda s: 0.01)
    assert old_w == new_w
    _assert_ordered_items_equal(old_w, new_w, "等权回退权重")
    assert old_fb == new_fb == 0
    assert new_w == {
        "a": pytest.approx(1 / 3),
        "b": pytest.approx(1 / 3),
        "c": pytest.approx(1 / 3),
    }


def test_compute_kelly_weights_fallback_and_mixed():
    """方差缺失回退中位数 + 混合正/非正分数（中位 kelly 填充）逐值一致。"""
    # 部分方差 None / 0 / 负值 ⇒ 均进 fallback（vol_sq > 0 判定）
    signals = {"s1": 0.9, "s2": 0.7, "s3": 0.5, "s4": 0.3, "s5": 0.1}
    table = {"s1": 0.04, "s2": 0.01, "s3": 0.09, "s4": 0.0025, "s5": None}

    def variance_fn(s):
        return table.get(s)

    old_w, old_fb = old_sizing.compute_kelly_weights(signals, variance_fn)
    new_w, new_fb = new_sizing.compute_kelly_weights(signals, variance_fn)
    assert old_w == new_w
    _assert_ordered_items_equal(old_w, new_w, "Kelly 权重（fallback）")
    assert old_fb == new_fb == 1
    # 锚定：s5 回退中位 vol_adjust = median{25,100,11.11,400} = 62.5
    assert new_w["s5"] == pytest.approx(0.2 * 62.5 / (25 + 80 + 6.666666666666667 + 160 + 12.5))

    # 方差全 None ⇒ vol_adjusts 空 ⇒ median_vol_adj=1.0，全部回退
    old_w2, old_fb2 = old_sizing.compute_kelly_weights(signals, lambda s: None)
    new_w2, new_fb2 = new_sizing.compute_kelly_weights(signals, lambda s: None)
    assert old_w2 == new_w2
    assert old_fb2 == new_fb2 == 5

    # 混合正/非正：非正股票拿中位 kelly（且互相相等）
    mixed = {"s1": 0.9, "s2": 0.7, "s3": 0.5, "s6": -0.2, "s7": 0.0}
    old_w3, old_fb3 = old_sizing.compute_kelly_weights(
        mixed, lambda s: {"s1": 0.04, "s2": 0.01, "s3": 0.09}.get(s)
    )
    new_w3, new_fb3 = new_sizing.compute_kelly_weights(
        mixed, lambda s: {"s1": 0.04, "s2": 0.01, "s3": 0.09}.get(s)
    )
    assert old_w3 == new_w3
    _assert_ordered_items_equal(old_w3, new_w3, "Kelly 权重（混合非正）")
    assert old_fb3 == new_fb3 == 0
    assert new_w3["s6"] == new_w3["s7"]  # 同为中位 kelly
    assert sum(new_w3.values()) == pytest.approx(1.0)


def test_compute_kelly_weights_half_and_leverage_cap():
    """半 Kelly 混合与单股杠杆上限迭代逐值一致（D8 拆分的两段提取件）。"""
    signals = {"s1": 0.9, "s2": 0.7, "s3": 0.5, "s4": 0.3, "s5": 0.1}
    variances = {"s1": 0.04, "s2": 0.01, "s3": 0.09, "s4": 0.0025, "s5": 0.0225}
    variance_fn = lambda s: variances.get(s)  # noqa: E731

    # half Kelly
    old_w, old_fb = old_sizing.compute_kelly_weights(signals, variance_fn, half=True)
    new_w, new_fb = new_sizing.compute_kelly_weights(signals, variance_fn, half=True)
    assert old_w == new_w
    _assert_ordered_items_equal(old_w, new_w, "半 Kelly 权重")
    assert old_fb == new_fb == 0
    assert sum(new_w.values()) == pytest.approx(1.0)
    # 半 Kelly 比全 Kelly 更接近等权（s4 权重被稀释）
    full_w, _ = new_sizing.compute_kelly_weights(signals, variance_fn)
    assert abs(new_w["s4"] - 0.2) < abs(full_w["s4"] - 0.2)

    # 单股上限（触发迭代重归一化）。登记旧实现固有行为：capped→重归一
    # 会回抬超限权重，10 轮迭代上限后 max 停在略超 cap 处（0.5703→0.3020），
    # 并不保证严格可行——行为逐字保留，锚定按真实终值断言。
    old_w2, _ = old_sizing.compute_kelly_weights(signals, variance_fn, max_leverage=0.3)
    new_w2, _ = new_sizing.compute_kelly_weights(signals, variance_fn, max_leverage=0.3)
    assert old_w2 == new_w2
    _assert_ordered_items_equal(old_w2, new_w2, "cap 截断权重")
    assert max(new_w2.values()) == pytest.approx(0.30203330932371325, rel=1e-9)
    assert max(new_w2.values()) < max(full_w.values())  # 相对无 cap 显著压低
    assert sum(new_w2.values()) == pytest.approx(1.0)

    # 上限不可行（1 股 cap=0.2 ⇒ 剩余现金）
    single = {"x": 1.0}
    old_w3, _ = old_sizing.compute_kelly_weights(single, lambda s: 0.01, max_leverage=0.2)
    new_w3, _ = new_sizing.compute_kelly_weights(single, lambda s: 0.01, max_leverage=0.2)
    assert old_w3 == new_w3


def test_kelly_leverage_cap_tolerance_boundary():
    """收敛容差 1e-9 边界 fixture（R1-T3-04 处置）：容差细节纳入锁定范围。

    构造：n=2、cap=0.5、同方差（score_rank s1=1.0 / s2=0.5 ⇒ 归一 2/3 与
    1/3，首轮溢出 x1=1/6）。溢出收缩序列 x_{k+1} = 0.5·x_k/(1−x_k) 使
    迭代上限第 10 轮时 x≈2.44e-4，恰落在 (1e-9, 1e-3]：生产 1e-9 容差版
    跑满 10 轮，1e-3 容差版在第 9 轮（x≈9.75e-4 ≤ 1e-3）提前停 ⇒ 输出
    可观测差异（评审量化：随机域 773/3000 样本、最大差 1.7e-3）。
    变异自证：把新侧 ``_apply_leverage_cap`` 换为 1e-3 容差副本，输出必须
    与生产实现不同——「未来误改容差不报警」缺口闭合。
    """
    signals = {"s1": 1.5, "s2": 1.0}
    old_w, _ = old_sizing.compute_kelly_weights(signals, lambda s: 0.01, max_leverage=0.5)
    new_w, _ = new_sizing.compute_kelly_weights(signals, lambda s: 0.01, max_leverage=0.5)
    assert old_w == new_w
    _assert_ordered_items_equal(old_w, new_w, "容差边界 Kelly 权重")
    # 输入确实落在容差边界域：max = cap + 2.44e-4 ∈ (1e-9, 1e-3]
    overflow = max(new_w.values()) - 0.5
    assert 1e-9 < overflow <= 1e-3, f"溢出 {overflow} 不在容差边界域"
    assert overflow == pytest.approx(0.00024402147388968842, rel=1e-9)

    # 变异自证：同源代码仅容差改 1e-3 的副本，必须产生不同输出
    def _apply_leverage_cap_tol(result, max_leverage):
        for _ in range(10):
            capped = {s: min(w, max_leverage) for s, w in result.items()}
            cap_total = sum(capped.values())
            if cap_total <= 0:
                break
            result = {s: w / cap_total for s, w in capped.items()}
            if all(w <= max_leverage + 1e-3 for w in result.values()):
                break
        return result

    original = new_sizing._apply_leverage_cap
    try:
        new_sizing._apply_leverage_cap = _apply_leverage_cap_tol
        w_tol, _ = new_sizing.compute_kelly_weights(signals, lambda s: 0.01, max_leverage=0.5)
    finally:
        new_sizing._apply_leverage_cap = original
    assert w_tol != new_w, "容差 1e-9→1e-3 变异未被捕获（锁定缺口）"
    assert max(abs(w_tol[k] - new_w[k]) for k in new_w) > 1e-4  # 差异非噪声级


def test_compute_kelly_weights_tie_and_unsorted_insertion_order():
    """并列正分 + 非排序插入顺序：输出键序与排名路径锁定（R2-T3-01 输入域补齐）。

    并列分数（s2/s1 同为 0.5）下百分位排名依赖**稳定排序**保持插入相对序
    （s2 先插入 ⇒ 百分位 0.5，s1 后插入 ⇒ 0.75）；非排序插入（最高分 s4
    第二个输入）下输出键序 = 输入键序（非分数序）。两者共同纳入顺序锁定；
    倒序替换包装下本组断言必失败（评审附录实验同型）。
    """
    signals = {"s2": 0.5, "s4": 0.9, "s1": 0.5, "s3": 0.1}  # 乱序插入 + s2/s1 并列
    variances = {"s1": 0.04, "s2": 0.01, "s3": 0.09, "s4": 0.0025}
    calls_old: List[str] = []
    calls_new: List[str] = []

    def variance_fn_old(s):
        calls_old.append(s)
        return variances.get(s)

    def variance_fn_new(s):
        calls_new.append(s)
        return variances.get(s)

    old_w, old_fb = old_sizing.compute_kelly_weights(signals, variance_fn_old)
    new_w, new_fb = new_sizing.compute_kelly_weights(signals, variance_fn_new)
    assert old_w == new_w
    _assert_ordered_items_equal(old_w, new_w, "并列/乱序 Kelly 权重")
    assert old_fb == new_fb == 0
    assert calls_old == calls_new == ["s2", "s4", "s1", "s3"]  # 回调序 = 输入序
    assert list(new_w) == ["s2", "s4", "s1", "s3"]  # 输出键序 = 输入键序（非分数序）
    # 锚定（防两侧同时错）：并列 0.5 下稳定排序保持插入相对序 ⇒ s2 得百分位
    # 0.5、s1 得 0.75；raw = {s2: 50, s4: 400, s1: 18.75, s3: 25/9}（rank ÷ σ²）
    total = 50.0 + 400.0 + 18.75 + 25.0 / 9.0
    assert new_w["s2"] == pytest.approx(50.0 / total)
    assert new_w["s4"] == pytest.approx(400.0 / total)
    assert new_w["s1"] == pytest.approx(18.75 / total)
    assert new_w["s3"] == pytest.approx((25.0 / 9.0) / total)


def test_compute_min_buy_value_threshold_cases():
    """最小买入阈值：正常/非法输入逐值一致。"""
    cases = [
        (1_000_000, 20, 0.1),
        (1_000_000, 20, 0.0),
        (1_000_000, 20, -0.5),
        (0, 20, 0.1),
        (-100.0, 20, 0.1),
        (1_000_000, 0, 0.1),
        (1_000_000, -3, 0.1),
        (None, 20, 0.1),
        (1_234_567.89, 7, 0.33),
    ]
    for assets, count, ratio in cases:
        old_v = old_sizing.compute_min_buy_value_threshold(assets, count, ratio)
        new_v = new_sizing.compute_min_buy_value_threshold(assets, count, ratio)
        assert old_v == new_v, f"min_buy_threshold({assets}, {count}, {ratio}) 不一致"
    assert new_sizing.compute_min_buy_value_threshold(1_000_000, 20, 0.1) == 5000.0


def test_resolve_trim_shares_cases():
    """减仓整手股数：整手取整/剩余不足一手整仓/边界逐值一致。"""
    cases = [
        (1000, 0.5, 100, 500),
        (250, 0.6, 100, 100),  # int(150)//100*100=100；余 150 ≥ 100
        (250, 0.8, 100, 250),  # trim=200，余 50 < 100 ⇒ 整仓
        (150, 0.5, 100, 0),  # trim=0，余 150 ≥ 100 ⇒ 不动
        (50, 0.5, 100, 50),  # 余 50 < 100 ⇒ 整仓（零股语义）
        (1000, 1.0, 100, 1000),
        (1000, 0.0, 100, 0),
        (1000, -0.1, 100, 0),
        (0, 0.5, 100, 0),
        (-100, 0.5, 100, 0),
        (1000, 0.5, 0, 0),
        (1000, 0.5, None, 0),
        (999, 0.333, 100, 300),
    ]
    for total, frac, lot, _ in cases:
        old_v = old_sizing.resolve_trim_shares(total, frac, lot)
        new_v = new_sizing.resolve_trim_shares(total, frac, lot)
        assert old_v == new_v, f"resolve_trim_shares({total}, {frac}, {lot}) 不一致"
    # 逐项锚定
    for total, frac, lot, expected in cases:
        assert (
            new_sizing.resolve_trim_shares(total, frac, lot) == expected
        ), f"锚定失败: resolve_trim_shares({total}, {frac}, {lot}) 应为 {expected}"


# ══════════════════ core/decision/buy_plan（字节级复制件） ══════════════════


def test_slot_match_result_dataclass_fields():
    """SlotMatchResult 字段集 + 默认值逐项一致（同 T1 口径）。"""
    _assert_dataclass_fields_equal(old_bp.SlotMatchResult, new_bp.SlotMatchResult)
    assert old_bp.REASON_ALREADY_BOUGHT == new_bp.REASON_ALREADY_BOUGHT
    assert old_bp.REASON_EXECUTION_FAILED == new_bp.REASON_EXECUTION_FAILED
    assert asdict(old_bp.SlotMatchResult()) == asdict(new_bp.SlotMatchResult())


def test_fill_slots_normal_matching():
    """正常顺位匹配：逐字段一致（filled/unfilled/bought 顺序）。"""
    slots = [{"slot": 0}, {"slot": 1}]
    candidates = ["A", "B", "C"]
    old_r = old_bp.fill_slots_from_candidates(
        slots, candidates, lambda c, s: (True, ""), lambda c, s: True
    )
    new_r = new_bp.fill_slots_from_candidates(
        slots, candidates, lambda c, s: (True, ""), lambda c, s: True
    )
    assert dataclasses.asdict(old_r) == dataclasses.asdict(new_r)
    assert new_r.bought == ["A", "B"]
    assert [f["stock"] for f in new_r.filled] == ["A", "B"]
    assert new_r.unfilled == []


def test_fill_slots_dedup_already_bought():
    """当日去重：候选已被前槽买入 ⇒ on_reject 收 REASON_ALREADY_BOUGHT。"""
    rejects_old: List[tuple] = []
    rejects_new: List[tuple] = []

    def make_eval(codes_ok):
        return lambda c, s: (c in codes_ok, "blocked")

    slots = [{"slot": 0}, {"slot": 1}]
    candidates = ["A", "A2", "B"]

    def run(mod, rejects):
        return mod.fill_slots_from_candidates(
            slots,
            candidates,
            make_eval({"A", "A2", "B"}),
            lambda c, s: c != "A2",  # A2 执行失败 ⇒ 槽 0 顺位到 A
            on_reject=lambda slot, cand, reason: rejects.append((cand, reason)),
        )

    old_r = run(old_bp, rejects_old)
    new_r = run(new_bp, rejects_new)
    assert dataclasses.asdict(old_r) == dataclasses.asdict(new_r)
    assert rejects_old == rejects_new
    assert ("A", new_bp.REASON_ALREADY_BOUGHT) in rejects_new
    assert ("A2", new_bp.REASON_EXECUTION_FAILED) in rejects_new


def test_fill_slots_evaluate_reject_and_exhausted():
    """evaluate 拒绝原因透传 + 槽位候选耗尽进 unfilled。"""
    rejects_old: List[tuple] = []
    rejects_new: List[tuple] = []
    slots = [{"slot": 0}, {"slot": 1}]
    candidates = ["A", "B"]

    def evaluate(c, s):
        return (False, "held") if c == "A" else (True, "")

    def run(mod, rejects):
        return mod.fill_slots_from_candidates(
            slots,
            candidates,
            evaluate,
            lambda c, s: False,  # B 执行恒失败
            on_reject=lambda slot, cand, reason: rejects.append((cand, reason)),
        )

    old_r = run(old_bp, rejects_old)
    new_r = run(new_bp, rejects_new)
    assert dataclasses.asdict(old_r) == dataclasses.asdict(new_r)
    assert rejects_old == rejects_new
    assert new_r.unfilled == [{"slot": 0}, {"slot": 1}]
    assert new_r.bought == []
    # 拒绝原因透传（evaluate 返回值）与执行失败常量均出现
    assert ("A", "held") in rejects_new
    assert ("B", new_bp.REASON_EXECUTION_FAILED) in rejects_new


def test_fill_slots_without_reject_callback():
    """on_reject=None 路径（不炸、结果一致）。"""
    slots = [{"slot": 0}]
    old_r = old_bp.fill_slots_from_candidates(
        slots, ["A"], lambda c, s: (False, "x"), lambda c, s: True
    )
    new_r = new_bp.fill_slots_from_candidates(
        slots, ["A"], lambda c, s: (False, "x"), lambda c, s: True
    )
    assert dataclasses.asdict(old_r) == dataclasses.asdict(new_r)
    assert new_r.unfilled == [{"slot": 0}]


# ══════════════════ core/decision/industry_constraint ══════════════════


def _sample_shenwan_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"],
            "sw_l1": ["银行", "房地产", "银行", "食品饮料"],
            "sw_l2": ["城商行", "地产开发", "股份行", "白酒"],
            "sw_l3": ["A股银行", None, "A股股份行", np.nan],
            "sw_name": ["平安银行", "万科A", "浦发银行", "贵州茅台"],
        }
    )


def test_load_industry_mapping_empty_and_missing_columns(synthetic_shenwan_config):
    """空表 / 缺 ts_code / 缺行业列：返回与异常逐字一致。"""
    # 空表 → {}
    assert (
        old_ic.load_industry_mapping(pd.DataFrame())
        == new_ic.load_industry_mapping(pd.DataFrame())
        == {}
    )
    assert old_ic.load_industry_mapping(None) == new_ic.load_industry_mapping(None) == {}
    # 缺 ts_code
    df_no_ts = pd.DataFrame({"sw_l1": ["银行"]})
    with pytest.raises(ValueError) as old_e:
        old_ic.load_industry_mapping(df_no_ts)
    with pytest.raises(ValueError) as new_e:
        new_ic.load_industry_mapping(df_no_ts)
    assert str(old_e.value) == str(new_e.value)
    # 无任何行业列
    df_no_ind = pd.DataFrame({"ts_code": ["000001.SZ"]})
    with pytest.raises(ValueError) as old_e2:
        old_ic.load_industry_mapping(df_no_ind)
    with pytest.raises(ValueError) as new_e2:
        new_ic.load_industry_mapping(df_no_ind)
    assert str(old_e2.value) == str(new_e2.value)


def test_load_industry_mapping_level_precedence(synthetic_shenwan_config):
    """l1/l2/l3 显式层级：优先列序逐值一致（含 None/NaN → 未知行业）。"""
    df = _sample_shenwan_df()
    for level in ("l1", "l2", "l3"):
        old_m = old_ic.load_industry_mapping(df, verbose=True, shenwan_level=level)
        new_m = new_ic.load_industry_mapping(df, verbose=True, shenwan_level=level)
        assert old_m == new_m, f"level={level} 行业映射不一致"
        _assert_ordered_items_equal(old_m, new_m, f"行业映射(level={level})")
    # 锚定：l1 用 sw_l1；l2 用 sw_l2；l3 用 sw_l3（None/NaN → 未知行业）
    m1 = new_ic.load_industry_mapping(df, shenwan_level="l1")
    assert m1["000001.SZ"] == "银行" and m1["600519.SH"] == "食品饮料"
    m2 = new_ic.load_industry_mapping(df, shenwan_level="l2")
    assert m2["000002.SZ"] == "地产开发"
    m3 = new_ic.load_industry_mapping(df, shenwan_level="l3")
    assert m3["000002.SZ"] == "未知行业" and m3["600519.SH"] == "未知行业"
    assert m3["000001.SZ"] == "A股银行"


def test_load_industry_mapping_config_default_level(synthetic_shenwan_config):
    """shenwan_level=None ⇒ 读配置默认口径（合成 stub = l2），两侧同一 stub。"""
    df = _sample_shenwan_df()
    old_m = old_ic.load_industry_mapping(df)
    new_m = new_ic.load_industry_mapping(df)
    assert old_m == new_m
    _assert_ordered_items_equal(old_m, new_m, "行业映射(配置默认口径)")
    assert synthetic_shenwan_config["reads"], "配置读取未发生（stub 未被消费）"
    assert new_m["000001.SZ"] == "城商行"  # l2 列生效


def test_load_industry_mapping_only_sw_industry_column(synthetic_shenwan_config):
    """仅 sw_industry 列（非 l1/l2/l3 命名）时的回退列序。"""
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "sw_industry": ["银行", ""],  # 空串 → 未知行业
        }
    )
    old_m = old_ic.load_industry_mapping(df, shenwan_level="l1")
    new_m = new_ic.load_industry_mapping(df, shenwan_level="l1")
    assert old_m == new_m
    _assert_ordered_items_equal(old_m, new_m, "行业映射(sw_industry 回退)")
    assert new_m["000001.SZ"] == "银行" and new_m["000002.SZ"] == "未知行业"


def test_apply_industry_constraint_invalid_max():
    """max_per_industry <= 0：异常消息逐字一致。"""
    for bad in (0, -1):
        with pytest.raises(ValueError) as old_e:
            old_ic.apply_industry_constraint([("A", 1.0)], {}, bad, 5)
        with pytest.raises(ValueError) as new_e:
            new_ic.apply_industry_constraint([("A", 1.0)], {}, bad, 5)
        assert str(old_e.value) == str(new_e.value)


def test_apply_industry_constraint_selection():
    """行业约束选股：正常/预占/未知行业归组逐值一致。"""
    ranked = [
        ("000001.SZ", 0.9),
        ("600000.SH", 0.8),  # 银行 ×2
        ("000002.SZ", 0.7),  # 房地产
        ("600519.SH", 0.6),  # 食品饮料
        ("999999.SZ", 0.5),  # 未映射 → 未知行业
    ]
    mapping = {
        "000001.SZ": "银行",
        "600000.SH": "银行",
        "000002.SZ": "房地产",
        "600519.SH": "食品饮料",
    }
    for kwargs in (
        {"max_per_industry": 1, "target_n": 5},  # 银行只取分数高者
        {"max_per_industry": 2, "target_n": 5},
        {"max_per_industry": 2, "target_n": 3, "verbose": True},
        {"max_per_industry": 1, "target_n": 5, "initial_industry_counts": {"银行": 1}},
    ):
        old_sel = old_ic.apply_industry_constraint(ranked, mapping, **kwargs)
        new_sel = new_ic.apply_industry_constraint(ranked, mapping, **kwargs)
        assert old_sel == new_sel, f"kwargs={kwargs} 选择不一致"
    # 锚定：银行上限 1 ⇒ 600000.SH 被跳过、未知行业入组
    sel = new_ic.apply_industry_constraint(ranked, mapping, max_per_industry=1, target_n=5)
    assert [s for s, _ in sel] == ["000001.SZ", "000002.SZ", "600519.SH", "999999.SZ"]
    # 锚定：银行已预占 1 + 上限 1 ⇒ 两只银行股都跳过
    sel2 = new_ic.apply_industry_constraint(
        ranked, mapping, max_per_industry=1, target_n=5, initial_industry_counts={"银行": 1}
    )
    assert "000001.SZ" not in [s for s, _ in sel2]


def test_apply_industry_constraint_edge_cases():
    """空候选 / target_n<=0：返回空列表逐值一致。"""
    assert (
        old_ic.apply_industry_constraint([], {"A": "银行"}, 2, 5)
        == new_ic.apply_industry_constraint([], {"A": "银行"}, 2, 5)
        == []
    )
    assert (
        old_ic.apply_industry_constraint([("A", 1.0)], {"A": "银行"}, 2, 0)
        == new_ic.apply_industry_constraint([("A", 1.0)], {"A": "银行"}, 2, 0)
        == []
    )


# ══════════════════ core/decision/weight_processor（D8 拆分件重点） ══════════════════


def test_cap_and_normalize_invalid_cap():
    """非法上限（<=0 / >1）：异常消息逐字一致（两个函数）。"""
    for bad in (0.0, -0.1, 1.01, 2.0):
        with pytest.raises(ValueError) as old_e:
            old_wp.cap_and_normalize_weights({"A": 0.5}, bad)
        with pytest.raises(ValueError) as new_e:
            new_wp.cap_and_normalize_weights({"A": 0.5}, bad)
        assert str(old_e.value) == str(new_e.value)
        with pytest.raises(ValueError) as old_e2:
            old_wp.resolve_tranche_weight_cap(bad, 0.5)
        with pytest.raises(ValueError) as new_e2:
            new_wp.resolve_tranche_weight_cap(bad, 0.5)
        assert str(old_e2.value) == str(new_e2.value)


def test_cap_and_normalize_empty_and_all_invalid():
    """空字典 / 全 NaN / 全 <=0：返回空字典逐值一致。"""
    cases = [
        {},
        {"A": float("nan"), "B": float("nan")},
        {"A": 0.0, "B": -0.1},
        {"A": float("nan"), "B": 0.0, "C": -1.0},
    ]
    for weights in cases:
        old_v = old_wp.cap_and_normalize_weights(weights, 0.4)
        new_v = new_wp.cap_and_normalize_weights(weights, 0.4)
        assert old_v == new_v == {}, f"weights={weights} 应为空字典"


def test_cap_and_normalize_no_cap_hit():
    """未触顶：按原比例归一（归一化数值 fixture），和为 1。"""
    weights = {"A": 0.3, "B": 0.2, "C": 0.5}
    old_v = old_wp.cap_and_normalize_weights(weights, 0.6)
    new_v = new_wp.cap_and_normalize_weights(weights, 0.6)
    assert old_v == new_v
    _assert_ordered_items_equal(old_v, new_v, "未触顶归一权重")
    assert sum(new_v.values()) == pytest.approx(1.0)
    assert new_v["A"] == pytest.approx(0.3) and new_v["C"] == pytest.approx(0.5)


def test_cap_and_normalize_iterative_cap():
    """触顶迭代：超限股固定上限、剩余按原比例分配（数值 fixture 逐值）。"""
    weights = {"A": 10.0, "B": 1.0, "C": 1.0}  # A 占 10/12 ⇒ 触顶 0.5
    old_v = old_wp.cap_and_normalize_weights(weights, 0.5)
    new_v = new_wp.cap_and_normalize_weights(weights, 0.5)
    assert old_v == new_v, "迭代限权不一致（D8 拆分引入漂移）"
    _assert_ordered_items_equal(old_v, new_v, "迭代限权权重")
    # 锚定：A=0.5，B/C 各分 0.25
    assert new_v["A"] == pytest.approx(0.5)
    assert new_v["B"] == pytest.approx(0.25)
    assert new_v["C"] == pytest.approx(0.25)
    assert sum(new_v.values()) == pytest.approx(1.0)


def test_cap_and_normalize_infeasible_cash():
    """不可行约束：3 股 cap=0.3 ⇒ 权重和 0.9、现金保留 0.1。"""
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    old_v = old_wp.cap_and_normalize_weights(weights, 0.3)
    new_v = new_wp.cap_and_normalize_weights(weights, 0.3)
    assert old_v == new_v
    _assert_ordered_items_equal(old_v, new_v, "不可行限权权重")
    assert all(w == pytest.approx(0.3) for w in new_v.values())
    assert sum(new_v.values()) == pytest.approx(0.9)  # 剩余 10% 保留为现金


def test_cap_and_normalize_nan_mixed_and_verbose():
    """NaN/零负混合过滤 + verbose 日志路径冒烟（日志不进对账门，仅不炸）。"""
    weights = {"A": 0.5, "B": float("nan"), "C": 0.0, "D": -0.2, "E": 0.5}
    old_v = old_wp.cap_and_normalize_weights(weights, 0.6, verbose=True)
    new_v = new_wp.cap_and_normalize_weights(weights, 0.6, verbose=True)
    assert old_v == new_v
    _assert_ordered_items_equal(old_v, new_v, "NaN 混合限权权重")
    assert set(new_v) == {"A", "E"}
    assert sum(new_v.values()) == pytest.approx(1.0)


def test_cap_and_normalize_tie_and_unsorted_insertion_order():
    """并列权重 + 非排序插入顺序：键序与分配额锁定（R2-T3-01 输入域补齐）。

    B/C 并列（同为 1.0）且 C 先于 A 插入；A 占 10/12 触顶 0.5 后，B/C 均分
    剩余 0.5。输出键序 = 输入键序（主函数按 valid_weights 重排），锁定
    「按权重降序输出」类重排漂移。
    """
    weights = {"C": 1.0, "A": 10.0, "B": 1.0}  # 乱序插入 + A 触顶
    old_v = old_wp.cap_and_normalize_weights(weights, 0.5)
    new_v = new_wp.cap_and_normalize_weights(weights, 0.5)
    assert old_v == new_v
    _assert_ordered_items_equal(old_v, new_v, "并列/乱序限权权重")
    assert list(new_v) == ["C", "A", "B"]  # 键序 = 输入键序（非权重序）
    # 锚定：A 固定 0.5；B/C 并列均分剩余 0.5
    assert new_v["A"] == pytest.approx(0.5)
    assert new_v["B"] == pytest.approx(0.25)
    assert new_v["C"] == pytest.approx(0.25)
    assert sum(new_v.values()) == pytest.approx(1.0)


def test_resolve_tranche_weight_cap_cases():
    """批内权重上限换算：正常/边界/非法逐值一致。"""
    cases = [
        (0.1, 0.5, 0.2),
        (0.8, 0.5, 1.0),  # 换算超 1 ⇒ 夹 1.0
        (1.0, 1.0, 1.0),
        (0.25, 1.0, 0.25),
        (0.06, 0.4, 0.15),
    ]
    for cap, frac, _ in cases:
        old_v = old_wp.resolve_tranche_weight_cap(cap, frac)
        new_v = new_wp.resolve_tranche_weight_cap(cap, frac)
        assert old_v == new_v, f"resolve_tranche_weight_cap({cap}, {frac}) 不一致"
    for cap, frac, expected in cases:
        assert new_wp.resolve_tranche_weight_cap(cap, frac) == expected
    # fraction 非法分支消息逐字一致
    for bad_frac in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError) as old_e:
            old_wp.resolve_tranche_weight_cap(0.5, bad_frac)
        with pytest.raises(ValueError) as new_e:
            new_wp.resolve_tranche_weight_cap(0.5, bad_frac)
        assert str(old_e.value) == str(new_e.value)


# ══════════════════ T3 归位改指锁定（§3.5 登记表 T3 到期义务） ══════════════════


# T3 归位后 v2 内禁止再引用的旧模块（精确段匹配，含函数体内延迟 import）
_REPOINTED_LEGACY_PREFIXES = (
    "src.lazybull.trading.sizing",
    "src.lazybull.trading.buy_plan",
    "src.lazybull.portfolio",
)

_REPOINT_SCAN_FILES = sorted(
    list((_REPO_ROOT / "src" / "lazybull" / "v2" / "core" / "decision").glob("*.py"))
    + [_REPO_ROOT / "src" / "lazybull" / "v2" / "core" / "accounting" / "state.py"]
)


def _iter_import_modules(path: Path):
    """收集文件内全部 import 模块名（ast.walk 覆盖函数体内延迟 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.module or ""
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_t3_repoint_no_legacy_imports():
    """归位改指锁定：v2/core/decision 与 accounting/state 无旧 sizing/portfolio import。"""
    offenders: List[str] = []
    for path in _REPOINT_SCAN_FILES:
        for module in _iter_import_modules(path):
            for prefix in _REPOINTED_LEGACY_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    offenders.append(f"{path.name}: {module}")
    assert offenders == [], f"旧模块引用残留（应改指 v2 复制件）: {offenders}"


def test_t3_repoint_same_function_object():
    """归位改指行为锁定：params/weighting 消费的正是 v2 sizing 的同一函数对象。"""
    from src.lazybull.v2.core.decision import params as new_params
    from src.lazybull.v2.core.decision import weighting as new_weighting

    assert new_params.compute_min_buy_value_threshold is new_sizing.compute_min_buy_value_threshold
    assert new_weighting.compute_kelly_weights is new_sizing.compute_kelly_weights
    assert new_weighting.estimate_variance_from_prices is new_sizing.estimate_variance_from_prices


def test_t3_suite_forbids_real_config_read(monkeypatch):
    """永久回归（范式同 T4 R2-T4-01）：禁止读取替身 + 计数断言。

    把新旧两侧行业约束模块绑定的 ``get_shenwan_level`` /
    ``normalize_shenwan_level`` 替换为**立即抛 RuntimeError 的禁止读取
    替身**，在该替身下复跑整个 T3 测试文件（本项除外，防递归）：必须
    全绿、真实配置读取 0 次。任一测试绕过合成 fixture 直接读取真实
    配置 ⇒ 替身抛出 ⇒ 该测试失败。
    """
    real_reads: List[int] = []

    def _reject_configuration_read():
        real_reads.append(1)
        raise RuntimeError("reviewer: unexpected configuration read")

    def _reject_normalize(level):
        real_reads.append(1)
        raise RuntimeError("reviewer: unexpected configuration read")

    monkeypatch.setattr(old_ic, "get_shenwan_level", _reject_configuration_read)
    monkeypatch.setattr(new_ic, "get_shenwan_level", _reject_configuration_read)
    monkeypatch.setattr(old_ic, "normalize_shenwan_level", _reject_normalize)
    monkeypatch.setattr(new_ic, "normalize_shenwan_level", _reject_normalize)

    counter = _PassedCounter()
    exit_code = pytest.main(
        [
            "-q",
            "-k",
            "not test_t3_suite_forbids_real_config_read",
            str(Path(__file__)),
        ],
        plugins=[counter],
    )
    assert exit_code == 0, "禁止真实配置读取下 T3 套件存在失败项"
    assert real_reads == [], f"真实配置读取发生 {len(real_reads)} 次"
    # 复跑非空跑：当前文件 33 项（含本项），inner 应跑 32 项；下限防未来漂移
    assert counter.passed >= 31, f"inner 复跑通过数异常: {counter.passed}"

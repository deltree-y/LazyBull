from scripts.respi.lcd35._context import Optional
from scripts.respi.lcd35.core import *  # noqa: F401,F403
from scripts.respi.lcd35.core import _coerce_float, _format_mmdd, _get_shenwan_levels_mapping


def _normalize_ts_code_key(ts_code: object) -> str:
    """将代码统一为去市场后缀的比较键。"""
    ts_code_str = str(ts_code).strip()
    if not ts_code_str:
        return ""
    return ts_code_str.split('.')[0]


def _value_color(value: float) -> tuple[int, int, int]:
    """正红负绿零浅灰。"""
    if value > 0:
        return COLOR_RED
    if value < 0:
        return COLOR_GREEN
    return COLOR_NEUTRAL

def _derive_intraday_pre_close(price: object, pct_chg: object) -> Optional[float]:
    """在昨收缺失时，依据现价与涨跌幅反推昨收。"""
    price_float = _coerce_float(price)
    pct_float = _coerce_float(pct_chg)
    if (
        price_float is None
        or not np.isfinite(price_float)
        or price_float <= 0
        or pct_float is None
        or not np.isfinite(pct_float)
        or abs(pct_float) > INTRADAY_STOCK_PCT_ABS_LIMIT
    ):
        return None

    ratio = 1.0 + pct_float / 100.0
    if abs(ratio) < 1e-8:
        return None

    pre_close = price_float / ratio
    if not np.isfinite(pre_close) or pre_close <= 0:
        return None
    return pre_close


def _build_industry_panel(snapshot: Optional[dict], mode: str = "cycle") -> Optional[dict]:
    """基于实时持仓快照构建行业统计面板数据。

    mode='cycle'：盘外/持仓周期口径，基于买入成本
    mode='intraday'：盘内当日口径，基于昨收
    """
    if snapshot is None:
        return None

    positions = snapshot.get('positions', {})
    rt_df = snapshot.get('quotes')
    if rt_df is None or rt_df.empty or not positions:
        return None

    # 避免与 data_pipeline/charting 形成顶层循环导入，按需懒加载。
    from scripts.respi.lcd35.charting import _normalize_cycle_price, _normalize_intraday_price

    industry_levels_mapping = _get_shenwan_levels_mapping()
    normalized_position_map: dict[str, object] = {}
    for pos_key, pos in positions.items():
        normalized_key = _normalize_ts_code_key(pos_key)
        if normalized_key:
            normalized_position_map[normalized_key] = pos
        bare_code = str(pos_key).split('.')[0].strip()
        bare_normalized_key = _normalize_ts_code_key(bare_code)
        if bare_normalized_key:
            normalized_position_map.setdefault(bare_normalized_key, pos)

    price_map: dict[str, float] = {}
    pre_close_map: dict[str, float] = {}
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', row.get('ts_code', ''))).strip()
        if not ts_code:
            continue
        pos = positions.get(ts_code)
        normalized_rt_code = _normalize_ts_code_key(ts_code)
        if pos is None and normalized_rt_code:
            pos = normalized_position_map.get(normalized_rt_code)
        if pos is None or getattr(pos, 'buy_price', 0) <= 0:
            continue

        pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
        if pre_close is None or pre_close <= 0:
            pre_close = _derive_intraday_pre_close(
                row.get('PRICE', row.get('price')),
                row.get('PCT_CHG', row.get('pct_chg')),
            )

        if mode == "intraday":
            current_price = _normalize_intraday_price(
                row.get('PRICE', row.get('price')),
                pre_close,
                INTRADAY_STOCK_PCT_ABS_LIMIT,
            )
        else:
            current_price = _normalize_cycle_price(
                row.get('PRICE', row.get('price')),
                pre_close,
                INTRADAY_STOCK_PCT_ABS_LIMIT,
            )
        if current_price is None:
            continue
        price_map[ts_code] = current_price
        if pre_close is not None and pre_close > 0:
            pre_close_map[ts_code] = pre_close

    return _build_industry_panel_from_prices(
        positions,
        price_map,
        industry_levels_mapping,
        pre_close_map=pre_close_map,
        mode=mode,
    )


def _build_industry_panel_from_prices(
    positions: dict,
    price_map: dict[str, float],
    industry_levels_mapping: Optional[dict[str, tuple[str, str, str]]] = None,
    pre_close_map: Optional[dict[str, float]] = None,
    mode: str = "cycle",
) -> Optional[dict]:
    """基于持仓与价格映射构建行业统计面板（按申万 L1 聚合）。"""
    if not positions:
        return None

    if industry_levels_mapping is None:
        industry_levels_mapping = _get_shenwan_levels_mapping()

    industry_stats: dict[str, dict] = {}
    total_positive = 0
    total_negative = 0
    total_pnl_amount = 0.0

    l1_set: set[str] = set()
    l2_set: set[str] = set()
    l3_set: set[str] = set()
    for ts_code in positions.keys():
        level_tuple = industry_levels_mapping.get(ts_code)
        if level_tuple is None:
            continue
        if level_tuple[0]:
            l1_set.add(level_tuple[0])
        if level_tuple[1]:
            l2_set.add(level_tuple[1])
        if level_tuple[2]:
            l3_set.add(level_tuple[2])

    for ts_code, pos in positions.items():
        shares = _coerce_float(getattr(pos, 'shares', 0))
        buy_price = _coerce_float(getattr(pos, 'buy_price', 0))
        if shares is None or buy_price is None or shares <= 0 or buy_price <= 0:
            continue

        current_price = price_map.get(ts_code)
        if current_price is None:
            continue
        pre_close = None if pre_close_map is None else pre_close_map.get(ts_code)
        if mode == "intraday":
            if pre_close is None or pre_close <= 0:
                continue
            base_price = pre_close
        else:
            base_price = buy_price

        pnl_pct = (current_price - base_price) / base_price * 100.0
        pnl_amount = (current_price - base_price) * shares
        total_pnl_amount += pnl_amount

        if pnl_pct >= 0:
            total_positive += 1
        elif pnl_pct < 0:
            total_negative += 1

        level_tuple = industry_levels_mapping.get(ts_code)
        industry_name = level_tuple[0] if level_tuple else '未知行业'
        if not industry_name:
            industry_name = '未知行业'
        item = industry_stats.setdefault(
            industry_name,
            {
                'industry': industry_name,
                'positive_count': 0,
                'negative_count': 0,
                'pnl_amount': 0.0,
            },
        )
        item['pnl_amount'] += pnl_amount
        if pnl_pct >= 0:
            item['positive_count'] += 1
        elif pnl_pct < 0:
            item['negative_count'] += 1

    if not industry_stats:
        return None

    positive_total_pnl_amount = sum(max(float(info['pnl_amount']), 0.0) for info in industry_stats.values())
    negative_total_pnl_amount_abs = sum(max(-float(info['pnl_amount']), 0.0) for info in industry_stats.values())

    industries = []
    contribution_basis = "intraday_total_pnl" if mode == "intraday" else "cycle_total_pnl"
    for _, info in industry_stats.items():
        pnl_amount = float(info['pnl_amount'])
        contribution_ratio = 0.0
        # 贡献比例按正负方向分别归一化：正行业合计 +100%，负行业合计 -100%。
        if pnl_amount > 0 and positive_total_pnl_amount > 1e-8:
            contribution_ratio = pnl_amount / positive_total_pnl_amount * 100.0
        elif pnl_amount < 0 and negative_total_pnl_amount_abs > 1e-8:
            contribution_ratio = -((-pnl_amount) / negative_total_pnl_amount_abs * 100.0)

        industries.append(
            {
                'industry': info['industry'],
                'positive_count': info['positive_count'],
                'negative_count': info['negative_count'],
                'pnl_amount': pnl_amount,
                'contribution_ratio': contribution_ratio,
            }
        )

    industries.sort(key=lambda item: (item['pnl_amount'], item['industry']), reverse=True)

    return {
        'total_positive': total_positive,
        'total_negative': total_negative,
        'position_count': len(positions),
        'total_pnl_amount': total_pnl_amount,
        'contribution_basis': contribution_basis,
        'l1_industry_count': len(l1_set),
        'l2_industry_count': len(l2_set),
        'l3_industry_count': len(l3_set),
        'industries': industries,
    }


def _format_rebalance_status(next_rebalance_date: Optional[str], days_to_rebalance: Optional[int]) -> str:
    """格式化顶部下次调仓文案。"""
    if next_rebalance_date and len(next_rebalance_date) == 8 and next_rebalance_date.isdigit():
        date_str = _format_mmdd(next_rebalance_date)
    else:
        date_str = "--/--"

    if days_to_rebalance is None:
        days_str = "--"
    else:
        days_str = str(max(int(days_to_rebalance), 0))

    return f"下次调仓:{date_str}/剩{days_str}天"
from scripts.respi.lcd35._context import Optional, datetime, timedelta, np, pd
from scripts.respi.lcd35.core import *  # noqa: F401,F403
from scripts.respi.lcd35.charting import *  # noqa: F401,F403


def _normalize_cycle_trade_date(date_value: object) -> Optional[str]:
    """将周期图日期统一规范为 YYYYMMDD 字符串。"""
    text = str(date_value or "").strip().replace("-", "")
    if len(text) < 8:
        return None
    text = text[:8]
    return text if text.isdigit() else None


def _build_cycle_close_map(df: Optional[pd.DataFrame]) -> dict[str, float]:
    """从日线 DataFrame 提取 {trade_date: close} 映射，过滤脏值。"""
    if (
        df is None
        or df.empty
        or 'trade_date' not in df.columns
        or 'close' not in df.columns
    ):
        return {}

    close_map: dict[str, float] = {}
    for trade_date, close in zip(df['trade_date'].tolist(), df['close'].tolist()):
        trade_date_norm = _normalize_cycle_trade_date(trade_date)
        close_float = _coerce_float(close)
        if (
            trade_date_norm is None
            or close_float is None
            or not np.isfinite(close_float)
            or close_float <= 0
        ):
            continue
        close_map[trade_date_norm] = close_float
    return close_map


def _normalize_cycle_positions(positions: dict) -> dict[str, dict[str, float]]:
    """将账户持仓归一化为周期图计算可直接消费的数值结构。"""
    normalized: dict[str, dict[str, float]] = {}
    for ts_code, pos in positions.items():
        shares_float = _coerce_float(getattr(pos, 'shares', 0))
        if shares_float is None or not np.isfinite(shares_float):
            continue
        shares = int(shares_float)
        if shares <= 0:
            continue

        buy_price_float = _coerce_float(getattr(pos, 'buy_price', 0.0))
        if buy_price_float is None or not np.isfinite(buy_price_float) or buy_price_float <= 0:
            buy_price_float = 0.0

        normalized[ts_code] = {
            'shares': shares,
            'buy_price': buy_price_float,
        }
    return normalized


def _fetch_cycle_chart_data() -> Optional[dict]:
    """获取持仓周期内的上证/深证/中证800指数和持仓组合涨跌幅数据。

    基于账户持仓状态 + TuShare daily API 计算每日组合市值，
    不依赖 NAV 记录（NAV 可能不完整）。

    Returns:
        dict: {
            'dates': list[str],          # 交易日期列表
            'index_pct': list[float],    # 上证指数累计涨跌幅(%)
            'shenzhen_pct': list[float], # 深证指数累计涨跌幅(%)
            'csi800_pct': list[float],   # 中证800累计涨跌幅(%)
            'portfolio_pct': list[float] # 持仓组合累计涨跌幅(%)
        }
        None: 数据不可用
    """
    from src.lazybull.paper import PaperStorage
    from src.lazybull.data.tushare_client import TushareClient

    paper_storage = PaperStorage(
        root_path=get_paper_root(), verbose=False
    )

    # 获取上次调仓日期作为周期起点
    rebalance_state = paper_storage.load_rebalance_state()
    if rebalance_state is None:
        return None
    start_date = rebalance_state.get('last_rebalance_date')
    rebalance_freq = rebalance_state.get('rebalance_freq')
    if not start_date:
        return None

    # 获取账户持仓
    account_state = paper_storage.load_account_state()
    if account_state is None or not account_state.positions:
        return None

    positions = account_state.positions  # {ts_code: Position}
    normalized_positions = _normalize_cycle_positions(positions)
    if not normalized_positions:
        _emit_diag("抓周期无有效持仓：持仓数值字段不可用，周期图跳过")
        return None

    cash = account_state.cash
    cash_float = _coerce_float(cash)
    if cash_float is None or not np.isfinite(cash_float):
        cash_float = 0.0
    current_dt = datetime.now()
    today_str = current_dt.strftime("%Y%m%d")
    cache_scope_date = today_str
    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    cycle_cache_key = _build_cycle_chart_cache_key(
        cache_scope_date,
        target_cycle_date,
        start_date,
        rebalance_freq,
        cash,
        positions,
    )
    cached_chart_data = _get_cached_cycle_chart_data(cycle_cache_key, cache_scope_date)
    if cached_chart_data is not None:
        _trace_diag("抓周期命中缓存，跳过网络抓取")
        return cached_chart_data

    fetch_started_at = time.monotonic()
    try:
        with _fetch_network_context():
            client = TushareClient(verbose=False)

            # 上证与深证指数日线（以此确定交易日序列）
            shanghai_df = client.query(
                "index_daily", ts_code=SHANGHAI_INDEX_CODE,
                start_date=start_date, end_date=today_str,
                fields="trade_date,close"
            )
            shenzhen_df = client.query(
                "index_daily", ts_code=SHENZHEN_INDEX_CODE,
                start_date=start_date, end_date=today_str,
                fields="trade_date,close"
            )
            csi800_df = client.query(
                "index_daily", ts_code=CSI800_INDEX_CODE,
                start_date=start_date, end_date=today_str,
                fields="trade_date,close"
            )
        if (
            shanghai_df is None
            or shanghai_df.empty
            or shenzhen_df is None
            or shenzhen_df.empty
            or csi800_df is None
            or csi800_df.empty
        ):
            _emit_diag_once(
                "tushare_index_daily_empty_for_cycle_chart",
                "周期图指数日线缺失（上证/深证/中证800至少一项为空），无法构建周期图",
            )
            return None
        shanghai_df = shanghai_df.sort_values('trade_date').reset_index(drop=True)
        shenzhen_df = shenzhen_df.sort_values('trade_date').reset_index(drop=True)
        csi800_df = csi800_df.sort_values('trade_date').reset_index(drop=True)
        shanghai_close_map = _build_cycle_close_map(shanghai_df)
        shenzhen_close_map = _build_cycle_close_map(shenzhen_df)
        csi800_close_map = _build_cycle_close_map(csi800_df)
        trade_dates = [
            d for d in shanghai_df['trade_date'].map(_normalize_cycle_trade_date).dropna().tolist()
            if d in shenzhen_close_map and d in csi800_close_map
        ]
        if len(trade_dates) < 1:
            return None

        # T0 为信号生成日（调仓日），股票于 T1 才实际买入，跳过 T0 让折线图从 T1 开始
        # 这样 T1 作为起点 = 原点（0%），避免多画一天导致 T1 当日已偏离原点
        if len(trade_dates) > 1 and trade_dates[0] == start_date:
            trade_dates = trade_dates[1:]

        # 逐股获取日线收盘价
        stock_closes: dict[str, dict[str, float]] = {}
        for ts_code in normalized_positions:
            with _fetch_network_context():
                df = client.query(
                    "daily", ts_code=ts_code,
                    start_date=start_date, end_date=today_str,
                    fields="trade_date,close"
                )
            if df is not None and not df.empty:
                stock_closes[ts_code] = _build_cycle_close_map(df)
    except Exception as exc:
        _emit_diag(
            "抓周期失败: "
            f"{type(exc).__name__}: {exc} | "
            f"proxy_bypass={_should_bypass_proxy_for_fetch()}"
        )
        return None

    # 计算每日组合市值
    base_value: Optional[float] = None
    portfolio_pct: list[float] = []
    for d in trade_dates:
        market_value = 0.0
        for ts_code, pos in normalized_positions.items():
            closes = stock_closes.get(ts_code, {})
            price = closes.get(d, pos['buy_price'])  # 停牌等无数据时回退到买入价
            price_float = _coerce_float(price)
            if price_float is None or not np.isfinite(price_float) or price_float <= 0:
                continue
            market_value += price_float * pos['shares']
        total_value = market_value + cash_float
        if base_value is None:
            base_value = total_value
        if base_value is None or abs(base_value) < 1e-12:
            portfolio_pct.append(0.0)
        else:
            portfolio_pct.append((total_value / base_value - 1) * 100)

    # 上证/深证指数涨跌幅
    shanghai_base_close = shanghai_close_map[trade_dates[0]]
    shenzhen_base_close = shenzhen_close_map[trade_dates[0]]
    csi800_base_close = csi800_close_map[trade_dates[0]]
    index_pct = [(shanghai_close_map[d] / shanghai_base_close - 1) * 100 for d in trade_dates]
    shenzhen_pct = [(shenzhen_close_map[d] / shenzhen_base_close - 1) * 100 for d in trade_dates]
    csi800_pct = [(csi800_close_map[d] / csi800_base_close - 1) * 100 for d in trade_dates]

    chart_data = _build_cycle_chart_payload(
        dates=trade_dates,
        index_pct=index_pct,
        shenzhen_pct=shenzhen_pct,
        csi800_pct=csi800_pct,
        portfolio_pct=portfolio_pct,
        rebalance_freq=rebalance_freq,
        base_value=base_value,
    )

    if chart_data is not None and (
        target_cycle_date is None or _has_cycle_data_for_target(chart_data, target_cycle_date)
    ):
        _save_cycle_chart_data_cache(cycle_cache_key, cache_scope_date, chart_data)

    _trace_diag(
        "抓周期完成: "
        f"trade_dates={len(trade_dates)}, stocks={len(normalized_positions)}, "
        f"cost={time.monotonic() - fetch_started_at:.2f}s"
    )

    return chart_data


# ---------- 个股盈亏排名 ----------

def _normalize_stock_codes(ts_codes: list[str]) -> set[str]:
    """将 ts_code 列表标准化为 6 位股票代码集合（过滤指数等非股票代码）。"""
    return {
        str(ts_code).split('.')[0].strip()
        for ts_code in ts_codes
        if str(ts_code).strip() and str(ts_code).split('.')[0].strip().isdigit()
    }


def _normalize_ts_code_key(ts_code: object) -> str:
    """将代码归一化为大写 ts_code（如 600000.SH），用于持仓与快照匹配。"""
    text = str(ts_code or '').strip().upper()
    if not text:
        return ''
    if '.' in text:
        code, suffix = text.split('.', 1)
        code = code.strip()
        suffix = suffix.strip()
        if code and suffix:
            return f"{code}.{suffix}"
        return ''
    if not text.isdigit():
        return ''
    if text.startswith('6'):
        return f"{text}.SH"
    if text.startswith(('0', '3')):
        return f"{text}.SZ"
    if text.startswith(('8', '4')):
        return f"{text}.BJ"
    return text


def _extract_stock_code6(value: object) -> str:
    """从多种代码口径中提取 6 位股票代码（如 sh600000、1.600000、600000.SH）。"""
    text = str(value or '').strip().upper()
    if not text:
        return ''

    if text.startswith(('SH', 'SZ', 'BJ')) and len(text) >= 8:
        candidate = text[2:8]
        if candidate.isdigit():
            return candidate

    if '.' in text:
        parts = [part.strip() for part in text.split('.') if part.strip()]
        for part in reversed(parts):
            if len(part) == 6 and part.isdigit():
                return part

    if len(text) == 6 and text.isdigit():
        return text

    if len(text) > 6:
        tail = text[-6:]
        if tail.isdigit():
            return tail

    return ''


def _fetch_realtime_quotes_efinance(ts_codes: list[str]) -> Optional[pd.DataFrame]:
    """使用 efinance 获取持仓实时行情，返回统一字段 DataFrame。"""
    if not ts_codes:
        return None

    stock_codes = _normalize_stock_codes(ts_codes)
    if not stock_codes:
        _trace_diag("E快照跳过: 传入代码为空或均非股票代码")
        return None

    fetch_started_at = time.monotonic()
    _trace_diag(f"E快照开始: req_stocks={len(stock_codes)}")
    _trace_diag(f"E快照请求代码: stock_codes={sorted(stock_codes)}")

    try:
        import efinance as ef  # type: ignore
    except Exception:
        _emit_diag_once(
            "efinance_holdings_import_error",
            "efinance导入失败，无法作为实时快照主来源",
            stderr=False,
        )
        return None

    # 为 efinance 配置自定义 Session，增强网络稳定性
    _configure_efinance_session()

    df = None
    last_exc: Optional[Exception] = None
    total_attempts = EFINANCE_RETRY_COUNT + 1
    for attempt_idx in range(total_attempts):
        attempt_no = attempt_idx + 1
        try:
            with _fetch_network_context():
                _trace_diag(
                    f"E快照尝试: attempt={attempt_no}/{total_attempts}, "
                    f"timeout={EFINANCE_CONNECT_TIMEOUT_SECONDS}s+{EFINANCE_READ_TIMEOUT_SECONDS}s"
                )
                df = ef.stock.get_latest_quote(sorted(stock_codes))
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            _emit_diag(
                "E快照失败: "
                f"attempt={attempt_no}/{total_attempts}, {type(exc).__name__}: {exc} | "
                f"proxy_bypass={_should_bypass_proxy_for_fetch()}"
            )
            if attempt_idx >= EFINANCE_RETRY_COUNT:
                return None
            retry_wait = max(2.0, float(EFINANCE_RETRY_MIN_INTERVAL_SECONDS))
            _trace_diag(
                "E快照重试等待: "
                f"wait={retry_wait:.1f}s, next_attempt={attempt_no + 1}/{total_attempts}"
            )
            time.sleep(retry_wait)

    if last_exc is not None:
        return None

    if df is None or df.empty:
        _emit_diag_once(
            "efinance_holdings_empty",
            "efinance持仓快照返回空数据，尝试回退AKShare",
            stderr=False,
        )
        return None

    code_col = next((c for c in ('代码', 'symbol', 'ts_code', 'TS_CODE') if c in df.columns), None)
    name_col = next((c for c in ('名称', 'name', 'NAME') if c in df.columns), None)
    price_col = next((c for c in ('最新价', '现价', 'price', 'PRICE') if c in df.columns), None)
    pre_close_col = next((c for c in ('昨收', '昨收价', '昨收盘', 'pre_close', 'PRE_CLOSE') if c in df.columns), None)
    pct_col = next((c for c in ('涨跌幅', 'pct_chg', 'PCT_CHG', '涨跌幅%') if c in df.columns), None)
    time_col = next((c for c in ('更新时间', '时间', 'time', 'TIME') if c in df.columns), None)

    if code_col is None or price_col is None:
        _emit_diag_once(
            "efinance_holdings_columns_missing",
            f"efinance持仓快照缺少关键列，当前列: {list(df.columns)}",
            stderr=False,
        )
        return None

    now_time = datetime.now().strftime("%H:%M:%S")
    rows = []
    for _, row in df.iterrows():
        code = str(row.get(code_col, '')).strip()
        if code not in stock_codes:
            continue
        ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        time_text = str(row.get(time_col, now_time)) if time_col else now_time
        if " " in time_text:
            time_text = time_text.split(" ")[-1]
        rows.append(
            {
                'TS_CODE': ts_code,
                'NAME': str(row.get(name_col, '')) if name_col else '',
                'PRICE': row.get(price_col),
                'PRE_CLOSE': row.get(pre_close_col) if pre_close_col else None,
                'PCT_CHG': row.get(pct_col) if pct_col else None,
                'TIME': time_text,
            }
        )

    if not rows:
        _emit_diag(
            "E快照为空: "
            f"req_stocks={len(stock_codes)}，接口返回行命中0，可能是代码映射或接口返回口径变化"
        )
        return None

    _trace_diag(
        "E快照成功: "
        f"hit_rows={len(rows)}, cost={time.monotonic() - fetch_started_at:.2f}s"
    )
    return pd.DataFrame(rows)


def _fetch_realtime_quotes_akshare(ts_codes: list[str]) -> Optional[pd.DataFrame]:
    """使用 AKShare 获取持仓实时行情，返回统一字段 DataFrame。"""
    if not ts_codes:
        return None

    stock_codes = _normalize_stock_codes(ts_codes)
    if not stock_codes:
        _trace_diag("AK快照跳过: 传入代码为空或均非股票代码")
        return None

    fetch_started_at = time.monotonic()
    _trace_diag(f"AK快照开始: req_stocks={len(stock_codes)}")

    try:
        import akshare as ak  # type: ignore
    except Exception:
        _emit_diag_once(
            "akshare_holdings_import_error",
            "AKShare导入失败，无法作为实时快照回退来源",
            stderr=False,
        )
        return None

    getter_candidates: list[tuple[str, object]] = []
    for getter_name in ("stock_zh_a_spot", "stock_zh_a_spot_em"):
        getter = getattr(ak, getter_name, None)
        if getter is not None:
            getter_candidates.append((getter_name, getter))

    if not getter_candidates:
        _emit_diag_once(
            "akshare_holdings_getter_missing",
            "AKShare缺少 stock_zh_a_spot_em/stock_zh_a_spot，无法回退持仓快照",
            stderr=False,
        )
        return None

    df = None
    for getter_name, getter in getter_candidates:
        _trace_diag(f"AK快照尝试接口: {getter_name}")
        if getter_name == "stock_zh_a_spot_em":
            _emit_diag_once(
                "akshare_holdings_spot_em_progress_hint",
                "AKShare stock_zh_a_spot_em 为全市场分页抓取，进度条如 24/58 中的 58 表示总分页数",
                stderr=False,
            )
        try:
            with _fetch_network_context():
                df = getter()
        except Exception as exc:
            _emit_diag(
                "AK快照失败: "
                f"api={getter_name}, {type(exc).__name__}: {exc} | "
                f"proxy_bypass={_should_bypass_proxy_for_fetch()}"
            )
            df = None
            continue

        if df is None or df.empty:
            _emit_diag(f"AK快照空返回: api={getter_name}")
            df = None
            continue

        _trace_diag(f"AK快照接口命中: api={getter_name}, rows={len(df)}")
        break

    if df is None or df.empty:
        _emit_diag_once(
            "akshare_holdings_empty",
            "AKShare持仓快照返回空数据，无法回退实时行情",
            stderr=False,
        )
        return None

    code_col = next((c for c in ('代码', 'symbol', 'ts_code', 'TS_CODE') if c in df.columns), None)
    name_col = next((c for c in ('名称', 'name', 'NAME') if c in df.columns), None)
    price_col = next((c for c in ('最新价', '现价', 'price', 'PRICE') if c in df.columns), None)
    pre_close_col = next((c for c in ('昨收', '昨收价', 'pre_close', 'PRE_CLOSE') if c in df.columns), None)
    pct_col = next((c for c in ('涨跌幅', 'pct_chg', 'PCT_CHG', '涨跌幅%') if c in df.columns), None)
    time_col = next((c for c in ('时间', '更新时间', 'time', 'TIME') if c in df.columns), None)

    if code_col is None or price_col is None:
        _emit_diag_once(
            "akshare_holdings_columns_missing",
            f"AKShare持仓快照缺少关键列，当前列: {list(df.columns)}",
            stderr=False,
        )
        return None

    now_time = datetime.now().strftime("%H:%M:%S")
    rows = []
    unmatched_samples: list[str] = []
    raw_code_samples: list[str] = []
    for _, row in df.iterrows():
        raw_code = str(row.get(code_col, '')).strip()
        code = _extract_stock_code6(raw_code)
        if raw_code and len(raw_code_samples) < 5:
            raw_code_samples.append(raw_code)
        if code not in stock_codes:
            if raw_code and len(unmatched_samples) < 5:
                unmatched_samples.append(f"{raw_code}->{code or '-'}")
            continue
        ts_code = _normalize_ts_code_key(code)
        if not ts_code:
            continue
        rows.append(
            {
                'TS_CODE': ts_code,
                'NAME': str(row.get(name_col, '')) if name_col else '',
                'PRICE': row.get(price_col),
                'PRE_CLOSE': row.get(pre_close_col) if pre_close_col else None,
                'PCT_CHG': row.get(pct_col) if pct_col else None,
                'TIME': str(row.get(time_col, now_time)) if time_col else now_time,
            }
        )

    if not rows:
        _emit_diag(
            "AK快照为空: "
            f"req_stocks={len(stock_codes)}，接口返回行命中0，可能是代码映射或接口返回口径变化 | "
            f"code_col={code_col}, sample_raw_codes={raw_code_samples}, "
            f"sample_unmatched={unmatched_samples}, sample_req={sorted(stock_codes)[:5]}"
        )
        return None
    _trace_diag(
        "AK快照成功: "
        f"hit_rows={len(rows)}, cost={time.monotonic() - fetch_started_at:.2f}s"
    )
    return pd.DataFrame(rows)


def _fetch_daily_snapshot_from_tushare(
    positions: dict,
    target_trade_date: str,
) -> Optional[dict]:
    """在盘后或非实时报价窗口内，用日线合成持仓快照。"""
    from src.lazybull.data.tushare_client import TushareClient

    if not positions or not target_trade_date:
        return None

    quote_rows: list[dict] = []
    index_pct_map: dict[str, float] = {}
    fallback_time = "15:00:00"

    try:
        with _fetch_network_context():
            client = TushareClient(verbose=False)

            for ts_code, pos in positions.items():
                df = client.query(
                    "daily",
                    ts_code=ts_code,
                    start_date=target_trade_date,
                    end_date=target_trade_date,
                    fields="ts_code,trade_date,close,pre_close,pct_chg",
                )
                close_float: Optional[float] = None
                pre_close_float: Optional[float] = None
                pct_float: float = 0.0

                if df is not None and not df.empty:
                    row = df.sort_values("trade_date").iloc[-1]
                    close_float = _coerce_float(row.get("close"))
                    pre_close_float = _coerce_float(row.get("pre_close"))
                    pct_raw = _coerce_float(row.get("pct_chg"))
                    if pct_raw is not None and np.isfinite(pct_raw):
                        pct_float = float(pct_raw)

                buy_price_float = _coerce_float(getattr(pos, "buy_price", 0.0))
                if (
                    close_float is None
                    or not np.isfinite(close_float)
                    or close_float <= 0
                ):
                    close_float = buy_price_float
                if (
                    pre_close_float is None
                    or not np.isfinite(pre_close_float)
                    or pre_close_float <= 0
                ):
                    pre_close_float = close_float
                if close_float is None or close_float <= 0:
                    continue
                if pre_close_float and pre_close_float > 0:
                    pct_float = (close_float / pre_close_float - 1.0) * 100.0

                quote_rows.append(
                    {
                        "TS_CODE": ts_code,
                        "NAME": str(ts_code).split(".")[0],
                        "PRICE": float(close_float),
                        "PRE_CLOSE": float(pre_close_float),
                        "PCT_CHG": float(pct_float),
                        "TIME": fallback_time,
                    }
                )

            window_start = (
                pd.to_datetime(target_trade_date, format="%Y%m%d") - timedelta(days=7)
            ).strftime("%Y%m%d")
            for index_code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
                df = client.query(
                    "index_daily",
                    ts_code=index_code,
                    start_date=window_start,
                    end_date=target_trade_date,
                    fields="trade_date,close",
                )
                if df is None or df.empty:
                    continue
                df = df.sort_values("trade_date").reset_index(drop=True)
                close_values = [_coerce_float(value) for value in df.get("close", []).tolist()]
                close_values = [
                    value
                    for value in close_values
                    if value is not None and np.isfinite(value) and value > 0
                ]
                if not close_values:
                    continue
                last_close = close_values[-1]
                prev_close = close_values[-2] if len(close_values) >= 2 else last_close
                if prev_close is None or prev_close <= 0:
                    pct_float = 0.0
                else:
                    pct_float = (last_close / prev_close - 1.0) * 100.0
                pct_sanitized = _sanitize_intraday_pct(pct_float, INTRADAY_INDEX_PCT_ABS_LIMIT)
                if pct_sanitized is not None:
                    index_pct_map[index_code] = pct_sanitized
    except Exception as exc:
        _emit_diag(
            "日线快照回退失败: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    if not quote_rows:
        return None

    return {
        "quotes": pd.DataFrame(quote_rows),
        "index_pct_map": index_pct_map,
        "trade_date": target_trade_date,
    }


def _build_post_close_daily_snapshot(
    snapshot: dict,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """当实时快照缺失时，尝试用目标交易日收盘数据补全持仓快照。"""
    if not isinstance(snapshot, dict):
        return None

    current_dt = now or datetime.now()
    if _is_realtime_quote_window(current_dt):
        return None

    positions = snapshot.get("positions", {})
    if not positions:
        return None

    target_trade_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    if not target_trade_date:
        return None

    fallback_payload = _fetch_daily_snapshot_from_tushare(positions, target_trade_date)
    if fallback_payload is None:
        return None

    merged_snapshot = dict(snapshot)
    merged_snapshot["quotes"] = fallback_payload["quotes"]
    merged_snapshot["index_pct_map"] = fallback_payload.get("index_pct_map", {})
    merged_snapshot["quote_source"] = "D"
    merged_snapshot["current_date"] = str(fallback_payload.get("trade_date", target_trade_date))
    return merged_snapshot

def _fetch_realtime_holdings_snapshot() -> Optional[dict]:
    """获取当前持仓实时行情快照。"""
    from src.lazybull.paper import PaperStorage

    paper_root = get_paper_root()
    paper_storage = PaperStorage(
        root_path=paper_root, verbose=False
    )
    config = paper_storage.load_config() or {}

    try:
        initial_capital = float(config.get('initial_capital', 500000.0))
    except (TypeError, ValueError):
        initial_capital = 500000.0

    try:
        horizon = int(config.get('horizon', 20))
    except (TypeError, ValueError):
        horizon = 20

    account_state = paper_storage.load_account_state()
    if account_state is None:
        positions = {}
        cash = initial_capital
    else:
        positions = getattr(account_state, 'positions', {}) or {}
        cash = _coerce_float(getattr(account_state, 'cash', initial_capital))
        if cash is None:
            cash = initial_capital

    account_start_date = str(config.get('account_start_date', '') or '').strip()
    if not account_start_date:
        try:
            nav_df = paper_storage.load_all_nav()
            if nav_df is not None and len(nav_df) > 0 and 'trade_date' in nav_df.columns:
                first_trade_date = str(nav_df['trade_date'].iloc[0]).strip()
                if first_trade_date.isdigit() and len(first_trade_date) == 8:
                    account_start_date = first_trade_date
        except Exception:
            account_start_date = ''

    def _annualized_return_from_snapshot(
        initial_capital_value: float,
        current_value: float,
        current_date: str,
    ) -> Optional[float]:
        if current_value <= 0 or initial_capital_value <= 0:
            return 0.0
        if not account_start_date or not current_date:
            return None
        try:
            start_dt = pd.to_datetime(account_start_date, format='%Y%m%d')
            current_dt = pd.to_datetime(str(current_date), format='%Y%m%d')
            days = int((current_dt - start_dt).days)
            if days < 1:
                return 0.0
            total_profit = current_value - initial_capital_value
            return (total_profit / initial_capital_value) * (365.0 / days) * 100
        except Exception:
            return None

    annualized_return_func = _annualized_return_from_snapshot

    snapshot = {
        'positions': positions,
        'cash': cash,
        'initial_capital': initial_capital,
        'current_date': datetime.now().strftime("%Y%m%d"),
        'annualized_return_func': annualized_return_func,
        'quote_source': '-',
        'index_pct_map': {},
        'quotes': None,
    }

    if not positions:
        _trace_diag("抓快照跳过: 当前无持仓")
        return snapshot

    if not _snapshot_fetch_lock.acquire(blocking=False):
        cached_snapshot = _get_cached_holdings_snapshot()
        if cached_snapshot is not None:
            _emit_diag("抓快照请求合并: 上一轮仍在执行，已复用最近缓存快照")
            return cached_snapshot
        _emit_diag("抓快照请求合并: 上一轮仍在执行且暂无缓存可用")
        return snapshot

    try:
        fetch_started_at = time.monotonic()
        rt_df = _fetch_realtime_quotes_efinance(list(positions.keys()))
        quote_source = 'E'

        if rt_df is None or rt_df.empty:
            _emit_diag("抓快照: efinance无可用数据，开始尝试AKShare兜底")
            rt_df = _fetch_realtime_quotes_akshare(list(positions.keys()))
            quote_source = 'A'

        if rt_df is None or rt_df.empty:
            fallback_snapshot = _build_post_close_daily_snapshot(snapshot)
            if fallback_snapshot is not None:
                _set_cached_holdings_snapshot(fallback_snapshot)
                _trace_diag(
                    "抓快照回退成功: "
                    f"source=D, rows={len(fallback_snapshot['quotes'])}, positions={len(positions)}"
                )
                return fallback_snapshot
            _emit_diag(
                "抓快照失败: efinance与AKShare均无可用行情，"
                f"positions={len(positions)}, proxy_bypass={_should_bypass_proxy_for_fetch()}"
            )
            return snapshot

        snapshot['quotes'] = rt_df
        snapshot['quote_source'] = quote_source
        index_pct_map = _extract_index_pct_map_from_quote_df(rt_df)
        cached_index_pct_map = _get_cached_realtime_index_pcts()
        for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
            if code in index_pct_map:
                continue
            pct = _sanitize_intraday_pct(
                cached_index_pct_map.get(code),
                INTRADAY_INDEX_PCT_ABS_LIMIT,
            )
            if pct is not None:
                index_pct_map[code] = pct
        if len(index_pct_map) < 3:
            _trace_diag(
                "快照阶段指数未补全，已切后台刷新: "
                f"cached_codes={sorted(index_pct_map.keys())}"
            )
            _refresh_realtime_index_pcts_async()
        snapshot['index_pct_map'] = index_pct_map
        _set_cached_holdings_snapshot(snapshot)
        _trace_diag(
            "抓快照成功: "
            f"source={quote_source}, rows={len(rt_df)}, positions={len(positions)}, "
            f"cost={time.monotonic() - fetch_started_at:.2f}s"
        )
        return snapshot
    finally:
        _snapshot_fetch_lock.release()


def _build_realtime_portfolio_summary(snapshot: Optional[dict]) -> Optional[dict]:
    """基于实时快照构建持仓摘要，复用已获取的持仓行情。"""
    if snapshot is None:
        return None

    # paper_trade.py 在 scripts/ 目录，用 __file__ 推导绝对路径后插入 sys.path，
    # 确保树莓派上无论从哪个工作目录运行都能正确找到该模块。
    import sys
    import pathlib as _pathlib
    _scripts_dir = str(_pathlib.Path(__file__).parent.parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from paper_trade import build_realtime_portfolio_summary_from_quotes

    cash = _coerce_float(snapshot.get('cash'))
    initial_capital = _coerce_float(snapshot.get('initial_capital'))
    if cash is None or initial_capital is None:
        return None

    annualized_return_func = snapshot.get('annualized_return_func')
    if not callable(annualized_return_func):
        annualized_return_func = None

    return build_realtime_portfolio_summary_from_quotes(
        positions=snapshot.get('positions', {}),
        cash=cash,
        initial_capital=initial_capital,
        current_date=str(snapshot.get('current_date', datetime.now().strftime("%Y%m%d"))),
        rt_df=snapshot.get('quotes'),
        annualized_return_func=annualized_return_func,
    )


def _build_stock_rankings(snapshot: Optional[dict]) -> Optional[list]:
    """基于实时快照构建个股总盈亏排名（按持仓成本）。"""
    if snapshot is None:
        return None

    positions = snapshot.get('positions', {})
    rt_df = snapshot.get('quotes')
    if not positions:
        _emit_diag("排行构建跳过: 持仓为空")
        return None
    if rt_df is None or rt_df.empty:
        _emit_diag("排行构建跳过: 快照行情为空")
        return None

    normalized_position_map: dict[str, object] = {}
    for pos_key, pos in positions.items():
        normalized_key = _normalize_ts_code_key(pos_key)
        if normalized_key:
            normalized_position_map[normalized_key] = pos
        bare_code = str(pos_key).split('.')[0].strip()
        bare_normalized_key = _normalize_ts_code_key(bare_code)
        if bare_normalized_key:
            normalized_position_map.setdefault(bare_normalized_key, pos)

    stocks = []
    rt_total = 0
    rt_valid_code_price = 0
    matched_rows = 0
    pnl_calc_failed = 0  # 新增: 统计 pnl 计算失败的行数
    buy_price_invalid = 0  # 新增: 统计 buy_price 无效的行数
    unmatched_samples: list[str] = []
    failed_pnl_samples: list[tuple] = []  # 新增: 记录计算失败的样本
    sample_rank_inputs: list[tuple] = []
    for _, row in rt_df.iterrows():
        rt_total += 1
        ts_code = str(row.get('TS_CODE', row.get('ts_code', ''))).strip()
        name = str(row.get('NAME', ''))
        price = row.get('PRICE', None)
        if not ts_code or price is None:
            continue
        rt_valid_code_price += 1
        normalized_rt_code = _normalize_ts_code_key(ts_code)
        pos = positions.get(ts_code)
        if pos is None and normalized_rt_code:
            pos = normalized_position_map.get(normalized_rt_code)
        if pos is None:
            if len(unmatched_samples) < 5:
                unmatched_samples.append(f"{ts_code}->{normalized_rt_code or '-'}")
            continue
        matched_rows += 1
        pre_close = row.get('PRE_CLOSE', row.get('pre_close'))
        current_price = _normalize_cycle_price(
            price,
            pre_close,
            INTRADAY_STOCK_PCT_ABS_LIMIT,
        )
        if len(sample_rank_inputs) < 5:
            sample_rank_inputs.append(
                (
                    ts_code,
                    _coerce_float(price),
                    _coerce_float(pre_close),
                    current_price,
                    _coerce_float(getattr(pos, 'buy_price', None)),
                )
            )
        if current_price is None:  # 新增: 统计 current_price 为空的情况
            pnl_calc_failed += 1
            if len(failed_pnl_samples) < 3:
                failed_pnl_samples.append(
                    (ts_code, price, pre_close, "current_price=None")
                )
            continue
        if pos.buy_price <= 0:  # 新增: 统计 buy_price 无效的情况
            buy_price_invalid += 1
            if len(failed_pnl_samples) < 3:
                failed_pnl_samples.append(
                    (ts_code, price, pre_close, f"buy_price={pos.buy_price}")
                )
            continue
        pnl_pct = (current_price - pos.buy_price) / pos.buy_price * 100
        code = ts_code.split('.')[0]
        stocks.append({'name': name[:4], 'code': code, 'pnl_pct': pnl_pct})

    _trace_diag(
        "排行匹配统计: "
        f"rt_total={rt_total}, rt_valid={rt_valid_code_price}, matched={matched_rows}, "
        f"positions={len(positions)}, position_aliases={len(normalized_position_map)}"
    )
    if sample_rank_inputs:
        _trace_diag(f"排行样本: {sample_rank_inputs}")

    if not stocks:
        position_samples = list(positions.keys())[:5]
        _emit_diag(
            "排行构建无结果: "
            f"matched={matched_rows}, rt_valid={rt_valid_code_price}, "
            f"pnl_calc_failed={pnl_calc_failed}, buy_price_invalid={buy_price_invalid}, "
            f"sample_rt_unmatched={unmatched_samples}, sample_pos_keys={position_samples}, "
            f"failed_pnl_samples={failed_pnl_samples}"
        )
        return None

    stocks.sort(key=lambda x: x['pnl_pct'], reverse=True)
    return stocks


def _fetch_stock_rankings() -> Optional[list]:
    """获取个股盈亏排名（盈利前2 + 亏损前2）。

    Returns:
        list[dict]: 按盈亏排序的个股列表，每项包含:
            name: str       - 股票名称
            code: str       - 6位股票代码
            pnl_pct: float  - 盈亏比率(%)
        None: 数据不可用
    """
    return _build_stock_rankings(_fetch_realtime_holdings_snapshot())


# ---------- 日内图数据获取 ----------

def _extract_pct_from_quote_row(row) -> Optional[float]:
    """从单条实时行情记录中提取当日涨跌幅。"""
    if row is None:
        return None
    pct_direct = _sanitize_intraday_pct(
        row.get('PCT_CHG', row.get('pct_chg')),
        INTRADAY_INDEX_PCT_ABS_LIMIT,
    )
    if pct_direct is not None:
        return pct_direct
    price = _coerce_float(row.get('PRICE', row.get('price')))
    pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
    if price is None or not np.isfinite(price) or price <= 0 or pre_close in (None, 0):
        return None
    return _sanitize_intraday_pct(
        (price / pre_close - 1) * 100,
        INTRADAY_INDEX_PCT_ABS_LIMIT,
    )


def _derive_pre_close_from_price_and_pct(price: object, pct_chg: object) -> Optional[float]:
    """根据现价与涨跌幅反推昨收。"""
    price_float = _coerce_float(price)
    pct_float = _sanitize_intraday_pct(pct_chg, INTRADAY_STOCK_PCT_ABS_LIMIT)
    if (
        price_float is None
        or not np.isfinite(price_float)
        or price_float <= 0
        or pct_float is None
    ):
        return None
    ratio = 1.0 + pct_float / 100.0
    if abs(ratio) < 1e-8:
        return None
    pre_close = price_float / ratio
    if not np.isfinite(pre_close) or pre_close <= 0:
        return None
    return pre_close


def _extract_index_pct_map_from_quote_df(rt_df) -> dict[str, float]:
    """从实时行情表中提取上证、深证与中证800当日涨跌幅。"""
    pct_map: dict[str, float] = {}
    if rt_df is None or rt_df.empty:
        return pct_map

    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', row.get('ts_code', '')))
        if ts_code not in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
            continue
        pct = _extract_pct_from_quote_row(row)
        if pct is not None:
            pct_map[ts_code] = pct

    return pct_map


def _extract_index_pct_from_akshare(df, target_code: str) -> Optional[float]:
    """从 akshare 指数现货表中提取指定指数当日涨跌幅。"""
    if df is None or df.empty:
        _emit_diag_once(
            f"akshare_spot_empty::{target_code}",
            f"AKShare现货表为空，无法提取指数涨跌幅: {target_code}",
            stderr=False,
        )
        return None

    code_columns = ['代码', 'symbol', 'ts_code']
    matched = None
    target_aliases = {target_code}
    if target_code == SHANGHAI_INDEX_CODE:
        target_aliases.update({'000001', 'sh000001'})
    elif target_code == SHENZHEN_INDEX_CODE:
        target_aliases.update({'399001', 'sz399001'})
    elif target_code == CSI800_INDEX_CODE:
        target_aliases.update({'000906', 'sh000906'})
    for col in code_columns:
        if col not in df.columns:
            continue
        code_series = df[col].astype(str)
        mask = code_series.isin(target_aliases)
        if mask.any():
            matched = df.loc[mask].iloc[0]
            break

    if matched is None:
        _emit_diag_once(
            f"akshare_spot_code_missing::{target_code}",
            f"AKShare现货表未命中指数代码: {target_code} | 可用列: {list(df.columns)}",
            stderr=False,
        )
        return None

    pct = _coerce_float(matched.get('涨跌幅', matched.get('pct_chg')))
    if pct is not None:
        return _sanitize_intraday_pct(pct, INTRADAY_INDEX_PCT_ABS_LIMIT)

    price = _coerce_float(matched.get('最新价', matched.get('最新')))
    pre_close = _coerce_float(
        matched.get('昨收', matched.get('昨收盘', matched.get('pre_close')))
    )
    if price is None or not np.isfinite(price) or price <= 0 or pre_close in (None, 0):
        _emit_diag_once(
            f"akshare_spot_price_invalid::{target_code}",
            f"AKShare现货记录缺少有效价格/昨收，无法计算涨跌幅: {target_code}",
            stderr=False,
        )
        return None
    return _sanitize_intraday_pct(
        (price / pre_close - 1) * 100,
        INTRADAY_INDEX_PCT_ABS_LIMIT,
    )


def _fetch_realtime_index_pcts_from_akshare() -> dict[str, float]:
    """使用 AKShare 拉取实时指数涨跌幅。"""
    pct_map: dict[str, float] = {}
    fetch_started_at = time.monotonic()
    _trace_diag("指数抓取开始: source=AKShare")

    try:
        import akshare as ak  # type: ignore

        getter_name = 'stock_zh_index_spot_sina'
        getter = getattr(ak, getter_name, None)
        if getter is None:
            _emit_diag_once(
                f"akshare_spot_getter_missing::{getter_name}",
                f"AKShare实时接口不存在: {getter_name}",
                stderr=False,
            )
        else:
            try:
                _trace_diag(f"指数主接口调用: api={getter_name}")
                with _fetch_network_context():
                    df = getter()
                row_count = 0 if df is None else int(len(df))
                _trace_diag(f"指数主接口返回: api={getter_name}, rows={row_count}")
                for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
                    if code in pct_map:
                        continue
                    pct = _extract_index_pct_from_akshare(df, code)
                    if pct is not None:
                        pct_map[code] = pct
            except Exception as exc:
                _emit_diag_once(
                    f"akshare_spot_getter_error::{getter_name}",
                    f"AKShare实时接口调用失败: {getter_name} | {type(exc).__name__}: {exc}",
                )

        # 单次抓取未命中中证800时再兜底二次请求，兼顾性能与稳健性。
        if CSI800_INDEX_CODE not in pct_map:
            _trace_diag("指数补抓开始: code=000906.SH, mode=fallback")
            csi800_pct = _fetch_csi800_realtime_pct_akshare()
            if csi800_pct is not None:
                pct_map[CSI800_INDEX_CODE] = csi800_pct
    except Exception:
        _emit_diag_once(
            "akshare_spot_import_or_loop_error",
            "AKShare实时指数获取主流程异常，已回退现有数据",
        )

    _trace_diag(
        "指数抓取结束: "
        f"codes={sorted(pct_map.keys())}, cost={time.monotonic() - fetch_started_at:.2f}s"
    )

    return pct_map


def _fetch_realtime_index_pcts(snapshot: Optional[dict] = None) -> dict[str, float]:
    """获取上证、深证与中证800当日实时涨跌幅。"""
    pct_map: dict[str, float] = {}

    if snapshot is not None:
        snapshot_pct_map = snapshot.get('index_pct_map')
        if isinstance(snapshot_pct_map, dict):
            for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
                pct = _sanitize_intraday_pct(
                    snapshot_pct_map.get(code),
                    INTRADAY_INDEX_PCT_ABS_LIMIT,
                )
                if pct is not None:
                    pct_map[code] = pct
        if len(pct_map) < 3:
            pct_map.update(
                _extract_index_pct_map_from_quote_df(snapshot.get('quotes'))
            )
        if len(pct_map) == 3:
            return pct_map

    cached_pct_map = _get_cached_realtime_index_pcts()
    for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE):
        if code in pct_map:
            continue
        pct = _sanitize_intraday_pct(
            cached_pct_map.get(code),
            INTRADAY_INDEX_PCT_ABS_LIMIT,
        )
        if pct is not None:
            pct_map[code] = pct

    if len(pct_map) < 3:
        _trace_diag(
            "指数缓存不足，触发后台刷新: "
            f"current_codes={sorted(pct_map.keys())}"
        )
        _refresh_realtime_index_pcts_async()

    missing_codes = [
        code for code in (SHANGHAI_INDEX_CODE, SHENZHEN_INDEX_CODE, CSI800_INDEX_CODE)
        if code not in pct_map
    ]
    if missing_codes:
        _emit_diag_once(
            "akshare_spot_missing_codes",
            f"实时指数涨跌幅缺失代码: {missing_codes} | 已获取: {sorted(pct_map.keys())}",
            stderr=False,
        )

    return pct_map


def _fetch_csi800_realtime_pct_akshare() -> Optional[float]:
    """按 stock_zh_index_spot_sina 接口获取中证800实时涨跌幅。"""
    fetch_started_at = time.monotonic()
    _trace_diag("中证800抓取开始: api=stock_zh_index_spot_sina")
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_import_error",
            f"AKShare导入失败，无法获取中证800实时涨跌幅 | {type(exc).__name__}: {exc}",
        )
        return None

    getter_sina = getattr(ak, 'stock_zh_index_spot_sina', None)
    if getter_sina is None:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_sina_missing",
            "AKShare缺少 stock_zh_index_spot_sina 接口，无法按指定链路获取中证800实时行情",
        )
        return None

    try:
        with _fetch_network_context():
            df = getter_sina()
        row_count = 0 if df is None else int(len(df))
        _trace_diag(f"中证800主接口返回: rows={row_count}")
    except Exception as exc:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_sina_error",
            f"AKShare实时接口调用失败: stock_zh_index_spot_sina | {type(exc).__name__}: {exc}",
        )
        return None

    if df is None or df.empty:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_empty",
            "AKShare stock_zh_index_spot_sina 返回空数据，无法提取中证800",
        )
        return None

    code_col = next((col for col in ('代码', 'symbol', 'ts_code') if col in df.columns), None)
    if code_col is None:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_code_col_missing",
            f"AKShare stock_zh_index_spot_sina 缺少代码列，当前列: {list(df.columns)}",
        )
        return None

    code_series = df[code_col].astype(str)
    matched = df.loc[code_series.isin({'sh000906', '000906', '000906.SH'})]
    if matched.empty:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_000906_missing",
            "AKShare stock_zh_index_spot_sina 未找到中证800(000906)",
            stderr=False,
        )
        return None

    row = matched.iloc[0]
    pct = _coerce_float(row.get('涨跌幅', row.get('pct_chg')))
    if pct is None:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_pct_missing",
            "AKShare stock_zh_index_spot_sina 命中000906但缺少涨跌幅字段",
        )
        return None

    pct_sanitized = _sanitize_intraday_pct(pct, INTRADAY_INDEX_PCT_ABS_LIMIT)
    if pct_sanitized is None:
        _emit_diag_once(
            "akshare_spot_stock_zh_index_spot_pct_invalid",
            f"AKShare stock_zh_index_spot_sina 000906涨跌幅异常: {pct}",
            stderr=False,
        )
    _trace_diag(
        "中证800抓取结束: "
        f"pct={pct_sanitized}, cost={time.monotonic() - fetch_started_at:.2f}s"
    )
    return pct_sanitized


def _compute_holdings_intraday_pct(snapshot: Optional[dict]) -> Optional[float]:
    """计算当前持仓股票相对昨收的实时涨跌幅（不含现金）。"""
    if snapshot is None:
        return None

    positions = snapshot.get('positions', {})
    rt_df = snapshot.get('quotes')
    if rt_df is None or rt_df.empty or not positions:
        return None

    quote_map = {}
    for _, row in rt_df.iterrows():
        ts_code = str(row.get('TS_CODE', ''))
        if ts_code:
            quote_map[ts_code] = row

    current_value = 0.0
    prev_close_value = 0.0
    valid_count = 0
    pre_close_derived_count = 0
    pre_close_missing_count = 0
    for ts_code, pos in positions.items():
        row = quote_map.get(ts_code)
        if row is None:
            continue
        current_price = _coerce_float(row.get('PRICE', row.get('price')))
        if current_price is None or not np.isfinite(current_price) or current_price <= 0:
            continue

        pre_close = _coerce_float(row.get('PRE_CLOSE', row.get('pre_close')))
        if pre_close is None or pre_close <= 0:
            pre_close = _derive_pre_close_from_price_and_pct(
                current_price,
                row.get('PCT_CHG', row.get('pct_chg')),
            )
            if pre_close is None:
                pre_close_missing_count += 1
                continue
            pre_close_derived_count += 1

        current_price = _normalize_intraday_price(
            current_price,
            pre_close,
            INTRADAY_STOCK_PCT_ABS_LIMIT,
        )
        if pre_close in (None, 0) or current_price is None:
            continue
        current_value += current_price * pos.shares
        prev_close_value += pre_close * pos.shares
        valid_count += 1

    if valid_count == 0 or prev_close_value <= 0:
        _trace_diag(
            "盘中收益计算失败: "
            f"valid=0, derived_pre_close={pre_close_derived_count}, "
            f"missing_pre_close={pre_close_missing_count}, positions={len(positions)}"
        )
        return None
    if pre_close_derived_count > 0:
        _trace_diag(
            "盘中收益计算: "
            f"derived_pre_close={pre_close_derived_count}, "
            f"missing_pre_close={pre_close_missing_count}, valid={valid_count}"
        )
    return _sanitize_intraday_pct(
        (current_value / prev_close_value - 1) * 100,
        INTRADAY_PORTFOLIO_PCT_ABS_LIMIT,
    )


def _build_intraday_chart(
    chart_data: Optional[dict],
    snapshot: Optional[dict],
    point_time: Optional[datetime] = None,
) -> Optional[dict]:
    """基于上证/深证/中证800实时涨跌与持仓股当日实时涨跌构建盘中图。"""
    current_time = point_time or _get_snapshot_quote_time(snapshot) or datetime.now()
    if snapshot is None or not _is_intraday_trading_time(current_time):
        return chart_data

    index_pct_map = _fetch_realtime_index_pcts(snapshot)
    holdings_pct = _compute_holdings_intraday_pct(snapshot)
    shanghai_pct = index_pct_map.get(SHANGHAI_INDEX_CODE)
    shenzhen_pct = index_pct_map.get(SHENZHEN_INDEX_CODE)
    csi800_pct = index_pct_map.get(CSI800_INDEX_CODE)

    # 中证800偶发取数失败时，沿用上一采样点，避免整次盘中刷新被短路
    if csi800_pct is None and isinstance(chart_data, dict):
        raw_csi800 = chart_data.get('raw_csi800_pct', chart_data.get('csi800_pct', []))
        if isinstance(raw_csi800, list) and raw_csi800:
            csi800_pct = _sanitize_intraday_pct(raw_csi800[-1], INTRADAY_INDEX_PCT_ABS_LIMIT)

    if (
        shanghai_pct is None
        or shenzhen_pct is None
        or holdings_pct is None
    ):
        quote_rows = 0
        if isinstance(snapshot, dict):
            quote_df = snapshot.get('quotes')
            quote_rows = 0 if quote_df is None else int(len(quote_df))
        _trace_diag(
            "盘中图跳过: "
            f"sh={shanghai_pct}, sz={shenzhen_pct}, hold={holdings_pct}, quote_rows={quote_rows}"
        )
        return chart_data

    if csi800_pct is None:
        csi800_pct = shanghai_pct

    return _upsert_intraday_chart(
        chart_data,
        current_time,
        index_pct=shanghai_pct,
        shenzhen_pct=shenzhen_pct,
        portfolio_pct=holdings_pct,
        csi800_pct=csi800_pct,
    )


def _should_keep_realtime_completion_active(
    cycle_chart_data: Optional[dict],
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断收盘后是否仍需继续补齐日内图最后一格。"""
    current_dt = now or datetime.now()
    if _is_realtime_quote_window(current_dt):
        return True
    if not _is_trade_day(current_dt, allow_load=True):
        return False
    target_cycle_date = _get_target_cycle_data_date(current_dt, allow_load=True)
    if target_cycle_date != current_dt.strftime("%Y%m%d"):
        return False
    close_deadline = datetime.combine(current_dt.date(), INTRADAY_WINDOW_END) + timedelta(
        seconds=POST_CLOSE_INTRADAY_GRACE_SECONDS
    )
    if current_dt > close_deadline:
        return False
    if _has_cycle_data_for_target(cycle_chart_data, target_cycle_date):
        return False
    return not _is_intraday_chart_complete(intraday_chart_data, current_dt)


def _should_keep_morning_close_completion_active(
    intraday_chart_data: Optional[dict],
    now: Optional[datetime] = None,
) -> bool:
    """判断午休开始后是否仍需补齐上午 11:30 最后一格。"""
    current_dt = now or datetime.now()
    if not _is_trade_day(current_dt, allow_load=True):
        return False

    morning_close_dt = datetime.combine(current_dt.date(), A_SHARE_MORNING_CLOSE)
    grace_deadline = morning_close_dt + timedelta(seconds=MORNING_CLOSE_INTRADAY_GRACE_SECONDS)
    if not (morning_close_dt < current_dt <= grace_deadline):
        return False

    return not _is_morning_intraday_chart_complete(intraday_chart_data, current_dt)


def _refresh_display_state(
    state: "DisplayState",
    refresh_realtime: bool = False,
    refresh_cycle: bool = False,
) -> None:
    """按需刷新共享显示状态。"""
    _trace_diag(
        "刷新开始: "
        f"realtime={refresh_realtime}, cycle={refresh_cycle}, "
        f"snapshot_timeout={REALTIME_SNAPSHOT_TIMEOUT_SECONDS:.1f}s, "
        f"ef_timeout=({EFINANCE_CONNECT_TIMEOUT_SECONDS:.1f}s,{EFINANCE_READ_TIMEOUT_SECONDS:.1f}s), "
        f"proxy_bypass={_should_bypass_proxy_for_fetch()}"
    )
    with state.lock:
        state.is_updating = refresh_realtime or refresh_cycle
        state.update_step = "刷新中"
        state.update_started_at = time.monotonic()

    holdings_snapshot = None
    latest_update_time: Optional[str] = None

    try:
        if refresh_realtime:
            try:
                with state.lock:
                    state.update_step = "抓快照"
                holdings_snapshot = _call_with_timeout(
                    _fetch_realtime_holdings_snapshot,
                    REALTIME_SNAPSHOT_TIMEOUT_SECONDS,
                    fallback=None,
                    timeout_diag_key="realtime_holdings_snapshot_timeout",
                    timeout_diag_message=(
                        "实时快照抓取超时，已跳过本轮刷新并保留上次有效数据显示"
                    ),
                )
                if isinstance(holdings_snapshot, dict):
                    pos_count = len(holdings_snapshot.get('positions', {}) or {})
                    quote_df = holdings_snapshot.get('quotes')
                    quote_rows = int(len(quote_df)) if quote_df is not None else 0
                    _trace_diag(
                        "抓快照完成: "
                        f"source={holdings_snapshot.get('quote_source', '-')}, "
                        f"positions={pos_count}, quote_rows={quote_rows}"
                    )
                    source = str(holdings_snapshot.get('quote_source', '')).strip().upper()
                    with state.lock:
                        state.quote_source_tag = source if source in ('T', 'A', 'D') else '-'
                    # 即便摘要计算失败，也至少更新时间戳，避免长期显示 --:--
                    latest_update_time = datetime.now().strftime("%H:%M")
                else:
                    cached_snapshot = _get_cached_holdings_snapshot()
                    if isinstance(cached_snapshot, dict):
                        holdings_snapshot = cached_snapshot
                        cached_source = str(cached_snapshot.get('quote_source', '-')).strip().upper()
                        cached_quote_df = cached_snapshot.get('quotes')
                        cached_rows = int(len(cached_quote_df)) if cached_quote_df is not None else 0
                        with state.lock:
                            state.quote_source_tag = cached_source if cached_source in ('T', 'A', 'D') else '-'
                        latest_update_time = datetime.now().strftime("%H:%M")
                        _emit_diag(
                            "抓快照结果为空（超时或异常），"
                            f"已回退使用最近缓存快照: source={cached_source}, quote_rows={cached_rows}"
                        )
                    else:
                        _emit_diag("抓快照结果为空（超时或异常），本轮实时面板将沿用旧值")
            except Exception:
                _emit_diag("抓快照阶段异常，已跳过本轮实时面板刷新")
                holdings_snapshot = None

            try:
                with state.lock:
                    state.update_step = "算摘要"
                summary = _build_realtime_portfolio_summary(holdings_snapshot)
                if summary is not None:
                    with state.lock:
                        state.summary = summary
                    latest_update_time = _format_quote_update_time(summary) or datetime.now().strftime("%H:%M")
                    _trace_diag(
                        "摘要更新成功: "
                        f"pos_count={summary.get('pos_count')}, quote_time={summary.get('quote_time', '')}"
                    )
                else:
                    q_rows = 0
                    if isinstance(holdings_snapshot, dict):
                        q_df = holdings_snapshot.get('quotes')
                        q_rows = int(len(q_df)) if q_df is not None else 0
                    _emit_diag(f"摘要为空: snapshot_quote_rows={q_rows}")
            except Exception:
                _emit_diag("算摘要阶段异常，摘要保持上次有效值")

            try:
                with state.lock:
                    current_intraday_chart = state.intraday_chart_data
                    state.update_step = "盘中图"
                intraday_chart_data = _call_with_timeout(
                    lambda: _build_intraday_chart(current_intraday_chart, holdings_snapshot),
                    REALTIME_INTRADAY_TIMEOUT_SECONDS,
                    fallback=current_intraday_chart,
                    timeout_diag_key="realtime_intraday_chart_timeout",
                    timeout_diag_message="盘中图构建超时，已保留上次盘中图数据",
                )
                if intraday_chart_data is not None:
                    with state.lock:
                        state.intraday_chart_data = intraday_chart_data
                    _save_intraday_chart(intraday_chart_data)
            except Exception:
                pass

            try:
                with state.lock:
                    state.update_step = "算排行"
                ranks = _build_stock_rankings(holdings_snapshot)
                if ranks is not None:
                    with state.lock:
                        state.stock_rankings = ranks
                    _trace_diag(f"排行更新成功: rows={len(ranks)}")
                else:
                    _emit_diag("排行为空: 快照缺少可用持仓行情")
            except Exception:
                _emit_diag("算排行阶段异常，排行保持上次有效值")

            try:
                with state.lock:
                    state.update_step = "算行业"
                industry_panel_cycle = _build_industry_panel(holdings_snapshot, mode="cycle")
                industry_panel_intraday = _build_industry_panel(holdings_snapshot, mode="intraday")
                if industry_panel_cycle is not None or industry_panel_intraday is not None:
                    with state.lock:
                        if industry_panel_cycle is not None:
                            state.industry_panel = industry_panel_cycle
                            state.industry_panel_cycle = industry_panel_cycle
                        if industry_panel_intraday is not None:
                            state.industry_panel_intraday = industry_panel_intraday
                    _trace_diag(
                        "行业更新成功: "
                        f"cycle={'Y' if industry_panel_cycle is not None else 'N'}, "
                        f"intraday={'Y' if industry_panel_intraday is not None else 'N'}"
                    )
                else:
                    _emit_diag("行业统计为空: 快照缺少可用持仓行情")
            except Exception:
                _emit_diag("算行业阶段异常，行业统计保持上次有效值")

        try:
            with state.lock:
                state.update_step = "算调仓"
            next_rebalance_date, days_to_rebalance = _calc_rebalance_status()
            with state.lock:
                state.next_rebalance_date = next_rebalance_date
                state.days_to_rebalance = days_to_rebalance
        except Exception:
            pass

        if refresh_cycle:
            try:
                with state.lock:
                    state.update_step = "抓周期"
                cycle_chart_data = _fetch_cycle_chart_data()
                if cycle_chart_data is not None:
                    with state.lock:
                        state.chart_data = cycle_chart_data
                    if latest_update_time is None:
                        latest_update_time = datetime.now().strftime("%H:%M")
                    _trace_diag(
                        "周期图更新成功: "
                        f"points={len(cycle_chart_data.get('dates', []) or [])}"
                    )
                else:
                    _emit_diag("抓周期返回空数据，周期图保持上次有效值")
            except Exception as exc:
                _emit_diag(
                    "抓周期阶段异常: "
                    f"{type(exc).__name__}: {exc}，周期图保持上次有效值"
                )
    finally:
        with state.lock:
            if latest_update_time is not None:
                state.update_time = latest_update_time
            state.update_step = ""
            state.update_started_at = 0.0
            state.is_updating = False
            summary_ready = state.summary is not None
            rank_ready = state.stock_rankings is not None
            industry_ready = state.industry_panel is not None
            chart_ready = state.chart_data is not None
            update_time = state.update_time
        _trace_diag(
            "刷新结束: "
            f"update_time={update_time}, summary={summary_ready}, "
            f"rank={rank_ready}, industry={industry_ready}, cycle_chart={chart_ready}"
        )


# ---------- 共享显示状态 ----------


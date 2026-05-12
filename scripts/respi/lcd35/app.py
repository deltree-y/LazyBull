from scripts.respi.lcd35._context import (
    Optional,
    datetime,
    get_config,
    random,
    setup_logger,
    signal,
    sys,
    time,
    threading,
)
from scripts.respi.lcd35.charting import (
    _describe_framebuffer_candidates,
    _emit_diag,
    _emit_diag_once,
    _get_data_worker_wait_seconds,
    _get_refresh_policy,
    _get_realtime_session_key,
    _get_target_cycle_data_date,
    _is_cycle_refresh_due,
    _is_realtime_refresh_due,
    _render_bootstrap_screen,
    _resolve_framebuffer_path,
)
from scripts.respi.lcd35.data_pipeline import _fetch_cycle_chart_data, _refresh_display_state
from scripts.respi.lcd35.core import (
    BACKLIGHT_BRIGHTNESS,
    REALTIME_RETRY_WAIT_SECONDS,
    REALTIME_SNAPSHOT_TIMEOUT_SECONDS,
    SCREENSAVER_INTERVAL,
    SCREENSAVER_RANGE_X,
    SCREENSAVER_RANGE_Y,
    _get_cached_holdings_snapshot,
    _should_bypass_proxy_for_fetch,
    _trace_diag,
)
from scripts.respi.lcd35.rendering import _render
from scripts.respi.lcd35.state import DisplayState
from scripts.respi.lcd35.system_io import (
    _cleanup_backlight,
    _clear_screen,
    _init_backlight,
    _render_error_screen,
    _set_backlight,
)


def _data_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """按分频策略获取实时行情和图表数据，更新共享状态。

    启动时立即获取一次（非交易日也会返回最近一个交易日的收盘数据）。
    """
    _emit_diag_once("data_worker_start", "数据线程已启动")
    _emit_diag_once(
        "fetch_proxy_mode_once",
        "抓数代理模式: "
        f"bypass={_should_bypass_proxy_for_fetch()} "
        "(可用 LAZYBULL_FETCH_BYPASS_PROXY=0 关闭)",
        stderr=False,
    )
    _emit_diag_once(
        "snapshot_timeout_config_once",
        "快照超时配置: "
        f"{REALTIME_SNAPSHOT_TIMEOUT_SECONDS:.1f}s "
        "(env=LAZYBULL_REALTIME_SNAPSHOT_TIMEOUT_SECONDS, default=60.0s)",
        stderr=False,
    )

    try:
        # 启动时立即获取一次（非交易日也能显示最近收盘数据）
        startup_dt = datetime.now()
        _refresh_display_state(state, refresh_realtime=True, refresh_cycle=True)
        last_realtime_refresh_at: Optional[datetime] = startup_dt
        last_realtime_session_key = _get_realtime_session_key(startup_dt)
        last_cycle_refresh_at: Optional[datetime] = startup_dt
        last_cycle_target_date = _get_target_cycle_data_date(startup_dt, allow_load=True)
        next_wait_override: Optional[float] = None

        while not stop_event.is_set():
            with state.lock:
                current_cycle_chart = state.chart_data
                current_intraday_chart = state.intraday_chart_data
            if next_wait_override is not None:
                wait_seconds = float(next_wait_override)
                next_wait_override = None
            else:
                wait_seconds = _get_data_worker_wait_seconds(
                    cycle_chart_data=current_cycle_chart,
                    intraday_chart_data=current_intraday_chart,
                )
            stop_event.wait(wait_seconds)
            if stop_event.is_set():
                break

            current_dt = datetime.now()
            with state.lock:
                current_cycle_chart = state.chart_data
                current_intraday_chart = state.intraday_chart_data
            refresh_policy = _get_refresh_policy(
                current_cycle_chart,
                intraday_chart_data=current_intraday_chart,
                now=current_dt,
            )
            refresh_realtime, realtime_session_key = _is_realtime_refresh_due(
                bool(refresh_policy['refresh_realtime']),
                last_realtime_refresh_at,
                last_realtime_session_key,
                current_dt,
            )
            refresh_cycle, cycle_target_date = _is_cycle_refresh_due(
                current_cycle_chart,
                bool(refresh_policy['refresh_cycle']),
                last_cycle_refresh_at,
                last_cycle_target_date,
                current_dt,
            )

            _trace_diag(
                "调度决策: "
                f"policy(realtime={refresh_policy['refresh_realtime']},cycle={refresh_policy['refresh_cycle']}), "
                f"due(realtime={refresh_realtime},cycle={refresh_cycle}), "
                f"session={realtime_session_key}, target_cycle={cycle_target_date}, wait={wait_seconds:.1f}s"
            )

            if refresh_cycle or refresh_realtime:
                _refresh_display_state(
                    state,
                    refresh_realtime=refresh_realtime,
                    refresh_cycle=refresh_cycle,
                )
                if refresh_realtime:
                    last_realtime_refresh_at = current_dt
                    last_realtime_session_key = realtime_session_key
                    cached_snapshot = _get_cached_holdings_snapshot()
                    if not isinstance(cached_snapshot, dict):
                        next_wait_override = float(REALTIME_RETRY_WAIT_SECONDS)
                        _emit_diag(
                            "实时抓取未命中有效快照，"
                            f"将于{REALTIME_RETRY_WAIT_SECONDS:.0f}s后快速重试"
                        )
                if refresh_cycle:
                    last_cycle_refresh_at = current_dt
                    last_cycle_target_date = cycle_target_date
            else:
                _trace_diag("本轮无需刷新，继续等待下一轮调度")
    except Exception as exc:
        _emit_diag(f"数据线程异常退出: {type(exc).__name__}: {exc}")


# ---------- 显示刷新线程 ----------

def _display_worker(state: DisplayState, stop_event: threading.Event) -> None:
    """每秒刷新画面，每 SCREENSAVER_INTERVAL 秒更新屏保偏移。

    23:00-6:00 自动息屏。
    """
    last_offset_time = 0.0
    _emit_diag_once("display_worker_start", "显示线程已启动")

    while not stop_event.is_set():
        try:
            hour = datetime.now().hour

            # ---- 息屏逻辑（23:00 - 6:00）----
            if hour >= 23 or hour < 6:
                _emit_diag_once(
                    "sleep_window_active",
                    f"当前命中自动息屏时段({hour:02d}:xx)，LCD 将保持黑屏直到 06:00",
                )
                if state.is_screen_on:
                    _clear_screen()
                    _set_backlight(0)
                    state.is_screen_on = False
                stop_event.wait(10)
                continue

            if not state.is_screen_on:
                _set_backlight(BACKLIGHT_BRIGHTNESS)
                with state.lock:
                    state.cpu_usage_pct = 0.0
                    state.memory_usage_pct = 0.0
                    state.cpu_usage_sample = None
                    state.usage_sampled_at = 0.0
                state.is_screen_on = True

            # ---- 屏保：每分钟随机偏移数据区 ----
            now_ts = time.monotonic()
            if now_ts - last_offset_time >= SCREENSAVER_INTERVAL:
                with state.lock:
                    state.offset_x = random.randint(-SCREENSAVER_RANGE_X, SCREENSAVER_RANGE_X)
                    state.offset_y = random.randint(-SCREENSAVER_RANGE_Y, SCREENSAVER_RANGE_Y)
                last_offset_time = now_ts

            # ---- 渲染（含实时时间）----
            _emit_diag_once("render_first_frame_start", "显示线程开始首帧渲染")
            _render(state)
            _emit_diag_once("render_first_frame_done", "显示线程已写出首帧")

            # ---- 每秒刷新 ----
            stop_event.wait(1)
        except Exception as exc:
            _render_error_screen(f"{type(exc).__name__}: {exc}")
            print(f"[lcd35_display] 渲染异常: {type(exc).__name__}: {exc}", file=sys.stderr)
            stop_event.wait(2)


# ---------- 入口 ----------

def main() -> None:
    _emit_diag("主程序启动")
    try:
        setup_logger(log_level="WARNING")
        _emit_diag_once("logger_ready", "日志初始化完成")
        get_config()
        _emit_diag_once("config_ready", "配置加载完成")

        selected_fb = _resolve_framebuffer_path()
        _emit_diag_once(
            f"fb_target::{selected_fb}",
            f"当前 framebuffer 目标: {selected_fb} | 可用设备: {_describe_framebuffer_candidates()}",
        )

        _init_backlight()
        _emit_diag_once("backlight_phase_done", "背光初始化阶段完成")
        _render_bootstrap_screen("准备启动数据与显示线程")
        _emit_diag_once("bootstrap_screen_written", "已尝试写入启动测试页")

        state = DisplayState()
        stop_event = threading.Event()

        def _shutdown(sig, frame):  # noqa: ANN001
            _emit_diag(f"收到退出信号: {sig}", stderr=False)
            stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # 数据获取线程（盘中实时 2 分钟，周期图与补数 10 分钟按需）
        data_t = threading.Thread(target=_data_worker, args=(state, stop_event), daemon=True)
        data_t.start()

        # 显示刷新线程（每秒）
        disp_t = threading.Thread(target=_display_worker, args=(state, stop_event), daemon=True)
        disp_t.start()
        _emit_diag_once("threads_started", "数据线程和显示线程已启动")

        try:
            while not stop_event.is_set():
                time.sleep(1)
        finally:
            _clear_screen()
            _cleanup_backlight()
            _emit_diag("主程序退出")
    except Exception as exc:
        _emit_diag(f"主程序启动失败: {type(exc).__name__}: {exc}")
        raise


if __name__ == '__main__':
    main()

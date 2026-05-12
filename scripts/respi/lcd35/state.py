from scripts.respi.lcd35._context import Optional, threading
from scripts.respi.lcd35.charting import _load_intraday_chart


class DisplayState:
    """数据线程与显示线程之间的共享状态。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.summary: Optional[dict] = None
        self.update_time: str = "--:--"
        self.is_updating: bool = False
        self.update_step: str = ""
        self.update_started_at: float = 0.0
        self.quote_source_tag: str = "-"
        self.next_rebalance_date: Optional[str] = None
        self.days_to_rebalance: Optional[int] = None
        self.chart_data: Optional[dict] = None
        self.intraday_chart_data: Optional[dict] = _load_intraday_chart()
        self.stock_rankings: Optional[list] = None  # 个股盈亏排名
        self.industry_panel: Optional[dict] = None  # 行业收益统计
        self.industry_panel_cycle: Optional[dict] = None  # 盘外/持仓周期口径
        self.industry_panel_intraday: Optional[dict] = None  # 盘内当日口径
        self.cpu_usage_pct: float = 0.0
        self.memory_usage_pct: float = 0.0
        self.cpu_usage_sample: Optional[tuple[int, int]] = None
        self.usage_sampled_at: float = 0.0
        # 屏保偏移（仅数据行参与）
        self.offset_x: int = 0
        self.offset_y: int = 0

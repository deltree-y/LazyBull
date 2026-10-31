"""v2 纸面宿主（P4-T2 起）：纸面交易场景的旧宿主装配迁移位。

- ``performance``：账户绩效口径（年化唯一实现新位置，D3 过渡例外迁移件）；
- ``reporting``：展示与格式化公共能力（D6 迁移件）；
- ``daily_assembly``：当日数据组装 + 量化就绪门（D4）；
- ``state_store``：纸面宿主状态库（v2 引擎运行时状态 ↔ 旧 ``data/paper/`` 布局适配；
  隔离根 / 单写入者写锁 / 同代原子提交点 / 半提交受控恢复 / SMB 远端只读）；
- ``config_adapter``：配置适配（config.yaml 分组模板 → v2 内核参数；engine 切换键
  解析 Q9 / 模型选择收口 F19 / 执行价格指纹 D7）；
- ``host``：纸面宿主核心（PaperHost；T1 内核 run_step/EngineState/live_schedule
  接入纸面场景，单日原子驱动 + run_t0/run_t1 双视图）；
- ``recovery``：半提交恢复职责族（PaperHostRecoveryMixin——完整代核对重建/
  幂等记录补写；经 mixin 继承，实例方法面不变）；
- ``instruction_projection``：T0 产出 → 旧 instructions 布局的投影纯函数族
  （经 state_store re-export，导入面不变）；
- ``factory``：纸面宿主装配工厂（create_paper_host + format_host_model_info
  展示同源件；自 host.py 行数硬上限拆分）。
- ``replay``：P4-T4 历史回放（窗口锚逐日步进、行域断言、D7 指纹与单 fold runs 出口）。
- ``shadow``：P4-T5 真实影子窗口逐日未裁剪证据累积与 paper_shadow runs 出口。
"""

from src.lazybull.v2.hosts.paper.config_adapter import (
    PAPER_CONFIG_TO_V2,
    PaperHostError,
    load_paper_config,
    resolve_engine,
    resolve_model_version,
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.daily_assembly import (
    DailyAssembly,
    ReadinessReport,
    assemble_daily_data,
)
from src.lazybull.v2.hosts.paper.factory import create_paper_host, format_host_model_info
from src.lazybull.v2.hosts.paper.host import PaperDayResult, PaperHost
from src.lazybull.v2.hosts.paper.replay import (
    assert_execution_configs_match,
    assert_single_day_row_domain,
    run_paper_replay,
    write_paper_replay_batch,
)
from src.lazybull.v2.hosts.paper.shadow import (
    PaperShadowAccumulator,
    write_paper_shadow_batch,
)
from src.lazybull.v2.hosts.paper.state_store import (
    PaperLockError,
    PaperStateError,
    PaperStateStore,
)

__all__ = [
    "DailyAssembly",
    "PAPER_CONFIG_TO_V2",
    "PaperDayResult",
    "PaperHost",
    "PaperHostError",
    "PaperLockError",
    "PaperShadowAccumulator",
    "PaperStateError",
    "PaperStateStore",
    "ReadinessReport",
    "assemble_daily_data",
    "assert_execution_configs_match",
    "assert_single_day_row_domain",
    "create_paper_host",
    "format_host_model_info",
    "load_paper_config",
    "resolve_engine",
    "resolve_model_version",
    "run_paper_replay",
    "to_engine_kwargs",
    "to_trading_config",
    "write_paper_replay_batch",
    "write_paper_shadow_batch",
]

# -*- coding: utf-8 -*-
"""P7 T1（P-1 同域静态可执行性）实现包。

模块划分
========
- :mod:`params`       冻结参数（逐项取自生效 prereg）与身份校验；
- :mod:`dataplane`    日历/行情面板/协变量/输入身份摘要（只读）；
- :mod:`matching`     六股匹配组构造与匹配质量；
- :mod:`execution`    共同入离场、费用与总回报经济价值；
- :mod:`randomization` 内层随机化、单侧临界值、误放行与功效；
- :mod:`verdict`      计数守恒、有效块、压力诊断与三向 verdict；
- :mod:`report`       逐格诊断、构成/匹配质量表与匹配组账本；
- :mod:`run_p1`       证据根发布入口（原子、禁覆盖、写域守卫）。
"""

__all__ = [
    "params",
    "dataplane",
    "matching",
    "execution",
    "randomization",
    "verdict",
    "report",
    "run_p1",
]

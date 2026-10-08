"""P2a-T7 验收工具包（B0 重放 / 数据身份门 / 闭包预检 / 冻结比对）。

薄入口脚本 + 可导入逻辑模块（供 pytest 合成 fixture 复用）：
- ``replay_b0``：B0 基线 14 折重放驱动（v2 内核 native replay → runs 出口）；
- ``input_manifest``：验收门 1（数据身份门）哈希清单采集/校验 + 窗口内容抽查；
- ``check_replay_closure_diff``：验收门 2（代码态预检，git diff 限重放执行闭包）；
- ``compare_replay_vs_b0``：验收门 3/4/5（配置指纹 / 成交逐笔 / 净值 1e-6 比对）。
"""

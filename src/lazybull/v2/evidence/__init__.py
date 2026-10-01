# -*- coding: utf-8 -*-
"""v2 证据机器（P5a-1 核心，只读三件套 + 转换器）。

模块边界（方案 §3.5 / §8-P5a-1）：
- 只读既有 walk-forward 产物与转换后的 runs 产物，不写任何资产；
- 配对制度重排（regime_resample）为裁决唯一口径；功效标定（power_calibration）
  与判据自洽性（criterion_calibration）为其上的标定层；
- 旧产物 → runs 契约 schema 的转换器（runs_convert）。
"""

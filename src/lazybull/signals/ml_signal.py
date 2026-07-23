"""ML 信号生成模块

基于训练好的机器学习模型生成交易信号
使用排序选股 Top N 方式
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.config import get_models_root
from ..ml import ModelRegistry
from .base import Signal


@dataclass
class SignalConfidenceGateState:
    """信号置信度门控状态。"""

    enabled: bool = False
    score: float = float("nan")
    exposure: float = 1.0
    candidate_count: int = 0
    top_k: int = 0
    top_mean: float = float("nan")
    baseline_mean: float = float("nan")
    score_std: float = float("nan")
    hit_threshold: Optional[float] = None
    reason: str = "未启用"

    # ── composite 模式新增字段 ──
    abs_quality_score: float = float("nan")  # 绝对收益质量分
    separation_percentile: float = float("nan")  # 分离度历史百分位
    composite_score: float = float("nan")  # 综合得分
    cost_gate_passed: bool = True  # 成本门控是否通过
    rolling_quality: float = float("nan")  # 滚动模型质量（由engine注入）


class MLSignal(Signal):
    """ML 信号生成器

    基于机器学习模型预测，选择预测收益最高的 Top N 股票
    """

    # 申万一级行业：银行(801780)、非银金融(801790，含保险/券商)
    _FINANCIAL_SW_L1_CODES = {"801780", "801790"}

    def __init__(
        self,
        top_n: int = 20,
        model_version: Optional[int] = None,
        models_dir: Optional[str] = None,
        signal_confidence_gate_enabled: bool = False,
        signal_confidence_gate_top_k: int = 10,
        signal_confidence_gate_thresholds: Optional[List[float]] = None,
        signal_confidence_gate_exposure_levels: Optional[List[float]] = None,
        signal_gate_mode: str = "legacy",
        signal_gate_cost_multiplier: float = 2.0,
        signal_gate_round_trip_cost: float = 0.003,
        signal_gate_percentile_warmup: int = 20,
        min_amount_ma20: float = 50000.0,
        min_total_mv: float = 500000.0,
        max_total_mv: float = 15000000.0,
        exclude_financial: bool = True,
        verbose: bool = True,
    ):
        """初始化 ML 信号

        Args:
            top_n: 选择 Top N 只股票
            model_version: 模型版本号，None 表示使用最新版本
            models_dir: 模型目录
            signal_confidence_gate_enabled: 是否启用信号置信度门控，默认False
            signal_confidence_gate_top_k: 置信度评估使用的头部股票数量，默认10
            signal_confidence_gate_thresholds: 置信度阈值列表，低于首档时持币
            signal_confidence_gate_exposure_levels: 各阈值对应的仓位系数列表
            signal_gate_mode: 门控模式 "legacy"|"composite"|"disabled"
            signal_gate_cost_multiplier: composite模式下预测收益至少覆盖成本的倍数
            signal_gate_round_trip_cost: 往返交易成本估算
            signal_gate_percentile_warmup: 百分位归一化预热期（调仓次数）
            min_amount_ma20: 20日均成交额下限（千元），默认50000（=5000万元）
            min_total_mv: 总市值下限（万元），默认500000（=50亿元）
            max_total_mv: 总市值上限（万元），默认15000000（=1500亿元）
            exclude_financial: 是否剔除金融股（银行/非银金融），默认True
            verbose: 是否输出详细日志，默认True
        """
        super().__init__("ml_signal")
        self.top_n = top_n
        self.model_version = model_version
        self.models_dir = models_dir or get_models_root()
        self.signal_confidence_gate_enabled = signal_confidence_gate_enabled
        self.signal_confidence_gate_top_k = signal_confidence_gate_top_k
        self.signal_confidence_gate_thresholds = signal_confidence_gate_thresholds or [
            0.8,
            1.2,
            1.6,
        ]
        self.signal_confidence_gate_exposure_levels = signal_confidence_gate_exposure_levels or [
            0.3,
            0.6,
            1.0,
        ]
        # composite 门控参数
        self.signal_gate_mode = signal_gate_mode
        self.signal_gate_cost_multiplier = signal_gate_cost_multiplier
        self.signal_gate_round_trip_cost = signal_gate_round_trip_cost
        self.signal_gate_percentile_warmup = signal_gate_percentile_warmup
        # composite 门控历史缓冲区（用于百分位归一化和自校准阈值）
        self._separation_history: List[float] = []
        self._composite_score_history: List[float] = []
        self._GATE_HISTORY_MAX_LEN = 60  # 最多保留60个调仓日的历史
        # 跨 split 持久化质量监控状态（walk-forward 复用 signal 实例时保留）
        self._persisted_quality_state: Optional[dict] = None
        # 条件式坏票分类器（v2，延迟加载）
        self._bad_pick_classifier = None

        self.min_amount_ma20 = min_amount_ma20
        self.min_total_mv = min_total_mv
        self.max_total_mv = max_total_mv
        self.exclude_financial = exclude_financial
        self.verbose = verbose
        # 延迟加载模型
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.registry = None  # 复用 registry 实例
        self._last_confidence_gate_state = SignalConfidenceGateState()

        self._validate_confidence_gate_params()

        confidence_gate_info = ""
        if self.signal_gate_mode == "composite":
            confidence_gate_info = (
                f", gate=composite(cost_mult={self.signal_gate_cost_multiplier}, "
                f"cost={self.signal_gate_round_trip_cost}, "
                f"top_k={self.signal_confidence_gate_top_k}, "
                f"warmup={self.signal_gate_percentile_warmup})"
            )
        elif self.signal_confidence_gate_enabled:
            confidence_gate_info = (
                ", confidence_gate="
                f"enabled(top_k={self.signal_confidence_gate_top_k}, "
                f"thresholds={self.signal_confidence_gate_thresholds}, "
                f"exposures={self.signal_confidence_gate_exposure_levels})"
            )

        logger.info(
            f"ML 信号初始化: top_n={top_n}, model_version={model_version}, "
            f"min_amount_ma20={min_amount_ma20:.0f}千元, "
            f"total_mv=[{min_total_mv/10000:.0f}亿,{max_total_mv/10000:.0f}亿], "
            f"exclude_financial={exclude_financial}{confidence_gate_info}"
        )

    def _validate_confidence_gate_params(self) -> None:
        """校验信号置信度门控参数。"""
        if self.signal_gate_mode not in ("legacy", "composite", "disabled"):
            raise ValueError(
                f"signal_gate_mode 必须为 'legacy'/'composite'/'disabled'，"
                f"当前值: {self.signal_gate_mode}"
            )

        if self.signal_gate_mode == "composite":
            if self.signal_gate_cost_multiplier <= 0:
                raise ValueError(
                    f"signal_gate_cost_multiplier 必须为正数，"
                    f"当前值: {self.signal_gate_cost_multiplier}"
                )
            if self.signal_gate_round_trip_cost <= 0:
                raise ValueError(
                    f"signal_gate_round_trip_cost 必须为正数，"
                    f"当前值: {self.signal_gate_round_trip_cost}"
                )

        if self.signal_confidence_gate_top_k <= 0:
            raise ValueError(
                "signal_confidence_gate_top_k 必须为正整数，"
                f"当前值: {self.signal_confidence_gate_top_k}"
            )

        # legacy 模式需要校验阈值和仓位列表
        if self.signal_gate_mode == "legacy":
            if len(self.signal_confidence_gate_thresholds) == 0:
                raise ValueError(
                    "legacy 模式下 signal_confidence_gate_thresholds 不能为空列表"
                )
            if len(self.signal_confidence_gate_exposure_levels) == 0:
                raise ValueError(
                    "legacy 模式下 signal_confidence_gate_exposure_levels 不能为空列表"
                )
            if len(self.signal_confidence_gate_thresholds) != len(
                self.signal_confidence_gate_exposure_levels
            ):
                raise ValueError(
                    "signal_confidence_gate_thresholds 与 "
                    "signal_confidence_gate_exposure_levels 长度必须一致"
                )

            for i in range(1, len(self.signal_confidence_gate_thresholds)):
                if (
                    self.signal_confidence_gate_thresholds[i]
                    <= self.signal_confidence_gate_thresholds[i - 1]
                ):
                    raise ValueError(
                        "signal_confidence_gate_thresholds 必须严格递增: "
                        f"{self.signal_confidence_gate_thresholds}"
                    )

            last_exposure = -1.0
            for exposure in self.signal_confidence_gate_exposure_levels:
                if exposure < 0 or exposure > 1:
                    raise ValueError(
                        "signal_confidence_gate_exposure_levels 必须在 [0, 1] 范围内: "
                        f"{self.signal_confidence_gate_exposure_levels}"
                    )
                if exposure < last_exposure:
                    raise ValueError(
                        "signal_confidence_gate_exposure_levels 必须非递减: "
                        f"{self.signal_confidence_gate_exposure_levels}"
                    )
                last_exposure = exposure

    def _get_primary_task(self) -> str:
        """获取主模型任务类型。"""
        if self.metadata is None:
            return "regression"
        return self.metadata.get("train_params", {}).get("task", "regression")

    def _get_label_transform(self) -> str:
        """获取标签变换类型（影响预测分数的尺度）。"""
        if self.metadata is None:
            return "raw"
        return self.metadata.get("train_params", {}).get("label_transform", "raw")

    def _calculate_confidence_gate_state(
        self, ranked_candidates: List[tuple], date: Optional[pd.Timestamp] = None
    ) -> SignalConfidenceGateState:
        """根据排序候选计算置信度门控状态，按 signal_gate_mode 路由。"""
        # disabled 模式：完全跳过门控
        if self.signal_gate_mode == "disabled":
            state = SignalConfidenceGateState(
                enabled=False, exposure=1.0, reason="门控已禁用(disabled)"
            )
            self._last_confidence_gate_state = state
            return state

        # composite 模式：使用新公式
        if self.signal_gate_mode == "composite":
            return self._compute_composite_gate_state(ranked_candidates, date=date)

        # legacy 模式：保持原有逻辑
        if not self.signal_confidence_gate_enabled:
            state = SignalConfidenceGateState(enabled=False, exposure=1.0, reason="未启用")
            self._last_confidence_gate_state = state
            return state

        return self._compute_legacy_gate_state(ranked_candidates, date=date)

    def _compute_legacy_gate_state(
        self, ranked_candidates: List[tuple], date: Optional[pd.Timestamp] = None
    ) -> SignalConfidenceGateState:
        """legacy 模式：原有的置信度门控逻辑。"""
        score_values = np.asarray([score for _, score in ranked_candidates], dtype=float)
        score_values = score_values[np.isfinite(score_values)]

        if len(score_values) == 0:
            state = SignalConfidenceGateState(
                enabled=True,
                score=0.0,
                exposure=0.0,
                candidate_count=0,
                top_k=0,
                reason="无有效候选分数，持币",
            )
            self._last_confidence_gate_state = state
            return state

        top_k = min(self.signal_confidence_gate_top_k, len(score_values))
        top_scores = score_values[:top_k]

        if len(score_values) > top_k:
            baseline_scores = score_values[top_k : min(len(score_values), top_k * 2)]
            if len(baseline_scores) == 0:
                baseline_scores = score_values[top_k:]
        else:
            baseline_scores = score_values

        top_mean = float(np.mean(top_scores))
        baseline_mean = float(np.mean(baseline_scores))
        score_std = float(np.std(score_values))

        confidence_score = 0.0
        reason = "分数离散度过低，持币"
        if score_std > 1e-12:
            confidence_score = max(0.0, (top_mean - baseline_mean) / score_std)
            reason = (
                f"score={confidence_score:.3f}, top_mean={top_mean:.4f}, "
                f"baseline={baseline_mean:.4f}, std={score_std:.4f}"
            )

        if self._get_primary_task() == "regression" and top_mean <= 0:
            confidence_score = 0.0
            reason = f"Top{top_k} 平均分={top_mean:.4f} <= 0，视为无正向alpha，持币"

        exposure = 0.0
        hit_threshold = None
        for threshold, level in zip(
            self.signal_confidence_gate_thresholds,
            self.signal_confidence_gate_exposure_levels,
        ):
            if confidence_score >= threshold:
                exposure = level
                hit_threshold = threshold
            else:
                break

        if hit_threshold is None:
            reason = f"{reason}，未达到首档阈值 " f"{self.signal_confidence_gate_thresholds[0]:.3f}"
        else:
            reason = f"{reason}，达到阈值 {hit_threshold:.3f}，" f"目标仓位 {exposure:.0%}"

        state = SignalConfidenceGateState(
            enabled=True,
            score=confidence_score,
            exposure=exposure,
            candidate_count=len(score_values),
            top_k=top_k,
            top_mean=top_mean,
            baseline_mean=baseline_mean,
            score_std=score_std,
            hit_threshold=hit_threshold,
            reason=reason,
        )
        self._last_confidence_gate_state = state
        return state

    def _compute_composite_gate_state(
        self, ranked_candidates: List[tuple], date: Optional[pd.Timestamp] = None
    ) -> SignalConfidenceGateState:
        """composite 模式：成本门控 + 绝对质量分 + 百分位归一化 + 自校准阈值。"""
        score_values = np.asarray([score for _, score in ranked_candidates], dtype=float)
        score_values = score_values[np.isfinite(score_values)]

        if len(score_values) == 0:
            state = SignalConfidenceGateState(
                enabled=True,
                score=0.0,
                exposure=0.0,
                candidate_count=0,
                top_k=0,
                cost_gate_passed=False,
                reason="无有效候选分数，持币",
            )
            self._last_confidence_gate_state = state
            return state

        top_k = min(self.signal_confidence_gate_top_k, len(score_values))
        top_scores = score_values[:top_k]

        if len(score_values) > top_k:
            baseline_end = min(len(score_values), top_k * 2)
            baseline_scores = score_values[top_k:baseline_end]
            if len(baseline_scores) == 0:
                baseline_scores = score_values[top_k:]
        else:
            baseline_scores = score_values

        top_mean = float(np.mean(top_scores))
        baseline_mean = float(np.mean(baseline_scores))
        score_std = float(np.std(score_values))

        # 判断标签尺度：cs_zscore 标签的预测值是无量纲 z 分数，不能直接与交易成本比较
        is_zscore_label = self._get_label_transform() == "cs_zscore"
        is_regression = self._get_primary_task() == "regression"

        # ── 方案1: 成本门控（硬门槛）──
        cost_gate_passed = True
        if is_regression:
            if is_zscore_label:
                # z 分数尺度：cost_multiplier 直接表示 top_mean 必须超过 score_std 的倍数
                # 例如 cost_multiplier=0.3 表示 top_mean > 0.3 × score_std
                # 典型范围：0.1（宽松）~ 1.0（严格）
                cost_threshold = self.signal_gate_cost_multiplier * (score_std + 1e-8)
                if top_mean < cost_threshold:
                    cost_gate_passed = False
                    reason = (
                        f"成本门控未通过(z分数模式): top_mean={top_mean:.4f} < "
                        f"{self.signal_gate_cost_multiplier}×std={cost_threshold:.4f}，"
                        f"信号强度不足(std={score_std:.4f})，持币"
                    )
            else:
                # 原始收益尺度：直接与交易成本比较
                cost_threshold = self.signal_gate_cost_multiplier * self.signal_gate_round_trip_cost
                if top_mean < cost_threshold:
                    cost_gate_passed = False
                    reason = (
                        f"成本门控未通过: top_mean={top_mean:.4f} < "
                        f"{self.signal_gate_cost_multiplier}×{self.signal_gate_round_trip_cost}"
                        f"={cost_threshold:.4f}，预期收益不足以覆盖交易成本，持币"
                    )

        if not cost_gate_passed:
            state = SignalConfidenceGateState(
                enabled=True,
                score=0.0,
                exposure=0.0,
                candidate_count=len(score_values),
                top_k=top_k,
                top_mean=top_mean,
                baseline_mean=baseline_mean,
                score_std=score_std,
                abs_quality_score=0.0,
                cost_gate_passed=False,
                composite_score=0.0,
                reason=reason,
            )
            self._last_confidence_gate_state = state
            return state

        # ── 方案3A: 绝对收益质量分 ──
        # 用 sigmoid 风格映射，让 cost_multiplier 控制"半满分"位置：
        # - 当 top_mean == cost_multiplier × 基准尺度 时，abs_quality_score ≈ 1.0（半满分）
        # - top_mean 越高于该门槛，越趋近 2.0；越低于该门槛，越趋近 0
        # - 不同 cost_multiplier 值移动这条 S 曲线，始终产生有区分度的连续值
        if is_zscore_label:
            midpoint = self.signal_gate_cost_multiplier * (score_std + 1e-8)
        elif self.signal_gate_round_trip_cost > 0:
            midpoint = self.signal_gate_cost_multiplier * self.signal_gate_round_trip_cost
        else:
            midpoint = 1e-8
        # sigmoid 映射: 2 / (1 + exp(-k*(x/midpoint - 1)))
        # k 控制曲线陡峭度，k=3 时在 midpoint 附近有良好区分度
        ratio = top_mean / (midpoint + 1e-8)
        abs_quality_score = float(2.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0))))

        # ── 方案3B: 分离度百分位归一化 ──
        separation = top_mean - baseline_mean
        self._separation_history.append(separation)
        if len(self._separation_history) > self._GATE_HISTORY_MAX_LEN:
            self._separation_history = self._separation_history[-self._GATE_HISTORY_MAX_LEN :]

        # 预热期不足时使用保守默认
        if len(self._separation_history) >= self.signal_gate_percentile_warmup:
            hist_arr = np.asarray(self._separation_history)
            # 当前separation在历史中的百分位（0~1）
            sep_percentile = float(np.mean(hist_arr <= separation))
        else:
            # 预热期: 基于分离度正负做简单判断
            sep_percentile = 0.5 if separation >= 0 else 0.3

        # ── 综合得分 ──
        composite_score = 0.5 * abs_quality_score + 0.5 * sep_percentile
        self._composite_score_history.append(composite_score)
        if len(self._composite_score_history) > self._GATE_HISTORY_MAX_LEN:
            self._composite_score_history = self._composite_score_history[
                -self._GATE_HISTORY_MAX_LEN :
            ]

        # ── 方案3C: 自校准阈值（基于历史分位决定仓位）──
        if len(self._composite_score_history) >= self.signal_gate_percentile_warmup:
            hist_scores = np.asarray(self._composite_score_history)
            score_pct = float(np.mean(hist_scores <= composite_score))
            if score_pct < 0.20:
                exposure = 0.0  # 低于20%分位 → 持币
            elif score_pct < 0.50:
                exposure = 0.5  # 20%-50%分位 → 半仓
            else:
                exposure = 1.0  # 50%以上 → 满仓
        else:
            # 预热期: 只做成本门控（已在上方通过），其余放行
            exposure = 1.0

        reason_parts = [
            f"composite={composite_score:.3f}",
            f"abs_quality={abs_quality_score:.3f}",
            f"sep_pct={sep_percentile:.2f}",
            f"top_mean={top_mean:.4f}",
            f"baseline={baseline_mean:.4f}",
        ]
        if exposure <= 0:
            reason_parts.append("低于20%分位，持币")
        elif exposure < 1.0:
            reason_parts.append(f"中等分位，仓位{exposure:.0%}")
        else:
            reason_parts.append("满仓通过")
        reason = "，".join(reason_parts)

        state = SignalConfidenceGateState(
            enabled=True,
            score=composite_score,
            exposure=exposure,
            candidate_count=len(score_values),
            top_k=top_k,
            top_mean=top_mean,
            baseline_mean=baseline_mean,
            score_std=score_std,
            abs_quality_score=abs_quality_score,
            separation_percentile=sep_percentile,
            composite_score=composite_score,
            cost_gate_passed=cost_gate_passed,
            reason=reason,
        )
        self._last_confidence_gate_state = state
        return state

    def evaluate_confidence_gate(
        self, ranked_candidates: List[tuple], date: Optional[pd.Timestamp] = None
    ) -> SignalConfidenceGateState:
        """对排序候选重新评估一次置信度门控。"""
        return self._calculate_confidence_gate_state(ranked_candidates, date=date)

    def apply_confidence_gate_to_weights(
        self,
        signals: Dict[str, float],
        confidence_state: Optional[SignalConfidenceGateState] = None,
        date: Optional[pd.Timestamp] = None,
        emit_log: bool = True,
    ) -> Dict[str, float]:
        """将置信度门控结果应用到最终权重，允许留出现金。"""
        if not signals:
            return signals

        state = confidence_state or self._last_confidence_gate_state
        if not state.enabled:
            return signals

        date_label = date.date() if isinstance(date, pd.Timestamp) else date

        if state.exposure <= 0:
            if emit_log:
                logger.warning(f"信号置信度门控: {date_label}, {state.reason}，本次持币")
            return {}

        if state.exposure < 1.0:
            if emit_log:
                logger.warning(
                    f"信号置信度门控: {date_label}, {state.reason}，"
                    f"仓位缩放到 {state.exposure:.0%}，剩余资金持币"
                )
            return {stock: weight * state.exposure for stock, weight in signals.items()}

        if emit_log:
            logger.info(f"信号置信度门控: {date_label}, {state.reason}，满仓通过")
        return signals

    def get_last_confidence_gate_state(self) -> SignalConfidenceGateState:
        """返回最近一次评估的置信度门控状态。"""
        return self._last_confidence_gate_state

    def update_model_version(self, new_version: int) -> None:
        """切换到新模型版本（walk-forward 跨 split 复用时调用）。

        仅重置模型相关缓存，保留门控历史缓冲区（_separation_history、
        _composite_score_history），以便百分位归一化和自校准阈值跨 split 积累。

        Args:
            new_version: 新模型版本号
        """
        if self.model_version == new_version and self.model is not None:
            return  # 版本未变，无需切换
        old_version = self.model_version
        self.model_version = new_version
        # 重置模型缓存，下次 generate 时触发延迟加载
        self.model = None
        self.metadata = None
        self.feature_columns = None
        logger.info(
            f"MLSignal 切换模型: v{old_version} → v{new_version}，"
            f"门控历史保留（separation={len(self._separation_history)}条，"
            f"composite={len(self._composite_score_history)}条）"
        )

    def _load_model(self) -> None:
        """加载模型（延迟加载）"""
        if self.model is None:
            if self.model_version is None:
                raise RuntimeError(
                    "MLSignal.model_version 为 None，请在 generate 前调用 update_model_version()"
                )
            logger.info("开始加载ML模型...")
            self.registry = ModelRegistry(models_dir=self.models_dir)
            # 严格检查：拒绝旧模型
            self.model, self.metadata = self.registry.load_model(
                version=self.model_version, strict_version_check=True
            )
            self.feature_columns = self.metadata["feature_columns"]
            risk_config = self.metadata.get("risk_penalty_config") or {}
            risk_enabled = bool(risk_config.get("enabled", False))
            if risk_enabled:
                risk_version = int(risk_config.get("version", 1))
                if risk_version >= 2:
                    # 条件式坏票模型（v2+）
                    bp_model_ver = int(risk_config.get("bad_pick_model_version", 0) or 0)
                    auc = float(risk_config.get("calibration_auc", 0.0) or 0)
                    regimes = risk_config.get("regime_configs") or {}
                    regime_info = ", ".join(
                        f"{n}:thr={c.get('threshold',0):.2f},lam={c.get('penalty_lambda',0):.3f}"
                        for n, c in regimes.items()
                    )
                    logger.warning(
                        f"模型条件式坏票惩罚已启用: AUC={auc:.3f}, "
                        f"bad_rate={float(risk_config.get('calibration_bad_rate', 0.0)):.2%}, "
                        f"classifier=v{bp_model_ver}, "
                        f"regimes={{{regime_info}}}"
                    )
                    # 延迟加载坏票分类器（直接用 XGBoost 原生加载，绕过注册表类型检测）
                    if bp_model_ver > 0:
                        try:
                            import xgboost as xgb
                            clf_file = self.models_dir / f"v{bp_model_ver}_model.json"
                            if not clf_file.exists():
                                raise FileNotFoundError(f"分类器模型文件不存在: {clf_file}")
                            clf = xgb.XGBClassifier()
                            clf.load_model(str(clf_file))
                            self._bad_pick_classifier = clf
                            logger.info(f"坏票分类器已加载: v{bp_model_ver}")
                        except Exception as exc:
                            logger.warning(f"坏票分类器加载失败: {exc}，惩罚将跳过")
                            self._bad_pick_classifier = None
                else:
                    # 旧版线性惩罚（v1）
                    top_features = ", ".join(
                        f"{item.get('name')}:{float(item.get('weight', 0.0)):.2f}"
                        for item in (risk_config.get("feature_weights") or [])[:3]
                    )
                    logger.warning(
                        f"模型风险惩罚已启用(v1): lambda={float(risk_config.get('penalty_lambda', 0.0) or 0.0):.3f}, "
                        f"bad_rate={float(risk_config.get('calibration_bad_rate', 0.0)):.2%}, "
                        f"top_features={top_features or '无'}"
                    )
            else:
                logger.info("模型风险惩罚未启用")
            logger.info(
                f"模型已加载: {self.metadata['version_str']}, "
                f"特征数={self.metadata['feature_count']}"
            )

    def _apply_selection_filters(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, int]:
        """选股阶段过滤（实盘/回测共用）

        规则（均使用特征文件中的原始列，z-score 归一化不影响这些列）：
        - 20日均成交额 >= min_amount_ma20（千元，默认50000=5000万）
        - 总市值 in [min_total_mv, max_total_mv]（万元，默认50亿~1500亿）
        - 剔除金融股（申万一级 银行801780/非银金融801790）

        Args:
            features_df: 特征DataFrame

        Returns:
            过滤后的DataFrame、过滤前数量、过滤后数量
        """
        if len(features_df) == 0:
            return features_df, 0, 0

        before = len(features_df)
        mask = pd.Series(True, index=features_df.index)

        # ── 成交额：amount_ma20 原始列（千元）────────────────────────────
        if "amount_ma20" in features_df.columns:
            amount_low = (features_df["amount_ma20"].fillna(0) < self.min_amount_ma20).sum()
            if amount_low > 0:
                # logger.info(
                #    f"  选股过滤-成交额: 剔除 amount_ma20 < {self.min_amount_ma20:.0f}千元"
                #    f"（={self.min_amount_ma20 / 10:.0f}万元）的 {amount_low} 只"
                # )
                pass
            mask &= features_df["amount_ma20"].fillna(0) >= self.min_amount_ma20
        else:
            logger.warning("选股过滤-成交额: amount_ma20 列不存在，跳过")

        # ── 市值：total_mv 原始列（万元）────────────────────────────────
        if "total_mv" in features_df.columns:
            mv_low = (features_df["total_mv"] < self.min_total_mv).sum()
            mv_high = (features_df["total_mv"] > self.max_total_mv).sum()
            if mv_low + mv_high > 0:
                # logger.info(
                #   f"  选股过滤-市值: 剔除 <{self.min_total_mv / 10000:.0f}亿 {mv_low}只, "
                #   f">{self.max_total_mv / 10000:.0f}亿 {mv_high}只"
                # )
                pass
            mask &= features_df["total_mv"].between(self.min_total_mv, self.max_total_mv)
        else:
            logger.warning("选股过滤-市值: total_mv 列不存在，跳过")

        # ── 金融股：sw_l1_code────────────────────────────────────────────
        if self.exclude_financial:
            if "sw_l1_code" not in features_df.columns:
                logger.warning(
                    "选股过滤-金融股: sw_l1_code 列不存在（申万行业数据未加载），跳过此规则"
                )
            else:
                fin_mask = features_df["sw_l1_code"].isin(self._FINANCIAL_SW_L1_CODES)
                fin_count = fin_mask.sum()
                # if fin_count > 0:
                #    logger.info(f"  选股过滤-金融股: 剔除银行/非银金融 {fin_count} 只")
                mask &= ~fin_mask

        result = features_df[mask].copy()
        return result, before, len(result)

    def _log_prediction_pipeline_summary(
        self, before_count: int, after_count: int, ranked: bool = False
    ) -> None:
        """将选股过滤与模型预测入口压缩为单行日志。"""
        if ranked:
            return
        stage = "选股/预测(ranked)" if ranked else "选股/预测"
        universe_text = f"{before_count}→{after_count}" if before_count != after_count else str(after_count)
        logger.info(f"{stage}: {universe_text}, 特征{len(self.feature_columns)}")

    @staticmethod
    def _rank_risk_feature(series: pd.Series) -> pd.Series:
        """将单日截面风险特征映射到 0~1 分位。"""
        valid = series.dropna()
        if len(valid) == 0:
            return pd.Series(0.5, index=series.index, dtype=float)
        ranked = series.rank(method="average", pct=True, na_option="keep")
        return ranked.fillna(0.5).clip(0.0, 1.0)

    def _apply_risk_penalty(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """按模型 metadata 中的风险配置生成 final_score。

        支持两种模式：
        - v1（旧版线性加权）: final_score = ml_score - λ × Σ(w_i × quantile_i)
        - v2（条件式坏票）: 市场状态检测 + 二分类器 + 阈值门控
        """
        features_df["risk_score"] = 0.0
        features_df["final_score"] = features_df["ml_score"]

        risk_config = (self.metadata or {}).get("risk_penalty_config") or {}
        if not risk_config.get("enabled"):
            return features_df, "ml_score"

        risk_version = int(risk_config.get("version", 1))

        if risk_version >= 2:
            # ── v2: 条件式坏票模型 ──
            from src.lazybull.risk.bad_pick import BadPickConfig, apply_conditional_penalty

            bp_config = BadPickConfig.from_dict(risk_config)
            if not bp_config.enabled or bp_config.bad_pick_model_version <= 0:
                return features_df, "ml_score"
            if self._bad_pick_classifier is None:
                return features_df, "ml_score"

            return apply_conditional_penalty(features_df, bp_config, self._bad_pick_classifier)

        # ── v1: 旧版线性加权惩罚（保留兼容）──
        penalty_lambda = float(risk_config.get("penalty_lambda", 0.0) or 0.0)
        feature_weights = risk_config.get("feature_weights") or []
        if penalty_lambda <= 0 or len(feature_weights) == 0:
            return features_df, "ml_score"

        risk_score = pd.Series(0.0, index=features_df.index, dtype=float)
        total_weight = 0.0
        for item in feature_weights:
            feature_name = item.get("name")
            weight = float(item.get("weight", 0.0) or 0.0)
            if not feature_name or weight <= 0:
                continue
            feature_series = (
                self._rank_risk_feature(features_df[feature_name])
                if feature_name in features_df.columns
                else pd.Series(0.5, index=features_df.index, dtype=float)
            )
            risk_score += feature_series * weight
            total_weight += weight

        if total_weight <= 0:
            return features_df, "ml_score"

        if total_weight != 1.0:
            risk_score = risk_score / total_weight

        features_df["risk_score"] = risk_score
        features_df["final_score"] = (
            features_df["ml_score"] - penalty_lambda * features_df["risk_score"]
        )
        return features_df, "final_score"

    def generate(self, date: pd.Timestamp, universe: List[str], data: Dict) -> Dict[str, float]:
        """生成 ML 信号

        Args:
            date: 当前日期
            universe: 股票池（股票代码列表）
            data: 数据字典，应包含 "features" 键，值为当日特征 DataFrame

        Returns:
            信号字典，{股票代码: 权重}
        """
        # 加载模型
        self._load_model()

        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            logger.info(
                f"data columns: {data['daily'].columns.tolist() if 'daily' in data else 'N/A'}"
            )
            return {}

        features_df = data["features"]

        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return {}

        # 过滤股票池（布尔索引已创建新对象，无需 .copy()）
        features_df = features_df[features_df["ts_code"].isin(universe)]

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return {}

        # 应用选股过滤（成交额/市值/金融股）
        features_df, filter_before_count, filter_after_count = self._apply_selection_filters(
            features_df
        )

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return {}

        # 特征列一致性检查 (缺失列以 NaN 自动补齐, 由 XGBoost/LightGBM 原生 NaN 处理)
        # 适用于另类因子 (如 north_flow / consensus) 在历史早期无数据导致的列缺失
        required_features = set(self.metadata.get("feature_columns", []))
        missing = required_features - set(features_df.columns)
        if missing:
            import numpy as np
            logger.debug(
                f"推理特征缺失 {len(missing)} 列, 自动补 NaN: "
                f"{sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}"
            )
            for col in missing:
                features_df[col] = np.nan
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return {}

        # XGB/LGB 原生支持 NaN，不做 fillna，保留缺失值信息

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get("train_params", {}).get("task", "regression")
        self._log_prediction_pipeline_summary(
            before_count=filter_before_count,
            after_count=filter_after_count,
            ranked=False,
        )

        if task == "classification" and hasattr(self.model, "predict_proba"):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == "classification":
                logger.warning(
                    f"模型声明为 classification，但无 predict_proba 方法，回退到 predict"
                )

        features_df["ml_score"] = predictions
        features_df, score_column = self._apply_risk_penalty(features_df)

        # 按预测分数排序，选择 Top N
        features_df = features_df.sort_values(score_column, ascending=False)
        ranked_candidates = list(
            zip(features_df["ts_code"].tolist(), features_df[score_column].tolist())
        )
        confidence_state = self.evaluate_confidence_gate(ranked_candidates, date=date)
        top_stocks = features_df.head(self.top_n)
        if self.verbose:
            logger.debug(
                "TOP预测概率抽样: {}".format(
                    features_df[["ts_code", "ml_score", "risk_score", "final_score"]]
                    .head(3)
                    .to_string(index=False)
                    .replace("\n", " | ")
                )
            )

        if len(top_stocks) == 0:
            logger.warning(f"{date.date()} 没有有效的预测结果")
            return {}

        # 输出原始 ml_score（供引擎层 _normalize_signals 统一做权重分配）
        # 正分数原样输出；全为负/零时回退到等权（避免引擎层收到无意义负值）
        positive_stocks = top_stocks[top_stocks[score_column] > 0]
        if len(positive_stocks) == 0:
            weight = 1.0 / len(top_stocks)
            signals = {stock: weight for stock in top_stocks["ts_code"].tolist()}
        else:
            scores = positive_stocks[score_column].values
            stocks = positive_stocks["ts_code"].values
            signals = dict(zip(stocks, scores.tolist()))

        signals = self.apply_confidence_gate_to_weights(
            signals, confidence_state=confidence_state, date=date
        )

        logger.debug(
            f"ML 信号生成完成: {date.date()}, 选择 {len(signals)} 只股票, "
            f"平均预测分数={top_stocks[score_column].mean():.6f}"
        )

        return signals

    def generate_ranked(self, date: pd.Timestamp, universe: List[str], data: Dict) -> List[tuple]:
        """生成排序后的候选股票列表（支持回填）

        返回所有候选股票的完整排序列表，而不仅仅是 top N。
        这样可以在 top N 中有不可交易股票时从后续候选中回填。

        Args:
            date: 当前日期
            universe: 股票池
            data: 数据字典，应包含 "features" 键

        Returns:
            排序后的 (股票代码, 预测分数) 元组列表，按分数降序排列
        """
        # 加载模型
        self._load_model()

        # 获取当日特征数据
        if "features" not in data:
            logger.warning(f"{date.date()} 没有特征数据")
            return []

        features_df = data["features"]

        if features_df is None or len(features_df) == 0:
            logger.warning(f"{date.date()} 特征数据为空")
            return []

        # 过滤股票池（布尔索引已创建新对象，无需 .copy()）
        features_df = features_df[features_df["ts_code"].isin(universe)]

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 股票池没有匹配的特征数据")
            return []

        # 应用选股过滤（成交额/市值/金融股）
        features_df, filter_before_count, filter_after_count = self._apply_selection_filters(
            features_df
        )

        if len(features_df) == 0:
            logger.warning(f"{date.date()} 选股过滤后无可选股票")
            return []

        # 特征列一致性检查 (缺失列以 NaN 自动补齐, 由 XGBoost/LightGBM 原生 NaN 处理)
        # 适用于另类因子 (如 north_flow / consensus) 在历史早期无数据导致的列缺失
        required_features = set(self.metadata.get("feature_columns", []))
        missing = required_features - set(features_df.columns)
        if missing:
            import numpy as np
            logger.debug(
                f"推理特征缺失 {len(missing)} 列, 自动补 NaN: "
                f"{sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}"
            )
            for col in missing:
                features_df[col] = np.nan
        available_features = features_df.columns.tolist()
        try:
            self.registry.check_feature_consistency(self.metadata, available_features)
        except ValueError as e:
            logger.error(f"特征列一致性检查失败: {e}")
            raise

        # 准备特征（XGBoost 不修改输入，无需 .copy()）
        try:
            X = features_df[self.feature_columns]
        except KeyError as e:
            logger.error(f"特征列缺失: {e}")
            return []

        # XGB/LGB 原生支持 NaN，不做 fillna

        # 预测（classification 模型使用 predict_proba 获取正类概率）
        task = self.metadata.get("train_params", {}).get("task", "regression")
        self._log_prediction_pipeline_summary(
            before_count=filter_before_count,
            after_count=filter_after_count,
            ranked=True,
        )

        if task == "classification" and hasattr(self.model, "predict_proba"):
            # 分类模型：使用正类概率作为分数
            predictions = self.model.predict_proba(X)[:, 1]  # 取正类（标签=1）的概率
            if self.verbose:
                logger.debug(f"使用 classification 模型预测概率（正类）作为分数")
        else:
            # 回归模型：使用预测值作为分数
            predictions = self.model.predict(X)
            if self.verbose and task == "classification":
                logger.warning(
                    f"模型声明为 classification，但无 predict_proba 方法，回退到 predict"
                )

        features_df["ml_score"] = predictions
        features_df, score_column = self._apply_risk_penalty(features_df)

        # 按预测分数排序，返回所有候选
        features_df = features_df.sort_values(score_column, ascending=False)
        if self.verbose:
            logger.debug(
                "TOP预测概率抽样: {}".format(
                    features_df[["ts_code", "ml_score", "risk_score", "final_score"]]
                    .head(3)
                    .to_string(index=False)
                    .replace("\n", " | ")
                )
            )

        # 返回 (股票代码, 分数) 元组列表
        ranked = list(zip(features_df["ts_code"].tolist(), features_df[score_column].tolist()))
        self.evaluate_confidence_gate(ranked, date=date)
        if False:
            logger.info(
                f"  ML排序候选生成: {date.date()}, "  # 候选数 {len(ranked)}, "
                f"平均预测分数[{features_df['ml_score'].mean():.3f}], "
                f"最高/最低[{features_df['ml_score'].max():.3f}/{features_df['ml_score'].min():.3f}]"
            )

        # 缓存本次完整排序候选列表，供持仓强势度评分（_score_holding_strength）查询 ml_raw_score
        # 与回测行为对齐：回测 engine 在 _generate_signal 末尾缓存 _last_ranked_candidates
        self._last_ranked_candidates = ranked

        return ranked

    def generate_with_features(
        self, date: pd.Timestamp, universe: List[str], features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """使用提供的特征数据生成信号（便捷方法）

        Args:
            date: 当前日期
            universe: 股票池
            features_df: 特征 DataFrame

        Returns:
            信号字典
        """
        data = {"features": features_df}
        return self.generate(date, universe, data)

    def get_model_info(self) -> Dict:
        """获取模型信息

        Returns:
            模型元数据字典
        """
        self._load_model()
        return self.metadata

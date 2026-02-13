# Copyright 2025 Valeo.

"""Reward functions for the simulator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from vmax.simulator import metrics, operations
from vmax.simulator.wrappers.base import Wrapper


# ========= 工具：解析/规范化 YAML 的 reward_config =========
def _normalize_reward_config(cfg: dict) -> dict[str, dict]:
    """
    输入可以是：
      - {"overlap": 2.0, "below_ttc": 1.0, ...}
      - {"overspeed": {"weight": 0.8, "overspeed_threshold_mps": 2.23}, ...}
    统一转为：
      {"name": {"weight": float, ...额外参数}}
    """
    out = {}
    for name, spec in (cfg or {}).items():
        if isinstance(spec, (int, float)):
            out[name] = {"weight": float(spec)}
        elif isinstance(spec, dict):
            if "weight" not in spec:
                raise ValueError(f"reward_config['{name}'] must include 'weight' when using dict form.")
            # 拷贝一份，避免原地修改
            d = dict(spec)
            d["weight"] = float(d["weight"])
            out[name] = d
        else:
            raise TypeError(f"reward_config['{name}'] must be number or dict, got {type(spec)}")
    return out


# ========= 线性加权 =========
class RewardLinearWrapper(Wrapper):
    """线性组合： sum_i w_i * r_i"""

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        super().__init__(env)
        self._cfg = _normalize_reward_config(reward_config)

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        total = 0.0
        for name, spec in self._cfg.items():
            w = spec["weight"]
            fn = _get_reward_fn(name)
            r = fn(state, **{k: v for k, v in spec.items() if k != "weight"})
            total = total + w * jnp.asarray(r, dtype=jnp.float32)
        return jnp.asarray(total, dtype=jnp.float32)


# ========= 乘法式（几何组合） =========
class RewardCustomWrapper(Wrapper):
    """
    几何组合：在 log 空间做线性和，再指数回去
        logR = sum_i w_i * log(score_i)
        R    = exp(logR)

    score 映射（无新超参）：
      - 违规布尔 (True=违规):       score = 1 - v    ∈ {0,1}
      - log_div (>=0):             score = 1 / (1+v) ∈ (0,1]
      - log_div_clip (bool pass):  score = v         ∈ {0,1}
      - progression (bool):        score = 1 + v     ∈ {1,2}
      - comfort (>=0):             score = 1 + tanh(v) ∈ (1,2)
    """

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        super().__init__(env)
        self._cfg = _normalize_reward_config(reward_config)
        self._tiny = jnp.array(1e-4, dtype=jnp.float32)  # 防 log(0) train stable

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        logR = 0.0
        for name, spec in self._cfg.items():
            w = spec["weight"]
            fn = _get_reward_fn(name)
            raw = fn(state, **{k: v for k, v in spec.items() if k != "weight"})  # bool/float
            score = self._to_score(name, raw)
            score = jnp.clip(score, self._tiny, None)
            logR = logR + w * jnp.log(score)
        return jnp.exp(logR).astype(jnp.float32)

    @staticmethod
    def _to_score(name: str, v) -> jax.Array:
        """
        将各个原始指标 v（布尔或实数）统一映射成一个 score ≥ 0，用于“乘法式奖励”几何组合。

        几类规则：

        1. 违规类（overlap / offroad / red_light 等）：
           - v 是布尔值或 {0,1}：1 表示发生违规，0 表示没违规
           - 映射规则：score = 1 - α * v
             · 无违规 v=0 → score = 1.0
             · 有违规 v=1 → score = min_score（例如 0.2）
           - min_score 决定“犯错时最低得分”，再乘上权重 weight 之后体现严重程度。

        2. 连续质量类：
           - log_div:
               v ≥ 0 代表某种“误差 / 偏离”，越大越差
               映射：score = 1 / (1 + v)，v 越大 score 越接近 0
           - log_div_clip:
               v 是 {0,1}，1=通过，0=未通过
               映射：score = v（通过=1, 不通过=0）

        3. 正向加成类：
           - progression:
               v ∈ {0,1}，1=有进展（沿路线前进），0=没进展
               映射：score = 1 + v ∈ {1,2}，有进展比原地不动多一个“倍数因子”
           - comfort:
               v ≥ 0 为“舒适度代价”或 jerk/加速度等
               映射：score = 1 + tanh(max(v,0)) ∈ (1,2)，越舒适（代价小）越接近 1，
                     代价越大 tanh→1，score 上限约 2

        4. 其他未列出 name：
           - 返回 1.0，表示对总 reward 是“中性因子”（乘上去不改变整体）。

        说明：
        外层 RewardCustomWrapper 使用的是几何组合：
            logR = Σ_i w_i * log(score_i)
            R    = exp(logR)
        所以：
            - score_i > 1  → 对总奖励是正向放大
            - score_i = 1  → 中性
            - 0 < score_i < 1 → 惩罚（成倍数缩小）
        """
        v_arr = jnp.asarray(v, dtype=jnp.float32)
        # ---- 违规类（True=违规） → 1 - v ----
        if name in (
            "overlap",
            "offroad",
            "off_route",
            "below_ttc",
            "red_light",
            "overspeed",
            "lane_deviation",
            "driving_direction",
        ):
                v_bool = jnp.asarray(v, dtype=jnp.float32)  # 违规=1

                # 统一形式：score = 1 - α * v   (α = 1 - min_score)
                # 无违规：score = 1
                # 有违规：score = min_score
                min_score = jnp.array(0.2, dtype=jnp.float32)  # min_score 越小，惩罚越重
                return jnp.where(v_arr > 0.0, min_score, 1.0)

        # ---- 连续质量 ----
        if name == "log_div":
            v_float = jnp.maximum(v_arr, 0.0)
            return 1.0 / (1.0 + v_float)# v=0 → 1；v 越大 → 越接近 0

        if name == "log_div_clip":
            # 通过=1，失败=0.5（给一点惩罚，避免 0）
            return jnp.where(v_arr > 0.0, 1.0, 0.5)
        # ---- 正向加成 ----
        if name == "progression":
            # 不前进：1.0；前进：1 + beta > 1
            beta = 1.0
            good = jnp.clip(v_arr, 0.0, 1.0)
            return 1.0 + beta * good

        if name == "comfort":
            # 假设 v 大表示舒适，tanh(v) ∈ (-1,1)
            # score ∈ [1,2)
            return 1.0 + jnp.tanh(v_arr)


        # 未知项：中性
        return jnp.array(1.0, dtype=jnp.float32)


# ========= registry：所有 reward 项都以 (state, **params) 形式 =========
def _get_reward_fn(reward_name: str):
    table = {
        "log_div_clip": _compute_log_divergence_clip_reward,
        "log_div": _compute_log_divergence_reward,

        "overlap": _compute_overlap_reward,
        "offroad": _compute_offroad_reward,
        "off_route": _compute_off_route_reward,
        "below_ttc": _compute_below_ttc_reward,
        "red_light": _compute_red_light_reward,
        "overspeed": _compute_overspeed_limit_reward,
        "driving_direction": _compute_driving_direction_reward,
        "lane_deviation": _compute_deviate_lane_reward,

        "progression": _compute_making_progress_reward,
        "comfort": _compute_comfort_reward,
        "safe_bubble": _compute_safe,
    }
    if reward_name not in table:
        raise ValueError(f"Reward function {reward_name} not implemented.")
    return table[reward_name]


# =========================
# Penalty-style (mostly bool)
# =========================

def _compute_overlap_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    overlap = waymax_metrics.OverlapMetric().compute(state).value
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree_util.tree_map(lambda x: x[sdc_idx], overlap)
    return jnp.asarray(sdc_overlap == 1.0, dtype=jnp.float32)


def _compute_offroad_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    offroad = waymax_metrics.OffroadMetric().compute(state).value
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree_util.tree_map(lambda x: x[sdc_idx], offroad)
    return jnp.asarray(sdc_offroad == 1.0, dtype=jnp.float32)


def _compute_red_light_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value
    return jnp.asarray(has_runned_red_light, dtype=jnp.float32)


def _compute_overspeed_limit_reward(
    state: datatypes.SimulatorState,
    overspeed_threshold_mps: float = 2.23,  # ~5 mph
    **_,
) -> jax.Array:
    speed_limit = metrics.infer_speed_limit_from_simulator_state(state)
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()
    return jnp.asarray(ego_speed > speed_limit + overspeed_threshold_mps, dtype=jnp.float32)


def _compute_below_ttc_reward(
    state: datatypes.SimulatorState,
    ttc_threshold_s: float = 1.5,
    **_,
) -> jax.Array:
    ttc = metrics.TimeToCollisionMetric().compute(state).value
    return jnp.asarray(ttc < ttc_threshold_s, dtype=jnp.float32)


def _compute_off_route_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    off_route_score = metrics.OffRouteMetric().compute(state).value  # 0/on-route；>0/off-route
    return jnp.asarray(off_route_score > 0, dtype=jnp.float32)


def _compute_driving_direction_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    driving_direction_metric = metrics.DrivingDirectionComplianceMetric().compute(state).value
    return jnp.asarray(driving_direction_metric > 0, dtype=jnp.float32)


def _compute_deviate_lane_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    on_multiple_lanes_metric = metrics.OnMultipleLanesMetric().compute(state).value
    return jnp.asarray(on_multiple_lanes_metric > 0, dtype=jnp.float32)


# =========================
# Reward-style (continuous)
# =========================

def _compute_log_divergence_clip_reward(
    state: datatypes.SimulatorState,
    log_div_threshold: float = 0.3,
    **_,
) -> jax.Array:
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)
    return jnp.asarray(log_divergence < log_div_threshold, dtype=jnp.float32)


def _compute_making_progress_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value
    return jnp.asarray(current_progression > previous_progression, dtype=jnp.float32)


def _compute_log_divergence_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)
    return jnp.asarray(log_divergence, dtype=jnp.float32)


def _compute_comfort_reward(state: datatypes.SimulatorState, **_) -> jax.Array:
    comfort_metric_reward_2 = metrics.ComfortMetric().compute_reward(state).value
    return jnp.asarray(comfort_metric_reward_2, dtype=jnp.float32)

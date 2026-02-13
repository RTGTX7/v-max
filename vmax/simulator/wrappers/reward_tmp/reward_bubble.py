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
               raise ValueError(
                   f"reward_config['{name}'] must include 'weight' when using dict form."
               )
           d = dict(spec)
           d["weight"] = float(d["weight"])
           out[name] = d
       else:
           raise TypeError(
               f"reward_config['{name}'] must be number or dict, got {type(spec)}"
           )
   return out




# ========= 线性加权 =========
class RewardLinearWrapper(Wrapper):
   """线性组合： sum_i w_i * r_i"""


   def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
       super().__init__(env)
       self._cfg = _normalize_reward_config(reward_config)


   def reward(
       self, state: datatypes.SimulatorState, action: datatypes.Action
   ) -> jax.Array:
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
     - safe_bubble (∈[0,1]):      score = 1 / (1 + v^2)  （越靠近 ego，reward 衰减越快）
   """


   def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
       super().__init__(env)
       self._cfg = _normalize_reward_config(reward_config)
       self._tiny = jnp.finfo(jnp.float32).tiny  # 防 log(0)


   def reward(
       self, state: datatypes.SimulatorState, action: datatypes.Action
   ) -> jax.Array:
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
           return 1.0 - v_bool  # 安全=1，违规=0


       # ---- 连续质量 ----
       if name == "log_div":
           v_float = jnp.asarray(v, dtype=jnp.float32)
           v_float = jnp.maximum(v_float, 0.0)
           return 1.0 / (1.0 + v_float)


       if name == "log_div_clip":
           v_bool_good = jnp.asarray(v, dtype=jnp.float32)  # 通过=1，未过=0
           return v_bool_good


       # ---- Safe Bubble：0=安全，1=深度侵入泡泡 ----
       if name == "safe_bubble":
           v_float = jnp.asarray(v, dtype=jnp.float32)
           v_float = jnp.clip(v_float, 0.0, 1.0)
           # 0 → score=1.0（无惩罚）
           # 1 → score=0.9（最多只衰减 10%）
           return 1.0 - 0.1 * v_float


       # ---- 正向加成 ----
       if name == "progression":
           good = jnp.asarray(v, dtype=jnp.float32)  # {0,1}
           return 1.0 + good


       if name == "comfort":
           v_float = jnp.asarray(v, dtype=jnp.float32)
           return 1.0 + jnp.tanh(jnp.maximum(v_float, 0.0))


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


       # === NEW: safe bubble ===
       "safe_bubble": _compute_safe_bubble_reward,
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


def _compute_safe_bubble_reward(
    state: datatypes.SimulatorState,
    p_base: float = 3.0,
    anisotropy: float = 1.25,
    **_,
) -> jax.Array:
    """
    返回 v_depth ∈ [0, 1]：Safe Bubble 侵入深度（已压缩）
      0   = 完全在泡泡外 / 很安全
      0.3 = 泡泡内但比较余裕
      1.0 = 深度侵入（危险）
    """

    # ------- 1. Ego 速度 -------
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()  # m/s
    ego_speed = jnp.asarray(ego_speed, dtype=jnp.float32)
    v_kmh = ego_speed * 3.6

    # ------- 2. a_cmd 先固定 0 -------
    a_cmd = jnp.array(0.0, dtype=jnp.float32)

     # ---------- 基本量 ----------
    v_norm = jnp.clip(v_kmh / 120.0, 0.0, 1.0)

    # 起步阶段左右观察
    v_start = 10.0
    w_start = jnp.clip(1.0 - v_kmh / v_start, 0.0, 1.0)

    a_norm = jnp.tanh(a_cmd / 2.0)   # -1~1

    # ---------- 注视权重 ----------
    w_front_raw = 0.4 + 0.4 * v_norm + 0.2 * jnp.maximum(a_norm, 0.0)
    w_back_raw  = 0.2 + 0.3 * jnp.maximum(-a_norm, 0.0)
    w_side_raw  = 1.0 - (w_front_raw + w_back_raw)
    w_side_raw  = jnp.maximum(w_side_raw, 0.0)
    w_side_raw  = w_side_raw + 0.5 * w_start

    w_sum = w_front_raw + w_side_raw + w_back_raw + 1e-6
    w_front = w_front_raw / w_sum
    w_side  = w_side_raw  / w_sum
    w_back  = w_back_raw  / w_sum

    # ---------- 前向尺度（你的新版本：更小的车距） ----------
    # 0 km/h → b_front_base ≈ 2m
    # 80 km/h → b_front_base ≈ 2 + 0.15*80 = 14m
    b_front_base = 2.0 + 0.15 * v_kmh
    b_front_base = jnp.clip(b_front_base, 2.0, 30.0)

    b_front = b_front_base * (0.8 + 0.4 * w_front) * (1.0 - 0.3 * w_start)

    # ---------- 后向尺度（比前向短很多） ----------
    # 0 km/h → b_back_base ≈ 0.5m
    # 前向 14m 时 → 后向 ≈ 0.5 + 0.2*14 = 3.3m
    b_back_base = 0.5 + 0.2 * b_front_base
    b_back = b_back_base * (0.25 + 0.4 * w_back)

    # ---------- 侧向 ----------
    lane_width = 3.6
    car_width  = 1.9
    car_half   = car_width / 2.0

    base_margin = 0.2
    v_norm2 = jnp.clip(v_kmh / 120.0, 0.0, 1.0)

    extra_margin = (0.15 + 0.7 * w_side) * v_norm2 * car_half
    side_radius = car_half + base_margin + extra_margin

    side_radius = jnp.clip(
        side_radius,
        car_half + 0.15,
        lane_width,
    )

    a_lat = side_radius / jnp.maximum(anisotropy, 1e-3)

    # ---------- 前向“深度”：用 TTC * v vs b_front ----------
    ttc = metrics.TimeToCollisionMetric().compute(state).value
    ttc = jnp.asarray(ttc, dtype=jnp.float32)
    ttc = jnp.where(ttc < 0.0, 0.0, ttc)
    ttc = jnp.where(jnp.isfinite(ttc), ttc, 1e6)

    front_dist_est = ego_speed * ttc
    front_depth_raw = jnp.maximum(b_front - front_dist_est, 0.0) / (b_front + 1e-6)
    # clamp 到 [0,1]
    front_depth = jnp.clip(front_depth_raw, 0.0, 1.0)

    # ---------- 侧向“深度”：OnMultipleLanes / a_lat ----------
    lateral_raw = metrics.OnMultipleLanesMetric().compute(state).value
    lateral_raw = jnp.asarray(lateral_raw, dtype=jnp.float32)

    lateral_depth_raw = lateral_raw / (a_lat + 1e-6)
    # 比前向弱一些：0.5 系数
    lateral_depth = jnp.clip(0.5 * lateral_depth_raw, 0.0, 1.0)

    # ---------- 总深度 ----------
    v_depth = jnp.maximum(front_depth, lateral_depth)
    # 最终保证在 [0,1]
    v_depth = jnp.clip(v_depth, 0.0, 1.0)

    return v_depth

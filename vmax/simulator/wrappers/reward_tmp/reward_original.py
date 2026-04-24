# Copyright 2025 Valeo.

"""Legacy baseline reward variant.

This file is retained for comparison only. The active reward implementation used
by the environment is exposed through ``vmax.simulator.wrappers.reward``.
"""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from vmax.simulator import metrics, operations
from vmax.simulator.wrappers.base import Wrapper


class RewardLinearWrapper(Wrapper):
    """Wraps the environment to compute rewards using a linear combination of functions."""

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        """Initialize the reward linear wrapper.

        Args:
            env: The environment to wrap.
            reward_config: Configuration for reward computation.

        """
        super().__init__(env)
        self._reward_config = reward_config

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Combine rewards provided from the reward functions.
        The reward is computed as a linear combination of the individual rewards,
        weighted by their respective coefficients.

        Args:
            state: Current simulator state.

        Returns:
            Combined reward value.
        """
        reward = 0.0

        for reward_name, reward_weigth in self._reward_config.items():
            reward_fn = _get_reward_fn(reward_name)
            reward += reward_fn(state) * reward_weigth

        return jnp.array(reward, dtype=jnp.float32)


class RewardCustomWrapper(Wrapper):
    """Wraps the environment to compute rewards using a custom function."""

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Compute a custom reward.

        This is a placeholder function, use it as you wish o7

        Args:
            state: Current simulator state.
            action: Action taken.

        Returns:
            Reward value.

        """
        # Implement your custom reward logic here
        return jnp.array(0.0)


def _get_reward_fn(reward_name: str) -> callable:
    """Retrieve a reward function by its name.

    Args:
        reward_name: Name identifier of the reward function.

    Returns:
        Callable reward function.

    """
    reward_dict = {
        "log_div_clip": _compute_log_divergence_clip_reward,
        "log_div": _compute_log_divergence_reward,
        "overlap": _compute_overlap_reward,
        "offroad": _compute_offroad_reward,
        "off_route": _compute_off_route_reward,
        "below_ttc": _compute_below_ttc_reward,
        "red_light": _compute_red_light_reward,
        "comfort": _compute_comfort_reward,
        "overspeed": _compute_overspeed_limit_reward,
        "driving_direction": _compute_driving_direction_reward,
        "lane_deviation": _compute_deviate_lane_reward,
        "progression": _compute_making_progress_reward,
    }

    if reward_name not in reward_dict:
        raise ValueError(f"Reward function {reward_name} not implemented.")

    return reward_dict[reward_name]


# Penalty based rewards


def _compute_overlap_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing overlaps between the SDC and other objects.

    Args:
        state: Current simulator state.

    Returns:
        True if overlap detected, False otherwise.

    """
    overlap = waymax_metrics.OverlapMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree_util.tree_map(lambda x: x[sdc_idx], overlap)

    return sdc_overlap == 1.0


def _compute_offroad_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing when the SDC drives off the road.

    Args:
        state: Current simulator state.

    Returns:
        True if off-road detected, False otherwise.

    """
    offroad = waymax_metrics.OffroadMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree_util.tree_map(lambda x: x[sdc_idx], offroad)

    return sdc_offroad == 1.0


def _compute_red_light_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing red light violations.

    Args:
        state: Current simulator state.

    Returns:
        True if red light violation detected, False otherwise.

    """
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value

    return has_runned_red_light


def _compute_overspeed_limit_reward(state: datatypes.SimulatorState, threshold: float = 2.23) -> bool:
    """Compute a reward based on speed-limit adherence.

    Returns True if the SDC's speed exceeds the road's speed limit by more than
    2.23 m/s (approximately 5 mph). m/s * 3.6 = km/h

    Args:
        state: Current simulator state.

    Returns:
        True if speed limit is exceeded by more than 2.23 m/s, False otherwise.

    """
    speed_limit = metrics.infer_speed_limit_from_simulator_state(state)
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()

    return ego_speed > speed_limit + threshold


def _compute_below_ttc_reward(state: datatypes.SimulatorState, threshold: float = 1.5) -> bool:
    """Compute a reward based on time-to-collision with other objects.

    The time-to-collision (TTC) is a measure of how long it would take for the SDC to
    collide with another object if both continue on their current trajectories.

    Args:
        state: Current simulator state.
        threshold: Minimum safe time-to-collision in seconds. Default is 1.5s.

    Returns:
        True if TTC is below threshold (unsafe), False otherwise (safe).

    """
    ttc = metrics.TimeToCollisionMetric().compute(state).value

    return ttc < threshold


def _compute_off_route_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing deviation from the planned route.

    Returns True if the SDC has deviated from its planned route. A deviation occurs
    when the SDC's position has a non-zero distance to the nearest point on the route.

    Args:
        state: Current simulator state.

    Returns:
        True if off-route detected, False otherwise.
    """
    # Waymax offroute's score: 0 if you're on-route, distance to route if you're offroute
    off_route_score = metrics.OffRouteMetric().compute(state).value

    return off_route_score > 0


def _compute_driving_direction_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward for driving direction compliance.

    This reward is based on the driving direction compliance metric, which evaluates
    whether the SDC is following the intended driving direction.

    Args:
        state: Current simulator state.

    Returns:
        True if driving direction is compliant, False otherwise.

    """
    driving_direction_metric = metrics.DrivingDirectionComplianceMetric().compute(state).value

    return driving_direction_metric > 0


def _compute_deviate_lane_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward penalizing lane deviations.

    This reward is based on the lane deviation metric, which evaluates how much the SDC
    deviates from its intended lane.

    Args:
        state: Current simulator state.

    Returns:
        True if lane deviation is detected, False otherwise.

    """
    on_multiple_lanes_metric = metrics.OnMultipleLanesMetric().compute(state).value

    return on_multiple_lanes_metric > 0


# Reward based rewards


def _compute_log_divergence_clip_reward(state: datatypes.SimulatorState, threshold: float = 0.3) -> bool:
    """Compute reward based on whether log divergence exceeds a threshold.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.
        threshold: Maximum acceptable divergence (in log space) before considering it a deviation.
                  Default is 0.3, higher values are more permissive.

    Returns:
        True if divergence exceeds threshold, False otherwise.
    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence < threshold


def _compute_making_progress_reward(state: datatypes.SimulatorState) -> bool:
    """Compute a reward promoting forward progress along the route.

    Compares the current progression metric with the previous timestep to determine
    if the SDC is making forward progress along its intended route.

    Args:
        state: Current simulator state.

    Returns:
        True if the SDC made forward progress, False otherwise.
    """
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value

    return current_progression > previous_progression


# Linar reward functions


def _compute_log_divergence_reward(state: datatypes.SimulatorState) -> float:
    """Compute reward based on log divergence.

    The log divergence measures how much the SDC's trajectory deviates from the expected path.

    Args:
        state: Current simulator state.

    Returns:
        Log divergence value.

    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence


def _compute_comfort_reward(state: datatypes.SimulatorState) -> float:
    """Compute a comfort-related reward.

    This reward is based on the comfort metric, which evaluates the smoothness of the SDC's
    trajectory. A higher comfort metric indicates a smoother and more comfortable ride.

    Args:
        state: Current simulator state.

    Returns:
        Comfort metric value.

    """
    comfort_metric_reward_2 = metrics.ComfortMetric().compute_reward(state).value

    return comfort_metric_reward_2

def _compute_safe_bubble_reward(
    state: datatypes.SimulatorState,
    # ---- bubble 尺度（TTC-boundary）----
    ttc_front: float = 1.2,         # 建议从 1.2 开始（你之前 1.5 更保守）
    ttc_rear: float = 1.0,
    d_static: float = 1.5,
    d_static_rear: float = 1.0,
    clip_front: float = 45.0,
    clip_rear: float = 25.0,
    rear_frac: float = 0.6,

    # ---- bubble 形状（superellipse）----
    anisotropy: float = 1.0,        # 侧向缩放
    p_base: float = 2.0,
    delta_p: float = 0.8,

    # ---- 距离惩罚（你要的：0.1m 最强，2m 以内指数衰减到 1）----
    threshold_m: float = 2.0,       # 2m 内开始惩罚
    dist_min: float = 0.1,          # 0.1m 以内视为最严重
    min_score: float = 0.2,         # 最小 score（用于乘法项）
    # 是否只在 bubble 内计算（推荐 True）
    gate_by_bubble: bool = True,
    # bubble gating 放宽一点（避免边界数值抖动）
    bubble_margin: float = 0.10,
    **_,
) -> jax.Array:
    """
    返回 score ∈ [min_score, 1]：
      - 若没有对象触发风险：score=1
      - 若存在对象 clearance < threshold_m：score 按指数衰减，clearance<=dist_min -> min_score

    说明：
    - clearance 是“近似几何间隙”，不是 TTC。
    - bubble 用于决定“哪些对象值得考虑”（可关/可开）。
    """

    # ========== 取 SDC 索引 ==========
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)

    # ========== 取当前时刻轨迹（Waymax traj） ==========
    traj = state.current_sim_trajectory

    # 位置/朝向/速度
    ego_xy = jnp.stack([traj.x[sdc_idx], traj.y[sdc_idx]], axis=-1).squeeze()          # (2,)
    ego_yaw = traj.yaw[sdc_idx].squeeze()
    ego_speed = traj.speed[sdc_idx].squeeze()                                          # m/s

    # ego 尺寸（用于近似几何间隙）
    ego_length = traj.length[sdc_idx].squeeze()
    ego_width  = traj.width[sdc_idx].squeeze()
    ego_rad = 0.5 * jnp.sqrt(ego_length**2 + ego_width**2)

    # ========== 其他对象 mask ==========
    valid = traj.valid.squeeze()                     # (N,)
    is_sdc = state.object_metadata.is_sdc.squeeze()  # (N,)
    others = valid & (~is_sdc)

    # 若没有其他对象：返回 1
    def _no_obj():
        return jnp.array(1.0, dtype=jnp.float32)

    def _has_obj():
        # ========== 取其他对象位置/尺寸 ==========
        obj_xy = jnp.stack([traj.x, traj.y], axis=-1)          # (N,2)
        rel = obj_xy - ego_xy                                  # (N,2)

        # 旋转到 ego 坐标系（x: 右，y: 前）
        c = jnp.cos(-ego_yaw)
        s = jnp.sin(-ego_yaw)
        R = jnp.array([[c, -s], [s, c]], dtype=jnp.float32)
        rel_ego = (rel @ R.T).astype(jnp.float32)              # (N,2)
        x = rel_ego[:, 0]
        y = rel_ego[:, 1]

        # 对象半径近似
        obj_rad = 0.5 * jnp.sqrt(traj.length.squeeze()**2 + traj.width.squeeze()**2)

        # ========== 计算 bubble 参数（TTC-boundary + 侧向半径） ==========
        # 侧向半径：用车宽+margin（你也可改成你那套 w_side 逻辑）
        lane_width = jnp.array(3.6, dtype=jnp.float32)
        base_margin = jnp.array(0.2, dtype=jnp.float32)
        car_half = ego_width * 0.5
        side_radius = jnp.clip(car_half + base_margin, car_half + 0.15, lane_width)
        a_lat = side_radius / jnp.maximum(anisotropy, 1e-3)

        # 前后 longitudinal base（TTC-boundary）
        front_base = d_static + ego_speed * ttc_front
        front_base = jnp.clip(front_base, d_static, clip_front)

        rear_base = d_static_rear + ego_speed * ttc_rear
        rear_base = jnp.minimum(rear_base, rear_frac * front_base)
        rear_base = jnp.clip(rear_base, d_static_rear, clip_rear)

        # superellipse 形状参数
        p_front = jnp.clip(p_base + delta_p, 1.0, 5.0)
        p_back  = jnp.clip(p_base - delta_p, 1.0, 5.0)

        # ========== bubble gating：计算是否在 bubble 内/附近 ==========
        # s = (|x|/a)^p + (|y|/b)^p  （前后 b 不同；用 y 的符号选）
        ax = jnp.maximum(a_lat, 1e-3)
        absx = jnp.abs(x)

        # front/back 分开
        y_pos = jnp.maximum(y, 0.0)
        y_neg = jnp.maximum(-y, 0.0)

        s_front = (absx / ax) ** p_front + (y_pos / jnp.maximum(front_base, 1e-3)) ** p_front
        s_back  = (absx / ax) ** p_back  + (y_neg / jnp.maximum(rear_base,  1e-3)) ** p_back

        # 只对“前方用 front，后方用 back”的那部分生效
        s_val = jnp.where(y >= 0.0, s_front, s_back)

        # margin：允许略微超出 bubble 也纳入（减少边缘抖动）
        bubble_ok = s_val <= (1.0 + bubble_margin)

        # ========== 计算 clearance（近似几何间隙） ==========
        # center distance - radii
        center_dist = jnp.linalg.norm(rel, axis=-1)  # (N,) 直接世界系距离也行（不依赖朝向）
        clearance = center_dist - (ego_rad + obj_rad)

        # 只考虑 others
        clearance = jnp.where(others, clearance, jnp.inf)
        if gate_by_bubble:
            clearance = jnp.where(bubble_ok & others, clearance, jnp.inf)

        # 取最危险的对象
        dmin = jnp.min(clearance)  # 可能是 inf

        # ========== 从 clearance -> score（指数衰减，2m 内开始惩罚） ==========
        # 规则：
        #   d >= threshold -> 1
        #   d <= dist_min  -> min_score
        #   中间：score = exp(-λ*(threshold - d))，并保证到 dist_min 时为 min_score
        lam = -jnp.log(jnp.maximum(min_score, 1e-6)) / jnp.maximum(threshold_m - dist_min, 1e-3)

        # delta = threshold - d
        delta = threshold_m - dmin
        score_mid = jnp.exp(-lam * delta)

        score = jnp.where(dmin >= threshold_m, 1.0, score_mid)
        score = jnp.where(dmin <= dist_min, min_score, score)
        score = jnp.clip(score, min_score, 1.0)
        return score.astype(jnp.float32)

    # 如果没有其他对象（others 全 False），直接 1
    has_any = jnp.any(others)
    return jax.lax.cond(has_any, _has_obj, _no_obj)

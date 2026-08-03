# vmax/simulator/metrics/safe_bubble.py
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from waymax import datatypes

from vmax.simulator import constants, operations

from typing import Tuple
def _estimate_sdc_accel_mps2(state: datatypes.SimulatorState, sdc_idx: int) -> jax.Array:
    dt = jnp.array(constants.TIME_DELTA, dtype=jnp.float32)

    def _a0():
        v = state.current_sim_trajectory.speed[sdc_idx].squeeze()
        return jnp.array(0.0, dtype=jnp.float32) * v

    def _a1():
        v_t = state.current_sim_trajectory.speed[sdc_idx].squeeze()
        prev_state = state.replace(timestep=state.timestep - 1)
        v_tm1 = prev_state.current_sim_trajectory.speed[sdc_idx].squeeze()
        a = (v_t - v_tm1) / jnp.maximum(dt, 1e-6)
        return a.astype(jnp.float32)

    return jax.lax.cond(state.timestep > 0, _a1, _a0)


def bubble_params(
    speed_kmh: jax.Array,
    accel_mps2: jax.Array,
    ego_width_m: jax.Array,
    *,
    # longitudinal
    ttc_front: float,
    ttc_rear: float,
    d_static: float,
    d_static_rear: float,
    clip_front: float,
    clip_rear: float,
    rear_frac: float,
    # lateral
    side_margin_min: float,
    side_margin_max: float,
    anisotropy: float,
    # superellipse
    p_base: float,
    delta_p: float,
        # NEW: parked rear bubble
    parked_rear_m: float = 6.0,   # v≈0 时后方至少 6m（你可以调）
    v_stop_kmh: float = 1,      # 低于这个算“停车/几乎停车”
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (a_lat, b_front, b_back, p_front, p_back) in meters."""
    v_mps = jnp.array(speed_kmh / 3.6, jnp.float32)
    a = jnp.array(accel_mps2, jnp.float32)

    # --- base longitudinal radii (meters) ---
    front_base = d_static + v_mps * float(ttc_front)
    front_base = jnp.clip(front_base, d_static, float(clip_front))

    rear_base = d_static_rear + v_mps * float(ttc_rear)
    rear_base = jnp.minimum(rear_base, float(rear_frac) * front_base)
    rear_base = jnp.clip(rear_base, d_static_rear, float(clip_rear))

    # --- accel-aware attention (soft, no hard gating) ---
    # a_norm in [-1,1], brake->negative, accel->positive
    a_norm = jnp.tanh(a / 2.0)

    # speed factor: very low speed reduces "front attention" (起步前不盯前方)
    v_norm = jnp.clip(jnp.array(speed_kmh / 15.0, jnp.float32), 0.0, 1.0)  # 0~15km/h transition

    # front grows when accelerating; rear grows when braking
    front_gain = 1.0 + 0.35 * v_norm * jnp.maximum(a_norm, 0.0)   # accel -> up to +35%
    rear_gain  = 1.0 + 0.65 * jnp.maximum(-a_norm, 0.0)           # brake -> up to +65%

    # additionally: at very low speed, front radius slightly shrinks (but never to zero)
    front_low_speed = 0.7 + 0.3 * v_norm                          # v=0 ->0.7, v>=15 ->1.0

    b_front = front_base * front_gain * front_low_speed
    is_stop = jnp.array(speed_kmh <= v_stop_kmh)
    b_back = jnp.where(is_stop, jnp.maximum(rear_base, parked_rear_m), rear_base) * rear_gain

    # lateral radius (meters)
    car_half = jnp.asarray(ego_width_m, jnp.float32) * 0.5
    v_norm = jnp.clip(jnp.asarray(speed_kmh, jnp.float32) / 120.0, 0.0, 1.0)
    side_margin = side_margin_min + (side_margin_max - side_margin_min) * v_norm
    side_radius = car_half + side_margin
    a_lat = side_radius / jnp.maximum(jnp.array(anisotropy, jnp.float32), 1e-3)

    # superellipse exponent
    p_front = jnp.clip(jnp.array(p_base + delta_p, jnp.float32), 1.0, 6.0)
    p_back  = jnp.clip(jnp.array(p_base - delta_p, jnp.float32), 1.0, 6.0)

    return a_lat, b_front, b_back, p_front, p_back

def bubble_score_from_xy(
    *,
    x: jnp.ndarray,
    y: jnp.ndarray,
    a_lat: jnp.ndarray,
    b_front: jnp.ndarray,
    b_back: jnp.ndarray,
    p_front: jnp.ndarray,
    p_back: jnp.ndarray,
    min_score: float,
    penetration_max_m: float,
) -> jnp.ndarray:
    """
    Pure penalty curve. NO speed logic, NO gating.
    """

    ax = jnp.maximum(a_lat, 1e-3)
    b  = jnp.where(y >= 0.0, b_front, b_back)
    p  = jnp.where(y >= 0.0, p_front, p_back)

    pnorm = ((jnp.abs(x) / ax) ** p + (jnp.abs(y) / b) ** p) ** (1.0 / p)

    scale_m = jnp.sqrt(ax * b)
    signed_m = (pnorm - 1.0) * scale_m

    pen_m = jnp.maximum(-signed_m, 0.0)
    k = -jnp.log(jnp.maximum(min_score, 1e-6)) / jnp.maximum(penetration_max_m, 1e-3)

    score = jnp.exp(-k * pen_m)
    return jnp.clip(score, min_score, 1.0)

def _score_from_signed_m_inside_only(
    signed_m: jax.Array,
    *,
    min_score: float,
    penetration_max_m: float,
):
    """
    signed_m < 0 inside bubble (meters). Outside (>=0) score=1.
    Inside: exp decay down to min_score over penetration_max_m.
    """
    pen_m = jnp.maximum(-signed_m, 0.0)
    min_s = jnp.array(min_score, jnp.float32)
    pen_max = jnp.maximum(jnp.array(penetration_max_m, jnp.float32), 1e-3)
    k = -jnp.log(jnp.maximum(min_s, 1e-6)) / pen_max
    score_in = jnp.exp(-k * pen_m)
    score_in = jnp.clip(score_in, min_s, 1.0)
    return jnp.where(pen_m > 0.0, score_in, 1.0)


def _masked_min_score(score: jax.Array, valid_non_sdc: jax.Array) -> jax.Array:
    """Return the worst score over valid non-ego object slots only."""
    masked_score = jnp.where(valid_non_sdc[..., None], score, 1.0)
    return jnp.min(masked_score).astype(jnp.float32)


def soft_safe_bubble_score(
    state: datatypes.SimulatorState,
    # ---- bubble shape knobs ----
    ttc_front: float = 3.0,
    ttc_rear: float = 1.5,
    d_static: float = 1.5,
    d_static_rear: float = 1.0,
    clip_front: float = 100.0,
    clip_rear: float = 40.0,
    rear_frac: float = 0.6,
    side_margin_min: float = 0.15,
    side_margin_max: float = 0.45,
    anisotropy: float = 1.0,
    p_base: float = 2.5,
    delta_p: float = 0.5,
    # ---- rollout ----
    time_horizon_s: float = 3.0,
    dt: float | None = None,
    # ---- penalty ----
    min_score: float = 0.20,
    penetration_max_m: float = 8.0,
    # ---- rules / gating ----
    v_front_on_kmh: float = 0.5,    # v > this enables FRONT penalty
    v_side_on_kmh: float = 0.5,      # v > this enables SIDE penalty
    v_rear_strict_kmh: float = 1e6,  # 0..5 km/h: REAR is strict (enabled)
    **_,
) -> jax.Array:
    """
    Simple rules:
      - v <= 0.5 km/h: only rear matters (front & side disabled)
      - 0.5 < v <= 5 km/h: rear matters + (front enabled) optional by thresholds above
      - v > 0.5 km/h: front starts; v > 0.5 km/h: side starts
    score in [min_score, 1], multiplicative-friendly.
    """
    dt_s = float(constants.TIME_DELTA if dt is None else dt)
    dt_arr = jnp.array(dt_s, dtype=jnp.float32)

    traj = state.current_sim_trajectory
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)

    ego_xy = jnp.stack([traj.x[sdc_idx], traj.y[sdc_idx]], axis=-1).squeeze()
    ego_yaw = traj.yaw[sdc_idx].squeeze()
    ego_vel = traj.vel_xy[sdc_idx].squeeze()
    ego_speed = traj.speed[sdc_idx].squeeze()
    ego_width = traj.width[sdc_idx].squeeze()

    v_kmh = (ego_speed * 3.6).astype(jnp.float32)
    a_cmd = jnp.clip(_estimate_sdc_accel_mps2(state, sdc_idx), -4.0, 4.0)

    a_lat, bF, bB, pF, pB = bubble_params(
        v_kmh, a_cmd, ego_width,
        ttc_front=ttc_front, ttc_rear=ttc_rear,
        d_static=d_static, d_static_rear=d_static_rear,
        clip_front=clip_front, clip_rear=clip_rear,
        rear_frac=rear_frac,
        side_margin_min=side_margin_min, side_margin_max=side_margin_max,
        anisotropy=anisotropy,
        p_base=p_base, delta_p=delta_p,
    )

    valid = traj.valid.squeeze()
    is_sdc = state.object_metadata.is_sdc.squeeze()
    others = valid & (~is_sdc)

    def _no_obj():
        return jnp.array(1.0, dtype=jnp.float32)
    def _has_obj():
        obj_xy = jnp.stack([traj.x, traj.y], axis=-1).squeeze()   # (N,2)
        obj_vel = traj.vel_xy.squeeze()                           # (N,2)

        rel0 = obj_xy - ego_xy                                    # (N,2)
        relv = obj_vel - ego_vel                                  # (N,2)

        num_steps = int(math.floor(float(time_horizon_s) / dt_s + 1e-6)) + 1
        t = jnp.arange(num_steps, dtype=jnp.float32) * dt_arr      # (T,)
        rel_t = rel0[:, None, :] + relv[:, None, :] * t[None, :, None]  # (N,T,2)

        # rotate to ego frame
        c = jnp.cos(-ego_yaw)
        s = jnp.sin(-ego_yaw)
        dx = rel_t[..., 0]
        dy = rel_t[..., 1]
        x = c * dx - s * dy
        y = s * dx + c * dy

        # -----------------------------
        # 1) rules / gating (scalar)
        # -----------------------------
        parked = v_kmh <= jnp.array(0.5, jnp.float32)

        front_on = v_kmh > jnp.array(float(v_front_on_kmh), jnp.float32)
        rear_on  = v_kmh <= jnp.array(float(v_rear_strict_kmh), jnp.float32)

        # parked: front off, rear on
        front_on = jnp.where(parked, False, front_on)
        rear_on  = jnp.where(parked, True,  rear_on)

        # side gating split (关键)
        side_on_front = v_kmh > jnp.array(float(v_side_on_kmh), jnp.float32)
        side_on_front = jnp.where(parked, False, side_on_front)

        side_on_rear  = v_kmh > jnp.array(float(v_side_on_kmh), jnp.float32)
        side_on_rear  = jnp.where(parked, True,  side_on_rear)   # parked rear keeps lateral

        # per-point direction allow (N,T)
        allow_dir = jnp.where(y >= 0.0, front_on, rear_on)

        # per-point lateral allow (N,T): front uses side_on_front, rear uses side_on_rear
        allow_lat = jnp.where(y >= 0.0, side_on_front, side_on_rear)

        # -----------------------------
        # 2) geometry (stable scale)
        # -----------------------------
        ax = jnp.maximum(a_lat, 1e-3)  # scalar, real

        b = jnp.where(y >= 0.0, jnp.maximum(bF, 1e-3), jnp.maximum(bB, 1e-3))
        p = jnp.where(y >= 0.0, pF, pB)

        absx = jnp.abs(x)
        absy = jnp.abs(y)

        # only drop x contribution where lateral is disabled
        absx = jnp.where(allow_lat, absx, jnp.zeros_like(absx))

        pnorm = ((absx / ax) ** p + (absy / b) ** p) ** (1.0 / p)

        scale_m = jnp.sqrt(ax * b)
        signed_m = (pnorm - 1.0) * scale_m

        # -----------------------------
        # 3) penalty + gating
        # -----------------------------
        score = _score_from_signed_m_inside_only(
            signed_m,
            min_score=min_score,
            penetration_max_m=penetration_max_m,
        )

        # direction off => neutral
        score = jnp.where(allow_dir, score, 1.0)

        return _masked_min_score(score, others)



    return jax.lax.cond(jnp.any(others), _has_obj, _no_obj)

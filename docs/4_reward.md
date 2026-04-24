# V-Max Reward Guide

This document explains how reward computation is organized in the repository, how the available reward terms behave, and where the project-specific reward extensions are implemented.

## 1. Reward Entry Points

Reward computation lives in simulator wrappers under:

- `vmax/simulator/wrappers/`
- `vmax/simulator/wrappers/reward_tmp/`

The two main patterns are:

- **linear reward composition**: weighted sum of reward terms
- **multiplicative reward composition**: reward terms mapped to scores and combined geometrically in log space

The canonical public reward-wrapper module is:

- `vmax/simulator/wrappers/reward.py`

That module currently re-exports the active project-specific implementation from:

- `vmax/simulator/wrappers/reward_tmp/reward_bubble2.py`
- `vmax/simulator/metrics/safe_bubble.py`

## 2. Reward Wrapper Modes

### Linear reward

`RewardLinearWrapper` computes:

```text
R = \sum_i w_i r_i
```

This is the standard weighted-sum design. It is simple and easy to tune locally, but positive terms can partially offset safety violations.

### Multiplicative reward

`RewardCustomWrapper` computes:

```text
log R = \sum_i w_i log(score_i)
R = exp(log R)
```

This makes each reward term behave like a multiplicative factor rather than a linear contribution. Safety-related scores below `1` shrink the total reward directly.

## 3. Score Mapping in Multiplicative Reward

The multiplicative wrapper converts raw metrics into nonnegative scores before combining them.

Important cases:

- violation-style terms such as `overlap`, `offroad`, `red_light`, `termination_switch`
  - safe: `1.0`
  - violated: `0.2`

- `off_route`
  - mapped as `1 / (1 + v)`

- `progression`
  - mapped as `1 + v`, where `v` is binary

- `forward_quality`
  - mapped as `1 + v`

- `comfort`
  - mapped as `1 + tanh(v)`

The exact mapping logic is implemented in:

- `vmax/simulator/wrappers/reward_tmp/reward_bubble2.py`

## 4. Reward Terms Available in the Wrapper

The reward registry currently includes:

- `overlap`
- `offroad`
- `off_route`
- `below_ttc`
- `red_light`
- `overspeed`
- `driving_direction`
- `lane_deviation`
- `log_div_clip`
- `log_div`
- `progression`
- `comfort`
- `forward_quality`
- `safe_bubble`
- `route_safe`
- `termination_switch`

These are all resolved through `_get_reward_fn(...)` in:

- `vmax/simulator/wrappers/reward_tmp/reward_bubble2.py`

## 5. Project-Specific Reward Structure

The current project-specific design uses three active multiplicative terms:

```text
R_t =
S_route_safe,t^alpha
*
S_termination_switch,t^beta
*
S_forward_quality,t^gamma
```

In the current default configuration:

- `route_safe = 0.5`
- `termination_switch = 0.8`
- `forward_quality = 1.2`

This configuration is defined through `reward_config` in:

- `vmax/config/base_config.yaml`

## 6. Safe Bubble and Route-Safe

### Safe bubble

`safe_bubble` is a continuous local safety score computed from a velocity-adaptive geometric envelope around the ego vehicle.

Implementation:

- `vmax/simulator/metrics/safe_bubble.py`

It uses:

- front and rear TTC-style longitudinal sizing
- velocity-adaptive lateral margin
- asymmetric superellipse geometry
- short-horizon relative-motion rollout
- exponential penalty based on bubble penetration depth

The output score is clipped into:

```text
[min_score, 1.0]
```

with `min_score = 0.2` by default.

### Route-safe

`route_safe` combines the safe-bubble score with smooth off-route suppression:

```text
route_safe = clip(safe_bubble * off_route_soft, min_score, 1.0)
```

where:

```text
off_route_soft = 1 / (1 + off_route_raw * off_route_beta)
```

This gives a single continuous score that penalizes both local safety intrusions and route deviation.

## 7. Terminal Safety Gate

`termination_switch` is a binary severe-violation signal that fires if any of the following occur:

- off-road
- overlap
- red-light violation

In multiplicative mode it maps to:

- `1.0` when safe
- `0.2` when violated

This acts as a hard safety gate without forcing the total reward to zero.

## 8. Forward Quality

`forward_quality` combines:

- binary making-progress signal
- continuous comfort signal

The implementation computes:

```text
forward_quality_raw = clip(progression * comfort, 0, 1)
score = 1 + forward_quality_raw
```

This encourages forward motion only when the motion remains reasonably smooth.

## 9. TTC vs Safe Bubble

The repository contains both TTC and safe-bubble logic, but they serve different purposes.

### TTC

`TimeToCollisionMetric` predicts the earliest future overlap time under constant-speed and constant-heading rollout.

Implementation:

- `vmax/simulator/metrics/ttc.py`

The reward term `below_ttc` is then a binary unsafe flag:

```text
below_ttc = 1 if TTC < threshold else 0
```

### Safe bubble

`safe_bubble` is not a TTC threshold. It defines a local protected region around the ego vehicle and penalizes predicted intrusions into that region continuously.

In short:

- TTC answers: "How soon until collision?"
- Safe bubble answers: "How deeply does a nearby agent intrude into the local safety envelope?"

## 10. Where to Modify Reward Behavior

To change reward composition:

- edit `reward_config` in `vmax/config/base_config.yaml`

To add a new reward term:

1. implement the metric function in `reward_bubble2.py` or a dedicated metric module
2. register it in `_get_reward_fn(...)`
3. define its score mapping in `RewardCustomWrapper._to_score(...)` if using multiplicative mode

To change safe-bubble geometry:

- edit `vmax/simulator/metrics/safe_bubble.py`

## 11. Practical Advice

- use the linear wrapper if you want a simpler baseline
- use the multiplicative wrapper when safety should act as a direct gating signal
- keep reward dimensions small enough to search systematically
- document the exact `reward_config` used for any reported result

For the current project, the multiplicative path is the one that matters most.

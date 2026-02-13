# V-Max Reward Function Documentation

## Overview

In V-Max, the reward function is a key component that guides the behavior of autonomous agents during simulation. Rewards are used to evaluate how well an agent is performing with respect to safety, comfort, rule compliance, and task completion. By designing appropriate reward functions, you can encourage agents to drive safely, follow traffic rules, and achieve their objectives efficiently.

## How Rewards Work in V-Max

V-Max uses a flexible reward system based on wrappers. The main reward wrapper is `RewardLinearWrapper`, which computes the total reward as a weighted sum of several individual reward functions. Each function measures a specific aspect of driving behavior (e.g., staying on the road, avoiding collisions, obeying speed limits).

You can configure which reward functions to use and their relative importance by providing a `reward_config` dictionary, where keys are reward function names and values are their weights.

## Configuring Rewards

To use the reward system, wrap your environment with `RewardLinearWrapper` and provide a configuration, for example:

```python
reward_config = {
    "overlap": -10.0,           # Penalize collisions
    "offroad": -5.0,            # Penalize driving off the road
    "progression": 1.0,         # Reward making progress
    "comfort": 0.5,             # Reward smooth driving
    # ... add more as needed
}

env = RewardLinearWrapper(env, reward_config)
```

Each time the agent takes an action, the wrapper computes the total reward as:

```
reward = sum(weight * reward_fn(state) for each reward_fn in config)
```

## Available Reward Functions

Here are the main reward functions you can use in your configuration:

| Name                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `overlap`           | Penalizes collisions with other objects                           |
| `offroad`           | Penalizes driving off the road                                   |
| `off_route`         | Penalizes deviating from the planned route                       |
| `below_ttc`         | Penalizes unsafe time-to-collision situations                    |
| `red_light`         | Penalizes running red lights                                     |
| `overspeed`         | Penalizes exceeding the speed limit                              |
| `driving_direction` | Penalizes driving in the wrong direction                         |
| `lane_deviation`    | Penalizes deviating from the intended lane                       |
| `log_div_clip`      | Penalizes large deviations from the expected trajectory           |
| `log_div`           | Measures log divergence from the expected trajectory              |
| `progression`       | Rewards making forward progress along the route                  |
| `comfort`           | Rewards smooth and comfortable driving                           |
# ✅ **NuPlan/V-Max Reward Functions 中文解释表**

| Reward 名称             | 中文含义              | 作用机制                                | 值的范围   | 为什么需要                     |
| --------------------- | ----------------- | ----------------------------------- | ------ | ------------------------- |
| **overlap**           | **碰撞惩罚**          | 如果 ego 与任何实体发生几何重叠 → 返回 1，否则 0      | {0,1}  | 避免撞车（最强惩罚）                |
| **offroad**           | **驶出道路惩罚**        | ego 有任意部分在车道外 → 1，否则 0              | {0,1}  | 防止跑到草地、路肩、建筑上             |
| **off_route**         | **偏离规划路线惩罚**      | 不在 route 上 → 1，否则 0                 | {0,1}  | 防止 agent 偏离路线、走错街         |
| **below_ttc**         | **低碰撞时间惩罚**       | TTC（碰撞时间）小于阈值 → 1                   | {0,1}  | 防止贴前车太近、危险接近              |
| **red_light**         | **闯红灯惩罚**         | 穿过红灯停止线 → 1                         | {0,1}  | 强制遵守信号灯                   |
| **overspeed**         | **超速惩罚**          | 当前速度 > 限速 + margin → 1              | {0,1}  | 避免危险超速                    |
| **driving_direction** | **逆行惩罚**          | 在道路几何方向相反方向行驶 → 1                   | {0,1}  | 避免逆行（最重要规则之一）             |
| **lane_deviation**    | **偏离当前车道惩罚**      | 在多个车道上行驶 → 1                        | {0,1}  | 防止乱变道、左右摇摆                |
| **log_div_clip**      | **轨迹发散强惩罚（阈值式）**  | log divergence < 阈值 → 1，否则 0        | {0,1}  | 保证轨迹和专家分布一致               |
| **log_div**           | **轨迹发散值（连续）**     | log divergence 数值（越大越差）             | [0,+∞) | 用于在连续 reward 中衡量轨迹偏离专家有多少 |
| **progression**       | **路线上是否前进**       | 当前 step 的 progression > 上一 step → 1 | {0,1}  | 保证 agent 不停在原地、向前走        |
| **comfort**           | **舒适性（加速度+jerk）** | - jerk 大 → 值小<br>- jerk 低 → 值高      | [0,1]  | 限制急刹车、急加速，提升舒适性           |

---

# 🌟 每个 reward 的简短总结（非常容易记住）

| 类别       | Reward                                                     | 简写记忆            |
| -------- | ---------------------------------------------------------- | --------------- |
| **安全性**  | overlap / below_ttc / overspeed / offroad                  | 不撞、不挨、不卡草、不超速   |
| **交通规则** | red_light / driving_direction / lane_deviation / off_route | 不闯灯、不逆行、不乱飘、不走错 |
| **行为质量** | log_div / log_div_clip                                     | 不偏离专家太多         |
| **性能效率** | progression                                                | 要动、要前进          |
| **乘坐体验** | comfort                                                    | 要稳，不要急刹急转弯      |

---

# 🚦 Multiplicative Reward 中的行为影响（很关键）

在 multiplicative reward 中：

* 每一项都会变成一个 **score**
* 再变成 log-space 再求和然后再 exponentiate

这意味着：

| 项目                         | 行为影响                             |
| -------------------------- | -------------------------------- |
| overlap=1 → score=0        | **整个 reward = 0**（训练瞬间崩溃）        |
| below_ttc=1 → score=0      | **前向太紧会被强压制**                    |
| lane_deviation=1 → score=0 | **一旦飘线，reward 值直接被大幅缩小 → 策略很敏感** |
| progression=1 → score=2    | **强鼓励一直走**（不走就死）                 |
| comfort 小 → score≈1        | **舒适性不重要**（除非权重高）                |



- **Penalty-based rewards** (e.g., `overlap`, `offroad`) usually return `True` (1.0) when a violation occurs, so you should assign them negative weights.
- **Reward-based rewards** (e.g., `progression`, `comfort`) return positive values when the agent behaves well, so you should assign them positive weights.

## Custom Rewards

If you need a custom reward function, you can subclass `RewardCustomWrapper` and implement your own logic in the `reward` method.

```python
class MyCustomRewardWrapper(RewardCustomWrapper):
    def reward(self, state, action):
        # Your custom reward logic here
        return jnp.array(...)
```

## Extending the Reward System

To add a new reward function:
1. Implement a new function (e.g., `_compute_my_new_reward(state)`) in `reward.py`.
2. Add it to the `_get_reward_fn` dictionary.
3. Use its name in your `reward_config`.

## Tips
- Tune the weights in `reward_config` to balance between safety, efficiency, and comfort.
- Use negative weights for penalties and positive weights for desirable behaviors.
- Test your configuration to ensure the agent learns the intended behavior.

## Summary
- The reward system in V-Max is modular and configurable.
- Use `RewardLinearWrapper` with a `reward_config` dictionary to combine multiple reward functions.
- Choose and tune reward functions to match your simulation goals.
- Extend or customize as needed for your use case.

For more details, see the source code in `vmax/simulator/wrappers/reward.py`.

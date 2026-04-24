# Copyright 2026 RTGTX7.

"""Canonical public reward-wrapper entry point.

The environment should import reward wrappers through this module, not from the
individual experimental files under ``reward_tmp``. The active implementation
currently lives in ``reward_tmp.reward_bubble2``.
"""

from vmax.simulator.wrappers.reward_tmp.reward_bubble2 import RewardCustomWrapper, RewardLinearWrapper

__all__ = ["RewardCustomWrapper", "RewardLinearWrapper"]

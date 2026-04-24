# Copyright 2025 Valeo.

"""Compatibility alias for the canonical reward wrappers.

Historically this repository carried several reward-wrapper variants under
different filenames. In the current workspace there is one canonical public
reward entry point:

- ``vmax.simulator.wrappers.reward``

That module re-exports the active implementation used by the environment. This
file is retained only so older imports continue to work.
"""

from vmax.simulator.wrappers.reward import RewardCustomWrapper, RewardLinearWrapper

__all__ = ["RewardCustomWrapper", "RewardLinearWrapper"]

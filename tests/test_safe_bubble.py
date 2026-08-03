import unittest

import jax.numpy as jnp

from vmax.simulator.metrics.safe_bubble import _masked_min_score


class MaskedMinScoreTest(unittest.TestCase):
    def test_excludes_ego_and_padding_slots(self):
        scores = jnp.array(
            [
                [0.2, 0.3],  # ego
                [0.6, 0.8],  # valid surrounding agent
                [0.1, 0.1],  # padded slot
            ],
            dtype=jnp.float32,
        )
        valid_non_sdc = jnp.array([False, True, False])

        self.assertEqual(float(_masked_min_score(scores, valid_non_sdc)), float(jnp.float32(0.6)))

    def test_uses_worst_valid_agent_over_rollout(self):
        scores = jnp.array(
            [
                [0.9, 0.7],
                [0.8, 0.5],
            ],
            dtype=jnp.float32,
        )
        valid_non_sdc = jnp.array([True, True])

        self.assertEqual(float(_masked_min_score(scores, valid_non_sdc)), float(jnp.float32(0.5)))


if __name__ == "__main__":
    unittest.main()

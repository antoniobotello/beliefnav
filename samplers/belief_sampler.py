import numpy as np
from typing import List


class BeliefSampler:
    def __init__(self, belief_regions: int = 5, candidates_per_region: int = 4):
        self.belief_regions = belief_regions
        self.candidates_per_region = candidates_per_region

    def sample(self, env) -> List[dict]:
        candidates: List[dict] = []
        max_num_candidates = self.belief_regions * self.candidates_per_region

        flat_indices = np.argsort(env.belief_map.ravel())[::-1][: self.belief_regions]
        ys, xs = np.unravel_index(flat_indices, env.belief_map.shape)
        belief_targets = list(zip(xs, ys))

        for tx, ty in belief_targets:
            for _ in range(self.candidates_per_region * 10):
                angle = env.np_random.uniform(-np.pi, np.pi)
                radius = env.np_random.uniform(2.0, env.sensor_range)

                x = int(np.round(tx + radius * np.cos(angle)))
                y = int(np.round(ty + radius * np.sin(angle)))

                x = int(np.clip(x, 0, env.grid_size - 1))
                y = int(np.clip(y, 0, env.grid_size - 1))

                if not env._is_valid_free_cell(x, y):
                    continue

                # Heading points toward high-belief region
                theta = np.arctan2(ty - y, tx - x)

                candidate_pose = np.array([x, y, theta], dtype=np.float32)

                # Simulate FOV
                visible_mask = env._ray_casting_visibility(pose=candidate_pose)

                # Discard the candidate if it is not in LOS (Line-Of-Sight)
                if not visible_mask[ty, tx]:
                    continue

                # Good FOV: candidate should see enough useful area
                free_visible_mask = visible_mask & (env.true_map == 0)
                visible_area = np.sum(free_visible_mask)

                # TODO: 5 is an initial try, adapt this threshold
                if visible_area < 5:
                    continue

                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "theta": theta,
                        "source": "belief",
                        "target_x": tx,
                        "target_y": ty,
                    }
                )

                if len(candidates) >= max_num_candidates:
                    return candidates

        return candidates

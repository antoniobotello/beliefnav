# samplers/frontier_sampler.py

import numpy as np
from typing import List


class FrontierSampler:
    def sample(self, env, n_candidates=30):
        frontiers = self._get_frontier_cells(env)

        if len(frontiers) == 0:
            return []

        candidates = []

        for _ in range(n_candidates):
            fx, fy = frontiers[env.np_random.integers(len(frontiers))]

            x, y = self._sample_near_frontier(env, fx, fy, radius=3)

            if not env.is_valid_free_cell(x, y):
                continue

            theta = np.arctan2(fy - y, fx - x)

            # TODO: Aqui en vez de diccionarios, tal vez seria mejor dar np.ndarray para evitar
            # TODO: convertirlo de nuevo en el futuro
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "source": "frontier",
                }
            )

        return candidates

    def _get_frontier_cells(self, env) -> List:
        frontiers = []

        for y in range(1, env.grid_size - 1):
            for x in range(1, env.grid_size - 1):
                # free cell
                #! Al princpio todo el mapa es desconocido,
                if env.occupancy_grid[y, x] != 0:
                    continue

                neighbors = env.occupancy_grid[y - 1 : y + 2, x - 1 : x + 2]

                # unknown neighbor
                if np.any(neighbors == -1):
                    frontiers.append((x, y))

        return frontiers

    def _sample_near_frontier(
        self, env, fx: int, fy: int, radius: int = 3
    ) -> tuple[int, int]:
        angle = env.np_random.uniform(-np.pi, np.pi)
        r = env.np_random.uniform(0, radius)

        x = int(fx + r * np.cos(angle))
        y = int(fy + r * np.sin(angle))

        x = int(np.clip(x, 0, env.grid_size - 1))
        y = int(np.clip(y, 0, env.grid_size - 1))

        return x, y

# samplers/frontier_sampler.py

import numpy as np
from typing import List
from sklearn.cluster import DBSCAN


class FrontierSampler:
    def __init__(self, num_candidates: int = 30):
        self.num_candidates = num_candidates
        self.candidates_per_cluster = 5
        self.sample_radius = 3

        # DBSCAN parameters
        self.eps = 0.35
        self.min_samples = 3

    # def sample(self, env):
    #     # Get all the frontier cells
    #     frontier_cells = self._get_frontier_cells(env)

    #     if len(frontier_cells) == 0:
    #         print("No frontiers found for the exploration mode.")
    #         return []

    #     # Agrupa los frontiers encontrados mediante connected components

    #     candidates = []

    #     for _ in range(self.num_candidates):
    #         # TODO: Aqui se estan sampleando frontier_cells al azar, cambia esto por frontier cells → frontier clusters → sample from clusters
    #         fx, fy = frontier_cells[env.np_random.integers(len(frontier_cells))]

    #         # TODO: Aqui se estan sampleando puntos cerca de los frontier_cells, cambia esto para que se samplen frontier_cells dentro de un frontier cluster
    #         x, y = self._sample_near_frontier(env, fx, fy, radius=3)

    #         if not env.is_valid_free_cell(x, y):
    #             continue

    #         theta = np.arctan2(fy - y, fx - x)

    #         candidates.append(
    #             {
    #                 "x": x,
    #                 "y": y,
    #                 "theta": theta,
    #                 "source": "frontier",
    #             }
    #         )
    #     return candidates

    def sample(self, env):
        # 1. Get all frontier cells
        frontier_cells = self._get_frontier_cells(env)

        if len(frontier_cells) == 0:
            print("No frontiers found for the exploration mode.")
            env.active_frontier_direction = None
            return []

        # 2. Cluster frontier cells using angle/distance DBSCAN
        clusters = self._cluster_frontiers(env, frontier_cells)

        if len(clusters) == 0:
            print("No frontier clusters found.")
            env.active_frontier_direction = None
            return []

        # 3. Select clusters to sample from
        if env.active_frontier_direction is None:
            # No active exploration region yet:
            # sample from all clusters so the scorer can choose the best region.
            selected_clusters = clusters
            candidates_per_cluster = max(
                1, self.num_candidates // len(selected_clusters)
            )
        else:
            # Continue exploring the same approximate direction
            similarities = [
                np.dot(cluster["direction"], env.active_frontier_direction)
                for cluster in clusters
            ]

            best_idx = int(np.argmax(similarities))
            selected_clusters = [clusters[best_idx]]
            candidates_per_cluster = self.num_candidates

        candidates = []

        # 4. Sample candidates from selected cluster(s)
        for cluster in selected_clusters:
            cells = cluster["cells"]  # frontier cells belonging to this cluster

            for _ in range(candidates_per_cluster):
                fx, fy = cells[env.np_random.integers(len(cells))]

                x, y = self._sample_near_frontier(
                    env,
                    int(fx),
                    int(fy),
                    radius=self.sample_radius,
                )

                if not env.is_valid_free_cell(x, y):
                    continue

                theta = np.arctan2(fy - y, fx - x)

                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "theta": float(theta),
                        "source": "frontier",
                        "cluster_id": cluster["id"],
                        "cluster_size": cluster["size"],
                        "cluster_direction": cluster["direction"],
                        "cluster_angle": cluster["angle"],
                    }
                )

                if len(candidates) >= self.num_candidates:
                    break

            if len(candidates) >= self.num_candidates:
                break

        # 5. If active region produced no valid candidates, reset it
        if len(candidates) == 0 and env.active_frontier_direction is not None:
            env.active_frontier_direction = None
            return self.sample(env)

        return candidates

    def _cluster_frontiers(self, env, frontiers):
        frontiers = np.array(frontiers)  # shape (N, 2)

        robot_x, robot_y, _ = env.robot_pose

        dx = frontiers[:, 0] - robot_x
        dy = frontiers[:, 1] - robot_y

        angles = np.arctan2(dy, dx)
        distances = np.sqrt(dx**2 + dy**2)
        distances_norm = distances / env.grid_size

        features = np.column_stack(
            [
                np.cos(angles),
                np.sin(angles),
                distances_norm,
            ]
        )
        labels = DBSCAN(
            eps=self.eps,  # max distance between two points to be considered neighbors
            min_samples=self.min_samples,  # minimum number of nearby points needed to form a dense cluster
        ).fit_predict(features)

        clusters = []

        for label in set(labels):
            if label == -1:  # noise points that do not belong to any cluster
                continue

            cells = frontiers[labels == label]

            mean_dx = np.mean(cells[:, 0] - robot_x)
            mean_dy = np.mean(cells[:, 1] - robot_y)
            mean_angle = np.arctan2(mean_dy, mean_dx)

            clusters.append(
                {
                    "id": int(label),
                    "cells": cells,
                    "size": len(cells),
                    "direction": np.array([np.cos(mean_angle), np.sin(mean_angle)]),
                    "angle": float(mean_angle),
                }
            )
        return clusters

    """
    1. Encuentra todas las frontier cells
    2. Agrúpalas con connected components
    3. Calcula score para cada cluster
    4. Elige top clusters o samplea clusters proporcional a su score
    5. Genera candidates cerca de esos clusters
    6. Filtra candidates:
    - known free
    - reachable
    - enough clearance
    7. CandidateScorer elige el mejor candidate final

    centroide: Cuidado! este puede caer dentro de unknown o de un obstaculo
    cx = mean(x coordinates)
    cy = mean(y coordinates)

    + cluster_size:
    cuántas frontier cells tiene el cluster

    - distance_to_robot:
        distancia o path cost al centroide del cluster

    - revisit_penalty:
        si está cerca del path reciente del robot

    + visible_unknown_potential:
        cuánta zona unknown hay alrededor del cluster


    _get_frontier_cells()
            ↓
    _cluster_frontiers()
            ↓
    _score_frontier_clusters()
            ↓
    sample candidate near selected cluster
            ↓
    filter candidate: known free + reachable
            ↓
    return candidates with metadata

    Clusters deberian tener (para priorizar clusters grandes y castigar a los clusters pequeños):
        cluster = {
        "cells": [...],
        "centroid": (cx, cy),
        "size": len(cells),
        "distance_to_robot": ...,
    }

    Samplear punto cerca del centroide del cluster seleccionado. 
    chosen_cluster → centroid/frontier cell representative → sample nearby candidate

    Metadata del candidate deberia verse mas o menos asi: 
        candidate = {
        "x": x,
        "y": y,
        "theta": theta,
        "source": "frontier",
        "frontier_cluster_size": cluster_size,
        "frontier_cluster_id": cluster_id,
    }

    Para poenalizar puntos cerca de regiones anteriormenete visitadas, anade por ejemplo penality a las ultimas 10 poisiciones
    """

    def _get_frontier_cells(self, env) -> List:
        frontiers = []

        for y in range(1, env.grid_size - 1):
            for x in range(1, env.grid_size - 1):
                # free cell
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

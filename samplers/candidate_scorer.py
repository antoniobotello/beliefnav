# samplers/candidate_scorer.py
import numpy as np


class CandidateScorer:
    def select_best(self, env, candidates):
        if len(candidates) == 0:
            raise ValueError("No candidates to evaluate at all.")

        best_candidate = None
        best_score = -float("inf")

        for candidate in candidates:
            score = self.score(env, candidate)

            candidate["score"] = score

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def score(self, env, candidate):
        candidate = np.array(
            [candidate["x"], candidate["y"], candidate["theta"]],
            dtype=np.float32,
        )
        visible_belief = self._compute_visible_belief_per_candidate(env, candidate)
        path_cost = self._compute_distance_to(env, candidate)
        visible_unknown_area = self._compute_visible_belief_per_candidate(
            env, candidate
        )

        return visible_belief + visible_unknown_area - path_cost

    def _compute_visible_belief_per_candidate(
        self, env, candidate: np.ndarray
    ) -> float:
        visible_mask = env._ray_casting_visibility(pose=candidate)
        free_visible_mask = visible_mask & (env.true_map == 0)
        visible_belief = np.sum(env.belief_map[free_visible_mask])
        # ?: Necesito el predicted belief map?
        # visible_belief = np.sum(self.predicted_belief_map[free_visible_mask])

        return float(visible_belief)

    def _compute_distance_to(self, env, candidate: np.ndarray):
        path_cost = np.linalg.norm(candidate[:2] - env.robot_pose[:2])
        return path_cost

    def _compute_visible_unknown_area_per_candidate(
        self, env, candidate: np.ndarray
    ) -> float:
        """
        candidate is a np.ndarray with the structure:
        candidate[0] = x, candidate[1] = y, candidate[2] = theta
        """

        visible_mask = env._ray_casting_visibility(pose=candidate)
        unknown_mask = env.occupancy_grid == -1
        visible_unknown_mask = visible_mask & unknown_mask
        visible_unknown_area = np.sum(visible_unknown_mask)

        return float(visible_unknown_area)

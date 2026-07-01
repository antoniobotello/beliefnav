import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict
from skimage.draw import line
from skimage.graph import route_through_array  # A* library
from scipy.ndimage import distance_transform_edt

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Wedge
from samplers.frontier_sampler import FrontierSampler
from samplers.belief_sampler import BeliefSampler
from samplers.reposition_sampler import RepositionSampler
from samplers.candidate_scorer import CandidateScorer


class TargetSearchEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # mas adelante, si quiero guardar frames o videos hace sentido usar
    # metadata = {
    #     "render_modes": ["human", "rgb_array"],
    #     "render_fps": 10,
    # }

    def __init__(self, grid_size=100, max_steps=100, render_mode="human"):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # * Action
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.5], dtype=np.float32),
            high=np.array([3.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Later: RL chooses between explore, exploit, reposition
        # self.action_space = spaces.Discrete(3)
        # self._action_to_strategy = ["explore", "exploit", "reposition"]

        # * Observation
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(grid_size, grid_size),
            dtype=np.float32,
        )

        # * Target
        self.target_pos = None
        self.target_vel = None
        self.target_speed = 1.0
        self.target_found = False

        # * Robot
        self.robot_pose = None  # [x, y, theta]
        self.robot_cells_per_step = 2

        # * Dynamics
        self.dt = 0.5

        # * Maps
        self.occupancy_grid = None
        self.true_map = None
        self.belief_map = None
        self.obstacle_distance_map = None

        # * Paths
        self.robot_path = []
        self.candidate_paths = []
        self.best_path = None

        # * LiDAR Sensor
        self.sensor_range = 5.0
        self.fov_angle = 2 * np.pi
        # self.fov_angle = np.deg2rad(90)
        self.num_rays = 360

        # * Rendering
        self.fig = None
        self.ax = None

        # * High Level Samplers
        # * Frontier Sampler
        self.frontier_sampler = FrontierSampler(num_candidates=30)
        self.active_frontier_direction = None
        self.active_frontier_direction = None  # np.array([cos(angle), sin(angle)])

        self.belief_sampler = BeliefSampler(belief_regions=5, candidates_per_region=4)
        self.reposition_sampler = RepositionSampler(
            n_candidates=30, max_attempts=500, min_clerance=3
        )

        # * Low Level Planner
        self.current_goal = None
        self.current_path = []
        self.steps_since_replan = 0
        self.replan_interval = 5

        # * Candidate Scorer
        self.candidate_scorer = CandidateScorer()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.target_found = False

        self.current_goal = None
        self.current_path = []
        self.steps_since_replan = 0

        self.step_count = 0

        # * Robot
        self.robot_pose = np.array([5.0, 5.0, 0.0], dtype=np.float32)
        self.robot_path = [self.robot_pose[:2].copy()]

        # * Target
        self.target_pos = np.array([80.0, 80.0], dtype=np.float32)
        target_angle = np.deg2rad(135)

        self.target_vel = self.target_speed * np.array(
            [np.cos(target_angle), np.sin(target_angle)],
            dtype=np.float32,
        )

        # * Belief map (Uniform distribution)
        self.belief_map = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.belief_map /= self.belief_map.sum()

        # * Obstacle map with hardcoded obstacles
        self.occupancy_grid = -np.ones(
            (self.grid_size, self.grid_size), dtype=np.int8
        )  # -1 = unknown, 0 = free, 1 = obstacle

        # * Real map known only by the simulator
        self.true_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # * Obstacles
        self.true_map[10:30, 10:20] = 1
        self.true_map[30:80, 45:55] = 1
        self.true_map[60:80, 20:30] = 1
        self.true_map[20:50, 80:90] = 1

        # # Add border walls
        self.true_map[0, :] = 1  # bottom wall
        self.true_map[-1, :] = 1  # top wall
        self.true_map[:, 0] = 1  # left wall
        self.true_map[:, -1] = 1  # right wall

        self.candidate_paths = []
        self.best_path = None

        # Initial sensing before first action
        visible_mask = self._ray_casting_visibility(pose=self.robot_pose)
        self._update_occupancy_grid(visible_mask)
        self._update_obstacle_distance_map()

        # If there is no obstacle
        free_visible_mask = visible_mask & (self.true_map == 0)
        self.target_found = self._update_belief_with_visibility(free_visible_mask)

        obs = self._get_obs()
        info = {
            "visible_mask": visible_mask,
            "robot_pose": self.robot_pose.copy(),
        }

        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.target_found:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"target_found": True}

        self.step_count += 1

        # if self.step_count < int(self.max_steps / 3):
        #     strategy = "explore"
        # elif self.step_count < 2 * int(self.max_steps / 3):
        #     strategy = "reposition"
        # else:
        #     strategy = "exploit"

        # TODO: Stratgy choosen with RL and not Hardcoded
        # strategy = self._action_to_strategy[action]

        # # * Hardcoded strategies
        strategy = "explore"
        # strategy = "exploit"
        # strategy = "reposition"

        best_candidate = None

        # if self._should_replan():
        #     best_candidate = self._select_new_candidate(strategy)

        #     if best_candidate is not None:
        #         path_found = self._plan_path_to_candidate(best_candidate)

        #         if not path_found:
        #             self.current_goal = None
        #             self.current_path = []
        #             self.steps_since_replan = 0

        path_found = False
        replanned = False

        if self._should_replan():
            replanned = True
            best_candidate = self._select_new_candidate(strategy)

            if best_candidate is not None:
                path_found = self._plan_path_to_candidate(best_candidate)

                if path_found:
                    # Commit to the frontier region only if the candidate was accepted
                    if best_candidate.get("source") == "frontier":
                        self.active_frontier_direction = best_candidate[
                            "cluster_direction"
                        ]

                else:
                    self.current_goal = None
                    self.current_path = []
                    self.steps_since_replan = 0

        self._follow_current_path()

        # * Target's Position Update at new step
        self._move_target_constant_velocity()

        # * Belief prediction: target motion model
        self._predict_belief_motion()

        # * Ray casting
        visible_mask = self._ray_casting_visibility(pose=self.robot_pose)

        # * Check if the target was detected
        target_cell = np.round(self.target_pos).astype(
            int
        )  # Target position only known by the simulator, not by the robot
        tx, ty = target_cell

        # if bool(visible_mask[tx, ty]):
        if bool(visible_mask[ty, tx]):
            print(f"Target was detected at cell {int(tx.item()), int(ty.item())}")

        # * Mapping update: what the robot knows about obstacles/free space
        self._update_occupancy_grid(visible_mask)
        self._update_obstacle_distance_map()

        # * Belief correction: where the robot looked and did not see the target
        # TODO: Comprueba si esta free visible mask es valida
        free_visible_mask = visible_mask & (self.true_map == 0)
        target_found = self._update_belief_with_visibility(free_visible_mask)

        terminated = target_found
        truncated = self.step_count >= self.max_steps

        if not terminated:
            # * Candidate Paths
            #! Esta decision para elegir paths es dummy hasta ahora corrigela
            self.candidate_paths = self._generate_candidate_paths()
            self.best_path = (
                self.candidate_paths[0] if len(self.candidate_paths) > 0 else None
            )

        reward = -0.1

        obs = self._get_obs()
        # info = {
        #     "visible_mask": visible_mask,
        #     "robot_pose": self.robot_pose.copy(),
        #     "strategy": strategy,
        #     "current_goal": self.current_goal,
        #     "current_path": self.current_path,
        #     "steps_since_replan": self.steps_since_replan,
        #     "target_found": target_found,
        # }
        info = {
            "visible_mask": visible_mask,
            "robot_pose": self.robot_pose.copy(),
            "strategy": strategy,
            "replanned": replanned,
            "path_found": path_found,
            "best_candidate": best_candidate,
            "current_goal": self.current_goal,
            "current_path": self.current_path,
            "steps_since_replan": self.steps_since_replan,
            "target_found": target_found,
        }

        return obs, reward, terminated, truncated, info

    def _select_new_candidate(self, strategy: str) -> dict | None:
        if strategy == "explore":
            candidates = self.frontier_sampler.sample(self)
        elif strategy == "exploit":
            candidates = self.belief_sampler.sample(self)
        elif strategy == "reposition":
            candidates = self.reposition_sampler.sample(self)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(candidates) == 0:
            return None

        return self.candidate_scorer.select_best(self, candidates)

    def _should_replan(self) -> bool:
        if self.current_goal is None:
            return True

        if len(self.current_path) < 2:
            return True

        if self.steps_since_replan >= self.replan_interval:
            return True

        next_y, next_x = self.current_path[1]

        if self._position_blocked_in_known_map(next_x, next_y):
            return True

        return False

    def _update_obstacle_distance_map(self) -> None:
        obstacle_mask = self.occupancy_grid == 1
        free_space_mask = ~obstacle_mask

        self.obstacle_distance_map = distance_transform_edt(free_space_mask)

    def _candidate_is_reachable(self, candidate: dict) -> bool:
        path = self._get_path_to_candidate(candidate)
        return path is not None

    def _get_path_to_candidate(self, candidate: dict) -> list | None:
        x_start, y_start, _ = self.robot_pose

        x_goal = int(round(candidate["x"]))
        y_goal = int(round(candidate["y"]))
        # x_goal, y_goal, _ = candidate

        start = (int(round(y_start)), int(round(x_start)))
        goal = (y_goal, x_goal)

        if self._position_blocked_in_known_map(x_goal, y_goal):
            return None

        cost_array = np.ones_like(self.occupancy_grid, dtype=float)
        cost_array[self.occupancy_grid == -1] = np.inf
        cost_array[self.occupancy_grid == 1] = np.inf
        cost_array[self.occupancy_grid == 0] = 1.0

        try:
            path, cost = route_through_array(
                cost_array,
                start=start,
                end=goal,
                fully_connected=True,
            )
        except ValueError:
            return None

        if len(path) < 2:
            return None

        return list(path)

    def _plan_path_to_candidate(self, candidate: dict) -> bool:
        path = self._get_path_to_candidate(candidate)

        if path is None:
            return False

        self.current_goal = candidate
        self.current_path = path
        self.steps_since_replan = 0

        return True

    def _follow_current_path(self) -> None:
        if len(self.current_path) < 2:
            return

        x_old, y_old, _ = self.robot_pose
        moved = False

        cells_to_move = int(self.robot_cells_per_step)

        for _ in range(cells_to_move):
            if len(self.current_path) < 2:
                break

            next_y, next_x = self.current_path[1]

            if self._position_blocked_in_known_map(next_x, next_y):
                break

            self.current_path.pop(0)

            moved = True

        if not moved:
            return

        y_new, x_new = self.current_path[0]

        dx = x_new - x_old
        dy = y_new - y_old
        theta = np.arctan2(dy, dx)

        self.robot_pose = np.array(
            [float(x_new), float(y_new), theta],
            dtype=np.float32,
        )

        self.robot_path.append(self.robot_pose[:2].copy())

        self.steps_since_replan += 1

    # def _position_in_collision(self, x: float, y: float) -> bool:
    #     ix = int(round(x))
    #     iy = int(round(y))

    #     if ix < 0 or ix >= self.grid_size or iy < 0 or iy >= self.grid_size:
    #         return True

    #     return self.true_map[iy, ix] == 1

    def _position_blocked_in_known_map(self, x: float, y: float) -> bool:
        ix = int(round(x))
        iy = int(round(y))

        if ix < 0 or ix >= self.grid_size or iy < 0 or iy >= self.grid_size:
            return True

        # unknown and obstacle are blocked
        return self.occupancy_grid[iy, ix] != 0

    def is_valid_free_cell(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        return self.occupancy_grid[y, x] == 0

    def _get_obs(self) -> np.ndarray:
        return self.belief_map.copy()

    def _generate_candidate_paths(
        self, num_paths: int = 25, path_len: int = 15
    ) -> List:
        paths = []

        for _ in range(num_paths):
            pts = [self.robot_pose[:2].copy()]
            current = self.robot_pose[:2].copy()

            for _ in range(path_len):
                step = self.np_random.normal(0, 1.2, size=2)
                current = current + step
                current = np.clip(current, 0, self.grid_size - 1)
                pts.append(current.copy())

            paths.append(np.array(pts))

        return paths

    def _predict_belief_motion(self) -> None:
        """
        Belief prediction for a constant-velocity target.

        The belief is shifted according to the estimated target velocity
        and slightly diffused to model uncertainty.
        """

        old_belief_map = self.belief_map.copy()

        vx, vy = self.target_vel

        # Convert continous velocity into cells displacement
        shift_x = int(np.round(vx * self.dt))
        shift_y = int(np.round(vy * self.dt))

        # * Belief shifted according to target motion
        predicted_belief_map = np.zeros_like(old_belief_map)

        h, w = old_belief_map.shape

        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)

        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)

        # Belif shift
        predicted_belief_map[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = (
            old_belief_map[
                src_y_start:src_y_end,
                src_x_start:src_x_end,
            ]
        )

        # Small diffusion for uncertainty
        diffused = predicted_belief_map.copy()
        diffused += 0.1 * self._shift_no_wrap(predicted_belief_map, 1, 0)
        diffused += 0.1 * self._shift_no_wrap(predicted_belief_map, -1, 0)
        diffused += 0.1 * self._shift_no_wrap(predicted_belief_map, 0, 1)
        diffused += 0.1 * self._shift_no_wrap(predicted_belief_map, 0, -1)

        # No belief for cells with obstacles
        diffused[self.occupancy_grid == 1] = 0.0

        total = diffused.sum()
        if total > 0:
            diffused /= total

        self.belief_map = diffused.astype(np.float32)

    def _shift_no_wrap(self, arr, shift_y, shift_x) -> np.ndarray:
        shifted = np.zeros_like(arr)

        h, w = arr.shape

        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(h, h + shift_y)

        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(w, w + shift_x)

        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[
            src_y_start:src_y_end,
            src_x_start:src_x_end,
        ]

        return shifted

    def _update_belief_with_visibility(self, visible_mask: np.ndarray) -> bool:
        target_cell = np.round(self.target_pos).astype(
            int
        )  # Target position only known by the simulator, not by the robot
        tx, ty = target_cell

        target_visible = False  # Assumption

        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            target_visible = visible_mask[ty, tx]  # Target cell visible?

        if target_visible:
            # Noisy/uncertain Target detection:
            # if the target is visible, increase the belief at the detected cell
            # but keep some probability mass elsewhere to avoid assuming perfect localization
            self.belief_map *= 0.1
            self.belief_map[ty, tx] += 1.0

            # Normalize
            total = self.belief_map.sum()

            if total > 0:
                self.belief_map /= total

            return True
            # TODO: replace this with a noisy detection model, where the
            # TODO: measurement is sampled around the true target position and the belief is
            # TODO: updated with a Gaussian likelihood centered at the noisy measurement.
        else:
            # Negative observation:
            # cells that were visible but target was not detected become less likely
            self.belief_map[visible_mask] *= 0.2

            # Known obstacles should not contain target probability
            self.belief_map[self.occupancy_grid == 1] = 0.0

            # Normalize
            total = self.belief_map.sum()

            if total > 0:
                self.belief_map /= total
            else:
                self.belief_map = np.ones(
                    (self.grid_size, self.grid_size),
                    dtype=np.float32,
                )

                self.belief_map[self.occupancy_grid == 1] = 0.0

                total = self.belief_map.sum()
                if total > 0:
                    self.belief_map /= total

                return False

    def _update_occupancy_grid(self, visible_mask: np.ndarray) -> None:
        visible_free = visible_mask & (self.true_map == 0)
        visible_obstacles = visible_mask & (self.true_map == 1)

        self.occupancy_grid[visible_free] = 0
        self.occupancy_grid[visible_obstacles] = 1

    def _draw_drone(self, ax: Axes) -> None:
        x, y, theta = self.robot_pose

        arm_length = 1.2
        rotor_size = 45

        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        local_points = np.array(
            [
                [arm_length, 0.0],
                [-arm_length, 0.0],
                [0.0, arm_length],
                [0.0, -arm_length],
            ]
        )

        world_points = local_points @ R.T + np.array([x, y])

        # arms
        ax.plot(
            [world_points[0, 0], world_points[1, 0]],
            [world_points[0, 1], world_points[1, 1]],
            linewidth=2.0,
            zorder=6,
        )

        ax.plot(
            [world_points[2, 0], world_points[3, 0]],
            [world_points[2, 1], world_points[3, 1]],
            linewidth=2.0,
            zorder=6,
        )

        # rotors
        ax.scatter(
            world_points[:, 0],
            world_points[:, 1],
            s=rotor_size,
            zorder=7,
        )

        # heading arrow
        ax.arrow(
            x,
            y,
            1.8 * np.cos(theta),
            1.8 * np.sin(theta),
            head_width=0.4,
            head_length=0.5,
            length_includes_head=True,
            zorder=8,
        )

    def _draw_fov(self, ax: Axes) -> None:
        x, y, theta = self.robot_pose

        if np.isclose(self.fov_angle, 2 * np.pi):
            lidar_range = Circle(
                (x, y),
                radius=self.sensor_range,
                alpha=0.18,
                zorder=3,
            )
            ax.add_patch(lidar_range)
            return

        left_angle = theta - self.fov_angle / 2
        right_angle = theta + self.fov_angle / 2

        fov = Wedge(
            center=(x, y),
            r=self.sensor_range,
            theta1=np.rad2deg(left_angle),
            theta2=np.rad2deg(right_angle),
            alpha=0.25,
            zorder=3,
        )

        ax.add_patch(fov)

    def _ray_casting_visibility(self, pose: dict | np.ndarray) -> np.ndarray:
        """
        Computes the visible cells from a pose using simulated 2D LiDAR ray casting.
        For 360-degree LiDAR, rays are cast in all directions around the robot.
        """
        if isinstance(pose, dict):
            x, y, theta = pose["x"], pose["y"], pose["theta"]
        elif isinstance(pose, np.ndarray):
            x, y, theta = pose
        else:
            raise ValueError(
                f"Data Type: {type(pose)} not supported for the robot pose."
            )

        visible_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        if np.isclose(self.fov_angle, 2 * np.pi):
            # endpoint=False avoids duplicating 0 and 2*pi, which represent the same ray direction.
            angles = np.linspace(
                0.0,
                2 * np.pi,
                self.num_rays,
                endpoint=False,
            )
        else:
            angles = np.linspace(
                theta - self.fov_angle / 2,
                theta + self.fov_angle / 2,
                self.num_rays,
            )

        x0 = int(round(x))
        y0 = int(round(y))

        for angle in angles:
            x1 = int(round(x + self.sensor_range * np.cos(angle)))
            y1 = int(round(y + self.sensor_range * np.sin(angle)))

            rr, cc = line(y0, x0, y1, x1)

            for iy, ix in zip(rr, cc):
                if ix < 0 or ix >= self.grid_size or iy < 0 or iy >= self.grid_size:
                    break

                visible_mask[iy, ix] = True

                if self.true_map[iy, ix] == 1:
                    break

        return visible_mask

    # Render solo va a dibujar lo que ya existe
    def render(self) -> None:
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
            plt.show(block=False)

        ax_main, ax_belief, ax_occ = self.axes

        ax_main.clear()
        ax_belief.clear()
        ax_occ.clear()

        # Colormap for occupancy grid:
        # -1 = unknown, 0 = free, 1 = obstacle
        occ_cmap = ListedColormap(
            [
                "gray",  # unknown
                "white",  # free
                "black",  # obstacle
            ]
        )

        occ_norm = BoundaryNorm(
            boundaries=[-1.5, -0.5, 0.5, 1.5],
            ncolors=3,
        )

        # =====================================================
        # 1. MAIN WORLD MAP
        # =====================================================
        ax_main.imshow(
            self.true_map,
            origin="lower",
            cmap=occ_cmap,
            norm=occ_norm,
            alpha=0.9,
        )

        # FOV
        self._draw_fov(ax_main)

        # Robot path
        if len(self.robot_path) > 1:
            path = np.array(self.robot_path)
            ax_main.plot(
                path[:, 0],
                path[:, 1],
                linewidth=2.0,
                alpha=0.8,
            )

        # Target
        ax_main.scatter(
            self.target_pos[0],
            self.target_pos[1],
            marker="x",
            s=100,
            linewidths=2.0,
            zorder=6,
            c="red",
        )

        # Robot
        self._draw_drone(ax_main)

        ax_main.set_title(f"World | step={self.step_count}")
        ax_main.set_xlim(0, self.grid_size - 1)
        ax_main.set_ylim(0, self.grid_size - 1)
        ax_main.set_aspect("equal")
        ax_main.set_xticks([])
        ax_main.set_yticks([])

        # =====================================================
        # 2. BELIEF MAP
        # =====================================================

        # if belief_display.max() > 0:
        #     belief_display = belief_display / belief_display.max()
        belief_display = self.belief_map.copy()

        bmin = belief_display.min()
        bmax = belief_display.max()

        if bmax > bmin:
            belief_display = (belief_display - bmin) / (bmax - bmin)
        else:
            belief_display = np.zeros_like(belief_display)

        ax_belief.imshow(
            belief_display,
            origin="lower",
            cmap="turbo",  # blue = low, red = high
            vmin=0.0,
            vmax=1.0,
        )

        ax_belief.scatter(
            self.robot_pose[0],
            self.robot_pose[1],
            s=50,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            c="black",
        )

        ax_belief.scatter(
            self.target_pos[0],
            self.target_pos[1],
            marker="x",
            s=80,
            linewidths=2.0,
            zorder=5,
            c="red",
        )

        ax_belief.set_title("Belief map")
        ax_belief.set_xlim(0, self.grid_size - 1)
        ax_belief.set_ylim(0, self.grid_size - 1)
        ax_belief.set_aspect("equal")
        ax_belief.set_xticks([])
        ax_belief.set_yticks([])

        # =====================================================
        # 3. OCCUPANCY GRID
        # =====================================================
        ax_occ.imshow(
            self.occupancy_grid,
            origin="lower",
            cmap=occ_cmap,
            norm=occ_norm,
        )

        ax_occ.scatter(
            self.robot_pose[0],
            self.robot_pose[1],
            s=50,
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            c="red",
        )

        ax_occ.set_title("Occupancy grid")
        ax_occ.set_xlim(0, self.grid_size - 1)
        ax_occ.set_ylim(0, self.grid_size - 1)
        ax_occ.set_aspect("equal")
        ax_occ.set_xticks([])
        ax_occ.set_yticks([])

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def _move_target_constant_velocity(self) -> None:
        new_pos = self.target_pos + self.target_vel * self.dt

        # * Bounce against the wall and position update

        x = int(np.round(new_pos[0]))
        y = int(np.round(new_pos[1]))

        collision = (
            x < 0
            or x >= self.grid_size
            or y < 0
            or y >= self.grid_size
            or self.true_map[y, x] == 1
        )

        if collision:
            self.target_vel *= -1.0
            new_pos = self.target_pos + self.target_vel * self.dt

        self.target_pos = np.clip(new_pos, 0, self.grid_size - 1)

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

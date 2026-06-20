import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict
from skimage.draw import line

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, BoundaryNorm
from samplers.frontier_sampler import FrontierSampler
from samplers.candidate_scorer import CandidateScorer

num_rays = 90


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
        )  # 2x2 vector. First column indicates v (range 0.0 to 1.0), second indicates w (range from -0.5 to 0.5)
        # self.action_space = spaces.Discrete(3)
        # self._action_to_strategy = ["explore", "exploit", "reposition"]
        # 0 = explore
        # 1 = exploit
        # 2 = reposition

        # * Observation
        # TODO: Agrega el occupany map
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(grid_size, grid_size),
            dtype=np.float32,
        )  # Observation represents the belief map, therefore the range goes from 0.0 to 1.0

        # * Target
        self.target_pos = None
        self.target_vel = None
        self.target_speed = 0.5
        self.dt = 1.0

        self.step_count = 0
        self.target_found = False

        # * Robot
        self.robot_pose: np.ndarray = None  # [x, y, theta]
        self.robot_pos = None

        # * Belief Map (Uniform Prior)
        self.belief_map = np.ones((grid_size, grid_size), dtype=np.float32)
        self.belief_map /= self.belief_map.sum()

        # * Occupancy Grid (Known by the robot)
        # -1 = unknown
        #  0 = free
        #  1 = obstacle
        self.occupancy_grid = -np.ones((grid_size, grid_size), dtype=np.int8)

        # * Paths
        self.robot_path = []
        self.candidate_paths = []
        self.best_path = None

        # * Sensor
        self.sensor_range = 10.0
        self.fov_angle = np.deg2rad(90)  # 90 degrees field of view

        self.fig = None
        self.ax = None

        # * Samplers
        self.frontier_sampler = FrontierSampler()

        # * Candidate Scorer
        self.candidate_scorer = CandidateScorer()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.step_count = 0

        # * Robot
        self.robot_pose = np.array([5.0, 5.0, 0.0], dtype=np.float32)
        self.robot_path = [self.robot_pose[:2].copy()]

        # * Target
        self.target_pos = np.array([40.0, 30.0], dtype=np.float32)
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

        # TODO: El robot no puede pasar por en medio de los obtaculos
        self.true_map[10:30, 10:20] = 1
        self.true_map[25:45, 22:32] = 1
        self.true_map[70:80, 10:15] = 1
        self.true_map[50:70, 80:90] = 1

        # # Add border walls
        self.true_map[0, :] = 1  # bottom wall
        self.true_map[-1, :] = 1  # top wall
        self.true_map[:, 0] = 1  # left wall
        self.true_map[:, -1] = 1  # right wall

        self.candidate_paths = []
        self.best_path = None

        # Initial sensing before first action
        visible_mask = self._ray_casting_visibility(
            pose=self.robot_pose, num_rays=num_rays
        )
        self._update_occupancy_grid(visible_mask)

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

        #####################
        # strategy = self._action_to_strategy[action]
        strategy = "explore"  # hardcoded
        # strategy = "exploit"
        # strategy = "reposition"

        if strategy == "explore":
            candidates = self.frontier_sampler.sample(self)
        elif strategy == "exploit":
            candidates = self.belief_sampler.sample(self)
        elif strategy == "reposition":
            candidates = self.reposition_sampler.sample(self)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        best_candidate = self.candidate_scorer.select_best(self, candidates)
        # best_candidate = candidates[0] if len(candidates) > 0 else None

        if best_candidate is None:
            self._move_robot(action)
        else:
            self._move_robot_to_candidate(best_candidate)

        #####################

        # * Robot's Position Update at new step
        # self._move_robot(action)

        # * Target's Position Update at new step
        self._move_target_constant_velocity()

        # * Belief prediction: target motion model
        self._predict_belief_motion()

        # * Ray casting
        visible_mask = self._ray_casting_visibility(
            pose=self.robot_pose, num_rays=num_rays
        )

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
        info = {
            "visible_mask": visible_mask,
            "robot_pose": self.robot_pose.copy(),
            "strategy": strategy,
            "candidates": candidates,
            "best_candidate": best_candidate,
            "target_found": target_found,
        }

        return obs, reward, terminated, truncated, info

    def _move_robot(self, action) -> None:
        v, omega = action

        x, y, theta = self.robot_pose

        # Apply rotation first
        theta = theta + omega
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Proposed motion
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)

        proposed_x = x + dx
        proposed_y = y + dy

        proposed_x = np.clip(proposed_x, 0, self.grid_size - 1)
        proposed_y = np.clip(proposed_y, 0, self.grid_size - 1)

        # Check collision separately in x and y direction
        collision_x = self._position_in_collision(proposed_x, y)
        collision_y = self._position_in_collision(x, proposed_y)

        # Bounce: reverse the component that hits the obstacle
        if collision_x:
            dx *= -1.0

        if collision_y:
            dy *= -1.0

        # If collision happened, update orientation according to reflected motion
        if collision_x or collision_y:
            theta = np.arctan2(dy, dx)

        new_x = x + dx
        new_y = y + dy

        new_x = np.clip(new_x, 0, self.grid_size - 1)
        new_y = np.clip(new_y, 0, self.grid_size - 1)

        # If the reflected position is still invalid, stay in place
        if self._position_in_collision(new_x, new_y):
            self.robot_pose = np.array([x, y, theta], dtype=np.float32)
            return

        self.robot_pose = np.array([new_x, new_y, theta], dtype=np.float32)
        self.robot_path.append(self.robot_pose[:2].copy())

    def _position_in_collision(self, x: float, y: float) -> bool:
        ix = int(round(x))
        iy = int(round(y))

        if ix < 0 or ix >= self.grid_size or iy < 0 or iy >= self.grid_size:
            return True

        return self.true_map[iy, ix] == 1

    def is_valid_free_cell(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False

        return self.occupancy_grid[y, x] == 0 and self.true_map[y, x] == 0

    def _move_robot_to_candidate(self, candidate: dict) -> None:
        x = float(candidate["x"])
        y = float(candidate["y"])
        theta = float(candidate["theta"])

        if self._position_in_collision(x, y):
            return

        self.robot_pose = np.array([x, y, theta], dtype=np.float32)
        self.robot_path.append(self.robot_pose[:2].copy())

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

        left_angle = theta - self.fov_angle / 2
        right_angle = theta + self.fov_angle / 2

        left_point = np.array(
            [
                x + self.sensor_range * np.cos(left_angle),
                y + self.sensor_range * np.sin(left_angle),
            ]
        )

        right_point = np.array(
            [
                x + self.sensor_range * np.cos(right_angle),
                y + self.sensor_range * np.sin(right_angle),
            ]
        )

        polygon = np.array(
            [
                [x, y],
                left_point,
                right_point,
            ]
        )

        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            alpha=0.25,
            zorder=3,
        )

    def _ray_casting_visibility(
        self, pose: np.ndarray, num_rays: int = num_rays
    ) -> np.ndarray:
        """
        Computes the visible cells from the robot pose using ray casting.

        The function:
        1. takes the current robot pose,
        2. casts multiple rays inside the robot field of view,
        3. advances each ray cell by cell,
        4. marks cells as visible,
        5. stops a ray when it hits an obstacle,
        6. returns a boolean visible_mask.

        Returns:
            visible_mask: Boolean grid where True means the cell is visible.
        """
        # x, y, theta = self.robot_pose
        x, y, theta = pose

        visible_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        angles = np.linspace(
            theta - self.fov_angle / 2,
            theta + self.fov_angle / 2,
            num_rays,
        )  # Ray angles range

        # Convert the robot's position into a grid cell
        x0 = int(round(x))
        y0 = int(round(y))

        # Shot a ray for every ray insie the ray angles range
        for angle in angles:
            # Get the ray's final point
            x1 = int(round(x + self.sensor_range * np.cos(angle)))
            y1 = int(round(y + self.sensor_range * np.sin(angle)))

            rr, cc = line(
                y0, x0, y1, x1
            )  # Get the cells saw by the ray. rr (rows) -> y and cc (columns) -> x

            for iy, ix in zip(rr, cc):
                if ix < 0 or ix >= self.grid_size or iy < 0 or iy >= self.grid_size:
                    break  # If ray outside the grid, stop it

                visible_mask[iy, ix] = True

                # The ground-truth map is only used internally by the simulator to generate
                # simulated sensor observations. The robot/agent does not receive this map;
                # it only receives the partially observed occupancy_grid.
                if self.true_map[iy, ix] == 1:
                    break  # if cell is an obstacle, stop it

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
        # belief_display = self.belief_map.copy()

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

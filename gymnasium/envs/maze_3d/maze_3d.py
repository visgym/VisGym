# Credit: https://github.com/koulanurag/maze-world

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import FuncConditional
import numpy as np
from typing import Tuple, Optional, Dict
from copy import copy
import ast

from ursina import Ursina, Entity, camera, color, window, load_texture, scene, destroy
from panda3d.core import PNMImage

_uapp = None

import random

class RandomMaze3DEnv(gym.Env, gym.VLMEnvMixin):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
    ACTION_TURN_LEFT = 0
    ACTION_TURN_RIGHT = 1
    ACTION_MOVE_FORWARD = 2
    ACTION_CAMERA_TURN_LEFT = 3
    ACTION_CAMERA_TURN_RIGHT = 4
    ACTION_CAMERA_U_TURN = 5
    DIRECTION_TO_VECTOR = {NORTH: (0, 1), EAST: (1, 0), SOUTH: (0, -1), WEST: (-1, 0)}
    DIRECTION_TO_ROTATION_Y = {NORTH: 0, EAST: 90, SOUTH: 180, WEST: 270}
    
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        generate_maze_fn: callable,
        maze_width: int = None,
        maze_height: int = None,
        draw_grids: bool = False,
        seed: int = None,
        cell_size: int = 10,
        render_size: Tuple[int, int] | None = (336, 336),
    ):
        self.generate_maze_fn = generate_maze_fn
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.draw_grids = draw_grids
        self.seed(seed)
        self._action_to_direction = {3: np.array([0, -1]), 0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1, 0])}
        self._cam_dir = 0
        self._cell_size = cell_size
        self._cam_h = cell_size / 2
        self._render_size = render_size
        self.initial_state = None

        self._app_initialized = False
        self._agent_sphere = None
        self._pnm_img = None
        self.action_space = FuncConditional({"move": spaces.Discrete(1), "turn": spaces.Discrete(3, start=1), "stop": spaces.Text(4)})
        self.observation_space = spaces.Dict({"agent": spaces.Box(low=0, high=2, shape=(self.maze_height, self.maze_width), dtype=int),
                            "target": spaces.Box(low=0, high=2, shape=(self.maze_height, self.maze_width), dtype=int)})
    
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        prompt = (
            f"You are navigating a {self.maze_height}x{self.maze_width} 3D maze environment. "
            "The maze consists of walls and open paths. You are given the first-person view from your current position and orientation.\n\n"
            "Your goal is to reach the target location which ismarked by a red sphere."
        )

        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Move one step forward in your current facing direction. "
                "Format: `('move', 0)`"
            )
        if "turn" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'turn': Rotate your view in the specified direction. "
                "Format: `('turn', direction)` where direction is 1 (left), 2 (right), or 3 (around)."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Stop the episode. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        prompt += "\n\nSuccess: You succeed when you reach the target location (red sphere)."

        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move forward: `('move', 0)`")
        if "turn" in actions:
            examples.append("- To turn left: `('turn', 1)`")
            examples.append("- To turn right: `('turn', 2)`")
            examples.append("- To turn around: `('turn', 3)`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        
        prompt += "\n\nNote: If you try to move forward into a wall, you will remain in your current position. Turning actions do not change your position, only your facing direction."
        
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'maze_map': self.maze_map.tolist(),
            'agent_location': self._agent_location.tolist(),
            'target_location': self._target_location.tolist(),
            'cam_dir': int(self._cam_dir),
        }
    
    def reset(self, seed: int = None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        # Initialize Ursina before clearing scene (needed for multiprocessing)
        self._initialize_ursina()
        scene.clear()
        self._agent_sphere = None

        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            self.maze_map = np.array(init_state['maze_map'])
            self._agent_location = np.array(init_state['agent_location'])
            self._target_location = np.array(init_state['target_location'])
            self._cam_dir = init_state['cam_dir']
        else:
            # Normal reset: generate a new maze
            self.maze_map, self._agent_location, self._target_location = self.generate_maze_fn()
            if not np.array_equal(self.maze_map.shape, [self.maze_height, self.maze_width]):
                raise ValueError("Shape of Generated Maze doesn't match with specified maze width and height.")
            
            self._cam_dir = self._pick_initial_camera_dir()
        
        if self.render_mode == "rgb_array":
             self._build_scene_entities()

        self.initial_state = f"maze_map: {self.maze_map}, agent_location: {self._agent_location}, target_location: {self._target_location}"
        
        return self._get_obs(), self._get_info()

    def step(self, action: str):
        terminated = truncated = False
        reward = 0.0
        self.env_feedback = None

        try:
            parsed = ast.literal_eval(action)
        except Exception as exc:
            self.env_feedback = f"Cannot parse action string {action!r}: {exc} so the action is invalid."
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()
        
        if (not isinstance(parsed, tuple)) or len(parsed) != 2:
            self.env_feedback = f"The action format is invalid. The action should be a tuple of two elements with the first element being the action name and the second element being the action payload."
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()
        
        branch, payload = parsed[0], parsed[1]

        # 3. Execute ------------------------------------------------------
        if branch == "move":
            if not self.action_space["move"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                direction = self._action_to_direction[self._cam_dir]
                new_loc = self._agent_location + direction
                if not (self.maze_map[new_loc[0], new_loc[1]] == 1):
                    self._agent_location = new_loc
                else:
                    self.env_feedback = "Cannot move forward - wall collision"
        elif branch == "turn":
            if not self.action_space["turn"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                if payload == 1: self._cam_dir = (self._cam_dir + 1) % 4
                elif payload == 2: self._cam_dir = (self._cam_dir + 3) % 4
                elif payload == 3: self._cam_dir = (self._cam_dir + 2) % 4
        elif branch == "stop":
            if not self.action_space["stop"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                terminated = True
                truncated = False
        else:
            self.env_feedback = f"The action name '{branch}' is not recognized in the available action space."
        
        # 4. Compute reward ------------------------------------------------------
        if terminated:
            reward = self._compute_reward()
        
        # 5. Return ------------------------------------------------------
        if self.env_feedback is None:
            self.env_feedback = "Action executed successfully."
        info = self._get_info()
        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def render(self, mode = "rgb_array"):
        if mode == "rgb_array":
            return self._render_frame()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = None, num_steps: int = None):
        """
        Returns a sequence of actions that will solve the maze.
        If num_steps is provided, pad the solution with dummy turn pairs to meet the
        minimum step requirement without changing the net orientation or position.
        
        Args:
            num_steps (int, optional): The minimum number of steps in the returned trajectory.
        
        Returns:
            list[int]: A sequence of actions (0=forward, 1=turn_left, 2=turn_right, 3=turn_around)
        """
        def _to_func_actions(int_actions: list[int]) -> list[str]:
            mapping = {
                0: "('move', 0)",
                1: "('turn', 1)",  # left
                2: "('turn', 2)",  # right
                3: "('turn', 3)",  # around
            }
            out = [mapping.get(int(a), "('move', 0)") for a in int_actions]
            # Always append an explicit stop signal to match the desired action space
            out.append("('stop', 'stop')")
            return out

        try:
            from .maze_solver import MazeSolver
        except ImportError:
            # Fallback to simple heuristic if maze_solver is not available
            actions = self._solve_heuristic()
            # Optional padding with dummy pairs (left then right)
            if num_steps is not None and len(actions) < num_steps:
                num_padding_needed = num_steps - len(actions)
                num_pairs_to_add = (num_padding_needed + 1) // 2
                insert_indices = list(self.np_random.integers(0, len(actions) + 1, size=num_pairs_to_add))
                insert_indices.sort(reverse=True)
                for idx in insert_indices:
                    # Insert a no-op pair: turn left then turn right
                    actions.insert(idx, 2)
                    actions.insert(idx, 1)
            return _to_func_actions(actions)
        
        # If already at target, just move forward (environment will handle termination)
        if np.array_equal(self._agent_location, self._target_location):
            return _to_func_actions([0])  # Move forward then stop
        
        # Prepare maze map for solver (remove agent marker)
        maze_map = copy(self.maze_map)
        
        # Solve the maze using MazeSolver
        solver = MazeSolver()
        path = solver.solve(maze_map, self._agent_location, self._target_location)
        
        if not path or len(path) < 2:
            # No path found, use heuristic fallback
            actions = self._solve_heuristic()
            if num_steps is not None and len(actions) < num_steps:
                num_padding_needed = num_steps - len(actions)
                num_pairs_to_add = (num_padding_needed + 1) // 2
                insert_indices = list(self.np_random.integers(0, len(actions) + 1, size=num_pairs_to_add))
                insert_indices.sort(reverse=True)
                for idx in insert_indices:
                    actions.insert(idx, 2)
                    actions.insert(idx, 1)
            return _to_func_actions(actions)
        
        # Convert path to actions based on current camera direction and env action semantics
        # Build a mapping from movement vectors to direction indices (cam_dir)
        vec_to_dir = {tuple(v.tolist()): k for k, v in self._action_to_direction.items()}
        actions = []
        current_dir = self._cam_dir

        for i in range(1, len(path)):
            prev_pos = np.array(path[i-1])
            next_pos = np.array(path[i])
            delta = tuple((next_pos - prev_pos).tolist())

            if delta not in vec_to_dir:
                continue

            desired_dir = vec_to_dir[delta]
            # Compute minimal rotation from current_dir to desired_dir
            turn_diff = (desired_dir - current_dir) % 4
            if turn_diff == 1:
                actions.append(1)  # turn left
            elif turn_diff == 2:
                actions.append(3)  # turn around
            elif turn_diff == 3:
                actions.append(2)  # turn right
            # Move forward
            actions.append(0)
            current_dir = desired_dir
        
        # Pad with dummy turn pairs if num_steps is specified
        if num_steps is not None and len(actions) < num_steps:
            num_padding_needed = num_steps - len(actions)
            # Each padding op adds a (turn_left, turn_right) pair => 2 steps
            num_pairs_to_add = (num_padding_needed + 1) // 2
            # Choose insertion indices along the action list
            insert_indices = list(self.np_random.integers(0, len(actions) + 1, size=num_pairs_to_add))
            insert_indices.sort(reverse=True)  # Insert from back to preserve earlier indices
            for idx in insert_indices:
                # Insert a no-op pair that preserves orientation overall
                actions.insert(idx, 2)  # turn right
                actions.insert(idx, 1)  # turn left

        return _to_func_actions(actions)

    def close(self):
        pass 

    def _get_obs(self):
        image = self.render()
        return image

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
                "agent": self._agent_location, "target": self._target_location, 
                "cam_dir": self._cam_dir, "maze_map": self.maze_map, "env_feedback": self.env_feedback}
    
    def _compute_reward(self):
        return 1.0 if np.array_equal(self._agent_location, self._target_location) else 0.0

########################################################
# End of all methods, start of helper methods
########################################################
    
    def _solve_heuristic(self):
        """
        Fallback heuristic when MazeSolver is not available.
        Uses simple greedy approach towards target with turning logic.
        """
        if np.array_equal(self._agent_location, self._target_location):
            return [0]  # Move forward
        
        # Calculate the direction vector we want to go
        diff = self._target_location - self._agent_location
        
        # Determine the ideal direction to face
        target_dir = None
        if abs(diff[1]) >= abs(diff[0]):  # Horizontal movement preferred
            if diff[1] > 0:
                target_dir = 1  # East
            else:
                target_dir = 3  # West
        else:  # Vertical movement preferred
            if diff[0] < 0:
                target_dir = 0  # North
            else:
                target_dir = 2  # South
        
        # Calculate the turn needed to face the target direction
        current_dir = self._cam_dir
        turn_diff = (target_dir - current_dir) % 4
        
        # Check if we can move forward in current direction
        direction_vec = self._action_to_direction[current_dir]
        next_pos = self._agent_location + direction_vec
        can_move_forward = (0 <= next_pos[0] < self.maze_height and 
                           0 <= next_pos[1] < self.maze_width and 
                           self._no_obstacle(next_pos))
        
        # If facing the right direction and can move forward, do so
        if turn_diff == 0 and can_move_forward:
            return [0]  # Move forward
        
        # Check if we can move forward in the target direction
        target_direction_vec = self._action_to_direction[target_dir]
        target_next_pos = self._agent_location + target_direction_vec
        target_path_clear = (0 <= target_next_pos[0] < self.maze_height and 
                            0 <= target_next_pos[1] < self.maze_width and 
                            self._no_obstacle(target_next_pos))
        
        # If target direction is clear, turn towards it
        if target_path_clear:
            if turn_diff == 1:
                return [1]  # Turn left
            elif turn_diff == 2:
                return [3]  # Turn around
            elif turn_diff == 3:
                return [2]  # Turn right
        
        # If target direction is blocked, try other directions
        for action in [1, 2, 3]:  # Try turning left, right, around
            test_dir = (current_dir + [1, 3, 2][action-1]) % 4
            test_vec = self._action_to_direction[test_dir]
            test_pos = self._agent_location + test_vec
            if (0 <= test_pos[0] < self.maze_height and 
                0 <= test_pos[1] < self.maze_width and 
                self._no_obstacle(test_pos)):
                return [action]
        
        # If no direction is available, just turn around and hope for the best
        return [3]

    def _no_obstacle(self, location):
        return not (self.maze_map[location[0], location[1]] == 1)  # check for walls
    
    def _cells_clear_between(self, start: np.ndarray, target: np.ndarray) -> bool:
        """Return True if start and target share a row or column and there are
        no walls between them (exclusive of endpoints)."""
        r0, c0 = int(start[0]), int(start[1])
        r1, c1 = int(target[0]), int(target[1])
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1, step):
                if not (0 <= r0 < self.maze_height and 0 <= c < self.maze_width):
                    return False
                if self.maze_map[r0, c] == 1:
                    return False
            return True
        if c0 == c1:
            step = 1 if r1 > r0 else -1
            for r in range(r0 + step, r1, step):
                if not (0 <= r < self.maze_height and 0 <= c0 < self.maze_width):
                    return False
                if self.maze_map[r, c0] == 1:
                    return False
            return True
        return False

    def _has_clear_los_in_direction(self, start: np.ndarray, target: np.ndarray, dir_idx: int) -> bool:
        """Does target lie straight ahead along dir_idx with no walls between?"""
        vec = self._action_to_direction[dir_idx]
        dr, dc = int(vec[0]), int(vec[1])
        r0, c0 = int(start[0]), int(start[1])
        r1, c1 = int(target[0]), int(target[1])
        # Horizontal
        if dr == 0 and r0 == r1:
            if dc > 0 and c1 > c0 and self._cells_clear_between(start, target):
                return True
            if dc < 0 and c1 < c0 and self._cells_clear_between(start, target):
                return True
        # Vertical
        if dc == 0 and c0 == c1:
            if dr > 0 and r1 > r0 and self._cells_clear_between(start, target):
                return True
            if dr < 0 and r1 < r0 and self._cells_clear_between(start, target):
                return True
        return False

    def _distance_to_wall(self, start: np.ndarray, dir_idx: int) -> int:
        """Return the number of free cells from start until the next wall in dir_idx."""
        vec = self._action_to_direction[dir_idx]
        dr, dc = int(vec[0]), int(vec[1])
        r, c = int(start[0]), int(start[1])
        dist = 0
        while True:
            r += dr
            c += dc
            if not (0 <= r < self.maze_height and 0 <= c < self.maze_width):
                break
            if self.maze_map[r, c] == 1:
                break
            dist += 1
        return dist

    def _pick_initial_camera_dir(self) -> int:
        """Pick a camera direction so the target isn't directly visible at reset.
        The camera is placed one cell ahead along cam dir and looks back at the agent,
        so the view direction aligns with cam_dir beyond the agent. We therefore try
        to avoid clear line-of-sight along cam_dir from agent to target.
        """
        # 1) Open directions (cell in front is free)
        open_dirs = []
        for d in range(4):
            nxt = self._agent_location + self._action_to_direction[d]
            if (0 <= nxt[0] < self.maze_height and 0 <= nxt[1] < self.maze_width and self.maze_map[tuple(nxt)] == 0):
                open_dirs.append(d)

        # 2) Keep those whose forward LOS does not hit target
        safe_dirs = []
        for d in open_dirs:
            if not self._has_clear_los_in_direction(self._agent_location, self._target_location, d):
                safe_dirs.append(d)

        if safe_dirs:
            idx = int(self.np_random.integers(0, len(safe_dirs)))
            return safe_dirs[idx]

        if open_dirs:
            # 3) If unavoidable, choose dir with nearest wall ahead to block view
            best_d = None
            best_dist = None
            for d in open_dirs:
                dist = self._distance_to_wall(self._agent_location, d)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_d = d
            return best_d

        # 4) Surrounded: default north
        return 0

    def _initialize_ursina(self):
        """Initializes the Ursina application. This should only be called once."""
        global _uapp
        # Check both flag and actual app existence (handles multiprocessing where
        # _app_initialized may be True from pickling but _uapp is None in new process)
        if self._app_initialized and _uapp is not None:
            return

        try:
            from panda3d.core import loadPrcFileData
            if self._render_size is not None:
                w, h = int(self._render_size[0]), int(self._render_size[1])
                loadPrcFileData("", f"win-size {w} {h}")
        except Exception:
            pass
        
        _uapp = Ursina(borderless=True, vsync=False, window_type='offscreen')
        
        self._pnm_img = PNMImage()
        self._app_initialized = True

    def _build_scene_entities(self):
        """Builds the visual entities for the current maze layout."""
        self._initialize_ursina()

        from pathlib import Path
        asset_dir = Path(__file__).parent / 'assets/images/textures'
        wall_tex = load_texture('wall_diffuse.jpg', path=str(asset_dir))
        gnd_tex  = load_texture('ground_diffuse.jpg', path=str(asset_dir))
        roof_tex = load_texture('roof_diffuse.jpg', path=str(asset_dir))

        h, w = self.maze_height, self.maze_width
        off_x = -w * self._cell_size / 2 + self._cell_size / 2
        off_z = h * self._cell_size / 2 - self._cell_size / 2

        for r in range(h):
            for c in range(w):
                cell = self.maze_map[r, c]
                x = c * self._cell_size + off_x
                z = -r * self._cell_size + off_z
                if cell == 1:            # Wall
                    Entity(model='cube', texture=wall_tex,
                        scale=(self._cell_size, self._cell_size, self._cell_size),
                        position=(x, self._cell_size/2, z), collider='box')

        ground = Entity(
            model='plane', texture=roof_tex, collider='box',
            scale=(w*self._cell_size, 1, h*self._cell_size),
            position=(0, 0, 0),
            texture_scale=(w, h)          # Make floor texture repeat by grid
        )
        if gnd_tex:
            gnd_tex.wrap_u = gnd_tex.wrap_v = 'repeat'

        roof = Entity(
            model='plane', texture=gnd_tex, collider='box',
            scale=(w*self._cell_size, 1, h*self._cell_size),
            position=(0, self._cell_size, 0), rotation_x=180,
            double_sided=True,
            texture_scale=(w, h)
        )
        if roof_tex:
            roof_tex.wrap_u = roof_tex.wrap_v = 'repeat'

        # Target sphere
        gx, gy = self._target_location[1], self._target_location[0]
        Entity(model='sphere', color=color.red, scale=self._cell_size * 0.3,
               position=(gx * self._cell_size + off_x, self._cell_size / 2, -gy * self._cell_size + off_z))

        if self._agent_sphere is None:
             self._agent_sphere = Entity(model='sphere', color=color.azure, scale=self._cell_size * 1e-8)


    def _render_frame(self):
        global _uapp
        self._initialize_ursina()

        # Rebuild scene if entities are missing or agent sphere wasn't created
        if not scene.entities or self._agent_sphere is None:
            self._build_scene_entities()
        
        h, w = self.maze_height, self.maze_width
        off_x = -w * self._cell_size / 2 + self._cell_size / 2
        off_z = h * self._cell_size / 2 - self._cell_size / 2
        ax, ay = self._agent_location[1], self._agent_location[0]
        self._agent_sphere.position = (
            ax * self._cell_size + off_x,
            self._cell_size / 2,
            -ay * self._cell_size + off_z
        )

        offset_map = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)} # North, West, South, East
        dx, dz = offset_map[self._cam_dir]
        camera.position = (
            self._agent_sphere.x + dx * self._cell_size,
            self._cam_h,
            self._agent_sphere.z + dz * self._cell_size
        )
        camera.look_at(self._agent_sphere.position)

        _uapp.step()
        
        base.win.getScreenshot(self._pnm_img) # this might be a global variable
        w_pix, h_pix = self._pnm_img.get_x_size(), self._pnm_img.get_y_size()
        
        arr = np.empty((h_pix, w_pix, 3), dtype=np.uint8)
        for y in range(h_pix):
            for x in range(w_pix):
                arr[y, x, 0] = self._pnm_img.get_red_val(x, h_pix - 1 - y)
                arr[y, x, 1] = self._pnm_img.get_green_val(x, h_pix - 1 - y)
                arr[y, x, 2] = self._pnm_img.get_blue_val(x, h_pix - 1 - y)

        return arr


class Maze3DEnv(RandomMaze3DEnv):
    r"""Extends the MazeEnv class to create random mazes of specified sizes at each reset.

    :example:
        >>> env = RandomMaze3DEnv(maze_width=5,maze_height=5)
    """

    def __init__(
        self,
        maze_width: int = 11,
        maze_height: int = 11,
        seed: int = None,
        render_size: Tuple[int, int] | None = (336, 336),
    ):
        r"""
        :param render_mode: Specify one of the following:

            - None (default): No render is computed.
            - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption. This rendering should occur during step() and render() doesn't need to be called. Returns None.
            - "rgb_array": Return a single frame representing the current state of the environment. A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
            - "ansi": Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
            - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the wrapper, gymnasium.wrappers.RenderCollection that is automatically applied during gymnasium.make(...,render_mode="rgb_array_list"). The frames collected are popped after render() is called or reset().

        :param maze_width: The width of the maze.
        :param maze_height: The height of the maze.

        :raises ValueError: If the width or height of the maze is not odd.
        """
        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height
        self.env_feedback = ""
        self._agent_sphere = None

        def _generate_maze():
            """
            Generates a random internal maze configuration.

            :return: A tuple containing the maze configuration, agent location, and target location.
            :rtype: tuple
            """

            # generate internal maze( other than outside wall area)
            generator = WilsonMazeGenerator(maze_height - 2, maze_width - 2, seed=int(self.np_random.integers(2**32)))
            generator.generate_maze()

            maze_config = np.ones((maze_height, maze_width), dtype=int)

            # fill maze by skipping outside walls
            maze_config[1:-1, 1:-1] = 1 - np.array(generator.grid)
            a=[
                    ([1, 1], [maze_config.shape[0] - 2, maze_config.shape[1] - 2]),
                    ([maze_config.shape[0] - 2, maze_config.shape[1] - 2], [1, 1]),
                    ([maze_config.shape[0] - 2, 1], [1, maze_config.shape[1] - 2]),
                    ([1, maze_config.shape[1] - 2], [maze_config.shape[0] - 2, 1]),
            ]
            idx = self.np_random.integers(0, len(a))
            agent_location, target_location = a[idx]
            agent_location, target_location = np.array(agent_location), np.array(target_location)
            return maze_config, agent_location, target_location

        RandomMaze3DEnv.__init__(
            self,
            generate_maze_fn=_generate_maze,
            maze_width=maze_width,
            maze_height=maze_height,
            seed=seed,
            render_size=render_size,
        )

class WilsonMazeGenerator:
    """
    Maze Generator utilizing Wilson's Loop Erased Random Walk Algorithm.

    Credit: https://github.com/CaptainFl1nt/WilsonMazeGenerator
    """

    def __init__(self, height: int, width: int, seed: Optional[int] = None):
        """
        Initializes a maze generator with the specified width and height.

        :param height: Height of the generated mazes.
        :type height: int
        :param width: Width of the generated mazes.
        :type width: int
        """
        self.width = 2 * (width // 2) + 1  # Make width odd
        self.height = 2 * (height // 2) + 1  # Make height odd
        self.rng = random.Random(seed)

        # grid of cells
        self.grid = [[0 for j in range(self.width)] for i in range(self.height)]

        # declare instance variable
        self.visited = []  # visited cells
        self.unvisited = []  # unvisited cells
        self.path = dict()  # random walk path

        # valid directions in random walk
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # indicates whether a maze is generated
        self.generated = False

        # shortest solution
        self.solution = []
        self.showSolution = False
        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)

    def __str__(self):
        """
        Returns a string representation of the maze grid.

        :return: String representation of the grid.
        :rtype: str
        """
        out = "##" * (self.width + 1) + "\n"
        for i in range(self.height):
            out += "#"
            for j in range(self.width):
                if self.grid[i][j] == 0:
                    out += "##"
                else:
                    if not self.showSolution:
                        out += "  "
                    elif (i, j) in self.solution:
                        out += "**"
                    else:
                        out += "  "
            out += "#\n"
        return out + "##" * (self.width + 1)

    def get_grid(self):
        """
        Retrieve the maze grid.

        :return: The maze grid.
        :rtype: list
        """
        return self.grid

    def get_solution(self):
        """
        Returns the solution to the maze as a list of tuples.

        :return: The solution to the maze.
        :rtype: list
        """
        return self.solution

    def show_solution(self, show):
        """
        Sets whether the __str__() method outputs the solution or not.

        :param show: Boolean value indicating whether to show the solution or not.
        :type show: bool
        """
        self.showSolution = show

    def generate_maze(self):
        """
        Generates the maze according to the Wilson Loop Erased Random Walk Algorithm.

        The algorithm works as follows:
            1. Reset the grid before generation.
            2. Choose the first cell to put in the visited list.
            3. Loop until all cells have been visited:
                a. Choose a random cell to start the walk.
                b. Loop until the random walk reaches a visited cell.
                c. Loop until the end of the path is reached:
                    - Add the cell to visited and cut into the maze.
                    - Follow the direction to the next cell.

        :return: None
        """
        # reset the grid before generation
        self.__initialize_grid()

        # choose the first cell to put in the visited list
        # see Step 1 of the algorithm.
        current = self.unvisited.pop(self.rng.randint(0, len(self.unvisited) - 1))
        self.visited.append(current)
        self._cut(current)

        # loop until all cells have been visited
        while len(self.unvisited) > 0:
            # choose a random cell to start the walk (Step 2)
            first = self.unvisited[self.rng.randint(0, len(self.unvisited) - 1)]
            current = first
            # loop until the random walk reaches a visited cell
            while True:
                # choose direction to walk (Step 3)
                dirNum = self.rng.randint(0, 3)
                # check if direction is valid. If not, choose new direction
                while not self.__is_valid_direction(current, dirNum):
                    dirNum = self.rng.randint(0, 3)
                # save the cell and direction in the path
                self.path[current] = dirNum
                # get the next cell in that direction
                current = self.__get_next_cell(current, dirNum, 2)
                if current in self.visited:  # visited cell is reached (Step 5)
                    break

            current = first  # go to start of path
            # loop until the end of path is reached
            while True:
                # add cell to visited and cut into the maze
                self.visited.append(current)
                self.unvisited.remove(current)  # (Step 6.b)
                self._cut(current)

                # follow the direction to next cell (Step 6.a)
                dirNum = self.path[current]
                crossed = self.__get_next_cell(current, dirNum, 1)
                self._cut(crossed)  # cut crossed edge

                current = self.__get_next_cell(current, dirNum, 2)
                if current in self.visited:  # end of path is reached
                    self.path = dict()  # clear the path
                    break

        self.generated = True

    def solve_maze(self):
        """Solves the maze according to the Wilson Loop Erased Random Walk Algorithm

        :return: None
        """
        # if there is no maze to solve, cut the method
        if not self.generated:
            return None

        # initialize with empty path at starting cell
        self.path = dict()
        current = self.start

        # loop until the ending cell is reached
        while True:
            while True:
                dirNum = self.rng.randint(0, 3)
                adjacent = self.__get_next_cell(current, dirNum, 1)
                if self.__is_valid_direction(current, dirNum):
                    hasWall = self.grid[adjacent[0]][adjacent[1]] == 0
                    if not hasWall:
                        break
            # add cell and direction to path
            self.path[current] = dirNum

            # get next cell
            current = self.__get_next_cell(current, dirNum, 2)
            if current == self.end:
                break  # break if ending cell is reached

        # go to start of path
        current = self.start
        self.solution.append(current)
        # loop until end of path is reached
        while not (current == self.end):
            dirNum = self.path[current]  # get direction
            # add adjacent and crossed cells to solution
            crossed = self.__get_next_cell(current, dirNum, 1)
            current = self.__get_next_cell(current, dirNum, 2)
            self.solution.append(crossed)
            self.solution.append(current)

        self.path = dict()

    ## Private Methods ##
    ## Do Not Use Outside This Class ##

    def __get_next_cell(self, cell, dirNum, fact):
        """WilsonMazeGenerator.get_next_cell(tuple,int,int) -> tuple
        Outputs the next cell when moved a distance fact in the the
        direction specified by dirNum from the initial cell.
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3
        fact: int distance to next cell"""
        dirTup = self.directions[dirNum]
        return (cell[0] + fact * dirTup[0], cell[1] + fact * dirTup[1])

    def __is_valid_direction(self, cell, dirNum):
        """WilsonMazeGenerator(tuple,int) -> boolean
        Checks if the adjacent cell in the direction specified by
        dirNum is within the grid
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3"""
        newCell = self.__get_next_cell(cell, dirNum, 2)
        tooSmall = newCell[0] < 0 or newCell[1] < 0
        tooBig = newCell[0] >= self.height or newCell[1] >= self.width
        return not (tooSmall or tooBig)

    def __initialize_grid(self):
        """
        Resets the maze grid to blank before generating a maze.

        :return: None
        """
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = 0

        # fill up unvisited cells
        for r in range(self.height):
            for c in range(self.width):
                if r % 2 == 0 and c % 2 == 0:
                    self.unvisited.append((r, c))

        self.visited = []
        self.path = dict()
        self.generated = False

    def _cut(self, cell):
        """
        Sets the value of the grid at the specified location to 1, indicating a cut.

        :param tuple cell: A tuple (y, x) representing the location where the cut should occur.
        :return: None
        """
        self.grid[cell[0]][cell[1]] = 1
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import random
import numpy as np
import trimesh
import ast
from gymnasium.utils import seeding
from gymnasium.spaces import FuncConditional, Text
from trimesh.transformations import euler_from_matrix, rotation_from_matrix, rotation_matrix
from typing import Dict, Optional

class MentalRotation3DCubeEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        num_segments: int = 4,
        length_range=(2, 5),
        image_size=(128, 128),
        angle_tol: float = np.deg2rad(10.0),
        cube_size: float = 1.0,
        seed: int = None, 
        action_frame: str = "object",
        *args, **kwargs     
    ):
        super().__init__()
        self.seed(seed)
        self.image_size = image_size
        self.angle_tol = angle_tol
        self.num_segments = num_segments
        self.length_range = length_range
        self.cube_size = cube_size
        if action_frame not in ("object", "world"):
            raise ValueError("action_frame must be 'object' or 'world'")
        self.action_frame = action_frame

        # actions: 3‐D continuous Euler increments + stop bit
        self.action_space = FuncConditional({
            "rotate": spaces.Box(
                low=np.array([-180.0, -180.0, -180.0], dtype=np.float32),
                high=np.array([ 180.0,  180.0,  180.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "stop":    Text(4)    # exactly like your other envs
        })

        H, W = image_size
        self.observation_space = spaces.Box(
            0, 255, shape=(H, W*2, 3), dtype=np.uint8
        )

        # placeholders
        self.original_img = None
        self.current_rot = None

    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            "You see two images side-by-side:\n"
            "- Left: the original 3D object (target orientation)\n"
            "- Right: the same object under an unknown rotation\n\n"
            "Your goal is to rotate the right object back to the original target orientation."
        )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "rotate" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'rotate': Rotate the 3D object by Euler angles. "
                "Format: `('rotate', [dyaw, dpitch, droll])` where each of dyaw, dpitch, droll "
                "is in degrees between -180 and 180."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Submit your final rotation. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Describe the coordinate frame used for rotations
        frame_note = (
            "\n\nNote: Rotations are applied in the object's local (intrinsic) axes"
            if self.action_frame == "object" else
            "\n\nNote: Rotations are applied in the world (extrinsic) axes"
        )
        prompt += frame_note

        # Add success criteria
        prompt += f"\n\nSuccess: You succeed if you end up within {np.rad2deg(self.angle_tol):.1f} degrees of target orientation."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "rotate" in actions:
            examples.append("- To rotate in yaw: `('rotate', [15.0, 0.0, 0.0])`")
            examples.append("- To rotate in roll: `('rotate', [0.0, 0.0, -10.0])`")
        if "stop" in actions:
            examples.append("- To submit: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'shape_state': self.shape_state,
            'secret_yaw': float(self.secret_yaw),
            'secret_pitch': float(self.secret_pitch),
            'secret_roll': float(self.secret_roll),
            'current_rot': self.current_rot.tolist(),
        }
    
    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            self.shape_state = init_state['shape_state']
            self.base_mesh, self.shape_state = _generate_segmented_shape(
                num_segments=self.num_segments,
                length_range=self.length_range,
                seed=None,
                cube_size=self.cube_size,
                state=self.shape_state,
            )
            
            # 1) pre-render the original (identity) view
            self.original_img = self._render(rotation=np.eye(3))
            
            # 2) restore the secret rotation from saved state
            self.secret_yaw = float(init_state['secret_yaw'])
            self.secret_pitch = float(init_state['secret_pitch'])
            self.secret_roll = float(init_state['secret_roll'])
            
            R0 = trimesh.transformations.euler_matrix(
                self.secret_yaw, self.secret_pitch, self.secret_roll, axes='szyx'
            )[:3,:3]
            
            self.current_rot = np.array(init_state['current_rot'])
            self.secret_matrix = np.linalg.inv(R0)
        else:
            # Normal reset: generate new shape and rotation
            self.base_mesh, self.shape_state = _generate_segmented_shape(
                num_segments=self.num_segments,
                length_range=self.length_range,
                seed=seed,
                cube_size=self.cube_size,
                state=None,
            )

            # 1) pre-render the original (identity) view
            self.original_img = self._render(rotation=np.eye(3))

            # 2) pick a random hidden rotation
            yaw, pitch, roll = self.np_random.uniform(0, 2*np.pi, size=3)
            self.secret_yaw = float(yaw)
            self.secret_pitch = float(pitch)
            self.secret_roll = float(roll)
            
            R0 = trimesh.transformations.euler_matrix(
                yaw, pitch, roll, axes='szyx'
            )[:3,:3]
            
            I = trimesh.transformations.euler_matrix(
                0, 0, 0, axes='szyx'
            )[:3,:3]
            
            self.current_rot = R0
            self.secret_matrix = np.linalg.inv(R0)

        # 3) return side‐by‐side
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
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
        if branch == "rotate":
            if not self.action_space["rotate"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                rot_inc = np.array(payload, dtype=np.float32)
                # Convert degrees to radians for trimesh
                rot_inc_rad = np.deg2rad(rot_inc)
                R_inc = trimesh.transformations.euler_matrix(
                    float(rot_inc_rad[0]), float(rot_inc_rad[1]), float(rot_inc_rad[2]),
                    axes="szyx"
                )[:3, :3]
                # Apply in chosen frame
                if self.action_frame == "object":
                    # local/intrinsic: post-multiply
                    self.current_rot = self.current_rot @ R_inc
                else:
                    # world/extrinsic: pre-multiply
                    self.current_rot = R_inc @ self.current_rot
                self.env_feedback = None
        
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

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._get_obs()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")


    def solve(self, num_steps: int = 3, uniform: bool = True, strategy: str = "pad3_by-axis") -> list[str]:
        """
        Returns a sequence of actions that will solve the mental rotation puzzle.
        Uses axis-angle representation to correctly subdivide the rotation.
        The rotation is split into a stochastic number of steps.

        Args:
            num_steps (int): The number of steps to spread the solution over.
            uniform (bool): Whether to subdivide the rotation uniformly.
            strategy (str): The strategy to use for solving. Can be "axis-angle", "by-axis",
                          or compositional strategies like "pad1_axis-angle", "pad1_by-axis",
                          "pad3_axis-angle", "pad3_by-axis".

        Returns:
            List[str]: A list of action strings.
        """
        # Check for compositional strategies
        if strategy.startswith("pad1_"):
            base_strategy = strategy[5:]  # Remove "pad1_" prefix
            return self._apply_pad1_strategy(base_strategy, num_steps, uniform)
        elif strategy.startswith("pad3_"):
            base_strategy = strategy[5:]  # Remove "pad3_" prefix
            return self._apply_pad3_strategy(base_strategy, num_steps, uniform)
        elif strategy == "by-axis":
            return self._solve_by_axis()
        elif strategy == "axis-angle":
            return self._solve_axis_angle(num_steps, uniform)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def close(self):
        """Clean up matplotlib state to prevent multiprocessing issues."""
        try:
            plt.close('all')
            plt.clf()
            plt.cla()
        except:
            pass

    def _get_obs(self):
        return np.concatenate([self.original_img, self._render(self.current_rot)], axis=1)
    
    def _get_info(self):
        return {
            "rotation_error": self._rotation_distance(self.current_rot, np.eye(3)),
            "env_feedback": self.env_feedback
        }

    def _compute_reward(self):
        angle_err = self._rotation_distance(self.current_rot, np.eye(3))
        return 1.0 if angle_err <= self.angle_tol else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _rotation_distance(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """
        Returns the geodesic angle (in radians) between two 3×3 rotation matrices.
        """
        R_rel = R1 @ R2.T
        # numerical safety: clamp the trace into [-1, 3]
        trace = np.clip(np.trace(R_rel), -1.0, 3.0)
        # arccos input is (trace - 1) / 2, which lies in [-1..1]
        theta = np.arccos((trace - 1.0) / 2.0)
        return theta

    def _render(self, rotation: np.ndarray):
        """
        Headless Matplotlib 3D render of self.base_mesh under `rotation`.
        Returns an (H,W,3) uint8 BGR image.
        """
        # Set backend locally to avoid multiprocessing issues
        import matplotlib
        matplotlib.use("Agg", force=True)
        
        H, W = self.image_size
        fig = plt.figure(figsize=(W/100, H/100), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        # rotate vertices
        verts = (self.base_mesh.vertices @ rotation.T)
        faces = self.base_mesh.faces
        tris = verts[faces]  # (F,3,3)

        # simple Lambert shading
        v1 = tris[:,1] - tris[:,0]
        v2 = tris[:,2] - tris[:,0]
        normals = np.cross(v1, v2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        light = np.array([0,0,1])
        intensity = 0.2 + 0.8*np.clip(normals @ light, 0,1)
        facecolors = plt.cm.gray(1.0 - intensity)

        coll = Poly3DCollection(tris, facecolors=facecolors, linewidths=0.1, edgecolors='k')
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(verts.flatten(), verts.flatten(), verts.flatten())
        ax.view_init(elev=30, azim=45)

        fig.canvas.draw()
        buf, (w,h) = fig.canvas.print_to_buffer()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((h,w,4))[...,:3]
        
        # Clean up matplotlib state
        plt.close(fig)
        plt.clf()
        plt.cla()
        
        return img

    def _apply_pad1_strategy(self, base_strategy: str, num_steps: int, uniform: bool) -> list[str]:
        """
        Apply pad1 strategy: tour around the object to view all 6 faces and return to origin.
        Path: Front -> Right -> Back -> Left -> Top -> Bottom -> Front (7 steps covering all 6 faces).
        
        Args:
            base_strategy (str): The base strategy to use after padding ("axis-angle" or "by-axis").
            num_steps (int): Number of steps for the base strategy.
            uniform (bool): Whether to subdivide uniformly for axis-angle strategy.
            
        Returns:
            List[str]: Padded action sequence.
        """
        # 7-step tour visiting all 6 different faces
        # Starting from front (initial view)
        padding_actions = [
            str(('rotate', [90.0, 0.0, 0.0])),     # 1. Front -> Right (yaw +90)
            str(('rotate', [90.0, 0.0, 0.0])),     # 2. Right -> Back (yaw +90)
            str(('rotate', [90.0, 0.0, 0.0])),     # 3. Back -> Left (yaw +90)
            str(('rotate', [0.0, 90.0, 0.0])),     # 4. Left -> Top (pitch +90, looking up from left)
            str(('rotate', [0.0, 180.0, 0.0])),    # 5. Top -> Bottom (pitch +180, flipping over)
            str(('rotate', [0.0, 90.0, 0.0])),     # 6. Bottom -> Left (pitch +90, back to horizontal at left)
            str(('rotate', [90.0, 0.0, 0.0])),     # 7. Left -> Front (yaw +90, return to front)
        ]
        
        # Get the actual solution actions
        if base_strategy == "axis-angle":
            solution_actions = self._solve_axis_angle(num_steps, uniform)
        elif base_strategy == "by-axis":
            solution_actions = self._solve_by_axis()
        else:
            raise ValueError(f"Unknown base strategy for pad1: {base_strategy}")
        
        # Remove the stop action from solution if present
        if solution_actions and solution_actions[-1] == "('stop', 'stop')":
            solution_actions = solution_actions[:-1]
        
        # Combine padding + solution + stop
        return padding_actions + solution_actions + ["('stop', 'stop')"]

    def _apply_pad3_strategy(self, base_strategy: str, num_steps: int, uniform: bool) -> list[str]:
        """
        Apply pad3 strategy:
        - For "axis-angle": add 4*(90.0, 0, 0), 4*(0, 90.0, 0), 4*(0, 0, 90.0) before solution.
        - For "by-axis": add 4 padding steps before each corresponding axis rotation.
        
        Args:
            base_strategy (str): The base strategy to use ("axis-angle" or "by-axis").
            num_steps (int): Number of steps for the base strategy.
            uniform (bool): Whether to subdivide uniformly for axis-angle strategy.
            
        Returns:
            List[str]: Padded action sequence.
        """
        if base_strategy == "axis-angle":
            # For axis-angle: add all padding before the solution
            padding_actions = (
                [str(('rotate', [90.0, 0.0, 0.0]))] * 4 +
                [str(('rotate', [0.0, 90.0, 0.0]))] * 4 +
                [str(('rotate', [0.0, 0.0, 90.0]))] * 4
            )
            
            # Get the actual solution
            solution_actions = self._solve_axis_angle(num_steps, uniform)
            
            # Remove stop action from solution if present
            if solution_actions and solution_actions[-1] == "('stop', 'stop')":
                solution_actions = solution_actions[:-1]
            
            # Combine padding + solution + stop
            return padding_actions + solution_actions + ["('stop', 'stop')"]
            
        elif base_strategy == "by-axis":
            # For by-axis: add 4 padding steps before each axis rotation
            total_rotation_matrix = np.linalg.inv(self.current_rot)
            
            M_total_4x4 = np.eye(4)
            M_total_4x4[:3, :3] = total_rotation_matrix
            
            yaw, pitch, roll = euler_from_matrix(M_total_4x4, axes='szyx')
            
            actions = []
            
            # Yaw axis with padding
            if not np.isclose(yaw, 0):
                # Add 4 padding steps
                actions.extend([str(('rotate', [90.0, 0.0, 0.0]))] * 4)
                # Add actual rotation
                yaw_deg = round(np.rad2deg(yaw), 1)
                actions.append(str(('rotate', [yaw_deg, 0.0, 0.0])))
            
            # Pitch axis with padding
            if not np.isclose(pitch, 0):
                # Add 4 padding steps
                actions.extend([str(('rotate', [0.0, 90.0, 0.0]))] * 4)
                # Add actual rotation
                pitch_deg = round(np.rad2deg(pitch), 1)
                actions.append(str(('rotate', [0.0, pitch_deg, 0.0])))
            
            # Roll axis with padding
            if not np.isclose(roll, 0):
                # Add 4 padding steps
                actions.extend([str(('rotate', [0.0, 0.0, 90.0]))] * 4)
                # Add actual rotation
                roll_deg = round(np.rad2deg(roll), 1)
                actions.append(str(('rotate', [0.0, 0.0, roll_deg])))
            
            if not actions:
                # No rotation needed
                return ["('stop', 'stop')"]
            
            actions.append("('stop', 'stop')")
            return actions
        else:
            raise ValueError(f"Unknown base strategy for pad3: {base_strategy}")

    def _solve_by_axis(self) -> list[str]:
        """
        Solves the puzzle by rotating along each axis (yaw, pitch, roll) separately.
        """
        total_rotation_matrix = np.linalg.inv(self.current_rot)
        
        M_total_4x4 = np.eye(4)
        M_total_4x4[:3, :3] = total_rotation_matrix
        
        yaw, pitch, roll = euler_from_matrix(M_total_4x4, axes='szyx')
        
        actions = []
        
        # Create separate actions for yaw, pitch, and roll
        if not np.isclose(yaw, 0):
            yaw_deg = round(np.rad2deg(yaw), 1)
            actions.append(str(('rotate', [yaw_deg, 0.0, 0.0])))
            
        if not np.isclose(pitch, 0):
            pitch_deg = round(np.rad2deg(pitch), 1)
            actions.append(str(('rotate', [0.0, pitch_deg, 0.0])))
            
        if not np.isclose(roll, 0):
            roll_deg = round(np.rad2deg(roll), 1)
            actions.append(str(('rotate', [0.0, 0.0, roll_deg])))

        if not actions:
            # No rotation needed
            return ["('stop', 'stop')"]

        actions.append("('stop', 'stop')")
        return actions

    def _solve_axis_angle(self, num_steps: int = 3, uniform: bool = True) -> list[str]:
        """
        Original implementation using axis-angle decomposition.
        """
        if num_steps <= 0:
            num_steps = 1

        # Total rotation needed to get back to identity
        total_rotation_matrix = np.linalg.inv(self.current_rot)

        # Convert total rotation to axis-angle representation.
        # We need to wrap it in a 4x4 homogeneous transformation matrix.
        M_total_4x4 = np.eye(4)
        M_total_4x4[:3, :3] = total_rotation_matrix
        
        # trimesh.transformations.rotation_from_matrix returns angle, direction, point
        # We only need angle and direction.
        try:
            angle, direction, _ = rotation_from_matrix(M_total_4x4)
        except np.linalg.LinAlgError:
            # This can happen if the matrix is not a valid rotation matrix.
            # In this case, we can't find a solution.
            return ["('stop', 'stop')"]

        
        # If there's no rotation to be done, angle can be ~0.
        if np.isclose(angle, 0):
            # No rotation needed, just stop.
            return ["('stop', 'stop')"]

        # Subdivide the rotation angle into `num_steps` parts.
        if num_steps > 1:
            if uniform:
                # Uniformly divide the rotation angle
                step_angles = [angle / num_steps] * num_steps
            else:
                # Generate `num_steps - 1` random split points
                # Use abs(angle) to prevent errors from very small negative angles due to float precision.
                split_points = np.sort(self.np_random.uniform(0, abs(angle), num_steps - 1))
                # Create the sequence of angles
                step_angles = np.diff(np.concatenate(([0], split_points, [angle])))
        else:
            step_angles = [angle]

        # Create the sequence of actions
        actions = []
        for step_angle in step_angles:
            # Create the rotation matrix for a single step
            M_step_4x4 = rotation_matrix(step_angle, direction)
            
            # Convert the single-step rotation matrix back to Euler angles
            yaw, pitch, roll = euler_from_matrix(M_step_4x4, axes='szyx')
            step_rotation_euler = np.array([yaw, pitch, roll], dtype=np.float32)
            
            # Convert radians to degrees
            step_rotation_degrees = np.rad2deg(step_rotation_euler)

            # Truncate to 1 decimal place
            truncated_rotation = [round(float(a), 1) for a in step_rotation_degrees]
            action_tuple = ('rotate', truncated_rotation)
            actions.append(str(action_tuple))
            
        # Add the final "stop" action
        actions.append("('stop', 'stop')")
        
        return actions

def _generate_segmented_shape(
    num_segments: int = 4,
    length_range=(2, 5),
    seed: int | None = None,
    cube_size: float = 1.0,
    state: Dict | None = None
) -> tuple[trimesh.Trimesh, Dict]:
    """
    Returns:
      - mesh: the merged trimesh of unit‑cubes
      - state: dict containing 'seg_lens' and 'dir_indices' for reproducibility
    """
    AXES = [
        np.array([1,0,0]), np.array([-1,0,0]),
        np.array([0,1,0]), np.array([0,-1,0]),
        np.array([0,0,1]), np.array([0,0,-1]),
    ]
    
    if state is not None:
        # Use provided state to reconstruct the shape deterministically
        seg_lens = state['seg_lens']
        dir_indices = state['dir_indices']
    else:
        # Generate new random shape
        rng = random.Random(seed)
        seg_lens = [rng.randint(*length_range) for _ in range(num_segments)]
        dir_indices = []
        
        # We'll collect the direction choices as we go
        pos = np.array([0,0,0], dtype=int)
        dir_vec = np.array([1,0,0], dtype=int)
        
        for length in seg_lens:
            # pick a new orthogonal direction for the next segment
            candidates = [d for d in AXES if abs(d @ dir_vec) == 0]
            chosen_dir = rng.choice(candidates)
            # Find the index of the chosen direction
            dir_idx = next(i for i, d in enumerate(AXES) if np.array_equal(d, chosen_dir))
            dir_indices.append(dir_idx)
            dir_vec = chosen_dir

    # Build the coords using the seg_lens and dir_indices
    pos = np.array([0,0,0], dtype=int)
    dir_vec = np.array([1,0,0], dtype=int)
    coords = {tuple(pos)}
    
    for i, length in enumerate(seg_lens):
        for _ in range(length):
            pos = pos + dir_vec
            coords.add(tuple(pos))
        # Set direction for next segment (if not the last segment)
        if i < len(dir_indices):
            dir_vec = AXES[dir_indices[i]]

    # build the mesh
    cubes = []
    for (x,y,z) in coords:
        box = trimesh.creation.box(extents=(cube_size,)*3)
        box.apply_translation((x*cube_size, y*cube_size, z*cube_size))
        cubes.append(box)
    mesh = trimesh.util.concatenate(cubes)
    mesh.apply_translation(-mesh.centroid)

    shape_state = {
        'seg_lens': seg_lens,
        'dir_indices': dir_indices
    }
    
    return mesh, shape_state

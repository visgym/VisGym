import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYGLET_HEADLESS'] = 'True'

import numpy as np
from typing import Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import trimesh
import ast
import subprocess
from trimesh.transformations import euler_from_matrix, euler_matrix, rotation_from_matrix, rotation_matrix
from .renderer import FastRenderer

# check number of gpus and randomly select one
try:
    # Use nvidia-smi to get GPU count
    result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                          capture_output=True, text=True, check=True)
    num_gpus = len(result.stdout.strip().split('\n'))
except (subprocess.CalledProcessError, FileNotFoundError):
    # Fallback: assume 1 GPU if nvidia-smi is not available
    num_gpus = 1

best_gpu_local = np.random.randint(0, num_gpus) if num_gpus > 0 else 0

# Get the actual GPU ID from CUDA_VISIBLE_DEVICES if set
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if cuda_visible_devices:
    visible_gpus = [int(x.strip()) for x in cuda_visible_devices.split(',')]
    best_gpu_global = visible_gpus[best_gpu_local]
    print(f"Using GPU {best_gpu_global} (local index {best_gpu_local} from CUDA_VISIBLE_DEVICES={cuda_visible_devices})")
    os.environ['EGL_DEVICE_ID'] = str(best_gpu_global)
else:
    print(f"Using GPU {best_gpu_local}")
    os.environ['EGL_DEVICE_ID'] = str(best_gpu_local)

class MentalRotation3DObjaverseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        render_size: int = 128,
        tolerance: float = 15.0, # This is in degrees
        seed: int | None = None,
        sample_dir: str = "./partial_datasets/objaverse/",
        downsample_faces: int = 0,
        use_textures: bool = True,
        action_frame: str = "object",
        max_output_long_side: int = 336):
        super().__init__()
        self.render_size = render_size
        # REFACTOR: Convert tolerance to radians for internal calculations
        self.angle_tol_rad = np.deg2rad(tolerance)
        self.downsample_faces = downsample_faces
        self._rng = np.random.RandomState(seed)
        self.use_textures = bool(use_textures)
        self.max_output_long_side = int(max_output_long_side)
        
        # Track used objects to ensure uniqueness across episodes
        self._used_object_indices = set()

        # Validate and store action frame ("object" intrinsic/post-multiply, or "world" extrinsic/pre-multiply)
        if action_frame not in ("object", "world"):
            raise ValueError("action_frame must be 'object' or 'world'")
        self.action_frame = action_frame

        # Auto face budget (this logic is fine)
        def _clamp(x, lo, hi): 
            return max(lo, min(hi, x))
        self._face_budget = _clamp(int(3 * self.max_output_long_side), 250, 900)
        if downsample_faces and downsample_faces > 0:
            self._face_budget = int(downsample_faces)

        # Action space is fine
        self.action_space = spaces.FuncConditional({
            "rotate": spaces.Box(low=-180.0, high=180.0, shape=(3,), dtype=np.float32),
            "stop": spaces.Text(4)
        })
        H, W = render_size, render_size
        self.observation_space = spaces.Box(0,255,(H,2*W,3),np.uint8)

        # Model path collection is fine
        self.local_obj_root = sample_dir
        self._all_model_paths = []
        for root,_,files in os.walk(self.local_obj_root):
            for fn in files:
                if fn.lower().endswith((".glb",".gltf")):
                    self._all_model_paths.append(os.path.join(root,fn))
        if not self._all_model_paths:
            raise RuntimeError(f"No GLB/GLTF under {self.local_obj_root}")

        # REFACTOR: Use the state representation from Rotation3DEnv
        self.base_scene = None
        self.rotated_scene = None
        self.ref_img = None
        self.current_rot = None # This will be our 4x4 transformation matrix
        self.fast_renderer = FastRenderer(w=render_size, h=render_size)
        
    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # This can be simplified slightly, but is mostly okay.
        prompt = (
            "You are solving a 3D mental rotation task. Two panels are shown:\n"
            "- Left: the target view of the object (identity orientation).\n"
            "- Right: the current view that you can rotate.\n"
            "Your job is to rotate the object on the right so it matches the left."
        )
        prompt += "\n\nAvailable actions:\n"
        prompt += "1. 'rotate': Apply an incremental Euler rotation (yaw, pitch, roll). Format: `('rotate', [d_roll, d_yaw, d_pitch])` with angles in degrees.\n"
        prompt += "2. 'stop': Submit your final orientation. Format: `('stop', 'stop')`"
        # REFACTOR: Use the degree value of tolerance for the prompt
        prompt += f"\n\nSuccess: You succeed if the final rotation error is less than or equal to {np.rad2deg(self.angle_tol_rad):.1f}°."
        # Clarify the rotation frame semantics
        frame_note = (
            "\n\nNote: Rotations are applied in the object's local (intrinsic) axes"
            if self.action_frame == "object" else
            "\n\nNote: Rotations are applied in the world (extrinsic) axes"
        )
        prompt += frame_note
        prompt += "\n\nFor example:\n- Roll by 15°: `('rotate', [15, 0, 0])`\n- To submit: `('stop', 'stop')`"
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        # Extract euler angles from current rotation matrix
        yaw, pitch, roll = trimesh.transformations.euler_from_matrix(self.current_rot, axes='szyx')
        
        return {
            'object_path': self.current_object_path,  # Relative path to the object file
            'secret_yaw': float(yaw),
            'secret_pitch': float(pitch),
            'secret_roll': float(roll),
            'current_rot': self.current_rot.tolist(),
        }
    

    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        if hasattr(self, 'fast_renderer') and self.fast_renderer is not None:
            self.fast_renderer.clear_cache()
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            self.current_object_path = init_state['object_path']
            
            # 1. Load the specific object by path
            try:
                raw_scene = self._load_obj_by_path(self.current_object_path)
            except Exception as exc:
                print(f"Error loading object from path '{self.current_object_path}': {exc}")
                raise
            
            if self._face_budget > 0:
                raw_scene = self._downsample_scene_meshes(raw_scene, self._face_budget)
            centre, radius = self._compute_center_and_radius(raw_scene)
            self.base_scene = raw_scene.copy()
            self._apply_center_and_scale(self.base_scene, centre, radius)
            
            # 2. Pre-render and cache the reference (identity) image
            try:
                self.ref_img = self._render_trimesh_scene_fast(self.base_scene.copy(), self.fast_renderer)
            except Exception as exc:
                print(f"Error rendering reference image: {exc}")
                raise
            
            # 3. Restore the secret rotation from saved state
            yaw = float(init_state['secret_yaw'])
            pitch = float(init_state['secret_pitch'])
            roll = float(init_state['secret_roll'])
            
            self.current_rot = trimesh.transformations.euler_matrix(
                yaw, pitch, roll, axes='szyx'
            )
            # Also allow direct matrix restoration if provided
            if 'current_rot' in init_state:
                self.current_rot = np.array(init_state['current_rot'])
        else:
            # Normal reset: generate new shape and rotation
            # 1. Load and process a new 3D model
            raw_scene, self.current_object_path = self._load_random_obj(seed=seed)
            if self._face_budget > 0:
                raw_scene = self._downsample_scene_meshes(raw_scene, self._face_budget)
            centre, radius = self._compute_center_and_radius(raw_scene)
            self.base_scene = raw_scene.copy()
            self._apply_center_and_scale(self.base_scene, centre, radius)

            # 2. Pre-render and cache the reference (identity) image
            try:
                self.ref_img = self._render_trimesh_scene_fast(self.base_scene.copy(), self.fast_renderer)
            except Exception as exc:
                print(f"Error rendering reference image: {exc}")
                raise

            # 3. Pick a random hidden rotation (as a 4x4 matrix)
            yaw, pitch, roll = self._rng.uniform(0, 360, size=3)
            # Using ZYX order to match the other env and common conventions.
            # trimesh wants degrees for this helper.
            self.current_rot = trimesh.transformations.euler_matrix(
                np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll), axes='szyx'
            )

        # 4. Render the rotated view
        self.rotated_scene = self.base_scene.copy()
        self.rotated_scene.apply_transform(self.current_rot)
        rot_img = self._render_trimesh_scene_fast(self.rotated_scene, self.fast_renderer)

        # 5. Return side-by-side observation
        obs = np.concatenate([self.ref_img, rot_img], axis=1)
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
                self.env_feedback = f"Action payload {payload} is invalid."
            else:
                d_yaw, d_pitch, d_roll = np.deg2rad(self._parse_payload_input(payload))
                R_inc = euler_matrix(d_yaw, d_pitch, d_roll, axes='szyx')
                if self.action_frame == "object":
                    self.current_rot = self.current_rot @ R_inc
                else:
                    self.current_rot = R_inc @ self.current_rot
                self.rotated_scene = self.base_scene.copy()
                self.rotated_scene.apply_transform(self.current_rot)
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

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self._get_obs()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, num_steps: int = 1, uniform: bool = True, strategy: str = "axis-angle") -> list[str]:
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
            return self._solve_by_axis(num_steps, uniform)
        elif strategy == "axis-angle":
            return self._solve_axis_angle(num_steps, uniform)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def close(self):
        """Clean up resources when the environment is closed."""
        if hasattr(self, 'fast_renderer') and self.fast_renderer is not None:
            self.fast_renderer.close()
            self.fast_renderer = None

    def _get_obs(self):
        """Get the current observation of the environment."""
        rot_img = self._render_trimesh_scene_fast(self.rotated_scene, self.fast_renderer)
        return np.concatenate([self.ref_img, rot_img], axis=1)
    
    def _get_info(self):
        return {"env_feedback": self.env_feedback}
    
    def _compute_reward(self):
        """Compute the reward for the current state."""
        angle_err_rad = self._rotation_distance(self.current_rot, np.eye(4))
        reward = 1.0 if angle_err_rad <= self.angle_tol_rad else 0.0
        return reward

########################################################
# End of all methods, start of helper methods
########################################################

    def _graph_to_scene_compat(self, scene: trimesh.Scene) -> trimesh.Scene:
        if hasattr(scene.graph, "to_scene"):
            return scene.graph.to_scene()
        new_geoms = []
        for node_name in scene.graph.nodes_geometry:
            T_node, geom_key = scene.graph.get(node_name)
            geom = scene.geometry[geom_key].copy()
            geom.apply_transform(T_node)
            new_geoms.append(geom)
        return trimesh.Scene(new_geoms)

    def _compute_center_and_radius(self, scene: trimesh.Scene):
        """Compute a robust scene center and radius.
        - Prefer oriented bounding box on a flattened scene;
        - Fallback to axis-aligned bounds across all available vertex data (meshes or point clouds);
        - If still empty, return (0,0,0) and a minimal radius to avoid divide-by-zero.
        """
        flat = self._graph_to_scene_compat(scene)
        # Try oriented bounds first (fast path)
        try:
            centre = flat.bounding_box_oriented.centroid
            # extents can be zero on degenerate; guard minimal radius
            ext = np.asarray(getattr(flat, 'extents', np.zeros(3)))
            r = float(np.linalg.norm(ext) * 0.5)
            if not np.isfinite(r) or r <= 0.0:
                raise ValueError('degenerate oriented bounds')
            return centre, r
        except Exception:
            pass

        # Fallback: gather points from all geometries
        pts_list = []
        try:
            for g in flat.geometry.values():
                try:
                    if isinstance(g, trimesh.Trimesh):
                        v = np.asarray(getattr(g, 'vertices', np.empty((0, 3))))
                        if v.size:
                            pts_list.append(v)
                    elif isinstance(g, trimesh.points.PointCloud):
                        v = np.asarray(getattr(g, 'vertices', np.empty((0, 3))))
                        if v.size:
                            pts_list.append(v)
                except Exception:
                    continue
        except Exception:
            pass

        if not pts_list:
            # Completely empty scene; choose safe defaults
            return np.zeros(3, dtype=float), 1e-3

        P = np.vstack(pts_list)
        mins, maxs = P.min(axis=0), P.max(axis=0)
        centre = (mins + maxs) * 0.5
        r = float(np.linalg.norm(maxs - mins) * 0.5)
        if not np.isfinite(r) or r <= 0.0:
            r = 1e-3
        return centre, r


    def _apply_center_and_scale(self, scene: trimesh.Scene, centre: np.ndarray, scale: float):
        scene.apply_translation(-centre)
        if scale > 0:
            scene.apply_scale(1.0 / scale)
        return scene

    def _parse_payload_input(self, payload):
        new_payload = np.ones_like(payload)
        new_payload[0] = - payload[2]
        new_payload[1] = payload[0]
        new_payload[2] = payload[1]
        return new_payload

    def _parse_payload_output(self, payload):
        new_payload = np.ones_like(payload)
        new_payload[0] = payload[1]
        new_payload[1] = payload[2]
        new_payload[2] = - payload[0]
        return new_payload


    def _downsample_scene_meshes(self, scene: trimesh.Scene, target_faces=1000):
        geoms = list(scene.geometry.items())
        total_faces = sum(len(g.faces) for _, g in geoms if hasattr(g, 'faces') and len(getattr(g, 'faces', [])) > 0)
        if total_faces == 0 or total_faces <= target_faces:
            return scene

        scale = max(0.0, float(target_faces) / float(total_faces))
        new_geoms = {}
        for name, g in geoms:
            if hasattr(g, 'faces') and len(getattr(g, 'faces', [])) > 0:
                original = int(len(g.faces))
                # proportional target, ensure it's strictly less than original and valid
                tgt = int(max(4, min(original - 1, int(np.ceil(original * scale)))))
                if tgt >= original or original <= 8:
                    new_geoms[name] = g
                else:
                    try:
                        simp = g.simplify_quadric_decimation(face_count=tgt)
                        if not hasattr(simp, 'faces') or len(getattr(simp, 'faces', [])) == 0:
                            # fallback if simplification produced empty/invalid mesh
                            new_geoms[name] = g
                        else:
                            new_geoms[name] = simp
                    except Exception:
                        new_geoms[name] = g
            else:
                new_geoms[name] = g

        s = trimesh.Scene(new_geoms)
        try:
            s.graph = scene.graph.copy()
        except Exception:
            pass
        return s

    def _render_trimesh_scene_fast(self, scene, fast_renderer):
        return fast_renderer.render(scene)

    def _euler_internal_to_payload(self, yaw_rad: float, pitch_rad: float, roll_rad: float) -> list[float]:
        """
        Convert internal Euler angles (yaw, pitch, roll) in radians, defined for axes='szyx',
        to the external action payload expected by step(), i.e. the inverse of parse_payload_input.

        Internal -> External mapping is given by parse_payload_output.
        """
        internal_deg = np.array([
            np.rad2deg(yaw_rad),
            np.rad2deg(pitch_rad),
            np.rad2deg(roll_rad)
        ], dtype=np.float32)
        external_deg = self._parse_payload_output(internal_deg)
        return [round(float(x), 1) for x in external_deg.tolist()]

    def _remove_negative_zero(self, values: list[float]) -> list[float]:
        """Ensure any -0.0 values become +0.0 for clean printing/serialization."""
        cleaned = []
        for v in values:
            cleaned.append(0.0 if v == 0.0 else float(v))
        return cleaned

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
        # Need to convert internal (yaw, pitch, roll) to external payload format
        
        # Step 1: Front -> Right (yaw +90)
        step1 = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
        
        # Step 2: Right -> Back (yaw +90)
        step2 = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
        
        # Step 3: Back -> Left (yaw +90)
        step3 = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
        
        # Step 4: Left -> Top (pitch +90)
        step4 = self._parse_payload_output(np.array([0.0, 90.0, 0.0], dtype=np.float32))
        
        # Step 5: Top -> Bottom (pitch +180)
        step5 = self._parse_payload_output(np.array([0.0, 180.0, 0.0], dtype=np.float32))
        
        # Step 6: Bottom -> Left (pitch +90)
        step6 = self._parse_payload_output(np.array([0.0, 90.0, 0.0], dtype=np.float32))
        
        # Step 7: Left -> Front (yaw +90)
        step7 = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
        
        padding_actions = [
            str(('rotate', [round(float(x), 1) for x in step1.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step2.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step3.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step4.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step5.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step6.tolist()])),
            str(('rotate', [round(float(x), 1) for x in step7.tolist()])),
        ]
        
        # Get the actual solution actions
        if base_strategy == "axis-angle":
            solution_actions = self._solve_axis_angle(num_steps, uniform)
        elif base_strategy == "by-axis":
            solution_actions = self._solve_by_axis(num_steps, uniform)
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
            # Convert internal (yaw, pitch, roll) to external payload format
            padding_yaw = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
            padding_pitch = self._parse_payload_output(np.array([0.0, 90.0, 0.0], dtype=np.float32))
            padding_roll = self._parse_payload_output(np.array([0.0, 0.0, 90.0], dtype=np.float32))
            
            padding_actions = (
                [str(('rotate', [round(float(x), 1) for x in padding_yaw.tolist()]))] * 4 +
                [str(('rotate', [round(float(x), 1) for x in padding_pitch.tolist()]))] * 4 +
                [str(('rotate', [round(float(x), 1) for x in padding_roll.tolist()]))] * 4
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
            
            yaw, pitch, roll = euler_from_matrix(total_rotation_matrix, axes='szyx')
            
            actions = []
            
            # Order depends on action frame semantics
            if self.action_frame == "object":
                # Intrinsic/body: apply roll -> pitch -> yaw via post-multiply
                if not np.isclose(roll, 0):
                    # Add 4 padding steps for roll axis
                    padding_roll = self._parse_payload_output(np.array([0.0, 0.0, 90.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_roll.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, 0.0, roll))
                    actions.append(str(('rotate', payload)))
                    
                if not np.isclose(pitch, 0):
                    # Add 4 padding steps for pitch axis
                    padding_pitch = self._parse_payload_output(np.array([0.0, 90.0, 0.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_pitch.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, pitch, 0.0))
                    actions.append(str(('rotate', payload)))
                    
                if not np.isclose(yaw, 0):
                    # Add 4 padding steps for yaw axis
                    padding_yaw = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_yaw.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(yaw, 0.0, 0.0))
                    actions.append(str(('rotate', payload)))
            else:
                # Extrinsic/world: apply yaw -> pitch -> roll via pre-multiply
                if not np.isclose(yaw, 0):
                    # Add 4 padding steps for yaw axis
                    padding_yaw = self._parse_payload_output(np.array([90.0, 0.0, 0.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_yaw.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(yaw, 0.0, 0.0))
                    actions.append(str(('rotate', payload)))
                    
                if not np.isclose(pitch, 0):
                    # Add 4 padding steps for pitch axis
                    padding_pitch = self._parse_payload_output(np.array([0.0, 90.0, 0.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_pitch.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, pitch, 0.0))
                    actions.append(str(('rotate', payload)))
                    
                if not np.isclose(roll, 0):
                    # Add 4 padding steps for roll axis
                    padding_roll = self._parse_payload_output(np.array([0.0, 0.0, 90.0], dtype=np.float32))
                    actions.extend([str(('rotate', [round(float(x), 1) for x in padding_roll.tolist()]))] * 4)
                    # Add actual rotation
                    payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, 0.0, roll))
                    actions.append(str(('rotate', payload)))

            if not actions:
                # No rotation needed
                return ["('stop', 'stop')"]

            actions.append("('stop', 'stop')")
            return actions
        else:
            raise ValueError(f"Unknown base strategy for pad3: {base_strategy}")

    def _solve_by_axis(self, num_steps: int = 1, uniform: bool = True) -> list[str]:
        """
        Solves the puzzle by rotating along each axis (yaw, pitch, roll) separately.
        """
        total_rotation_matrix = np.linalg.inv(self.current_rot)
        
        yaw, pitch, roll = euler_from_matrix(total_rotation_matrix, axes='szyx')
        
        actions = []
        # Order depends on action frame semantics
        if self.action_frame == "object":
            # Intrinsic/body: apply roll -> pitch -> yaw via post-multiply
            if not np.isclose(roll, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, 0.0, roll))
                actions.append(str(('rotate', payload)))
            if not np.isclose(pitch, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, pitch, 0.0))
                actions.append(str(('rotate', payload)))
            if not np.isclose(yaw, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(yaw, 0.0, 0.0))
                actions.append(str(('rotate', payload)))
        else:
            # Extrinsic/world: apply yaw -> pitch -> roll via pre-multiply
            if not np.isclose(yaw, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(yaw, 0.0, 0.0))
                actions.append(str(('rotate', payload)))
            if not np.isclose(pitch, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, pitch, 0.0))
                actions.append(str(('rotate', payload)))
            if not np.isclose(roll, 0):
                payload = self._remove_negative_zero(self._euler_internal_to_payload(0.0, 0.0, roll))
                actions.append(str(('rotate', payload)))

        if not actions:
            # No rotation needed
            return ["('stop', 'stop')"]

        actions.append("('stop', 'stop')")
        return actions

    def _solve_axis_angle(self, num_steps: int = 1, uniform: bool = True) -> list[str]:
        """
        Original implementation using axis-angle decomposition.
        """
        if num_steps <= 0:
            num_steps = 1

        # The total rotation needed is the inverse of the current rotation
        total_rotation_needed = np.linalg.inv(self.current_rot)

        try:
            angle, direction, _ = rotation_from_matrix(total_rotation_needed)
        except np.linalg.LinAlgError:
            return ["('stop', 'stop')"]

        if np.isclose(angle, 0):
            return ["('stop', 'stop')"]

        # Subdivide the rotation angle into `num_steps` parts.
        if num_steps > 1:
            if uniform:
                # Uniformly divide the rotation angle
                step_angles = [angle / num_steps] * num_steps
            else:
                # Generate `num_steps - 1` random split points
                split_points = np.sort(self._rng.uniform(0, abs(angle), num_steps - 1))
                # Create the sequence of angles
                step_angles = np.diff(np.concatenate(([0], split_points, [angle])))
        else:
            step_angles = [angle]

        # Create the sequence of actions; simulate frame semantics to be exact
        actions = []
        R_sim = self.current_rot.copy()
        for step_angle in step_angles:
            # Desired world-frame incremental rotation for this step
            M_step_world = rotation_matrix(step_angle, direction)

            if self.action_frame == "object":
                # Convert world increment to an equivalent local increment via conjugation
                M_step_local = np.linalg.inv(R_sim) @ M_step_world @ R_sim
                yaw, pitch, roll = euler_from_matrix(M_step_local, axes='szyx')
                payload = self._euler_internal_to_payload(yaw, pitch, roll)
                actions.append(str(('rotate', payload)))
                # Update simulated pose as env would (post-multiply by local increment)
                R_sim = R_sim @ M_step_local
            else:
                # World-frame: use world increment directly
                yaw, pitch, roll = euler_from_matrix(M_step_world, axes='szyx')
                payload = self._euler_internal_to_payload(yaw, pitch, roll)
                actions.append(str(('rotate', payload)))
                # Update simulated pose as env would (pre-multiply by world increment)
                R_sim = M_step_world @ R_sim

        actions.append("('stop', 'stop')")
        return actions

    def _load_random_obj(self, seed=None) -> tuple[trimesh.Scene, str]:
        """
        Load a random 3D object from the dataset.
        Returns: (scene, relative_path) where relative_path is relative to local_obj_root
        """
        # Get available (unused) object indices
        available_indices = [i for i in range(len(self._all_model_paths)) if i not in self._used_object_indices]
        
        if not available_indices:
            raise RuntimeError("All objects have been used. Cannot create more episodes.")
        
        # Choose a random index from available objects
        if seed is not None:
            rng = np.random.RandomState(seed)
            chosen_idx = rng.choice(available_indices)
        else:
            chosen_idx = self._rng.choice(available_indices)
        
        # Mark this object as used
        self._used_object_indices.add(chosen_idx)
        
        # Load the selected object
        p = self._all_model_paths[chosen_idx]
        # Store the relative path (from local_obj_root) for reproducibility
        relative_path = os.path.relpath(p, self.local_obj_root)
        scene = trimesh.load(p, force='scene', skip_materials=not self.use_textures, process=True)
        return scene, relative_path
    
    def _load_obj_by_path(self, relative_path: str) -> trimesh.Scene:
        """
        Load a 3D object by its relative path (relative to local_obj_root).
        """
        full_path = os.path.join(self.local_obj_root, relative_path)
        if not os.path.exists(full_path):
            raise ValueError(f"Object file '{relative_path}' not found in {self.local_obj_root}")
        
        return trimesh.load(full_path, force='scene', skip_materials=not self.use_textures, process=True)

    def _rotation_distance(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """ Returns the geodesic angle (in radians) between two 3×3 rotation matrices. """
        R_rel = R1[:3, :3] @ R2[:3, :3].T
        trace = np.clip(np.trace(R_rel), -1.0, 3.0)
        theta = np.arccos((trace - 1.0) / 2.0)
        return theta
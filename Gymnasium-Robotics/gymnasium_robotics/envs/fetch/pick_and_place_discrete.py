import os
import ast
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.spaces import FuncConditional, Text

from gymnasium_robotics.envs.fetch.pick_and_place import (
    MujocoFetchPickAndPlaceEnv,
    MujocoPyFetchPickAndPlaceEnv,
)
# from gymnasium_robotics.envs.vlm_mixin import VLMEnvMixin
# from gym import VLMEnvMixin
import gymnasium as gym


MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")


class _DiscreteMixin(gym.VLMEnvMixin):
    """Mixin providing FuncConditional action mapping and pick-and-place oracle.

    Action space:
    - "move": 3-element array [x, y, z] where each element is -1, 0, or 1
    - "gripper": 0 (open) or 1 (close) 
    - "stop": terminate episode
    """

    # Underlying BaseFetchEnv scales position deltas by 0.05 per step
    _STEP_SIZE: float = 0.05

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Set image dimensions before parent initialization
        self._image_height = 480
        self._image_width = 480
        # Dual camera setup: wide and wrist/front camera names
        self._primary_camera_name = kwargs.pop("primary_camera_name", "external_camera_0")
        self._secondary_camera_name = kwargs.pop("secondary_camera_name", "gripper_camera_rgb")
        # Gripper-follow camera config
        self._use_follow_camera = kwargs.pop("use_follow_camera", True)
        self._follow_cam_distance = float(kwargs.pop("follow_cam_distance", 0.38))
        self._follow_cam_azimuth = float(kwargs.pop("follow_cam_azimuth", 180.0))
        self._follow_cam_elevation = float(kwargs.pop("follow_cam_elevation", -60.0))
        self._follow_cam_lookat_offset = np.array(
            kwargs.pop("follow_cam_lookat_offset", [0.0, 0.0, -0.01]), dtype=float
        )
        # Camera jitter configuration (applied on reset)
        self._jitter_camera_on_reset = kwargs.pop("jitter_camera_on_reset", True)
        self._camera_base_lookat = np.array(kwargs.pop("camera_base_lookat", [1.3, 0.75, 0.55]), dtype=float)
        self._camera_jitter_azimuth = float(kwargs.pop("camera_jitter_azimuth", 10.0))  # degrees
        self._camera_jitter_elevation = float(kwargs.pop("camera_jitter_elevation", 10.0))  # degrees
        self._camera_jitter_distance = float(kwargs.pop("camera_jitter_distance", 0.1))  # meters
        self._camera_jitter_lookat = float(kwargs.pop("camera_jitter_lookat", 0.2))  # meters
        # Fallback defaults used if renderer has no baseline camera state
        self._camera_default_azimuth = float(kwargs.pop("camera_default_azimuth", 132.0))
        self._camera_default_elevation = float(kwargs.pop("camera_default_elevation", -14.0))
        # Slightly zoom in default external view
        self._camera_default_distance = float(kwargs.pop("camera_default_distance", 2))
        
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        
        # FuncConditional action space like mental rotation 3D
        self.action_space = FuncConditional({
            "move": spaces.Box(
                low=np.array([-1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1], dtype=np.float32),
                dtype=np.float32,
            ),
            "gripper": spaces.Discrete(2),  # 0=open, 1=close
            "stop": Text(4)       # "stop"
        })
        
        # Add image to observation space
        original_spaces = self.observation_space.spaces
        new_spaces = original_spaces.copy()
        concat_factor = 2 if self._use_follow_camera else 1
        self._image_concat_factor = concat_factor
        new_spaces['image'] = spaces.Box(
            low=0, high=255,
            shape=(self._image_height, self._image_width * concat_factor, 3),
            dtype=np.uint8
        )
        self.observation_space = spaces.Dict(new_spaces)
        
        # Oracle state variables
        self._oracle_stage: int = 0
        self._prev_finger_opening: Optional[float] = None
        self._open_action_candidate: int = 7  # assume 7 opens until detected
        self._close_action_candidate: int = 8
        self._open_detected: bool = False
        self._last_action: int = 0
        # Heuristics
        self._xy_align_tol: float = 0.03
        self._z_contact_margin: float = 0.005
        # Stage-3 closing stabilization
        self._close_hold_steps: int = 3
        self._stage3_wait_count: int = 0
        self._stabilize_eps: float = 1e-4
        self._stabilize_required: int = 2
        self._stable_count: int = 0
        # Sticky gripper bias so moves keep squeezing/opening
        self._gripper_bias: float = 0.0

    def _map_action_to_continuous(self, action) -> np.ndarray:
        """Convert FuncConditional action to continuous action for MuJoCo.
        
        Args:
            action: Either a tuple ("move", [x,y,z]) or ("gripper", "open/close") or ("stop", "stop")
            
        Returns:
            np.ndarray: 4-element continuous action [dx, dy, dz, gripper]
        """
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            # Default noop action
            return np.array([0.0, 0.0, 0.0, self._gripper_bias], dtype=np.float32)
        
        action_type, payload = action
        
        if action_type == "move":
            # payload should be [x, y, z] with values -1, 0, or 1
            if isinstance(payload, (list, tuple, np.ndarray)) and len(payload) == 3:
                dx, dy, dz = float(payload[0]), float(payload[1]), float(payload[2])
                return np.array([dx, dy, dz, self._gripper_bias], dtype=np.float32)
        
        elif action_type == "gripper":
            # payload should be 0 (open) or 1 (close)
            if payload == 0:
                self._gripper_bias = 1.0
            elif payload == 1:
                self._gripper_bias = -1.0
            return np.array([0.0, 0.0, 0.0, self._gripper_bias], dtype=np.float32)
        
        elif action_type == "stop":
            # Stop action - return noop
            return np.array([0.0, 0.0, 0.0, self._gripper_bias], dtype=np.float32)
        
        # Default noop
        return np.array([0.0, 0.0, 0.0, self._gripper_bias], dtype=np.float32)

    def get_init_state(self) -> dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        # Get object position from joint state
        if hasattr(self, 'data') and self.has_object:
            object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
            object_pos = object_qpos[:3].copy().tolist()  # x, y, z position
        else:
            object_pos = None
        
        # Get goal position
        goal_pos = self.goal.copy().tolist() if hasattr(self, 'goal') else None
        
        # Get camera configuration
        camera_config = None
        if hasattr(self, 'mujoco_renderer') and self.mujoco_renderer is not None:
            cam_cfg = getattr(self.mujoco_renderer, 'default_cam_config', None)
            if cam_cfg is not None:
                camera_config = {
                    'azimuth': float(cam_cfg.get('azimuth', self._camera_default_azimuth)),
                    'elevation': float(cam_cfg.get('elevation', self._camera_default_elevation)),
                    'distance': float(cam_cfg.get('distance', self._camera_default_distance)),
                    'lookat': cam_cfg.get('lookat', self._camera_base_lookat).tolist() if isinstance(cam_cfg.get('lookat'), np.ndarray) else list(cam_cfg.get('lookat', self._camera_base_lookat)),
                }
        
        return {
            'object_pos': object_pos,
            'goal_pos': goal_pos,
            'camera_config': camera_config,
        }
    
    def reset(self, seed=None, options=None, init_state=None):  # type: ignore[override]
        # Set the RNG seed for reproducible camera jitter and object positioning
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        obs, info = super().reset(seed=seed, options=options)
        self._oracle_stage = 0
        self._prev_finger_opening = None
        self._open_action_candidate = ("gripper", 0)
        self._close_action_candidate = ("gripper", 1)
        self._open_detected = False
        self._last_action = ("move", [0, 0, 0])
        self._stage3_wait_count = 0
        self._stable_count = 0
        self._gripper_bias = 0.0
        
        if init_state is not None:
            # Restore state from init_state
            # Restore object position
            if init_state.get('object_pos') is not None and self.has_object:
                object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
                object_qpos[:3] = np.array(init_state['object_pos'], dtype=np.float64)
                self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)
                self._mujoco.mj_forward(self.model, self.data)
            
            # Restore goal position
            if init_state.get('goal_pos') is not None:
                self.goal = np.array(init_state['goal_pos'], dtype=np.float64)
            
            # Restore camera configuration
            if init_state.get('camera_config') is not None:
                cam_cfg = init_state['camera_config']
                self._set_camera_config({
                    'azimuth': cam_cfg['azimuth'],
                    'elevation': cam_cfg['elevation'],
                    'distance': cam_cfg['distance'],
                    'lookat': np.array(cam_cfg['lookat'], dtype=np.float64),
                })
            
            # Get updated observation after restoring state
            obs = self._get_obs()
        else:
            # Apply a small random camera jitter at each episode reset for visual diversity
            self._apply_camera_jitter()
        
        return obs, info

    def _get_obs(self) -> dict:
        """Get observation including rendered image."""
        # Get original observation
        obs = super()._get_obs()
        
        # Add rendered image - only if goal is properly initialized
        if hasattr(self, 'goal') and self.goal.shape == (3,):
            # Randomize camera view for more diverse observations
            # self._randomize_camera_view()
            self._render_callback()
            if self._use_follow_camera:
                img = self._render_dual_camera()
            else:
                img = self._render_primary_camera()
            # Resize if needed
            target_w = self._image_width * (2 if self._use_follow_camera else 1)
            if img.shape[:2] != (self._image_height, target_w):
                import cv2
                img = cv2.resize(img, (target_w, self._image_height))
            obs['image'] = img
        else:
            # Goal not initialized yet, create black image
            obs['image'] = np.zeros((self._image_height, self._image_width, 3), dtype=np.uint8)
        
        return obs

    def _randomize_camera_view(self):
        """Randomize camera viewpoint for more diverse observations."""
        if not self._randomize_camera:
            return
        
        if hasattr(self, 'mujoco_renderer') and self.mujoco_renderer is not None:
            # Randomize camera parameters
            azimuth = self.np_random.uniform(*self._camera_azimuth_range)
            elevation = self.np_random.uniform(*self._camera_elevation_range)
            distance = self.np_random.uniform(*self._camera_distance_range)
            
            # Update camera configuration
            self._set_camera_config({
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'lookat': np.array([1.3, 0.75, 0.55])  # Keep lookat fixed at table center
            })

    def _apply_camera_jitter(self) -> None:
        """Apply small camera pose jitter using the env RNG.

        Jitters azimuth, elevation, distance, and lookat around sensible defaults.
        """
        if not getattr(self, "_jitter_camera_on_reset", True):
            return
        if not hasattr(self, "mujoco_renderer") or self.mujoco_renderer is None:
            return

        # Sample small, centered offsets
        azimuth = float(
            self._camera_default_azimuth
            + self.np_random.uniform(-self._camera_jitter_azimuth, self._camera_jitter_azimuth)
        )
        elevation = float(
            self._camera_default_elevation
            + self.np_random.uniform(-self._camera_jitter_elevation, self._camera_jitter_elevation)
        )
        distance = float(
            self._camera_default_distance
            + self.np_random.uniform(-self._camera_jitter_distance, self._camera_jitter_distance)
        )
        lookat_noise = self.np_random.uniform(-self._camera_jitter_lookat, self._camera_jitter_lookat, size=3)
        lookat = (self._camera_base_lookat + lookat_noise).astype(float)

        self._set_camera_config({
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance,
            'lookat': lookat,
        })

    def _set_camera_config(self, cam_cfg: dict) -> None:
        """Set camera config via MujocoRenderer API and live viewer if present."""
        if not hasattr(self, "mujoco_renderer") or self.mujoco_renderer is None:
            return
        # Ensure defaults applied to any future viewer
        self.mujoco_renderer.default_cam_config = cam_cfg
        # Use free camera so azimuth/elevation/distance/lookat take effect
        try:
            self.mujoco_renderer.camera_id = -1
        except Exception:
            pass
        # If a viewer already exists, set attributes directly
        viewer = getattr(self.mujoco_renderer, "viewer", None)
        if viewer is not None:
            # Ensure free camera mode
            try:
                viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE  # type: ignore[attr-defined]
                viewer.cam.fixedcamid = -1
            except Exception:
                pass
            for key, value in cam_cfg.items():
                if hasattr(viewer.cam, key):
                    if isinstance(value, np.ndarray):
                        getattr(viewer.cam, key)[:] = value
                    else:
                        setattr(viewer.cam, key, value)

    def _render_dual_camera(self) -> np.ndarray:
        """Render and horizontally concatenate primary (free) and wrist/follow cameras."""
        import numpy as _np
        prev_cam_id = getattr(self.mujoco_renderer, "camera_id", None)
        # Primary image: same as _render_primary_camera (free camera)
        # Re-apply free camera config to avoid lingering tracking settings
        cam_cfg = getattr(self.mujoco_renderer, "default_cam_config", None)
        if cam_cfg is not None:
            self._set_camera_config(cam_cfg)
        self.mujoco_renderer.camera_id = -1
        img1 = self.mujoco_renderer.render(render_mode="rgb_array")
        # Secondary: wrist/follow camera
        img2 = self._render_follow_camera() if self._use_follow_camera else None
        if img2 is None:
            for cand_name in [self._secondary_camera_name, "head_camera_rgb", "external_camera_0"]:
                cand_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_CAMERA, cand_name)
                if int(cand_id) < 0:
                    continue
                self.mujoco_renderer.camera_id = int(cand_id)
                img2 = self.mujoco_renderer.render(render_mode="rgb_array")
                if img2 is not None:
                    break
        # Restore previous camera
        if prev_cam_id is not None:
            self.mujoco_renderer.camera_id = prev_cam_id
        # Resize and hstack
        if img1.shape[:2] != (self._image_height, self._image_width):
            import cv2
            img1 = cv2.resize(img1, (self._image_width, self._image_height))
        if img2 is None:
            img2 = _np.zeros_like(img1)
        elif img2.shape[:2] != (self._image_height, self._image_width):
            import cv2
            img2 = cv2.resize(img2, (self._image_width, self._image_height))
        return _np.hstack([img1, img2])

    def _render_follow_camera(self) -> np.ndarray:
        """Render a camera that follows the gripper position without rotating.

        Uses MuJoCo tracking camera mode with `trackbodyid` pointing to
        `robot0:gripper_link`. Azimuth/elevation are kept in world frame, so
        the view does not spin with the wrist.
        """
        # Ensure viewer exists and use non-fixed camera id
        prev_cam_id = getattr(self.mujoco_renderer, "camera_id", None)
        self.mujoco_renderer.camera_id = -1
        viewer = self.mujoco_renderer._get_viewer(render_mode="rgb_array")
        try:
            body_id = int(self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link"
            ))
            # World-space target position for lookat with small offset
            grip = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
            lookat = grip + self._follow_cam_lookat_offset
            # Configure tracking camera
            viewer.cam.type = self._mujoco.mjtCamera.mjCAMERA_TRACKING  # type: ignore[attr-defined]
            viewer.cam.trackbodyid = body_id
            viewer.cam.fixedcamid = -1
            for i in range(3):
                viewer.cam.lookat[i] = float(lookat[i])
            viewer.cam.distance = float(self._follow_cam_distance)
            viewer.cam.azimuth = float(self._follow_cam_azimuth)
            viewer.cam.elevation = float(self._follow_cam_elevation)
        except Exception:
            pass
        img = self.mujoco_renderer.render(render_mode="rgb_array")
        if prev_cam_id is not None:
            self.mujoco_renderer.camera_id = prev_cam_id
        return img

    def _render_primary_camera(self) -> np.ndarray:
        """Render the old default free camera view (not front fixed camera)."""
        prev_cam_id = getattr(self.mujoco_renderer, "camera_id", None)
        # Ensure free camera config is applied (reset any tracking from wrist view)
        cam_cfg = getattr(self.mujoco_renderer, "default_cam_config", None)
        if cam_cfg is not None:
            self._set_camera_config(cam_cfg)
        # Use free camera; defaults/jitter are applied on reset via _apply_camera_jitter
        self.mujoco_renderer.camera_id = -1
        img = self.mujoco_renderer.render(render_mode="rgb_array")
        if prev_cam_id is not None:
            self.mujoco_renderer.camera_id = prev_cam_id
        return img

    def _get_info(self) -> dict:
        """Get additional information about the environment state."""
        obs = self._get_obs()
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "distance_to_goal": np.linalg.norm(obs["achieved_goal"] - self.goal),
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": self.goal,
        }
        return info

    def render(self):
        image = self._get_obs()["image"]
        return image

    def step(self, action):  # type: ignore[override]
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
                # Convert FuncConditional action to continuous action
                cont_action = self._map_action_to_continuous((branch, payload))
                self._set_action(cont_action)
                self._mujoco_step(cont_action)
                self._step_callback()
                self.env_feedback = None

        elif branch == "gripper":
            if not self.action_space["gripper"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                # Convert FuncConditional action to continuous action
                cont_action = self._map_action_to_continuous((branch, payload))
                self._set_action(cont_action)
                self._mujoco_step(cont_action)
                self._step_callback()
                self.env_feedback = None

        elif branch == "stop":
            if not self.action_space["stop"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                terminated = True
                truncated = False
        else:
            self.env_feedback = f"The action name '{branch}' is not recognized in the available action space."

        if self.render_mode == "human":
            self.render()

        # 4. Compute reward ------------------------------------------------------
        obs = self._get_obs()
        info = self._get_info()
        info["is_success"] = self._is_success(obs["achieved_goal"], self.goal)
        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info) or terminated
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        # Simple binary reward: 1 for success, 0 otherwise
        if terminated:
            reward = 1.0 if info["is_success"] else 0.0

        # 5. Return ------------------------------------------------------
        if self.env_feedback is None:
            self.env_feedback = "Action executed successfully."
        
        info["env_feedback"] = self.env_feedback
        return obs, reward, terminated, truncated, info

    def _axis_greedy_move(self, current: np.ndarray, target: np.ndarray, thresh: float) -> tuple:
        # Per-axis greedy move similar to reach oracle, tolerant to 5 cm step size
        err = target - current
        # Use a tolerance at least half the discrete step to avoid oscillation
        tol = max(thresh, self._STEP_SIZE * 0.6)
        if np.all(np.abs(err) <= tol):
            return ("move", [0, 0, 0])
        axis = int(np.argmax(np.abs(err)))
        sign = 1 if err[axis] >= 0 else -1
        
        move_array = [0, 0, 0]
        move_array[axis] = sign
        return ("move", move_array)

    def oracle_action(
        self,
        approach_z_margin: float = 0.10,
        place_z_margin: float = 0.10,
        pos_threshold: float = 0.03,
        open_thresh: float = 0.08,
        closed_thresh: float = 0.01,
    ) -> tuple:
        """Oracle for pick and place with a simple staged state machine.

        Stages:
          0) Move above object (xy align at safe approach height)
          1) Open gripper
          2) Descend to object top and center
          3) Close gripper (grasp)
          4) Lift to carry height
          5) Move above goal (xy align at carry height)
          6) Descend to place height
          7) Open gripper (release)

        Args:
            approach_z_margin: Height above object for approach.
            place_z_margin: Height above goal for transit/carry.
            pos_threshold: Axis-wise tolerance to consider at target.
            open_thresh: Gripper finger position sum to consider open.
            closed_thresh: Gripper finger position sum to consider closed.

        Returns:
            Action tuple ("move", [x,y,z]) or ("gripper", 0/1) or ("stop", "stop").
        """
        obs = self._get_obs()
        # Parse positions
        grip_pos = obs["observation"][0:3]
        obj_pos = obs["observation"][3:6]
        achieved_obj = obs["achieved_goal"]  # alias
        goal_pos = obs["desired_goal"]
        gripper_fingers = obs["observation"][9:11]
        finger_opening = float(np.sum(gripper_fingers))

        # Define safe/carry heights
        safe_approach_z = float(max(obj_pos[2] + approach_z_margin, grip_pos[2]))
        carry_z = float(max(goal_pos[2], obj_pos[2]) + place_z_margin)

        # Stage transitions and actions
        if self._oracle_stage == 0:
            # Move above object
            target = np.array([obj_pos[0], obj_pos[1], safe_approach_z], dtype=float)
            act = self._axis_greedy_move(grip_pos, target, pos_threshold)
            if act[0] == "move" and all(x == 0 for x in act[1]):
                self._oracle_stage = 1
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        if self._oracle_stage == 1:
            # Ensure gripper open
            if finger_opening >= open_thresh:
                self._oracle_stage = 2
                act = ("move", [0, 0, 0])
            else:
                # Auto-detect which action opens the gripper by observing change
                if self._prev_finger_opening is not None and isinstance(self._last_action, tuple) and self._last_action[0] == "gripper":
                    delta = finger_opening - self._prev_finger_opening
                    eps = 1e-4
                    if delta > eps:
                        # Last action increased opening -> it's the open action
                        self._open_action_candidate = self._last_action
                        self._close_action_candidate = ("gripper", 1)
                        self._open_detected = True
                    elif delta < -eps:
                        # Last action decreased opening -> opposite is open
                        self._open_action_candidate = ("gripper", 0)
                        self._close_action_candidate = self._last_action
                        self._open_detected = True
                act = ("gripper", 0)
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        if self._oracle_stage == 2:
            # Align XY tightly, then force descend to near-contact regardless of step tol
            xy_err = obj_pos[:2] - grip_pos[:2]
            if abs(xy_err[0]) > self._xy_align_tol:
                act = ("move", [1 if xy_err[0] > 0 else -1, 0, 0])
            elif abs(xy_err[1]) > self._xy_align_tol:
                act = ("move", [0, 1 if xy_err[1] > 0 else -1, 0])
            elif grip_pos[2] > obj_pos[2] + self._z_contact_margin:
                act = ("move", [0, 0, -1])  # descend
            else:
                self._oracle_stage = 3
                act = ("move", [0, 0, 0])
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        if self._oracle_stage == 3:
            # Close to grasp with stabilization: keep sending close for a few steps
            # and/or until finger opening stabilizes.
            if self._stage3_wait_count == 0:
                self._stable_count = 0
            # Detect stabilization of finger opening
            if self._prev_finger_opening is not None:
                if abs(finger_opening - self._prev_finger_opening) < self._stabilize_eps:
                    self._stable_count += 1
                else:
                    self._stable_count = 0
            act = ("gripper", 1)
            self._stage3_wait_count += 1
            # Transition when enough hold steps or stabilized
            if (
                self._stage3_wait_count >= self._close_hold_steps
                or self._stable_count >= self._stabilize_required
            ):
                self._oracle_stage = 4
                # reset counters for next time
                self._stage3_wait_count = 0
                self._stable_count = 0
                # keep act as close this step to ensure grasp
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        if self._oracle_stage == 4:
            # After grasp: directly move object toward goal in 3D while keeping gripper closed.
            # Use object error to pick dominant axis movement each step.
            obj_err = goal_pos - achieved_obj
            # If already basically at goal, transition to release
            if np.all(np.abs(obj_err) <= pos_threshold):
                self._oracle_stage = 7
                act = ("move", [0, 0, 0])
            else:
                axis = int(np.argmax(np.abs(obj_err)))
                move_array = [0, 0, 0]
                if axis == 0:
                    move_array[0] = 1 if obj_err[0] > 0 else -1
                elif axis == 1:
                    move_array[1] = 1 if obj_err[1] > 0 else -1
                else:
                    move_array[2] = 1 if obj_err[2] > 0 else -1
                act = ("move", move_array)
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        if self._oracle_stage == 5:
            # Unused with direct 3D move; keep in sync if reached by legacy logic.
            self._oracle_stage = 4
            return ("move", [0, 0, 0])

        if self._oracle_stage == 6:
            # Unused with direct 3D move; keep in sync if reached by legacy logic.
            self._oracle_stage = 4
            return ("move", [0, 0, 0])

        if self._oracle_stage == 7:
            # Reached goal: keep gripper closed and hold position to satisfy success metric
            self._gripper_bias = -1.0  # keep squeezing
            act = ("move", [0, 0, 0])
            self._last_action = act
            self._prev_finger_opening = finger_opening
            return act

        # Stage 8+: hold position (noop)
        self._last_action = ("move", [0, 0, 0])
        self._prev_finger_opening = finger_opening
        return ("move", [0, 0, 0])

    def get_camera_info(self) -> dict:
        """Get information about available cameras."""
        n_cameras = self.model.ncam
        camera_info = {
            "total_cameras": n_cameras,
            "current_camera_id": None,  # Using default camera
            "image_dimensions": (self._image_height, self._image_width)
        }
        
        if n_cameras > 0:
            camera_info["camera_names"] = []
            for i in range(n_cameras):
                name = self.model.cam(i).name
                camera_info["camera_names"].append(name)
        
        return camera_info

    # VLM mixin API
    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            "You are controlling a robotic arm to pick up a grey cube and place it at the target 3D position. "
            "The robot end-effector (gripper) and the cube are visible in the image. "
            "The left image is a front view of the robot, and the right image moving camera attached to the gripper. "
            "Your goal is to grasp the cube and move it to the red target marker. "
            "Each action moves the end-effector by a fixed step size in the specified direction."
        )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Move the end-effector in 3D space. "
                "Format: `('move', [x, y, z])` where each element is -1, 0, or 1:\n"
                "   - x: -1=left, 0=no change, 1=right\n"
                "   - y: -1=backward, 0=no change, 1=forward\n"
                "   - z: -1=down, 0=no change, 1=up"
            )
        if "gripper" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'gripper': Control the gripper (sticky behavior). "
                "Format: `('gripper', value)` where value is:\n"
                "   - 0: open the gripper\n"
                "   - 1: close the gripper\n"
                "Note: Gripper state is sticky - once set, it continues until changed."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the pick and place session. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed when you pick up the cube and place it at the target position (red marker)."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move right and up: `('move', [1, 0, 1])`")
            examples.append("- To move forward: `('move', [0, 1, 0])`")
            examples.append("- To move down and left: `('move', [-1, 0, -1])`")
            examples.append("- To stay in place: `('move', [0, 0, 0])`")
        if "gripper" in actions:
            examples.append("- To open gripper: `('gripper', 0)`")
            examples.append("- To close gripper: `('gripper', 1)`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        
        # Add gripper behavior explanation
        prompt += "\n\nGripper Behavior: The gripper has sticky behavior - once you open or close it, "
        "that state persists until you change it. This is useful for grasping and holding objects while moving."
        
        # Add feedback if available
        if self.env_feedback:
            prompt += f"\n\nFeedback: {self.env_feedback}"
        
        return prompt

    def _snapshot_state(self):
        """Snapshot MuJoCo FULLPHYSICS state plus time and ctrl; save oracle vars."""
        import mujoco
        from gymnasium.envs.mujoco import utils as mj_utils  # type: ignore

        state = {
            "mjstate": mj_utils.get_state(self, mujoco.mjtState.mjSTATE_FULLPHYSICS),
            "time": float(self.data.time),
            "ctrl": self.data.ctrl.copy(),
            "mocap_pos": self.data.mocap_pos.copy() if hasattr(self.data, "mocap_pos") else None,
            "mocap_quat": self.data.mocap_quat.copy() if hasattr(self.data, "mocap_quat") else None,
        }
        kind = "mjstate"

        # Save controller-side fields that influence oracle behavior
        oracle_state = {
            "_oracle_stage": getattr(self, "_oracle_stage", 0),
            "_prev_finger_opening": getattr(self, "_prev_finger_opening", None),
            "_open_action_candidate": getattr(self, "_open_action_candidate", ("gripper", 0)),
            "_close_action_candidate": getattr(self, "_close_action_candidate", ("gripper", 1)),
            "_open_detected": getattr(self, "_open_detected", False),
            "_last_action": getattr(self, "_last_action", ("move", [0, 0, 0])),
            "_stage3_wait_count": getattr(self, "_stage3_wait_count", 0),
            "_stable_count": getattr(self, "_stable_count", 0),
            "_gripper_bias": getattr(self, "_gripper_bias", 0.0),
        }
        return (kind, state, oracle_state)

    def _restore_state(self, snapshot) -> None:
        kind, state, oracle_state = snapshot
        import mujoco
        from gymnasium.envs.mujoco import utils as mj_utils  # type: ignore

        assert kind == "mjstate"
        mj_utils.set_state(self, state["mjstate"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.time = state["time"]
        self.data.ctrl[:] = state["ctrl"]
        if state.get("mocap_pos") is not None and hasattr(self.data, "mocap_pos"):
            self.data.mocap_pos[:] = state["mocap_pos"]
        if state.get("mocap_quat") is not None and hasattr(self.data, "mocap_quat"):
            self.data.mocap_quat[:] = state["mocap_quat"]
        # Ensure derived quantities are consistent
        if hasattr(self, "_mujoco"):
            self._mujoco.mj_forward(self.model, self.data)  # type: ignore[attr-defined]
        if state.get("mocap_pos") is not None and hasattr(self.data, "mocap_pos"):
            self.data.mocap_pos[:] = state["mocap_pos"]
        if state.get("mocap_quat") is not None and hasattr(self.data, "mocap_quat"):
            self.data.mocap_quat[:] = state["mocap_quat"]

        # Restore Python-side oracle fields
        for k, v in oracle_state.items():
            setattr(self, k, v)

    def solve(self, max_steps: int = 1000, num_steps: int = None, strategy: str = None) -> list[str]:
        """Return a list of oracle actions that solve from current state.

        Uses a closed-loop rollout on a temporary snapshot so the live env state
        is unchanged. Includes gripper actions as needed.
        """
        snapshot = self._snapshot_state()
        import mujoco  # type: ignore
        disableflags_backup: int = int(self.model.opt.disableflags)
        self.model.opt.disableflags = disableflags_backup | mujoco.mjtDisableBit.mjDSBL_WARMSTART
        actions: list[str] = []
        # Debug trace: (grip_pos, obj_pos, achieved_obj, action)
        debug_trace: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple]] = []
        # Record initial observation for alignment checks
        init_obs = self._get_obs()
        self._last_plan_debug = {
            "init_obs": init_obs,
            "trace": debug_trace,
        }
        try:
            for _ in range(max_steps):
                a = self.oracle_action()
                actions.append(str(a))
                # Convert to string for step method
                action_str = str(a)
                obs, _, terminated, truncated, info = self.step(action_str)
                # Append debug info
                grip_pos = obs["observation"][0:3].copy()
                obj_pos = obs["observation"][3:6].copy()
                achieved_obj = obs["achieved_goal"].copy()
                debug_trace.append((grip_pos, obj_pos, achieved_obj, a))
                if terminated or truncated or info.get("is_success", False):
                    break
        finally:
            self._restore_state(snapshot)
            self.model.opt.disableflags = disableflags_backup
            # Save final debug trace
            self._last_plan_debug = {
                "init_obs": init_obs,
                "trace": debug_trace,
            }
        # Append a final stop action
        actions.append(str(("stop", "stop")))
        return actions


class MujocoFetchPickAndPlaceDiscreteEnv(_DiscreteMixin, MujocoFetchPickAndPlaceEnv, EzPickle):
    """Discrete-action variant of FetchPickAndPlace with a staged oracle controller."""

    def __init__(self, reward_type: str = "sparse", distance_threshold: float = 0.05, **kwargs):
        super().__init__(reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)


class MujocoPyFetchPickAndPlaceDiscreteEnv(_DiscreteMixin, MujocoPyFetchPickAndPlaceEnv, EzPickle):
    def __init__(self, reward_type: str = "sparse", distance_threshold: float = 0.05, **kwargs):
        super().__init__(reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)


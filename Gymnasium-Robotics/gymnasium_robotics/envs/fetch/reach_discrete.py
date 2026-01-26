import os
import ast
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.spaces import FuncConditional, Text

from gymnasium_robotics.envs.fetch.reach import (
    MujocoFetchReachEnv,
    MujocoPyFetchReachEnv,
)
# from gymnasium_robotics.envs.vlm_mixin import VLMEnvMixin
import gymnasium as gym


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "reach.xml")


class _DiscreteMixin(gym.VLMEnvMixin):
    """Mixin providing FuncConditional action mapping and a greedy oracle controller.

    Action space:
    - "move": 3-element array [x, y, z] where each element is -1, 0, or 1
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
        self._camera_jitter_lookat = float(kwargs.pop("camera_jitter_lookat", 0.1))  # meters
        # Fallback defaults used if renderer has no baseline camera state
        self._camera_default_azimuth = float(kwargs.pop("camera_default_azimuth", 132.0))
        self._camera_default_elevation = float(kwargs.pop("camera_default_elevation", -14.0))
        # Slightly zoom in default external view
        self._camera_default_distance = float(kwargs.pop("camera_default_distance", 2))
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        
        # FuncConditional action space - only move and stop for Reach task
        self.action_space = FuncConditional({
            "move": spaces.Box(
                low=np.array([-1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1], dtype=np.float32),
                dtype=np.float32,
            ),
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

    def get_init_state(self) -> dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
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
            'goal_pos': goal_pos,
            'camera_config': camera_config,
        }
    
    def reset(self, seed=None, options=None, init_state=None):  # type: ignore[override]
        # Set the RNG seed for reproducible camera jitter
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        obs, info = super().reset(seed=seed, options=options)
        
        if init_state is not None:
            # Restore state from init_state
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
        cam_cfg = getattr(self.mujoco_renderer, "default_cam_config", None)
        if cam_cfg is not None:
            self._set_camera_config(cam_cfg)
        self.mujoco_renderer.camera_id = -1
        img1 = self.mujoco_renderer.render(render_mode="rgb_array")
        # Secondary: wrist/follow camera
        img2 = self._render_follow_camera() if self._use_follow_camera else None
        if img2 is None:
            # Fallback to named secondary if follow disabled
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
        """Render a camera that follows gripper position without rotating."""
        # Ensure viewer exists and use non-fixed camera id
        prev_cam_id = getattr(self.mujoco_renderer, "camera_id", None)
        self.mujoco_renderer.camera_id = -1
        viewer = self.mujoco_renderer._get_viewer(render_mode="rgb_array")
        try:
            body_id = int(self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link"
            ))
            grip = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
            lookat = grip + self._follow_cam_lookat_offset
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
        cam_cfg = getattr(self.mujoco_renderer, "default_cam_config", None)
        if cam_cfg is not None:
            self._set_camera_config(cam_cfg)
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

    def _map_action_to_continuous(self, action) -> np.ndarray:
        """Convert FuncConditional action to continuous action for MuJoCo.
        
        Args:
            action: Either a tuple ("move", [x,y,z]) or ("stop", "stop")
            
        Returns:
            np.ndarray: 4-element continuous action [dx, dy, dz, gripper]
        """
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            # Default noop action
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        action_type, payload = action
        
        if action_type == "move":
            # payload should be [x, y, z] with values -1, 0, or 1
            if isinstance(payload, (list, tuple, np.ndarray)) and len(payload) == 3:
                dx, dy, dz = float(payload[0]), float(payload[1]), float(payload[2])
                return np.array([dx, dy, dz, 0.0], dtype=np.float32)
        
        elif action_type == "stop":
            # Stop action - return noop
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Default noop
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

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
                
                # Apply action through parent machinery
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

    def oracle_action(
        self,
        threshold: float = 0.002,
        prefer_order: Tuple[int, int, int] = (0, 1, 2),
        num_steps: int = None,
    ) -> tuple:
        """Return a greedy action to move towards the goal.

        Returns a single action tuple for the current step.
        """
        obs = self._get_obs()
        achieved = np.array(obs["achieved_goal"], dtype=float)
        goal = np.array(obs["desired_goal"], dtype=float)
        err = goal - achieved

        # If already within axis-wise threshold, just hold
        if np.all(np.abs(err) <= threshold):
            return ("move", [0, 0, 0])
        
        # Find the axis with the largest error
        axis = int(np.argmax(np.abs(err)))
        
        # Create move array
        move_array = [0, 0, 0]
        if err[axis] > 0:
            move_array[axis] = 1
        else:
            move_array[axis] = -1
            
        return ("move", move_array)

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

    def _snapshot_state(self):
        """Snapshot MuJoCo FULLPHYSICS state plus time and ctrl."""
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
        return (kind, state)

    def render(self):
        image = self._get_obs()["image"]
        return image

    def _restore_state(self, snapshot) -> None:
        kind, state = snapshot
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

    # VLM mixin API
    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            "You are controlling a robotic arm to reach a target 3D position. "
            "The left image is a front view of the robot, and the right image moving camera attached to the gripper. "
            "Your goal is to move it to the red target marker. "
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
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the reaching session. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed when the end-effector reaches the target position (red marker)."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move right and up: `('move', [1, 0, 1])`")
            examples.append("- To move forward: `('move', [0, 1, 0])`")
            examples.append("- To move down and left: `('move', [-1, 0, -1])`")
            examples.append("- To stay in place: `('move', [0, 0, 0])`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        
        # Add feedback if available
        if self.env_feedback:
            prompt += f"\n\nFeedback: {self.env_feedback}"
        
        return prompt

    def solve(self, num_steps: int = None, strategy: str = None) -> list[str]:
        """Return a list of oracle actions that solve from current state.
        
        Uses a closed-loop rollout on a temporary snapshot so the live env state
        is unchanged. Similar to pick_and_place implementation.
        """
        # Take a snapshot of current state
        snapshot = self._snapshot_state()
        actions = []
        max_steps = num_steps if num_steps is not None else 10
        max_steps = max(max_steps, 10)
        
        try:
            # Run greedy actions until we reach the goal or hit max steps
            for _ in range(max_steps):
                action = self.oracle_action()
                actions.append(str(action))
                
                # Apply the action to update state for next iteration
                cont_action = self._map_action_to_continuous(action)
                self._set_action(cont_action)
                self._mujoco_step(cont_action)
                self._step_callback()
                
                # Check if we've reached the goal after applying the action
                obs = self._get_obs()
                achieved = np.array(obs["achieved_goal"], dtype=float)
                goal = np.array(obs["desired_goal"], dtype=float)
                err = goal - achieved
                
                if np.all(np.abs(err) <= 0.002):  # Success threshold
                    break
        finally:
            # Restore the original state
            self._restore_state(snapshot)
        
        # Add final stop action
        actions.append(str(("stop", "stop")))
        return actions


class MujocoFetchReachDiscreteEnv(_DiscreteMixin, MujocoFetchReachEnv, EzPickle):
    """
    FuncConditional-action variant of FetchReach with move/stop actions and a greedy oracle.

    This environment maps FuncConditional actions to the underlying 4D continuous control
    used by Fetch: (dx, dy, dz, gripper). For FetchReach, only movement actions are needed
    as the task only requires reaching a target position.
    """

    def __init__(self, reward_type: str = "sparse", distance_threshold: float = 0.1, **kwargs):
        super().__init__(reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)


class MujocoPyFetchReachDiscreteEnv(_DiscreteMixin, MujocoPyFetchReachEnv, EzPickle):
    def __init__(self, reward_type: str = "sparse", distance_threshold: float = 0.1, **kwargs):
        super().__init__(reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, distance_threshold=distance_threshold, **kwargs)


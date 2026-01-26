import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import FuncConditional
import numpy as np
import cv2
import ast
from pathlib import Path
from typing import Optional, List, Dict

class MentalRotation2DEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(
        self,
        sample_dir: str,
        seed: Optional[int] = None,
        image_size: int = 128,
        tolerance: float = 5.0,
        *args, **kwargs
    ):
        super().__init__()
        self.seed(seed)
        self.image_size = image_size
        self.tolerance = tolerance

        # Load image files
        sample_dir = Path(sample_dir)
        if not sample_dir.is_dir():
            raise ValueError(f"{sample_dir!r} is not a valid directory")
        # Recursively find all image files (supports nested directory structure)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.candidates = [
            p for p in sample_dir.rglob('*') 
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        # Sort first for determinism, then shuffle with the seeded RNG
        self.candidates = sorted(self.candidates, key=lambda p: str(p))
        self.np_random.shuffle(self.candidates)
        self.curr_idx = -1

        # Define action space: rotate or stop
        self.action_space = FuncConditional({
            "rotate": spaces.Discrete(361, start=-180),  # -180 to 180 degrees
            "stop":    spaces.Text(4)
        })

        # Observation shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size * 2, 3),
            dtype=np.uint8
        )

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'image_name': self.current_image_name,
            'secret_angle': float(self.secret_angle),
            'agent_angle': float(self.agent_angle),
        }

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            "You are solving a mental rotation task. Two panels appear side by side:\n"
            "- Left: the original circular image.\n"
            "- Right: the image has been rotated by a secret angle.\n"
            "Your job is to undo that rotation and align the right image back to match the left."
        )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "rotate" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'rotate': Rotate the right image by an integer angle. "
                "Format: `('rotate', angle)` where angle is an integer between -180 and 180 degrees "
                "(positive is clockwise, negative is counterclockwise)."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Submit your final adjustment. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += f"\n\nSuccess: You succeed if your final adjustment undoes the secret rotation within ±{self.tolerance}°."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "rotate" in actions:
            examples.append("- To rotate clockwise: `('rotate', 45)`")
            examples.append("- To rotate counterclockwise: `('rotate', -30)`")
        if "stop" in actions:
            examples.append("- To submit: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt
    
    def reset(self, *, seed: Optional[int] = None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed) # TODO: do we call seed here or in the constructor?
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            image_name = init_state['image_name']
            self.orig_img = self._load_image_by_name(image_name)
            self.secret_angle = float(init_state['secret_angle'])
            self.agent_angle = float(init_state['agent_angle'])
        else:
            # Normal reset: load next image and generate random angles
            self.orig_img = self._load_next_image()
            # Secret rotation is clockwise: apply -secret
            self.secret_angle = float(self.np_random.integers(0, 360))
            self.agent_angle = 0.0
        
        self._update_agent_img()
        self.env_feedback = None
        obs = self._get_obs()
        return obs, {}

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

        # 3. Execute ------------------------------------------------------
        if branch == "rotate":
            if not self.action_space["rotate"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self.agent_angle = (self.agent_angle + float(payload)) % 360
                self._update_agent_img()
        
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

    def render(self, mode='rgb_array') -> np.ndarray:
        if mode == 'rgb_array':
            return self._get_obs()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, num_steps: int = None) -> List[str]:
        # Compute needed CCW adjustment to undo a CW secret rotation
        # num_steps = 10
        delta = (self.secret_angle - self.agent_angle) % 360
        delta = delta if delta <= 180 else delta - 360
        if abs(delta) <= self.tolerance:
            return ["('stop', 'stop')"]
        total = int(round(abs(delta)))
        if total <= 1:
            # If the rotation is too small to be split, just return the single move
            return [f"('rotate', {int(round(delta))})", "('stop', 'stop')"]

        if num_steps is None:
            steps = int(self.np_random.integers(2, 5))
        else:
            # Coerce to at least 1 step to avoid invalid ranges
            steps = max(1, int(num_steps))

        # If caller asks for 1 step, just take the whole remaining rotation
        if steps == 1:
            angle = int(np.sign(delta) * total)
            return [f"('rotate', {angle})", "('stop', 'stop')"]

        # Ensure we don't try to sample more points than the range allows
        num_cuts = min(steps - 1, total - 1)
        if num_cuts <= 0:
            cuts = []
        else:
            population = np.arange(1, total, dtype=int)
            if num_cuts >= population.size:
                cuts = population.tolist()
            else:
                cuts = sorted(map(int, self.np_random.choice(population, size=num_cuts, replace=False).tolist()))
        pts = [0] + cuts + [total]
        actions: List[str] = []
        for i in range(1, len(pts)):
            step = pts[i] - pts[i-1]
            angle = int(np.sign(delta) * step)
            actions.append(f"('rotate', {angle})")
        actions.append("('stop', 'stop')")
        return actions

    def close(self):
        pass

    def _get_info(self) -> dict:
        # Error between agent_angle and secret_angle
        err = (self.agent_angle - self.secret_angle) % 360
        if err > 180:
            err -= 360
        return {
            "rotation_error": abs(err),
            "secret_angle": self.secret_angle,
            "agent_angle": self.agent_angle,
            "env_feedback": self.env_feedback
        }

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.orig_img, self.agent_img], axis=1)
    
    def _compute_reward(self) -> float:
        err = self._get_info()["rotation_error"]
        return 1.0 if err <= self.tolerance else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _update_agent_img(self):
        # Apply net rotation = agent CCW minus secret CW
        total = (self.agent_angle - self.secret_angle) % 360
        rotated = self._rotate_image(self.orig_img, total)
        self.agent_img = self._circular_crop(rotated)

    def _circular_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        cv2.circle(mask, center, radius, 255, -1)
        mask_3 = cv2.merge([mask, mask, mask])
        return cv2.bitwise_and(img, mask_3)

    def _load_next_image(self) -> np.ndarray:
        self.curr_idx += 1
        if self.curr_idx >= len(self.candidates):
            raise ValueError("No more images to load")
        path = str(self.candidates[self.curr_idx])
        self.current_image_name = self.candidates[self.curr_idx].name  # Store the image filename
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        return self._circular_crop(img_resized)
    
    def _load_image_by_name(self, image_name: str) -> np.ndarray:
        """Load an image by its filename instead of using the sequential index."""
        # Find the image in candidates by name
        target_path = None
        for candidate in self.candidates:
            if candidate.name == image_name:
                target_path = candidate
                break
        
        if target_path is None:
            raise ValueError(f"Image '{image_name}' not found in sample directory")
        
        self.current_image_name = image_name
        path = str(target_path)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.image_size, self.image_size))
        return self._circular_crop(img_resized)

    def _rotate_image(self, img, angle_deg):
        """
        Rotate the image about its center by angle_deg:
          • positive → clockwise
          • negative → counterclockwise

        OpenCV’s getRotationMatrix2D uses positive=CCW, so we invert the sign.
        """
        # invert because OpenCV’s sign is opposite
        cv_angle = -angle_deg  
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, cv_angle, 1.0)
        rotated = cv2.warpAffine(
            img, mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0)
        )
        return rotated
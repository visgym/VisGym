import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import FuncConditional
import numpy as np
import cv2
import os
import ast
from pathlib import Path
from typing import Optional, List, Tuple, Dict

class ColorizationEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(
        self,
        sample_dir: str,
        seed: Optional[int] = None,
        circle_size: int = 50,
        region_radius: Optional[int] = None,
        max_steps: int = 10,
        hue_tolerance: int = 10,
        sat_tolerance: int = 20,
        img_size: int = 256,
        accuracy_radius: Optional[int] = None,
        min_brightness: int = 45,  
        max_brightness: int = 210,  
    ):
        super().__init__()
        self.seed(seed)
        self.max_steps = max_steps
        self.hue_tolerance = hue_tolerance
        self.circle_size = circle_size
        self.img_size = img_size
        self.region_radius = region_radius
        self.curr_img_idx = -1
        self.sat_tolerance = sat_tolerance
        self.accuracy_radius = accuracy_radius if accuracy_radius is not None \
                            else circle_size // 1.75
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        # Load and shuffle candidate images
        sample_dir = Path(sample_dir)
        self.sample_dir = sample_dir
        if not sample_dir.is_dir():
            raise ValueError(f"{sample_dir!r} is not a valid directory")
        # Recursively find all image files (supports nested directory structure)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.candidates = [
            p for p in sample_dir.rglob('*') 
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        # Sort first for filesystem-order determinism, then shuffle with seeded RNG
        self.candidates = sorted(self.candidates, key=lambda p: str(p))
        self.np_random.shuffle(self.candidates)

        # Define action space and placeholder for observation
        self.action_space = FuncConditional({
            "rotate": spaces.Box(low=-360, high=360, shape=(), dtype=np.int32),
            "saturate": spaces.Box(low=-255, high=255, shape=(), dtype=np.int32),
            "stop":    spaces.Text(4)
        })
        self.observation_space = None

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # generate the prompt dynamically, akin to the JigsawEnv style
        prompt = (
            f"You are performing a color-matching task. You see two images side by side:\n"
            f"- LEFT: A color wheel showing your current hue and saturation selection\n"
            f"- RIGHT: An image with a circular region colored with your current selection (gray outside the circle)\n\n"
            f"Your goal is to adjust the hue and saturation to match the original color that appears at the center "
            f"of the circular region in the right image. The circle's border shows the exact target location.\n\n"
            f"Success criteria: You succeed when your color selection closely matches the target color in both "
            f"hue and saturation."
        )
        prompt += "\n\nAvailable actions:\n"
        # build action descriptions
        action_descriptions = []
        actions = self.action_space.get_function_names()
        if "rotate" in actions:
            action_descriptions.append(
                f"{len(action_descriptions)+1}. 'rotate': Adjust the hue by rotating around the color wheel (circular motion). "
                "Format: `('rotate', angle)` where angle is an integer between -360 and 360 degrees."
            )
        if "saturate" in actions:
            action_descriptions.append(
                f"{len(action_descriptions)+1}. 'saturate': Adjust the saturation by moving toward or away from the center of the wheel. "
                "Format: `('saturate', delta)` where delta is an integer between -255 and 255."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions)+1}. 'stop': Submit your final color choice when you're satisfied with the match. "
                "Format: `('stop', 'stop')`."
            )
        prompt += "\n".join(action_descriptions)

        # examples
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "rotate" in actions:
            examples.append("- `('rotate', 30)` to rotate the hue +30 degrees clockwise")
            examples.append("- `('rotate', -45)` to rotate the hue 45 degrees counter-clockwise")
        if "saturate" in actions:
            examples.append("- `('saturate', 20)` to move away from center (increase saturation, more vivid)")
            examples.append("- `('saturate', -30)` to move toward the center (decrease saturation, more muted)")
        if "stop" in actions:
            examples.append("- `('stop', 'stop')` to submit your answer when the colors match")
        prompt += "\n".join(examples)
        return prompt
    
    # TODO: init state
    # 1. _init_episode can take init_state and initialize based on it, image path / object path
    # 2. get_init_state(): returns the init_state needed for _init_episode, image path / object path
    
    def get_init_state(self) -> Dict:
        return {
            'user_image_path': self.user_image_path,
            'target_pos': self.target_pos,
            'target_hue': self.target_hue,
            'target_sat': self.target_sat,
            'target_val': self.target_val,
            'current_hue': self.current_hue,
            'current_sat': self.current_sat
        }
        
    def reset(self, *, init_state: Optional[dict] = None, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if init_state is not None:
            self.user_image_path = init_state['user_image_path']
            self.user_image_path = os.path.join(self.sample_dir, self.user_image_path)
            self.target_pos = init_state['target_pos']
            self.target_hue = init_state['target_hue']
            self.target_sat = init_state['target_sat']
            self.target_val = init_state['target_val']
            self.current_hue = init_state['current_hue']
            self.current_sat = init_state['current_sat']
            self.user_img_bgr = cv2.imread(self.user_image_path, cv2.IMREAD_COLOR)
        else:
            self._init_episode()

        obs = self._get_obs()
        # Define observation space
        if self.observation_space is None:
            h, w, c = obs.shape
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
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
        if branch == "rotate":
            if not self.action_space["rotate"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self.current_hue = (self.current_hue + float(payload)) % 360
        
        elif branch == "saturate":
            if not self.action_space["saturate"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self.current_sat = int(np.clip(int(getattr(self, 'current_sat', 0)) + int(payload), 0, 255))

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

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        if mode == 'rgb_array':
            return self._get_obs()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = "rotate", num_steps: Optional[int] = None) -> List[str]:
        delta = (self.target_hue - self.current_hue) % 360
        delta = delta if delta <= 180 else delta - 360
        diff = abs(delta)

        # If already within tolerance (both hue and saturation), optionally pad
        sat_err = abs(int(getattr(self, 'current_sat', 0)) - int(self.target_sat))
        if diff <= self.hue_tolerance and sat_err <= getattr(self, 'sat_tolerance', 20):
            actions: List[str] = []
            if num_steps is not None and num_steps > 0:
                target = num_steps
                max_over = 2
                sim_sat = int(getattr(self, 'current_sat', 0))
                while len(actions) < target:
                    # Prefer 4-step reversible block if it won't exceed target + 2
                    if len(actions) + 4 <= target + max_over:
                        k = int(self.np_random.integers(1, 6))
                        # choose safe saturation delta to avoid clipping
                        safe_sat_delta = min(10, sim_sat, 255 - sim_sat)
                        if safe_sat_delta < 1:
                            safe_sat_delta = 1
                        s = int(self.np_random.integers(1, safe_sat_delta + 1))
                        actions.extend([
                            f"('rotate', {k})",
                            f"('rotate', {-k})",
                            f"('saturate', {s})",
                            f"('saturate', {-s})",
                        ])
                        # sim_sat returns to original since we guaranteed no clipping
                    else:
                        # fallback to 2-step rotate pair to stay within +2
                        k = int(self.np_random.integers(1, 6))
                        actions.extend([
                            f"('rotate', {k})",
                            f"('rotate', {-k})",
                        ])
            actions.append("('stop', 'stop')")
            return actions

        # Break diff into a random number of chunks (2-4 steps)
        diff_int = int(round(diff))
        n_steps = int(self.np_random.integers(2, 5))
        # Choose (n_steps-1) cut points
        if diff_int <= 1:
            # If difference is too small, just make one step
            points = [0, diff_int]
        else:
            # Ensure we don't try to sample more elements than available
            max_cuts = min(n_steps-1, diff_int-1)
            if max_cuts > 0:
                cut_points = sorted(map(int, self.np_random.choice(range(1, diff_int), \
                    size=max_cuts, replace=False)))
                points = [0] + cut_points + [diff_int]
            else:
                points = [0, diff_int]

        actions: List[str] = []
        for i in range(1, len(points)):
            step = points[i] - points[i-1]
            # Preserve sign of delta
            angle = int(np.sign(delta) * step)
            actions.append(f"('rotate', {angle})")

        # Add saturation adjustment steps toward target
        sat_delta = int(self.target_sat) - int(getattr(self, 'current_sat', 0))
        sat_diff = abs(sat_delta)
        if sat_diff > 0:
            n_sat_steps = int(self.np_random.integers(1, 4))
            if sat_diff < n_sat_steps:
                n_sat_steps = sat_diff
            if n_sat_steps <= 0:
                n_sat_steps = 1
            sat_points = [0, sat_diff]
            if sat_diff > 1 and n_sat_steps > 1:
                cnum = n_sat_steps - 1
                # Ensure we don't try to sample more elements than available
                max_sat_cuts = min(cnum, sat_diff-1)
                if max_sat_cuts > 0:
                    sat_cuts = sorted(map(int, self.np_random.choice(range(1, sat_diff), \
                        size=max_sat_cuts, replace=False)))
                    sat_points = [0] + sat_cuts + [sat_diff]
            for i in range(1, len(sat_points)):
                step = sat_points[i] - sat_points[i-1]
                sgn = 1 if sat_delta > 0 else -1
                actions.append(f"('saturate', {sgn*int(step)})")

        # Pad with reversible pairs to reach at least num_steps (allow up to +2 overflow)
        if num_steps is not None and len(actions) < num_steps:
            target = num_steps
            max_over = 2
            while len(actions) < target:
                remain = target - len(actions)
                # Prefer a 4-step block (rotate pair + saturate pair) if it keeps us within +2
                if len(actions) + 4 <= target + max_over and remain >= 3:
                    idx = int(self.np_random.integers(0, len(actions) + 1)) if len(actions) > 0 else 0
                    # Simulate state up to the insertion point to choose safe saturation
                    _, pre_saturate = self._get_state_after_actions(actions[:idx])
                    safe_sat_delta = min(10, pre_saturate, 255 - pre_saturate)
                    if safe_sat_delta < 1:
                        safe_sat_delta = 1
                    s = int(self.np_random.integers(1, safe_sat_delta + 1))
                    k = int(self.np_random.integers(1, 6))
                    actions.insert(idx, f"('rotate', {-k})")
                    actions.insert(idx, f"('rotate', {k})")
                    actions.insert(idx, f"('saturate', {-s})")
                    actions.insert(idx, f"('saturate', {s})")
                else:
                    # Use a 2-step rotate pair as a safe filler to avoid overshooting by > +2
                    idx = int(self.np_random.integers(0, len(actions) + 1)) if len(actions) > 0 else 0
                    k = int(self.np_random.integers(1, 6))
                    actions.insert(idx, f"('rotate', {-k})")
                    actions.insert(idx, f"('rotate', {k})")
            # No trimming; final length is in [num_steps, num_steps+2]

        # Finally submit
        actions.append("('stop', 'stop')")
        return actions

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        bgr_obs = self._combine_obs(
            self._render_wheel_observation(), self._set_hue(self.current_hue)
        )
        return cv2.cvtColor(bgr_obs, cv2.COLOR_BGR2RGB)

    def _get_info(self) -> dict:
        # Calculate color space distance for info
        hue_diff = min(abs(self.current_hue - self.target_hue), 360 - abs(self.current_hue - self.target_hue))
        sat_diff = abs(int(getattr(self, 'current_sat', 0)) - int(self.target_sat))
        val_diff = abs(self.target_val - self.target_val)  # Always 0 since we use same value
        
        color_distance = np.sqrt((hue_diff * 2.0)**2 + sat_diff**2 + val_diff**2)
        max_distance = np.sqrt((180 * 2.0)**2 + 255**2 + 255**2)
        threshold_distance = (self.accuracy_radius / self.circle_size) * max_distance * 0.3
        
        return {
            "hue_error": hue_diff,
            "sat_error": sat_diff,
            "color_distance": color_distance,
            "threshold_distance": threshold_distance,
            "accuracy_radius": self.accuracy_radius,
            "env_feedback": self.env_feedback
        }
    
    def _compute_reward(self):
        # Calculate color space distance between prediction and ground truth
        pred_hue = self.current_hue
        pred_sat = int(getattr(self, 'current_sat', 0))
        pred_val = self.target_val
        
        gt_hue = self.target_hue
        gt_sat = int(self.target_sat)
        gt_val = self.target_val
        
        hue_diff = min(abs(pred_hue - gt_hue), 360 - abs(pred_hue - gt_hue))
        sat_diff = abs(pred_sat - gt_sat)
        val_diff = abs(pred_val - gt_val)
        
        # Weighted distance (hue is most important)
        color_distance = np.sqrt((hue_diff * 2.0)**2 + sat_diff**2 + val_diff**2)
        
        max_distance = np.sqrt((180 * 2.0)**2 + 255**2 + 255**2)
        threshold_distance = (self.accuracy_radius / self.circle_size) * max_distance * 0.3
        
        # Calculate reward based on distance
        if color_distance <= threshold_distance:
            return 1.0
        else:
            return 0.0

########################################################
# End of all methods, start of helper methods
########################################################


    def _init_episode(self, init_state: Dict = None):
        if init_state is not None:
            self.user_image_path = init_state['user_image_path']
            self.target_pos = init_state['target_pos']
            self.target_hue = init_state['target_hue']
            self.target_sat = init_state['target_sat']
            self.target_val = init_state['target_val']
            return
        # Advance image index
        self.curr_img_idx += 1
        if self.curr_img_idx >= len(self.candidates):
            raise ValueError("No more images to load")
        user_image_path = str(self.candidates[self.curr_img_idx])
        self.user_image_path = user_image_path
        # Load image and pick a target pixel with low color variance
        img_bgr = cv2.imread(user_image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {user_image_path}")
        self.user_img_bgr = img_bgr
        h_img, w_img = img_bgr.shape[:2]
        margin = self.circle_size
        
        # Find candidate points with low color variance
        self.target_pos = self._find_low_variance_point(img_bgr, margin)

        # Determine true hue and its saturation/value
        self.target_hue = self._get_target_hue(self.target_pos, user_image_path)
        hsv_full = cv2.cvtColor(self.user_img_bgr, cv2.COLOR_BGR2HSV)
        tx, ty = map(int, self.target_pos)
        _, self.target_sat, self.target_val = hsv_full[ty, tx]

        # Initialize current hue and saturation
        self.current_hue = float(self.np_random.uniform(0, 360))
        self.current_sat = int(self.np_random.integers(0, 256))
        
        # Recompute wheel if target value changed
        if hasattr(self, '_precomputed_wheel') and hasattr(self, '_last_target_val'):
            if self._last_target_val != self.target_val:
                self._precompute_wheel()
        self._last_target_val = self.target_val

    def _find_low_variance_point(self, img_bgr, margin, variance_threshold=1500.0, num_candidates=50):
        """
        Find a point with low color variance in its neighborhood.
        
        Args:
            img_bgr: Input BGR image
            margin: Margin to avoid edges
            variance_threshold: Maximum variance threshold for random sampling
            num_candidates: Number of candidate points to evaluate
            
        Returns:
            Tuple (x, y) of the selected point
        """
        h_img, w_img = img_bgr.shape[:2]
        
        # Define search region (avoiding margins)
        min_x, max_x = margin, w_img - margin
        min_y, max_y = margin, h_img - margin
        
        # Sample candidate points
        candidates = []
        for _ in range(num_candidates):
            x = self.np_random.integers(min_x, max_x)
            y = self.np_random.integers(min_y, max_y)
            candidates.append((x, y))
        
        # Evaluate color variance and brightness for each candidate
        valid_candidates = []
        best_point = None
        min_variance = float('inf')
        
        for x, y in candidates:
            variance = self._calculate_color_variance(img_bgr, x, y)
            brightness = self._calculate_brightness(img_bgr, x, y)
            
            # Check brightness constraints
            brightness_valid = self.min_brightness <= brightness <= self.max_brightness
            
            # Track best point regardless of threshold (but still check brightness)
            if variance < min_variance and brightness_valid:
                min_variance = variance
                best_point = (x, y)
            
            # Collect points within threshold and brightness constraints
            if variance <= variance_threshold and brightness_valid:
                valid_candidates.append((x, y, variance, brightness))
        
        # If we found points within threshold, randomly sample one
        if valid_candidates:
            # Sort by variance (lower is better) and take top 20% for random selection
            valid_candidates.sort(key=lambda x: x[2])
            top_candidates = valid_candidates[:max(1, len(valid_candidates) // 5)]
            selected_idx = self.np_random.integers(0, len(top_candidates))
            return (top_candidates[selected_idx][0], top_candidates[selected_idx][1])
        
        # If no points within threshold, return best point (if any)
        if best_point is not None:
            return best_point
        
        # If no valid points found at all, fallback to center of image
        h_img, w_img = img_bgr.shape[:2]
        fallback_x = w_img // 2
        fallback_y = h_img // 2
        print(f"Warning: No valid points found within brightness range [{self.min_brightness}, {self.max_brightness}]. Using fallback point.")
        return (fallback_x, fallback_y)
    
    def _calculate_color_variance(self, img_bgr, x, y):
        """Calculate color variance for a point's neighborhood."""
        h_img, w_img = img_bgr.shape[:2]
        
        # Define neighborhood around the point
        neighborhood_size = min(self.circle_size // 2, 20)
        x1 = max(0, x - neighborhood_size)
        y1 = max(0, y - neighborhood_size)
        x2 = min(w_img, x + neighborhood_size + 1)
        y2 = min(h_img, y + neighborhood_size + 1)
        
        # Extract neighborhood
        neighborhood = img_bgr[y1:y2, x1:x2]
        
        # Calculate color variance (using HSV for better color space)
        neighborhood_hsv = cv2.cvtColor(neighborhood, cv2.COLOR_BGR2HSV)
        
        # Calculate variance for each channel
        h_var = np.var(neighborhood_hsv[:, :, 0])
        s_var = np.var(neighborhood_hsv[:, :, 1])
        v_var = np.var(neighborhood_hsv[:, :, 2])
        
        # Combined variance (weighted)
        total_variance = h_var * 2.0 + s_var + v_var  # Hue variance is more important
        
        return total_variance

    def _calculate_brightness(self, img_bgr, x, y):
        """Calculate the brightness (value) of a point in HSV color space."""
        h_img, w_img = img_bgr.shape[:2]
        
        # Define neighborhood around the point
        neighborhood_size = min(self.circle_size // 2, 20)
        x1 = max(0, x - neighborhood_size)
        y1 = max(0, y - neighborhood_size)
        x2 = min(w_img, x + neighborhood_size + 1)
        y2 = min(h_img, y + neighborhood_size + 1)
        
        # Extract neighborhood
        neighborhood = img_bgr[y1:y2, x1:x2]
        
        # Convert to HSV and calculate mean brightness
        neighborhood_hsv = cv2.cvtColor(neighborhood, cv2.COLOR_BGR2HSV)
        mean_brightness = np.mean(neighborhood_hsv[:, :, 2])  # V channel is brightness
        
        return mean_brightness

    def _get_target_hue(self, target_pos, user_image_path) -> float:
        if not os.path.isfile(user_image_path):
            raise FileNotFoundError(f"Image not found: {user_image_path}")
        img_bgr = cv2.imread(user_image_path)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hue_0_179 = img_hsv[int(target_pos[1]), int(target_pos[0]), 0]
        return float(hue_0_179) * 2.0

    def _set_hue(self, hue_color: float) -> np.ndarray:
        # Start with the original colored image
        base_bgr = self.user_img_bgr.copy()
        h_img, w_img = base_bgr.shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cx, cy = map(int, self.target_pos)
        cv2.circle(mask, (cx, cy), int(self.circle_size), 255, thickness=-1)

        hsv = cv2.cvtColor(self.user_img_bgr, cv2.COLOR_BGR2HSV)
        h_idx = int((hue_color % 360) / 2)
        s_idx = int(np.clip(int(getattr(self, 'current_sat', 0)), 0, 255))
        hsv[..., 0][mask == 255] = h_idx
        hsv[..., 1][mask == 255] = s_idx
        colored_full = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        base_bgr[mask == 255] = colored_full[mask == 255]
        if self.region_radius is not None:
            # Create outer circle mask
            mask_outer = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(mask_outer, (cx, cy), int(self.region_radius), 255, thickness=-1)
            
            # Create inner circle mask
            mask_inner = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.circle(mask_inner, (cx, cy), int(self.circle_size), 255, thickness=-1)
            
            # Create ring mask (outer circle minus inner circle)
            mask_ring = mask_outer - mask_inner
            
            # Apply gray color to the ring region
            gray_color = [128, 128, 128]  # Gray color in BGR
            base_bgr[mask_ring == 255] = gray_color

        cv2.circle(base_bgr, (cx, cy), int(self.circle_size), (0, 0, 0), thickness=2)
        return base_bgr

    def _render_wheel_observation(self) -> np.ndarray:
        # Use pre-computed wheel if available and target value hasn't changed, otherwise compute it
        if not hasattr(self, '_precomputed_wheel') or \
            not hasattr(self, '_precomputed_target_val') or \
                self._precomputed_target_val != self.target_val:
            self._precompute_wheel()
            self._precomputed_target_val = self.target_val
        
        # Copy the pre-computed wheel and add the pointer
        wheel = self._precomputed_wheel.copy()
        
        # Add pointer at current position
        mid = self.img_size // 2
        rad_ang = self.current_hue * np.pi / 180.0
        rad_len = int((np.clip(int(getattr(self, 'current_sat', 0)), 0, 255) / 255.0) * (mid - 2))
        x = mid + int(rad_len * np.cos(rad_ang))
        y = mid + int(rad_len * np.sin(rad_ang))
        cv2.circle(wheel, (x, y), 5, (0, 0, 0), thickness=2)
                
        return wheel
    
    def _precompute_wheel(self):
        """Pre-compute the static color wheel using the exact same method as original."""
        mid = self.img_size // 2
        wheel = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Use the exact same nested loop logic as the original to ensure identical results
        for r in range(self.img_size):
            for c in range(self.img_size):
                dx, dy = c - mid, r - mid
                dist = np.hypot(dx, dy)
                if dist <= mid:
                    angle = (np.arctan2(dy, dx) * 180.0 / np.pi) % 360
                    h = int(angle / 2)
                    s = int(np.clip((dist / mid) * 255.0, 0, 255))
                    hsv = np.uint8([[[h, s, self.target_val]]])
                    wheel[r, c] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        
        self._precomputed_wheel = wheel

    def _combine_obs(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        hL, wL, _ = left_img.shape
        hR, wR, _ = right_img.shape
        new_h = max(hL, hR)
        if hL < new_h:
            left_img = cv2.copyMakeBorder(left_img, 0, new_h - hL, 0, 0, cv2.BORDER_CONSTANT)
        if hR < new_h:
            right_img = cv2.copyMakeBorder(right_img, 0, new_h - hR, 0, 0, cv2.BORDER_CONSTANT)
        return np.concatenate([left_img, right_img], axis=1)

    def _get_state_after_actions(self, actions: List[str]) -> Tuple[float, int]:
        sim_hue = self.current_hue
        sim_sat = int(getattr(self, 'current_sat', 0))

        for action_str in actions:
            try:
                branch, payload = ast.literal_eval(action_str)
                if branch == 'rotate':
                    sim_hue = (sim_hue + payload) % 360
                elif branch == 'saturate':
                    sim_sat = np.clip(sim_sat + payload, 0, 255)
            except (ValueError, SyntaxError):
                continue # Ignore malformed actions like ('stop', 'stop')
                
        return sim_hue, sim_sat

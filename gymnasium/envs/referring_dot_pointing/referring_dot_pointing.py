import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import os
from gymnasium.utils import seeding
from gymnasium.spaces import FuncConditional
import ast
from typing import Dict, Optional, Any

from gymnasium.envs.referring_dot_pointing.refer_dataloader.refer import REFER


class ReferringDotPointingEnv(gym.Env, gym.VLMEnvMixin):
    """
    Referring‐expression pointing using ground‐truth segmentation masks.
    Reward = 1.0 if the click lies *inside* the annotated mask, else 0.0.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        sample_dir: str,
        splitBy: str = "unc",
        radius: int = 5,
        render_mode: str = "rgb_array",
        seed = None,
        accept_any_same_category: bool = False,
        data_chunk: tuple[int, int] = None, # the partition for subsampling, default to None
        max_steps: int = 50,
        *args, **kwargs
    ):
        super().__init__()
        self.render_mode = render_mode
        self.accept_any_same_category = accept_any_same_category
        self.max_steps = max_steps
        # load REFER API
        self.refer = REFER(sample_dir, dataset="refcoco+", splitBy=splitBy)
        all_ref_ids = self.refer.getRefIds()
        # Validate that every ref_id has valid data (image exists and mask is valid)
        valid_ref_ids = []
        for ref_id in all_ref_ids:
            try:
                ref = self.refer.Refs[ref_id]
                ann = self.refer.refToAnn[ref_id]
                img_info = self.refer.Imgs[ann["image_id"]]
                img_path = os.path.join(self.refer.IMAGE_DIR, img_info["file_name"])
                
                # Check if image file exists and can be loaded
                if not os.path.exists(img_path):
                    continue
                    
                # Check if mask is valid (has non-zero area)
                M = self.refer.getMask(ref)
                mask = M["mask"].astype(bool)
                if not mask.any():  # Check if mask has any True values
                    continue
                    
                valid_ref_ids.append(ref_id)
            except (KeyError, Exception):
                # Skip invalid references
                continue
        
        if not valid_ref_ids:
            raise ValueError("No valid references found with valid images and masks")
        
        # Sort and subsample the valid references
        valid_ref_ids.sort()

        if data_chunk is not None:
            valid_ref_ids = valid_ref_ids[data_chunk[0]:data_chunk[1]]
        
        self.seed(seed)
        
        # Shuffle the reference IDs based on seed for reproducibility
        shuffled_ref_ids = valid_ref_ids.copy()
        self.np_random.shuffle(shuffled_ref_ids)
        
        self.ref_ids = shuffled_ref_ids
        self.image_dir = self.refer.IMAGE_DIR
        self.curr_idx = -1
        # episode settings
        self.radius = radius

        sample_ann = self.refer.refToAnn[self.ref_ids[0]]
        sample_info = self.refer.Imgs[sample_ann["image_id"]]
        sample_img = cv2.imread(os.path.join(self.image_dir, sample_info["file_name"]))
        if sample_img is None:
            raise ValueError(f"Could not load sample image {sample_info['file_name']!r}")
        self.height, self.width = sample_img.shape[:2]

        # continuous 2D action: (x,y) normalized coords [0.0, 1.0]
        self.action_space = FuncConditional({
            "mark": spaces.Box(
                            low=np.array([0.0, 0.0], dtype=np.float32),
                            high=np.array([1.0, 1.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
            "stop":    spaces.Text(4)
        })
        
        
        # RGB image observation
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # internal state
        self.current_img: np.ndarray
        self.current_mask: np.ndarray
        self.class_mask: np.ndarray | None = None  # union of same-category instances (optional)
        self.current_description: str
        self.dot = None
        self.env_feedback = None

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = f"Your goal is to point to '{self.current_description}' in the image."

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "mark" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'mark': Mark a point at the specified coordinates. "
                "Format: `('mark', (x_norm, y_norm))` where x_norm, y_norm are normalized coordinates "
                f"between 0.0 and 1.0, which map to pixel x = round(x_norm*({self.width}-1)), "
                f"y = round(y_norm*({self.height}-1))."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Submit your current mark as final answer. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "mark" in actions:
            examples.append("- To mark center: `('mark', (0.5, 0.5))`")
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
            'ref_id': self.current_ref_id,
            'sentence_idx': self.current_sentence_idx,
        }
    
    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        self.dot = None
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            rid = init_state['ref_id']
            sentence_idx = init_state['sentence_idx']
            
            # Verify that ref_id is valid
            if rid not in self.refer.Refs:
                raise ValueError(f"Reference ID {rid} not found in dataset")
            
            self.current_ref_id = rid
            self.current_sentence_idx = sentence_idx
        else:
            # Normal reset: use sequential sampling
            self.curr_idx += 1
            if self.curr_idx >= len(self.ref_ids):
                raise ValueError("All references exhausted")
            rid = self.ref_ids[self.curr_idx]
            self.current_ref_id = rid
            self.current_sentence_idx = 0  # Default to first sentence
        
        # Common loading logic for both paths
        ref = self.refer.Refs[rid]
        ann = self.refer.refToAnn[rid]
        img_info = self.refer.Imgs[ann["image_id"]]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        # load image
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
        
        if img_bgr is None:
            raise RuntimeError(f"Could not load image: {img_path}")
        self.current_img = img_bgr
        self.height, self.width = self.current_img.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )

        # get segmentation mask from REFER
        M = self.refer.getMask(ref)
        mask = M["mask"].astype(bool)  # H×W bool
        self.current_mask = mask

        # build union-of-category mask if enabled
        if self.accept_any_same_category:
            try:
                cat_id = ref["category_id"]
                img_id = ann["image_id"]
                union = np.zeros_like(self.current_mask, dtype=bool)
                for a in self.refer.imgToAnns.get(img_id, []):
                    if a.get("category_id") == cat_id:
                        Mm = self.refer.getMask(ann=a)
                        union |= Mm["mask"].astype(bool)
                # fallback to instance mask if union ended empty (shouldn't happen but safe)
                self.class_mask = union if union.any() else self.current_mask
            except Exception:
                self.class_mask = self.current_mask
        else:
            self.class_mask = self.current_mask
       

        # Use the specified sentence index
        if self.current_sentence_idx < len(ref["sentences"]):
            self.current_description = ref["sentences"][self.current_sentence_idx]["sent"]
        else:
            # Fallback to first sentence if index is out of range
            self.current_description = ref["sentences"][0]["sent"]

        return self.render(), {}

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
        if branch == "mark":
            if not self.action_space["mark"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                # args are normalized [0..1]
                x_norm, y_norm = float(payload[0]), float(payload[1])
                # convert to pixel coords
                ix = int(np.clip(x_norm, 0.0, 1.0) * (self.width - 1))
                iy = int(np.clip(y_norm, 0.0, 1.0) * (self.height - 1))
                self.dot = (ix, iy)
        
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
            img = self.current_img.copy()
            if self.dot is not None:
                x, y = self.dot
                h, w = img.shape[:2]
                # sample a small local patch around the click
                x0, x1 = max(0, x-1), min(w-1, x+1)
                y0, y1 = max(0, y-1), min(h-1, y+1)
                patch = img[y0:y1+1, x0:x1+1]

                # compute average brightness (convert to grayscale mean)
                b, g, r = patch[...,0], patch[...,1], patch[...,2]
                brightness = ((0.299*r + 0.587*g + 0.114*b).mean())

                # choose dot color for max contrast
                # if patch is bright, use green; if dark, use red
                if brightness > 127:
                    dot_color = (0, 255, 0)  # green in BGR
                else:
                    dot_color = (0, 0, 255)  # red in BGR

                cv2.circle(img, (x, y), self.radius, dot_color, thickness=-1)

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = "mark", num_steps: int = 3, noise_level: float = 0.5) -> list[str]:
        """
        Returns a sequence of actions that will lead to the groundtruth solution.
        It starts with a random point inside the mask and gradually moves to the center.
        
        Args:
            num_steps (int): The number of 'mark' steps to take before stopping.
            noise_level (float): The magnitude of the initial noise, relative to the target's size.

        Returns:
            list[str]: A list of actions including 'mark' steps and a final 'stop'.
        """
        if num_steps <= 0:
            num_steps = 1

        # 1. Find the ground truth center and sample points within the mask
        rows, cols = np.where(self.current_mask)
        h, w = self.current_img.shape[:2]
        if len(rows) == 0:
            # Fallback: use image center if mask is empty
            h, w = self.current_img.shape[:2]
            gt_center_x, gt_center_y = (w - 1) / 2, (h - 1) / 2
            start_x, start_y = gt_center_x, gt_center_y
        else:
            # Calculate the center of mass (more accurate than bbox center)
            gt_center_x = cols.mean()
            gt_center_y = rows.mean()
            
            if len(rows) > 1:
                # Randomly select a point from the mask
                idx = self.np_random.integers(0, len(rows))
                start_x, start_y = cols[idx], rows[idx]
            else:
                # If mask has only one pixel, use it
                start_x, start_y = cols[0], rows[0]

        actions = []
        # 2. Interpolate from the random start to the center of mass
        for i in range(1, num_steps + 1):
            # The interpolation factor t goes from (1/N) to 1.0
            t = i / num_steps
            interp_x = start_x + t * (gt_center_x - start_x)
            interp_y = start_y + t * (gt_center_y - start_y)
            
            # Normalize coordinates to [0.0, 1.0] range
            x_norm = np.clip(interp_x / (self.width - 1), 0.0, 1.0)
            y_norm = np.clip(interp_y / (self.height - 1), 0.0, 1.0)
            
            # Create the mark action with normalized coordinates
            mark_action = f"('mark', ({x_norm:.4f}, {y_norm:.4f}))"
            actions.append(mark_action)

        # 3. Add the final stop action
        actions.append("('stop', 'stop')")
        
        return actions

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        return self.render()
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "env_feedback": self.env_feedback
        }

    def _compute_reward(self) -> float:
        if self.dot is not None:
            x, y = self.dot
            in_mask = False
            mask_to_check = self.class_mask if self.accept_any_same_category else self.current_mask
            if 0 <= y < mask_to_check.shape[0] and 0 <= x < mask_to_check.shape[1]:
                in_mask = bool(mask_to_check[y, x])
            return 1.0 if in_mask else 0.0
        return 0.0

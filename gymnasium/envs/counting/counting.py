import os
import ast
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import FuncConditional
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from lvis import LVIS
from typing import Optional, Dict

class CountingEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(
        self,
        annotation_file: str = "./partial_datasets/counting/lvis_v1_train.json",
        sample_dir: str = "./partial_datasets/counting",
        categories: list[str] | None = None,
        max_count: int = 10,
        min_count: int = 2,
        radius: int = 5,
        seed: int | None = 42,
        data_chunk: tuple[int, int] | None = None, # used for sft data gen parallelization
        require_exhaustive: bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.sample_dir = sample_dir
        self.max_count = max_count
        self.min_count = min_count
        self.radius = radius
        self.require_exhaustive = require_exhaustive

        print("Loading LVIS annotation file...")
        self.lvis = LVIS(annotation_file)
        
        if categories:
            self.cat_ids = self.lvis.get_cat_ids(cat_names=categories)
        else:
            self.cat_ids = self.lvis.get_cat_ids()

        print("Filtering valid samples...")
        all_img_ids = self.lvis.get_img_ids()
        valid_samples = []
        
        batch_size = 500
        for i in range(0, len(all_img_ids), batch_size):
            batch_img_ids = all_img_ids[i:i + batch_size]
            
            batch_img_infos = self.lvis.load_imgs(batch_img_ids)
            
            for img_id, img_info in zip(batch_img_ids, batch_img_infos):
                not_exhaustive = set(img_info.get("not_exhaustive_category_ids", []))
                
                # Get annotation IDs for this image
                ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
                if not ann_ids:
                    continue
                
                # Load annotations for this image
                anns = self.lvis.load_anns(ann_ids)
                
                # Count annotations per category
                cat_counts = {}
                for ann in anns:
                    cat_id = ann["category_id"]
                    if cat_id not in cat_counts:
                        cat_counts[cat_id] = 0
                    cat_counts[cat_id] += 1
                
                # Check if any category meets our criteria
                for cat_id, count in cat_counts.items():
                    # Skip if not in our target categories
                    if categories and cat_id not in self.cat_ids:
                        continue
                    
                    # Skip if not exhaustive (if required)
                    if self.require_exhaustive and cat_id in not_exhaustive:
                        continue
                    
                    # Check count constraints
                    if self.min_count <= count <= self.max_count:
                        # Get file name
                        file_name = img_info.get("file_name")
                        if not file_name and "coco_url" in img_info:
                            file_name = img_info["coco_url"].split("/")[-1]
                        
                        if file_name:
                            valid_samples.append({
                                'img_id': img_id,
                                'category_id': cat_id,
                                'count': count,
                                'file_name': file_name,
                                'not_exhaustive': cat_id in not_exhaustive
                            })
                            break  # Only need one valid category per image
            
            # Clear batch data immediately
            del batch_img_infos
            del anns

        if not valid_samples:
            raise ValueError("No valid LVIS samples found under the given constraints.")

        # ---- Sort and apply data chunk ----
        print("Sorting and applying data chunk...")
        valid_samples.sort(key=lambda x: (x['img_id'], x['category_id']))
        
        if data_chunk is not None:
            start_idx, end_idx = data_chunk
            valid_samples = valid_samples[start_idx:end_idx]
        
        # Store only the essential data
        self.samples = valid_samples
        
        # Stats
        total = len(all_img_ids)
        print("LVIS Counting Environment Statistics:")
        print(f"  Total images: {total}")
        print(f"  Valid samples (min={min_count}, max={max_count}, exhaustive={require_exhaustive}): {len(valid_samples)}")
        print(f"  Filtered out: {total - len(valid_samples)}")

        self.seed(seed)
        self.sequence = self.np_random.permutation(len(self.samples))
        self.seq_idx = -1

        # Defer image loading until first reset to save memory
        self.height, self.width = 480, 640  # Default size, will be updated in reset()

        # ---- Action/Observation spaces ----
        self.action_space = FuncConditional({
            "mark": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "undo": spaces.Text(max_length=6),
            "guess": spaces.Discrete(self.max_count + 1),
            "stop": spaces.Text(max_length=4),
        })
        self.observation_space = spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

        # Internals
        self.base_img = None
        self.dots = []
        self.current_guess = None
        self.cat_id = None
        self.category_name = None
        self.target_anns = []
        self.centers = []
        self.true_count = 0
        
        # Clear the LVIS object to free memory after filtering
        del self.lvis
        self.lvis = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            f"You are solving a {self.category_name}-counting task. "
            f"Count the number of {self.category_name} in the image. "
            "You can place dots to mark instances and then record your final count."
        )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "mark" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'mark': Place a dot at normalized coordinates. "
                "Format: `('mark', (x, y))` where x and y are normalized coordinates between 0.0 and 1.0."
            )
        if "undo" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'undo': Remove your last placed dot. "
                "Format: `('undo', 'undo')`"
            )
        if "guess" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'guess': Record your count guess. "
                f"Format: `('guess', N)` where N is an integer between {self.min_count} and {self.max_count}."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the counting session. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed if your final count guess matches the true number of objects."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "mark" in actions:
            examples.append("- To mark a point: `('mark', (0.5, 0.3))`")
        if "undo" in actions:
            examples.append("- To undo: `('undo', 'undo')`")
        if "guess" in actions:
            examples.append(f"- To guess count: `('guess', 5)`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt
    
    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'file_name': self.samples[self.sequence[self.seq_idx]]['file_name'],
            'category_name': self.category_name,
            'true_count': self.true_count,
            'cat_id': self.cat_id,
            'img_id': self.img_id,
        }

    def reset(self, *, seed=None, options=None, init_state: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.dots = []
        self.current_guess = None
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            file_name = init_state['file_name']
            category_name = init_state['category_name']
            self.true_count = init_state['true_count']
            self.cat_id = init_state['cat_id']
            # Reload LVIS if needed to get cat_id and img_id
            if self.lvis is None:
                self.lvis = LVIS(self.annotation_file)
            
            self.category_name = category_name
            self.img_id = init_state['img_id']
            
            # Find img_id from file_name
            # Note: LVIS doesn't have a direct file_name -> img_id lookup, so we need to search
            all_img_ids = self.lvis.get_img_ids()
            self.img_id = None
            
        else:
            # Normal reset: use sequential sampling
            self.seq_idx += 1
            if self.seq_idx >= len(self.sequence):
                raise ValueError("All LVIS samples exhausted.")
            
            # Get the sample data
            sample = self.samples[self.sequence[self.seq_idx]]
            self.img_id = sample['img_id']
            self.cat_id = sample['category_id']
            self.true_count = sample['count']
            file_name = sample['file_name']
            
            # Load category name (need to reload LVIS for this)
            if self.lvis is None:
                self.lvis = LVIS(self.annotation_file)
            
            self.category_name = self.lvis.load_cats([self.cat_id])[0]["name"]

        # Common loading logic for both paths
        # Load image only when needed
        path = os.path.join(self.sample_dir, file_name)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not load image: {path}")
        
        # Clear previous image to free memory
        if self.base_img is not None:
            del self.base_img
        
        self.base_img = bgr
        new_height, new_width = bgr.shape[:2]
        
        # Update dimensions and observation space if changed
        if new_height != self.height or new_width != self.width:
            self.height, self.width = new_height, new_width
            self.observation_space = spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

        # Load annotations only for this specific image and category
        ann_ids = self.lvis.get_ann_ids(img_ids=[self.img_id])
        anns = self.lvis.load_anns(ann_ids)
        
        # Collect target annotations for this category
        self.target_anns = [a for a in anns if a["category_id"] == self.cat_id]

        # Precompute centers
        self.centers = []
        for a in self.target_anns:
            # Prefer mask centers; LVIS provides 'segmentation' that we can convert to a binary mask
            m = self._ann_to_mask(a)
            if m is not None:
                ys, xs = np.nonzero(m)
                if len(xs):
                    self.centers.append((float(xs.mean()), float(ys.mean())))
                    continue
            # Fallback: bbox center
            x, y, w, h = a["bbox"]
            self.centers.append((x + w / 2.0, y + h / 2.0))

        # Clear annotations to free memory
        del anns

        return self._get_obs(), {}

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
        if branch == "mark":
            if not self.action_space["mark"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                x, y = float(payload[0]), float(payload[1])
                ix = int(np.clip(x, 0.0, 1.0) * (self.width - 1))
                iy = int(np.clip(y, 0.0, 1.0) * (self.height - 1))
                self.dots.append((ix, iy))

        elif branch == "undo":
            if not self.action_space["undo"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                if self.dots:
                    self.dots.pop()

        elif branch == "guess":
            if not self.action_space["guess"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self.current_guess = int(payload)

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

    def solve(self, strategy: str = "mark_all", num_steps: int = None) -> list[str]:
        if strategy == "mark_all":
            actions = []
            for cx, cy in self.centers:
                x_norm = np.clip(cx / max(1, self.width - 1), 0.0, 1.0)
                y_norm = np.clip(cy / max(1, self.height - 1), 0.0, 1.0)
                actions.append(f"('mark', ({x_norm:.4f}, {y_norm:.4f}))")
            actions.append(f"('guess', {self.true_count})")
            actions.append("('stop', 'stop')")
            return actions
        elif strategy == "guess_only":
            return [f"('guess', {self.true_count})", "('stop', 'stop')"]
        else:
            raise ValueError("strategy must be 'mark_all' or 'guess_only'")

    def close(self):
        """Clean up resources to prevent memory leaks."""
        if hasattr(self, 'base_img') and self.base_img is not None:
            del self.base_img
            self.base_img = None
        
        if hasattr(self, 'target_anns'):
            self.target_anns.clear()
        
        if hasattr(self, 'centers'):
            self.centers.clear()
        
        if hasattr(self, 'dots'):
            self.dots.clear()
        
        # Clear samples data
        if hasattr(self, 'samples'):
            self.samples.clear()
        
        # Clear LVIS object if it exists
        if hasattr(self, 'lvis') and self.lvis is not None:
            del self.lvis
            self.lvis = None

    def _get_obs(self):
        img = self.base_img.copy()
        for x, y in self.dots:
            cv2.circle(img, (x, y), self.radius, (0, 0, 255), -1)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _get_info(self):
        return {
            "true_count": self.true_count,
            "current_guess": self.current_guess,
            "env_feedback": self.env_feedback
        }
    
    def _compute_reward(self):
        return 1.0 if (self.current_guess is not None and self.current_guess == self.true_count) else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _ann_to_mask(self, ann):
        segm = ann.get("segmentation", None)
        if segm is None:
            return None
        h, w = self.height, self.width
        if isinstance(segm, list):
            # polygons
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(segm, dict) and "counts" in segm:
            # RLE
            rle = segm
        else:
            return None
        return maskUtils.decode(rle)

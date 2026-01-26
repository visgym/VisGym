from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium.spaces import (
    Box,
    MultiDiscrete,
    Permutation,
    Text,
    FuncConditional,
)
import os
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import ast

# Constants
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
DEFAULT_IMAGE_MODE = "RGB"
VIEW_SIZE = 224  # Size of each zoomed view

class ZoomInPuzzleEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        sample_dir: str,
        *,
        seed: Optional[int] = None,
        zoom_gap: float = 1.0,
        zoom_std: float = 0.3,
        min_zoom_level: float = 2.0,
        num_zoom_views: int = 4,
        nested: bool = False,  # new ▶
    ) -> None:
        super().__init__()

        if not os.path.exists(sample_dir):
            raise ValueError(f"Sample directory {sample_dir} does not exist")

        self.sample_dir = sample_dir
        self.zoom_gap = zoom_gap
        self.zoom_std = zoom_std
        self.min_zoom_level = min_zoom_level
        self.num_zoom_views = num_zoom_views
        self.nested = nested  # ▶ remember nesting choice

        self.seed(seed)
        self.np_random = np.random.RandomState(seed)
        self.images = self._populate_images()
        if not self.images:
            raise ValueError(f"No images found in {sample_dir}")

        self.sample_sequence = self.np_random.permutation(len(self.images))
        self.sample_idx = -1

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(VIEW_SIZE, VIEW_SIZE * (self.num_zoom_views + 1), 3),
            dtype=np.uint8,
        )

        self.num_rows, self.num_cols = 1, self.num_zoom_views
        self.num_positions = self.num_rows * self.num_cols

        # Rich action space via FuncConditional ---------------------------
        self.action_space = FuncConditional(
            {
                "swap": MultiDiscrete([self.num_positions, self.num_positions], start=[1, 1]),
                "reorder": Permutation(self.num_positions, start=1),
                "stop": Text(4),  # e.g. "stop"
            }
        )

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            f"You are given an original image and {self.num_zoom_views} zoomed-in "
            "views laid out left→right. Your goal is to rearrange them so they are "
            "ordered from *least* to *most* zoomed."
        )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "swap" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'swap': Swap two zoom-tiles by specifying their 1-based positions. "
                "Format: `('swap', (i, j))` where i and j are 1-based positions."
            )
        if "reorder" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'reorder': Provide the complete ordering in one shot. "
                f"Format: `('reorder', [1, 2, ..., {self.num_zoom_views}])` where the list gives the desired left-to-right arrangement."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Finish and lock in the current ordering. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: Arrange the views from least to most zoomed (ascending zoom level order)."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "swap" in actions:
            examples.append("- To swap views: `('swap', (1, 3))`")
        if "reorder" in actions:
            examples.append(f"- To reorder all: `('reorder', [1, 3, 2, 4])`")
        if "stop" in actions:
            examples.append("- To finalize: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        # Get relative path to the current image (relative to sample_dir)
        current_image_path = self.images[self.sample_sequence[self.sample_idx]]
        relative_path = os.path.relpath(current_image_path, self.sample_dir)
        
        return {
            'image_name': relative_path,
            'zoom_levels': self.zoom_levels.copy(),
            'crop_boxes': self.crop_boxes.copy(),
            'current_view_order': self.current_view_order.copy(),
        }
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # if seed is not None:
        #     self.np_random.seed(seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            image_name = init_state['image_name']
            image_path = os.path.join(self.sample_dir, image_name)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            self.sample = Image.open(image_path).convert(DEFAULT_IMAGE_MODE)
            
            # Restore zoom levels and crop boxes
            self.zoom_levels = init_state['zoom_levels'].copy()
            self.crop_boxes = init_state['crop_boxes'].copy()
            
            # Rebuild zoomed views from saved crop boxes
            self._rebuild_zoomed_views_from_boxes()
            
            # Restore view order
            self.current_view_order = init_state['current_view_order'].copy()
            return self._get_obs(), {}
        
        # Normal reset: sequential sampling
        if self.sample_idx is None:
            self.sample_idx = 0
        else:
            self.sample_idx += 1
        if self.sample_idx >= len(self.sample_sequence):
            raise Exception("All samples exhausted")

        path = self.images[self.sample_sequence[self.sample_idx]]
        self.sample = Image.open(path).convert(DEFAULT_IMAGE_MODE)

        self._generate_zoomed_views()

        # Start with random ordering each episode
        self.current_view_order = list(range(self.num_zoom_views))
        self.np_random.shuffle(self.current_view_order)
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
        if branch == "swap":
            if not self.action_space["swap"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self._swap(payload)
        
        elif branch == "reorder":
            if not self.action_space["reorder"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self._reorder(payload)
        # customized action ends here

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

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self._get_obs()
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = 'reorder', num_steps: int = None):
        """
        Returns the action(s) that will lead to the groundtruth solution.
        In zoom ordering, this means arranging views from least to most zoomed.
        
        Args:
            strategy (str): 'reorder' or 'swap'.
                            'reorder' gives a single action to solve the puzzle.
                            'swap' gives a sequence of swap actions to solve the puzzle.
        
        Returns:
            A list of action strings.
        """
        if strategy == 'reorder':
            # Get the correct order: views should be arranged by ascending zoom levels
            true_order = np.argsort(self.zoom_levels)
            
            # Create the reorder permutation
            # reorder expects a list where position i should get content from position reorder[i]-1
            # We want position i to show true_order[i], so we need to find where true_order[i] is currently
            reorder_action = [0] * self.num_zoom_views
            for target_pos in range(self.num_zoom_views):
                desired_view = true_order[target_pos]  # Which original view should be at this position
                current_pos = self.current_view_order.index(desired_view)  # Where is it currently?
                reorder_action[target_pos] = current_pos + 1  # Convert to 1-based
            
            return [f"('reorder', {reorder_action})", "('stop','stop')"]

        elif strategy == 'swap':
            swaps = []
            target_order = list(np.argsort(self.zoom_levels))
            mutable_state = self.current_view_order.copy()
            view_to_pos = {view_idx: i for i, view_idx in enumerate(mutable_state)}

            for i in range(len(mutable_state)):
                correct_view = target_order[i]
                if mutable_state[i] != correct_view:
                    # Find where the correct view is currently located
                    j = view_to_pos[correct_view]
                    
                    # The view currently at position i
                    view_at_i = mutable_state[i]

                    # Perform the swap
                    mutable_state[i], mutable_state[j] = mutable_state[j], view_at_i
                    
                    # Update the position map for the two swapped views
                    view_to_pos[view_at_i] = j
                    view_to_pos[correct_view] = i

                    # Record the swap action (1-based indices)
                    swap_action = ('swap', (i + 1, j + 1))
                    swaps.append(str(swap_action))
            return swaps + ["('stop','stop')"]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies are 'reorder' and 'swap'.")

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        """
        Observation = original image (at slot 0) + N zoom tiles.

        The white-on-black numbers you see in each zoom slot (1…N)
        are redrawn here *every* time from their slot index, so they
        stay fixed even after swap / reorder actions.
        """
        # ── create base canvas ──────────────────────────────────────────
        original = self.sample.resize((VIEW_SIZE, VIEW_SIZE),
                                      Image.Resampling.LANCZOS)
        total_w  = VIEW_SIZE * (self.num_zoom_views + 1)
        canvas   = Image.new(DEFAULT_IMAGE_MODE, (total_w, VIEW_SIZE))
        canvas.paste(original, (0, 0))

        # We’ll use a single Draw object to overlay the slot numbers
        draw_canvas = ImageDraw.Draw(canvas)
        font        = self._pick_font(28)
        pad         = 10   # padding for the black background box

        # ── paste zoom tiles + slot labels ──────────────────────────────
        for disp_pos, src_idx in enumerate(self.current_view_order):
            x_offset = VIEW_SIZE * (disp_pos + 1)
            canvas.paste(self.zoomed_views[src_idx], (x_offset, 0))

            # slot index is 1-based
            slot_text = str(disp_pos + 1)
            tw, th    = draw_canvas.textbbox((0, 0), slot_text, font=font)[2:]
            tx        = x_offset + VIEW_SIZE - tw - pad
            ty        = VIEW_SIZE  - th - pad

            # black semi-transparent box behind the text
            draw_canvas.rectangle(
                (tx - pad, ty - pad, tx + tw + pad, ty + th + pad),
                fill=(0, 0, 0, 128)
            )
            draw_canvas.text((tx, ty), slot_text, fill=(255, 255, 255), font=font)

        return np.array(canvas, dtype=np.uint8)

    def _get_info(self) -> Dict[str, Any]:
        # Inverse mapping: orig idx → displayed pos (1‑based for convenience)
        inv = [0] * self.num_zoom_views
        for disp, orig in enumerate(self.current_view_order):
            inv[orig] = disp + 1
        return {
            "zoom_levels": self.zoom_levels,
            "crop_boxes": self.crop_boxes,
            "original_view_order": inv,
            "env_feedback": self.env_feedback
        }
    
    def _compute_reward(self) -> float:
        true_order = np.argsort(self.zoom_levels)
        guess = np.array([self.current_view_order.index(i) for i in range(self.num_zoom_views)])
        return 1.0 if np.array_equal(guess, true_order) else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _populate_images(self) -> List[str]:
        paths: List[str] = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            paths.extend(glob(os.path.join(self.sample_dir, f"**/*{ext}"), recursive=True))
        # Sort for determinism before seeded permutation
        return sorted(paths)

    def _rebuild_zoomed_views_from_boxes(self):
        """
        Rebuild zoomed_views from saved crop_boxes (used when loading from init_state).
        """
        self.zoomed_views: List[Image.Image] = []
        
        # Helper to crop + resize + label a region
        def _crop_and_label(box: Tuple[int, int, int, int], idx: int) -> Image.Image:
            crop = self.sample.crop(box).resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(crop)
            text = str(idx)
            font = self._pick_font(28)
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            pad = 10
            tx, ty = VIEW_SIZE - tw - pad, VIEW_SIZE - th - pad
            draw.rectangle((tx - pad, ty - pad, tx + tw + pad, ty + th + pad), fill=(0, 0, 0, 128))
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
            return crop
        
        for idx, box in enumerate(self.crop_boxes):
            self.zoomed_views.append(_crop_and_label(box, idx + 1))
    
    def _generate_zoomed_views(self):
        self.zoomed_views: List[Image.Image] = []
        self.zoom_levels: List[float] = []
        self.crop_boxes: List[Tuple[int, int, int, int]] = []  # (l, u, r, d)

        # 1. Determine zoom factors ------------------------------------------------
        raw_levels = []
        for i in range(self.num_zoom_views):
            μ = self.min_zoom_level + (i + 1) * self.zoom_gap
            z = max(self.np_random.normal(μ, self.zoom_std), self.min_zoom_level)
            raw_levels.append(z)
        self.zoom_levels = sorted(raw_levels)

        # 2. Produce crops ---------------------------------------------------------
        img_w, img_h = self.sample.size

        # Helper to crop + resize + label a region
        def _crop_and_label(box: Tuple[int, int, int, int], idx: int) -> Image.Image:
            crop = self.sample.crop(box).resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(crop)
            text = str(idx)
            font = self._pick_font(28)
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            pad = 10
            tx, ty = VIEW_SIZE - tw - pad, VIEW_SIZE - th - pad
            draw.rectangle((tx - pad, ty - pad, tx + tw + pad, ty + th + pad), fill=(0, 0, 0, 128))
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
            return crop

        if not self.nested:
            # Unnested – independent random centers as in original
            for z in self.zoom_levels:
                w, h = int(img_w / z), int(img_h / z)
                cx = self.np_random.randint(w // 2, img_w - w // 2)
                cy = self.np_random.randint(h // 2, img_h - h // 2)
                l, u = cx - w // 2, cy - h // 2
                box = (l, u, l + w, u + h)
                self.crop_boxes.append(box)
                self.zoomed_views.append(_crop_and_label(box, len(self.zoomed_views) + 1))
        else:
            # Nested – choose outer crop first then progressively zoom inwards
            #  → we treat zoom_levels sorted ascending so z[0] is least zoomed (largest crop)
            prev_box = (0, 0, img_w, img_h)
            for z in self.zoom_levels:
                prev_l, prev_u, prev_r, prev_d = prev_box
                prev_w, prev_h = prev_r - prev_l, prev_d - prev_u

                # compute new crop size
                w, h = int(img_w / z), int(img_h / z)
                w = min(w, prev_w)
                h = min(h, prev_h)

                # random position inside previous box such that crop fits entirely
                cx = self.np_random.randint(prev_l + w // 2, prev_r - w // 2 + 1)
                cy = self.np_random.randint(prev_u + h // 2, prev_d - h // 2 + 1)
                l, u = cx - w // 2, cy - h // 2
                box = (l, u, l + w, u + h)

                self.crop_boxes.append(box)
                self.zoomed_views.append(_crop_and_label(box, len(self.zoomed_views) + 1))
                prev_box = box  # next must reside inside this one

    def _swap(self, pair):
        if pair is None:
            raise ValueError("swap expects two indices – got None")
        i, j = pair
        i -= 1  # convert to 0‑based
        j -= 1
        for idx in (i, j):
            if not 0 <= idx < self.num_positions:
                raise ValueError(f"swap indices out of range: {pair}")
        self.current_view_order[i], self.current_view_order[j] = (
            self.current_view_order[j],
            self.current_view_order[i],
        )

    def _reorder(self, perm):
        if perm is None or len(perm) != self.num_positions:
            raise ValueError("reorder expects full permutation list")
        perm0 = [p - 1 for p in perm]  # to 0‑based positions
        if sorted(perm0) != list(range(self.num_positions)):
            raise ValueError("reorder payload is not a permutation of 1…N")
        self.current_view_order = [self.current_view_order[p] for p in perm0]

    def _pick_font(self, size: int) -> ImageFont.ImageFont:
        for name in ("Arial", "Helvetica", "DejaVuSans", "LiberationSans"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

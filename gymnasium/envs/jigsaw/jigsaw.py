import os
import re
import ast
from glob import glob
from typing import Optional, Dict
from copy import deepcopy

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from gymnasium.spaces import MultiDiscrete, Permutation, FuncConditional, Text

class JigsawEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, num_rows, num_cols, sample_dir, seed: Optional[int] = None, border_width: int = 1):
        super().__init__()
        # Validate inputs
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("num_rows and num_cols must be positive integers")
        
        if not os.path.isdir(sample_dir):
            raise ValueError(f"sample_dir {sample_dir} does not exist")

        # number of rows and columns in the puzzle
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.border_width = border_width

        # directory to sample puzzles from
        self.sample_dir = sample_dir
        
        # deterministic random number generator
        self.seed(seed)

        # get the images from the sample directory
        self.images = self._populate_images()
        if not self.images:
            raise ValueError(f"No valid images found in {sample_dir}")

        # sample sequence
        self.sample_sequence = self.np_random.permutation(len(self.images))
        self.sample_idx = None
        
        # Dictionary to store piece images {piece_id: image_piece}
        self.piece_dict = {}
        
        # Current and target states
        self.current_state = []  # Initialize as empty list
        self.target_state = []   # Initialize as empty list

        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.num_rows * self.num_cols - 1,
            shape=(self.num_rows * self.num_cols, ),
            dtype=np.int32
        )

        self.action_space = FuncConditional({
            "swap": MultiDiscrete(np.array([[self.num_rows, self.num_cols], [self.num_rows, self.num_cols]])),
            "reorder": Permutation(self.num_rows * self.num_cols),
            "stop": Text(4) # dummy text space, we just need the stop signal
        })

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_prompt(self, **kwargs) -> str:
        # generate the prompt here
        prompt = f"You are solving a {self.num_rows}x{self.num_cols} jigsaw puzzle. The puzzle pieces are currently scrambled. " \
            "Your goal is to rearrange the pieces to recover the image. " \
            "\n\nAvailable actions:\n"
        
        # Dynamically build the action list based on available actions
        action_descriptions = []
        actions = self.action_space.get_function_names()
        if "swap" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'swap': Swap two pieces by specifying their coordinates. " \
                "Format: `('swap', ((row1, col1), (row2, col2)))` where coordinates start from (0,0) at the top-left corner."
            )
        if "reorder" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'reorder': Reorder all pieces at once. " \
                f"Format: `('reorder', [0, 1, 2, ..., {self.num_rows * self.num_cols - 1}])` where the list represents the desired order of pieces from top-left to bottom-right."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the puzzle solving session. " \
                "Format: `('stop', 'stop')`"
            )
        
        # Add the action descriptions to the prompt
        prompt += "\n".join(action_descriptions)
        
        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. " \
            "For example:\n"
        
        examples = []
        if "swap" in actions:
            examples.append("- To swap two pieces: `('swap', ((0, 0), (1, 1)))`")
        if "reorder" in actions:
            examples.append(f"- To reorder all pieces: `('reorder', [0, 1, 2, ..., {self.num_rows * self.num_cols - 1}])`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        
        # Clarify how indices map to board cells so 0/1/2/3 are unambiguous
        prompt += (
            "\n\nIndex-to-cell mapping (0-based rows/cols):\n"
            f"- Index = row * {self.num_cols} + col.\n"
            f"- Row 0 covers indices 0..{self.num_cols - 1}, row 1 covers {self.num_cols}..{2*self.num_cols - 1}, etc.\n"
            "Example for 2x2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.\n"
        )
        return prompt
    
    # 1. get_init_state(): returns the init_state needed for _init_episode, image path / object path
    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        # Get the relative path of the image file from sample_dir
        image_path = self.images[self.sample_sequence[self.sample_idx]]
        # Store relative path to sample_dir, or just the filename if in the root
        image_name = os.path.relpath(image_path, self.sample_dir)
        
        return {
            'image_name': image_name,
            'current_state': self.current_state.copy(),
            'target_state': self.target_state.copy(),
        }
        
    def reset(self, *, init_state: Optional[dict] = None, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            image_name = init_state['image_name']
            self.current_state = init_state['current_state'].copy()
            self.target_state = init_state['target_state'].copy()
            
            # Construct the full path to the image
            image_path = os.path.join(self.sample_dir, image_name)
            
            # Verify the image exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file '{image_name}' not found in {self.sample_dir}")
            
            # Load the image
            self.sample = image_path
            loaded_image = Image.open(self.sample).convert('RGB')
            self.unshuffled_sample = deepcopy(loaded_image)
            
            # Rebuild the piece dictionary based on the saved states
            self._rebuild_pieces_from_state()
            
        else:
            # Normal reset: use sequential sampling
            if self.sample_idx is None:
                self.sample_idx = 0
            else:
                self.sample_idx += 1     
                if self.sample_idx >= len(self.sample_sequence):
                    raise Exception('All samples exhausted')
            self.sample = self.images[self.sample_sequence[self.sample_idx]]
            self.sample = Image.open(self.sample).convert('RGB')
            self.unshuffled_sample = deepcopy(self.sample)
            self._initialize_pieces()
            
        return self._get_obs(), {}

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
        if branch == "swap":
            if not self.action_space["swap"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                (row1, col1), (row2, col2) = payload
                pos1 = row1 * self.num_cols + col1
                pos2 = row2 * self.num_cols + col2
                self.current_state[pos1], self.current_state[pos2] = self.current_state[pos2], self.current_state[pos1]
        
        elif branch == "reorder":
            if not self.action_space["reorder"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self.current_state = [self.current_state[i] for i in payload]

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

    def solve(self, strategy: str = 'reorder', num_steps: int = None):
        """
        Returns the action(s) that will lead to the groundtruth solution.

        Args:
            strategy (str): 'reorder' or 'swap'.
            num_steps (int, optional): If provided for 'swap' strategy, ensures the
                                     solution has at least this many steps, adding
                                     no-op swaps if necessary.
        
        Returns:
            A list of action strings.
        """
        # Use a new, unseeded RNG for full randomness in every call
        local_rng = np.random.default_rng()

        # Check if the puzzle is already solved
        if self.current_state == self.target_state:
            return [str(('stop', 'stop'))]
        
        if strategy == 'reorder':
            current_state_inv = {piece_id: index for index, piece_id in enumerate(self.current_state)}
            payload = [current_state_inv[piece_id] for piece_id in self.target_state]
            action = ('reorder', payload)
            return [str(action), str(('stop', 'stop'))]
        
        elif strategy == 'swap':
            # --- Generate the minimal, randomized solution first ---
            swaps = []
            mutable_state = self.current_state.copy()
            piece_to_pos = {piece_id: i for i, piece_id in enumerate(mutable_state)}
            
            misplaced_indices = [i for i, piece in enumerate(mutable_state) if piece != self.target_state[i]]
            local_rng.shuffle(misplaced_indices)

            while misplaced_indices:
                i = misplaced_indices.pop(0)
                if mutable_state[i] == self.target_state[i]:
                    continue

                correct_piece = self.target_state[i]
                j = piece_to_pos[correct_piece]
                piece_at_i = mutable_state[i]

                mutable_state[i], mutable_state[j] = mutable_state[j], piece_at_i
                piece_to_pos[piece_at_i] = j
                piece_to_pos[correct_piece] = i

                pos1 = (i // self.num_cols, i % self.num_cols)
                pos2 = (j // self.num_cols, j % self.num_cols)
                swap_action = ('swap', (pos1, pos2))
                swaps.append(str(swap_action))

                if mutable_state[j] == self.target_state[j]:
                    if j in misplaced_indices:
                        misplaced_indices.remove(j)
            
            # --- Now, pad with "meaningful" (but net-zero effect) swaps if requested ---
            if num_steps is not None and num_steps > len(swaps):
                num_padding_needed = num_steps - len(swaps)
                # Each padding operation adds a pair of (action, undo_action), adding 2 steps.
                num_pairs_to_add = (num_padding_needed + 1) // 2
                
                for _ in range(num_pairs_to_add):
                    if len(self.current_state) < 2:
                        continue  # Not enough pieces to swap

                    # Pick two different random indices to form a meaningful swap
                    idx1, idx2 = local_rng.choice(len(self.current_state), 2, replace=False)
                    pos1 = (idx1 // self.num_cols, idx1 % self.num_cols)
                    pos2 = (idx2 // self.num_cols, idx2 % self.num_cols)
                    
                    meaningful_swap_action = str(('swap', (pos1, pos2)))
                    
                    # This pair consists of the action and its inverse (which is the same action).
                    # It must be inserted as a contiguous block to have a net-zero effect.
                    padding_pair = [meaningful_swap_action, meaningful_swap_action]
                    
                    # Insert this pair at a random location within the existing solution path
                    insert_pos = local_rng.integers(0, len(swaps) + 1)
                    swaps[insert_pos:insert_pos] = padding_pair

            swaps.append(str(('stop', 'stop')))
            return swaps
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies are 'reorder' and 'swap'.")

    def close(self):
        pass

    def _get_obs(self):
        image = self._create_image_from_state(self.current_state)
        return image

    def _get_info(self):
        proportion_matched = sum([1 for i, j in zip(self.current_state, self.target_state) \
                                  if i == j]) / len(self.current_state)
        return {
            'proportion_matched': proportion_matched,
            'current_state': self.current_state,
            'target_state': self.target_state,
            'unshuffled_image': self.unshuffled_sample,
            'env_feedback': self.env_feedback
        }
    
    def _compute_reward(self):
        return 1.0 if self.current_state == self.target_state else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _initialize_pieces(self):
        """
        Initializes the puzzle pieces.
        1. Generate a shuffled target state
        2. Create piece dictionary by mapping each number in the target state
           to patches extracted sequentially from the original image
        """
        img_w, img_h = self.unshuffled_sample.size
        
        # Calculate piece dimensions - allow for non-divisible sizes
        base_piece_w = img_w // self.num_cols
        base_piece_h = img_h // self.num_rows
        extra_w = img_w % self.num_cols
        extra_h = img_h % self.num_rows
        
        num_pieces = self.num_rows * self.num_cols
        
        # Generate the shuffled target state
        self.target_state = list(range(num_pieces))

        self.np_random.shuffle(self.target_state)
        # need to make sure that the shuffled target state is not the same as the initial state
        while self.target_state == list(range(num_pieces)):
            self.np_random.shuffle(self.target_state)
        
        # Set the current state as the ordered list
        self.current_state = list(range(num_pieces))
        
        # Clear any existing pieces
        self.piece_dict = {}
        
        # Extract all patches from the original image
        patches = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Calculate piece dimensions
                piece_w = base_piece_w + (1 if j < extra_w else 0)
                piece_h = base_piece_h + (1 if i < extra_h else 0)
                
                # Calculate coordinates
                left = sum(base_piece_w + (1 if x < extra_w else 0) for x in range(j))
                upper = sum(base_piece_h + (1 if y < extra_h else 0) for y in range(i))
                right = left + piece_w
                lower = upper + piece_h
                
                # Extract the patch
                patch = self.unshuffled_sample.crop((left, upper, right, lower))
                patches.append(patch)
        
        # Map each number in target_state to patches sequentially
        for idx, piece_id in enumerate(self.target_state):
            # Get the patch at sequential position idx
            patch = patches[idx]
            if self.border_width > 0:
                patch = ImageOps.expand(patch, border=self.border_width, fill='white')
            self.piece_dict[piece_id] = patch
    
    def _rebuild_pieces_from_state(self):
        """
        Rebuilds the puzzle pieces from saved state (used when loading from init_state).
        Uses the already-set current_state and target_state instead of generating new ones.
        """
        img_w, img_h = self.unshuffled_sample.size
        
        # Calculate piece dimensions - allow for non-divisible sizes
        base_piece_w = img_w // self.num_cols
        base_piece_h = img_h // self.num_rows
        extra_w = img_w % self.num_cols
        extra_h = img_h % self.num_rows
        
        # Clear any existing pieces
        self.piece_dict = {}
        
        # Extract all patches from the original image
        patches = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # Calculate piece dimensions
                piece_w = base_piece_w + (1 if j < extra_w else 0)
                piece_h = base_piece_h + (1 if i < extra_h else 0)
                
                # Calculate coordinates
                left = sum(base_piece_w + (1 if x < extra_w else 0) for x in range(j))
                upper = sum(base_piece_h + (1 if y < extra_h else 0) for y in range(i))
                right = left + piece_w
                lower = upper + piece_h
                
                # Extract the patch
                patch = self.unshuffled_sample.crop((left, upper, right, lower))
                patches.append(patch)
        
        # Map each number in target_state to patches sequentially
        # (same logic as _initialize_pieces but without shuffling)
        for idx, piece_id in enumerate(self.target_state):
            # Get the patch at sequential position idx
            patch = patches[idx]
            if self.border_width > 0:
                patch = ImageOps.expand(patch, border=self.border_width, fill='white')
            self.piece_dict[piece_id] = patch

    def _populate_images(self):
        images = sorted(glob(os.path.join(self.sample_dir, '*.png')) + \
                        glob(os.path.join(self.sample_dir, '*.jpg')) + \
                        glob(os.path.join(self.sample_dir, '*.jpeg')) + \
                        glob(os.path.join(self.sample_dir, '**/*.png')) + \
                        glob(os.path.join(self.sample_dir, '**/*.jpg')) + \
                        glob(os.path.join(self.sample_dir, '**/*.jpeg')))
        print(f"Found {len(images)} images in {self.sample_dir}")
        return images

    def _create_image_from_state(self, state):
        """
        Creates an image from a given state by pasting pieces according to the state.
        """
        img_w, img_h = self.unshuffled_sample.size
        
        new_img_w = img_w + self.num_cols * 2 * self.border_width
        new_img_h = img_h + self.num_rows * 2 * self.border_width
        new_image = Image.new('RGB', (new_img_w, new_img_h))
        
        # Calculate piece dimensions
        base_piece_w = img_w // self.num_cols
        base_piece_h = img_h // self.num_rows
        extra_w = img_w % self.num_cols
        extra_h = img_h % self.num_rows
        
        # Place each piece according to the state
        for pos in range(len(state)):
            piece_id = state[pos]  # Which piece should be at this position
            piece = self.piece_dict[piece_id]  # Get the piece from dictionary
            
            # Calculate target position
            row = pos // self.num_cols
            col = pos % self.num_cols
            
            # Calculate coordinates accounting for variable piece sizes
            left = sum(base_piece_w + (1 if x < extra_w else 0) for x in range(col))
            upper = sum(base_piece_h + (1 if y < extra_h else 0) for y in range(row))
            
            left += col * 2 * self.border_width
            upper += row * 2 * self.border_width
            
            # Paste the piece
            new_image.paste(piece, (left, upper))
        
        new_image = np.array(new_image)
        return new_image
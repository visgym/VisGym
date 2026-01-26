import gymnasium as gym
import json
import os
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import ast
from gymnasium.spaces import (
    MultiDiscrete,
    Permutation,
    Text,
    FuncConditional,
)

class VideoUnshuffleEnv(gym.Env, gym.VLMEnvMixin):
    """An environment for video unshuffling tasks using Something-Something V2 dataset."""
    
    def __init__(
        self,
        root_dir: str = "./partial_datasets/ssv2",
        num_frames: int = 4,
        seed: Optional[int] = None,
        sampling_strategy: str = "distinct",  # "uniform", "salient", or "distinct"
        min_frame_diff: float = 0.1,  # Minimum frame difference threshold, better to be larger than 0.1
        max_frames_to_analyze: int = 10000,  # Maximum number of frames to analyze for salient points, better to be larger than 100
        video_label: str = None, # e.g., 'with your camera', refer to the labels.json file for all possible labels
    ):
        """
        Initialize the environment.
        
        Args:
            root_dir: Root directory containing the dataset and labels
            num_frames: Number of frames to extract and shuffle
            seed: Random seed for reproducibility
            sampling_strategy: Strategy for frame sampling ("uniform", "salient", or "distinct")
            min_frame_diff: Minimum frame difference threshold for salient sampling
            max_frames_to_analyze: Maximum number of frames to analyze for salient points
        """
        super().__init__()
        
        self.root_dir = root_dir
        self.labels_path = os.path.join(root_dir, "labels", "train.json")
        self.videos_dir = os.path.join(root_dir, "20bn-something-something-v2")
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.min_frame_diff = min_frame_diff
        self.max_frames_to_analyze = max_frames_to_analyze
        self.video_label = video_label
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Load and process labels
        with open(self.labels_path, 'r') as f:
            self.original_labels = json.load(f)
        
        # Create shuffled mapping of video IDs to labels
        self.video_mapping: Dict[str, str] = {}
        self.pairs: List[Tuple[str, str]] = []
        self.selection_index: int = 0
        self._shuffle_videos()
        
        # Initialize state
        self.current_video_id: Optional[str] = None
        self.current_video_path: Optional[str] = None
        self.current_label: Optional[str] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.frame_positions: List[int] = []
        self.correct_order: List[int] = []
        self.current_observation: Optional[np.ndarray] = None
        self.env_feedback: str | None = None
        self.original_frames: List[np.ndarray] = []
        
        # Define observation and action spaces
        # Observation is a concatenated image of shuffled frames
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(224, 224 * num_frames, 3), dtype=np.uint8
        )
        # Action space is the permutation of frame indices
        self.action_space = FuncConditional(
            {
                "swap": MultiDiscrete([self.num_frames, self.num_frames], start=[1, 1]),
                "reorder": Permutation(self.num_frames, start=1),
                "stop": Text(4),  # e.g. "stop"
            }
        )
    
    def get_prompt(self, **kwargs) -> str:
        """
        Build an instruction prompt for the frame-ordering puzzle using the same
        structure as the zoom-ordering reference prompt.
        """
        # Intro sentence
        prompt = (
            f"You are given {self.num_frames} video frames extracted from a short "
            "clip, laid out left→right in a shuffled order and labeled 1 through "
            f"{self.num_frames}. The action being performed in the video is: "
            f"'{self.current_label}'.\n\n"
            "Your goal is to rearrange the frames so they appear in their original "
            "chronological order from left to right. Pay attention to the temporal "
            "progression of the action described above.\n\n"
            "Available actions:\n"
        )

        # Dynamically assemble the action list
        action_descriptions = []
        actions = self.action_space.get_function_names()

        if "swap" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'swap': Swap two frames by "
                "specifying their 1-based positions. "
                "Format: `('swap', (i, j))`"
            )

        if "reorder" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'reorder': Provide the complete "
                "ordering in one shot. "
                f"Format: `('reorder', [1, 2, ..., {self.num_frames}])` where "
                "the list gives the desired left-to-right arrangement."
            )

        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Finish and lock in the "
                "current ordering. Format: `('stop', 'stop')`"
            )

        prompt += "\n".join(action_descriptions)

        # Examples
        prompt += (
            "\n\nPlease respond with exactly one action and its arguments in the "
            "specified format. For example:\n"
        )

        examples = []
        if "swap" in actions:
            examples.append("- Swap the first and third frames: `('swap', (1, 3))`")
        if "reorder" in actions:
            examples.append(
                f"- Reorder all frames: `('reorder', [2, 1, 4, 3, ..., {self.num_frames}])`"
            )
        if "stop" in actions:
            examples.append("- Finalize the ordering: `('stop', 'stop')`")

        prompt += "\n".join(examples)
        return prompt

    def solve(self, strategy: str = "swap", num_steps: int = None):
        actions = []
        if strategy == "reorder":
            target_order = list(range(1, self.num_frames + 1))
            reorder_action = [0] * self.num_frames
            for target_pos_idx in range(self.num_frames):
                desired_frame_id = target_order[target_pos_idx]
                current_pos_idx = self.shuffled_order.index(desired_frame_id)
                reorder_action[target_pos_idx] = current_pos_idx + 1  # 1-based
            actions.append(f"('reorder', {reorder_action})")

        elif strategy == "swap":
            swaps = []
            target_order = list(range(1, self.num_frames + 1))
            mutable_state = self.shuffled_order.copy()
            frame_to_pos = {frame_id: i for i, frame_id in enumerate(mutable_state)}
            for i in range(len(mutable_state)):
                correct_frame_id = target_order[i]
                if mutable_state[i] != correct_frame_id:
                    j = frame_to_pos[correct_frame_id]
                    frame_at_i = mutable_state[i]
                    mutable_state[i], mutable_state[j] = mutable_state[j], frame_at_i
                    frame_to_pos[frame_at_i] = j
                    frame_to_pos[correct_frame_id] = i
                    swaps.append(str(('swap', (i + 1, j + 1))))
            actions.extend(swaps)

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Available: 'reorder', 'swap'.")

        # Always finish with a stop so the episode terminates cleanly
        actions.append("('stop', 'stop')")
        return actions

    def _shuffle_videos(self):
        """Shuffle the video-label mapping."""
        # Create a list of (id, label) pairs
        pairs = [(item['id'], item['label']) for item in self.original_labels]
        if self.video_label is not None:
            pairs = [pair for pair in pairs if self.video_label in str(pair[1])]
            print(f"after filter {len(pairs)}")
        # Sort by id for determinism before seeded shuffle
        pairs.sort(key=lambda x: str(x[0]))
        self.np_random.shuffle(pairs)
        
        # Create new mapping
        self.pairs = pairs
        self.video_mapping = {id: label for id, label in pairs}

    def _uniform_positions(self, total_frames: int) -> List[int]:
        """Evenly spaced positions strictly inside [0, total_frames-1]."""
        limit = min(total_frames, self.max_frames_to_analyze)
        if limit <= self.num_frames:
            # fall back to first N valid frames
            return list(range(max(0, limit - self.num_frames), limit))
        # use linspace and drop endpoints
        idxs = np.linspace(0, limit - 1, self.num_frames + 2, dtype=int)[1:-1]
        # ensure strictly increasing & unique
        idxs = np.unique(idxs)
        # pad if de-dup shrank the set
        while len(idxs) < self.num_frames:
            extra = np.random.randint(0, limit)
            if extra not in idxs:
                idxs = np.sort(np.append(idxs, extra))
        return idxs.tolist()

    def _find_salient_frames(self) -> List[int]:
        """Find frames with significant visual changes; fall back to uniform if needed."""
        if self.video_capture is None:
            return []
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_analyze = min(total_frames, self.max_frames_to_analyze)
        if frames_to_analyze <= 0:
            return []

        # seed with frame 0
        selected = [0]
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, ref = self.video_capture.read()
        if not ret:
            return self._uniform_positions(total_frames)
        ref = cv2.cvtColor(cv2.resize(ref, (224, 224)), cv2.COLOR_BGR2RGB)

        for i in range(1, frames_to_analyze):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            if self._compute_frame_difference(ref, frame) >= self.min_frame_diff:
                selected.append(i)
                ref = frame
                if len(selected) >= self.num_frames:
                    break

        if len(selected) < self.num_frames:
            return self._uniform_positions(total_frames)
        return sorted(selected)

    def _find_most_distinct_frames(self) -> List[int]:
        """
        Find the 4 most visually distinct frames by computing pairwise differences
        between all frames and selecting the subset that maximizes total distinctness.
        """
        if self.video_capture is None:
            return []
        
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_analyze = min(total_frames, self.max_frames_to_analyze)
        if frames_to_analyze <= self.num_frames:
            return self._uniform_positions(total_frames)

        # Extract and preprocess all frames
        frames_data = []
        for i in range(frames_to_analyze):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            frames_data.append((i, frame))
        
        if len(frames_data) < self.num_frames:
            return self._uniform_positions(total_frames)

        # Compute pairwise differences between all frames
        num_frames = len(frames_data)
        differences = np.zeros((num_frames, num_frames))
        
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                diff = self._compute_frame_difference(frames_data[i][1], frames_data[j][1])
                differences[i][j] = diff
                differences[j][i] = diff

        # Find the subset of frames that maximizes total distinctness
        # Use a greedy approach: start with the frame pair with highest difference,
        # then iteratively add frames that maximize the minimum distance to selected frames
        
        # Find the two most different frames to start
        max_diff = 0
        start_pair = (0, 1)
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                if differences[i][j] > max_diff:
                    max_diff = differences[i][j]
                    start_pair = (i, j)
        
        selected_indices = set(start_pair)
        
        # Greedily add remaining frames
        while len(selected_indices) < self.num_frames and len(selected_indices) < num_frames:
            best_candidate = None
            best_min_distance = -1
            
            for candidate in range(num_frames):
                if candidate in selected_indices:
                    continue
                
                # Find minimum distance from candidate to any selected frame
                min_distance = float('inf')
                for selected in selected_indices:
                    min_distance = min(min_distance, differences[candidate][selected])
                
                # Choose candidate that maximizes minimum distance
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_indices.add(best_candidate)
            else:
                break

        # Convert back to frame positions and sort
        selected_frames = [frames_data[i][0] for i in selected_indices]
        return sorted(selected_frames)

    def _extract_frames(self) -> List[np.ndarray]:
        if self.video_capture is None:
            return []

        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.sampling_strategy == "uniform":
            self.frame_positions = self._uniform_positions(total_frames)
        elif self.sampling_strategy == "salient":
            self.frame_positions = self._find_salient_frames()
        elif self.sampling_strategy == "distinct":
            self.frame_positions = self._find_most_distinct_frames()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}. "
                           "Available options: 'uniform', 'salient', 'distinct'")

        # hard guarantee: exactly num_frames indices, all in-range ints
        self.frame_positions = [int(max(0, min(p, total_frames - 1))) for p in self.frame_positions]
        if len(self.frame_positions) != self.num_frames:
            return []

        frames = []
        for pos in self.frame_positions:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = self.video_capture.read()
            if not ret:
                return []  # abort this video; caller will pick the next one
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames
    
    def _extract_frames_at_positions(self, positions: List[int]) -> List[np.ndarray]:
        """Extract frames at specific positions (used when loading from init_state)."""
        if self.video_capture is None:
            return []
        
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Validate positions
        positions = [int(max(0, min(p, total_frames - 1))) for p in positions]
        
        frames = []
        for pos in positions:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = self.video_capture.read()
            if not ret:
                return []
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames
    
    def _create_observation_from_order(self, frames: List[np.ndarray], order: List[int]) -> np.ndarray:
        """Create observation from frames using a given order (used when loading from init_state)."""
        if len(frames) < self.num_frames:
            raise RuntimeError(f"Need {self.num_frames} frames, got {len(frames)}")
        
        if not self._valid_order(order):
            raise RuntimeError(f"Invalid order: {order}")
        
        # Build shuffled frames based on the given order
        shuffled_frames = [frames[fid - 1] for fid in order]
        numbered_frames = self._add_frame_numbers(shuffled_frames)
        concatenated = np.concatenate(numbered_frames, axis=1)
        return concatenated

    def _add_frame_numbers(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Add frame numbers to each frame."""
        numbered_frames = []
        for i, frame in enumerate(frames):
            # Create a copy to avoid modifying the original
            frame = frame.copy()
            
            # Add semi-transparent background for text
            text = str(i + 1)  # 1-based indexing
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calculate text position (bottom right corner with padding)
            padding = 5
            x = frame.shape[1] - text_width - padding
            y = frame.shape[0] - padding
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + padding),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw text
            cv2.putText(
                frame,
                text,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            numbered_frames.append(frame)
        
        return numbered_frames
        

    # 1) tiny util
    def _valid_order(self, order: List[int]) -> bool:
        return isinstance(order, list) and len(order) == self.num_frames and all(1 <= x <= self.num_frames for x in order)

    # 2) fix color conversion (your frames are RGB by construction)
    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff) / 255.0)

    # 3) keep _uniform_positions/_find_salient_frames/_extract_frames as you have them
    #    (they're okay now), but make _create_shuffled_observation and _update_current_observation
    #    defensive against bad orders.

    def _create_shuffled_observation(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, List[int]]:
        if len(frames) < self.num_frames:
            raise RuntimeError(f"Need {self.num_frames} frames, got {len(frames)}")

        self.original_order = list(range(1, self.num_frames + 1))
        self.shuffled_order = self.original_order.copy()
        self.np_random.shuffle(self.shuffled_order)

        if not self._valid_order(self.shuffled_order):
            raise RuntimeError(f"Invalid shuffled_order: {self.shuffled_order}")

        # correct_order maps frame-id → 1-based position after shuffle
        self.correct_order = [0] * self.num_frames
        for i, fid in enumerate(self.shuffled_order):
            if not (1 <= fid <= self.num_frames):
                raise RuntimeError(f"Frame id out of range: {fid}")
            self.correct_order[fid - 1] = i + 1

        try:
            shuffled_frames = [frames[fid - 1] for fid in self.shuffled_order]
        except IndexError as e:
            raise RuntimeError(f"IndexError building shuffled frames: order={self.shuffled_order}, "
                            f"len(frames)={len(frames)}") from e

        numbered_frames = self._add_frame_numbers(shuffled_frames)
        concatenated = np.concatenate(numbered_frames, axis=1)
        return concatenated, self.shuffled_order

    def _update_current_observation(self) -> None:
        if not self._valid_order(self.shuffled_order):
            # regenerate a trivial identity order rather than crash
            self.shuffled_order = list(range(1, self.num_frames + 1))
        try:
            frames = [self.original_frames[fid - 1] for fid in self.shuffled_order]
        except IndexError:
            # fallback to identity if something went wrong
            self.shuffled_order = list(range(1, self.num_frames + 1))
            frames = [self.original_frames[fid - 1] for fid in self.shuffled_order]
        numbered = self._add_frame_numbers(frames)
        self.current_observation = np.concatenate(numbered, axis=1)

    # 4) make reset() skip any problematic video cleanly (no IndexError escapes)
    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'video_id': self.current_video_id,
            'label': self.current_label,
            'shuffled_order': self.shuffled_order.copy(),
            'frame_positions': self.frame_positions.copy(),
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, init_state: Optional[Dict] = None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        if self.video_capture is not None:
            self.video_capture.release()

        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            vid = init_state['video_id']
            lbl = init_state['label']
            
            self.current_video_id, self.current_label = vid, lbl
            self.current_video_path = os.path.join(self.videos_dir, f"{vid}.webm")
            
            self.video_capture = cv2.VideoCapture(self.current_video_path)
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Could not open video: {self.current_video_path}")
            
            # Use the saved frame positions to extract the same frames
            self.frame_positions = init_state['frame_positions'].copy()
            frames = self._extract_frames_at_positions(self.frame_positions)
            
            if len(frames) != self.num_frames:
                raise RuntimeError(f"Expected {self.num_frames} frames but got {len(frames)}")
            
            self.original_frames = frames
            # Use the saved shuffled order instead of generating a new one
            self.shuffled_order = init_state['shuffled_order'].copy()
            self.correct_order = list(range(1, self.num_frames + 1))
            
            # Rebuild the observation with the saved shuffled order
            self.current_observation = self._create_observation_from_order(frames, self.shuffled_order)
            
            info = {
                "video_id": self.current_video_id,
                "label": self.current_label,
                "shuffled_order": self.shuffled_order,
                "correct_order": self.correct_order,
                "env_feedback": self.env_feedback,
            }
            return self.current_observation, info
        
        # Normal reset: sequential sampling
        tries = 0
        max_tries = 100  # don't loop forever if dataset is broken
        while tries < max_tries:
            tries += 1
            if not self.pairs:  # sanity guard
                raise RuntimeError("No videos available after filtering.")

            # pick next video round-robin
            try:
                vid, lbl = self.pairs[self.selection_index]
            except IndexError:
                # defend against any weird concurrent mutation
                self.selection_index = 0
                continue

            self.current_video_id, self.current_label = vid, lbl
            self.current_video_path = os.path.join(self.videos_dir, f"{vid}.webm")
            self.selection_index = (self.selection_index + 1) % len(self.pairs)

            self.video_capture = cv2.VideoCapture(self.current_video_path)
            if not self.video_capture.isOpened():
                continue  # try next video

            frames = self._extract_frames()
            if len(frames) != self.num_frames:
                continue

            try:
                self.original_frames = frames
                self.current_observation, self.shuffled_order = self._create_shuffled_observation(frames)
            except Exception as e:
                # log minimal context and move on
                print(f"[WARN] skipping video {vid}: {e}")
                continue

            # success
            info = {
                "video_id": self.current_video_id,
                "label": self.current_label,
                "shuffled_order": self.shuffled_order,
                "correct_order": self.correct_order,
                "env_feedback": self.env_feedback,
            }
            return self.current_observation, info

        # If we get here, we failed to find any usable video
        raise RuntimeError("Could not load a usable video after many attempts.")


    # -----------------------------------------------------------------------------#
    # Public API
    # -----------------------------------------------------------------------------#
    def step(
        self, action: str
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
                self.env_feedback = (
                    f"Swap payload {payload} is invalid. It must contain two distinct "
                    f"1-based indices between 1 and {self.num_frames}."
                )
            else:
                self._swap(payload)

        elif branch == "reorder":
            if not self.action_space["reorder"].contains(payload):
                self.env_feedback = (
                    f"Reorder payload {payload} is invalid. It must be a permutation "
                    f"of [1, {self.num_frames}]."
                )
            else:
                self._reorder(payload)

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


    # -----------------------------------------------------------------------------#
    # Private helpers
    # -----------------------------------------------------------------------------#
    # _swap()  – update list **and** image
    def _swap(self, indices: Tuple[int, int]) -> None:
        """Swap two 1-based positions in the current ordering and refresh view."""
        i, j = (idx - 1 for idx in indices)             # 0-based
        self.shuffled_order[i], self.shuffled_order[j] = \
            self.shuffled_order[j], self.shuffled_order[i]
        self._update_current_observation()              # ②  refresh
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    def _reorder(self, ordering: List[int]) -> None:
        """
        Payload is a permutation of positions 1..N.
        Convert it to the corresponding frame-ID list, then refresh the view.
        """
        # Validate length once (FuncConditional already checks permutation, but
        # this keeps the function usable outside the action_space guard).
        if len(ordering) != self.num_frames:
            raise ValueError("Payload length must equal num_frames")

        # Translate: take the i-th requested position → fetch that frame ID
        # from the *current* shuffled_order.
        self.shuffled_order = [self.shuffled_order[pos - 1] for pos in ordering]

        # Re-render the observation so the agent sees the change
        self._update_current_observation()
    # ─────────────────────────────────────────────────────────────────────────────


    def _compute_reward(self) -> float:
        current_order = self.shuffled_order
        target_order = list(range(1, self.num_frames + 1))  # [1, 2, 3, 4]
        return 1.0 if current_order == target_order else 0.0

    def _get_obs(self) -> np.ndarray:
        """Return the current observation (single-step, so unchanged)."""
        return self.current_observation

    def _get_info(self) -> Dict[str, Any]:
        """Collect diagnostic information for the caller."""
        return {
            "video_id": self.current_video_id,
            "label": self.current_label,
            "shuffled_order": self.shuffled_order,
            "correct_order": self.correct_order,
            "tau": self._compute_reward(),
            "env_feedback": self.env_feedback,
        }

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.current_observation
        else:
            raise NotImplementedError
        
    def close(self):
        """Clean up resources."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

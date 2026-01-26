import random
from copy import deepcopy
from typing import List, Set, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import (
    Box,
    Discrete,
    MultiDiscrete,
    Text,        
)
from gymnasium.spaces import FuncConditional    

from PIL import Image, ImageDraw
import ast


class PatchReassemblyEnv(gym.Env, gym.VLMEnvMixin):
    """
    A toy “jigsaw” environment.

    • Action branches (FuncConditional):
        – {'place':  [patch_id, row, col]}
        – {'remove': patch_id}
        – {'stop' :  'stop'}        (optional stop signal)

    • Each patch has an *anchor* cell chosen at generation time.
      A “place” action snaps that anchor onto the chosen grid coordinate.

    • “remove” returns the patch to its own pre-allocated parking slot
      outside the grid (no overlaps with labels).
    """

    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(self,
                 grid_size: Tuple[int, int] = (5, 5),
                 num_patches: int = 5,
                 render_mode: str = "rgb_array",
                 seed: int | None = None):
        super().__init__()

        self.render_mode = render_mode

        self.grid_rows, self.grid_cols = grid_size
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.seed(seed)
        self.random = random.Random(seed)

        # Colours & shapes
        self.colors = self._generate_colors(self.num_patches)
        self.patches, self.patch_anchor = self._generate_irregular_patches()

        # Parking slots are fixed for the whole episode
        self.parking_slot = self._make_parking_slots()

        # State = current anchor position of every patch
        self.current_state: List[Tuple[int, int]] = []

        # ----------------------- action space ---------------------------
        self.action_space = FuncConditional({
            "place": MultiDiscrete(
                np.array([self.num_patches, self.grid_rows, self.grid_cols],
                         dtype=np.int32)
            ),
            "remove": Discrete(self.num_patches),
            "stop":   Text(4)            # dummy “quit” branch
        })

        # ---------------------- observation space -----------------------
        if self.render_mode == "ansi":
            self.observation_space = Text(max_length=10000)
        else:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(1024, 1024, 3),
                dtype=np.uint8,
            )

        self.reset() #TODO: ideally we should remove this in init, but this might break the test case reproducibility

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            f"You are solving a {self.grid_rows}×{self.grid_cols} patch-reassembly puzzle. "
            f"There are {self.num_patches} irregular pieces parked around the board. "
            "Your goal is to place them so the grid is completely filled—no gaps, no overlaps."
        )
        if self.render_mode == 'ansi':
            prompt += ("\nYou are given a text-based representation of the grid. Patches on the board are shown with their ID number, and empty cells are marked with '.'. "
                       "Parked patches are displayed below the grid as small ASCII shapes. The anchor cell for each parked patch is marked with a '*' instead of its ID number.")
        else: # rgb_array
            prompt += "\nYou see an image of the board and the parked patches."

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "place" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'place': Snap a patch onto the grid by aligning its "
                "anchor (the cell that shows the patch's ID number) with a chosen board coordinate. "
                f"Format: `('place', (patch_id, row, col))` where patch_id ∈ [0, {self.num_patches-1}], "
                f"row ∈ [0, {self.grid_rows-1}], col ∈ [0, {self.grid_cols-1}]."
            )
        if "remove" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'remove': Return a patch to its parking slot. "
                f"Format: `('remove', patch_id)` where patch_id ∈ [0, {self.num_patches-1}]."
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the episode and submit your solution. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed if the grid is completely filled with no gaps or overlaps."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "place" in actions:
            examples.append("- To place patch 0 at (2, 3): `('place', (0, 2, 3))`")
        if "remove" in actions:
            examples.append("- To remove patch 0: `('remove', 0)`")
        if "stop" in actions:
            examples.append("- To finish: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        # Convert patches (list of sets) to list of lists for JSON serialization
        patches_serializable = [sorted(list(patch)) for patch in self.patches]
        
        return {
            'patches': patches_serializable,
            'patch_anchor': self.patch_anchor.copy(),
            'parking_slot': self.parking_slot.copy(),
            'current_state': self.current_state.copy(),
        }
    
    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        """
        Start a fresh episode with brand-new irregular patches.

        • If `seed` is given, the new set is repeatable.
        • Otherwise the RNG state continues from the previous episode.
        • If `init_state` is given, restore the exact patch configuration.
        """
        # handle seeding ----------------------------------------------------
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.random = random.Random(seed)

        if init_state is not None:
            self.patches = [set(tuple(cell) for cell in patch) for patch in init_state['patches']]
            self.patch_anchor = init_state['patch_anchor'].copy()
            self.parking_slot = init_state['parking_slot'].copy()
            self.current_state = init_state['current_state'].copy()
        else:
            self.patches, self.patch_anchor = self._generate_irregular_patches()
            self.parking_slot = self._make_parking_slots()
            self.current_state = list(self.parking_slot)

        self._freeze_canvas()

        self.done = False
        self.env_feedback = None
        return self._get_obs(), self._get_info()

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
        if branch == "place":
            if not self.action_space["place"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self._place_patch(payload)

        elif branch == "remove":
            if not self.action_space["remove"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self._remove_patch(payload)

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
    
    def render(self, mode: str | None = None):
        """
        Draw the environment using the *pre-computed* canvas bounds
        (_min_r/_max_r/_min_c/_max_c set in reset).  Size is constant
        for every call until the next reset.
        """
        if mode is None:
            mode = self.render_mode

        if mode == 'ansi':
            return self._render_ansi()
        elif mode == 'rgb_array':
            cell      = 40                   # pixels per logical cell
            margin    = 1                    # blank ring around everything
            black     = (0, 0, 0)
            grey      = (200, 200, 200)
            coord_clr = (0, 120, 255)        # cool blue for row/col numbers

            # -------------------- fixed extents -----------------------------
            Hcells = (self._max_r - self._min_r + 1) + 2 * margin
            Wcells = (self._max_c - self._min_c + 1) + 2 * margin
            off_r  = margin - self._min_r            # world→canvas row shift
            off_c  = margin - self._min_c            # world→canvas col shift

            Hpx, Wpx = Hcells * cell, Wcells * cell
            img  = Image.new("RGB", (Wpx, Hpx), (255, 255, 255))
            draw = ImageDraw.Draw(img)

            # -------------------- grid lines --------------------------------
            gx0, gy0 = (off_c + 0) * cell, (off_r + 0) * cell
            gW,  gH  = self.grid_cols * cell, self.grid_rows * cell

            for r in range(self.grid_rows + 1):
                y = gy0 + r * cell
                draw.line([(gx0, y), (gx0 + gW, y)], fill=grey, width=1)
            for c in range(self.grid_cols + 1):
                x = gx0 + c * cell
                draw.line([(x, gy0), (x, gy0 + gH)], fill=grey, width=1)

            # -------------------- coordinate numbers ------------------------
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("DejaVuSans.ttf", int(cell * 0.8))
            except Exception:
                font = None

            # columns (above grid)
            for c in range(self.grid_cols):
                cx = gx0 + c * cell + cell // 2
                cy = gy0 - cell // 2
                draw.text((cx, cy), str(c), anchor="mm", fill=coord_clr, font=font)

            # rows (left of grid)
            for r in range(self.grid_rows):
                cx = gx0 - cell // 2
                cy = gy0 + r * cell + cell // 2
                draw.text((cx, cy), str(r), anchor="mm", fill=coord_clr, font=font)

            # -------------------- draw every patch --------------------------
            for pid, anchor in enumerate(self.current_state):
                cells  = self._cells_at(pid, anchor)
                color  = self.colors[pid]
                bright = sum(color) / 3
                txt_clr = (0, 0, 0) if bright > 128 else (255, 255, 255)

                # cells
                for r, c in cells:
                    px = (off_c + c) * cell
                    py = (off_r + r) * cell
                    draw.rectangle([px, py, px + cell, py + cell],
                                fill=color, outline=black, width=1)

                # patch id on its anchor
                ar, ac = anchor
                px = (off_c + ac) * cell + cell // 2
                py = (off_r + ar) * cell + cell // 2
                draw.text((px, py), str(pid), anchor="mm", fill=txt_clr, font=font)

            return np.asarray(img, dtype=np.uint8)
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = None, num_steps: int = None):
        """
        Returns a sequence of actions that will solve the patch reassembly puzzle.
        Uses a backtracking algorithm to find a valid solution and can add randomized,
        "mistake-and-correct" steps to meet a minimum step count.
        
        Args:
            num_steps (int, optional): If provided, the returned action sequence will be
                                     padded with extra steps to have at least this
                                     many 'place' actions.

        Returns:
            List[str]: Sequence of actions to solve the puzzle.
        """
        # If already solved, just stop
        if self._is_solved():
            return ["('stop', 'stop')"]
        
        # 1. Get the shortest, deterministic solution first
        solution = self._solve_puzzle()
        
        if not solution:
            # If no solution found, return a heuristic move
            return self._solve_heuristic()
        # 2. Pad with steps if needed, using a robust hybrid strategy
        def _count_place(actions: list[str]) -> int:
            cnt = 0
            for s in actions:
                try:
                    br, _ = ast.literal_eval(s)
                    if br == 'place':
                        cnt += 1
                except Exception:
                    continue
            return cnt

        if num_steps is not None and num_steps < 0:
            num_steps = 0

        max_iters = 5_000  # safety cap against infinite loops
        iters = 0
        while num_steps is not None and _count_place(solution) < num_steps and iters < max_iters:
            iters += 1
            
            # --- Strategy 1: "Mistake-and-Correct" (adds 1 step) ---
            # Find all possible mistake opportunities in the current solution
            mistake_opportunities = []
            for i, action_str in enumerate(solution):
                try:
                    br, payload = ast.literal_eval(action_str)
                except Exception:
                    continue
                if br == 'place':
                    # This is the state of the board right before this action would be executed
                    pre_state = self._get_state_after_actions(solution[:i])

                    try:
                        pid, correct_r, correct_c = payload
                    except Exception:
                        continue
                    correct_pos = (correct_r, correct_c)

                    # Find valid but wrong places for this piece in the current pre_state
                    for r in range(self.grid_rows):
                        for c in range(self.grid_cols):
                            pos = (r, c)
                            if pos != correct_pos and self._is_valid_placement_against_state(pid, pos, pre_state):
                                mistake_opportunities.append({'idx': i, 'pid': pid, 'wrong_pos': pos, 'correct_action': action_str})

            if mistake_opportunities:
                # Apply a random valid mistake
                mistake = self.random.choice(mistake_opportunities)
                idx = mistake['idx']
                pid = mistake['pid']
                wrong_pos = mistake['wrong_pos']
                correct_action = mistake['correct_action']
                
                place_wrong_action = f"('place', ({pid}, {wrong_pos[0]}, {wrong_pos[1]}))"
                
                # Rebuild the solution list with the mistake-then-correct sequence
                solution = solution[:idx] + [place_wrong_action, correct_action] + solution[idx+1:]
                continue # Restart the while loop to re-evaluate length and find new opportunities

            # --- Strategy 2: Fallback "Remove-and-Replace" (adds 2 steps) ---
            # This is guaranteed to be valid and will always increase the step count
            place_action_indices = []
            for i, s in enumerate(solution):
                try:
                    br, _ = ast.literal_eval(s)
                except Exception:
                    continue
                if br == 'place':
                    place_action_indices.append(i)
            if not place_action_indices:
                break # No more 'place' actions to pad, cannot continue

            idx_to_pad_after = self.random.choice(place_action_indices)
            action_to_dupe = solution[idx_to_pad_after]
            try:
                _, payload = ast.literal_eval(action_to_dupe)
            except Exception:
                break
            pid, _, _ = payload

            remove_action = f"('remove', {pid})"
            
            # Insert the remove/place pair immediately after the original place
            new_solution = solution[:idx_to_pad_after+1] + [remove_action, action_to_dupe] + solution[idx_to_pad_after+1:]
            solution = new_solution
            # Loop continues

    # 3. Add the final stop action
        solution.append("('stop', 'stop')")
        return solution
    
    def close(self):
        pass

    def _get_grid(self):
        grid = np.full((self.grid_rows, self.grid_cols), -1, dtype=np.int32)
        for pid, anchor in enumerate(self.current_state):
            if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
                for r, c in self._cells_at(pid, anchor):
                    grid[r, c] = pid
        return grid

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        filled = np.sum(self._get_grid() != -1)
        return {
            "proportion_filled": filled / (self.grid_rows * self.grid_cols),
            "current_state": deepcopy(self.current_state),
            "env_feedback": self.env_feedback
        }
    
    def _compute_reward(self) -> float:
        return 1.0 if self._is_solved() else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _get_state_after_actions(self, actions: list[str]) -> list[tuple[int, int]]:
        """
        Simulates the execution of place/remove actions to determine the final board state.
        """
        # Start from the current state of the environment
        simulated_state = deepcopy(self.current_state)
        
        for action_str in actions:
            try:
                branch, payload = ast.literal_eval(action_str)
                if branch == 'place':
                    pid, r, c = payload
                    simulated_state[pid] = (r, c)
                elif branch == 'remove':
                    pid = payload
                    simulated_state[pid] = self.parking_slot[pid]
            except (ValueError, SyntaxError):
                continue # Ignore malformed actions like ('stop', 'stop')
                
        return simulated_state

    def _solve_puzzle(self):
        """
        Use backtracking to solve the puzzle completely.
        Returns a list of actions that solve the puzzle, or None if no solution exists.
        """
        # Save current state
        original_state = deepcopy(self.current_state)
        
        # Find patches that are not on the board
        unplaced_patches = []
        for pid in range(self.num_patches):
            anchor = self.current_state[pid]
            if not (0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols):
                unplaced_patches.append(pid)
        
        # If all patches are placed, check if solved
        if not unplaced_patches:
            if self._is_solved():
                return []
            else:
                # Need to rearrange - use heuristic for now
                return self._solve_heuristic()
        
        # Try to place patches using backtracking
        solution = []
        if self._backtrack_solve(unplaced_patches, 0, solution):
            # Restore original state
            self.current_state = original_state
            return solution
        
        # Restore original state if no solution found
        self.current_state = original_state
        return None
    
    def _backtrack_solve(self, unplaced_patches, patch_idx, solution):
        """
        Backtracking algorithm to place patches.
        """
        # Base case: all patches placed
        if patch_idx >= len(unplaced_patches):
            return self._is_solved()
        
        pid = unplaced_patches[patch_idx]
        
        # Try all possible positions for this patch
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self._is_valid_placement(pid, (row, col)):
                    # Place the patch
                    original_pos = self.current_state[pid]
                    self.current_state[pid] = (row, col)
                    solution.append(f"('place', ({pid}, {row}, {col}))")
                    
                    # Recursively try to place remaining patches
                    if self._backtrack_solve(unplaced_patches, patch_idx + 1, solution):
                        return True
                    
                    # Backtrack
                    self.current_state[pid] = original_pos
                    solution.pop()
        
        return False
    
    def _solve_heuristic(self):
        """
        Fallback heuristic when complete solution is not immediately found.
        Returns the next best move to make progress.
        """
        # Find a patch that's not on the board
        for pid in range(self.num_patches):
            anchor = self.current_state[pid]
            if not (0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols):
                # Try to find a valid position for this patch
                for row in range(self.grid_rows):
                    for col in range(self.grid_cols):
                        if self._is_valid_placement(pid, (row, col)):
                            return [f"('place', ({pid}, {row}, {col}))"]
        
        # If all patches are on board but puzzle not solved, try to rearrange
        # Remove a patch that might be blocking the solution
        for pid in range(self.num_patches):
            anchor = self.current_state[pid]
            if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
                # Check if removing this patch would allow better placement
                original_pos = self.current_state[pid]
                self.current_state[pid] = self.parking_slot[pid]
                
                # Try to find a better position
                for row in range(self.grid_rows):
                    for col in range(self.grid_cols):
                        if self._is_valid_placement(pid, (row, col)):
                            # Test if this leads to a better state
                            self.current_state[pid] = (row, col)
                            if self._count_filled_cells() > self._count_filled_cells_at_state(original_pos, pid):
                                # Restore and return the move sequence
                                self.current_state[pid] = original_pos
                                return [f"('remove', {pid})", f"('place', ({pid}, {row}, {col}))"]
                            self.current_state[pid] = self.parking_slot[pid]
                
                # Restore original position
                self.current_state[pid] = original_pos
        
        # If nothing else works, just stop
        return ["('stop', 'stop')"]
    
    def _count_filled_cells(self):
        """Count how many grid cells are currently filled."""
        grid = self._get_grid()
        return np.sum(grid != -1)
    
    def _count_filled_cells_at_state(self, test_pos, test_pid):
        """Count filled cells if test_pid were at test_pos."""
        count = 0
        for pid in range(self.num_patches):
            if pid == test_pid:
                anchor = test_pos
            else:
                anchor = self.current_state[pid]
            
            if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
                cells = self._cells_at(pid, anchor)
                for r, c in cells:
                    if 0 <= r < self.grid_rows and 0 <= c < self.grid_cols:
                        count += 1
        return count

    def _freeze_canvas(self):
        """
        Compute canonical min/max row/col for the *whole episode*:

        • include every grid cell;
        • include every patch in its parking slot;
        • that superset is enough to cover any later board position.
        """
        all_cells = {(r, c)
                    for r in range(self.grid_rows)
                    for c in range(self.grid_cols)}          # the board itself
        for pid, park_anchor in enumerate(self.parking_slot):  # parked patches
            all_cells.update(self._cells_at(pid, park_anchor))

        rs, cs  = zip(*all_cells)
        self._min_r, self._max_r = min(rs), max(rs)
        self._min_c, self._max_c = min(cs), max(cs)



    def _render_ansi(self):
        """
        Renders the environment to an ANSI string.
        - The main grid is shown with coordinates.
        - Patches on the grid are represented by their IDs.
        - Parked patches are listed below, each with a small ASCII art representation of its shape.
          The anchor cell of each parked patch is marked with a '*'.
        """
        R, C = self.grid_rows, self.grid_cols
        
        # Create the grid from observation
        obs_grid = self._get_grid()
        
        # Build the string representation
        output = []
        
        # Column headers
        header = "   " + " ".join(f"{c:<2}" for c in range(C))
        output.append(header)
        output.append("  " + "-" * (C * 3))
        
        # Grid rows
        for r in range(R):
            row_str = f"{r:<2}|"
            for c in range(C):
                patch_id = obs_grid[r, c]
                cell = f" {patch_id:<2}" if patch_id != -1 else " . "
                row_str += cell
            output.append(row_str)
        
        output.append("\n--- Parked Patches ---")
        
        # List and draw parked patches
        has_parked = False
        for pid, anchor in enumerate(self.current_state):
            if not (0 <= anchor[0] < R and 0 <= anchor[1] < C):
                has_parked = True
                output.append(f"\nPatch {pid}:")
                
                # Get relative coordinates to draw the shape
                anchor_r, anchor_c = self.patch_anchor[pid]
                rel_coords = {(r - anchor_r, c - anchor_c) for r, c in self.patches[pid]}
                
                # Get bounding box of the relative shape
                min_r = min(r for r, c in rel_coords)
                max_r = max(r for r, c in rel_coords)
                min_c = min(c for r, c in rel_coords)
                max_c = max(c for r, c in rel_coords)
                
                # Create a small canvas for the patch shape
                shape_height = max_r - min_r + 1
                shape_width = max_c - min_c + 1
                patch_canvas = [[' ' for _ in range(shape_width)] for _ in range(shape_height)]
                
                for r_rel, c_rel in rel_coords:
                    # The anchor's relative coordinate is (0,0)
                    char = '*' if (r_rel, c_rel) == (0, 0) else str(pid)
                    patch_canvas[r_rel - min_r][c_rel - min_c] = char
                
                # Add the shape to the output
                for row in patch_canvas:
                    output.append("  " + "".join(row))

        if not has_parked:
            output.append("  (None)")
            
        return "\n".join(output)


    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        rng = np.random.default_rng(42)
        return [tuple(rng.integers(50, 230, size=3)) for _ in range(n)]

    def _generate_irregular_patches(
        self,
    ) -> Tuple[List[Set[Tuple[int, int]]], List[Tuple[int, int]]]:
        """Partition the *R × C* grid into **exactly** ``self.num_patches``
        contiguous regions that together cover the board with no gaps or
        overlaps.

        The *anchor* for each patch is chosen as ``min(shape)`` under normal
        tuple ordering, which is the genuine top‑left cell of that patch.
        """
        R, C, P = self.grid_rows, self.grid_cols, self.num_patches
        assert P <= R * C, "More patches than cells!"

        # 1) choose ``P`` distinct seed cells to start the multi‑source BFS
        all_cells = [(r, c) for r in range(R) for c in range(C)]
        seed_cells = self.random.sample(all_cells, P)

        # 2) initialise assignment grid  (-1 = unclaimed)
        owner = np.full((R, C), fill_value=-1, dtype=int)

        # 3) multi‑source BFS that grows every patch until the board is full
        from collections import deque
        frontiers = [deque([a]) for a in seed_cells]  # one queue per patch

        for pid, (r, c) in enumerate(seed_cells):
            owner[r, c] = pid

        remaining = R * C - P  # unassigned cell counter

        nbr = lambda r, c: [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

        while remaining:
            for pid in range(P):  # round‑robin growth for irregularity
                if not remaining:
                    break
                if not frontiers[pid]:
                    continue

                r, c = frontiers[pid].popleft()
                for nr, nc in nbr(r, c):
                    if 0 <= nr < R and 0 <= nc < C and owner[nr, nc] == -1:
                        owner[nr, nc] = pid
                        frontiers[pid].append((nr, nc))
                        remaining -= 1
                        if remaining == 0:
                            break

        # 4) collect absolute cell sets for every patch ------------------
        patches: List[Set[Tuple[int, int]]] = [
            set(map(tuple, np.argwhere(owner == pid))) for pid in range(P)
        ]

        # 5) anchor = **lexicographic min** cell in each patch ------------
        anchors: List[Tuple[int, int]] = [min(shape) for shape in patches]

        return patches, anchors

    def _make_parking_slots(self) -> List[Tuple[int, int]]:
        """
        Anchor positions so that an entire patch sits ≥2 cells above the grid
        if parked on the *top* edge, ≥2 cells left of the grid on the *left*
        edge, and ≥1 cell away on the right / bottom edges.
        """
        R, C = self.grid_rows, self.grid_cols
        slots, cursors = [], {"top": 0, "right": 0, "bottom": 0, "left": 0}
        sides = ["top", "right", "bottom", "left"]

        GAP_REG   = 1    # normal clearance (bottom / right)
        GAP_LABEL = 2    # clearance for edges that have labels (top / left)

        # offset ranges of each patch relative to its anchor
        def rel_bounds(pid):
            ar, ac = self.patch_anchor[pid]
            rs = [r - ar for r, _ in self.patches[pid]]
            cs = [c - ac for _, c in self.patches[pid]]
            return min(rs), max(rs), min(cs), max(cs)   # min_r, max_r, min_c, max_c

        for pid in range(self.num_patches):
            min_dr, max_dr, min_dc, max_dc = rel_bounds(pid)
            h = max_dr - min_dr + 1
            w = max_dc - min_dc + 1
            side = sides[pid % 4]

            if side == "top":                              # ↑
                anchor_r = -(GAP_LABEL + max_dr)
                anchor_c = cursors["top"] - min_dc
                cursors["top"] += w + GAP_LABEL

            elif side == "right":                          # →
                anchor_c = C + GAP_REG - min_dc
                anchor_r = cursors["right"] - min_dr
                cursors["right"] += h + GAP_REG

            elif side == "bottom":                         # ↓
                anchor_r = R + GAP_REG - min_dr
                anchor_c = cursors["bottom"] - min_dc
                cursors["bottom"] += w + GAP_REG

            else:  # "left"                                # ←
                anchor_c = -(GAP_LABEL + max_dc)
                anchor_r = cursors["left"] - min_dr
                cursors["left"] += h + GAP_LABEL

            slots.append((anchor_r, anchor_c))

        return slots

    def _cells_at(self, pid: int, anchor_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        World-space cells occupied by PATCH pid whose anchor sits at anchor_pos.
        """
        ar, ac = self.patch_anchor[pid]
        dr, dc = anchor_pos[0] - ar, anchor_pos[1] - ac
        return {(r + dr, c + dc) for (r, c) in self.patches[pid]}

    def _is_valid_placement(self, pid: int, target: Tuple[int, int]) -> bool:
        cells = self._cells_at(pid, target)

        # inside grid?
        for r, c in cells:
            if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
                return False

        # no overlap with in-board patches
        for other, anchor in enumerate(self.current_state):
            if other == pid:
                continue
            if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
                if cells & self._cells_at(other, anchor):
                    return False
        return True

    def _is_valid_placement_against_state(self, pid: int, target: Tuple[int, int], state: List[Tuple[int, int]]) -> bool:
        """Checks if placing a patch is valid against a given hypothetical state."""
        cells = self._cells_at(pid, target)

        # inside grid?
        for r, c in cells:
            if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
                return False

        # no overlap with in-board patches in the given state
        for other, anchor in enumerate(state):
            if other == pid:
                continue
            if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
                if cells & self._cells_at(other, anchor):
                    return False
        return True

    def _is_solved(self) -> bool:
        grid = np.full((self.grid_rows, self.grid_cols), -1)
        for pid, anchor in enumerate(self.current_state):
            if not (0 <= anchor[0] < self.grid_rows
                    and 0 <= anchor[1] < self.grid_cols):
                return False
            for r, c in self._cells_at(pid, anchor):
                grid[r, c] = pid
        return -1 not in grid

    def _place_patch(self, payload: Tuple[int, int, int]):
        pid, row, col = payload
        if self._is_valid_placement(pid, (row, col)):
            self.current_state[pid] = (row, col)
        else:
            self.env_feedback = f"The action is not valid because the patch {pid} cannot be placed at ({row}, {col})."

    def _remove_patch(self, payload: int):
        anchor = self.current_state[payload]
        if 0 <= anchor[0] < self.grid_rows and 0 <= anchor[1] < self.grid_cols:
            self.current_state[payload] = self.parking_slot[payload]
        else:
            self.env_feedback = f"The action is not valid because the patch {payload} is already in the parking slot."

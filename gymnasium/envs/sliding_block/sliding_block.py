import ast
import copy
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, FuncConditional, MultiDiscrete, Text

class SlidingBlockEnv(gym.Env, gym.VLMEnvMixin):
    """
    A sliding block puzzle environment based on Klotski.
    The goal is to slide rectangular blocks within a framed board to reach a target configuration.
    """
    
    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        num_shuffle_moves: int = 0,  # Number of random moves to shuffle the board
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the sliding block puzzle environment.
        
        Args:
            num_shuffle_moves: Number of random moves to shuffle the board
            seed: Random seed for reproducibility
            render_mode: Render mode for the environment
        """
        super().__init__()
        
        # Fixed board size for Klotski
        self.board_size = (5, 4)
        self.rows, self.cols = self.board_size
        
        # Number of random moves to shuffle the board
        self.num_shuffle_moves = num_shuffle_moves
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Classic Klotski configuration matching the provided image
        self.puzzle_config = {
            "blocks": [
                {"id": 1, "size": (2, 2), "color": "red", "is_target": True},  
                {"id": 2, "size": (1, 2), "color": "green"},                   
                {"id": 3, "size": (2, 1), "color": "blue"},                    
                {"id": 4, "size": (2, 1), "color": "blue"},                  
                {"id": 5, "size": (2, 1), "color": "blue"}, 
                {"id": 6, "size": (2, 1), "color": "blue"}, 
                {"id": 7, "size": (1, 1), "color": "yellow"},
                {"id": 8, "size": (1, 1), "color": "yellow"},
                {"id": 9, "size": (1, 1), "color": "yellow"},
                {"id": 10, "size": (1, 1), "color": "yellow"},
            ],
            "initial_positions": [
                {"id": 1, "position": (0, 1)},   # Red 2x2 at (0,1)
                {"id": 2, "position": (2, 1)},   # Green 2x1 at (2,1)
                {"id": 3, "position": (0, 0)},   # Blue 2x1 at (0,0)
                {"id": 4, "position": (0, 3)},   # Blue 2x1 at (0,3)
                {"id": 5, "position": (2, 0)},   # Blue 2x1 at (2,0)
                {"id": 6, "position": (2, 3)},   # Blue 2x1 at (2,3)
                {"id": 7, "position": (4, 0)},   # Yellow 1x1 at (4,0)
                {"id": 8, "position": (4, 3)},   # Yellow 1x1 at (4,3)
                {"id": 9, "position": (3, 1)},   # Yellow 1x1 at (3,1)
                {"id": 10, "position": (3, 2)},   # Yellow 1x1 at (3,2)
            ]
        }
        
        self.num_to_dir = {
            0: "up",
            1: "right",
            2: "down",
            3: "left"
        }
        # Initialize state
        self.blocks = {}  # Dictionary to store block information
        self.board = np.zeros(self.board_size, dtype=np.int32)  # Board state
        self.original_board = None  # Store the original board state
        self.original_blocks = None  # Store the original block positions
        self.moves = 0
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=0,
            high=len(self.puzzle_config["blocks"]),
            shape=self.board_size,
            dtype=np.int32
        )

        self.action_space = FuncConditional({
            "move": MultiDiscrete([
                len(self.puzzle_config["blocks"]) + 1,  # block_id
                4  # direction
            ]),
            "stop": Text(4) # dummy text space, we just need the stop signal
        })
        
        # Initialize the environment
        self.reset() #TODO: ideally we should remove this in init, but this might break the test case reproducibility
        
        self.render_mode = render_mode

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
            'original_blocks': copy.deepcopy(self.original_blocks),
            'target_blocks': copy.deepcopy(self.target_blocks),
            'current_blocks': copy.deepcopy(self.blocks),
        }
    
    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = (
            f"You are solving a {self.rows}x{self.cols} sliding block puzzle (Klotski). "
            "The puzzle consists of rectangular blocks that can be moved in four directions. "
            "Your goal is to rearrange the blocks to match the target configuration."
        )

        if self.render_mode == 'ansi':
            prompt += (
                "\nYou are given a text-based representation of the puzzle. "
                "You see two boards side-by-side:\n"
                "- Left: the target configuration\n"
                "- Right: the current configuration that you need to rearrange\n"
                "Each number on the board represents a block, and '.' represents an empty space."
            )
        else:  # Default to rgb_array description
            prompt += (
                "\nYou see two boards side-by-side:\n"
                "- Left: the target configuration\n"
                "- Right: the current configuration that you need to rearrange\n"
            )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Move a block in one of four directions. "
                "Format: `('move', (block_id, direction))` where:\n"
                "   - block_id is the number of the block to move (1-10)\n"
                "   - direction is 0=up, 1=right, 2=down, 3=left"
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the puzzle solving session. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed if all blocks are in their target positions when you stop."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move block 1 up: `('move', (1, 0))`")
            examples.append("- To move block 2 right: `('move', (2, 1))`")
            examples.append("- To move block 3 down: `('move', (3, 2))`")
            examples.append("- To move block 4 left: `('move', (4, 3))`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, init_state: Optional[Dict] = None):
        """Reset the environment to its initial state and shuffle it with improved strategy."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            self.original_blocks = copy.deepcopy(init_state['original_blocks'])
            self.target_blocks = copy.deepcopy(init_state['target_blocks'])
            self.blocks = copy.deepcopy(init_state['current_blocks'])
            
            # Rebuild the original_board from original_blocks
            self.board = np.zeros(self.board_size, dtype=np.int32)
            self.original_board = np.zeros(self.board_size, dtype=np.int32)
            for block_id, block in self.original_blocks.items():
                h, w = block["size"]
                r, c = block["position"]
                self.original_board[r:r+h, c:c+w] = block_id
                self.board[r:r+h, c:c+w] = block_id
            
            # Rebuild the target_board from target_blocks
            self.target_board = np.zeros(self.board_size, dtype=np.int32)
            for block_id, block in self.target_blocks.items():
                h, w = block["size"]
                r, c = block["position"]
                self.target_board[r:r+h, c:c+w] = block_id
            
            self.moves = 0
            
            return self.board.copy(), {
                "moves": self.moves,
                "blocks": self.blocks,
                "original_board": self.original_board,
                "target_board": self.target_board
            }
        
        # Normal reset: generate new puzzle
        # Initialize blocks in their original (solved) positions
        self._initialize_blocks()
        
        # Store the solved state
        self.original_board = self.board.copy()
        self.original_blocks = copy.deepcopy(self.blocks)
        
        # Improved shuffling strategy
        self._improved_shuffle()
        
        self.target_board = self.board.copy()
        self.target_blocks = copy.deepcopy(self.blocks)
        
        # Restore the board and blocks to the solved state for the agent to start from
        self.board = self.original_board.copy()
        for block_id, block in self.blocks.items():
            block["position"] = self.original_blocks[block_id]["position"]
        
        self.moves = 0
        
        return self.board.copy(), {
            "moves": self.moves,
            "blocks": self.blocks,
            "original_board": self.original_board,
            "target_board": self.target_board
        }

    
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
        if branch == "move":
            if not self.action_space["move"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                block_id, direction = payload
                moved = self._move_block(block_id, direction)
                if not moved:
                    self.env_feedback = f"The action {payload} is not valid because the block {block_id} cannot move in the direction {self.num_to_dir[direction]}."
                else:
                    self.moves += 1
        
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
        if mode == 'ansi':
            def draw_board_ansi(board, blocks):
                grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
                for r in range(self.rows):
                    for c in range(self.cols):
                        block_id = board[r, c]
                        if block_id != 0:
                            if block_id == 10:
                                grid[r][c] = '0'
                            else:
                                grid[r][c] = str(block_id)
                return ["".join(row) for row in grid]

            target_lines = draw_board_ansi(self.target_board, self.target_blocks)
            current_lines = draw_board_ansi(self.board, self.blocks)

            # Combine them side-by-side
            header = "Target".center(self.cols) + "   " + "Current".center(self.cols)
            combined_lines = [header, "-" * (self.cols * 2 + 3)]
            for i in range(self.rows):
                combined_lines.append(f"{target_lines[i]} | {current_lines[i]}")

            return "\n".join(combined_lines)
        elif mode == 'rgb_array':
            cell_size = 50
            gap = 20  # pixels between boards
            title_height = 40  # pixels for the title area
            colors = {
                0: (255, 255, 255),  # Empty space
                1: (255, 0, 0),      # Red (target block)
                2: (0, 255, 0),      # Green
                3: (0, 0, 255),      # Blue
                4: (0, 0, 255),      # Blue
                5: (0, 0, 255),      # Blue
                6: (0, 0, 255),      # Blue
                7: (255, 255, 0),    # Yellow
                8: (255, 255, 0),    # Yellow
                9: (255, 255, 0),    # Yellow
                10: (255, 255, 0),   # Yellow
            }
            def draw_board(board, blocks):
                img = np.zeros((self.rows * cell_size, self.cols * cell_size, 3), dtype=np.uint8)
                for i in range(self.rows):
                    for j in range(self.cols):
                        block_id = board[i, j]
                        color = colors[block_id]
                        y1, y2 = i * cell_size, (i + 1) * cell_size
                        x1, x2 = j * cell_size, (j + 1) * cell_size
                        img[y1:y2, x1:x2] = color
                for block_id, block in blocks.items():
                    if block["position"] is None:
                        continue
                    h, w = block["size"]
                    row, col = block["position"]
                    y1, y2 = row * cell_size, (row + h) * cell_size
                    x1, x2 = col * cell_size, (col + w) * cell_size
                    cv2.rectangle(img, (x1, y1), (x2-1, y2-1), (0, 0, 0), thickness=2)
                    text = str(block_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = x1 + (x2 - x1 - text_width) // 2
                    text_y = y1 + (y2 - y1 + text_height) // 2
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
                return img
            # Draw both boards
            left = draw_board(self.target_board, self.target_blocks)
            right = draw_board(self.board, self.blocks)
            # Add titles above each board
            board_h, board_w, _ = left.shape
            total_w = 2 * board_w + gap
            total_h = board_h + title_height
            both = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
            # Place left board
            both[title_height:title_height+board_h, 0:board_w] = left
            # Place right board
            both[title_height:title_height+board_h, board_w+gap:board_w+gap+board_w] = right
            # Draw titles
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # slightly larger font
            thickness = 2
            # Target board title
            text1 = "target board"
            (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            text_x1 = (board_w - text_width1) // 2
            text_y1 = (title_height + text_height1) // 2
            cv2.putText(both, text1, (text_x1, text_y1), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            # Current board title
            text2 = "current board"
            (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            text_x2 = board_w + gap + (board_w - text_width2) // 2
            text_y2 = (title_height + text_height2) // 2
            cv2.putText(both, text2, (text_x2, text_y2), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            return both
        else:
            raise NotImplementedError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = "bfs", num_steps: int = None):
        """
        Finds the shortest sequence of moves to solve the puzzle using Breadth-First Search.
        If num_steps is provided, pads the sequence with dummy back-and-forth moves
        that leave the state unchanged, similar to toy_maze2d.
        
        Args:
            num_steps (int, optional): Minimum number of steps to return.
        
        Returns:
            list[str]: Sequence of move action strings to reach target (optionally padded).
        """
        def get_state_key(blocks_dict):
            """Gets a hashable representation of the blocks' positions."""
            return tuple(sorted((k, v['position']) for k, v in blocks_dict.items()))

        initial_blocks = copy.deepcopy(self.blocks)
        initial_board = self.board.copy()
        target_key = get_state_key(self.target_blocks)

        if get_state_key(initial_blocks) == target_key:
            # Already solved; optionally add dummy pairs to meet num_steps
            base_moves: List[Tuple[int, int]] = []
            if num_steps is not None and num_steps > 0:
                # Build opportunities from current state
                def collect_valid_moves(blocks_dict, board_arr):
                    val = []
                    for bid, blk in blocks_dict.items():
                        h, w = blk['size']
                        r, c = blk['position']
                        for direction in range(4):
                            dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
                            new_r, new_c = r + dr, c + dc
                            if not (0 <= new_r <= self.rows - h and 0 <= new_c <= self.cols - w):
                                continue
                            tmp = board_arr.copy()
                            tmp[r:r+h, c:c+w] = 0
                            if np.all(tmp[new_r:new_r+h, new_c:new_c+w] == 0):
                                val.append((bid, direction))
                    return val
                candidates = collect_valid_moves(initial_blocks, initial_board)
                # If none, return empty list
                if candidates:
                    pairs_needed = (num_steps + 1) // 2
                    idxs = list(self.np_random.randint(0, len(candidates), size=pairs_needed))
                    for i in idxs:
                        bid, direction = candidates[i]
                        base_moves.append((bid, direction))
                        base_moves.append((bid, (direction + 2) % 4))
            actions = [f"('move', {action})" for action in base_moves[:num_steps or 0]]
            actions.append("('stop', 'stop')")
            return actions

        # Queue for BFS: (blocks_dict, board_array, path_of_actions)
        queue = [(initial_blocks, initial_board, [])]
        visited = {get_state_key(initial_blocks)}

        solution_moves: Optional[List[Tuple[int, int]]] = None
        while queue and solution_moves is None:
            current_blocks, current_board, path = queue.pop(0)

            # Try moving each block in each direction
            for block_id in current_blocks:
                block = current_blocks[block_id]
                h, w = block['size']
                r, c = block['position']
                
                for direction in range(4):  # 0:up, 1:right, 2:down, 3:left
                    dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
                    new_r, new_c = r + dr, c + dc
                    
                    # 1. Check boundaries
                    if not (0 <= new_r <= self.rows - h and 0 <= new_c <= self.cols - w):
                        continue

                    # 2. Check for collision by temporarily removing the block
                    temp_board = current_board.copy()
                    temp_board[r:r+h, c:c+w] = 0  # Clear current position
                    
                    is_valid_move = np.all(temp_board[new_r:new_r+h, new_c:new_c+w] == 0)
                    
                    if not is_valid_move:
                        continue
                        
                    # 3. If move is valid, create the next state
                    move = (block_id, direction)
                    
                    next_blocks = copy.deepcopy(current_blocks)
                    next_blocks[block_id]['position'] = (new_r, new_c)
                    
                    next_key = get_state_key(next_blocks)
                    if next_key not in visited:
                        new_path = path + [move]
                        if next_key == target_key:
                            # We found the solution path
                            solution_moves = new_path
                            break
                        
                        visited.add(next_key)
                        
                        # Create the new board for the next state in the queue
                        next_board = temp_board
                        next_board[new_r:new_r+h, new_c:new_c+w] = block_id
                        
                        queue.append((next_blocks, next_board, new_path))
            # continue BFS if not found yet
        
        if solution_moves is None:
            return ["('stop', 'stop')"]  # No solution found; ensure termination

        # Optional padding with dummy back-and-forth moves
        if num_steps is not None and len(solution_moves) < num_steps:
            # Reconstruct intermediate states to find valid dummy opportunities
            def apply_move(blocks_dict, board_arr, move):
                bid, direction = move
                blk = blocks_dict[bid]
                h, w = blk['size']
                r, c = blk['position']
                dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
                new_r, new_c = r + dr, c + dc
                # create deep copies
                nb = copy.deepcopy(blocks_dict)
                b2 = board_arr.copy()
                # clear and place
                b2[r:r+h, c:c+w] = 0
                b2[new_r:new_r+h, new_c:new_c+w] = bid
                nb[bid]['position'] = (new_r, new_c)
                return nb, b2

            # Build list of states before each action index
            states: List[Tuple[Dict[int, dict], np.ndarray]] = []
            s_blocks = copy.deepcopy(initial_blocks)
            s_board = initial_board.copy()
            states.append((s_blocks, s_board))
            for mv in solution_moves:
                s_blocks, s_board = apply_move(s_blocks, s_board, mv)
                states.append((s_blocks, s_board))  # state after move

            # For each state, collect valid moves (that can be immediately reversed)
            padding_opportunities = []  # {insert_at, move}
            for insert_at, (blk_dict, brd) in enumerate(states):
                for bid, blk in blk_dict.items():
                    h, w = blk['size']
                    r, c = blk['position']
                    for direction in range(4):
                        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
                        new_r, new_c = r + dr, c + dc
                        if not (0 <= new_r <= self.rows - h and 0 <= new_c <= self.cols - w):
                            continue
                        tmp = brd.copy()
                        tmp[r:r+h, c:c+w] = 0
                        if np.all(tmp[new_r:new_r+h, new_c:new_c+w] == 0):
                            padding_opportunities.append({'insert_at': insert_at, 'move': (bid, direction)})

            # Fallback: use solution moves as opportunities if none were found
            if not padding_opportunities and solution_moves:
                for i, mv in enumerate(solution_moves):
                    padding_opportunities.append({'insert_at': i, 'move': mv})

            pairs_needed = (num_steps - len(solution_moves) + 1) // 2
            if padding_opportunities and pairs_needed > 0:
                idxs = list(self.np_random.randint(0, len(padding_opportunities), size=pairs_needed))
                pads = [padding_opportunities[i] for i in idxs]
                pads.sort(key=lambda p: p['insert_at'], reverse=True)
                moves = list(solution_moves)
                for pad in pads:
                    idx = pad['insert_at']
                    bid, direction = pad['move']
                    rev = (bid, (direction + 2) % 4)
                    # Insert pair: move then reverse
                    moves.insert(idx, rev)
                    moves.insert(idx, (bid, direction))
                solution_moves = moves

        actions = [f"('move', {action})" for action in solution_moves]
        actions.append("('stop', 'stop')")
        return actions

    def close(self):
        pass

    def _get_obs(self):
        return self.render()
    
    def _get_info(self):
        return {"moves": self.moves, "blocks": self.blocks, "env_feedback": self.env_feedback}
    
    def _compute_reward(self) -> bool:
        for block_id, block in self.blocks.items():
            target_block = self.target_blocks[block_id]
            if block["position"] != target_block["position"]:
                return 0.0
        return 1.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _initialize_blocks(self):
        """Initialize blocks with their positions and properties."""
        self.blocks = {}
        self.board = np.zeros(self.board_size, dtype=np.int32)
        for block_config in self.puzzle_config["blocks"]:
            block_id = block_config["id"]
            self.blocks[block_id] = {
                "size": block_config["size"],
                "color": block_config["color"],
                "is_target": block_config.get("is_target", False),
                "position": None
            }
        for pos_config in self.puzzle_config["initial_positions"]:
            block_id = pos_config["id"]
            position = pos_config["position"]
            self._place_block(block_id, position, is_initial_placement=True)
    
    def _place_block(self, block_id: int, position: Tuple[int, int], is_initial_placement: bool = False):
        """Place a block on the board at the specified position."""
        block = self.blocks[block_id]
        height, width = block["size"]
        row, col = position
        
        for i in range(height):
            for j in range(width):
                self.board[row + i, col + j] = block_id
        
        # Update block position
        block["position"] = position
    
    def _is_valid_placement(self, block_id: int, position: Tuple[int, int], is_initial_placement: bool = False) -> bool:
        """Check if a block can be placed at the specified position."""
        block = self.blocks[block_id]
        height, width = block["size"]
        row, col = position
        
        # Check board boundaries
        if row < 0 or col < 0 or row + height > self.rows or col + width > self.cols:
            return False
        
        # During initial placement, we only check board boundaries
        if is_initial_placement:
            return True
        
        # Check for overlapping blocks
        for i in range(height):
            for j in range(width):
                current_block = self.board[row + i, col + j]
                if current_block != 0 and current_block != block_id:
                    return False
        
        return True
    
    def _can_move(self, block_id: int, direction: int) -> bool:
        """Check if a block can move in the specified direction."""
        if block_id not in self.blocks:
            return False
            
        block = self.blocks[block_id]
        height, width = block["size"]
        row, col = block["position"]
        
        # Calculate new position
        if direction == 0:  # up
            new_pos = (row - 1, col)
        elif direction == 1:  # right
            new_pos = (row, col + 1)
        elif direction == 2:  # down
            new_pos = (row + 1, col)
        else:  # left
            new_pos = (row, col - 1)
        
        return self._is_valid_placement(block_id, new_pos)
    
    def _move_block(self, block_id: int, direction: int) -> bool:
        """Move a block in the specified direction if possible."""
        if not self._can_move(block_id, direction):
            return False
            
        block = self.blocks[block_id]
        height, width = block["size"]
        row, col = block["position"]
        
        # Clear current position
        for i in range(height):
            for j in range(width):
                self.board[row + i, col + j] = 0
        
        # Calculate and set new position
        if direction == 0:  # up
            new_pos = (row - 1, col)
        elif direction == 1:  # right
            new_pos = (row, col + 1)
        elif direction == 2:  # down
            new_pos = (row + 1, col)
        else:  # left
            new_pos = (row, col - 1)
        
        self._place_block(block_id, new_pos)
        return True
    
    def _get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves in the current state."""
        valid_moves = []
        for block_id in self.blocks:
            for direction in range(4):
                if self._can_move(block_id, direction):
                    valid_moves.append((block_id, direction))
        return valid_moves

    def _improved_shuffle(self):
        """Improved shuffling strategy that ensures significant block movement."""
        if self.num_shuffle_moves <= 0:
            return
            
        # Strategy 1: Targeted movement of key blocks
        self._shuffle_key_blocks()
        
        # Strategy 2: Random walk with movement tracking
        self._random_walk_shuffle()
        
        # Strategy 3: Ensure minimum displacement
        self._ensure_minimum_displacement()

    def _shuffle_key_blocks(self):
        """Move key blocks (especially the target block) in specific patterns."""
        target_block_id = None
        for block_id, block in self.blocks.items():
            if block.get("is_target", False):
                target_block_id = block_id
                break
        
        if target_block_id is None:
            return
            
        # Try to move the target block first (usually the hardest to move)
        moves_attempted = 0
        max_attempts = self.num_shuffle_moves // 3
        
        while moves_attempted < max_attempts:
            direction = self.np_random.randint(4)
            if self._move_block(target_block_id, direction):
                moves_attempted += 1
            else:
                # If target block can't move, move a blocking block
                valid_moves = self._get_valid_moves()
                if valid_moves:
                    move_idx = self.np_random.randint(len(valid_moves))
                    block_id, direction = valid_moves[move_idx]
                    self._move_block(block_id, direction)
                    moves_attempted += 1

    def _random_walk_shuffle(self):
        """Perform random walk ensuring no immediate reversals."""
        moves_made = 0
        max_moves = (self.num_shuffle_moves * 2) // 3
        last_move = None
        
        while moves_made < max_moves:
            valid_moves = self._get_valid_moves()
            if not valid_moves:
                break
                
            # Filter out moves that immediately reverse the last move
            if last_move is not None:
                last_block, last_dir = last_move
                reverse_dir = (last_dir + 2) % 4  # Opposite direction
                valid_moves = [
                    move for move in valid_moves 
                    if not (move[0] == last_block and move[1] == reverse_dir)
                ]
            
            if not valid_moves:
                valid_moves = self._get_valid_moves()
            
            if valid_moves:
                move_idx = self.np_random.randint(len(valid_moves))
                block_id, direction = valid_moves[move_idx]
                if self._move_block(block_id, direction):
                    last_move = (block_id, direction)
                    moves_made += 1

    def _ensure_minimum_displacement(self):
        """Ensure blocks have moved a minimum distance from their original positions."""
        min_displacement = 1  # Minimum number of cells a block should move
        remaining_moves = self.num_shuffle_moves // 6  # Reserve some moves for this
        
        # Check if original_blocks is available
        if self.original_blocks is None:
            return
        
        for block_id, block in self.blocks.items():
            if block_id not in self.original_blocks:
                continue
                
            original_pos = self.original_blocks[block_id]["position"]
            current_pos = block["position"]
            
            # Skip if positions are None
            if original_pos is None or current_pos is None:
                continue
            
            # Calculate Manhattan distance
            displacement = abs(original_pos[0] - current_pos[0]) + abs(original_pos[1] - current_pos[1])
            
            if displacement < min_displacement and remaining_moves > 0:
                # Try to move this block further
                attempts = 0
                max_attempts = 10
                
                while displacement < min_displacement and attempts < max_attempts and remaining_moves > 0:
                    # Find a direction that moves the block away from its original position
                    best_moves = []
                    
                    for direction in range(4):
                        if self._can_move(block_id, direction):
                            # Calculate where the block would be after this move
                            row, col = current_pos
                            if direction == 0:  # up
                                new_pos = (row - 1, col)
                            elif direction == 1:  # right
                                new_pos = (row, col + 1)
                            elif direction == 2:  # down
                                new_pos = (row + 1, col)
                            else:  # left
                                new_pos = (row, col - 1)
                            
                            new_displacement = abs(original_pos[0] - new_pos[0]) + abs(original_pos[1] - new_pos[1])
                            if new_displacement > displacement:
                                best_moves.append((direction, new_displacement))
                    
                    if best_moves:
                        # Choose the move that maximizes displacement
                        best_moves.sort(key=lambda x: x[1], reverse=True)
                        chosen_direction = best_moves[0][0]
                        
                        if self._move_block(block_id, chosen_direction):
                            current_pos = block["position"]
                            if current_pos is None:
                                break
                            displacement = abs(original_pos[0] - current_pos[0]) + abs(original_pos[1] - current_pos[1])
                            remaining_moves -= 1
                    
                    attempts += 1
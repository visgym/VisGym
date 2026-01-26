from collections import deque
import numpy as np

class MazeSolver:
    """
    A solver for grid-based mazes using Breadth-First Search (BFS) to find the shortest path.
    """

    def solve(self, maze_map: np.ndarray, start_pos: tuple, target_pos: tuple) -> list[tuple]:
        """
        Finds the shortest path from a start to a target position in a maze.

        Args:
            maze_map (np.ndarray): A 2D array representing the maze (0=path, 1=wall).
            start_pos (tuple): The starting (row, col) coordinates.
            target_pos (tuple): The target (row, col) coordinates.

        Returns:
            list[tuple]: A list of (row, col) tuples representing the path, or an empty list if no path is found.
        """
        rows, cols = maze_map.shape
        start_pos = tuple(start_pos)
        target_pos = tuple(target_pos)

        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}

        while queue:
            (current_r, current_c), path = queue.popleft()

            if (current_r, current_c) == target_pos:
                return path

            for dr, dc in directions:
                next_r, next_c = current_r + dr, current_c + dc

                if (0 <= next_r < rows and 0 <= next_c < cols and
                        maze_map[next_r, next_c] == 0 and
                        (next_r, next_c) not in visited):
                    
                    visited.add((next_r, next_c))
                    new_path = path + [(next_r, next_c)]
                    queue.append(((next_r, next_c), new_path))
        
        return [] # Return empty list if no path is found

def convert_path_to_toymaze2d_actions(path: list[tuple]) -> list[str]:
    """
    Converts a coordinate path to a sequence of actions for ToyMaze2DEnv.
    """
    actions = []
    # Direction mapping for ToyMaze2D: 0=right, 1=up, 2=left, 3=down
    # Vector: (row_delta, col_delta) -> action_id
    dir_to_action = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        dr, dc = r2 - r1, c2 - c1
        
        action_id = dir_to_action.get((dr, dc))
        if action_id is not None:
            actions.append(f"('move', {action_id})")
        else:
            raise ValueError(f"Invalid move in path: from {path[i]} to {path[i+1]}")
            
    actions.append("('stop', 'stop')") # Terminate the episode
    return actions


    """
    Converts a coordinate path to a sequence of integer actions for ToyMaze3DEnv.
    """
    actions = []
    # Camera directions in ToyMaze3D: 0=North, 1=West, 2=South, 3=East
    # In the env, actions are: 0=forward, 1=left, 2=right, 3=around
    current_cam_dir = 0 # Assume agent starts facing North (up)

    # Map movement vector (dr, dc) to the required camera direction
    vector_to_cam_dir = {(-1, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3}

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        dr, dc = r2 - r1, c2 - c1
        
        target_cam_dir = vector_to_cam_dir.get((dr, dc))
        if target_cam_dir is None:
            raise ValueError(f"Invalid move in path: from {path[i]} to {path[i+1]}")

        # --- 1. Generate turning actions ---
        # 0=no turn, 1=left, 2=right, 3=around
        turn_diff = (target_cam_dir - current_cam_dir + 4) % 4
        
        turn_action = None
        if turn_diff == 1: # Need to turn left
            turn_action = 1
        elif turn_diff == 3: # Need to turn right
            turn_action = 2
        elif turn_diff == 2: # Need to turn around
            turn_action = 3

        if turn_action is not None:
            actions.append(turn_action)

        # --- 2. Generate move forward action ---
        actions.append(0) # Action 0 is always "move forward"
        
        # --- 3. Update simulated camera direction ---
        current_cam_dir = target_cam_dir
        
    return actions 
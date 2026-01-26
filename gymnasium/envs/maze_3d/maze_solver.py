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

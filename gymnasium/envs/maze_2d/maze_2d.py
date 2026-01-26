# Credit: https://github.com/koulanurag/maze-world

from copy import copy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.spaces import FuncConditional, Text, Discrete
import ast
import random
from typing import Optional, Dict


class RandomMaze2DEnv(gym.Env, gym.VLMEnvMixin):
    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        generate_maze_fn: callable,
        maze_width: int = None,  # columns
        maze_height: int = None,  # rows
        draw_grids: bool = False,
        seed: int = None,
    ):
        """

        :param generate_maze_fn: This function is called during every reset of the environment and is expected to return three items in following order:

            - maze-map:  numpy array of  map where "1" represents wall and "0" represents floor.
            - agent location: tuple (x,y) where x and y represent location  of agent
            - target location: tuple (x,y) where x and y represent target location  of the agent

        :param maze_width: The width of the maze
        :param maze_height:  The height of the maze
        """

        self.generate_maze_fn = generate_maze_fn
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.draw_grids = draw_grids
        self.seed(seed)

        # The size of the PyGame window
        self._window_pixel_size = 25
        self._window_size = (
            maze_width * self._window_pixel_size,
            maze_height * self._window_pixel_size,
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 1]),  # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),  # down
        }

        self.action_space = FuncConditional({
            "move": Discrete(4),
            "stop": Text(4) # dummy text space, we just need the stop signal
        })
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.zeros((self.maze_height, self.maze_width)),
                    high=np.ones((self.maze_height, self.maze_width)) * 2,
                    shape=(
                        self.maze_height,
                        self.maze_width,
                    ),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.zeros((self.maze_height, self.maze_width)),
                    high=np.ones((self.maze_height, self.maze_width)) * 2,
                    shape=(
                        self.maze_height,
                        self.maze_width,
                    ),
                    dtype=int,
                ),
            }
        )

        self._canvas = None
        self._window = None
        self._clock = None
    def seed(self, seed: int = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_prompt(self, **kwargs) -> str:
        # Start with the basic task description
        prompt = f"You are navigating a {self.maze_height}x{self.maze_width} maze. "

        if self.render_mode == "ansi":
            prompt += (
                "You are given a text-based representation of the maze where:\n"
                "- Walls are represented by '#'.\n"
                "- Paths are represented by ' ' (empty space).\n"
                "- Your position is marked with 'A'.\n"
                "- The target is marked with 'T'.\n"
                "Your goal is to navigate from 'A' to 'T'."
            )
        else:  # rgb_array
            prompt += (
                "The maze consists of walls (gray) and paths (white). "
                "You are represented by a blue circle, and your goal is to reach the red target square."
            )

        # Dynamically build action descriptions
        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Move in one of four directions. "
                "Format: `('move', direction)` where direction is an integer:\n"
                "   - 0=right, 1=up, 2=left, 3=down"
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': End the navigation session. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)

        # Add success criteria
        prompt += "\n\nSuccess: You succeed if you reach the red target square."

        # Add examples for available actions
        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move right: `('move', 0)`")
            examples.append("- To move up: `('move', 1)`")
            examples.append("- To move left: `('move', 2)`")
            examples.append("- To move down: `('move', 3)`")
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
            'maze_map': self.maze_map.tolist(),
            'agent_location': self._agent_location.tolist(),
            'target_location': self._target_location.tolist(),
        }

    def reset(self, seed: int = None, options=None, init_state: Optional[Dict] = None):
        """
        Resets the environment to its initial state and generates a new random maze configuration.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._prev_agent_location = None
        
        if init_state is not None:
            # Initialize from provided state (for reproducibility without random seed)
            self.maze_map = np.array(init_state['maze_map'])
            self._agent_location = np.array(init_state['agent_location'])
            self._target_location = np.array(init_state['target_location'])
        else:
            # Normal reset: generate a new maze
            self.maze_map, self._agent_location, self._target_location = (
                self.generate_maze_fn()
            )
        
        if not np.array_equal(self.maze_map.shape, [self.maze_height, self.maze_width]):
            raise ValueError(
                f"Shape of Generated Maze doesn't match with"
                f" specified maze width and height."
                f" Generate maze shape is {self.maze_map.shape}, "
                f"whereas specified maze width is {self.maze_width}"
                f" and height is {self.maze_height}"
            )

        # return initial parameters
        observation = self._get_obs()
        info = self._get_info()
        self._canvas = None

        return observation, info

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
                direction = self._action_to_direction[payload]
                new_agent_location = self._agent_location + direction
                
                if self._no_obstacle(new_agent_location):
                    self._prev_agent_location, self._agent_location = (
                        self._agent_location,
                        new_agent_location,
                    )
                else:
                    self.env_feedback = "Cannot move into a wall."
        
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

    def render(self, mode = "rgb_array"):
        """
        Compute the render frames as specified by `mode`.
        """
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "rgb_array":
            return self._render_frame()
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def solve(self, strategy: str = None, num_steps: int = None):
        """
        Returns a sequence of actions that will solve the maze.
        If num_steps is provided, it will pad the solution with back-and-forth
        moves to meet the minimum step requirement.
        
        Args:
            strategy (str, optional): The strategy to use for solving the maze.
            num_steps (int, optional): The minimum number of steps in the returned trajectory.
        
        Returns:
            list[str]: A sequence of action strings to solve the maze.
        """
        try:
            from .maze_solver import MazeSolver, convert_path_to_toymaze2d_actions
        except ImportError:
            # Fallback to simple heuristic if maze_solver is not available
            action = self._solve_heuristic()
            if action == "('stop', 'stop')":
                 return [action]
            # Note: Heuristic doesn't support num_steps padding and only gives one step.
            return [action, "('stop', 'stop')"]

        # If already at target, stop
        if np.array_equal(self._agent_location, self._target_location):
            return ["('stop', 'stop')"]
        
        # Solve the maze using MazeSolver to get the optimal path
        solver = MazeSolver()
        path = solver.solve(copy(self.maze_map), self._agent_location, self._target_location)
        
        if not path or len(path) < 2:
            return ["('stop', 'stop')"]
        
        # Convert the path to a sequence of actions
        actions = convert_path_to_toymaze2d_actions(path)
        
        if not actions:
            return ["('stop', 'stop')"]

        # Pad with back-and-forth moves if num_steps is specified
        if num_steps is not None and len(actions) < num_steps:
            num_padding_needed = num_steps - len(actions)
            # Each padding op adds a (move, move_back) pair, i.e., 2 steps
            num_pairs_to_add = (num_padding_needed + 1) // 2

            padding_opportunities = []
            # We can insert a pad at any point along the path, from start to end.
            for path_idx, location in enumerate(path):
                for direction in range(4):
                    next_loc = location + self._action_to_direction[direction]
                    if (0 <= next_loc[0] < self.maze_height and 
                        0 <= next_loc[1] < self.maze_width and 
                        self._no_obstacle(next_loc)):
                        # An opportunity is defined by where to insert in the action list
                        # and which direction to use for the back-and-forth move.
                        padding_opportunities.append({'insert_at': path_idx, 'dir': direction})
            
            if padding_opportunities:
                # Randomly select opportunities to add padding
                indices_to_sample = self.np_random.integers(0, len(padding_opportunities), size=num_pairs_to_add)
                pads_to_insert = [padding_opportunities[i] for i in indices_to_sample]

                # Sort by insertion index descending to avoid messing up indices
                pads_to_insert.sort(key=lambda p: p['insert_at'], reverse=True)

                for pad in pads_to_insert:
                    idx = pad['insert_at']
                    direction = pad['dir']
                    reverse_direction = (direction + 2) % 4
                    
                    pad_action = f"('move', {direction})"
                    rev_action = f"('move', {reverse_direction})"
                    
                    # Insert the pair. The agent will move and then immediately move back.
                    actions.insert(idx, rev_action)
                    actions.insert(idx, pad_action)

        # Always end with a stop action
        actions.append("('stop', 'stop')")
        
        return actions

    def close(self):
        """
        Closes the environment.

        This method shuts down the Pygame display if it was initialized.

        """
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _get_obs(self):
        image = self.render()
        return image

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "agent": self._agent_location,
            "target": self._target_location,
            "maze_map": self.maze_map,
            "env_feedback": self.env_feedback,
        }
    
    def _compute_reward(self):
        if np.array_equal(self._agent_location, self._target_location):
            return 1.0
        else:
            return 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _render_ansi(self):
        """
        Render the environment to an ANSI string.
        """
        # Create a character array from the maze map
        maze_chars = np.full(self.maze_map.shape, " ", dtype="<U1")
        maze_chars[self.maze_map == 1] = "#"

        # Place the target and agent
        maze_chars[self._target_location[0], self._target_location[1]] = "T"
        maze_chars[self._agent_location[0], self._agent_location[1]] = "A"

        # Add a border for clarity
        h, w = self.maze_map.shape
        bordered_maze = np.full((h + 2, w + 2), "#", dtype="<U1")
        bordered_maze[1:-1, 1:-1] = maze_chars

        # Convert to a single string
        return "\n".join("".join(row) for row in bordered_maze)

    def _render_frame(self):
        if self._canvas is None:
            self._canvas = pygame.Surface(self._window_size)
            self._canvas.fill((255, 255, 255))

            # draw walls
            for x in range(self.maze_height):
                for y in range(self.maze_width):
                    if self.maze_map[x, y] == 1:
                        pygame.draw.rect(
                            self._canvas,
                            (169, 169, 169),
                            pygame.Rect(
                                np.array([y, x]) * self._window_pixel_size,
                                (self._window_pixel_size, self._window_pixel_size),
                            ),
                        )
            
            # draw grids if enabled
            if self.draw_grids:
                for x in range(self.maze_height):
                    pygame.draw.line(
                        self._canvas,
                        (0, 0, 0),
                        (0, x * self._window_pixel_size),
                        (self.maze_width * self._window_pixel_size, x * self._window_pixel_size),
                        1,
                    )
                for y in range(self.maze_width):
                    pygame.draw.line(
                        self._canvas,
                        (0, 0, 0),
                        (y * self._window_pixel_size, 0),
                        (y * self._window_pixel_size, self.maze_height * self._window_pixel_size),
                        1,
                    )

            # draw the target
            pygame.draw.rect(
                self._canvas,
                (255, 0, 0),
                pygame.Rect(
                    self._target_location[::-1] * self._window_pixel_size,
                    (self._window_pixel_size, self._window_pixel_size),
                ),
            )

        # Clean previous agent location
        if self._prev_agent_location is not None:
            pygame.draw.circle(
                self._canvas,
                (255, 255, 255),
                (self._prev_agent_location[::-1] + 0.5) * self._window_pixel_size,
                self._window_pixel_size / 4,
            )
        # Draw new agent location
        pygame.draw.circle(
            self._canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * self._window_pixel_size,
            self._window_pixel_size / 4,
        )

        # Return rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._canvas)), axes=(1, 0, 2)
        )

    def _solve_heuristic(self):
        """
        Fallback heuristic when MazeSolver is not available.
        Uses simple greedy approach towards target.
        """
        if np.array_equal(self._agent_location, self._target_location):
            return "('stop', 'stop')"
        
        # Calculate direction to target
        diff = self._target_location - self._agent_location
        
        # Priority: larger absolute difference first
        if abs(diff[1]) >= abs(diff[0]):  # Move horizontally first
            if diff[1] > 0:
                next_pos = self._agent_location + np.array([0, 1])  # right
                direction = 0
            else:
                next_pos = self._agent_location + np.array([0, -1])  # left  
                direction = 2
        else:  # Move vertically
            if diff[0] < 0:
                next_pos = self._agent_location + np.array([-1, 0])  # up
                direction = 1
            else:
                next_pos = self._agent_location + np.array([1, 0])  # down
                direction = 3
        
        # Check if the move is valid (no wall)
        if (0 <= next_pos[0] < self.maze_height and 
            0 <= next_pos[1] < self.maze_width and 
            self._no_obstacle(next_pos)):
            return f"('move', {direction})"
        
        # If direct path is blocked, try other directions
        for dir_idx, direction_vec in self._action_to_direction.items():
            next_pos = self._agent_location + direction_vec
            if (0 <= next_pos[0] < self.maze_height and 
                0 <= next_pos[1] < self.maze_width and 
                self._no_obstacle(next_pos)):
                return f"('move', {dir_idx})"
        
        # If no move is possible, stop
        return "('stop', 'stop')"
    

    def _no_obstacle(self, location):
        return not (self.maze_map[location[0], location[1]] == 1)  # check for walls

class Maze2DEnv(RandomMaze2DEnv):
    r"""Extends the MazeEnv class to create random mazes of specified sizes at each reset.

    :example:
        >>> env = RandomToyMaze2DEnv(maze_width=5,maze_height=7)
    """

    def __init__(
        self,
        maze_width: int = 11,
        maze_height: int = 11,
        seed: int = None,
    ):
        r"""
        :param render_mode: Specify one of the following:

            - None (default): No render is computed.
            - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption. This rendering should occur during step() and render() doesn't need to be called. Returns None.
            - "rgb_array": Return a single frame representing the current state of the environment. A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
            - "ansi": Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
            - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the wrapper, gymnasium.wrappers.RenderCollection that is automatically applied during gymnasium.make(...,render_mode="rgb_array_list"). The frames collected are popped after render() is called or reset().

        :param maze_width: The width of the maze.
        :param maze_height: The height of the maze.

        :raises ValueError: If the width or height of the maze is not odd.
        """
        if maze_width % 2 == 0 or maze_height % 2 == 0:
            raise ValueError("width/height of maze should be odd")

        self.maze_width = maze_width
        self.maze_height = maze_height
        self.env_feedback = ""
        # Seeding is handled by the superclass `ToyMaze2DEnv` which is called below.

        def _generate_maze():
            """
            Generates a random internal maze configuration.

            :return: A tuple containing the maze configuration, agent location, and target location.
            :rtype: tuple
            """

            # generate internal maze( other than outside wall area)
            generator = WilsonMazeGenerator(maze_height - 2, maze_width - 2, seed=int(self.np_random.integers(2**32)))
            generator.generate_maze()

            maze_config = np.ones((maze_height, maze_width), dtype=int)

            # fill maze by skipping outside walls
            maze_config[1:-1, 1:-1] = 1 - np.array(generator.grid)
            a=[
                    ([1, 1], [maze_config.shape[0] - 2, maze_config.shape[1] - 2]),
                    ([maze_config.shape[0] - 2, maze_config.shape[1] - 2], [1, 1]),
                    ([maze_config.shape[0] - 2, 1], [1, maze_config.shape[1] - 2]),
                    ([1, maze_config.shape[1] - 2], [maze_config.shape[0] - 2, 1]),
            ]
            idx = self.np_random.integers(0, len(a))
            agent_location, target_location = a[idx]
            agent_location, target_location = np.array(agent_location), np.array(target_location)
            return maze_config, agent_location, target_location

        RandomMaze2DEnv.__init__(
            self,
            generate_maze_fn=_generate_maze,
            maze_width=maze_width,
            maze_height=maze_height,
            seed=seed,
        )

class WilsonMazeGenerator:
    """
    Maze Generator utilizing Wilson's Loop Erased Random Walk Algorithm.

    Source: https://github.com/CaptainFl1nt/WilsonMazeGenerator
    """

    def __init__(self, height: int, width: int, seed: Optional[int] = None):
        """
        Initializes a maze generator with the specified width and height.

        :param height: Height of the generated mazes.
        :type height: int
        :param width: Width of the generated mazes.
        :type width: int
        """
        self.width = 2 * (width // 2) + 1  # Make width odd
        self.height = 2 * (height // 2) + 1  # Make height odd
        self.rng = random.Random(seed)

        # grid of cells
        self.grid = [[0 for j in range(self.width)] for i in range(self.height)]

        # declare instance variable
        self.visited = []  # visited cells
        self.unvisited = []  # unvisited cells
        self.path = dict()  # random walk path

        # valid directions in random walk
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # indicates whether a maze is generated
        self.generated = False

        # shortest solution
        self.solution = []
        self.showSolution = False
        self.start = (0, 0)
        self.end = (self.height - 1, self.width - 1)

    def __str__(self):
        """
        Returns a string representation of the maze grid.

        :return: String representation of the grid.
        :rtype: str
        """
        out = "##" * (self.width + 1) + "\n"
        for i in range(self.height):
            out += "#"
            for j in range(self.width):
                if self.grid[i][j] == 0:
                    out += "##"
                else:
                    if not self.showSolution:
                        out += "  "
                    elif (i, j) in self.solution:
                        out += "**"
                    else:
                        out += "  "
            out += "#\n"
        return out + "##" * (self.width + 1)

    def get_grid(self):
        """
        Retrieve the maze grid.

        :return: The maze grid.
        :rtype: list
        """
        return self.grid

    def get_solution(self):
        """
        Returns the solution to the maze as a list of tuples.

        :return: The solution to the maze.
        :rtype: list
        """
        return self.solution

    def show_solution(self, show):
        """
        Sets whether the __str__() method outputs the solution or not.

        :param show: Boolean value indicating whether to show the solution or not.
        :type show: bool
        """
        self.showSolution = show

    def generate_maze(self):
        """
        Generates the maze according to the Wilson Loop Erased Random Walk Algorithm.

        The algorithm works as follows:
            1. Reset the grid before generation.
            2. Choose the first cell to put in the visited list.
            3. Loop until all cells have been visited:
                a. Choose a random cell to start the walk.
                b. Loop until the random walk reaches a visited cell.
                c. Loop until the end of the path is reached:
                    - Add the cell to visited and cut into the maze.
                    - Follow the direction to the next cell.

        :return: None
        """
        # reset the grid before generation
        self.__initialize_grid()

        # choose the first cell to put in the visited list
        # see Step 1 of the algorithm.
        current = self.unvisited.pop(self.rng.randint(0, len(self.unvisited) - 1))
        self.visited.append(current)
        self._cut(current)

        # loop until all cells have been visited
        while len(self.unvisited) > 0:
            # choose a random cell to start the walk (Step 2)
            first = self.unvisited[self.rng.randint(0, len(self.unvisited) - 1)]
            current = first
            # loop until the random walk reaches a visited cell
            while True:
                # choose direction to walk (Step 3)
                dirNum = self.rng.randint(0, 3)
                # check if direction is valid. If not, choose new direction
                while not self.__is_valid_direction(current, dirNum):
                    dirNum = self.rng.randint(0, 3)
                # save the cell and direction in the path
                self.path[current] = dirNum
                # get the next cell in that direction
                current = self.__get_next_cell(current, dirNum, 2)
                if current in self.visited:  # visited cell is reached (Step 5)
                    break

            current = first  # go to start of path
            # loop until the end of path is reached
            while True:
                # add cell to visited and cut into the maze
                self.visited.append(current)
                self.unvisited.remove(current)  # (Step 6.b)
                self._cut(current)

                # follow the direction to next cell (Step 6.a)
                dirNum = self.path[current]
                crossed = self.__get_next_cell(current, dirNum, 1)
                self._cut(crossed)  # cut crossed edge

                current = self.__get_next_cell(current, dirNum, 2)
                if current in self.visited:  # end of path is reached
                    self.path = dict()  # clear the path
                    break

        self.generated = True

    def solve_maze(self):
        """Solves the maze according to the Wilson Loop Erased Random Walk Algorithm

        :return: None
        """
        # if there is no maze to solve, cut the method
        if not self.generated:
            return None

        # initialize with empty path at starting cell
        self.path = dict()
        current = self.start

        # loop until the ending cell is reached
        while True:
            while True:
                # choose valid direction
                # must remain in the grid
                # also must not cross a wall
                dirNum = self.rng.randint(0, 3)
                adjacent = self.__get_next_cell(current, dirNum, 1)
                if self.__is_valid_direction(current, dirNum):
                    hasWall = self.grid[adjacent[0]][adjacent[1]] == 0
                    if not hasWall:
                        break
            # add cell and direction to path
            self.path[current] = dirNum

            # get next cell
            current = self.__get_next_cell(current, dirNum, 2)
            if current == self.end:
                break  # break if ending cell is reached

        # go to start of path
        current = self.start
        self.solution.append(current)
        # loop until end of path is reached
        while not (current == self.end):
            dirNum = self.path[current]  # get direction
            # add adjacent and crossed cells to solution
            crossed = self.__get_next_cell(current, dirNum, 1)
            current = self.__get_next_cell(current, dirNum, 2)
            self.solution.append(crossed)
            self.solution.append(current)

        self.path = dict()

    ## Private Methods ##
    ## Do Not Use Outside This Class ##

    def __get_next_cell(self, cell, dirNum, fact):
        """WilsonMazeGenerator.get_next_cell(tuple,int,int) -> tuple
        Outputs the next cell when moved a distance fact in the the
        direction specified by dirNum from the initial cell.
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3
        fact: int distance to next cell"""
        dirTup = self.directions[dirNum]
        return (cell[0] + fact * dirTup[0], cell[1] + fact * dirTup[1])

    def __is_valid_direction(self, cell, dirNum):
        """WilsonMazeGenerator(tuple,int) -> boolean
        Checks if the adjacent cell in the direction specified by
        dirNum is within the grid
        cell: tuple (y,x) representing position of initial cell
        dirNum: int with values 0,1,2,3"""
        newCell = self.__get_next_cell(cell, dirNum, 2)
        tooSmall = newCell[0] < 0 or newCell[1] < 0
        tooBig = newCell[0] >= self.height or newCell[1] >= self.width
        return not (tooSmall or tooBig)

    def __initialize_grid(self):
        """
        Resets the maze grid to blank before generating a maze.

        :return: None
        """
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = 0

        # fill up unvisited cells
        for r in range(self.height):
            for c in range(self.width):
                if r % 2 == 0 and c % 2 == 0:
                    self.unvisited.append((r, c))

        self.visited = []
        self.path = dict()
        self.generated = False

    def _cut(self, cell):
        """
        Sets the value of the grid at the specified location to 1, indicating a cut.

        :param tuple cell: A tuple (y, x) representing the location where the cut should occur.
        :return: None
        """
        self.grid[cell[0]][cell[1]] = 1
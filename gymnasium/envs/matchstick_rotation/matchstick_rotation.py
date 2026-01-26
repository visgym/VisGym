import ast
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import FuncConditional
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from typing import Optional, Dict

matplotlib.use('Agg')

class MatchstickRotationEnv(gym.Env, gym.VLMEnvMixin):
    """
    Matchstick Rotation Env:
      - The agent sees a blue stick and a red target stick on a canvas.
      - The agent can move and rotate the blue stick to match the red target stick.
      - The agent can stop and the episode is terminated if the blue stick is within the target stick.
    """

    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        canvas_size=(100, 100),
        stick_length=20,
        seed: int = None,
        scale_range=(0.3, 3.0),
        pos_tolerance=5,
        ang_tolerance=10,
    ):
        super().__init__()
        self.canvas_size = canvas_size
        self.stick_length = stick_length
        self.rng = np.random.default_rng(seed)
        self.scale_range = scale_range
        self.pos_tolerance = pos_tolerance
        self.ang_tolerance = ang_tolerance

        # two actions: move or stop
        self.action_space = FuncConditional({
            "move": spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            "stop": spaces.Text(4),
        })

        H, W = canvas_size
        self.observation_space = spaces.Box(0,255,shape=(H,W,3),dtype=np.uint8)
        self.env_feedback = None

        self._sample_scale()
        self._sample_target()

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_prompt(self, **kwargs) -> str:
        """
        Build an instruction prompt for the match-move stick rotation puzzle using the same
        structure as the zoom-in prompt.
        """
        # Intro sentence
        prompt = (
            "You see a blue stick and a red target stick on a canvas. "
            "Your goal is to move and rotate the blue stick to match the red target stick. "
            "Unit moves are in an *unknown* scale (could be tiny or huge).\n\n"
            "Available actions:\n"
        )

        action_descriptions = []
        actions = self.action_space.get_function_names()

        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Translate by dx,dy units (any real number) "
                "and rotate by dθ° (any real number). "
                "Format: `('move', [dx, dy, dθ])`"
            )

        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Submit your current stick as final. "
                f"You win (+1) if your stick is within {self.pos_tolerance}px and {self.ang_tolerance}° of the target. "
                "Format: `('stop', 'stop')`"
            )

        prompt += "\n".join(action_descriptions)

        prompt += (
            "\n\nPlease respond with exactly one action and its arguments in the "
            "specified format. For example:\n"
        )

        examples = []
        if "move" in actions:
            examples.append("- Move right by 2 units and rotate 45°: `('move', [2, 0, 45])`")
            examples.append("- Move diagonally and rotate: `('move', [1.5, -0.8, 90])`")
        if "stop" in actions:
            examples.append("- Finalize the stick position: `('stop', 'stop')`")

        prompt += "\n".join(examples)
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        return {
            'scale': float(self._scale),
            'target_mid': self.target["mid"].tolist(),
            'target_ang': float(self.target["ang"]),
            'initial_mid': self.state["mid"].tolist(),
            'initial_ang': float(self.state["ang"]),
        }
    
    def reset(self, *, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        if init_state is not None:
            self._scale = init_state['scale']
            self.target = {
                "mid": np.array(init_state['target_mid']),
                "ang": init_state['target_ang']
            }
            self.state = {
                "mid": np.array(init_state['initial_mid']),
                "ang": init_state['initial_ang'],
                "done": False
            }
        else:
            self._sample_scale()
            self._sample_target()
            W, H = self.canvas_size
            x0 = self.rng.uniform(0, W / self._scale)
            y0 = self.rng.uniform(0, H / self._scale)
            a0 = self.rng.uniform(0,360)
            self.state = {"mid": np.array([x0,y0]), "ang": a0, "done":False}
        
        self.env_feedback = None
        return self._get_obs(), {}

    def _move(self, payload):
        dx, dy, dθ = map(float, payload)
        self.state["mid"] += np.array([dx, dy]) * self._scale
        self.state["ang"] = (self.state["ang"] + dθ) % 360
        W, H = self.canvas_size
        self.state["mid"] = np.clip(self.state["mid"], [0,0], [W/self._scale, H/self._scale])



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
        if branch == "move":
            if not self.action_space["move"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                self._move(payload)
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

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            import matplotlib
            matplotlib.use("Agg", force=True)
            
            fig, ax = plt.subplots(figsize=(4,4))
            ax.set_xlim(0, self.canvas_size[0])
            ax.set_ylim(0, self.canvas_size[1])
            ax.axis("off")

            def draw(mid, ang, **kw):
                θ = np.deg2rad(ang)
                dx = (self.stick_length/2)*np.cos(θ)
                dy = (self.stick_length/2)*np.sin(θ)
                # map scaled coords back to pixels
                mx, my = mid * self._scale
                ax.plot([mx-dx, mx+dx],[my-dy, my+dy], **kw)

            # target in red dashed
            draw(self.target["mid"], self.target["ang"],
                color="red", linestyle="--", linewidth=2)
            # current in blue
            draw(self.state["mid"], self.state["ang"],
                color="blue", linewidth=3)

            # Add black border around the canvas
            ax.add_patch(Rectangle((0, 0), self.canvas_size[0], self.canvas_size[1],
                                fill=False, edgecolor="black", linewidth=2, zorder=1000))

            fig.canvas.draw()
            # Use print_to_buffer like mental_rotation_3d does for better memory management
            buf, (width, height) = fig.canvas.print_to_buffer()
            img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))[..., :3]
            
            # Clean up matplotlib state like mental_rotation_3d
            plt.close(fig)
            plt.clf()
            plt.cla()
            
            return img
        else:
            raise ValueError(f"Invalid render mode: {mode}")


    def solve(self, num_steps: int = 5, strategy: str = "unit_first") -> list:
        """
        Returns a sequence of actions that will solve the match-move puzzle.
        The total movement is split stochastically over the given number of steps.

        Args:
            num_steps (int): The number of steps to spread the solution over.
            strategy (str): The solution strategy. Can be 'stochastic' or 'unit_first'.

        Returns:
            List[str]: A list of action strings.
        """
        # Calculate the total required change in position and angle
        delta_pos = self.target["mid"] - self.state["mid"]
        # For a symmetric stick, angles that differ by 180° are equivalent.
        # Choose the smallest rotation in [-90, 90] to align modulo 180°.
        raw_diff = (self.target["ang"] - self.state["ang"]) % 180
        if raw_diff > 90:
            raw_diff -= 180
        delta_ang = raw_diff

        # The agent's dx, dy are scaled by self._scale before being applied.
        # So, to achieve delta_pos, we must provide a pre-scaled dx, dy.
        dx, dy = delta_pos / self._scale
        d_theta = delta_ang

        if strategy == "unit_first":
            actions = []
            if num_steps <= 2:
                # Step 1: Unit translation based on angle to target
                angle_to_target = np.rad2deg(np.arctan2(delta_pos[1], delta_pos[0]))
                angle_to_target = (angle_to_target + 360) % 360  # Normalize to [0, 360]

                sector = int((angle_to_target + 22.5) % 360 / 45)
                unit_moves = [
                    (1, 0), (1, 1), (0, 1), (-1, 1),
                    (-1, 0), (-1, -1), (0, -1), (1, -1)
                ]
                unit_dx, unit_dy = unit_moves[sector]

                # First action is unit translation only
                action1_payload = [round(float(unit_dx), 1), round(float(unit_dy), 1), 0.0]
                actions.append(str(('move', action1_payload)))

                # Step 2: Move directly to the target from the new position
                remaining_dx = dx - unit_dx
                remaining_dy = dy - unit_dy
                remaining_d_theta = d_theta # Full rotation is in the second step

                action2_payload = [round(float(v), 1) for v in [remaining_dx, remaining_dy, remaining_d_theta]]
                actions.append(str(('move', action2_payload)))
            else: # num_steps > 2
                current_pos_unscaled = self.state["mid"].copy()
                total_dx_applied = 0
                total_dy_applied = 0

                for i in range(num_steps - 1):
                    # Recalculate direction to target from the current simulated position
                    current_delta_pos = self.target["mid"] - current_pos_unscaled
                    
                    # Determine the direction for the unit move
                    angle_to_target = np.rad2deg(np.arctan2(current_delta_pos[1], current_delta_pos[0]))
                    angle_to_target = (angle_to_target + 360) % 360

                    sector = int((angle_to_target + 22.5) % 360 / 45)
                    unit_moves = [
                        (1, 0), (1, 1), (0, 1), (-1, 1),
                        (-1, 0), (-1, -1), (0, -1), (1, -1)
                    ]
                    base_unit_dx, base_unit_dy = unit_moves[sector]

                    # Scale the unit move by decreasing length
                    move_length = num_steps - 1 - i
                    step_dx = base_unit_dx * move_length
                    step_dy = base_unit_dy * move_length

                    # Append the action for this step (translation only)
                    action_payload = [round(float(step_dx), 1), round(float(step_dy), 1), 0.0]
                    actions.append(str(('move', action_payload)))

                    # Update the simulated position and total applied deltas
                    current_pos_unscaled += np.array([step_dx, step_dy])
                    total_dx_applied += step_dx
                    total_dy_applied += step_dy

                # Final step: move the remaining distance and do all the rotation
                remaining_dx = dx - total_dx_applied
                remaining_dy = dy - total_dy_applied
                remaining_d_theta = d_theta

                final_action_payload = [round(float(v), 1) for v in [remaining_dx, remaining_dy, remaining_d_theta]]
                actions.append(str(('move', final_action_payload)))

        else: # Default to "stochastic" strategy
            if num_steps is None or num_steps < 1:
                num_steps = 3

            total_move = np.array([dx, dy, d_theta])

            # Stochastically divide the total move into smaller steps
            if num_steps > 1:
                # Generate `num_steps - 1` random split points (fractions)
                split_points = np.sort(self.rng.uniform(0, 1, num_steps - 1))
                # Create the sequence of movement fractions for each step
                step_fractions = np.diff(np.concatenate(([0], split_points, [1])))
                step_moves = [total_move * f for f in step_fractions]
            else:
                step_moves = [total_move]

            # Create the sequence of 'move' actions
            actions = []
            for move in step_moves:
                # The payload should be a list of floats with only one decimal place
                action_payload = [round(float(v), 1) for v in move]
                action_tuple = ('move', action_payload)
                actions.append(str(action_tuple))

        # Add the final "stop" action
        actions.append("('stop', 'stop')")

        return actions

    def close(self): 
        """Clean up matplotlib state to prevent multiprocessing issues."""
        try:
            plt.close('all')
            plt.clf()
            plt.cla()
        except:
            pass

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        return {"env_feedback": self.env_feedback}

    def _compute_reward(self):
        tgt_mid = self.target["mid"] * self._scale
        pos_err = np.linalg.norm(self.state["mid"] * self._scale - tgt_mid)
        pos_err /= self._scale
        raw_diff = (self.state["ang"] - self.target["ang"]) % 180
        angle_err = raw_diff if raw_diff <= 90 else 180 - raw_diff
        return 1.0 if (pos_err <= self.pos_tolerance and angle_err <= self.ang_tolerance) else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _sample_scale(self):
        lo, hi = self.scale_range
        self._scale = self.rng.uniform(lo, hi)

    def _sample_target(self):
        W, H = self.canvas_size
        x = self.rng.uniform(0, W / self._scale)
        y = self.rng.uniform(0, H / self._scale)
        ang = self.rng.uniform(0,360)
        self.target = {"mid": np.array([x,y]), "ang": ang}

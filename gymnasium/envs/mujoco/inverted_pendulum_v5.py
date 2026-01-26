__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Union

import numpy as np
import ast
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import gymnasium as gym
from gymnasium.spaces import Box, FuncConditional, Text

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}


class InvertedPendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
    This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
    The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.


    ## Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |


    ## Observation Space
    The observation space consists of the following parts (in order):
    - *qpos (2 element):* Position values of the robot's cart and pole.
    - *qvel (2 elements):* The velocities of cart and pole (their derivatives).

    The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s)  |


    ## Rewards
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.

    The pole is considered upright if:
    $|angle| < 0.2$.

    and `info` also contains the reward.


    ## Starting State
    The initial position state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
    The initial velocity state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.

    where $\\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The environment terminates when the Inverted Pendulum is unhealthy.
    The Inverted Pendulum is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    InvertedPendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `xml_file`              | **str**    |`"inverted_pendulum.xml"`| Path to a MuJoCo model                                                                        |
    | `reset_noise_scale`     | **float**  | `0.01`                  | Scale of random perturbations of initial position and velocity (see `Starting State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the Pendulum is healthy (not terminated) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
        - Added `xml_file` argument.
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `info["reward_survive"]` which contains the reward.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.5.
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "inverted_pendulum.xml",
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        terminated = bool(
            not np.isfinite(observation).all() or (np.abs(observation[1]) > 0.2)
        )

        reward = int(not terminated)

        info = {"reward_survive": reward}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

class InvertedPendulumVLMEnv(InvertedPendulumEnv, gym.VLMEnvMixin):
    def __init__(self, **kwargs):
        # 1) initialize the base MuJoCo env
        super().__init__(**kwargs)
        # 2) ensure we render offscreen to RGB arrays
        self.render_mode = "rgb_array"
        # 3) burn one reset+render to get frame shape
        super().reset()
        frame = super().render()
        H, W, C = frame.shape

        # 4) override to image‐in API
        self.observation_space = Box(low=0, high=255, shape=(H, W, C), dtype=np.uint8)
        self.action_space = FuncConditional({
            "act":  Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32),
            "stop": Text(4),
        })
    
    def get_prompt(self, **kwargs) -> str:
        return (
            "You are balancing an inverted pendulum on a cart.  The observation is an RGB image showing:\n"
            "  • the cart on its track\n"
            "  • the pole upright on the cart\n\n"
            "Your objective is to keep the pole balanced (within ±0.2 rad of vertical) for as long as possible.\n\n"
            "At each step, output exactly one action as a string, in one of these two formats:\n\n"
            "  1. Operate:\n"
            "     (\"act\", [f])\n"
            "     • f: horizontal force on the cart ∈ [−3, 3]\n\n"
            "  2. Submit and end the episode:\n"
            "     (\"stop\", \"stop\")\n\n"
            "Examples:\n"
            "  (\"act\", [1.2])    # push right with force 1.2\n"
            "  (\"act\", [-0.8])   # push left with force 0.8\n"
            "  (\"stop\", \"stop\")\n\n"
            "Respond with exactly one of these tuples—nothing else."
        )



    def step(self, action_str: str):
        # parse the VLM’s text
        func, args = ast.literal_eval(action_str)

        if func == "stop":
            # run one zero‐force step to compute the final +1/0 reward
            zero = np.zeros(1, dtype=np.float32)
            _, reward, done, truncated, info = super().step(zero)
            done = True
            frame = super().render()
            return frame, reward, done, truncated, info

        if func != "act":
            raise ValueError(f"Unrecognized action {func!r}")

        # extract force
        if not (isinstance(args, (list,tuple)) and len(args) == 1):
            raise ValueError("`act` requires a single‐element list: ['act',[f]]")
        f = float(args[0])

        # step the physics
        _, reward, done, truncated, info = super().step(np.array([f], dtype=np.float32))
        frame = super().render()
        return frame, reward, done, truncated, info
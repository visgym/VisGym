"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from typing import Any
import os
from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_MODULE_DIR, '..', '..', 'data')
_PARTIAL_DATASET_DIR = os.path.join(_MODULE_DIR, '..', '..', 'inference', 'inference_dataset', 'partial_datasets')

# Helper function to get path with fallback to partial dataset
def _get_path_with_fallback(original_path, fallback_path):
    return original_path if os.path.isdir(original_path) else fallback_path

def _get_file_with_fallback(original_path, fallback_path):
    return original_path if os.path.isfile(original_path) else fallback_path

# Original default paths
DEFAULT_IMAGE_DIR = os.getenv('VIS_GYM_IMAGE_DIR', os.path.join(_DATA_DIR, 'images'))
DEFAULT_VIDEO_DIR = os.getenv('VIS_GYM_VIDEO_DIR', os.path.join(_DATA_DIR, 'videos'))
DEFAULT_3D_DIR = os.getenv('VIS_GYM_3D_DIR', os.path.join(_DATA_DIR, '3d', 'glbs'))
DEFAULT_REFCOCO_DIR = os.getenv('VIS_GYM_REFCOCO_DIR', os.path.join(_DATA_DIR, 'refcoco'))
DEFAULT_LVIS_DIR = os.getenv('VIS_GYM_LVIS_DIR', os.path.join(_DATA_DIR, 'lvis', 'train2017'))
DEFAULT_LVIS_ANNOTATION_FILE = os.getenv('VIS_GYM_LVIS_ANNOTATION_FILE', os.path.join(_DATA_DIR, 'lvis', 'lvis_v1_train.json'))

# Environment-specific paths with fallback to partial datasets
_COLORIZATION_DIR = _get_path_with_fallback(DEFAULT_IMAGE_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'colorization'))
_JIGSAW_DIR = _get_path_with_fallback(DEFAULT_IMAGE_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'jigsaw'))
_MENTAL_ROTATION_2D_DIR = _get_path_with_fallback(DEFAULT_IMAGE_DIR, os.path.join(_PARTIAL_DATASET_DIR, '2d_rotation'))
_MENTAL_ROTATION_3D_OBJAVERSE_DIR = _get_path_with_fallback(DEFAULT_3D_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'objaverse'))
_REFCOCO_SAMPLE_DIR = _get_path_with_fallback(DEFAULT_REFCOCO_DIR, _PARTIAL_DATASET_DIR)
_VIDEO_UNSHUFFLE_DIR = _get_path_with_fallback(DEFAULT_VIDEO_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'ssv2'))
_ZOOM_IN_PUZZLE_DIR = _get_path_with_fallback(DEFAULT_IMAGE_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'zoomin'))
_COUNTING_DIR = _get_path_with_fallback(DEFAULT_LVIS_DIR, os.path.join(_PARTIAL_DATASET_DIR, 'counting'))
_COUNTING_ANNOTATION_FILE = _get_file_with_fallback(DEFAULT_LVIS_ANNOTATION_FILE, os.path.join(_PARTIAL_DATASET_DIR, 'counting', 'lvis_v1_train.json'))

# VisGym Environments
# ----------------------------------------

# Colorization
# ----------------------------------------
register(
    id="colorization/easy",
    entry_point="gymnasium.envs.colorization.colorization:ColorizationEnv",
    kwargs={"accuracy_radius": 11, "sample_dir": _COLORIZATION_DIR},
    disable_env_checker=True,
)

register(
    id="colorization/hard",
    entry_point="gymnasium.envs.colorization.colorization:ColorizationEnv",
    kwargs={"accuracy_radius": 16, "sample_dir": _COLORIZATION_DIR},
    disable_env_checker=True,
)

# Counting
# ----------------------------------------
register(
    id="counting/easy",
    entry_point="gymnasium.envs.counting.counting:CountingEnv",
    kwargs={"min_count": 2, "max_count": 20, "annotation_file": _COUNTING_ANNOTATION_FILE, "sample_dir": _COUNTING_DIR},
    disable_env_checker=True,
)

register(
    id="counting/hard",
    entry_point="gymnasium.envs.counting.counting:CountingEnv",
    kwargs={"min_count": 5, "max_count": 30, "annotation_file": _COUNTING_ANNOTATION_FILE, "sample_dir": _COUNTING_DIR},
    disable_env_checker=True,
)

# Jigsaw
# ----------------------------------------
register(
    id="jigsaw/easy",
    entry_point="gymnasium.envs.jigsaw.jigsaw:JigsawEnv",
    kwargs={"num_rows": 2, "num_cols": 2, "sample_dir": _JIGSAW_DIR},
    disable_env_checker=True,
)

register(
    id="jigsaw/hard",
    entry_point="gymnasium.envs.jigsaw.jigsaw:JigsawEnv",
    kwargs={"num_rows": 3, "num_cols": 3, "sample_dir": _JIGSAW_DIR},
    disable_env_checker=True,
)

# Matchstick Equation
# ----------------------------------------
register(
    id="matchstick_equation/easy",
    entry_point="gymnasium.envs.matchstick_equation.matchstick_equation:MatchstickEquationEnv",
    kwargs={"break_moves": 1},
    disable_env_checker=True,
)

register(
    id="matchstick_equation/hard",
    entry_point="gymnasium.envs.matchstick_equation.matchstick_equation:MatchstickEquationEnv",
    kwargs={"break_moves": 2},
    disable_env_checker=True,
)

# Matchstick Rotation
# ----------------------------------------
register(
    id="matchstick_rotation/easy",
    entry_point="gymnasium.envs.matchstick_rotation.matchstick_rotation:MatchstickRotationEnv",
    kwargs={"pos_tolerance": 10, "ang_tolerance": 15},
    disable_env_checker=True,
)

register(
    id="matchstick_rotation/hard",
    entry_point="gymnasium.envs.matchstick_rotation.matchstick_rotation:MatchstickRotationEnv",
    kwargs={"pos_tolerance": 5, "ang_tolerance": 10},
    disable_env_checker=True,
)

# Maze 2D
# ----------------------------------------
register(
    id="maze_2d/easy",
    entry_point="gymnasium.envs.maze_2d.maze_2d:Maze2DEnv",
    kwargs={"maze_width": 9, "maze_height": 9},
    disable_env_checker=True,
)

register(
    id="maze_2d/hard",
    entry_point="gymnasium.envs.maze_2d.maze_2d:Maze2DEnv",
    kwargs={"maze_width": 11, "maze_height": 11},
    disable_env_checker=True,
)

# Maze 3D
# ----------------------------------------
register(
    id="maze_3d/easy",
    entry_point="gymnasium.envs.maze_3d.maze_3d:Maze3DEnv",
    kwargs={"maze_width": 7, "maze_height": 7},
    disable_env_checker=True,
)

register(
    id="maze_3d/hard",
    entry_point="gymnasium.envs.maze_3d.maze_3d:Maze3DEnv",
    kwargs={"maze_width": 9, "maze_height": 9},
    disable_env_checker=True,
)

# Mental Rotation 2D
# ----------------------------------------
register(
    id="mental_rotation_2d/easy",
    entry_point="gymnasium.envs.mental_rotation_2d.mental_rotation_2d:MentalRotation2DEnv",
    kwargs={"tolerance": 10.0, "sample_dir": _MENTAL_ROTATION_2D_DIR},
    disable_env_checker=True,
)

register(
    id="mental_rotation_2d/hard",
    entry_point="gymnasium.envs.mental_rotation_2d.mental_rotation_2d:MentalRotation2DEnv",
    kwargs={"tolerance": 5.0, "sample_dir": _MENTAL_ROTATION_2D_DIR},
    disable_env_checker=True,
)

# Mental Rotation 3D (Cube)
# ----------------------------------------
register(
    id="mental_rotation_3d_cube/easy",
    entry_point="gymnasium.envs.mental_rotation_3d_cube.mental_rotation_3d_cube:MentalRotation3DCubeEnv",
    kwargs={"num_segments": 4},
    disable_env_checker=True,
)

register(
    id="mental_rotation_3d_cube/hard",
    entry_point="gymnasium.envs.mental_rotation_3d_cube.mental_rotation_3d_cube:MentalRotation3DCubeEnv",
    kwargs={"num_segments": 6},
    disable_env_checker=True,
)

# Mental Rotation 3D (Objaverse)
# ----------------------------------------
register(
    id="mental_rotation_3d_objaverse/easy",
    entry_point="gymnasium.envs.mental_rotation_3d_objaverse.mental_rotation_3d_objaverse:MentalRotation3DObjaverseEnv",
    kwargs={"tolerance": 15.0, "sample_dir": _MENTAL_ROTATION_3D_OBJAVERSE_DIR},
    disable_env_checker=True,
)

register(
    id="mental_rotation_3d_objaverse/hard",
    entry_point="gymnasium.envs.mental_rotation_3d_objaverse.mental_rotation_3d_objaverse:MentalRotation3DObjaverseEnv",
    kwargs={"tolerance": 5.0, "sample_dir": _MENTAL_ROTATION_3D_OBJAVERSE_DIR},
    disable_env_checker=True,
)

# MuJoCo Fetch Pick-and-Place
# ----------------------------------------
register(
    id="fetch_pick_and_place/easy",
    entry_point="gymnasium_robotics.envs.fetch.pick_and_place_discrete:MujocoFetchPickAndPlaceDiscreteEnv",
    disable_env_checker=True,
)

register(
    id="fetch_pick_and_place/hard",
    entry_point="gymnasium_robotics.envs.fetch.pick_and_place_discrete:MujocoFetchPickAndPlaceDiscreteEnv",
    disable_env_checker=True,
)

# MuJoCo Fetch Reach
# ----------------------------------------
register(
    id="fetch_reach/easy",
    entry_point="gymnasium_robotics.envs.fetch.reach_discrete:MujocoFetchReachDiscreteEnv",
    disable_env_checker=True,
)

register(
    id="fetch_reach/hard",
    entry_point="gymnasium_robotics.envs.fetch.reach_discrete:MujocoFetchReachDiscreteEnv",
    disable_env_checker=True,
)

# Patch Reassembly
# ----------------------------------------
register(
    id="patch_reassembly/easy",
    entry_point="gymnasium.envs.patch_reassembly.patch_reassembly:PatchReassemblyEnv",
    kwargs={"grid_size": (6, 6), "num_patches": 5},
    disable_env_checker=True,
)

register(
    id="patch_reassembly/hard",
    entry_point="gymnasium.envs.patch_reassembly.patch_reassembly:PatchReassemblyEnv",
    kwargs={"grid_size": (8, 8), "num_patches": 6},
    disable_env_checker=True,
)

# Referring Dot-Pointing
# ----------------------------------------
register(
    id="referring_dot_pointing/easy",
    entry_point="gymnasium.envs.referring_dot_pointing.referring_dot_pointing:ReferringDotPointingEnv",
    kwargs={"sample_dir": _REFCOCO_SAMPLE_DIR},
    disable_env_checker=True,
)

register(
    id="referring_dot_pointing/hard",
    entry_point="gymnasium.envs.referring_dot_pointing.referring_dot_pointing:ReferringDotPointingEnv",
    kwargs={"sample_dir": _REFCOCO_SAMPLE_DIR},
    disable_env_checker=True,
)

# Sliding Block
# ----------------------------------------
register(
    id="sliding_block/easy",
    entry_point="gymnasium.envs.sliding_block.sliding_block:SlidingBlockEnv",
    kwargs={"num_shuffle_moves": 30},
    disable_env_checker=True,
)

register(
    id="sliding_block/hard",
    entry_point="gymnasium.envs.sliding_block.sliding_block:SlidingBlockEnv",
    kwargs={"num_shuffle_moves": 90},
    disable_env_checker=True,
)

# Video Unshuffle
# ----------------------------------------
register(
    id="video_unshuffle/easy",
    entry_point="gymnasium.envs.video_unshuffle.video_unshuffle:VideoUnshuffleEnv",
    kwargs={"num_frames": 4, "root_dir": _VIDEO_UNSHUFFLE_DIR},
    disable_env_checker=True,
)

register(
    id="video_unshuffle/hard",
    entry_point="gymnasium.envs.video_unshuffle.video_unshuffle:VideoUnshuffleEnv",
    kwargs={"num_frames": 5, "root_dir": _VIDEO_UNSHUFFLE_DIR},
    disable_env_checker=True,
)

# Zoom-In Puzzle
# ----------------------------------------
register(
    id="zoom_in_puzzle/easy",
    entry_point="gymnasium.envs.zoom_in_puzzle.zoom_in_puzzle:ZoomInPuzzleEnv",
    kwargs={"num_zoom_views": 4, "sample_dir": DEFAULT_IMAGE_DIR},
    disable_env_checker=True,
)

register(
    id="zoom_in_puzzle/hard",
    entry_point="gymnasium.envs.zoom_in_puzzle.zoom_in_puzzle:ZoomInPuzzleEnv",
    kwargs={"num_zoom_views": 5, "sample_dir": DEFAULT_IMAGE_DIR},
    disable_env_checker=True,
)

# Classic
# ----------------------------------------

register(
    id="CartPole-v0",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPole-v1",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="Pendulum-v1",
    entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="gymnasium.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# Phys2d (jax classic control)
# ----------------------------------------

register(
    id="phys2d/CartPole-v0",
    entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    vector_entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
    disable_env_checker=True,
)

register(
    id="phys2d/CartPole-v1",
    entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    vector_entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxVectorEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
    disable_env_checker=True,
)

register(
    id="phys2d/Pendulum-v0",
    entry_point="gymnasium.envs.phys2d.pendulum:PendulumJaxEnv",
    vector_entry_point="gymnasium.envs.phys2d.pendulum:PendulumJaxVectorEnv",
    max_episode_steps=200,
    disable_env_checker=True,
)

# Box2d
# ----------------------------------------

register(
    id="LunarLander-v3",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v3",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalker-v3",
    entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcore-v3",
    entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id="CarRacing-v3",
    entry_point="gymnasium.envs.box2d.car_racing:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="gymnasium.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="FrozenLake-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FrozenLake8x8-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="CliffWalking-v0",
    entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
)

register(
    id="Taxi-v3",
    entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200,
)


# Tabular
# ----------------------------------------

register(
    id="tabular/Blackjack-v0",
    entry_point="gymnasium.envs.tabular.blackjack:BlackJackJaxEnv",
    disable_env_checker=True,
)

register(
    id="tabular/CliffWalking-v0",
    entry_point="gymnasium.envs.tabular.cliffwalking:CliffWalkingJaxEnv",
    disable_env_checker=True,
)



# Mujoco
# ----------------------------------------

# manipulation

register(
    id="Reacher-v2",
    entry_point="gymnasium.envs.mujoco.reacher:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v4",
    entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v5",
    entry_point="gymnasium.envs.mujoco.reacher_v5:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v5-vlm",
    entry_point="gymnasium.envs.mujoco.reacher_v5:ReacherVLMEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Pusher-v2",
    entry_point="gymnasium.envs.mujoco.pusher:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Pusher-v4",
    entry_point="gymnasium.envs.mujoco.pusher_v4:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Pusher-v5",
    entry_point="gymnasium.envs.mujoco.pusher_v5:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

# balance

register(
    id="InvertedPendulum-v2",
    entry_point="gymnasium.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v4",
    entry_point="gymnasium.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v5",
    entry_point="gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v5-vlm",
    entry_point="gymnasium.envs.mujoco.inverted_pendulum_v5:InvertedPendulumVLMEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedDoublePendulum-v2",
    entry_point="gymnasium.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v4",
    entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v5",
    entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v5-vlm",
    entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v5:InvertedDoublePendulumVLMEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)


# runners

register(
    id="HalfCheetah-v2",
    entry_point="gymnasium.envs.mujoco.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v3",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v4",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v5",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v2",
    entry_point="gymnasium.envs.mujoco.hopper:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v3",
    entry_point="gymnasium.envs.mujoco.hopper_v3:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v4",
    entry_point="gymnasium.envs.mujoco.hopper_v4:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v5",
    entry_point="gymnasium.envs.mujoco.hopper_v5:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v5-vlm",
    entry_point="gymnasium.envs.mujoco.hopper_v5:HopperVLMEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Swimmer-v2",
    entry_point="gymnasium.envs.mujoco.swimmer:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v3",
    entry_point="gymnasium.envs.mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v4",
    entry_point="gymnasium.envs.mujoco.swimmer_v4:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v5",
    entry_point="gymnasium.envs.mujoco.swimmer_v5:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v5-vlm",
    entry_point="gymnasium.envs.mujoco.swimmer_v5:SwimmerVLMEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v2",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d:Walker2dEnv",
)

register(
    id="Walker2d-v3",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="Walker2d-v4",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv",
)

register(
    id="Walker2d-v5",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v5:Walker2dEnv",
)

register(
    id="Ant-v2",
    entry_point="gymnasium.envs.mujoco.ant:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v3",
    entry_point="gymnasium.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v4",
    entry_point="gymnasium.envs.mujoco.ant_v4:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v5",
    entry_point="gymnasium.envs.mujoco.ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Humanoid-v2",
    entry_point="gymnasium.envs.mujoco.humanoid:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v3",
    entry_point="gymnasium.envs.mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v4",
    entry_point="gymnasium.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v5",
    entry_point="gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="gymnasium.envs.mujoco.humanoidstandup:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v4",
    entry_point="gymnasium.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v5",
    entry_point="gymnasium.envs.mujoco.humanoidstandup_v5:HumanoidStandupEnv",
    max_episode_steps=1000,
)


# --- For shimmy compatibility
def _raise_shimmy_error(*args: Any, **kwargs: Any):
    raise ImportError(
        'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)

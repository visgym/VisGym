"""Run FetchPickAndPlaceDiscrete using an oracle-derived open-loop sequence.

This script executes episodes in the discrete-action Fetch PickAndPlace
environment using the environment-provided `solve()` to obtain
an open-loop sequence that performs the pick-and-place.

Usage:
  python -m examples.run_fetch_pick_and_place_discrete_oracle --args.episodes 5 --args.render --args.sleep_s 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Tuple

import gymnasium as gym
import numpy as np

import gymnasium_robotics


try:
    import tyro
except Exception as exc:  # pragma: no cover - optional CLI dep
    raise RuntimeError(
        "tyro is required for this script. Install with `pip install tyro`."
    ) from exc


gym.register_envs(gymnasium_robotics)


@dataclass
class Args:
    """CLI arguments for running the discrete pick-and-place oracle.

    Attributes:
        env_id: Gymnasium env id to create.
        episodes: Number of episodes to run.
        render: Whether to render in a window.
        reward_type: Reward configuration for the env.
        seed: Base random seed; incremented per-episode.
        sleep_s: Optional per-step sleep in seconds for visualization.
        approach_z_margin: Height above object for approach.
        place_z_margin: Height above goal for transit/carry.
        pos_threshold: Axis-wise tolerance to consider aligned.
        open_thresh: Gripper opening threshold considered open (sum of fingers).
        closed_thresh: Gripper opening threshold considered closed (sum of fingers).
    """

    env_id: str = "FetchPickAndPlaceDiscrete-v4"
    episodes: int = 3
    render: bool = True
    reward_type: Literal["sparse", "dense"] = "sparse"
    seed: int = 0
    sleep_s: float = 0.0
    approach_z_margin: float = 0.12
    place_z_margin: float = 0.12
    pos_threshold: float = 0.03
    open_thresh: float = 0.08
    closed_thresh: float = 0.01


def run_episode(
    env: gym.Env,
    approach_z_margin: float,
    place_z_margin: float,
    pos_threshold: float,
    open_thresh: float,
    closed_thresh: float,
    sleep_s: float,
) -> Tuple[float, bool, int]:
    """Run a single episode using the env's ground-truth action sequence.

    Args:
        env: The Gymnasium environment instance.
        approach_z_margin: Height above object for approach.
        place_z_margin: Height above goal for transit/carry.
        pos_threshold: Axis-wise tolerance to consider aligned.
        open_thresh: Opening threshold to consider gripper open.
        closed_thresh: Opening threshold to consider gripper closed.
        sleep_s: Optional sleep time per step for visualization.

    Returns:
        total_reward: Sum of rewards over the episode.
        success: Whether the episode achieved success.
        steps: Number of steps taken in the episode.
    """
    obs, _ = env.reset()
    done = False
    total_reward: float = 0.0
    steps = 0

    # Get the entire sequence from the current state snapshot
    actions: list[int] = env.unwrapped.solve(max_steps=1000)

    # Optional: verify first-step alignment between planning and rollout
    if hasattr(env.unwrapped, "_last_plan_debug"):
        plan_debug = env.unwrapped._last_plan_debug
        init_obs = plan_debug.get("init_obs", {})
        plan_first = plan_debug.get("trace", [])
        if plan_first:
            p_grip, p_obj, p_ach, p_act = plan_first[0]
            # Compare with current obs before applying first action
            # If mismatch is large, print a warning for debugging
            cur_obs = env.unwrapped._get_obs()
            cg = cur_obs["observation"][0:3]
            co = cur_obs["observation"][3:6]
            if np.linalg.norm(cg - p_grip) > 1e-6 or np.linalg.norm(co - p_obj) > 1e-6:
                print("[warn] Replay start differs from plan start: |grip|=", float(np.linalg.norm(cg - p_grip)), 
                      "|obj|=", float(np.linalg.norm(co - p_obj)))

    # For deterministic contact replay, disable warmstart during replay as well
    disableflags_backup = None
    try:
        import mujoco  # type: ignore
        disableflags_backup = int(env.unwrapped.model.opt.disableflags)
        env.unwrapped.model.opt.disableflags = disableflags_backup | mujoco.mjtDisableBit.mjDSBL_WARMSTART
    except Exception:
        pass

    # Clear warmstart accelerations if present
    try:
        if hasattr(env.unwrapped.data, "qacc_warmstart"):
            env.unwrapped.data.qacc_warmstart[:] = 0
    except Exception:
        pass

    try:
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
            if info.get("is_success", False) > 0 or done:
                break
            if env.render_mode == "human" and sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        if disableflags_backup is not None:
            try:
                env.unwrapped.model.opt.disableflags = disableflags_backup
            except Exception:
                pass

    success = bool(info.get("is_success", False))
    return total_reward, success, steps


def main(args: Args) -> None:
    """Entrypoint to run multiple episodes with the pick-and-place oracle.

    Args:
        args: Parsed CLI arguments.
    """
    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, reward_type=args.reward_type, render_mode=render_mode)

    returns: list[float] = []
    successes: list[bool] = []
    steps_taken: list[int] = []

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        ep_return, ep_success, ep_steps = run_episode(
            env=env,
            approach_z_margin=args.approach_z_margin,
            place_z_margin=args.place_z_margin,
            pos_threshold=args.pos_threshold,
            open_thresh=args.open_thresh,
            closed_thresh=args.closed_thresh,
            sleep_s=args.sleep_s,
        )
        returns.append(ep_return)
        successes.append(ep_success)
        steps_taken.append(ep_steps)
        print(
            f"Episode {ep+1}/{args.episodes}: return={ep_return:.3f}, "
            f"success={ep_success}, steps={ep_steps}"
        )

    env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_steps = float(np.mean(steps_taken)) if steps_taken else 0.0
    print(
        "\nSummary: "
        f"avg_return={avg_return:.3f}, success_rate={success_rate:.2f}, avg_steps={avg_steps:.1f}"
    )


if __name__ == "__main__":
    tyro.cli(main)



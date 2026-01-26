"""Run FetchReachDiscrete with an oracle-derived open-loop action sequence.

This script executes one or more episodes in the discrete-action Fetch Reach
environment and uses the environment-provided `solve()` to obtain
an open-loop sequence of greedy actions that move the end-effector toward the goal.

Usage:
  python -m examples.run_fetch_reach_discrete_oracle --episodes 5 --render True
  python -m examples.run_fetch_reach_discrete_oracle --args.episodes 5 --args.render --args.reward_type sparse
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
    """CLI arguments for running the discrete oracle.

    Attributes:
        env_id: Gymnasium env id to create.
        episodes: Number of episodes to run.
        render: Whether to render in a window.
        reward_type: Reward configuration for the env.
        seed: Base random seed; incremented per-episode.
        sleep_s: Optional per-step sleep in seconds for visualization.
        axis_threshold: Per-axis threshold for oracle to decide noop.
    """

    env_id: str = "FetchReachDiscrete-v4"
    episodes: int = 3
    render: bool = True
    reward_type: Literal["sparse", "dense"] = "sparse"
    seed: int = 0
    sleep_s: float = 0.0
    axis_threshold: float = 0.002


def run_episode(env: gym.Env, axis_threshold: float, sleep_s: float) -> Tuple[float, bool, int]:
    """Run a single episode using the env's ground-truth action sequence.

    Args:
        env: The Gymnasium environment instance.
        axis_threshold: Per-axis tolerance for noop in the oracle.
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

    # Basic safety check that we are using the discrete oracle-capable env
    if not hasattr(env.unwrapped, "solve"):
        raise RuntimeError(
            "Environment does not expose `solve`. Use FetchReachDiscrete-* env ids."
        )

    # Compute the full open-loop sequence from the current state snapshot
    actions: list[int] = env.unwrapped.solve()

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps += 1
        if info.get("is_success", False) > 0 or done:
            break
        if env.render_mode == "human" and sleep_s > 0:
            time.sleep(sleep_s)

    success = bool(info.get("is_success", False))
    return total_reward, success, steps


def main(args: Args) -> None:
    """Entrypoint to run multiple episodes with the oracle controller.

    Args:
        args: Parsed CLI arguments.
    """
    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, reward_type=args.reward_type, render_mode=render_mode)

    # Run episodes
    returns: list[float] = []
    successes: list[bool] = []
    steps_taken: list[int] = []

    for ep in range(args.episodes):
        # Reseed for variability
        env.reset(seed=args.seed + ep)
        ep_return, ep_success, ep_steps = run_episode(
            env=env, axis_threshold=args.axis_threshold, sleep_s=args.sleep_s
        )
        returns.append(ep_return)
        successes.append(ep_success)
        steps_taken.append(ep_steps)
        print(
            f"Episode {ep+1}/{args.episodes}: return={ep_return:.3f}, "
            f"success={ep_success}, steps={ep_steps}"
        )

    env.close()

    # Summary
    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_steps = float(np.mean(steps_taken)) if steps_taken else 0.0
    print(
        "\nSummary: "
        f"avg_return={avg_return:.3f}, success_rate={success_rate:.2f}, avg_steps={avg_steps:.1f}"
    )


if __name__ == "__main__":
    tyro.cli(main)


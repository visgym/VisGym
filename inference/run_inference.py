#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-Gym Inference Runner

Run VLM inference on pre-saved environment states.

Example:
    python run_inference_from_state.py \
        --model "qwen/qwen3-vl-235b-a22b-instruct" \
        --api_option openrouter \
        --episodes_root ./inference_dataset/test_set_v10_easy \
        --outdir debug \
        --make-gif
"""

# Fix for MuJoCo OpenGL issues in headless environments - MUST be before other imports
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import datetime as dt
import json
import multiprocessing as mp
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Optional

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from tqdm import tqdm

from gymnasium.vlm.vlm import TransformersVLM, OpenAIVLM
from gymnasium.vlm.interactor import Interactor
from utils import ENV_REGISTRY
from visualize_gifs import save_episode_gif, list_render_images

gym.register_envs(gymnasium_robotics)

# API base URLs
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _env_slug(env_id: str) -> str:
    """Convert env_id to a filesystem-safe slug."""
    return env_id.replace("/", "__").replace(":", "__")


def iter_jsonl_files(root_or_file: str):
    """Iterate over JSONL files in a directory (recursively) or yield a single file."""
    p = Path(root_or_file)
    if p.is_file():
        yield p
        return
    for f in sorted(p.rglob("*.jsonl")):
        yield f


def load_episodes(jsonl_path: Path):
    """Load episodes from a JSONL file, yielding one dict per line."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _json_safe(obj):
    """Recursively convert numpy types to JSON-serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _require_env(var_names: list) -> str:
    """Return the first set env var among var_names, else raise a helpful error."""
    for v in var_names:
        val = os.getenv(v)
        if val:
            return val.strip()
    raise RuntimeError(
        f"Missing API key. Please export one of: {', '.join(var_names)}\n"
        "  e.g., export OPENAI_API_KEY='sk-...'\n"
        "        export OPENROUTER_API_KEY='or-...'"
    )


def build_vlm(api_option: str, model_name: str):
    """Build a VLM instance based on the API option and model name."""
    if api_option == "openai":
        api_key = _require_env(["OPENAI_API_KEY", "OPENAI_KEY"])
        ok = any(k in model_name.lower() for k in ("gpt", "o1", "o3", "o4", "4o"))
        if not ok:
            raise AssertionError(
                f"Model '{model_name}' doesn't look like an OpenAI model."
            )
        return OpenAIVLM(api_key=api_key, model_name=model_name)

    elif api_option == "openrouter":
        api_key = _require_env(["OPENROUTER_API_KEY"])
        return OpenAIVLM(api_key=api_key, model_name=model_name, base_url=OPENROUTER_BASE_URL)

    elif api_option == "transformers":
        return TransformersVLM(model_name=model_name)

    else:
        raise ValueError(f"Invalid api_option: {api_option}")


def make_env_from_registry(env_id: str, episode_seed: int):
    """
    Build env using ENV_REGISTRY.
    If registry has a 'seed_key', inject the episode_seed into env kwargs at make().
    """
    if env_id not in ENV_REGISTRY:
        env = gym.make(env_id)
        return env, None, None

    registry_entry = ENV_REGISTRY[env_id]
    env_kwargs = deepcopy(registry_entry.get("env_kwargs", {}))
    seed_key = registry_entry.get("seed_key", None)
    extra_state_key = registry_entry.get("extra_state", None)

    if seed_key is not None:
        env_kwargs[seed_key] = episode_seed

    env = gym.make(env_id, **env_kwargs)
    return env, seed_key, extra_state_key


def save_step_render(render_img, out_dir: Path, step_idx: int):
    """Save a single step render image."""
    try:
        import matplotlib.pyplot as plt
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step_idx:04d}.png"
        plt.imsave(path.as_posix(), render_img)
        plt.close()
    except Exception:
        pass


def set_global_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_inference_on_episode(
    env,
    extra_state_key,
    env_id: str,
    seed: int,
    episode_seed: int,
    render: bool,
    max_steps: int,
    verbose: bool,
    save_verbose: bool,
    out_episode_dir: Path,
    max_long_side: int,
    elicit_verbalization: bool,
    include_env_feedback: bool,
    num_interaction_history: int,
    make_gif: bool = False,
    gif_ms_per_frame: int = 300,
    gif_start_hold_ms: int = 2500,
    gif_end_hold_ms: int = 2000,
    gif_max_text_chars: int = 400,
    gif_max_width: int = 1024,
    gif_preserve_quality: bool = True,
    init_state: Optional[dict] = None,
    provide_final_goal: bool = False,
    text_mode: bool = False,
):
    """
    Run one Interactor episode, save stats and optional renders.
    """
    vlm = get_vlm()
    interactor = Interactor(vlm, env, max_steps=max_steps, master_seed=seed, max_long_side=max_long_side)
    
    if text_mode:
        stats = interactor.run_episode_text_mode(
            render=render,
            verbose=verbose,
            save_verbose=save_verbose,
            seed=episode_seed,
            extra_state=extra_state_key,
            elicit_verbalization=elicit_verbalization,
            include_env_feedback=include_env_feedback,
            num_interaction_history=num_interaction_history,
            init_state=init_state,
        )
    else:
        stats = interactor.run_episode(
            render=render,
            verbose=verbose,
            save_verbose=save_verbose,
            seed=episode_seed,
            extra_state=extra_state_key,
            elicit_verbalization=elicit_verbalization,
            include_env_feedback=include_env_feedback,
            num_interaction_history=num_interaction_history,
            init_state=init_state,
            provide_final_goal=provide_final_goal,
        )

    out_episode_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "env_id": env_id,
        "seed": seed,
        "episode_seed": episode_seed,
        "timestamp": dt.datetime.now().isoformat(),
        "init_args": interactor.init_args,
        "run_args": interactor._run_args,
        "history": interactor.hist,
        "stats": {
            "step": stats.get("step"),
            "reward": stats.get("reward"),
            "terminated": stats.get("terminated"),
            "truncated": stats.get("truncated"),
        },
        "version": "v1",
        "extra_state": stats.get("extra_state", None),
    }

    # Write stats JSON
    with open((out_episode_dir / "episode_stats.json").as_posix(), "w") as f:
        json.dump(_json_safe(result), f, indent=2)

    # Write history JSONL (one line per step)
    if save_verbose and interactor.hist:
        with open((out_episode_dir / "history.jsonl").as_posix(), "w") as f:
            for step in interactor.hist:
                f.write(json.dumps(step) + "\n")

    # Save render frames
    if render and interactor.hist:
        for i, step in enumerate(interactor.hist):
            img = None
            for k in ("image", "frame", "obs", "render", "rgb"):
                if k in step:
                    img = step[k]
                    break
            if img is None:
                try:
                    img = env.render(mode='rgb_array')
                except Exception:
                    img = None
            if img is not None:
                arr = np.array(img)
                save_step_render(arr, out_episode_dir / "renders", i)

    # Create a per-episode GIF
    if make_gif:
        try:
            frame_paths = list_render_images(out_episode_dir)
            reward_val = result.get("stats", {}).get("reward", None)
            if isinstance(reward_val, (int, float)):
                reward_str = f"{float(reward_val):.3f}"
            else:
                reward_str = "NA"
            tag = f"seed{seed}_episode{episode_seed}"
            env_slug_dir = out_episode_dir.parent
            out_gif = env_slug_dir / f"{tag}_r{reward_str}.gif"
            save_episode_gif(
                episode_data=result,
                frame_paths=frame_paths,
                out_gif=out_gif,
                ms_per_frame=gif_ms_per_frame,
                start_hold_ms=gif_start_hold_ms,
                end_hold_ms=gif_end_hold_ms,
                max_text_chars=gif_max_text_chars,
                max_width=gif_max_width,
                captions=True,
                preserve_quality=gif_preserve_quality,
            )
        except Exception:
            pass

    return result


# Global state for multiprocessing workers
args = None
out_root: Optional[Path] = None
_vlm_cache = None


def _init_worker(ns, out_root_str):
    """Initialize worker process with shared state."""
    global args, out_root, _vlm_cache
    args = ns
    out_root = Path(out_root_str)
    out_root.mkdir(parents=True, exist_ok=True)
    _vlm_cache = None  # Reset cache for new worker


def get_vlm():
    """Get or create cached VLM instance for this worker."""
    global _vlm_cache
    if _vlm_cache is None:
        _vlm_cache = build_vlm(args.api_option, args.model)
    return _vlm_cache


def run_inference_on_episode_mp(jsonl_path):
    """Process episodes from a JSONL file (multiprocessing worker function)."""
    global args, out_root

    env = None
    extra_state_key = None

    for i, ep in enumerate(tqdm(load_episodes(jsonl_path))):
        env_id = ep.get("env_id")
        seed = ep.get("seed")
        episode_seed = ep.get("episode_seed")

        # Skip if output already exists
        if (out_root / _env_slug(env_id) / f"seed{seed}_episode{episode_seed}").exists():
            continue

        # Initialize env on first episode
        if env is None:
            set_global_seed(seed)
            env, _, extra_state_key = make_env_from_registry(env_id, seed)

        if args.env_filter and (env_id not in args.env_filter):
            continue

        env_slug = _env_slug(env_id)
        tag = f"seed{seed}_episode{episode_seed}"
        out_dir = out_root / env_slug / tag
        init_dir = Path(args.initdir) / env_slug / tag

        # Load initial state
        initial_state_file = init_dir / "initial_state.json"
        with open(initial_state_file, 'r') as f:
            init_state = json.load(f)
        run_inference_on_episode(
            env=env,
            extra_state_key=extra_state_key,
            env_id=env_id,
            seed=int(seed),
            episode_seed=int(episode_seed),
            render=args.render,
            max_steps=args.max_steps,
            verbose=args.verbose,
            save_verbose=args.save_verbose,
            out_episode_dir=out_dir,
            max_long_side=args.max_long_side,
            elicit_verbalization=args.elicit_verbalization,
            include_env_feedback=args.include_env_feedback,
            num_interaction_history=args.num_interaction_history,
            make_gif=args.make_gif,
            gif_ms_per_frame=args.gif_ms_per_frame,
            gif_start_hold_ms=args.gif_start_hold_ms,
            gif_end_hold_ms=args.gif_end_hold_ms,
            gif_max_text_chars=args.gif_max_text_chars,
            gif_max_width=args.gif_max_width,
            gif_preserve_quality=args.gif_preserve_quality,
            init_state=init_state,
            provide_final_goal=args.provide_final_goal,
            text_mode=args.text_mode,
        )
        print(f"Saved episode to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="VLM-Gym Inference Runner")

    # Model settings
    p.add_argument("--model", required=True,
                   help="Model name (OpenAI, OpenRouter alias, or HF Transformers id)")
    p.add_argument("--api_option", choices=["openai", "openrouter", "transformers"],
                   default="openrouter", help="API backend to use")

    # Input/output paths
    p.add_argument("--episodes_root", required=True,
                   help="Path to directory containing JSONL shards, or a single JSONL file")
    p.add_argument("--outdir", default="runs/infer",
                   help="Output root directory")
    p.add_argument("--initdir", default="./inference_dataset/initial_states",
                   help="Directory containing initial states")

    # Inference settings
    p.add_argument("--max-steps", type=int, default=20,
                   help="Maximum steps per episode")
    p.add_argument("--max-long-side", type=int, default=336,
                   help="Maximum long side for image rendering")
    p.add_argument("--env-filter", nargs="+", default=None,
                   help="Only run these env ids (must match JSONL 'env_id')")
                   
    # Prompt settings
    p.add_argument("--elicit-verbalization", action="store_true", default=False,
                   help="Explicitly ask for reasoning verbalization")
    p.add_argument("--include-env-feedback", action="store_true", default=False,
                   help="Include textual feedback (action executed, action not parsable, etc.)")
    p.add_argument("--num-interaction-history", type=int, default=-1,
                   help="Number of interaction history steps to include (-1 = all)")
    p.add_argument("--provide-final-goal", action="store_true", default=False,
                   help="Provide the final goal to the VLM")
    p.add_argument("--text-mode", action="store_true", default=False,
                   help="Run in text mode")
    # Output settings
    p.add_argument("--render", action="store_true",
                   help="Save per-step renders if available")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output")
    p.add_argument("--save-verbose", action="store_true",
                   help="Write step-by-step history.jsonl")

    # GIF generation
    p.add_argument("--make-gif", action="store_true",
                   help="Generate a GIF per episode")
    p.add_argument("--gif-ms-per-frame", type=int, default=800)
    p.add_argument("--gif-start-hold-ms", type=int, default=2500)
    p.add_argument("--gif-end-hold-ms", type=int, default=2000)
    p.add_argument("--gif-max-text-chars", type=int, default=400)
    p.add_argument("--gif-max-width", type=int, default=1024)
    p.add_argument("--gif-preserve-quality", action="store_true")

    args_ns = p.parse_args()

    out_root_path = Path(args_ns.outdir)
    out_root_path.mkdir(parents=True, exist_ok=True)

    # Collect JSONL files
    jsonl_paths = sorted(iter_jsonl_files(args_ns.episodes_root), reverse=False)

    print(f"Found {len(jsonl_paths)} JSONL files")

    # Run inference using multiprocessing
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=1,
        initializer=_init_worker,
        initargs=(args_ns, str(out_root_path)),
        maxtasksperchild=1,
    ) as pool:
        list(pool.imap_unordered(run_inference_on_episode_mp, jsonl_paths))

    print(f"\nDone. Outputs saved to {out_root_path}")


if __name__ == "__main__":
    main()

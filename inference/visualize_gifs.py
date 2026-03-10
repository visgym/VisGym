#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create captioned GIF visualizations from VLM-Gym inference outputs.

This script expects an episode folder produced by inference/run_inference.py
containing:
  - episode_stats.json
  - renders/step_*.png (or other image formats) when --render was used

You can point --input to:
  - a single episode_stats.json file
  - a single episode directory (containing episode_stats.json)
  - a root directory; the script will recursively find all episode_stats.json

Examples:
  # Single episode
  python inference/visualize_gifs.py \
    --input VisGym/inference/eval/openai_gpt_5_v4_new/colorization__ColorizationEnv-v2/seed10716_episode1233051550/episode_stats.json

  # All episodes under a root
  python inference/visualize_gifs.py \
    --input VisGym/inference/eval/openai_gpt_5_v5 --out-root VisGym/inference/results/openai_gpt_5_v5

Output:
  By default, writes episode.gif inside each episode directory.
  Use --out-root to mirror episode structure under another root.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# -------------------------- Image helpers (robust loader) --------------------------

def _to_pil(img_like):
    """
    Convert various image representations to PIL.Image.
    Supports:
      - np.ndarray (H,W,[C]) uint8 or float [0,1]
      - PIL.Image
      - dict with [image, rgb, frame, render, obs, pixels]
      - base64 string (optionally data URL) or bytes
      - file path string
    """
    if img_like is None:
        return None

    # Already a PIL image
    if isinstance(img_like, Image.Image):
        return img_like

    # Dict with a nested image
    if isinstance(img_like, dict):
        for k in ("image", "rgb", "frame", "render", "obs", "pixels"):
            if k in img_like:
                im = _to_pil(img_like[k])
                if im is not None:
                    return im
        return None

    # Bytes-like → try PIL
    if isinstance(img_like, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img_like)).convert("RGB")
        except Exception:
            return None

    # String cases
    if isinstance(img_like, str):
        s = img_like.strip()

        # data URL?
        if s.startswith("data:image"):
            try:
                _, b64 = s.split(",", 1)
                data = base64.b64decode(b64)
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                return None

        # Try plain base64 (PNG/JPEG/GIF magic check)
        try:
            data = base64.b64decode(s, validate=True)
            if (
                data[:8] == b"\x89PNG\r\n\x1a\n"
                or data[:2] == b"\xff\xd8"
                or data[:6] in (b"GIF87a", b"GIF89a")
            ):
                return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            pass

        # Maybe it's a file path
        if os.path.exists(s):
            try:
                return Image.open(s).convert("RGB")
            except Exception:
                return None

        return None

    # Numpy array
    try:
        import numpy as _np

        arr = _np.array(img_like)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            arr = _np.transpose(arr, (1, 2, 0))  # CHW→HWC
        if arr.dtype != _np.uint8:
            arr = _np.clip(arr, 0, 1) if arr.max() <= 1.0 else _np.clip(arr, 0, 255)
            arr = (arr * 255).astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
        if arr.ndim == 2:
            arr = _np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(arr)
    except Exception:
        return None


def _wrap_lines(lines: List[str], draw: ImageDraw.ImageDraw, font: Optional[ImageFont.ImageFont], max_width_px: int) -> List[str]:
    wrapped: List[str] = []
    for line in [l for l in lines if l]:
        words = str(line).split()
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width_px:
                cur = test
            else:
                if cur:
                    wrapped.append(cur)
                cur = w
        if cur:
            wrapped.append(cur)
    return wrapped


def _compose_captioned_frame(img: Image.Image, lines: List[str], max_width: Optional[int] = None, pad: int = 12, line_gap: int = 8, bg: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if max_width and img.width > max_width:
        h = int(img.height * (max_width / img.width))
        img = img.resize((max_width, h), Image.Resampling.LANCZOS)

    tmp = Image.new("RGB", (img.width, 10), bg)
    draw = ImageDraw.Draw(tmp)
    try:
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Times New Roman.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 14)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = None

    wrapped = _wrap_lines(lines, draw, font, img.width - 2 * pad)
    if font:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = bbox[3] - bbox[1]
    else:
        line_h = 12

    caption_h = 0 if not wrapped else (len(wrapped) * line_h + (len(wrapped) - 1) * line_gap)
    H = img.height + (0 if caption_h == 0 else (pad + caption_h + pad))
    canvas = Image.new("RGB", (img.width, H), bg)
    canvas.paste(img, (0, 0))

    y = img.height + pad
    draw_canvas = ImageDraw.Draw(canvas)
    for line in wrapped:
        draw_canvas.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h + line_gap
    return canvas


# -------------------------- Episode IO --------------------------

def find_episode_jsons(input_path: Path) -> Iterable[Path]:
    """Yield episode_stats.json paths for a file/dir tree."""
    if input_path.is_file():
        if input_path.name == "episode_stats.json":
            yield input_path
        else:
            # treat file as the JSON
            yield input_path
        return

    # directory
    stats = list(input_path.rglob("episode_stats.json"))
    for p in sorted(stats):
        yield p


def load_episode(json_path: Path) -> Tuple[dict, Path]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episode_dir = json_path.parent
    return data, episode_dir


def list_render_images(episode_dir: Path) -> List[Path]:
    renders_dir = episode_dir / "renders"
    if not renders_dir.exists() or not renders_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in sorted(renders_dir.iterdir()) if p.suffix.lower() in exts]
    return imgs


def _truncate_text(s: str, max_chars: int) -> str:
    s = s.replace("\n", " ")
    if len(s) <= max_chars:
        return s
    truncated = s[: max_chars - 3]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


def build_caption_lines(step: dict, step_index: int, max_text_chars: int) -> List[str]:
    lines: List[str] = []
    prompt = step.get("prompt")
    if prompt:
        lines.append(_truncate_text(str(prompt), max_text_chars))
    lines.append(f"step {step_index}")

    action = step.get("action")
    if action is not None:
        lines.append(_truncate_text(str(action), max_text_chars))

    reason = step.get("verbalization") or step.get("reason") or step.get("vlm_output") or step.get("env_feedback")
    if reason:
        lines.append(_truncate_text(str(reason), max_text_chars))

    reward = step.get("reward")
    if reward is not None:
        try:
            lines.append(f"reward={float(reward):.3f}")
        except Exception:
            lines.append(f"reward={reward}")
    return lines


def save_episode_gif(
    episode_data: dict,
    frame_paths: List[Path],
    out_gif: Path,
    ms_per_frame: int = 300,
    start_hold_ms: Optional[int] = 2500,
    end_hold_ms: Optional[int] = 2000,
    max_text_chars: int = 400,
    max_width: Optional[int] = 1024,
    captions: bool = True,
    preserve_quality: bool = True,
) -> bool:
    history = episode_data.get("history") or []

    frames: List[Image.Image] = []
    num_frames = len(frame_paths)
    if num_frames == 0:
        # Fallback: try to render from history (if images embedded)
        for i, step in enumerate(history):
            img = None
            for k in ("obs", "frame", "image", "render", "extras"):
                if k in step:
                    img = _to_pil(step[k])
                    if img is not None:
                        break
            if img is None:
                continue
            lines = build_caption_lines(step, i, max_text_chars) if captions else []
            frames.append(_compose_captioned_frame(img, lines, max_width=None if preserve_quality else max_width))
    else:
        for i, fp in enumerate(frame_paths):
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            step = history[i] if i < len(history) else {}
            lines = build_caption_lines(step, i, max_text_chars) if captions else []
            frames.append(_compose_captioned_frame(img, lines, max_width=None if preserve_quality else max_width))

    if not frames:
        return False

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    # Per-frame durations
    per_frame_ms = [max(20, int(ms_per_frame)) for _ in frames]
    if start_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[0] = max(20, int(start_hold_ms))
    if end_hold_ms is not None and len(per_frame_ms) > 0:
        per_frame_ms[-1] = max(20, int(end_hold_ms))

    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=per_frame_ms,
        disposal=2,
        optimize=True,
        quality=100,
    )
    return True

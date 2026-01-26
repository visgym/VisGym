import json, base64, io
from pathlib import Path
import ipywidgets as w
from IPython.display import display, HTML
from PIL import Image

_PRE = "white-space:pre-wrap;font-size:0.82em;line-height:1.25;margin:0;"

def _acc(child, title):
    box = w.Accordion([child])
    box.set_title(0, title)
    box.selected_index = None
    return box

def _b64_img_tag(b64, px):
    return f"<img src='data:image/jpeg;base64,{b64}' style='max-width:{px}px;'>"

def view_episode(json_path: str | Path, img_px: int = 300):
    """
    Display every step stored in a HumanInteractor-generated JSON log.

    Parameters
    ----------
    json_path : str or Path
        Path to the saved episode JSON.
    img_px : int, default 300
        Longest-side pixel size for the displayed screenshots.
    """
    json_path = Path(json_path).expanduser()
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    init_args = data.get("init_args", {})
    run_args  = data.get("run_args", {})
    history   = data["history"]

    # Episode header
    header_html = (
        "<h3>Episode metadata</h3>"
        "<b>Init args:</b><pre style='{0}'>{1}</pre>"
        "<b>Run args:</b><pre style='{0}'>{2}</pre>"
        "<hr/>"
    ).format(_PRE, json.dumps(init_args, indent=2), json.dumps(run_args, indent=2))
    display(HTML(header_html))

    # Per-step widgets
    for step in history:
        img_b64 = step["image"]
        prompt  = step.get("prompt", "")
        think   = step.get("think", "")
        # payload or action; if none, then empty string
        answer  = step.get("answer", step.get("action", step.get("payload", "")))
        reward  = step.get("reward", 0)
        fb_text = ""
        info    = step.get("info", {})
        if isinstance(info, dict) and "env_feedback" in info:
            fb_text = info["env_feedback"]

        img_html    = w.HTML(_b64_img_tag(img_b64, img_px))
        prompt_html = w.HTML(f"<pre style='{_PRE}'>{prompt}</pre>")
        answer_html = w.HTML(f"<pre style='{_PRE}'>{answer}</pre>")
        reward_html = w.HTML(f"<pre style='{_PRE}'>Reward: {reward}</pre>")

        parts = [
            _acc(img_html,    "Image"),
            _acc(prompt_html, "Prompt")
        ]
        if think:
            think_html = w.HTML(f"<pre style='{_PRE}'>{think}</pre>")
            parts.append(_acc(think_html, "Reasoning"))
        parts.append(_acc(answer_html, "Answer"))
        if fb_text:
            fb_html = w.HTML(f"<pre style='{_PRE}'>{fb_text}</pre>")
            parts.append(_acc(fb_html, "Env Feedback"))
        parts.append(_acc(reward_html, "Reward"))

        step_box = w.VBox([w.HTML(f"<b>Step {step['step']}</b>")] + parts)
        display(step_box)

# interactor.py
import base64
import io
import json
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import re
import gymnasium as gym
import numpy as np
from PIL import Image
import ipywidgets as w
from IPython.display import display

from .vlm import BaseVLM
import re
import ast

# ---------------------------------------------------------------------
# Widget helper – now ordered:
#   1. Image   2. Input Prompt   3. VLM Output   4. Env Feedback
# ---------------------------------------------------------------------
def _make_step_widget(
    step_idx: int,
    img_b64: str,
    prompt: str,
    vlm_out: str | None = None,   # show ⏳ until available
    env_fb: str | None = None,    # blank until env tells us
    img_px: int = 250,
) -> Tuple[w.VBox, w.HTML, w.HTML]:
    """Return (box, out_html, fb_html)."""
    # --- small CSS for all <pre> blocks ---
    _PRE_STYLE = (
        "white-space:pre-wrap;"
        "font-size:0.82em;"
        "line-height:1.25;"
        "margin:0;"
    )

    # plain content widgets ---------------------------------------------------
    img_html  = w.HTML(
        f"<img src='data:image/jpeg;base64,{img_b64}' "
        f"style='max-width:{img_px}px;'>"
    )
    prompt_html = w.HTML(f"<pre style='{_PRE_STYLE}'>{prompt}</pre>")
    out_html    = w.HTML(
        f"<pre style='{_PRE_STYLE}'>{vlm_out or '⏳ …'}</pre>"
    )
    fb_html     = w.HTML(
        f"<pre style='{_PRE_STYLE}'>{env_fb or ''}</pre>"
    )

    # each piece in its own 1-child accordion so they open independently ------
    def _solo_acc(child: w.Widget, title: str) -> w.Accordion:
        acc = w.Accordion([child])
        acc.set_title(0, title)
        acc.selected_index = None
        return acc

    img_acc  = _solo_acc(img_html,  "Image")
    prm_acc  = _solo_acc(prompt_html, "Input Prompt")
    out_acc  = _solo_acc(out_html,   "VLM Output")
    fb_acc   = _solo_acc(fb_html,    "Env Feedback")

    # header label for the step ----------------------------------------------
    header = w.HTML(f"<b>Step&nbsp;{step_idx}</b>")

    # top-level VBox for this step -------------------------------------------
    box = w.VBox([header, img_acc, prm_acc, out_acc, fb_acc])
    return box, out_html, fb_html


# ---------------------------------------------------------------------
# Optional plain-Matplotlib renderer (unchanged from your draft)
# ---------------------------------------------------------------------
def _default_render_fn(env, episode_stats, abs_size=300):
    import matplotlib.pyplot as plt

    im = env.render()  # rgb_array
    h, w = im.shape[:2]
    scale = abs_size / max(h, w)
    fig_w, fig_h = w * scale / 100, h * scale / 100  # inches (≈100 dpi)

    plt.figure(figsize=(fig_w, fig_h), dpi=100)
    plt.imshow(im)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------
# The updated Interactor
# ---------------------------------------------------------------------
class Interactor:
    def __init__(
        self,
        vlm: BaseVLM,
        env: gym.Env,
        max_steps: int = 100,
        render_fn: Callable = None,
        abs_size: int = 300,
        master_seed: Optional[int] = 42,
        max_long_side: Optional[int] = None,
        image_quality: Optional[int] = None,
        num_step_range: Optional[Tuple[int, int]] = None,
    ):
        self.vlm = vlm
        self.env = env
        self.max_steps = max_steps
        self.abs_size = abs_size
        self.max_long_side = max_long_side
        self.image_quality = image_quality
        if master_seed is not None:
            self.master_rng = np.random.RandomState(master_seed)
        else:
            self.master_rng = None

        # unwrap Gymnasium wrappers
        while hasattr(self.env, "env"):
            self.env = self.env.env

        self.render_fn = (
            lambda e, s: _default_render_fn(e, s, self.abs_size)
            if render_fn is None
            else render_fn
        )

        # Trajectory saving attributes
        self.hist: List[Dict[str, Any]] = []
        self._run_args: Dict[str, Any] = {}
        self.init_args = dict(
            env_repr=str(env),
            vlm_repr=str(vlm),
            max_steps=max_steps,
            abs_size=abs_size,
            master_seed=master_seed,
            max_long_side=max_long_side,
            image_quality=image_quality,
        )
        if render_fn is not None:
            self.init_args["custom_render_fn"] = repr(render_fn)

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _xml_prompt() -> str:
        # return (
        #     "\n\nYou MUST verbalize your reasoning in <think>…</think> and give the "
        #     "final result in <answer>…</answer>."
        # )
        return (
            "\n\nVerbalize your reasoning in <think>…</think> and give the "
            "final result in <answer>…</answer>."
        )

    def _image_to_b64(self, img: np.ndarray | Image.Image) -> str:
        pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        
        # Resize image if max_long_side is specified
        if self.max_long_side is not None:
            w, h = pil.size
            max_dim = max(w, h)
            if max_dim > self.max_long_side:
                scale = self.max_long_side / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
        if pil.mode in ("I", "F", "I;16", "I;16B", "P", "LA"):
            pil = pil.convert("RGB")
        elif pil.mode == "RGBA":
            pil = pil.convert("RGB")  
            
        buf = io.BytesIO()
        if self.image_quality is not None:
            pil.save(buf, format="JPEG", quality=self.image_quality)
        else:
            pil.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _safe(x: Any) -> Any:
        """Make object JSON serializable."""
        try:
            json.dumps(x)
            return x
        except TypeError:
            return x.tolist() if isinstance(x, np.ndarray) else str(x)

    def save_history(self, path: str | Path) -> str:
        """Save interaction history to JSON file."""
        pkg = dict(
            init_args=self.init_args,
            run_args=self._run_args,
            history=self.hist
        )
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(pkg, f, indent=2)
        print("Saved interaction log to:", p)
        return str(p)
    

    @staticmethod
    def _normalize(text: str) -> str:
        t = str(text)

        # Remove code fences like ```json or ```python and closing ```
        t = re.sub(r"```[\w-]*", "", t)
        t = t.replace("```", "")

        # Remove inline backticks
        t = t.replace("`", "")

        # Remove special box / think markers commonly seen in traces
        t = t.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        t = t.replace("\u25c1think\u25b7", "").replace("\u25c1/think\u25b7", "")

        # Drop "Action:" labels (case-insensitive)
        t = re.sub(r"\bAction\s*:\s*", " ", t, flags=re.IGNORECASE)

        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # Detects a simple bare identifier used as the first tuple element, e.g. (reorder, [...])
    # and quotes it -> ('reorder', [...])
    _IDENT = re.compile(r"""
        ^\(\s*                 # opening paren
        ([A-Za-z_]\w*)         # bare identifier
        (\s*,)                 # followed by comma
    """, re.VERBOSE)
    
    @staticmethod
    def _quote_first_identifier(s: str) -> str:
        m = Interactor._IDENT.search(s)
        if not m:
            return s
        ident, comma = m.group(1), m.group(2)
        return s[:m.start()] + "(" + f"'{ident}'" + comma + s[m.end():]
        
    @staticmethod
    def _is_action_tuple(obj) -> bool:
        # Accept tuples with at least one element and string action name
        return isinstance(obj, tuple) and len(obj) >= 1 and isinstance(obj[0], str)

    @staticmethod
    def action_correctness(action: str) -> bool:
        if not action or action[0] != "(" or action[-1] != ")":
            return False
        try:
            node = ast.literal_eval(action)
        except Exception:
            return False
        return Interactor._is_action_tuple(node)


    def parse_output(self, output: str, prefer: str = "last") -> str:
        """
        prefer: 'last' (default) or 'first' — which valid action tuple to return.
        """

        text = Interactor._normalize(output)
        if not text:
            return output

        candidates = []
        n = len(text)
        i = 0
        while i < n:
            if text[i] != "(":
                i += 1
                continue
            depth = 0
            j = i
            while j < n:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        raw = text[i:j+1]
                        attempt = Interactor._quote_first_identifier(raw)
                        try:
                            node = ast.literal_eval(attempt)
                            if Interactor._is_action_tuple(node):
                                if prefer == "first":
                                    return attempt            # early return
                                candidates.append(attempt)
                        except Exception:
                            pass
                        j += 1
                        break
                j += 1
            i += 1

        return candidates[-1] if candidates else output

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def run_episode(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_verbose: bool = False,
        elicit_verbalization: bool = False,
        repeat_instructions: bool = False,
        include_env_feedback: bool = False,
        num_interaction_history: int = -1,  # -1 means all
        provide_final_goal: bool = False,
        widget_log: bool = False,  # NEW
        autosave_path: str | None = None,
        save_history_at_episode_end: bool = False,
        extra_state: Optional[str] = None,
        init_state: Optional[dict] = None,
    ) -> Dict[str, Any]:
        # Determine the seed for this episode
        if seed is not None:
            episode_seed = seed
        elif self.master_rng is not None:
            # Generate a new deterministic seed for this episode
            episode_seed = self.master_rng.randint(0, 2**32 - 1)
        else:
            episode_seed = None  # Let the env handle it (usually random)

        # Store run-time args for provenance
        self._run_args = dict(
            render=render,
            seed=episode_seed,
            verbose=verbose,
            save_verbose=save_verbose,
            elicit_verbalization=elicit_verbalization,
            repeat_instructions=repeat_instructions,
            include_env_feedback=include_env_feedback,
            num_interaction_history=num_interaction_history,
            widget_log=widget_log,
            autosave_path=autosave_path,
            save_history_at_episode_end=save_history_at_episode_end,
        )

        # Clear history for new episode
        self.hist.clear()
        
        obs, info = self.env.reset(init_state=init_state, seed=episode_seed)

        final_goal = None
        if provide_final_goal:
            from copy import deepcopy
            env_solved = deepcopy(self.env) # copy the env to avoid modifying the original env
            actions = env_solved.solve()
            for action in actions:
                obs, reward, terminated, truncated, info = env_solved.step(str(action))
            if terminated and reward == 1:
                final_goal = env_solved.render()
            else:
                raise ValueError("The episode did not terminate successfully.")
            del env_solved # free memory

        self.vlm.reset_conversation()

        stats: Dict[str, Any] = dict(
            step=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            final_proportion_matched=0,
            extra_state=(getattr(self.env.unwrapped, extra_state, None) if extra_state else None),
        )
        widgets: List[w.Accordion] = [] if widget_log else []
        image_b64s = []
        while True:
            if stats["step"] >= self.max_steps:
                stats["truncated"] = True
                break

            frame = self.env.render()
            img_b64 = self._image_to_b64(frame)
            # append all the history images to the list
            image_b64s.append(img_b64)
            instructions = self.env.get_prompt()

            if elicit_verbalization:
                instructions += self._xml_prompt()

            
            if save_verbose:
                stats["info"] = info.get("current_state", info)

            step_msg = f"This is step {stats['step'] + 1}. You are allowed to take {self.max_steps - stats['step'] - 1} more steps."

            # Build prompt
            if stats["step"] == 0 or num_interaction_history == 0:
                prompt = instructions + "\n" + step_msg
            else:
                prompt_parts = []
                if include_env_feedback and "env_feedback" in info:
                    prompt_parts.append(
                        "Environment feedback: " + info["env_feedback"]
                    )
                
                prompt_parts.append(step_msg)

                if repeat_instructions:
                    prompt_parts.append(instructions)
                
                prompt = "\n\n".join(prompt_parts)

            # Optionally display / collect pretty widgets
            if verbose or widget_log:
                # create and display the grouped widget ---------------------------------
                step_box, out_html, fb_html = _make_step_widget(
                    stats["step"] + 1,
                    img_b64,
                    prompt,
                    vlm_out=None,
                    env_fb=None,
                    img_px=self.abs_size,
                )
                display(step_box)
                if widget_log:
                    widgets.append(step_box)

            # ------------------------------------------------------------------
            # Query VLM
            # ------------------------------------------------------------------
            if num_interaction_history != -1:
                pruned = self.vlm.prune_conversation_history(num_interaction_history, instructions)

            final_goal_prompt, final_goal_img_b64 = None, None
            if provide_final_goal and stats["step"] == 0:
                assert num_interaction_history == -1, "partial history is not supported when providing the final goal"
                final_goal_prompt = f"The final goal of the episode should look like the following image: "
                final_goal_img_b64 = self._image_to_b64(final_goal)
            

            vlm_out = self.vlm.query(img_b64, prompt, 
            final_goal_prompt=final_goal_prompt, final_goal_img_b64=final_goal_img_b64)
            from copy import deepcopy
            current_conversation_history = deepcopy(self.vlm.conversation_history)
            # print("Current conversation history:", len(current_conversation_history))

            # Update the widget with the real answer (if we showed one)
            if verbose or widget_log:
                out_html.value = (
                    f"<pre style='white-space:pre-wrap;"
                    f"font-size:0.82em;line-height:1.25;margin:0;'>{vlm_out}</pre>"
                )

            # Parse action
            action = (
                vlm_out.split("<answer>")[1].split("</answer>")[0]
                if elicit_verbalization and "<answer>" in vlm_out
                else vlm_out
            )
            action = self.parse_output(action) # add robust parsing for all models

            # Environment step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Log step to history
            step_data = dict(
                step=stats["step"],
                prompt=prompt,
                vlm_output=vlm_out,
                action=action,
                reward=reward,
                image=img_b64,
                info=self._safe(info),
                conversation_history=current_conversation_history,
            )
            if elicit_verbalization and "<think>" in vlm_out:
                think_part = vlm_out.split("<think>")[1].split("</think>")[0] if "<think>" in vlm_out else None
                step_data["think"] = think_part
            self.hist.append(step_data)

            # Stats
            stats["step"] += 1
            # print("Reward:", reward)
            stats["reward"] += reward
            stats["terminated"] = terminated
            stats["truncated"] = truncated

            # fill Env Feedback panel if requested and available
            if include_env_feedback and "env_feedback" in info and (verbose or widget_log):
                fb_html.value = (
                    f"<pre style='white-space:pre-wrap;"
                    f"font-size:0.82em;line-height:1.25;margin:0;'>{info['env_feedback']}</pre>"
                )

            if terminated or truncated:
                break

            if render and not widget_log:  # avoid double-show
                self.render_fn(self.env, stats)

        # Save trajectory if requested
        if autosave_path:
            self.save_history(autosave_path)
        elif save_history_at_episode_end:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_history(f"episode_{ts}.json")

        if widget_log:
            stats["widgets"] = widgets
        return stats

    def run_episode_get_init_state(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_verbose: bool = False,
        elicit_verbalization: bool = False,
        repeat_instructions: bool = False,
        include_env_feedback: bool = False,
        num_interaction_history: int = -1,  # -1 means all
        provide_final_goal: bool = False,
        widget_log: bool = False,  # NEW
        autosave_path: str | None = None,
        save_history_at_episode_end: bool = False,
        extra_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Determine the seed for this episode
        if seed is not None:
            episode_seed = seed
        elif self.master_rng is not None:
            # Generate a new deterministic seed for this episode
            episode_seed = self.master_rng.randint(0, 2**32 - 1)
        else:
            episode_seed = None  # Let the env handle it (usually random)

        # Store run-time args for provenance
        self._run_args = dict(
            render=render,
            seed=episode_seed,
            verbose=verbose,
            save_verbose=save_verbose,
            elicit_verbalization=elicit_verbalization,
            repeat_instructions=repeat_instructions,
            include_env_feedback=include_env_feedback,
            num_interaction_history=num_interaction_history,
            widget_log=widget_log,
            autosave_path=autosave_path,
            save_history_at_episode_end=save_history_at_episode_end,
        )

        # Clear history for new episode
        self.hist.clear()
        
        obs, info = self.env.reset(seed=episode_seed)
        init_state = self.env.get_init_state()
        return init_state
    
    def run_episode_text_mode(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_verbose: bool = False,
        elicit_verbalization: bool = False,
        repeat_instructions: bool = False,
        include_env_feedback: bool = False,
        num_interaction_history: int = -1,  # -1 means all
        widget_log: bool = False,  # NEW
        autosave_path: str | None = None,
        save_history_at_episode_end: bool = False,
        extra_state: Optional[str] = None,
        init_state: Optional[dict] = None,
    ) -> Dict[str, Any]:
        # Determine the seed for this episode
        if seed is not None:
            episode_seed = seed
        elif self.master_rng is not None:
            # Generate a new deterministic seed for this episode
            episode_seed = self.master_rng.randint(0, 2**32 - 1)
        else:
            episode_seed = None  # Let the env handle it (usually random)

        # Store run-time args for provenance
        self._run_args = dict(
            render=render,
            seed=episode_seed,
            verbose=verbose,
            save_verbose=save_verbose,
            elicit_verbalization=elicit_verbalization,
            repeat_instructions=repeat_instructions,
            include_env_feedback=include_env_feedback,
            num_interaction_history=num_interaction_history,
            widget_log=widget_log,
            autosave_path=autosave_path,
            save_history_at_episode_end=save_history_at_episode_end,
        )

        # Clear history for new episode
        self.hist.clear()
        
        obs, info = self.env.reset(init_state=init_state, seed=episode_seed)
        self.env.render_mode = "ansi"
        self.vlm.reset_conversation()

        stats: Dict[str, Any] = dict(
            step=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            final_proportion_matched=0,
            extra_state=(getattr(self.env.unwrapped, extra_state, None) if extra_state else None),
        )
        widgets: List[w.Accordion] = [] if widget_log else []
        while True:
            if stats["step"] >= self.max_steps:
                stats["truncated"] = True
                break

            frame = self.env.render(mode="ansi")
            # frame = self.env._render_ansi()
            print(frame)
            # append all the history images to the list
            instructions = self.env.get_prompt()

            if elicit_verbalization:
                instructions += self._xml_prompt()

            
            if save_verbose:
                stats["info"] = info.get("current_state", info)

            step_msg = f"This is step {stats['step'] + 1}. You are allowed to take {self.max_steps - stats['step'] - 1} more steps."

            # Build prompt
            if stats["step"] == 0:
                prompt = instructions + "\n" + step_msg
            else:
                prompt_parts = []
                if include_env_feedback and "env_feedback" in info:
                    prompt_parts.append(
                        "Environment feedback: " + info["env_feedback"]
                    )
                
                prompt_parts.append(step_msg)

                if repeat_instructions:
                    prompt_parts.append(instructions)
                
                prompt = "\n\n".join(prompt_parts)

            # ------------------------------------------------------------------
            # Query VLM
            # ------------------------------------------------------------------
            if num_interaction_history != -1:
                pruned = self.vlm.prune_conversation_history(num_interaction_history, f'{instructions}')
            vlm_out = self.vlm.query_text_mode(frame + "\n" + str(prompt))

            # Parse action
            action = (
                vlm_out.split("<answer>")[1].split("</answer>")[0]
                if elicit_verbalization and "<answer>" in vlm_out
                else vlm_out
            )
            action = self.parse_output(action) # add robust parsing for all models

            # Environment step
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Log step to history
            step_data = dict(
                step=stats["step"],
                prompt=frame + "\n" + str(prompt),
                vlm_output=vlm_out,
                action=action,
                reward=reward,
                image=frame,
                info=self._safe(info)
            )
            if elicit_verbalization and "<think>" in vlm_out:
                think_part = vlm_out.split("<think>")[1].split("</think>")[0] if "<think>" in vlm_out else None
                step_data["think"] = think_part
            self.hist.append(step_data)

            # Stats
            stats["step"] += 1
            # print("Reward:", reward)
            stats["reward"] += reward
            stats["terminated"] = terminated
            stats["truncated"] = truncated

            if terminated or truncated:
                break

            if render and not widget_log:  # avoid double-show
                self.render_fn(self.env, stats)

        # Save trajectory if requested
        if autosave_path:
            self.save_history(autosave_path)
        elif save_history_at_episode_end:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_history(f"episode_{ts}.json")

        if widget_log:
            stats["widgets"] = widgets
        return stats
        

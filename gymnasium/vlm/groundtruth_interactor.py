# groundtruth_interactor.py
import base64
import io
import json
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image
import ipywidgets as w
from IPython.display import display

from .interactor import Interactor, _make_step_widget
from .vlm import BaseVLM
from .reasoner import BaseReasoner

class MockVLM(BaseVLM):
    """
    A mock VLM that doesn't actually call any external API.
    It's used by GroundtruthInteractor to maintain the same interface.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "GroundtruthVLM"
        
    def query(self, image_base64: str, prompt: str) -> str:
        """
        Mock query method that returns a placeholder response.
        This method is never actually called in GroundtruthInteractor.
        
        Args:
            image_base64: Base64 encoded image string (unused)
            prompt: The text prompt (unused)
            
        Returns:
            A placeholder response string
        """
        return "Mock VLM response - not used in groundtruth mode"


class GroundtruthInteractor(Interactor):
    """
    A subclass of Interactor that uses the environment's solve method
    instead of calling an actual VLM. This generates text sequences just like the
    regular Interactor but uses the groundtruth actions.
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_steps: int = 100,
        render_fn: Callable = None,
        abs_size: int = 300,
        master_seed: Optional[int] = 42,
        max_long_side: Optional[int] = None,
        groundtruth_strategy: str = None,
        groundtruth_kwargs: Optional[Dict[str, Any]] = None,
        reasoner: Optional[BaseReasoner] = None,      # <--- NEW
        enable_reasoning: bool = True,                 # <--- NEW
        max_reason_chars: int = 1000,                  # <--- NEW
        num_step_range: Tuple[int, int] = None,     # <--- NEW
        image_quality: Optional[int] = None,
    ):
        """
        Initialize the GroundtruthInteractor.
        
        Args:
            env: The gymnasium environment
            max_steps: Maximum number of steps per episode
            render_fn: Custom render function
            abs_size: Absolute size for image rendering
            master_seed: Master seed for deterministic episode generation
            groundtruth_strategy: Strategy parameter for solve (if needed)
        """
        # Create a mock VLM for compatibility with parent class
        mock_vlm = MockVLM()
        
        # Initialize parent class with mock VLM
        super().__init__(
            vlm=mock_vlm,
            env=env,
            max_steps=max_steps,
            num_step_range=num_step_range,
            render_fn=render_fn,
            abs_size=abs_size,
            master_seed=master_seed,
            max_long_side=max_long_side,
            image_quality=image_quality,
        )
        
        self.groundtruth_strategy = groundtruth_strategy
        self.groundtruth_kwargs = groundtruth_kwargs or {}
        self.reasoner = reasoner
        self.enable_reasoning = enable_reasoning
        self.max_reason_chars = max_reason_chars
        self.num_step_range = num_step_range
        # Update init_args to reflect groundtruth mode
        self.init_args.update({
            "interactor_type": "GroundtruthInteractor",
            "groundtruth_strategy": groundtruth_strategy,
            "groundtruth_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in (groundtruth_kwargs or {}).items()},
            "enable_reasoning": enable_reasoning,
            "reasoner_class": reasoner.__class__.__name__ if reasoner else None,
        })
        
    def _get_groundtruth_action(self) -> list:
        """
        Get the groundtruth action from the environment.
        
        Returns:
            The groundtruth action as a string
        """
        # range both inclusive
        num_steps = None
        if self.num_step_range is not None:
            num_steps = np.random.randint(self.num_step_range[0], self.num_step_range[1] + 1)
        try:
            if self.groundtruth_strategy is not None:
                # Some environments accept a strategy parameter
                if num_steps is not None:
                    actions = self.env.solve(strategy=self.groundtruth_strategy, num_steps=num_steps, **(self.groundtruth_kwargs or {}))
                else:
                    actions = self.env.solve(strategy=self.groundtruth_strategy, **(self.groundtruth_kwargs or {}))
            else:
                # Most environments don't need a strategy parameter
                if num_steps is not None:
                    actions = self.env.solve(num_steps=num_steps, **(self.groundtruth_kwargs or {}))
                else:
                    actions = self.env.solve(**(self.groundtruth_kwargs or {}))
            # Ensure there is a final stop to terminate the episode for envs
            # that don't auto-terminate on goal (e.g., sliding_block)
            if not actions or not isinstance(actions, list):
                actions = ["('stop', 'stop')"]
            else:
                # If the last action is not a stop, append one
                last = str(actions[-1]) if actions else ""
                if "('stop'" not in last:
                    actions = list(actions) + ["('stop', 'stop')"]

            return actions
                
        except Exception as e:
            # Fallback to stop action if solve fails
            print(f"Warning: solve failed: {e}")
            return ["('stop', 'stop')"]

    def _generate_reasoning(
        self,
        instructions: str,
        prev_action: str | None,
        next_action: str,
        step_idx: int,
        prev_image_b64: Optional[str],
        next_image_b64: Optional[str],
    ) -> str:
        if not self.enable_reasoning or self.reasoner is None:
            return "<think></think>"
        try:
            text = self.reasoner.generate(
                instructions=instructions,
                prev_action=prev_action,
                next_action=next_action,
                step_idx=step_idx,
                prev_image_b64=prev_image_b64,
                next_image_b64=next_image_b64,
            )
            return text[: self.max_reason_chars].strip()
        except Exception as e:
            print(f"Warning: reasoner failed: {e}")
            return "<think></think>"
            
    def _compose_response(self, reasoning: str, action: str) -> str:
        return reasoning + f"<answer>{action}</answer>"

    def run_episode(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
        save_verbose: bool = False,
        elicit_verbalization: bool = False,
        repeat_instructions: bool = False,
        include_env_feedback: bool = False,
        num_interaction_history: int = 0,
        widget_log: bool = False,
        autosave_path: str | None = None,
        save_history_at_episode_end: bool = False,
        extra_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run an episode using groundtruth actions instead of VLM calls.
        This method overrides the parent's run_episode to use solve.
        """
        # Determine the seed for this episode
        if seed is not None:
            episode_seed = seed
        elif self.master_rng is not None:
            # Generate a new deterministic seed for this episode
            episode_seed = self.master_rng.randint(0, 2**32 - 1)
        else:
            episode_seed = None  # Let the env handle it (usually random)

        # print(f"Running episode with seed {episode_seed}")

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
        # Compute ground-truth actions for the freshly reset episode state
        self.groundtruth_actions = self._get_groundtruth_action()

        stats: Dict[str, Any] = dict(
            step=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            final_proportion_matched=0,
            seed=episode_seed,
            extra_state=(getattr(self.env.unwrapped, extra_state, None) if extra_state else None),
        )
        widgets: List[w.Accordion] = [] if widget_log else []
        prev_action: Optional[str] = None
        while True:
            if stats["step"] >= self.max_steps:
                stats["truncated"] = True
                break

            frame = self.env.render()
            img_b64 = self._image_to_b64(frame)
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
            # Generate groundtruth response instead of querying VLM
            # ------------------------------------------------------------------
            prev_frame = self.env.render()
            prev_b64 = self._image_to_b64(prev_frame)
            
            # Pick next action; if we ran out, issue a stop to terminate
            if stats["step"] < len(self.groundtruth_actions):
                next_action = self.groundtruth_actions[stats["step"]]
            else:
                next_action = "('stop', 'stop')"
            
            obs, reward, terminated, truncated, info = self.env.step(next_action)
            next_frame = self.env.render()
            next_b64 = self._image_to_b64(next_frame)
            

            reasoning_text = self._generate_reasoning(
                instructions=instructions,
                prev_action=prev_action,
                next_action=next_action,
                step_idx=stats["step"],
                prev_image_b64=prev_b64,   # <-- pass images through
                next_image_b64=next_b64,
            )

            # Compose a VLM-like response with <think> + <answer>
            vlm_out = self._compose_response(reasoning_text, next_action)

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

            # Environment step


            # Log step to history
            step_data = dict(
                step=stats["step"],
                prompt=prompt,
                vlm_output=vlm_out,
                action=next_action,
                reward=reward,
                image=img_b64,              # keep your original field; or store both:
                image_prev=prev_b64,         # optional new fields
                image_next=next_b64,
                info=self._safe(info),
                think=reasoning_text,
            )

            step_data["think"] = reasoning_text
            self.hist.append(step_data)

            # Stats
            stats["step"] += 1
            stats["reward"] += reward
            stats["terminated"] = terminated
            stats["truncated"] = truncated

            # fill Env Feedback panel if requested and available
            if include_env_feedback and "env_feedback" in info and (verbose or widget_log):
                fb_html.value = (
                    f"<pre style='white-space:pre-wrap;"
                    f"font-size:0.82em;line-height:1.25;margin:0;'>{info['env_feedback']}</pre>"
                )
            
            prev_action = action

            if terminated or truncated:
                break

            if render and not widget_log:  # avoid double-show
                self.render_fn(self.env, stats)

        # Save trajectory if requested
        if autosave_path:
            self.save_history(autosave_path)
        elif save_history_at_episode_end:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_history(f"episodes/episode_{ts}.json")

        if widget_log:
            stats["widgets"] = widgets
        return stats 
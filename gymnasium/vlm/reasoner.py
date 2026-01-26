# reasoner.py
from typing import Optional, Dict, Any
import time
import base64
try:
    # OpenAI SDK v1+
    from openai import OpenAI
    from openai import APIError, RateLimitError, AuthenticationError
except Exception:
    OpenAI = None
    APIError = RateLimitError = AuthenticationError = Exception  # basic fallback

class BaseReasoner:
    """Interface for generating a short reasoning span (the <think> block)."""
    def generate(
        self,
        instructions: str,
        prev_action: Optional[str],
        next_action: str,
        step_idx: int,
    ) -> str:
        raise NotImplementedError


def _to_data_url(b64: str, mime: str = "image/jpeg") -> str:
    """
    Accepts raw base64 (no header) or a pre-formed data URL.
    Returns a data URL usable in Chat Completions image messages.
    """
    if not b64:
        return ""
    if b64.startswith("data:"):
        return b64
    return f"data:{mime};base64,{b64}"


class GPTReasoner(BaseReasoner):
    """
    Minimal LLM wrapper with real GPT call.
    - Sends prev/next images alongside text.
    - Prefers the Responses API; falls back to Chat Completions.
    - temperature=0 for deterministic-ish bootstrap.
    """

    def __init__(self, model: str = "gpt-4o-mini", client=None, max_tokens: int = 256, temperature: float = 0.0,
                 request_timeout: Optional[float] = 30.0, max_retries: int = 3, backoff_base: float = 1.5):
        self.model = model
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _ensure_client(self):
        if self.client is None:
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai>=1.0.0` and set OPENAI_API_KEY.")
            self.client = OpenAI()  # uses env var OPENAI_API_KEY

    def _build_prompt(
        self,
        instructions: str,
        prev_action: Optional[str],
        next_action: str,
        step_idx: int,
    ) -> str:
        prev_str = "NONE" if prev_action is None else str(prev_action)
        return (
            "You are generating a concise internal chain-of-thought explaining why the next action is appropriate. "
            "Do NOT restate the action; explain your reasoning (2–3 sentences).\n\n"
            f"Task instructions:\n{instructions}\n\n"
            f"Step index: {step_idx}\n"
            f"Previous action: {prev_str}\n"
            f"Next action (ground-truth): {next_action}\n\n"
            "You also receive two images: the previous observation and the next observation after taking the action. "
            "Your reasoning should be based on the images. try to analysis the images and the instructions to generate the reasoning."
            "Produce only the reasoning text (no XML tags)."
        )

    # ---------- RESPONSES API (multimodal) ----------
    def _call_responses_api(self, prompt: str, prev_data_url: str, next_data_url: str) -> str:
        content = [{"type": "text", "text": prompt}]
        if prev_data_url:
            # Either image_url form or inline base64 depending on SDK support.
            content.append({"type": "input_image", "image_url": {"url": prev_data_url}})
        if next_data_url:
            content.append({"type": "input_image", "image_url": {"url": next_data_url}})

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        try:
            return resp.output_text.strip()
        except Exception:
            # Manual extraction fallback
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ("output_text", "text"):
                            text = getattr(c, "text", "") or getattr(c, "value", "")
                            if text:
                                return text.strip()
            raise RuntimeError("Unexpected Responses API payload structure")

    # ---------- CHAT COMPLETIONS (multimodal) ----------
    def _call_chat_api(self, prompt: str, prev_data_url: str, next_data_url: str) -> str:
        user_content = [{"type": "text", "text": prompt}]
        if prev_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": prev_data_url, "detail": "auto"}})
        if next_data_url:
            user_content.append({"type": "image_url", "image_url": {"url": next_data_url, "detail": "auto"}})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise internal rationales that justify the next action. "
                        "Avoid restating the action; explain the why in 2–3 sentences."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        return resp.choices[0].message.content.strip()

    def _generate_with_retries(self, prompt: str, prev_url: str, next_url: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                try:
                    return self._call_responses_api(prompt, prev_url, next_url)
                except Exception:
                    return self._call_chat_api(prompt, prev_url, next_url)
            except (RateLimitError, APIError, AuthenticationError, TimeoutError) as e:
                last_err = e
                time.sleep((self.backoff_base ** attempt) + (0.01 * attempt))
            except Exception as e:
                last_err = e
                break
        raise last_err if last_err else RuntimeError("Unknown generation error")

    def generate(
        self,
        instructions: str,
        prev_action: Optional[str],
        next_action: str,
        step_idx: int,
        prev_image_b64: Optional[str] = None,   # <--- NEW
        next_image_b64: Optional[str] = None,   # <--- NEW
        image_mime: str = "image/jpeg",
    ) -> str:
        """
        prev_image_b64 / next_image_b64: base64 *without* header ('iVBORw0K...').
        If you already have data URLs, pass them—this function will accept both.
        """
        self._ensure_client()
        prompt = self._build_prompt(instructions, prev_action, next_action, step_idx)

        prev_data_url = _to_data_url(prev_image_b64 or "", mime=image_mime)
        next_data_url = _to_data_url(next_image_b64 or "", mime=image_mime)

        text = self._generate_with_retries(prompt, prev_data_url, next_data_url)

        # Robustness
        text = (text or "").strip()
        if not text:
            text = "This action advances toward the goal from the current state."
        if len(text) > 1000:
            text = text[:1000].rstrip()
        return text

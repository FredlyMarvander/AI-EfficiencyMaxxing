"""llama.cpp GGUF inference wrapper for zero-token local tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.router import TaskKind
from agent.tokens import estimate_tokens


class LocalModelError(RuntimeError):
    """Raised when local GGUF inference is unavailable or fails."""


SYSTEM_PROMPTS: dict[TaskKind, str] = {
    TaskKind.FACTUAL: (
        "Answer the factual question directly. If it asks for an explanation, "
        "explain concisely but completely, covering every part of the "
        "question. Otherwise return just the answer plus essential context."
    ),
    TaskKind.SENTIMENT: (
        "Classify sentiment. Return only one label, using the label set the "
        "prompt requests (e.g. favorable/unfavorable); otherwise Positive, "
        "Negative, or Neutral. Use Neutral for factual statements that "
        "express no opinion or emotion."
    ),
    TaskKind.SUMMARY: (
        "Summarize faithfully and concisely. Preserve the main claims and avoid "
        "new information."
    ),
    # No concrete example entities here: an earlier prompt included one and
    # the model copied it into answers for unrelated sentences.
    TaskKind.NER: (
        "List every named entity that appears in the given text: all people, "
        "organizations, locations, dates, and other proper nouns. Follow the "
        "requested format if one is given. Never output an entity that is not "
        "in the text."
    ),
    TaskKind.CODE: (
        "Write the requested code exactly as specified, using the required "
        "names. Output one fenced code block and nothing else unless an "
        "explanation is requested."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "Answer accurately and concisely. Return only the requested answer."
)


@dataclass
class LocalGGUFModel:
    settings: Settings
    _llm: Any = field(default=None, init=False, repr=False)

    def complete(self, prompt: str, kind: TaskKind, max_tokens: int) -> str:
        # Fail fast (and fall back to Fireworks) instead of letting llama.cpp
        # error out after a slow prefill; 128 covers template + system prompt.
        if estimate_tokens(prompt) + max_tokens + 128 > self.settings.local_n_ctx:
            raise LocalModelError("prompt likely exceeds the local context window")

        llm = self._load()
        system_prompt = SYSTEM_PROMPTS.get(kind, DEFAULT_SYSTEM_PROMPT)

        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=0.9,
                max_tokens=max_tokens,
                repeat_penalty=1.05,
            )
        except Exception as exc:
            raise LocalModelError(f"local inference failed: {exc}") from exc

        answer = _extract_content(response)
        if not answer:
            raise LocalModelError("local model returned an empty answer")
        return answer

    def _load(self):
        if self._llm is not None:
            return self._llm

        model_path = Path(self.settings.local_model_path)
        if not model_path.is_file():
            raise LocalModelError(f"local GGUF model not found at {model_path}")

        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise LocalModelError("llama-cpp-python is not installed") from exc

        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.settings.local_n_ctx,
                n_threads=self.settings.local_n_threads,
                n_batch=self.settings.local_n_batch,
                verbose=False,
            )
        except Exception as exc:
            raise LocalModelError(f"could not load local GGUF model: {exc}") from exc

        return self._llm

    def close(self) -> None:
        if self._llm is None:
            return

        llm = self._llm
        self._llm = None
        close = getattr(llm, "close", None)
        if callable(close):
            close()


def call_local_model(
    prompt: str,
    settings: Settings,
    kind: TaskKind = TaskKind.FACTUAL,
    max_tokens: int = 128,
) -> dict[str, str]:
    """Compatibility wrapper returning the same shape as older scaffolds."""

    model = LocalGGUFModel(settings)
    try:
        answer = model.complete(prompt, kind, max_tokens)
        return {"answer": answer}
    finally:
        model.close()


def _extract_content(response: Any) -> str:
    if not isinstance(response, dict):
        return str(response or "").strip()

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return str(first or "").strip()

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = first.get("text", "")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return str(content or "").strip()

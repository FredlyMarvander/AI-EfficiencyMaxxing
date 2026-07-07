"""Fireworks AI client using the OpenAI-compatible chat completions API."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import requests

from agent.config import Settings


SYSTEM_PROMPT = (
    "You are a highly efficient assistant. Think carefully, then provide the "
    "accurate final answer in English with no conversational filler."
)


class FireworksAPIError(RuntimeError):
    """Raised for transport or response errors from Fireworks AI."""


@dataclass
class FireworksClient:
    settings: Settings

    def __post_init__(self) -> None:
        self.chat_url = _chat_completions_url(self.settings.fireworks_base_url)
        self.last_usage: dict[str, int | str] = _empty_usage()
        self.total_usage: dict[str, int] = _empty_total_usage()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.settings.fireworks_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        selected_model = model or self.settings.allowed_models[0]
        if selected_model not in self.settings.allowed_models:
            raise FireworksAPIError(
                f"model {selected_model!r} is not present in ALLOWED_MODELS"
            )

        token_limit = max_tokens or self.settings.default_max_tokens
        timeout = timeout_seconds or self.settings.request_timeout_seconds
        self.last_usage = _empty_usage()
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": token_limit,
        }

        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                timeout=min(timeout, 30.0),
            )
        except requests.RequestException as exc:
            raise FireworksAPIError(f"request failed: {exc}") from exc

        if response.status_code >= 400:
            raise FireworksAPIError(_format_http_error(response))

        try:
            data = response.json()
        except ValueError as exc:
            raise FireworksAPIError("response was not valid JSON") from exc

        self.last_usage = _extract_usage(data, selected_model)
        self.total_usage["prompt_tokens"] += int(self.last_usage["prompt_tokens"])
        self.total_usage["completion_tokens"] += int(
            self.last_usage["completion_tokens"]
        )
        self.total_usage["total_tokens"] += int(self.last_usage["total_tokens"])
        return _clean_answer(_extract_content(data))


def call_fireworks_api(
    prompt: str,
    settings: Settings,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """Call Fireworks with the first allowed model and return a concise answer."""

    client = FireworksClient(settings)
    return client.complete(
        prompt=prompt,
        model=settings.allowed_models[0],
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _chat_completions_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _empty_usage() -> dict[str, int | str]:
    return {
        "model": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _empty_total_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _extract_usage(data: dict[str, Any], model: str) -> dict[str, int | str]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise FireworksAPIError("response did not include choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise FireworksAPIError("choice had an unexpected shape")

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

    answer = str(content or "").strip()
    if not answer:
        raise FireworksAPIError("model returned an empty answer")
    return answer


def _clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(sure|certainly|of course)[,!.:\s]+", "", answer, flags=re.I)
    answer = re.sub(
        r"^here (is|are) (the|a|an)?\s*(concise\s*)?(answer|response)[:\s]+",
        "",
        answer,
        flags=re.I,
    )
    return answer.strip()


def _format_http_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text[:300]
    return f"HTTP {response.status_code}: {body}"

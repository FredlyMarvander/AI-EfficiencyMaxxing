"""Fireworks AI client using the OpenAI-compatible chat completions API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import re
import threading
import time
from typing import Any

import requests

from agent.config import Settings
from agent.prompts import system_prompt_for
from agent.router import TaskKind


MAX_RETRY_TOKENS = 2048

# Transient statuses worth retrying; anything else 4xx/5xx fails immediately.
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
# Bounded so one rate-limited task cannot monopolize the container's shared
# runtime budget: worst case is a handful of capped backoffs, not minutes.
MAX_TRANSPORT_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.5
BACKOFF_CAP_SECONDS = 20.0

# Process-wide request pacing shared by every client instance, so multiple
# clients (threads) cannot jointly exceed the account's per-minute quota.
#
# `_adaptive_interval` starts at 0 and only grows when the API actually
# returns 429 (using its Retry-After header when present, else multiplicative
# backoff), then decays after sustained successes. A generously rate-limited
# key (e.g. the harness's) never trips this and pays zero extra latency; a
# tightly quota'd key converges to a safe spacing without anyone having to
# guess the account's exact limit up front -- a fixed guess (e.g. a manual
# FIREWORKS_MIN_INTERVAL) can still be wrong, and restarting the process
# forgets it, so every fresh run re-collides with the server-side quota.
_pace_lock = threading.Lock()
_next_request_at = 0.0
_adaptive_interval = 0.0
_consecutive_successes = 0

ADAPTIVE_INTERVAL_CAP_SECONDS = 20.0
ADAPTIVE_GROWTH_FACTOR = 1.6
ADAPTIVE_GROWTH_FLOOR_SECONDS = 2.0
ADAPTIVE_DECAY_FACTOR = 0.7
ADAPTIVE_DECAY_AFTER_SUCCESSES = 5


def _reserve_request_slot(min_interval: float) -> None:
    global _next_request_at
    with _pace_lock:
        interval = max(min_interval, _adaptive_interval)
        if interval <= 0:
            return
        now = time.monotonic()
        start_at = max(now, _next_request_at)
        _next_request_at = start_at + interval
    wait = start_at - now
    if wait > 0:
        time.sleep(wait)


def _note_rate_limited(retry_after: float | None) -> None:
    global _adaptive_interval, _consecutive_successes
    with _pace_lock:
        _consecutive_successes = 0
        grown = max(ADAPTIVE_GROWTH_FLOOR_SECONDS, _adaptive_interval * ADAPTIVE_GROWTH_FACTOR)
        floor = retry_after if retry_after is not None else grown
        _adaptive_interval = min(ADAPTIVE_INTERVAL_CAP_SECONDS, max(_adaptive_interval, floor))


def _note_request_succeeded() -> None:
    global _adaptive_interval, _consecutive_successes
    if _adaptive_interval <= 0:
        return
    with _pace_lock:
        _consecutive_successes += 1
        if _consecutive_successes >= ADAPTIVE_DECAY_AFTER_SUCCESSES:
            _adaptive_interval = max(0.0, _adaptive_interval * ADAPTIVE_DECAY_FACTOR)
            _consecutive_successes = 0


class FireworksAPIError(RuntimeError):
    """Raised for transport or response errors from Fireworks AI."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelUnavailableError(FireworksAPIError):
    """A model that definitively does not exist or is not deployed.

    Unlike transient failures (429/5xx/timeouts), this cannot heal within one
    run, so the model is cached as dead and skipped for all later tasks.
    """

    def __init__(self, message: str, model: str, status: int | None = None) -> None:
        super().__init__(message, status=status)
        self.model = model


def _is_model_unavailable_error(status: int | None, message: str) -> bool:
    if status != 404:
        return False
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ("model", "not found", "not_found", "not deployed", "no such")
    )


@dataclass
class FireworksClient:
    settings: Settings

    def __post_init__(self) -> None:
        self.chat_url = _chat_completions_url(self.settings.fireworks_base_url)
        self.last_usage: dict[str, int | str] = _empty_usage()
        self.total_usage: dict[str, int] = _empty_total_usage()
        self._usage_lock = threading.Lock()
        self._unavailable_models: set[str] = set()
        self._unavailable_lock = threading.Lock()
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
        kind: TaskKind | None = None,
    ) -> str:
        answer, _ = self.complete_with_usage(
            prompt,
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            kind=kind,
        )
        return answer

    def complete_with_usage(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        kind: TaskKind | None = None,
        _retry_on_truncation: bool = True,
    ) -> tuple[str, dict[str, int | str]]:
        selected_model = model or self.settings.allowed_models[0]
        if selected_model not in self.settings.allowed_models:
            raise FireworksAPIError(
                f"model {selected_model!r} is not present in ALLOWED_MODELS"
            )
        if self.is_model_unavailable(selected_model):
            raise ModelUnavailableError(
                f"model {selected_model!r} was already reported unavailable",
                model=selected_model,
            )

        token_limit = max_tokens or self.settings.default_max_tokens
        timeout = timeout_seconds or self.settings.request_timeout_seconds
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt_for(kind)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": token_limit,
        }

        try:
            data = self._post_with_retries(payload, min(timeout, 30.0))
        except ModelUnavailableError:
            raise
        except FireworksAPIError as exc:
            if _is_model_unavailable_error(exc.status, str(exc)):
                with self._unavailable_lock:
                    self._unavailable_models.add(selected_model)
                logging.warning(
                    "model %s marked unavailable for the rest of this run",
                    selected_model,
                )
                raise ModelUnavailableError(
                    str(exc), model=selected_model, status=exc.status
                ) from exc
            raise

        usage = _extract_usage(data, selected_model)
        content, finish_reason = _extract_content(data)
        usage["finish_reason"] = finish_reason
        with self._usage_lock:
            self.last_usage = usage
            self.total_usage["prompt_tokens"] += int(usage["prompt_tokens"])
            self.total_usage["completion_tokens"] += int(usage["completion_tokens"])
            self.total_usage["total_tokens"] += int(usage["total_tokens"])

        answer = _clean_answer(_strip_reasoning(content))

        # A "length" stop means the answer was cut off mid-generation (for
        # reasoning models possibly inside a <think> block, leaving nothing
        # usable). One retry with a bigger budget costs extra tokens but a
        # truncated answer risks failing the accuracy gate entirely.
        if finish_reason == "length" and _retry_on_truncation:
            retry_limit = min(MAX_RETRY_TOKENS, token_limit * 3)
            if retry_limit > token_limit:
                logging.info(
                    "answer truncated at %s tokens; retrying with %s",
                    token_limit,
                    retry_limit,
                )
                try:
                    retry_answer, retry_usage = self.complete_with_usage(
                        prompt,
                        model=selected_model,
                        max_tokens=retry_limit,
                        timeout_seconds=timeout_seconds,
                        kind=kind,
                        _retry_on_truncation=False,
                    )
                    # The retry's own usage doesn't include the discarded
                    # first attempt; a per-task caller reading just the
                    # returned usage would otherwise silently undercount the
                    # true cost of this task by the truncated attempt's cost.
                    combined_usage: dict[str, int | str] = {
                        "model": retry_usage.get("model", selected_model),
                        "prompt_tokens": int(usage["prompt_tokens"])
                        + int(retry_usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage["completion_tokens"])
                        + int(retry_usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage["total_tokens"])
                        + int(retry_usage.get("total_tokens", 0)),
                        "finish_reason": retry_usage.get("finish_reason", ""),
                        "retried": True,
                    }
                    return retry_answer, combined_usage
                except FireworksAPIError as exc:
                    if not answer:
                        raise
                    logging.warning(
                        "truncation retry failed (%s); keeping truncated answer", exc
                    )

        if not answer:
            raise FireworksAPIError("model returned an empty answer")
        return answer, usage

    def _post_with_retries(self, payload: dict[str, Any], timeout: float) -> dict:
        """POST the payload, absorbing rate limits and transient failures.

        A single 429 must not surface as a wrong final answer: the judge
        grades whatever lands in results.json, so waiting a few seconds is
        always cheaper than failing the task.
        """

        last_error = "request failed"
        last_status: int | None = None
        for attempt in range(MAX_TRANSPORT_ATTEMPTS):
            _reserve_request_slot(self.settings.min_request_interval)
            try:
                response = self.session.post(
                    self.chat_url, json=payload, timeout=timeout
                )
            except requests.RequestException as exc:
                last_error = f"request failed: {exc}"
                response = None

            if response is not None:
                if response.status_code < 400:
                    _note_request_succeeded()
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise FireworksAPIError(
                            "response was not valid JSON"
                        ) from exc
                last_error = _format_http_error(response)
                last_status = response.status_code
                if response.status_code == 429:
                    _note_rate_limited(_retry_after_seconds(response))
                if response.status_code not in RETRYABLE_STATUS:
                    raise FireworksAPIError(last_error, status=response.status_code)

            if attempt == MAX_TRANSPORT_ATTEMPTS - 1:
                break
            delay = min(
                BACKOFF_CAP_SECONDS,
                BACKOFF_BASE_SECONDS * (2**attempt) + random.uniform(0.0, 1.0),
            )
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                delay = max(delay, min(retry_after, BACKOFF_CAP_SECONDS))
            logging.warning(
                "transient Fireworks error (attempt %s/%s), retrying in %.1fs: %s",
                attempt + 1,
                MAX_TRANSPORT_ATTEMPTS,
                delay,
                last_error[:200],
            )
            time.sleep(delay)

        raise FireworksAPIError(last_error, status=last_status)

    def is_model_unavailable(self, model: str) -> bool:
        with self._unavailable_lock:
            return model in self._unavailable_models


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
        "finish_reason": "",
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


def _extract_content(data: dict[str, Any]) -> tuple[str, str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise FireworksAPIError("response did not include choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise FireworksAPIError("choice had an unexpected shape")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        # Reasoning models put chain-of-thought in reasoning_content and the
        # final answer in content. If generation was cut off mid-reasoning,
        # content is empty; the reasoning tail is a better fallback than an
        # empty answer, which would fail the task outright.
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]
    else:
        content = first.get("text", "")

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    finish_reason = str(first.get("finish_reason") or "")
    return str(content or "").strip(), finish_reason


def _strip_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    # An unterminated <think> means generation was cut off mid-reasoning;
    # everything after the tag is scratch work, not an answer.
    cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.S | re.I)
    return cleaned.strip()


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


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _format_http_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text[:300]
    return f"HTTP {response.status_code}: {body}"

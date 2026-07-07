"""Batch entry point for the Track 1 General-Purpose AI Agent harness."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

from agent.config import DEFAULT_OUTPUT_PATH, ConfigError, Settings, load_settings
from agent.fireworks import FireworksAPIError, FireworksClient
from agent.io import read_tasks, write_results
from agent.local_model import LocalGGUFModel, LocalModelError
from agent.router import (
    RouteTarget,
    SemanticRouter,
    ordered_models,
    output_budget,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)


def process_tasks(settings: Settings) -> list[dict[str, str]]:
    tasks = read_tasks(settings.input_path)
    router = SemanticRouter(settings)
    client = FireworksClient(settings)
    local_model = LocalGGUFModel(settings) if settings.enable_local_model else None
    results: list[dict[str, str]] = []
    deadline = time.monotonic() + settings.max_runtime_seconds

    for index, task in enumerate(tasks, start=1):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 1:
            logging.error("maximum runtime reached before task %s", task["task_id"])
            results.extend(
                {
                    "task_id": pending_task["task_id"],
                    "answer": "Unable to produce an answer within the time limit.",
                }
                for pending_task in tasks[index - 1 :]
            )
            break

        logging.info("processing task %s (%s/%s)", task["task_id"], index, len(tasks))
        results.append(
            answer_task(
                task,
                client,
                settings,
                router=router,
                local_model=local_model,
                remaining_seconds=remaining_seconds,
            )
        )

    logging.info(
        "tokens total prompt=%s completion=%s total=%s",
        client.total_usage["prompt_tokens"],
        client.total_usage["completion_tokens"],
        client.total_usage["total_tokens"],
    )
    if local_model is not None:
        local_model.close()
    return results


def answer_task(
    task: dict[str, str],
    client: FireworksClient,
    settings: Settings,
    router: SemanticRouter | None = None,
    local_model: LocalGGUFModel | None = None,
    remaining_seconds: float | None = None,
) -> dict[str, str]:
    task_id = task["task_id"]
    prompt = task["prompt"].strip()
    if not prompt:
        return {"task_id": task_id, "answer": ""}

    router = router or SemanticRouter(settings)
    decision = router.route(prompt)
    max_tokens = output_budget(decision.kind, prompt, settings.default_max_tokens)
    timeout_seconds = min(
        settings.request_timeout_seconds,
        max(1.0, remaining_seconds or settings.request_timeout_seconds),
    )

    logging.info(
        "route task=%s kind=%s target=%s confidence=%.3f margin=%.3f reason=%s",
        task_id,
        decision.kind.value,
        decision.target.value,
        decision.confidence,
        decision.margin,
        decision.reason,
    )

    if decision.target is RouteTarget.LOCAL and settings.enable_local_model:
        model = local_model or LocalGGUFModel(settings)
        try:
            answer = model.complete(prompt, decision.kind, max_tokens)
            return {"task_id": task_id, "answer": answer}
        except LocalModelError as exc:
            logging.warning(
                "local model failed for task %s; falling back to Fireworks: %s",
                task_id,
                exc,
            )

    last_error: Exception | None = None
    for model in ordered_models(
        settings.allowed_models,
        decision.kind,
        preferred_model=settings.preferred_fireworks_model,
    ):
        try:
            answer = client.complete(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            _log_token_usage(task_id, client.last_usage)
            return {"task_id": task_id, "answer": answer}
        except FireworksAPIError as exc:
            last_error = exc
            logging.warning("model %s failed for task %s: %s", model, task_id, exc)
        except Exception as exc:
            last_error = exc
            logging.exception(
                "unexpected failure from model %s for task %s", model, task_id
            )

    logging.error("task %s failed: %s", task_id, last_error)
    return {
        "task_id": task_id,
        "answer": "Unable to produce an answer due to an API error.",
    }


def run() -> int:
    settings = load_settings()
    results = process_tasks(settings)
    write_results(settings.output_path, _valid_result_shape(results))
    logging.info("wrote %s result(s) to %s", len(results), settings.output_path)
    return 0


def _log_token_usage(task_id: str, usage: dict[str, int | str]) -> None:
    if int(usage.get("total_tokens", 0)) <= 0:
        logging.info("tokens task=%s unavailable", task_id)
        return

    logging.info(
        "tokens task=%s model=%s prompt=%s completion=%s total=%s",
        task_id,
        usage.get("model", ""),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
    )


def main() -> int:
    try:
        return run()
    except (ConfigError, OSError, ValueError) as exc:
        logging.error("fatal startup error: %s", exc)
    except Exception:
        logging.exception("unexpected fatal error")

    _write_empty_results_best_effort()
    return 1


def _valid_result_shape(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"task_id": str(item.get("task_id", "")), "answer": str(item.get("answer", ""))}
        for item in results
    ]


def _write_empty_results_best_effort() -> None:
    output_path = os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
    try:
        write_results(output_path, [])
    except Exception:
        logging.exception("could not write fallback results file")


if __name__ == "__main__":
    sys.exit(main())

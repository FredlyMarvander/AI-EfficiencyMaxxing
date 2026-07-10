"""Batch entry point for the Track 1 General-Purpose AI Agent harness."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import threading
import time
from typing import Any

# Captured before any heavy model imports/loads so the runtime budget counts
# startup cost against the harness's 10-minute container limit.
_PROCESS_START = time.monotonic()

from agent.config import DEFAULT_OUTPUT_PATH, ConfigError, Settings, load_settings
from agent.fireworks import FireworksAPIError, FireworksClient
from agent.io import read_tasks, write_results
from agent.local_model import LocalGGUFModel, LocalModelError
from agent.router import (
    RouteDecision,
    RouteTarget,
    SemanticRouter,
    ordered_models,
    output_budget,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

TIME_LIMIT_ANSWER = "Unable to produce an answer within the time limit."
API_ERROR_ANSWER = "Unable to produce an answer due to an API error."


def process_tasks(settings: Settings) -> list[dict[str, str]]:
    tasks = read_tasks(settings.input_path)
    router = SemanticRouter(settings)
    client = FireworksClient(settings)
    local_model = LocalGGUFModel(settings) if settings.enable_local_model else None
    deadline = _PROCESS_START + settings.max_runtime_seconds

    # Placeholders keep /output/results.json valid at every moment, so even a
    # hard kill mid-run leaves a complete, well-formed submission behind.
    results = [
        {"task_id": task["task_id"], "answer": TIME_LIMIT_ANSWER} for task in tasks
    ]
    results_lock = threading.Lock()
    write_lock = threading.Lock()

    def flush() -> None:
        with write_lock:
            with results_lock:
                snapshot = _valid_result_shape(results)
            write_results(settings.output_path, snapshot)

    flush()

    def run_one(index: int, task: dict[str, str], decision: RouteDecision | None) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 2:
            logging.error("skipping task %s: time limit reached", task["task_id"])
            return
        logging.info("processing task %s (%s/%s)", task["task_id"], index + 1, len(tasks))
        try:
            result = answer_task(
                task,
                client,
                settings,
                router=router,
                local_model=local_model,
                remaining_seconds=remaining,
                decision=decision,
            )
        except Exception:
            logging.exception("unexpected failure while answering task %s", task["task_id"])
            result = {"task_id": task["task_id"], "answer": API_ERROR_ANSWER}
        with results_lock:
            results[index] = result
        flush()

    decisions: list[RouteDecision | None] = [
        router.route(task["prompt"].strip()) if task["prompt"].strip() else None
        for task in tasks
    ]

    # Fireworks calls are network-bound and run concurrently; llama.cpp is not
    # thread-safe, so local tasks get a single serial lane alongside them.
    fireworks_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.fireworks_concurrency, thread_name_prefix="fireworks"
    )
    local_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="local"
    )

    futures = []
    for index, (task, decision) in enumerate(zip(tasks, decisions)):
        use_local_lane = (
            local_model is not None
            and decision is not None
            and decision.target is RouteTarget.LOCAL
        )
        pool = local_pool if use_local_lane else fireworks_pool
        futures.append(pool.submit(run_one, index, task, decision))

    _, pending = concurrent.futures.wait(
        futures, timeout=max(1.0, deadline - time.monotonic())
    )
    fireworks_pool.shutdown(wait=False, cancel_futures=True)
    local_pool.shutdown(wait=False, cancel_futures=True)
    flush()

    logging.info(
        "tokens total prompt=%s completion=%s total=%s",
        client.total_usage["prompt_tokens"],
        client.total_usage["completion_tokens"],
        client.total_usage["total_tokens"],
    )

    if pending:
        # In-flight inference cannot be cancelled and non-daemon workers would
        # block interpreter shutdown past the container limit; results are
        # already on disk, so exit immediately with success.
        logging.error(
            "%s task(s) still running at the time limit; exiting with partial results",
            len(pending),
        )
        logging.shutdown()
        os._exit(0)

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
    decision: RouteDecision | None = None,
) -> dict[str, str]:
    task_id = task["task_id"]
    prompt = task["prompt"].strip()
    if not prompt:
        return {"task_id": task_id, "answer": ""}

    if decision is None:
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

    if decision.target is RouteTarget.DETERMINISTIC and decision.answer is not None:
        # Locally computed exact answer: zero Fireworks tokens, no network.
        return {"task_id": task_id, "answer": decision.answer}

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
            answer, usage = client.complete_with_usage(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                kind=decision.kind,
            )
            _log_token_usage(task_id, usage)
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
    return {"task_id": task_id, "answer": API_ERROR_ANSWER}


def run() -> int:
    settings = load_settings()
    results = process_tasks(settings)
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

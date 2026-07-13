"""Batch entry point for the Track 1 General-Purpose AI Agent harness."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
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
    TaskKind,
    local_output_budget,
    ordered_models,
    output_budget,
)
from agent.validate import repair_local_answer, validate_local_answer


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

TIME_LIMIT_ANSWER = "Unable to produce an answer within the time limit."
API_ERROR_ANSWER = "Unable to produce an answer due to an API error."


# Retried at a nonzero temperature (the first attempt is temperature=0.0,
# fully deterministic) so a validation failure gets a genuinely different
# second sample instead of regenerating the identical rejected answer.
# Costs no Fireworks tokens -- only local CPU time -- so it's worth trying
# before paying for an escalation.
_LOCAL_RETRY_TEMPERATURE = 0.4

_STRICT_FORMAT_PATTERN = re.compile(r"\bexactly\b|\bprecisely\b")


def _retry_unlikely_to_help(kind: TaskKind, prompt: str) -> bool:
    # A different sampling temperature rarely fixes a missed *exact* count
    # requirement (e.g. "exactly two sentences", "exactly three bullet
    # points") -- that's a structural mismatch a resample doesn't reliably
    # correct, not a content-quality issue. Retrying anyway still escalates
    # afterward, so it only adds local CPU time for near-zero expected
    # benefit; skip straight to one attempt for these prompts.
    return kind is TaskKind.SUMMARY and bool(
        _STRICT_FORMAT_PATTERN.search(prompt.lower())
    )


def _local_attempt(
    task: dict[str, str],
    prompt: str,
    settings: Settings,
    local_model: LocalGGUFModel,
    decision: RouteDecision,
) -> dict[str, str] | None:
    """Try the local model for a LOCAL-routed task, with one free retry.

    Returns the finished result if either attempt validates, or None if the
    caller should fall back to Fireworks.
    """
    task_id = task["task_id"]
    local_budget = local_output_budget(
        decision.kind, prompt, settings.default_max_tokens
    )
    temperatures = (
        (0.0,)
        if _retry_unlikely_to_help(decision.kind, prompt)
        else (0.0, _LOCAL_RETRY_TEMPERATURE)
    )
    first_answer: str | None = None
    for attempt, temperature in enumerate(temperatures):
        try:
            answer = local_model.complete(
                prompt, decision.kind, local_budget, temperature=temperature
            )
        except LocalModelError as exc:
            # A load/context/inference failure won't be fixed by resampling.
            logging.warning(
                "local model failed for task %s; falling back to Fireworks: %s",
                task_id,
                exc,
            )
            return None

        if first_answer is None:
            first_answer = answer
        if validate_local_answer(decision.kind, prompt, answer):
            return {"task_id": task_id, "answer": answer}

        # Free second chance before escalating: a deterministic repair (e.g.
        # re-adding NER entities dropped by the 1.5B model, verbatim from the
        # source text) that then PASSES validation is safe to ship -- unlike
        # the zero-token mode's ship-anything, nothing unvalidated leaves.
        repaired = repair_local_answer(decision.kind, prompt, answer)
        if repaired != answer and validate_local_answer(
            decision.kind, prompt, repaired
        ):
            logging.info(
                "task %s: local answer repaired deterministically; shipping",
                task_id,
            )
            return {"task_id": task_id, "answer": repaired}

        if attempt < len(temperatures) - 1:
            logging.info(
                "local answer for task %s failed validation; retrying locally",
                task_id,
            )
        else:
            logging.info(
                "local answer for task %s failed validation (attempt %s/%s)",
                task_id,
                attempt + 1,
                len(temperatures),
            )

    if not settings.allow_escalation and first_answer is not None:
        # Zero-token mode: an unvalidated local answer still clears the 50%
        # accuracy gate far more cheaply than any Fireworks call. Ship the
        # temperature-0 attempt (deterministic, most-likely decoding), after
        # a deterministic repair pass (e.g. re-adding dropped NER entities).
        logging.info(
            "task %s: shipping unvalidated local answer (escalation disabled)",
            task_id,
        )
        repaired = repair_local_answer(decision.kind, prompt, first_answer)
        return {"task_id": task_id, "answer": repaired}
    return None


def _fireworks_attempt(
    task: dict[str, str],
    prompt: str,
    client: FireworksClient,
    settings: Settings,
    decision: RouteDecision,
    timeout_seconds: float,
) -> dict[str, str]:
    task_id = task["task_id"]
    max_tokens = output_budget(decision.kind, prompt, settings.default_max_tokens)

    last_error: Exception | None = None
    for model in ordered_models(
        settings.allowed_models,
        decision.kind,
        preferred_model=settings.preferred_fireworks_model,
    ):
        if client.is_model_unavailable(model):
            logging.info("skipping unavailable model %s for task %s", model, task_id)
            continue
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


def _answer_fireworks_lane(
    task: dict[str, str],
    client: FireworksClient,
    settings: Settings,
    decision: RouteDecision,
    remaining_seconds: float | None,
) -> dict[str, str]:
    if decision.target is RouteTarget.DETERMINISTIC and decision.answer is not None:
        # Locally computed exact answer: zero Fireworks tokens, no network.
        return {"task_id": task["task_id"], "answer": decision.answer}

    prompt = task["prompt"].strip()
    timeout_seconds = min(
        settings.request_timeout_seconds,
        max(1.0, remaining_seconds or settings.request_timeout_seconds),
    )
    return _fireworks_attempt(task, prompt, client, settings, decision, timeout_seconds)


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

    def record(index: int, result: dict[str, str]) -> None:
        with results_lock:
            results[index] = result
        flush()

    # Fireworks calls are network-bound and run concurrently; llama.cpp is not
    # thread-safe, so local tasks get a single serial lane alongside them. A
    # local answer that fails validation escalates onto the *concurrent*
    # Fireworks pool (submit_fireworks) rather than running inline on this
    # single-worker lane, so a fallback call never has to wait behind
    # whatever other local generation happens to be running.
    fireworks_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.fireworks_concurrency, thread_name_prefix="fireworks"
    )
    local_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="local"
    )
    futures: list[concurrent.futures.Future] = []
    futures_lock = threading.Lock()

    def submit_fireworks(
        index: int, task: dict[str, str], decision: RouteDecision
    ) -> None:
        with futures_lock:
            futures.append(
                fireworks_pool.submit(run_fireworks, index, task, decision)
            )

    def run_fireworks(index: int, task: dict[str, str], decision: RouteDecision) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 2:
            logging.error("skipping task %s: time limit reached", task["task_id"])
            return
        logging.info(
            "processing task %s (%s/%s)", task["task_id"], index + 1, len(tasks)
        )
        try:
            result = _answer_fireworks_lane(
                task, client, settings, decision, remaining
            )
        except Exception:
            logging.exception(
                "unexpected failure while answering task %s", task["task_id"]
            )
            result = {"task_id": task["task_id"], "answer": API_ERROR_ANSWER}
        record(index, result)

    # Time-pressure spillover: the local lane is serial, so on hardware
    # slower than the dev machine the queue can outlast the 540s budget and
    # the tail would ship as placeholder answers -- guaranteed accuracy-gate
    # damage that no token saving justifies. Before each local generation,
    # project the whole remaining local queue at the observed per-task pace
    # and divert to the concurrent Fireworks pool when it no longer fits.
    local_lane = {
        "queued": 0,  # local tasks not yet started
        "avg_seconds": 6.0,  # conservative prior until real completions land
        "completions": 0,
    }
    local_lane_lock = threading.Lock()
    _SPILL_SAFETY_SECONDS = 30.0
    _SPILL_EMA_ALPHA = 0.3

    def run_local(index: int, task: dict[str, str], decision: RouteDecision) -> None:
        remaining = deadline - time.monotonic()
        with local_lane_lock:
            local_lane["queued"] -= 1
            projected = local_lane["avg_seconds"] * (local_lane["queued"] + 1)
        if remaining <= 2:
            logging.error("skipping task %s: time limit reached", task["task_id"])
            return
        if projected > remaining - _SPILL_SAFETY_SECONDS:
            logging.warning(
                "task %s: local lane projected %.0fs but %.0fs remain; "
                "spilling to Fireworks",
                task["task_id"],
                projected,
                remaining,
            )
            submit_fireworks(index, task, decision)
            return
        logging.info(
            "processing task %s (%s/%s)", task["task_id"], index + 1, len(tasks)
        )
        started = time.monotonic()
        try:
            result = _local_attempt(
                task, task["prompt"].strip(), settings, local_model, decision
            )
            elapsed = time.monotonic() - started
            with local_lane_lock:
                if local_lane["completions"] == 0:
                    local_lane["avg_seconds"] = elapsed
                else:
                    local_lane["avg_seconds"] += _SPILL_EMA_ALPHA * (
                        elapsed - local_lane["avg_seconds"]
                    )
                local_lane["completions"] += 1
        except Exception:
            logging.exception(
                "unexpected failure while answering task %s", task["task_id"]
            )
            record(index, {"task_id": task["task_id"], "answer": API_ERROR_ANSWER})
            return
        if result is not None:
            record(index, result)
            return
        submit_fireworks(index, task, decision)

    decisions: list[RouteDecision | None] = [
        router.route(task["prompt"].strip()) if task["prompt"].strip() else None
        for task in tasks
    ]

    for index, (task, decision) in enumerate(zip(tasks, decisions)):
        if decision is None:
            record(index, {"task_id": task["task_id"], "answer": ""})
            continue
        logging.info(
            "route task=%s kind=%s target=%s confidence=%.3f margin=%.3f reason=%s",
            task["task_id"],
            decision.kind.value,
            decision.target.value,
            decision.confidence,
            decision.margin,
            decision.reason,
        )
        use_local_lane = (
            local_model is not None and decision.target is RouteTarget.LOCAL
        )
        if use_local_lane:
            with local_lane_lock:
                local_lane["queued"] += 1
            futures.append(local_pool.submit(run_local, index, task, decision))
        else:
            futures.append(fireworks_pool.submit(run_fireworks, index, task, decision))

    # The local lane can enqueue new Fireworks futures while we wait (a
    # validation-triggered escalation), so keep waiting on the latest
    # snapshot until the future list stops growing or the deadline passes.
    pending: list[concurrent.futures.Future] = []
    while True:
        with futures_lock:
            snapshot = list(futures)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pending = [f for f in snapshot if not f.done()]
            break
        _, still_pending = concurrent.futures.wait(snapshot, timeout=remaining)
        with futures_lock:
            grew = len(futures) > len(snapshot)
        if not grew:
            pending = list(still_pending)
            break

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
    """Synchronous local-then-Fireworks-fallback entry point.

    Kept for eval/eval.py and eval/run_modes.py, which call this directly and
    expect one blocking call. process_tasks uses the split
    _local_attempt/_answer_fireworks_lane helpers instead, so a validation
    escalation can run on the concurrent Fireworks pool rather than inline.
    """
    task_id = task["task_id"]
    prompt = task["prompt"].strip()
    if not prompt:
        return {"task_id": task_id, "answer": ""}

    if decision is None:
        router = router or SemanticRouter(settings)
        decision = router.route(prompt)

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
        result = _local_attempt(task, prompt, settings, model, decision)
        if result is not None:
            return result

    return _answer_fireworks_lane(task, client, settings, decision, remaining_seconds)


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
        "tokens task=%s model=%s prompt=%s completion=%s total=%s "
        "finish_reason=%s retried=%s",
        task_id,
        usage.get("model", ""),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
        usage.get("finish_reason", ""),
        bool(usage.get("retried", False)),
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

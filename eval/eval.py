"""Local eval harness: accuracy, routing, and token cost per category."""

from __future__ import annotations

import json
import os
import time

from agent.config import load_settings
from agent.fireworks import FireworksClient
from agent.local_model import LocalGGUFModel, LocalModelError
from agent.router import RouteTarget, SemanticRouter
from eval.grading import looks_correct
from main import answer_task


def main() -> None:
    settings = load_settings()
    client = FireworksClient(settings)
    router = SemanticRouter(settings)
    local_model = LocalGGUFModel(settings) if settings.enable_local_model else None

    if local_model is not None:
        try:
            local_model._load()
        except LocalModelError as exc:
            print(f"WARNING: local model unavailable ({exc}).")
            print("Local-routed tasks will fall back to Fireworks, so this run does")
            print("NOT measure local answer quality. Run inside the Docker image.")
            print()

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "test_cases.json"), encoding="utf-8") as handle:
        cases = json.load(handle)

    stats: dict[str, dict[str, float]] = {}

    print(
        f"{'id':>8}  {'ok':>2}  {'routed as':<17} {'target':<9} "
        f"{'tokens':>6} {'time':>6}  answer"
    )
    print("-" * 100)

    for case in cases:
        task_id = str(case.get("task_id", case.get("id", "")))
        prompt = str(case.get("prompt", case.get("task", "")))
        category = str(case.get("category", ""))

        decision = router.route(prompt)
        tokens_before = int(client.total_usage["total_tokens"])
        case_start = time.monotonic()
        result = answer_task(
            {"task_id": task_id, "prompt": prompt},
            client,
            settings,
            router=router,
            local_model=local_model,
            decision=decision,
        )
        elapsed = time.monotonic() - case_start
        tokens = int(client.total_usage["total_tokens"]) - tokens_before

        ok = looks_correct(case, result["answer"])
        route_ok = not category or decision.kind.value == category
        kind_label = decision.kind.value[:15] + ("" if route_ok else "*")

        bucket = stats.setdefault(
            category or decision.kind.value,
            {"n": 0, "ok": 0, "route_ok": 0, "local": 0, "tokens": 0, "seconds": 0.0},
        )
        bucket["n"] += 1
        bucket["ok"] += int(ok)
        bucket["route_ok"] += int(route_ok)
        bucket["local"] += int(decision.target is RouteTarget.LOCAL)
        bucket["tokens"] += tokens
        bucket["seconds"] += elapsed

        answer_preview = " ".join(result["answer"].split())[:45]
        print(
            f"{task_id:>8}  {'OK' if ok else ' X':>2}  {kind_label:<17} "
            f"{decision.target.value:<9} {tokens:>6} {elapsed:>5.1f}s  {answer_preview}"
        )

    print("-" * 100)
    print("* = router category differs from the test case category")
    print()
    print(
        f"{'category':<26} {'accuracy':>9} {'route-ok':>9} "
        f"{'local':>6} {'tokens':>7} {'time':>8}"
    )
    total = {"n": 0, "ok": 0, "route_ok": 0, "local": 0, "tokens": 0, "seconds": 0.0}
    for category, bucket in stats.items():
        for key in total:
            total[key] += bucket[key]
        print(_summary_row(category, bucket))
    print("-" * 68)
    print(_summary_row("TOTAL", total))

    accuracy = 100.0 * total["ok"] / max(1.0, total["n"])
    print()
    print(
        f"Overall: {accuracy:.0f}% heuristic accuracy, "
        f"{int(total['tokens'])} Fireworks tokens, {total['seconds']:.0f}s wall time"
    )

    if local_model is not None:
        local_model.close()


def _summary_row(label: str, bucket: dict[str, float]) -> str:
    count = int(bucket["n"])
    accuracy = f"{int(bucket['ok'])}/{count}"
    route_ok = f"{int(bucket['route_ok'])}/{count}"
    local = f"{int(bucket['local'])}/{count}"
    return (
        f"{label:<26} {accuracy:>9} {route_ok:>9} "
        f"{local:>6} {int(bucket['tokens']):>7} {bucket['seconds']:>7.1f}s"
    )


if __name__ == "__main__":
    main()

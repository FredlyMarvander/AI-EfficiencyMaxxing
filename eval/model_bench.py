"""Benchmark every accessible Fireworks model on the Fireworks-only remainder.

Runs the categories that still reach Fireworks after deterministic solving
and local-first validation (debugging, logic, and the NER cases the local
model fails), grading with the shared heuristics and recording exact billed
tokens per model per category.

  python -m eval.model_bench                     # all accessible models
  python -m eval.model_bench --models gemma      # substring filter
  python -m eval.model_bench --categories code_debugging --limit 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time

os.environ.setdefault("LOG_LEVEL", "WARNING")

from agent.config import load_settings
from agent.fireworks import FireworksAPIError, FireworksClient
from agent.router import TaskKind, output_budget
from eval.grading import looks_correct


# NER ids the local model cannot answer (validation escalates them); these
# are exactly the NER calls Fireworks still receives.
NER_ESCALATION_IDS = {
    "ner-01", "ner-02", "ner-05", "ner-06", "ner-09",
    "ner-11", "ner-13", "ner-19", "ner-24",
}

DEFAULT_CATEGORIES = ("code_debugging", "logical_reasoning", "named_entity_recognition")

KIND_BY_CATEGORY = {kind.value: kind for kind in TaskKind}


def load_cases(path: str, categories: list[str], limit: int) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    picked = []
    for case in cases:
        category = case.get("category", "")
        if category not in categories:
            continue
        if (
            category == "named_entity_recognition"
            and str(case.get("id")) not in NER_ESCALATION_IDS
        ):
            continue
        picked.append(case)
    if limit:
        by_category: dict[str, int] = {}
        trimmed = []
        for case in picked:
            count = by_category.get(case["category"], 0)
            if count < limit:
                trimmed.append(case)
                by_category[case["category"]] = count + 1
        picked = trimmed
    return picked


def accessible_models(settings, name_filter: str) -> list[str]:
    report = os.path.join(os.path.dirname(__file__), "reports", "model_check.json")
    models = list(settings.allowed_models)
    if os.path.exists(report):
        with open(report, encoding="utf-8") as handle:
            checked = json.load(handle)
        usable = {row["model"] for row in checked if row.get("accessible")}
        models = [model for model in models if model in usable]
    if name_filter:
        models = [model for model in models if name_filter in model]
    return models


def bench_model(model: str, cases: list[dict], settings, concurrency: int) -> dict:
    client = FireworksClient(settings)
    rows: list[dict | None] = [None] * len(cases)

    def _run(index: int, case: dict) -> None:
        kind = KIND_BY_CATEGORY[case["category"]]
        budget = output_budget(kind, str(case["prompt"]), settings.default_max_tokens)
        started = time.monotonic()
        try:
            answer, usage = client.complete_with_usage(
                prompt=str(case["prompt"]),
                model=model,
                max_tokens=budget,
                timeout_seconds=settings.request_timeout_seconds,
                kind=kind,
            )
            rows[index] = {
                "id": str(case["id"]),
                "category": case["category"],
                "ok": looks_correct(case, answer),
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "tokens": int(usage["total_tokens"]),
                "seconds": round(time.monotonic() - started, 1),
                "answer": answer[:400],
            }
        except FireworksAPIError as exc:
            rows[index] = {
                "id": str(case["id"]),
                "category": case["category"],
                "ok": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens": 0,
                "seconds": round(time.monotonic() - started, 1),
                "answer": f"API ERROR: {exc}"[:200],
            }

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures = [pool.submit(_run, i, case) for i, case in enumerate(cases)]
    concurrent.futures.wait(futures)
    pool.shutdown()
    return {"model": model, "results": [row for row in rows if row]}


def summarize(report: dict) -> None:
    stats: dict[str, dict[str, float]] = {}
    for row in report["results"]:
        bucket = stats.setdefault(
            row["category"],
            {"n": 0, "ok": 0, "prompt": 0, "completion": 0, "tokens": 0, "seconds": 0.0},
        )
        bucket["n"] += 1
        bucket["ok"] += int(row["ok"])
        bucket["prompt"] += row["prompt_tokens"]
        bucket["completion"] += row["completion_tokens"]
        bucket["tokens"] += row["tokens"]
        bucket["seconds"] += row["seconds"]

    print(f"\n--- {report['model']}")
    print(f"{'category':<26} {'acc':>7} {'prompt':>7} {'compl':>7} {'total':>7} {'avg':>6} {'time':>7}")
    for category, bucket in sorted(stats.items()):
        n = int(bucket["n"])
        print(
            f"{category:<26} {int(bucket['ok'])}/{n:<5} {int(bucket['prompt']):>7} "
            f"{int(bucket['completion']):>7} {int(bucket['tokens']):>7} "
            f"{bucket['tokens'] / max(1, n):>6.0f} {bucket['seconds']:>6.0f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="substring filter")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--limit", type=int, default=0, help="cases per category")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases_v2.json"),
    )
    args = parser.parse_args()

    settings = load_settings()
    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    cases = load_cases(args.cases, categories, args.limit)
    models = accessible_models(settings, args.models)
    if not models:
        print("no accessible models match; run eval.model_check first")
        return 2

    print(f"{len(cases)} case(s) x {len(models)} model(s)")
    reports = []
    for model in models:
        print(f"\nbenchmarking {model} ...", flush=True)
        report = bench_model(model, cases, settings, args.concurrency)
        reports.append(report)
        out = os.path.join(
            os.path.dirname(__file__), "reports",
            f"bench_{model.rsplit('/', 1)[-1]}.json",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        summarize(report)

    print("\n=== ranking (total tokens on identical cases; accuracy first) ===")
    for report in reports:
        ok = sum(row["ok"] for row in report["results"])
        n = len(report["results"])
        tokens = sum(row["tokens"] for row in report["results"])
        print(f"{report['model']:<55} {ok}/{n:<5} {tokens:>8} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())

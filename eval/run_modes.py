"""Multi-mode eval runner: compares routing configurations on one case set.

Modes:
  fireworks  everything (except deterministic=off too) goes to Fireworks
  det        deterministic local solvers + Fireworks for the rest
  hybrid     deterministic + local GGUF model + Fireworks (current default)

Usage (inside the Docker image so all models are present):
  python -m eval.run_modes --modes fireworks,det,hybrid
  python -m eval.run_modes --modes det --category math --limit 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep per-task INFO logs quiet

from agent.config import Settings, load_settings
from agent.fireworks import FireworksClient
from agent.local_model import LocalGGUFModel, LocalModelError
from agent.router import RouteTarget, SemanticRouter
from eval.grading import looks_correct
from main import answer_task


MODES: dict[str, dict[str, bool]] = {
    "fireworks": {"enable_local_model": False, "enable_deterministic": False},
    "det": {"enable_local_model": False, "enable_deterministic": True},
    "hybrid": {"enable_local_model": True, "enable_deterministic": True},
}

DEFAULT_CASES = os.path.join(os.path.dirname(__file__), "test_cases_v2.json")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def run_case(
    case: dict,
    settings: Settings,
    router: SemanticRouter,
    local_model: LocalGGUFModel | None,
) -> dict:
    task_id = str(case.get("id", case.get("task_id", "")))
    prompt = str(case["prompt"])
    decision = router.route(prompt)

    # One client per case so token usage attribution is exact even when
    # cases run concurrently.
    client = FireworksClient(settings)
    started = time.monotonic()
    result = answer_task(
        {"task_id": task_id, "prompt": prompt},
        client,
        settings,
        router=router,
        local_model=local_model,
        decision=decision,
    )
    elapsed = time.monotonic() - started

    answer = result["answer"]
    return {
        "id": task_id,
        "category": str(case.get("category", "")),
        "kind": decision.kind.value,
        "target": decision.target.value,
        "ok": looks_correct(case, answer),
        "route_ok": (
            not case.get("category") or decision.kind.value == case["category"]
        ),
        "tokens": int(client.total_usage["total_tokens"]),
        "seconds": round(elapsed, 2),
        "answer": answer,
    }


def run_mode(mode: str, cases: list[dict], base: Settings, concurrency: int) -> dict:
    settings = dataclasses.replace(base, **MODES[mode])
    router = SemanticRouter(settings)
    local_model = None
    local_available = False
    if settings.enable_local_model:
        local_model = LocalGGUFModel(settings)
        try:
            local_model._load()
            local_available = True
        except LocalModelError as exc:
            print(f"  WARNING: local model unavailable ({exc}); "
                  "local-routed cases will fall back to Fireworks.")

    rows: list[dict | None] = [None] * len(cases)

    def _run(index: int, case: dict) -> None:
        try:
            rows[index] = run_case(case, settings, router, local_model)
        except Exception as exc:  # keep the sweep alive; report the failure
            rows[index] = {
                "id": str(case.get("id", "")),
                "category": str(case.get("category", "")),
                "kind": "?", "target": "?", "ok": False, "route_ok": False,
                "tokens": 0, "seconds": 0.0, "answer": f"RUNNER ERROR: {exc}",
            }

    # llama.cpp is not thread-safe: local-lane cases run on one thread,
    # network-bound cases fan out.
    remote_pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    local_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []
    for index, case in enumerate(cases):
        decision = router.route(str(case["prompt"]))
        lane = (
            local_pool
            if settings.enable_local_model and decision.target is RouteTarget.LOCAL
            else remote_pool
        )
        futures.append(lane.submit(_run, index, case))
    concurrent.futures.wait(futures)
    remote_pool.shutdown()
    local_pool.shutdown()
    if local_model is not None:
        local_model.close()

    results = [row for row in rows if row is not None]
    return {
        "mode": mode,
        "local_model_available": local_available,
        "results": results,
    }


def summarize(report: dict) -> None:
    results = report["results"]
    stats: dict[str, dict[str, float]] = {}
    for row in results:
        bucket = stats.setdefault(
            row["category"] or row["kind"],
            {"n": 0, "ok": 0, "route_ok": 0, "local": 0, "det": 0,
             "tokens": 0, "seconds": 0.0},
        )
        bucket["n"] += 1
        bucket["ok"] += int(row["ok"])
        bucket["route_ok"] += int(row["route_ok"])
        bucket["local"] += int(row["target"] == "local")
        bucket["det"] += int(row["target"] == "deterministic")
        bucket["tokens"] += row["tokens"]
        bucket["seconds"] += row["seconds"]

    print(f"\n=== mode: {report['mode']} ===")
    header = (
        f"{'category':<26} {'acc':>7} {'route':>7} {'local':>6} "
        f"{'det':>5} {'tokens':>8} {'time':>8}"
    )
    print(header)
    print("-" * len(header))
    total = {"n": 0, "ok": 0, "route_ok": 0, "local": 0, "det": 0,
             "tokens": 0, "seconds": 0.0}
    for category in sorted(stats):
        bucket = stats[category]
        for key in total:
            total[key] += bucket[key]
        print(_row(category, bucket))
    print("-" * len(header))
    print(_row("TOTAL", total))
    accuracy = 100.0 * total["ok"] / max(1, total["n"])
    print(
        f"overall: {accuracy:.1f}% | fireworks tokens: {int(total['tokens'])} "
        f"| wall time (sum): {total['seconds']:.0f}s"
    )

    failures = [row for row in results if not row["ok"]]
    if failures:
        print(f"\nfailures ({len(failures)}):")
        for row in failures:
            preview = " ".join(str(row["answer"]).split())[:90]
            print(f"  {row['id']:<10} [{row['kind']}/{row['target']}] {preview}")


def _row(label: str, bucket: dict[str, float]) -> str:
    count = int(bucket["n"])
    return (
        f"{label:<26} {int(bucket['ok'])}/{count:<4} "
        f"{int(bucket['route_ok'])}/{count:<4} "
        f"{int(bucket['local']):>6} {int(bucket['det']):>5} "
        f"{int(bucket['tokens']):>8} {bucket['seconds']:>7.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="fireworks,det,hybrid")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--category", default="")
    parser.add_argument("--ids", default="", help="comma-separated case ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        print(f"unknown mode(s): {unknown}; choose from {sorted(MODES)}")
        return 2

    with open(args.cases, encoding="utf-8") as handle:
        cases = json.load(handle)
    if args.category:
        cases = [case for case in cases if case.get("category") == args.category]
    if args.ids:
        wanted = {piece.strip() for piece in args.ids.split(",") if piece.strip()}
        cases = [case for case in cases if str(case.get("id", "")) in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected")
        return 2

    base = load_settings()
    os.makedirs(REPORT_DIR, exist_ok=True)

    # A filtered run (category/ids/limit) covers a subset of the full case
    # set; writing it to the same {mode}.json as a full sweep would silently
    # clobber the canonical report with partial data.
    is_filtered = bool(args.category or args.ids or args.limit)

    print(f"{len(cases)} case(s), modes: {', '.join(modes)}")
    reports = []
    for mode in modes:
        print(f"\nrunning mode {mode!r} ...")
        started = time.monotonic()
        report = run_mode(mode, cases, base, args.concurrency)
        report["elapsed_seconds"] = round(time.monotonic() - started, 1)
        reports.append(report)
        filename = f"{mode}.partial.json" if is_filtered else f"{mode}.json"
        path = os.path.join(REPORT_DIR, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
        print(f"mode {mode!r} finished in {report['elapsed_seconds']}s "
              f"-> {path}")

    for report in reports:
        summarize(report)

    if len(reports) > 1:
        print("\n=== comparison ===")
        print(f"{'mode':<12} {'accuracy':>9} {'tokens':>9} {'elapsed':>9}")
        for report in reports:
            results = report["results"]
            ok = sum(row["ok"] for row in results)
            tokens = sum(row["tokens"] for row in results)
            print(
                f"{report['mode']:<12} {ok}/{len(results):<6} {tokens:>9} "
                f"{report['elapsed_seconds']:>8.1f}s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

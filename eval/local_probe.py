"""Offline probe: run ONLY the local GGUF model on selected categories.

Zero Fireworks tokens. Measures whether the local model is accurate enough
for categories currently hard-wired to Fireworks, and saves raw answers so
candidate validators can be evaluated afterwards without re-running the
model.

  python -m eval.local_probe --categories named_entity_recognition,code_debugging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from agent.config import load_settings
from agent.local_model import LocalGGUFModel, LocalModelError
from agent.router import TaskKind
from eval.grading import looks_correct


# Local generation budgets: the 1.5B instruct model does not emit
# chain-of-thought, so these only need to fit the literal answer.
PROBE_BUDGETS = {
    "named_entity_recognition": 192,
    "code_debugging": 320,
    "logical_reasoning": 192,
    "code_generation": 384,
    "factual": 96,
    "sentiment": 24,
    "summarisation": 256,
    "math": 192,
}

KIND_BY_CATEGORY = {kind.value: kind for kind in TaskKind}

DEFAULT_CATEGORIES = (
    "named_entity_recognition,code_debugging,logical_reasoning,code_generation"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases_v2.json"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "reports", "local_probe.json"
        ),
    )
    args = parser.parse_args()
    wanted = {part.strip() for part in args.categories.split(",") if part.strip()}

    settings = load_settings()
    model = LocalGGUFModel(settings)
    model._load()

    with open(args.cases, encoding="utf-8") as handle:
        cases = [case for case in json.load(handle) if case.get("category") in wanted]

    rows = []
    correct_by_category: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        category = case["category"]
        kind = KIND_BY_CATEGORY[category]
        budget = PROBE_BUDGETS.get(category, 256)
        started = time.monotonic()
        try:
            answer = model.complete(str(case["prompt"]), kind, budget)
        except LocalModelError as exc:
            answer = f"LOCAL ERROR: {exc}"
        elapsed = time.monotonic() - started

        ok = looks_correct(case, answer)
        bucket = correct_by_category.setdefault(category, [0, 0])
        bucket[0] += int(ok)
        bucket[1] += 1
        rows.append(
            {
                "id": str(case.get("id", "")),
                "category": category,
                "ok": ok,
                "seconds": round(elapsed, 1),
                "answer": answer,
            }
        )
        print(
            f"[{index + 1}/{len(cases)}] {case.get('id', ''):<10} "
            f"{'OK ' if ok else 'X  '} {elapsed:5.1f}s",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle, ensure_ascii=False, indent=1)

    print("\nlocal-only accuracy by category:")
    for category, (ok, count) in sorted(correct_by_category.items()):
        print(f"  {category:<26} {ok}/{count}")
    print(f"saved raw answers -> {args.out}")
    model.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

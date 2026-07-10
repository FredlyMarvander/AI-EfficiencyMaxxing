"""Offline routing sweep: no API calls, just router decisions per case.

  python -m eval.route_check [--cases eval/test_cases_v2.json]

Requires only the embedding model (present in the Docker image); with the
model missing it still runs on the hashing fallback, clearly labelled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from agent.config import load_settings
from agent.router import SemanticRouter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases_v2.json"),
    )
    args = parser.parse_args()

    settings = load_settings()
    router = SemanticRouter(settings)
    if router.encoder_is_fallback:
        print("NOTE: hashing fallback encoder in use (embedding model missing)\n")

    with open(args.cases, encoding="utf-8") as handle:
        cases = json.load(handle)

    wrong: list[tuple[str, str, str, str]] = []
    targets: Counter[str] = Counter()
    by_category: dict[str, list[int]] = {}

    for case in cases:
        category = str(case.get("category", ""))
        decision = router.route(str(case["prompt"]))
        targets[decision.target.value] += 1
        ok = decision.kind.value == category
        bucket = by_category.setdefault(category, [0, 0])
        bucket[0] += ok
        bucket[1] += 1
        if not ok:
            wrong.append(
                (str(case.get("id", "")), category, decision.kind.value,
                 decision.reason)
            )

    print(f"{'category':<26} {'route accuracy':>15}")
    print("-" * 43)
    total_ok = total_n = 0
    for category, (ok, count) in sorted(by_category.items()):
        total_ok += ok
        total_n += count
        print(f"{category:<26} {ok:>7}/{count}")
    print("-" * 43)
    print(f"{'TOTAL':<26} {total_ok:>7}/{total_n}")
    print(f"\ntargets: {dict(targets)}")

    if wrong:
        print(f"\nmisrouted ({len(wrong)}):")
        for case_id, want, got, reason in wrong:
            print(f"  {case_id:<10} {want:<26} -> {got:<26} ({reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

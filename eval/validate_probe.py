"""Offline check: how would validate.py have judged the saved probe answers?

Confusion matrix per category:
  accept+ok   -> saved Fireworks call (good)
  accept+bad  -> WRONG ANSWER SHIPPED (must be zero)
  reject+ok   -> wasted escalation (tokens, not accuracy)
  reject+bad  -> correct escalation (good)
"""

from __future__ import annotations

import json
import os
import sys

from agent.router import TaskKind
from agent.validate import validate_local_answer

KIND_BY_CATEGORY = {kind.value: kind for kind in TaskKind}


def main() -> int:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "test_cases_v2.json"), encoding="utf-8") as handle:
        cases = {str(case["id"]): case for case in json.load(handle)}
    with open(
        os.path.join(here, "reports", "local_probe.json"), encoding="utf-8"
    ) as handle:
        probe = json.load(handle)["results"]

    matrix: dict[str, dict[str, int]] = {}
    shipped_bad: list[str] = []
    for row in probe:
        case = cases.get(row["id"])
        if case is None:
            continue
        category = row["category"]
        if category not in ("named_entity_recognition", "code_generation"):
            continue
        kind = KIND_BY_CATEGORY[category]
        accepted = validate_local_answer(kind, str(case["prompt"]), row["answer"])
        key = ("accept" if accepted else "reject") + ("+ok" if row["ok"] else "+bad")
        bucket = matrix.setdefault(category, {})
        bucket[key] = bucket.get(key, 0) + 1
        if accepted and not row["ok"]:
            shipped_bad.append(row["id"])

    for category, bucket in sorted(matrix.items()):
        total = sum(bucket.values())
        print(f"{category} (n={total}):")
        for key in ("accept+ok", "accept+bad", "reject+ok", "reject+bad"):
            print(f"  {key:<11} {bucket.get(key, 0)}")
    if shipped_bad:
        print(f"\nDANGER accepted-but-wrong: {shipped_bad}")
        return 1
    print("\nno wrong answer would have shipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

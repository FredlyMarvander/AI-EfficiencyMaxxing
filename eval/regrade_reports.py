"""Re-score a saved report's answers with the current grading.looks_correct,
without re-calling the API. Useful after fixing a grading bug.

Usage: python -m eval.regrade_reports eval/reports/det.json eval/reports/fireworks.json
"""

from __future__ import annotations

import json
import os
import sys

from eval.grading import looks_correct
from eval.run_modes import summarize

CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases_v2.json")


def main() -> int:
    with open(CASES_PATH, encoding="utf-8") as handle:
        cases_by_id = {str(case["id"]): case for case in json.load(handle)}

    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        changed = 0
        for row in report["results"]:
            case = cases_by_id.get(row["id"])
            if case is None:
                continue
            new_ok = looks_correct(case, row["answer"])
            if new_ok != row["ok"]:
                changed += 1
            row["ok"] = new_ok
        print(f"{path}: {changed} case(s) changed verdict")
        summarize(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

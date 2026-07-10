"""Print the same per-category summary as run_modes, from saved report JSON.

Usage: python -m eval.summarize_reports eval/reports/det.json eval/reports/fireworks.json
"""

from __future__ import annotations

import json
import sys

from eval.run_modes import summarize


def main() -> int:
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        summarize(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

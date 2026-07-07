"""Small local smoke harness for the file-based runner."""

import json
import os

from agent.config import load_settings
from agent.fireworks import FireworksClient
from main import answer_task


def looks_correct(output: str, expected: str) -> bool:
    if not expected:
        return len((output or "").strip()) > 20
    return expected.strip().lower() in (output or "").strip().lower()


def main():
    settings = load_settings()
    client = FireworksClient(settings)
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "test_cases.json"), encoding="utf-8") as handle:
        cases = json.load(handle)

    passed = 0

    print(f"{'id':>12}  {'ok':>3}  answer")
    print("-" * 72)

    for case in cases:
        task = {
            "task_id": str(case.get("task_id", case.get("id", ""))),
            "prompt": str(case.get("prompt", case.get("task", ""))),
        }
        result = answer_task(task, client, settings)
        ok = looks_correct(result["answer"], case.get("expected", ""))
        passed += int(ok)

        print(
            f"{task['task_id']:>12}  "
            f"{'OK' if ok else 'X':>3}  "
            f"{result['answer'][:80]}"
        )

    print("-" * 72)
    print(f"Accuracy:     {passed}/{len(cases)}")


if __name__ == "__main__":
    main()

"""
Local eval harness.

Runs every task in test_cases.json through the agent and prints a
per-task breakdown plus totals (accuracy + tokens). This mirrors how
the competition scores you: token count + output accuracy.

Run:  python -m eval.eval
"""
import json
import os

from agent.router import run_agent


def looks_correct(output: str, expected: str) -> bool:
    """
    Very simple accuracy check. For open-ended tasks (expected == ""),
    we just check the answer is non-trivial. Swap this for the real
    grading method once it's known on launch day.
    """
    if not expected:
        return len((output or "").strip()) > 20
    return expected.strip().lower() in (output or "").strip().lower()


def main():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "test_cases.json")) as f:
        cases = json.load(f)

    total_tokens = 0
    passed = 0

    print(f"{'id':>3}  {'model':<7} {'tokens':>7}  {'ok':>3}  reason")
    print("-" * 52)

    for case in cases:
        result = run_agent(case["task"])
        ok = looks_correct(result["output"], case.get("expected", ""))

        total_tokens += result["total_tokens"]
        passed += int(ok)

        print(
            f"{case['id']:>3}  "
            f"{result['model_used']:<7} "
            f"{result['total_tokens']:>7}  "
            f"{'OK' if ok else 'X':>3}  "
            f"{result['route_reason']}"
        )

    print("-" * 52)
    print(f"Accuracy:     {passed}/{len(cases)}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()

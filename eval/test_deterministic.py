"""Offline unit tests for the deterministic solvers and the router.

Runs without network, models, or API keys:
    python -m eval.test_deterministic
"""

from __future__ import annotations

import sys

from agent.deterministic import try_solve
from agent.router import TaskKind, _lexical_kind


# (prompt, expected answer or None when the solver must decline)
SOLVER_CASES: list[tuple[str, str | None]] = [
    ("Calculate 17 multiplied by 23.", "391"),
    ("What is 15 percent of 240?", "36"),
    ("What is 15% of 240?", "36"),
    ("Calculate the average of these numbers: 4, 8, 15, 16, 23, 42.", "18"),
    ("What is the average of 10, 20 and 33?", "21"),
    ("What is 128 + 256?", "384"),
    ("Compute (12 + 8) * 3", "60"),
    ("What is 2^10?", "1024"),
    ("Evaluate 144 / 12.", "12"),
    ("What is 7 plus 5 minus 2?", "10"),
    ("What is 45 times 12?", "540"),
    ("What is 10 divided by 4?", "2.5"),
    ("what is the mean of 5, 10, 15?", "10"),
    # Single-variable linear equations: exact, so solvable.
    ("Solve the equation 2x + 5 = 19 for x.", "x = 7"),
    ("Solve for x: 3x - 7 = 20.", "x = 9"),
    ("Solve 3y - 4 = 11.", "y = 5"),
    ("Solve -3x - 4 = 11", "x = -5"),
    ("What is x if 5x = 45?", "x = 9"),
    ("If x + 5 = 19, what is x?", "x = 14"),
    # Temperature conversions: exact formulas.
    ("Convert 100 Fahrenheit to Celsius.", "37.78°C"),
    ("Convert 100 Fahrenheit to Celsius (give the number).", "37.78°C"),
    ("What is 37 Celsius in Fahrenheit?", "98.6°F"),
    ("Convert 0 C to K", "273.15K"),
    # Must decline: context, units, nonlinear algebra, or anything ambiguous.
    ("A shirt costs 40 dollars and is discounted by 25 percent. What is the final price?", None),
    ("A train travels 180 km in 2.5 hours. What is its speed?", None),
    ("What is the capital of France?", None),
    ("Solve the equation 2x + 5 = 19 for y.", None),
    ("Solve 2x + 3y = 19.", None),
    ("Solve x^2 + 5 = 19.", None),
    ("x = 5", None),
    ("Convert 25 Celsius to Celsius.", None),
    ("Convert 100 dollars to euros.", None),
    ("What is 15 percent of the population of France?", None),
    ("I bought 3 apples and 4 pears, how many fruits do I have?", None),
    ("What year is 100 years after 1923?", None),
    ("What is 1,000,000 divided by permutations of 5?", None),
    ("What is the probability of rolling a 6 twice: 1/6 * 1/6?", None),
    ("9 ** 9 ** 9 ** 9", None),
    ("What is 5!", None),
]

# (prompt, expected lexical category or None for "no lexical opinion")
ROUTER_CASES: list[tuple[str, TaskKind | None]] = [
    # Hidden-style sentiment
    ("What is the tone of this review: 'The food was cold and the staff rude.'", TaskKind.SENTIMENT),
    ("Describe the attitude expressed in this tweet.", TaskKind.SENTIMENT),
    ("Is this comment favorable or unfavorable toward the product?", TaskKind.SENTIMENT),
    # Hidden-style summarization
    ("What is the key takeaway from the following article? " + "word " * 50, TaskKind.SUMMARY),
    ("Give me the TL;DR of this thread.", TaskKind.SUMMARY),
    ("In one sentence, what does this long passage say? " + "word " * 50, TaskKind.SUMMARY),
    # Hidden-style NER
    ("List all the people, places, and organizations mentioned in this text.", TaskKind.NER),
    ("Extract the names and dates from the paragraph below.", TaskKind.NER),
    ("Identify the organizations in this press release.", TaskKind.NER),
    # Hidden-style debugging (needs code context)
    ("Why does this fail?\n\ndef f(x):\n    return x + '1'", TaskKind.DEBUGGING),
    ("What is wrong with this function?\n\ndef f():\n    pass", TaskKind.DEBUGGING),
    ("This code returns the wrong result:\nfor i in range(10): print(i)", TaskKind.DEBUGGING),
    # Hidden-style code generation
    ("Implement a stack with push and pop methods.", TaskKind.CODE),
    ("Write a function that merges two sorted lists.", TaskKind.CODE),
    ("Create a program that counts vowels in a string.", TaskKind.CODE),
    ("Fill in the missing code so the test passes.", TaskKind.CODE),
    # Guard rails: things that must NOT be hijacked by the new patterns
    ("What is wrong with this argument: all birds fly, penguins are birds?", TaskKind.LOGIC),
    ("Who wrote Pride and Prejudice?", TaskKind.FACTUAL),
    # No lexical opinion -> semantic router decides; that is correct here.
    ("In one sentence, define photosynthesis.", None),
    ("Summarize the customer feedback in two sentences.", TaskKind.SUMMARY),
    ("Classify the sentiment of this review as positive or negative.", TaskKind.SENTIMENT),
    ("Calculate 15 percent of 240 plus the cost of shipping.", TaskKind.MATH),
]


def main() -> int:
    failures = 0

    print("deterministic solver:")
    for prompt, want in SOLVER_CASES:
        got = try_solve(prompt)
        ok = got == want
        failures += not ok
        print(f"  {'OK ' if ok else 'FAIL'} {prompt[:64]!r} -> {got!r} (want {want!r})")

    print("lexical router:")
    for prompt, want in ROUTER_CASES:
        got = _lexical_kind(prompt)
        ok = got == want
        failures += not ok
        want_label = want.value if want else None
        got_label = got.value if got else None
        print(f"  {'OK ' if ok else 'FAIL'} {prompt[:64]!r} -> {got_label} (want {want_label})")

    print(f"\n{failures} failure(s)" if failures else "\nALL OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

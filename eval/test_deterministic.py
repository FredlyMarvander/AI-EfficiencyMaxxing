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
    # (Discount and plain speed prompts moved to WORD_PROBLEM_CASES: the
    # dedicated templates now solve them exactly.)
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

# Word-problem templates: (prompt, exact answer or None to decline).
# Positive cases mirror eval/test_cases_v2.json phrasing; negatives are
# near-misses that MUST fall through to the LLM.
WORD_PROBLEM_CASES: list[tuple[str, str | None]] = [
    # percent change / increase / decrease
    ("A price rises from 80 dollars to 100 dollars. What is the percent increase?", "25%"),
    ("The value fell from 200 to 150. What is the percent decrease?", "25%"),
    ("What is 80 increased by 25 percent?", "100"),
    ("What is 200 decreased by 10%?", "180"),
    ("A price rises from 80 dollars to 100 dollars. What is the percent decrease?", None),
    # discount / tip / tax / interest
    ("A shirt costs 40 dollars and is discounted by 25 percent. What is the final price in dollars?", "30"),
    ("A jacket costs 90 dollars and is on sale for 30% off. How much is the discount?", "27"),
    ("A dinner bill is 85 dollars. How much is a 20 percent tip in dollars?", "17"),
    ("An item costs 200 dollars plus 10 percent GST. What is the total price?", "220"),
    ("How much simple interest does 1000 dollars earn at 5 percent per year over 3 years, in dollars?", "150"),
    ("A shirt costs 40 dollars and is discounted by 25 percent, then taxed at 8%. What is the final price?", None),
    # fractions
    ("What is three quarters of 200?", "150"),
    ("What is half of 37?", "18.5"),
    ("What is two thirds of 90?", "60"),
    # speed / distance / time
    ("A train travels 180 kilometers in 2.5 hours. What is its average speed in km/h?", "72 km/h"),
    ("Driving at 60 miles per hour for 2.5 hours, how many miles do you cover?", "150 miles"),
    ("A train travels 180 kilometers in 2.5 hours. What is its average speed in mph?", None),
    ("A runner runs 100 meters in 20 seconds. What is his average speed?", "5 m/s"),
    ("A man walks 3000 meters in 2 hours. What is his average speed?", None),
    # proportional scaling
    ("If 5 pencils cost 2.50 dollars, how much do 8 pencils cost in dollars?", "4"),
    ("If 5 pencils cost 2.50 dollars, how much do 8 erasers cost?", None),
    ("A recipe for 4 servings uses 2 cups of flour. How many cups are needed for 10 servings?", "5 cups"),
    ("A car uses 6 liters of fuel per 100 km. How many liters does it need for a 250 km trip?", "15"),
    ("Three workers build a wall in 12 days. Working at the same rate, how many days would six workers need?", "6 days"),
    # unit conversion
    ("Convert 2.5 hours into minutes.", "150 minutes"),
    ("How many meters are there in 5 kilometers?", "5000 meters"),
    ("Convert 5 kilometers to miles.", None),
    ("Convert 100 dollars to euros.", None),
    # integers and lists
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("What is the least common multiple of 4 and 6?", "12"),
    ("What is the median of 3, 7, 9, 15, 21?", "9"),
    ("What is the median of 3, 7, 9, 15?", "8"),
    ("What is the next number in the sequence 2, 6, 18, 54?", "162"),
    ("What is the next number in the sequence 1, 1, 2, 3, 5, 8, 13?", "21"),
    ("What is the next number in the sequence 2, 6, 18?", None),
    ("What is the next number in the sequence 7, 9, 4, 11?", None),
    ("What is the square root of 144?", "12"),
    ("What is the square root of 145?", None),
    ("A positive number squared equals 169. What is the number?", "13"),
    # ratios, ages, counting
    ("A class of 30 students has boys and girls in the ratio 2:3. How many girls are there?", "18"),
    ("A class of 30 students has boys and girls in the ratio 2:3. How many teachers are there?", None),
    ("A class of 31 students has boys and girls in the ratio 2:3. How many girls are there?", None),
    ("Anna is twice as old as Ben. Together their ages add up to 36. How old is Ben?", "12"),
    ("Anna is twice as old as Ben. Together their ages add up to 36. How old is Anna?", "24"),
    ("There are twice as many cats as dogs. There are 5 dogs. How many cats?", "10"),
    ("A store had 120 phones and sold 45. How many are left?", "75"),
    ("A store had 120 phones, sold 45, then received 30 more. How many phones does it have now?", "105"),
    # probability
    ("What is the probability of getting heads on both of two fair coin flips?", "1/4 (0.25)"),
    ("What is the probability of getting heads 3 times in a row with a fair coin?", "1/8 (0.125)"),
    ("What is the probability of rolling an even number on a standard six-sided die?", "1/2 (0.5)"),
    ("What is the probability of rolling a 6 on a fair die?", "1/6 (0.1667)"),
    ("What is the probability of rolling two sixes in a row?", None),
    # geometry
    ("A rectangle is 12 meters long and 8 meters wide. What is its area in square meters?", "96"),
    ("A rectangle is 12 meters long and 8 centimeters wide. What is its area?", None),
    ("What is the perimeter of a square with side length 7 cm, in centimeters?", "28"),
    ("What is the area of a circle with radius 5? Round to one decimal place.", "78.5"),
    ("What is the sum of the interior angles of a triangle in degrees?", "180 degrees"),
    # deterministic logic
    ("Today is Wednesday. What day of the week will it be in exactly 100 days?", "Friday"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Answer with just the name.", "Carol"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is the oldest?", None),
    ("Alice is taller than Bob. Dave is taller than Carol. Who is the shortest?", None),
    ("Ann likes tea more than coffee, and coffee more than juice. Which of the three drinks does she like least?", "Juice"),
    # story problems that must stay with the LLM
    ("Tom finished the race before Jane but after Sam. Who finished first? Answer with just the name.", None),
    ("A bat and a ball cost 1.10 together. The bat costs 1 more than the ball. What does the ball cost?", None),
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

    print("word problems:")
    for prompt, want in WORD_PROBLEM_CASES:
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

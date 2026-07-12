"""Anchored template solvers for standard one-step word problems.

Same contract as agent.deterministic: every solver must be effectively
infallible. Each template anchors on the full prompt structure and declines
on anything it does not provably understand — a wrong exact answer is
unrecoverable, so precision always beats coverage. All returned strings are
plain values with the unit the question itself used.
"""

from __future__ import annotations

import math
import re


NUM = r"(-?\d+(?:\.\d+)?)"
NUM_NC = r"-?\d+(?:\.\d+)?"  # non-capturing variant for list patterns
INT = r"(\d+)"
MONEY_WORD = r"(?:dollars?|usd|\$|euros?|eur|pounds?|gbp|rupees?|inr|cents?)"
Q_PREFIX = r"(?:what\s+is|whats|what\s+will|how\s+much\s+is|calculate|compute|find|determine)?\s*"
TAIL = r"\s*[?.!]?\s*$"

# Word fractions that are exact by definition.
_FRACTION_WORDS = {
    "half": 0.5, "a half": 0.5, "one half": 0.5,
    "a third": 1 / 3, "one third": 1 / 3, "two thirds": 2 / 3,
    "a quarter": 0.25, "one quarter": 0.25, "three quarters": 0.75,
    "a fifth": 0.2, "one fifth": 0.2, "two fifths": 0.4,
    "three fifths": 0.6, "four fifths": 0.8,
    "a tenth": 0.1, "one tenth": 0.1,
}

# Exact metric / time conversion factors: target unit per source unit.
_CONVERSIONS: dict[tuple[str, str], float] = {}


def _register_conversion(source: str, target: str, factor: float) -> None:
    _CONVERSIONS[(source, target)] = factor
    _CONVERSIONS[(target, source)] = 1.0 / factor


for _src, _dst, _factor in (
    ("hours", "minutes", 60.0),
    ("minutes", "seconds", 60.0),
    ("hours", "seconds", 3600.0),
    ("days", "hours", 24.0),
    ("weeks", "days", 7.0),
    ("kilometers", "meters", 1000.0),
    ("meters", "centimeters", 100.0),
    ("meters", "millimeters", 1000.0),
    ("centimeters", "millimeters", 10.0),
    ("kilometers", "centimeters", 100000.0),
    ("kilograms", "grams", 1000.0),
    ("grams", "milligrams", 1000.0),
    ("tonnes", "kilograms", 1000.0),
    ("liters", "milliliters", 1000.0),
):
    _register_conversion(_src, _dst, _factor)

_UNIT_ALIASES = {
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours", "h": "hours",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
    "second": "seconds", "seconds": "seconds", "sec": "seconds", "secs": "seconds", "s": "seconds",
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
    "kilometer": "kilometers", "kilometers": "kilometers",
    "kilometre": "kilometers", "kilometres": "kilometers", "km": "kilometers",
    "meter": "meters", "meters": "meters", "metre": "meters", "metres": "meters", "m": "meters",
    "centimeter": "centimeters", "centimeters": "centimeters",
    "centimetre": "centimeters", "centimetres": "centimeters", "cm": "centimeters",
    "millimeter": "millimeters", "millimeters": "millimeters",
    "millimetre": "millimeters", "millimetres": "millimeters", "mm": "millimeters",
    "kilogram": "kilograms", "kilograms": "kilograms", "kg": "kilograms",
    "gram": "grams", "grams": "grams", "g": "grams",
    "milligram": "milligrams", "milligrams": "milligrams", "mg": "milligrams",
    "tonne": "tonnes", "tonnes": "tonnes", "ton": "tonnes", "tons": "tonnes",
    "liter": "liters", "liters": "liters", "litre": "liters", "litres": "liters", "l": "liters",
    "milliliter": "milliliters", "milliliters": "milliliters",
    "millilitre": "milliliters", "millilitres": "milliliters", "ml": "milliliters",
}


def try_solve_word_problem(text: str) -> str | None:
    for solver in _SOLVERS:
        answer = solver(text)
        if answer is not None:
            return answer
    return None


def _fmt(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        formatted = f"{value:.{decimals}f}"
        return formatted
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _rounding_from(text: str) -> int | None:
    match = re.search(
        r"round(?:ed)?\s+(?:\w+\s+)?to\s+(?:the\s+nearest\s+(whole\s+number|integer)"
        r"|(one|two|three|1|2|3)\s+decimal\s+places?)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    if match.group(1):
        return 0
    return {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}[
        match.group(2).lower()
    ]


def _clean_int(value: float) -> str | None:
    """Format a value that must be a whole number, or refuse."""
    if abs(value - round(value)) > 1e-9:
        return None
    return str(int(round(value)))


# --- percentages -----------------------------------------------------------


def _percent_change(text: str) -> str | None:
    # "A price rises from 80 dollars to 100 dollars. What is the percent increase?"
    match = re.fullmatch(
        r"\s*(?:a|the)?\s*[\w ]{0,30}?(?:rises?|rose|increases?d?|grows?|grew|went\s+up|"
        r"falls?|fell|decreases?d?|drops?|dropped|went\s+down)\s+from\s+" + NUM +
        r"\s*(?:" + MONEY_WORD + r"|\w{0,12})?\s+to\s+" + NUM +
        r"\s*(?:" + MONEY_WORD + r"|\w{0,12})?\s*[.,]?\s*"
        r"(?:what\s+is|whats|calculate|compute|find)?\s*(?:the\s+)?"
        r"percent(?:age)?\s+(increase|decrease|change)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    old, new = float(match.group(1)), float(match.group(2))
    direction = match.group(3).lower()
    if old == 0:
        return None
    change = (new - old) / old * 100.0
    if direction == "increase" and change < 0:
        return None
    if direction == "decrease":
        if change > 0:
            return None
        change = -change
    if direction == "change":
        change = abs(change)
    return f"{_fmt(change)}%"


def _increase_decrease_by_percent(text: str) -> str | None:
    # "What is 80 increased by 25 percent?" / "Increase 80 by 25%."
    match = re.fullmatch(
        r"\s*(?:what\s+is|whats)?\s*(?:(increase|decrease)\s+)?" + NUM +
        r"\s+(?:(increased|decreased)\s+)?by\s+" + NUM + r"\s*(?:%|percent)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    verb = (match.group(1) or match.group(3) or "").lower()
    if not verb:
        return None
    base, percent = float(match.group(2)), float(match.group(4))
    sign = 1.0 if verb.startswith("increase") else -1.0
    return _fmt(base * (1.0 + sign * percent / 100.0))


def _discount(text: str) -> str | None:
    # "A shirt costs 40 dollars and is discounted by 25 percent. What is the
    # final price (in dollars)?" — also "...on sale for 25% off..."
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w -]{0,30}?\s*costs?\s+" + NUM +
        r"\s*(?:" + MONEY_WORD + r")?\s+and\s+is\s+"
        r"(?:discounted\s+by|on\s+sale\s+(?:for|at)|reduced\s+by|marked\s+down\s+(?:by)?)\s+"
        + NUM + r"\s*(?:%|percent)(?:\s+off)?\s*[.,]?\s*"
        r"(?:what\s+is|whats|calculate|compute|find|how\s+much\s+is)\s*(?:the\s+)?"
        r"(final\s+price|sale\s+price|new\s+price|discounted\s+price|price\s+now|"
        r"discount(?:\s+amount)?|amount\s+(?:you\s+)?saved?|savings?)"
        r"(?:\s+in\s+" + MONEY_WORD + r")?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    price, percent = float(match.group(1)), float(match.group(2))
    asked = match.group(3).lower()
    if percent < 0 or percent > 100:
        return None
    saved = price * percent / 100.0
    if "price" in asked:
        return _fmt(price - saved)
    return _fmt(saved)


def _tip_or_tax_amount(text: str) -> str | None:
    # "A dinner bill is 85 dollars. How much is a 20 percent tip (in dollars)?"
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w ]{0,30}?(?:bill|price|cost|total|amount)\s+"
        r"(?:is|comes\s+to|was)\s+" + NUM + r"\s*(?:" + MONEY_WORD + r")?\s*[.,]?\s*"
        r"(?:how\s+much\s+is|what\s+is|calculate|compute|find)\s+(?:a|an|the)?\s*"
        + NUM + r"\s*(?:%|percent)\s+(tip|tax|gst|vat|service\s+charge|surcharge)"
        r"(?:\s+in\s+" + MONEY_WORD + r")?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    base, percent = float(match.group(1)), float(match.group(2))
    return _fmt(base * percent / 100.0)


def _total_with_tax(text: str) -> str | None:
    # "An item costs 200 dollars plus 10 percent GST. What is the total price?"
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w -]{0,30}?costs?\s+" + NUM +
        r"\s*(?:" + MONEY_WORD + r")?\s*(?:,)?\s+plus\s+(?:a\s+)?" + NUM +
        r"\s*(?:%|percent)\s+(?:tip|tax|gst|vat|service\s+charge|surcharge)\s*[.,]?\s*"
        r"(?:what\s+is|whats|calculate|compute|find|how\s+much\s+is)\s+(?:the\s+)?"
        r"(?:total(?:\s+(?:price|cost|amount))?|final\s+(?:price|cost|amount))"
        r"(?:\s+in\s+" + MONEY_WORD + r")?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    base, percent = float(match.group(1)), float(match.group(2))
    return _fmt(base * (1.0 + percent / 100.0))


def _simple_interest(text: str) -> str | None:
    # "How much simple interest does 1000 dollars earn at 5 percent per year
    # over 3 years (in dollars)?"
    match = re.fullmatch(
        r"\s*(?:how\s+much\s+)?(?:simple\s+)?interest\s+(?:does|will|is\s+earned\s+(?:on|by))\s+"
        + NUM + r"\s*(?:" + MONEY_WORD + r")?\s+earn\s+at\s+" + NUM +
        r"\s*(?:%|percent)(?:\s+(?:per\s+year|per\s+annum|annually|yearly))?\s+"
        r"(?:over|for|in|after)\s+" + NUM + r"\s+years?"
        r"(?:\s*,?\s+in\s+" + MONEY_WORD + r")?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    principal, rate, years = (float(match.group(i)) for i in (1, 2, 3))
    return _fmt(principal * rate / 100.0 * years)


def _fraction_of(text: str) -> str | None:
    # "What is three quarters of 200?"
    words = "|".join(re.escape(w) for w in _FRACTION_WORDS)
    match = re.fullmatch(
        Q_PREFIX + r"(" + words + r")\s+of\s+" + NUM + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = _FRACTION_WORDS[match.group(1).lower()] * float(match.group(2))
    return _fmt(round(value, 9))


# --- rates, proportion, scaling ---------------------------------------------


_DISTANCE_UNITS = r"(kilometers?|kilometres?|km|miles?|meters?|metres?|m)"
_TIME_UNITS = r"(hours?|hrs?|h|minutes?|mins?|seconds?|secs?)"


def _canon_distance(raw: str) -> str:
    lowered = raw.lower()
    if lowered.startswith(("km", "kilo")):
        return "kilometers"
    if lowered.startswith("mile"):
        return "miles"
    return "meters"


def _canon_time(raw: str) -> str:
    lowered = raw.lower()
    if lowered.startswith("h"):
        return "hours"
    if lowered.startswith("min"):
        return "minutes"
    return "seconds"


def _speed_from_distance_time(text: str) -> str | None:
    # "A train travels 180 kilometers in 2.5 hours. What is its (average) speed ...?"
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w ]{0,30}?(?:travels?|travelled|traveled|covers?|drives?|"
        r"drove|runs?|ran|goes|went|flies|flew|cycles?|walks?|walked)\s+" + NUM +
        r"\s*" + _DISTANCE_UNITS + r"\s+in\s+" + NUM + r"\s*" + _TIME_UNITS +
        r"\s*[.,]?\s*(?:what\s+is|whats|calculate|compute|find)\s+(?:its|the|his|her|their)\s+"
        r"(?:average\s+)?speed(?:\s+in\s+([\w/]+))?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    distance = float(match.group(1))
    distance_unit = _canon_distance(match.group(2))
    duration = float(match.group(3))
    time_unit = _canon_time(match.group(4))
    asked_unit = (match.group(5) or "").lower().replace("per", "/").replace(" ", "")

    if duration == 0:
        return None
    if distance_unit == "meters":
        # Meter-scale speeds are conventionally m/s; with any other time base
        # the expected answer unit is ambiguous ("18000 m/h" would be marked
        # wrong for a 100m sprint), so decline and let the LLM phrase it.
        if time_unit != "seconds":
            return None
        speed = distance / duration
        unit_label = "m/s"
    else:
        hours = duration / {"hours": 1.0, "minutes": 60.0, "seconds": 3600.0}[time_unit]
        speed = distance / hours
        unit_label = "km/h" if distance_unit == "kilometers" else "mph"
    if asked_unit:
        aliases = {
            "km/h": "km/h", "kmh": "km/h", "km/hour": "km/h", "kph": "km/h",
            "mph": "mph", "miles/hour": "mph",
            "m/s": "m/s", "mps": "m/s", "meters/second": "m/s",
            "metres/second": "m/s",
        }
        wanted = aliases.get(asked_unit)
        if wanted is None or wanted != unit_label:
            return None  # asked in a unit we cannot exactly provide
    return f"{_fmt(speed)} {unit_label}"


def _distance_from_speed_time(text: str) -> str | None:
    # "Driving at 60 miles per hour for 2.5 hours, how many miles do you cover?"
    match = re.fullmatch(
        r"\s*(?:driving|traveling|travelling|walking|running|cycling|flying|going|moving)\s+at\s+"
        + NUM + r"\s*" + _DISTANCE_UNITS + r"\s*(?:per\s+hour|/h|an\s+hour)\s+for\s+"
        + NUM + r"\s*hours?\s*[.,]?\s*how\s+(?:many|far|much)\s*"
        r"(?:kilometers?|kilometres?|km|miles?|meters?|metres?)?\s*"
        r"(?:do\s+you|does\s+\w+|will\s+\w+|is)?\s*"
        r"(?:cover(?:ed)?|travel(?:led|ed)?|go(?:ne)?|driven?|drive)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    speed = float(match.group(1))
    unit = {"kilometers": "km", "miles": "miles", "meters": "m"}[
        _canon_distance(match.group(2))
    ]
    hours = float(match.group(3))
    return f"{_fmt(speed * hours)} {unit}"


def _proportional_cost(text: str) -> str | None:
    # "If 5 pencils cost 2.50 dollars, how much do 8 pencils cost (in dollars)?"
    match = re.fullmatch(
        r"\s*if\s+" + INT + r"\s+([\w -]{1,25}?)\s+costs?\s+" + NUM +
        r"\s*(?:" + MONEY_WORD + r")?\s*,?\s+how\s+much\s+(?:do|does|would|will)\s+"
        + INT + r"\s+([\w -]{1,25}?)\s+cost(?:\s+in\s+" + MONEY_WORD + r")?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    count_a = float(match.group(1))
    item_a = match.group(2).strip().lower()
    cost_a = float(match.group(3))
    count_b = float(match.group(4))
    item_b = match.group(5).strip().lower()
    if count_a == 0 or item_a != item_b:
        return None
    return _fmt(round(cost_a / count_a * count_b, 9))


def _recipe_scaling(text: str) -> str | None:
    # "A recipe for 4 servings uses 2 cups of flour. How many cups are needed
    # for 10 servings?"
    match = re.fullmatch(
        r"\s*(?:a|the)?\s*recipe\s+for\s+" + INT + r"\s+(?:servings?|people|persons?)\s+"
        r"(?:uses|needs|requires|calls\s+for)\s+" + NUM + r"\s+([\w ]{1,20}?)\s+of\s+"
        r"([\w -]{1,25})\s*[.,]?\s*how\s+(?:much|many)\s+(?:[\w ]{1,20}?\s+)?"
        r"(?:is|are)\s+(?:needed|required)\s+for\s+" + INT +
        r"\s+(?:servings?|people|persons?)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    base_servings = float(match.group(1))
    quantity = float(match.group(2))
    unit = match.group(3).strip()
    target_servings = float(match.group(5))
    if base_servings == 0:
        return None
    scaled = quantity * target_servings / base_servings
    return f"{_fmt(round(scaled, 9))} {unit}"


def _rate_per_100(text: str) -> str | None:
    # "A car uses 6 liters of fuel per 100 km. How many liters does it need
    # for a 250 km trip?"
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w ]{0,20}?uses?\s+" + NUM +
        r"\s+(liters?|litres?)\s+(?:of\s+[\w ]{1,15}\s+)?per\s+100\s*"
        r"(?:km|kilometers?|kilometres?)\s*[.,]?\s*how\s+many\s+(?:liters?|litres?)\s+"
        r"(?:does\s+it\s+need|are\s+needed|will\s+it\s+use|does\s+it\s+use)\s+for\s+"
        r"(?:a\s+)?" + NUM + r"\s*(?:km|kilometers?|kilometres?)(?:\s+trip|\s+journey|\s+drive)?"
        + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    per_100 = float(match.group(1))
    distance = float(match.group(3))
    return _fmt(round(per_100 * distance / 100.0, 9))


def _inverse_work_rate(text: str) -> str | None:
    # "Three workers build a wall in 12 days. ... how many days would six
    # workers need?" — requires the same-job phrasing and integer word or digit
    # counts on both sides.
    number_word = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)"
    match = re.fullmatch(
        r"\s*" + number_word + r"\s+(workers?|people|persons?|builders?|painters?|machines?)\s+"
        r"(?:build|builds|paint|paints|complete|completes|finish|finishes|do|does|make|makes)\s+"
        r"(?:a|an|the)\s+[\w -]{1,25}\s+in\s+" + number_word + r"\s+(days?|hours?|weeks?)\s*[.,]?\s*"
        r"(?:working\s+at\s+the\s+same\s+rate\s*,?\s*)?how\s+(?:many\s+)?(days?|hours?|weeks?)\s+"
        r"(?:would|will|do|does)\s+" + number_word + r"\s+(?:workers?|people|persons?|builders?|"
        r"painters?|machines?)\s+(?:need|take|require)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    }

    def _num(raw: str) -> float | None:
        raw = raw.lower()
        if raw.isdigit():
            return float(raw)
        return float(words[raw]) if raw in words else None

    crew_a = _num(match.group(1))
    duration = _num(match.group(3))
    time_word_given = match.group(4).rstrip("s")
    time_word_asked = match.group(5).rstrip("s")
    crew_b = _num(match.group(6))
    if None in (crew_a, duration, crew_b) or crew_b == 0:
        return None
    if time_word_given != time_word_asked:
        return None
    answer = crew_a * duration / crew_b
    return f"{_fmt(round(answer, 9))} {time_word_asked}s" if answer != 1 else f"1 {time_word_asked}"


def _unit_conversion(text: str) -> str | None:
    # "Convert 2.5 hours into minutes." / "How many meters are there in 5 kilometers?"
    match = re.fullmatch(
        r"\s*(?:please\s+)?convert\s+" + NUM + r"\s*([a-z]+)\s+(?:to|into|in)\s+([a-z]+)" + TAIL,
        text,
        re.IGNORECASE,
    ) or re.fullmatch(
        r"\s*how\s+many\s+([a-z]+)\s+(?:are\s+(?:there\s+)?in|is|make(?:\s+up)?)\s+"
        + NUM + r"\s*([a-z]+)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    groups = match.groups()
    if groups[0] and groups[0][0].isdigit() or "." in groups[0] or groups[0].lstrip("-").replace(".", "").isdigit():
        value_s, source_s, target_s = groups
    else:
        target_s, value_s, source_s = groups
    source = _UNIT_ALIASES.get(source_s.lower())
    target = _UNIT_ALIASES.get(target_s.lower())
    if not source or not target or source == target:
        return None
    factor = _CONVERSIONS.get((source, target))
    if factor is None:
        return None
    value = float(value_s) * factor
    return f"{_fmt(round(value, 9))} {target}"


# --- integers, lists, sequences ---------------------------------------------


def _gcd_lcm(text: str) -> str | None:
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?(greatest\s+common\s+(?:divisor|factor)|gcd|gcf|hcf|"
        r"highest\s+common\s+factor|least\s+common\s+multiple|lowest\s+common\s+multiple|lcm)\s+"
        r"of\s+(\d+(?:\s*(?:,\s*and|,|and)\s*\d+)+)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    numbers = [int(part) for part in re.findall(r"\d+", match.group(2))]
    if len(numbers) < 2 or any(n == 0 for n in numbers):
        return None
    op = match.group(1).lower()
    result = numbers[0]
    for n in numbers[1:]:
        result = math.gcd(result, n) if ("divisor" in op or "factor" in op or op in {"gcd", "gcf", "hcf"}) else result * n // math.gcd(result, n)
    return str(result)


def _median_or_range(text: str) -> str | None:
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?(median|range|sum|product|largest|smallest|maximum|minimum)\s+of\s+"
        r"(?:these\s+numbers\s*:?\s*|the\s+numbers\s*:?\s*|the\s+following(?:\s+numbers)?\s*:?\s*)?"
        r"(" + NUM_NC + r"(?:\s*(?:,\s*and|,|and)\s*" + NUM_NC + r")+)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    op = match.group(1).lower()
    numbers = [float(part) for part in re.findall(r"-?\d+(?:\.\d+)?", match.group(2))]
    if len(numbers) < 2:
        return None
    if op == "median":
        ordered = sorted(numbers)
        mid = len(ordered) // 2
        value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    elif op == "range":
        value = max(numbers) - min(numbers)
    elif op == "sum":
        value = sum(numbers)
    elif op == "product":
        value = 1.0
        for n in numbers:
            value *= n
    elif op in {"largest", "maximum"}:
        value = max(numbers)
    else:
        value = min(numbers)
    return _fmt(round(value, 9))


def _next_in_sequence(text: str) -> str | None:
    # Detects arithmetic, geometric, and Fibonacci-style sequences; declines
    # whenever fewer than four terms are given or patterns disagree.
    match = re.fullmatch(
        r"\s*(?:what\s+is|whats|find|give)?\s*(?:the\s+)?next\s+(?:number|term|value)\s+"
        r"(?:in|of)\s+(?:the\s+)?(?:sequence|series)\s*:?\s*"
        r"(" + NUM_NC + r"(?:\s*,\s*" + NUM_NC + r")+)"
        r"\s*(?:,\s*)?(?:\.\.\.|…)?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    terms = [float(part) for part in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(terms) < 4:
        return None

    candidates: set[float] = set()
    diffs = [b - a for a, b in zip(terms, terms[1:])]
    if all(abs(d - diffs[0]) < 1e-9 for d in diffs):
        candidates.add(terms[-1] + diffs[0])
    if all(t != 0 for t in terms):
        ratios = [b / a for a, b in zip(terms, terms[1:])]
        if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
            candidates.add(terms[-1] * ratios[0])
    if len(terms) >= 5 and all(
        abs(terms[i] - (terms[i - 1] + terms[i - 2])) < 1e-9
        for i in range(2, len(terms))
    ):
        candidates.add(terms[-1] + terms[-2])

    if len(candidates) != 1:
        return None
    return _fmt(round(candidates.pop(), 9))


def _square_root(text: str) -> str | None:
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?square\s+root\s+of\s+" + INT + TAIL,
        text,
        re.IGNORECASE,
    )
    if match:
        n = int(match.group(1))
        root = math.isqrt(n)
        return str(root) if root * root == n else None

    # "A positive number squared equals 169. What is the number?"
    match = re.fullmatch(
        r"\s*(?:a|the)\s+positive\s+number\s+squared\s+(?:equals|is)\s+" + INT +
        r"\s*[.,]?\s*(?:what\s+is|whats|find)\s+(?:the|that)\s+number" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    n = int(match.group(1))
    root = math.isqrt(n)
    return str(root) if root * root == n else None


# --- ratios, ages, counting --------------------------------------------------


def _ratio_split(text: str) -> str | None:
    # "A class of 30 students has boys and girls in the ratio 2:3. How many
    # girls are there?"
    match = re.fullmatch(
        r"\s*(?:a|the)?\s*[\w ]{0,20}?of\s+" + INT + r"\s+([\w]+)\s+has\s+([\w]+)\s+and\s+([\w]+)\s+"
        r"in\s+(?:the\s+|a\s+)?ratio\s+(?:of\s+)?" + INT + r"\s*(?::|to)\s*" + INT +
        r"\s*[.,]?\s*how\s+many\s+([\w]+)\s+(?:are\s+there|are\s+in\s+the\s+[\w]+|does\s+it\s+have)?"
        + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    total = int(match.group(1))
    label_a = match.group(3).lower()
    label_b = match.group(4).lower()
    part_a = int(match.group(5))
    part_b = int(match.group(6))
    asked = match.group(7).lower()
    if part_a + part_b == 0 or total % (part_a + part_b) != 0:
        return None
    unit = total // (part_a + part_b)
    if asked == label_a:
        return str(unit * part_a)
    if asked == label_b:
        return str(unit * part_b)
    return None


def _age_multiple_sum(text: str) -> str | None:
    # "Anna is twice as old as Ben. Together their ages add up to 36. How old
    # is Ben?"
    multiples = {"twice": 2, "double": 2, "three times": 3, "thrice": 3,
                 "four times": 4, "five times": 5}
    words = "|".join(re.escape(w) for w in multiples)
    match = re.fullmatch(
        r"\s*([A-Z][a-z]+)\s+is\s+(" + words + r")\s+as\s+old\s+as\s+([A-Z][a-z]+)\s*[.,]\s*"
        r"(?:together\s+)?(?:their\s+ages\s+)?(?:add\s+up\s+to|sum\s+to|total)\s+" + INT +
        r"\s*[.,]?\s*how\s+old\s+is\s+([A-Z][a-z]+)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    older, mult_word, younger = match.group(1), match.group(2).lower(), match.group(3)
    total = int(match.group(4))
    asked = match.group(5)
    factor = multiples[mult_word]
    if total % (factor + 1) != 0:
        return None
    younger_age = total // (factor + 1)
    if asked.lower() == younger.lower():
        return str(younger_age)
    if asked.lower() == older.lower():
        return str(younger_age * factor)
    return None


def _twice_as_many(text: str) -> str | None:
    # "There are twice as many cats as dogs. There are 5 dogs. How many cats?"
    multiples = {"twice": 2, "three times": 3, "four times": 4, "five times": 5}
    words = "|".join(re.escape(w) for w in multiples)
    match = re.fullmatch(
        r"\s*there\s+are\s+(" + words + r")\s+as\s+many\s+([\w]+)\s+as\s+([\w]+)\s*[.,]\s*"
        r"there\s+are\s+" + INT + r"\s+([\w]+)\s*[.,]?\s*how\s+many\s+([\w]+)(?:\s+are\s+there)?"
        + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    factor = multiples[match.group(1).lower()]
    many_label = match.group(2).lower()
    few_label = match.group(3).lower()
    count = int(match.group(4))
    count_label = match.group(5).lower()
    asked = match.group(6).lower()
    if count_label == few_label and asked == many_label:
        return str(count * factor)
    if count_label == many_label and asked == few_label:
        return str(count // factor) if count % factor == 0 else None
    return None


def _remaining_stock(text: str) -> str | None:
    # "A store had 120 phones and sold 45. How many are left?" — optional
    # restock clause: "then received 30 more".
    match = re.fullmatch(
        r"\s*(?:a|an|the)?\s*[\w ]{0,25}?(?:had|has|starts?\s+with|started\s+with)\s+" + INT +
        r"\s+([\w -]{1,25}?)\s*(?:,|and)?\s+(?:sold|sells|used|uses|gave\s+away|gives\s+away|"
        r"shipped|ships|lost|loses|removed|removes)\s+" + INT +
        r"(?:\s+(?:of\s+them|more))?\s*"
        r"(?:[,;]?\s*(?:and\s+)?(?:then\s+)?(?:received|receives|bought|buys|added|adds|restocked|"
        r"restocks|got|gets)\s+" + INT + r"(?:\s+more)?\s*)?"
        r"[.,]?\s*how\s+many\s+(?:[\w -]{1,25}?\s+)?(?:are\s+left|remain(?:ing)?|are\s+remaining|"
        r"does\s+(?:it|the\s+[\w]+)\s+have\s+(?:left|now|remaining))" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    start = int(match.group(1))
    sold = int(match.group(3))
    restocked = int(match.group(4)) if match.group(4) else 0
    result = start - sold + restocked
    return str(result) if result >= 0 else None


# --- probability -------------------------------------------------------------


def _coin_probability(text: str) -> str | None:
    # "What is the probability of getting heads on both of two fair coin flips?"
    # / "...heads 3 times in a row?"
    count_words = {"both": 2, "two": 2, "three": 3, "four": 4, "five": 5}
    match = re.fullmatch(
        r"\s*(?:what\s+is|whats|find|calculate)?\s*(?:the\s+)?probability\s+of\s+"
        r"(?:getting|flipping|tossing)\s+(heads?|tails?)\s+"
        r"(?:on\s+(?:both|all)\s+of\s+(two|three|four|five)\s+(?:fair\s+)?coin\s+(?:flips?|tosses?)"
        r"|(\d+|two|three|four|five)\s+times\s+in\s+a\s+row(?:\s+(?:with|on)\s+a\s+fair\s+coin)?)"
        + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    raw = (match.group(2) or match.group(3) or "").lower()
    n = count_words.get(raw) or (int(raw) if raw.isdigit() else None)
    if not n or n < 1 or n > 20:
        return None
    denominator = 2**n
    return f"1/{denominator} ({_fmt(round(1.0 / denominator, 9))})"


def _die_probability(text: str) -> str | None:
    # "What is the probability of rolling an even number on a standard
    # six-sided die?" / "...rolling a 6 on a fair die?"
    match = re.fullmatch(
        r"\s*(?:what\s+is|whats|find|calculate)?\s*(?:the\s+)?probability\s+of\s+rolling\s+"
        r"(?:an?\s+)?(even\s+number|odd\s+number|[1-6]|number\s+greater\s+than\s+[1-5]|"
        r"number\s+less\s+than\s+[2-6])\s+(?:on|with)\s+a\s+"
        r"(?:standard\s+|fair\s+|single\s+)*(?:six[- ]sided\s+)?(?:die|dice)" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    event = match.group(1).lower()
    if "even" in event or "odd" in event:
        favorable = 3
    elif "greater" in event:
        favorable = 6 - int(event[-1])
    elif "less" in event:
        favorable = int(event[-1]) - 1
    else:
        favorable = 1
    gcd = math.gcd(favorable, 6)
    numerator, denominator = favorable // gcd, 6 // gcd
    return f"{numerator}/{denominator} ({_fmt(round(numerator / denominator, 4))})"


# --- geometry ----------------------------------------------------------------


def _rectangle(text: str) -> str | None:
    # "A rectangle is 12 meters long and 8 meters wide. What is its area
    # (in square meters)?"
    match = re.fullmatch(
        r"\s*(?:a|the)?\s*rectangle(?:\s+is)?\s+" + NUM + r"\s*([a-z]+)?\s+long\s+and\s+"
        + NUM + r"\s*([a-z]+)?\s+wide\s*[.,]?\s*(?:what\s+is|whats|calculate|compute|find)\s+"
        r"(?:its|the)\s+(area|perimeter)(?:\s+in\s+(?:square\s+)?[a-z]+)?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    length, width = float(match.group(1)), float(match.group(3))
    unit_a = (match.group(2) or "").lower()
    unit_b = (match.group(4) or "").lower()
    if unit_a and unit_b and _UNIT_ALIASES.get(unit_a) != _UNIT_ALIASES.get(unit_b):
        return None
    asked = match.group(5).lower()
    value = length * width if asked == "area" else 2 * (length + width)
    return _fmt(round(value, 9))


def _square_shape(text: str) -> str | None:
    # "What is the perimeter of a square with side length 7 cm (in centimeters)?"
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?(area|perimeter)\s+of\s+a\s+square\s+with\s+"
        r"(?:a\s+)?sides?(?:\s+length)?(?:\s+of)?\s+" + NUM + r"\s*[a-z]*"
        r"(?:\s*,?\s+in\s+(?:square\s+)?[a-z]+)?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    asked = match.group(1).lower()
    side = float(match.group(2))
    value = side * side if asked == "area" else 4 * side
    return _fmt(round(value, 9))


def _circle(text: str) -> str | None:
    # "What is the area of a circle with radius 5? Round to one decimal place."
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?(area|circumference)\s+of\s+a\s+circle\s+with\s+"
        r"(?:a\s+)?(radius|diameter)(?:\s+of)?\s+" + NUM + r"\s*[a-z]*"
        r"\s*[.?!,]?\s*(round(?:ed)?\s+to\s+[\w ]+)?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    asked = match.group(1).lower()
    radius = float(match.group(3))
    if match.group(2).lower() == "diameter":
        radius /= 2.0
    value = math.pi * radius * radius if asked == "area" else 2 * math.pi * radius
    decimals = _rounding_from(text)
    if decimals is None:
        decimals = 2
    return _fmt(value, decimals)


def _polygon_angles(text: str) -> str | None:
    # "What is the sum of the interior angles of a triangle in degrees?"
    shapes = {
        "triangle": 180, "quadrilateral": 360, "rectangle": 360, "square": 360,
        "pentagon": 540, "hexagon": 720, "heptagon": 900, "octagon": 1080,
    }
    match = re.fullmatch(
        Q_PREFIX + r"(?:the\s+)?sum\s+of\s+(?:the\s+)?interior\s+angles\s+"
        r"(?:of|in)\s+a\s+([a-z]+)(?:\s+in\s+degrees)?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    shape = match.group(1).lower()
    if shape not in shapes:
        return None
    return f"{shapes[shape]} degrees"


# --- deterministic logic -----------------------------------------------------


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")


def _weekday_arithmetic(text: str) -> str | None:
    # "Today is Wednesday. What day of the week will it be in exactly 100 days?"
    match = re.fullmatch(
        r"\s*(?:if\s+)?today\s+is\s+([A-Za-z]+)\s*[.,]?\s*what\s+day(?:\s+of\s+the\s+week)?\s+"
        r"(?:will\s+it\s+be|is\s+it)\s+in\s+(?:exactly\s+)?" + INT + r"\s+days?" + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    day = match.group(1).lower()
    if day not in _WEEKDAYS:
        return None
    offset = int(match.group(2))
    return _WEEKDAYS[(_WEEKDAYS.index(day) + offset) % 7].capitalize()


_COMPARATIVE_AXES = {
    "taller": ("tallest", "shortest"),
    "older": ("oldest", "youngest"),
    "faster": ("fastest", "slowest"),
    "stronger": ("strongest", "weakest"),
    "heavier": ("heaviest", "lightest"),
    "bigger": ("biggest", "smallest"),
    "larger": ("largest", "smallest"),
    "richer": ("richest", "poorest"),
}


def _transitive_chain(text: str) -> str | None:
    # "Alice is taller than Bob. Bob is taller than Carol. Who is the
    # shortest? (Answer with just the name.)"
    comparatives = "|".join(_COMPARATIVE_AXES)
    match = re.fullmatch(
        r"\s*([A-Z][a-z]+)\s+is\s+(" + comparatives + r")\s+than\s+([A-Z][a-z]+)\s*[.,;]\s*"
        r"([A-Z][a-z]+)\s+is\s+(" + comparatives + r")\s+than\s+([A-Z][a-z]+)\s*[.,;]?\s*"
        r"(?:deduce\s+)?who\s+is\s+(?:the\s+)?([a-z]+)\s*\??\s*"
        r"(?:answer\s+with\s+just\s+the\s+name\.?|answer\s+with\s+the\s+name\.?|"
        r"give\s+just\s+the\s+name\.?)?\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    a, comp1, b = match.group(1), match.group(2).lower(), match.group(3)
    c, comp2, d = match.group(4), match.group(5).lower(), match.group(6)
    asked = match.group(7).lower()
    if comp1 != comp2 or comp1 not in _COMPARATIVE_AXES:
        return None
    # Chain must link: A > B, B > C.
    if b.lower() != c.lower():
        return None
    top_word, bottom_word = _COMPARATIVE_AXES[comp1]
    if asked == top_word:
        return a
    if asked == bottom_word:
        return d
    return None


def _preference_chain(text: str) -> str | None:
    # "Ann likes tea more than coffee, and coffee more than juice. Which of
    # the three drinks does she like least?"
    match = re.fullmatch(
        r"\s*([A-Z][a-z]+)\s+likes\s+([\w ]{1,15}?)\s+more\s+than\s+([\w ]{1,15}?)\s*,?\s*"
        r"and\s+([\w ]{1,15}?)\s+more\s+than\s+([\w ]{1,15}?)\s*[.,]\s*"
        r"which\s+(?:of\s+the\s+[\w ]{1,20}\s+)?does\s+(?:she|he|they)\s+like\s+(least|most|best)"
        + TAIL,
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    first = match.group(2).strip().lower()
    second = match.group(3).strip().lower()
    second_again = match.group(4).strip().lower()
    third = match.group(5).strip().lower()
    asked = match.group(6).lower()
    if second != second_again:
        return None
    if asked == "least":
        return third.capitalize()
    return first.capitalize()


_SOLVERS = (
    _percent_change,
    _increase_decrease_by_percent,
    _discount,
    _tip_or_tax_amount,
    _total_with_tax,
    _simple_interest,
    _fraction_of,
    _speed_from_distance_time,
    _distance_from_speed_time,
    _proportional_cost,
    _recipe_scaling,
    _rate_per_100,
    _inverse_work_rate,
    _unit_conversion,
    _gcd_lcm,
    _median_or_range,
    _next_in_sequence,
    _square_root,
    _ratio_split,
    _age_multiple_sum,
    _twice_as_many,
    _remaining_stock,
    _coin_probability,
    _die_probability,
    _rectangle,
    _square_shape,
    _circle,
    _polygon_angles,
    _weekday_arithmetic,
    _transitive_chain,
    _preference_chain,
)

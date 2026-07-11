"""Zero-token deterministic solvers for trivially verifiable prompts.

Every solver here must be effectively infallible: it only fires when the
entire prompt reduces to a single well-formed arithmetic question, so a hit
means the answer is exact. Anything ambiguous (word problems, units, story
context) returns None and flows to the LLM instead. A wrong deterministic
answer is unrecoverable, so precision is prioritized over coverage.
"""

from __future__ import annotations

import ast
import operator
import re


# Filler that may surround the actual expression in an arithmetic prompt.
# The remainder after stripping these must be ONLY an expression; any other
# word (e.g. "dollars", "apples", "then") disqualifies the prompt.
_FILLER_PATTERN = re.compile(
    r"\b(?:what|whats|is|the|value|of|result|calculate|compute|evaluate|find|"
    r"solve|answer|give|me|please|equal|equals|to|following|expression|"
    r"arithmetic|exactly|tell)\b",
    re.IGNORECASE,
)

_WORD_OPERATORS = (
    (re.compile(r"\bmultiplied\s+by\b|\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided\s+by\b|\bover\b", re.IGNORECASE), "/"),
    (re.compile(r"\bplus\b|\badded\s+to\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b|\bsubtract(?:ed)?\b|\bless\b", re.IGNORECASE), "-"),
    (re.compile(r"\bto\s+the\s+power\s+of\b|\braised\s+to\b", re.IGNORECASE), "^"),
    (re.compile(r"[×x]", re.IGNORECASE), "*"),
    (re.compile(r"[÷]"), "/"),
)

# Only these node types may appear in a candidate expression.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Guard against pathological inputs like 9**9**9 hanging the process.
_MAX_EXPONENT = 64
_MAX_OPERAND = 1e12

_PERCENT_OF_PATTERN = re.compile(
    r"^\s*(?:what\s+is|whats|calculate|compute|find|evaluate)?\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_AVERAGE_PATTERN = re.compile(
    r"^\s*(?:what\s+is|whats|calculate|compute|find|evaluate)?\s*(?:the\s+)?"
    r"(average|mean)\s+of\s+(?:these\s+numbers\s*:?\s*|the\s+numbers\s*:?\s*"
    r"|the\s+following(?:\s+numbers)?\s*:?\s*)?"
    r"((?:-?\d+(?:\.\d+)?)(?:\s*(?:,\s*and|,|and)\s*-?\d+(?:\.\d+)?)+)\s*[?.!]?\s*$",
    re.IGNORECASE,
)

# Single-variable linear equation: [coef]VAR [± const] = rhs. Exactly one
# variable letter may appear; anything else (second variable, powers, story
# context) fails the anchored match and flows to the LLM.
_EQUATION_CORE = (
    r"(-?\d+(?:\.\d+)?)?\s*\*?\s*([a-z])\s*"
    r"(?:([+-])\s*(\d+(?:\.\d+)?))?\s*=\s*(-?\d+(?:\.\d+)?)"
)

_EQUATION_SOLVE_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:solve|find|determine|calculate)?\s*"
    r"(?:for\s+([a-z])\s*)?[:,]?\s*(?:the\s+)?(?:linear\s+)?(?:equation\s*)?"
    r"[:,]?\s*" + _EQUATION_CORE +
    r"\s*(?:for\s+(?:the\s+value\s+of\s+)?([a-z]))?\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_EQUATION_WHATIS_PATTERN = re.compile(
    r"^\s*(?:what\s+is|find)\s+([a-z])\s+(?:if|when|given(?:\s+that)?)\s+"
    + _EQUATION_CORE + r"\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_EQUATION_IF_PATTERN = re.compile(
    r"^\s*if\s+" + _EQUATION_CORE +
    r"\s*,?\s*(?:then\s+)?what\s+is\s+([a-z])\s*[?.!]?\s*$",
    re.IGNORECASE,
)

# Temperature conversion between Fahrenheit, Celsius, and Kelvin: the
# formulas are exact, so a matched prompt is always answerable.
_TEMPERATURE_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:convert\s+|what\s+is\s+)?(-?\d+(?:\.\d+)?)\s*"
    r"(?:°\s*|degrees?\s+)?(fahrenheit|celsius|centigrade|kelvin|[fck])\s+"
    r"(?:to|into|in)\s+(?:°\s*|degrees?\s+)?(fahrenheit|celsius|centigrade|kelvin|[fck])"
    r"\s*(?:\([^)]{0,60}\))?\s*[?.!]?\s*$",
    re.IGNORECASE,
)

_TEMPERATURE_UNITS = {
    "f": "F", "fahrenheit": "F",
    "c": "C", "celsius": "C", "centigrade": "C",
    "k": "K", "kelvin": "K",
}


def try_solve(prompt: str) -> str | None:
    """Return an exact answer string, or None when not provably safe."""

    text = prompt.strip()
    if not text or len(text) > 300:
        return None

    for solver in (
        _solve_percent_of,
        _solve_average,
        _solve_temperature,
        _solve_linear_equation,
        _solve_arithmetic,
    ):
        answer = solver(text)
        if answer is not None:
            return answer
    return None


def can_solve(prompt: str) -> bool:
    return try_solve(prompt) is not None


def _solve_percent_of(text: str) -> str | None:
    match = _PERCENT_OF_PATTERN.match(text)
    if not match:
        return None
    percent, base = float(match.group(1)), float(match.group(2))
    return _format_number(percent * base / 100.0)


def _solve_average(text: str) -> str | None:
    match = _AVERAGE_PATTERN.match(text)
    if not match:
        return None
    numbers = [
        float(part)
        for part in re.findall(r"-?\d+(?:\.\d+)?", match.group(2))
    ]
    if not numbers:
        return None
    return _format_number(sum(numbers) / len(numbers))


def _solve_linear_equation(text: str) -> str | None:
    asked_prefix = asked_suffix = None
    match = _EQUATION_SOLVE_PATTERN.match(text)
    if match:
        asked_prefix, coef_s, var, sign, const_s, rhs_s, asked_suffix = match.groups()
    else:
        match = _EQUATION_WHATIS_PATTERN.match(text)
        if match:
            asked_prefix, coef_s, var, sign, const_s, rhs_s = match.groups()
        else:
            match = _EQUATION_IF_PATTERN.match(text)
            if not match:
                return None
            coef_s, var, sign, const_s, rhs_s, asked_suffix = match.groups()

    # A bare "x = 5" has nothing to solve; require an actual operation.
    if coef_s is None and const_s is None:
        return None

    var = var.lower()
    for asked in (asked_prefix, asked_suffix):
        if asked and asked.lower() != var:
            return None  # asked about a different symbol than the equation uses

    coefficient = float(coef_s) if coef_s else 1.0
    if coefficient == 0:
        return None
    constant = float(const_s) if const_s else 0.0
    if sign == "-":
        constant = -constant
    value = (float(rhs_s) - constant) / coefficient
    return f"{var} = {_format_number(value)}"


def _solve_temperature(text: str) -> str | None:
    match = _TEMPERATURE_PATTERN.match(text)
    if not match:
        return None

    value = float(match.group(1))
    source = _TEMPERATURE_UNITS[match.group(2).lower()]
    target = _TEMPERATURE_UNITS[match.group(3).lower()]
    if source == target:
        return None

    celsius = {
        "F": (value - 32.0) * 5.0 / 9.0,
        "K": value - 273.15,
        "C": value,
    }[source]
    converted = {
        "C": celsius,
        "F": celsius * 9.0 / 5.0 + 32.0,
        "K": celsius + 273.15,
    }[target]
    suffix = {"C": "°C", "F": "°F", "K": "K"}[target]
    formatted = f"{converted:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}{suffix}"


def _solve_arithmetic(text: str) -> str | None:
    candidate = text
    for pattern, symbol in _WORD_OPERATORS:
        candidate = pattern.sub(f" {symbol} ", candidate)
    candidate = _FILLER_PATTERN.sub(" ", candidate)
    candidate = candidate.replace("=", " ").replace("?", " ")
    candidate = re.sub(r"[,.!:;]\s*$", " ", candidate)
    # Thousands separators inside numbers ("1,000") are ambiguous with
    # argument lists; refuse rather than guess.
    if re.search(r"\d,\d", candidate):
        return None
    candidate = candidate.replace(",", " ")

    # After stripping filler, only an arithmetic expression may remain.
    if not re.fullmatch(r"[\d\s()+\-*/^.%]+", candidate):
        return None
    if not re.search(r"\d\s*[+\-*/^%]|\)\s*[+\-*/^%]", candidate):
        return None  # a lone number is not a question worth answering

    expression = re.sub(r"\^", "**", candidate).strip()
    # A trailing % is "percent", not modulo; that's not plain arithmetic.
    if expression.endswith("%"):
        return None

    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return _format_number(value)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        value = float(node.value)
        if abs(value) > _MAX_OPERAND:
            raise ValueError("operand too large")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (
            abs(right) > _MAX_EXPONENT or abs(left) > 1e6
        ):
            raise ValueError("exponent too large")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        if abs(result) > 1e15:
            raise ValueError("result too large")
        return result
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted

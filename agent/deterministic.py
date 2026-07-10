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


def try_solve(prompt: str) -> str | None:
    """Return an exact answer string, or None when not provably safe."""

    text = prompt.strip()
    if not text or len(text) > 300:
        return None

    for solver in (_solve_percent_of, _solve_average, _solve_arithmetic):
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

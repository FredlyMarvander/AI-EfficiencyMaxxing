"""Shared answer-checking heuristics for the local eval harnesses.

The real judge is an LLM and stricter; these checks only need to be tight
enough that mode-to-mode comparisons are meaningful. Three matching rules:

- numeric terms match any number token in the output with exact value
  ("36" matches "= 36.0" but not "360");
- plain alphanumeric terms match left-word-boundary-anchored ("favorable"
  does not match inside "unfavorable", "valid" does not match inside
  "invalid" -- both lack a boundary right before the term -- but "recycl"
  still matches "recycling" and "garden" still matches "gardens", since
  intentional word-stem terms need the suffix left open);
- anything containing symbols falls back to substring ("% 2 == 0", "1/4").
"""

from __future__ import annotations

import re
import unicodedata


def looks_correct(case: dict, output: str) -> bool:
    # NFKC folds compatibility variants to their plain form (H₂O -> H2O,
    # fullwidth digits, etc.) so formatting choices the real judge would
    # accept don't fail this cheaper heuristic.
    normalized = unicodedata.normalize("NFKC", output or "")
    text = " ".join(normalized.split()).lower()
    if not text:
        return False

    expected_all = case.get("expected_all")
    if expected_all:
        return all(_match_term(str(term), text) for term in expected_all)

    expected_any = case.get("expected_any")
    if expected_any:
        return any(_match_term(str(term), text) for term in expected_any)

    expected = str(case.get("expected", "") or "").strip()
    if expected:
        return _match_term(expected, text)
    return len(text) > 20


def _match_term(term: str, text: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False

    if _is_number(term):
        target = float(term)
        for token in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
            try:
                value = float(token.replace(",", ""))
            except ValueError:
                continue
            if abs(value - target) < 1e-9:
                return True
        return False

    if re.fullmatch(r"[a-z0-9 ]+", term):
        # Left-anchored only (no trailing \b): blocks "unfavorable" matching
        # "favorable", but still lets stem terms match their inflections.
        pattern = r"\b" + r"\s+".join(re.escape(word) for word in term.split())
        return re.search(pattern, text) is not None

    return term in text


def _is_number(term: str) -> bool:
    try:
        float(term)
    except ValueError:
        return False
    return True

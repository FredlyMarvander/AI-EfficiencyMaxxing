"""Deterministic validation of local-model answers.

A local answer is returned to the judge only if it passes these checks;
anything rejected escalates to Fireworks. Rejecting a good answer costs a
few hundred tokens; accepting a bad answer costs the accuracy gate, so every
rule errs toward rejection.
"""

from __future__ import annotations

import ast
import re

from agent.router import TaskKind


_STOPWORD_PREFIXES = {
    "the", "a", "an", "in", "on", "at", "after", "before", "during", "dr",
    "mr", "mrs", "ms", "prof",
}

# Capitalized words a well-formed NER answer may legitimately introduce:
# entity-type labels and scaffolding, never entity names themselves.
_NER_LABEL_WORDS = {
    "people", "person", "persons", "organization", "organizations",
    "organisation", "organisations", "location", "locations", "place",
    "places", "date", "dates", "misc", "miscellaneous", "entity", "entities",
    "company", "companies", "city", "cities", "name", "names", "named",
    "per", "org", "loc", "gpe", "time", "event", "events", "none", "json",
    "here", "note", "extracted", "output", "answer", "type", "types",
    "object", "objects", "artwork", "painting", "monument", "landmark",
    "building", "buildings", "product", "products", "ship", "prize",
    "award", "title", "work", "country", "countries", "facility",
}


def validate_local_answer(kind: TaskKind, prompt: str, answer: str) -> bool:
    answer = (answer or "").strip()
    if not answer or len(answer) > 4000:
        return False
    if re.search(r"\bas an ai\b|\bi cannot\b|\bi'm sorry\b", answer.lower()):
        return False

    if kind is TaskKind.SENTIMENT:
        return _validate_sentiment(prompt, answer)
    if kind is TaskKind.NER:
        return _validate_ner(prompt, answer)
    if kind is TaskKind.SUMMARY:
        return _validate_summary(prompt, answer)
    if kind is TaskKind.FACTUAL:
        return _validate_factual(answer)
    if kind is TaskKind.CODE:
        return _validate_code(prompt, answer)
    return False  # unknown kind: never trust the local model blindly


def _validate_sentiment(prompt: str, answer: str) -> bool:
    lowered_prompt = prompt.lower()
    if "favorable" in lowered_prompt or "favourable" in lowered_prompt:
        labels = ("favorable", "unfavorable", "favourable", "unfavourable")
    elif re.search(r"\bhappy\b.{0,15}\bunhappy\b", lowered_prompt):
        labels = ("happy", "unhappy")
    else:
        labels = ("positive", "negative", "neutral")

    lowered = answer.lower()
    hits = {
        label for label in labels if re.search(rf"(?<![a-z]){label}(?![a-z])", lowered)
    }
    # "unfavorable" contains no bare "favorable" under the lookarounds above,
    # but count base/negated pairs as distinct labels either way.
    if len(hits) != 1:
        return False
    wants_reason = bool(re.search(r"\breason|\bexplain|\bwhy\b", lowered_prompt))
    if not wants_reason and len(answer) > 120:
        return False
    return True


def _extract_ner_source(prompt: str) -> str | None:
    quoted = re.findall(r"[\"'‘’“”](.{15,}?)[\"'‘’“”]", prompt, re.S)
    if quoted:
        return max(quoted, key=len)
    head, sep, tail = prompt.partition(":")
    if sep and len(tail.strip()) >= 15 and re.search(
        r"\b(extract|list|identify|find|pull|return|name|entities|who)\b",
        head.lower(),
    ):
        return tail.strip()
    return None


def _capitalized_runs(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][\w'’.-]*|\S", text)
    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if re.match(r"^[A-Z]", token):
            current.append(token)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    cleaned: list[str] = []
    for run in runs:
        while run and run[0].lower().rstrip(".") in _STOPWORD_PREFIXES:
            run = run[1:]
        if run:
            cleaned.append(" ".join(run).rstrip(".,;"))
    return cleaned


def _validate_ner(prompt: str, answer: str) -> bool:
    source = _extract_ner_source(prompt)
    if source is None:
        return False  # cannot locate the text to check against: escalate

    normalized_answer = " ".join(answer.split()).lower()
    normalized_source = " ".join(source.split()).lower()

    # Completeness: every capitalized run and year in the source must appear.
    # Possessives count as the bare name ("Google's" is covered by "Google").
    for entity in _capitalized_runs(source):
        needle = entity.lower()
        bare = re.sub(r"['’]s\b", "", needle)
        if needle not in normalized_answer and bare not in normalized_answer:
            return False
    for year in re.findall(r"\b(?:1[5-9]\d\d|20\d\d)\b", source):
        if year not in normalized_answer:
            return False

    # Anti-hallucination: capitalized tokens in the answer must come from the
    # source (or be known label words).
    for token in re.findall(r"\b[A-Z][\w'’-]*", answer):
        lowered = token.lower().rstrip(".,;:")
        if lowered in _NER_LABEL_WORDS or lowered in _STOPWORD_PREFIXES:
            continue
        if lowered not in normalized_source:
            return False
    return True


def _validate_summary(prompt: str, answer: str) -> bool:
    words = len(answer.split())
    if words < 3:
        return False

    lowered_prompt = prompt.lower()
    limit = re.search(r"\b(?:in|within|under|at most|no more than)\s+(\d+)\s+words", lowered_prompt)
    if limit and words > int(limit.group(1)) * 1.2 + 2:
        return False

    sentences_asked = re.search(
        r"\bin (one|two|three|a single|1|2|3)(?:\s+or\s+(?:two|three|fewer))?\s+sentences?\b",
        lowered_prompt,
    )
    if sentences_asked:
        counts = {"one": 1, "a single": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3}
        allowed = counts[sentences_asked.group(1)]
        if lowered_prompt.count(" or two") or " or three" in lowered_prompt:
            allowed += 1
        sentences = len([s for s in re.split(r"[.!?]+", answer) if s.strip()])
        if sentences > allowed + 1:
            return False

    # A summary should be meaningfully shorter than its source.
    if words > max(120, len(prompt.split())):
        return False
    return True


def _validate_factual(answer: str) -> bool:
    if len(answer) > 400:
        return False
    return True


def _required_code_names(prompt: str) -> list[str]:
    names = re.findall(
        r"\b(?:function|method|class)\s+(?:called|named)\s+([A-Za-z_]\w*)", prompt
    )
    names += re.findall(r"\bcalled\s+([A-Za-z_]\w*)\s*\(", prompt)
    return list(dict.fromkeys(names))


def _extract_code_block(answer: str) -> str:
    fenced = re.findall(r"```[a-zA-Z]*\n(.*?)```", answer, re.S)
    if fenced:
        return "\n".join(fenced)
    return answer


def _validate_code(prompt: str, answer: str) -> bool:
    code = _extract_code_block(answer).strip()
    if not code:
        return False

    lowered_prompt = prompt.lower()
    names = _required_code_names(prompt)

    if "sql" in lowered_prompt:
        return bool(re.search(r"\bselect\b.*\bfrom\b", code, re.I | re.S))

    if "javascript" in lowered_prompt or "typescript" in lowered_prompt:
        for name in names:
            if not re.search(
                rf"\bfunction\s+{re.escape(name)}\b|\b(?:const|let|var)\s+{re.escape(name)}\s*=",
                code,
            ):
                return False
        return bool(names) or "function" in code or "=>" in code

    # Default: treat as Python. It must parse, and every required name must
    # be defined as a function or class.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for name in names:
        if name not in defined:
            return False
    if names:
        return True
    # No required name (e.g. "create a program that ..."): accept only if the
    # code actually does something.
    return any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Expr, ast.For,
                          ast.While, ast.Assign, ast.If))
        for node in tree.body
    )

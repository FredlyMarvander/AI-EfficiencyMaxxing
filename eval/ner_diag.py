"""One-off: explain the NER validator's verdict for each probed answer."""

from __future__ import annotations

import json
import os
import re

from agent.validate import (
    _capitalized_runs,
    _extract_ner_source,
    _NER_LABEL_WORDS,
    _STOPWORD_PREFIXES,
    validate_local_answer,
)
from agent.router import TaskKind

here = os.path.dirname(__file__)
with open(os.path.join(here, "test_cases_v2.json"), encoding="utf-8") as handle:
    cases = {c["id"]: c for c in json.load(handle) if c["category"] == "named_entity_recognition"}
with open(os.path.join(here, "reports", "local_probe_ner.json"), encoding="utf-8") as handle:
    probe = json.load(handle)["results"]

for row in probe:
    case = cases[row["id"]]
    prompt, answer = str(case["prompt"]), row["answer"]
    accepted = validate_local_answer(TaskKind.NER, prompt, answer)
    verdict = "ACCEPT" if accepted else "reject"
    reasons = []
    source = _extract_ner_source(prompt)
    if source is None:
        reasons.append("no source found")
    else:
        norm_answer = " ".join(answer.split()).lower()
        norm_source = " ".join(source.split()).lower()
        for entity in _capitalized_runs(source):
            needle = entity.lower()
            bare = re.sub(r"['’]s\b", "", needle)
            if needle not in norm_answer and bare not in norm_answer:
                reasons.append(f"missing:{entity}")
        for year in re.findall(r"\b(?:1[5-9]\d\d|20\d\d)\b", source):
            if year not in norm_answer:
                reasons.append(f"missing-year:{year}")
        for token in re.findall(r"\b[A-Z][\w'’-]*", answer):
            lowered = token.lower().rstrip(".,;:")
            if lowered in _NER_LABEL_WORDS or lowered in _STOPWORD_PREFIXES:
                continue
            if lowered not in norm_source:
                reasons.append(f"hallucinated:{token}")
    print(f"{row['id']} ok={row['ok']} {verdict} {'; '.join(reasons[:4])}")
    if not accepted:
        print(f"   answer: {' '.join(answer.split())[:150]}")

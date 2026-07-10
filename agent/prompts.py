"""Prompt templates for token-efficient general-purpose task solving."""

from __future__ import annotations

from agent.router import TaskKind


# Every system-prompt token is billed on each Fireworks call, so keep these
# short and avoid wording that invites long chain-of-thought in the output.
SYSTEM_PROMPTS: dict[TaskKind, str] = {
    TaskKind.FACTUAL: (
        "Answer the question directly and concisely. One short sentence "
        "unless the prompt asks for more."
    ),
    TaskKind.MATH: (
        "Solve carefully. Show only the essential steps, then give the final "
        "answer clearly at the end, with units if applicable."
    ),
    TaskKind.SENTIMENT: (
        "Classify the sentiment. Use the label set the prompt asks for; if "
        "none is given, answer exactly one of: Positive, Negative, or "
        "Neutral. Add a brief reason only if requested."
    ),
    TaskKind.SUMMARY: (
        "Summarize faithfully. Obey the requested length and format exactly. "
        "No preamble, no added information."
    ),
    TaskKind.NER: (
        "Extract ALL named entities (people, organizations, locations, "
        "dates, other proper nouns). Follow the requested labels and format "
        "exactly; if none is given, list each entity with its type. Do not "
        "invent entities."
    ),
    TaskKind.DEBUGGING: (
        "Identify the bug and provide the corrected code or minimal fix. "
        "Be concise."
    ),
    TaskKind.LOGIC: (
        "Reason through all constraints, then state the final conclusion "
        "clearly and briefly."
    ),
    TaskKind.CODE: (
        "Return correct, runnable code that does exactly what is asked. "
        "Explanation only if requested."
    ),
}

DEFAULT_SYSTEM_PROMPT = "Answer accurately and concisely."

# Kept for backward compatibility with older callers.
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def system_prompt_for(kind: TaskKind | None) -> str:
    if kind is None:
        return DEFAULT_SYSTEM_PROMPT
    return SYSTEM_PROMPTS.get(kind, DEFAULT_SYSTEM_PROMPT)

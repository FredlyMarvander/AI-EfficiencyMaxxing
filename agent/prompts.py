"""Prompt templates for token-efficient general-purpose task solving."""

from __future__ import annotations

from agent.router import TaskKind


# Every system-prompt token is billed on each Fireworks call, so keep these
# minimal: answer-only phrasing, no step-by-step invitations.
SYSTEM_PROMPTS: dict[TaskKind, str] = {
    TaskKind.FACTUAL: "Answer directly in one short sentence.",
    TaskKind.MATH: (
        "Solve. End with only the final answer, with units if applicable."
    ),
    TaskKind.SENTIMENT: (
        "Reply with one label: Positive, Negative, or Neutral — or the label "
        "set the prompt requests. No reason unless asked."
    ),
    TaskKind.SUMMARY: (
        "Summarize faithfully. Match the requested length and format. No "
        "preamble."
    ),
    TaskKind.NER: (
        "Extract every named entity (people, organizations, locations, "
        "dates). Use the requested format. Do not invent entities."
    ),
    TaskKind.DEBUGGING: (
        "State the bug and give the minimal corrected code. Be brief."
    ),
    TaskKind.LOGIC: "State the conclusion briefly.",
    TaskKind.CODE: "Return correct runnable code only, unless asked to explain.",
}

DEFAULT_SYSTEM_PROMPT = "Answer accurately and concisely."

# Kept for backward compatibility with older callers.
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def system_prompt_for(kind: TaskKind | None) -> str:
    if kind is None:
        return DEFAULT_SYSTEM_PROMPT
    return SYSTEM_PROMPTS.get(kind, DEFAULT_SYSTEM_PROMPT)

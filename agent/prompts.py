"""Prompt templates for token-efficient general-purpose task solving."""

# Every system-prompt token is billed on each Fireworks call, so keep this
# short and avoid wording that invites long chain-of-thought in the output.
SYSTEM_PROMPT = "Answer accurately and concisely in English. No filler."

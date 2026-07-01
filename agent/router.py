from agent.config import QUALITY_THRESHOLD
from agent.local_model import call_local_model
from agent.fireworks import call_fireworks

# Signals that a task is cheap/easy enough for the local model.
SIMPLE_KEYWORDS = (
    "translate", "define", "what is", "what are", "convert",
    "calculate", "list", "spell", "count", "when was",
    "who is", "how many", "capital of",
)

# Signals that a task is hard enough to be worth a remote call.
COMPLEX_KEYWORDS = (
    "analyze", "compare", "explain why", "write an essay",
    "summarize", "evaluate", "argue", "critique", "design",
    "step by step", "reason",
)


# ── Layer 1: rule-based (free) ───────────────────────────────────────
def rule_based_route(task: str):
    """
    Return "local", "remote", or None (None = not sure, let Layer 2 decide).
    Costs zero tokens.
    """
    text = task.lower()
    words = len(task.split())

    looks_simple = any(k in text for k in SIMPLE_KEYWORDS) and words < 15
    looks_complex = any(k in text for k in COMPLEX_KEYWORDS) or words > 60

    if looks_complex:
        return "remote"
    if looks_simple:
        return "local"
    return None


# ── Quality check for Layer 2 ────────────────────────────────────────
def quality_score(output: str) -> float:
    """
    A cheap 0..1 heuristic for "does this local answer look usable?"
    No ML, no extra tokens — just red flags. Refine after you see real
    tasks on launch day.
    """
    text = (output or "").strip().lower()
    if len(text) < 5:
        return 0.0

    red_flags = (
        "i don't know", "i'm not sure", "i am not sure",
        "i cannot", "as an ai", "i'm unable",
    )
    penalty = sum(0.3 for flag in red_flags if flag in text)
    return max(0.0, 1.0 - penalty)


# ── Main entry point ─────────────────────────────────────────────────
def run_agent(task: str) -> dict:
    """
    Route one task and return the answer plus full token accounting.
    """
    decision = rule_based_route(task)

    # Clear-cut remote
    if decision == "remote":
        return _format(call_fireworks(task), "rule:complex", escalated=False)

    # Clear-cut local
    if decision == "local":
        return _format(call_local_model(task), "rule:simple", escalated=False)

    # Ambiguous: try local first, escalate only if it looks weak.
    local = call_local_model(task)
    if quality_score(local["output"]) >= QUALITY_THRESHOLD:
        return _format(local, "ambiguous:local-ok", escalated=False)

    # Local answer looked weak -> spend the remote call.
    # We still "paid" the local tokens, so count BOTH for honest scoring.
    remote = call_fireworks(task)
    remote["input_tokens"] += local["input_tokens"]
    remote["output_tokens"] += local["output_tokens"]
    remote["total_tokens"] += local["total_tokens"]
    return _format(remote, "ambiguous:escalated", escalated=True)


def _format(result: dict, reason: str, escalated: bool) -> dict:
    return {
        "output": result["output"],
        "model_used": result["model"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"],
        "route_reason": reason,
        "escalated": escalated,
    }

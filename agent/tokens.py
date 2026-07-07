"""Tiny token estimate fallback without optional third-party dependencies."""


def estimate_tokens(text: str) -> int:
    """Approximate tokens as one token per four characters."""

    return max(1, (len(text or "") + 3) // 4)

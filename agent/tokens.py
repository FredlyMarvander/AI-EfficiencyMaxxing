import tiktoken

# cl100k_base is a reasonable general-purpose tokenizer for estimates.
_ENCODER = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Only used when no exact count is available."""
    return len(_ENCODER.encode(text or ""))

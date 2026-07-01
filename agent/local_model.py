import ollama

from agent.config import OLLAMA_HOST, LOCAL_MODEL, MAX_TOKENS
from agent.tokens import estimate_tokens

# A client pointed at our Ollama server (localhost in dev, the
# "ollama" container in docker-compose).
_client = ollama.Client(host=OLLAMA_HOST)


def _get(resp, key, default=None):
    """Read a field whether resp is a plain dict or an ollama object."""
    if isinstance(resp, dict):
        return resp.get(key, default)
    return getattr(resp, key, default)


def call_local_model(task: str, model: str = LOCAL_MODEL) -> dict:
    """
    Run the task on a small local model.

    Returns the SAME shape as call_fireworks. Ollama usually hands back
    exact token counts, so we use them; if a version doesn't, we fall
    back to a tiktoken estimate so nothing crashes.
    """
    response = _client.chat(
        model=model,
        messages=[{"role": "user", "content": task}],
        options={"num_predict": MAX_TOKENS},
    )

    message = _get(response, "message", {}) or {}
    content = (_get(message, "content", "") or "").strip()

    input_tokens = _get(response, "prompt_eval_count")
    output_tokens = _get(response, "eval_count")
    if input_tokens is None:
        input_tokens = estimate_tokens(task)
    if output_tokens is None:
        output_tokens = estimate_tokens(content)

    return {
        "output": content,
        "model": "local",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens),
    }

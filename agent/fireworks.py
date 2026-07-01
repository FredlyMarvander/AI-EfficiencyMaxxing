"""Remote model via Fireworks AI official SDK."""
from fireworks.client import Fireworks

from agent.config import REMOTE_MODEL, MAX_TOKENS

# Client otomatis baca FIREWORKS_API_KEY dari environment
_client = Fireworks()


def call_fireworks(task: str, model: str = REMOTE_MODEL) -> dict:
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": task}],
        max_tokens=MAX_TOKENS,
    )

    usage = response.usage
    return {
        "output": response.choices[0].message.content.strip(),
        "model": "remote",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
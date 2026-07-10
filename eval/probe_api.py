"""One-shot raw API probe: dump the full response JSON for inspection."""

from __future__ import annotations

import json
import os
import sys
import time

import requests

base = os.environ["FIREWORKS_BASE_URL"].rstrip("/")
model = os.environ["ALLOWED_MODELS"].split(",")[0].strip()
payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": (
                "Classify the sentiment. Answer exactly one of: Positive, "
                "Negative, or Neutral."
            ),
        },
        {
            "role": "user",
            "content": "Classify: 'Best purchase I have made all year!'",
        },
    ],
    "temperature": 0,
    "max_tokens": 2000,
}
response = None
for attempt in range(10):
    response = requests.post(
        f"{base}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
        timeout=60,
    )
    print("status:", response.status_code, flush=True)
    if response.status_code < 400:
        break
    time.sleep(25)
else:
    sys.exit(1)
data = response.json()
choice = (data.get("choices") or [{}])[0]
message = choice.get("message", {})
print("finish_reason:", choice.get("finish_reason"))
print("message keys:", sorted(message.keys()))
print("usage:", json.dumps(data.get("usage", {})))
for key, value in message.items():
    if isinstance(value, str):
        print(f"--- {key} ({len(value)} chars) ---")
        print(value[:1500])

"""Verify which ALLOWED_MODELS are actually accessible with the current key.

One minimal request per model. 429s are retried by the client (accessible,
just throttled); 403/404 and similar mark the model unusable.

  python -m eval.model_check
"""

from __future__ import annotations

import json
import os
import sys
import time

from agent.config import load_settings
from agent.fireworks import FireworksAPIError, FireworksClient


def main() -> int:
    settings = load_settings()
    client = FireworksClient(settings)

    rows = []
    for model in settings.allowed_models:
        started = time.monotonic()
        try:
            answer, usage = client.complete_with_usage(
                prompt="Reply with the single word: ok",
                model=model,
                max_tokens=256,  # reasoning models need headroom even here
                timeout_seconds=30,
            )
            rows.append(
                {
                    "model": model,
                    "accessible": True,
                    "error": "",
                    "sample": answer[:40],
                    "tokens": int(usage["total_tokens"]),
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
        except FireworksAPIError as exc:
            rows.append(
                {
                    "model": model,
                    "accessible": False,
                    "error": str(exc)[:160],
                    "sample": "",
                    "tokens": 0,
                    "seconds": round(time.monotonic() - started, 1),
                }
            )

    out = os.path.join(os.path.dirname(__file__), "reports", "model_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)

    print(f"{'model':<55} {'ok':>3} {'tok':>5} {'time':>6}  error/sample")
    for row in rows:
        detail = row["sample"] if row["accessible"] else row["error"]
        print(
            f"{row['model']:<55} {'yes' if row['accessible'] else 'NO':>3} "
            f"{row['tokens']:>5} {row['seconds']:>5.1f}s  {detail}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

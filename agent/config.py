import os
from dotenv import load_dotenv

load_dotenv()

# ── Remote model (Fireworks AI) ──────────────────────────────────────
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1/chat/completions",
)
# Placeholder until launch day. Swap for the announced model.
REMOTE_MODEL = os.getenv(
    "REMOTE_MODEL",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
)

# ── Local model (Ollama) ─────────────────────────────────────────────
# In docker-compose this points at the "ollama" service.
# For local dev (Ollama installed on your machine) it's localhost.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "phi3:mini")

# ── Routing knobs ────────────────────────────────────────────────────
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
# Below this quality score, a local answer is rejected and we escalate.
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.6"))

"""Runtime configuration loaded from harness-injected environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 1024
# The harness kills the container at 600s; stop early so results are written.
HARD_RUNTIME_LIMIT_SECONDS = 600.0
DEFAULT_MAX_RUNTIME_SECONDS = 540.0
DEFAULT_FIREWORKS_CONCURRENCY = 4
DEFAULT_EMBEDDING_MODEL_PATH = "/models/all-MiniLM-L6-v2"
DEFAULT_LOCAL_MODEL_PATH = "/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_LOCAL_N_CTX = 8192
DEFAULT_ROUTER_CONFIDENCE_THRESHOLD = 0.24
DEFAULT_ROUTER_MARGIN_THRESHOLD = 0.015


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    fireworks_api_key: str
    fireworks_base_url: str
    allowed_models: list[str]
    input_path: str = DEFAULT_INPUT_PATH
    output_path: str = DEFAULT_OUTPUT_PATH
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH
    local_model_path: str = DEFAULT_LOCAL_MODEL_PATH
    enable_local_model: bool = True
    fireworks_concurrency: int = DEFAULT_FIREWORKS_CONCURRENCY
    local_n_ctx: int = DEFAULT_LOCAL_N_CTX
    local_n_threads: int = 4
    local_n_batch: int = 256
    router_confidence_threshold: float = DEFAULT_ROUTER_CONFIDENCE_THRESHOLD
    router_margin_threshold: float = DEFAULT_ROUTER_MARGIN_THRESHOLD
    preferred_fireworks_model: str | None = None


def load_settings() -> Settings:
    """Read dynamic runtime settings without relying on a .env file."""

    api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    base_url = os.getenv("FIREWORKS_BASE_URL", "").strip()
    allowed_models = _split_models(os.getenv("ALLOWED_MODELS", ""))

    missing = []
    if not api_key:
        missing.append("FIREWORKS_API_KEY")
    if not base_url:
        missing.append("FIREWORKS_BASE_URL")
    if not allowed_models:
        missing.append("ALLOWED_MODELS")
    if missing:
        raise ConfigError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    preferred_model = os.getenv("PREFERRED_FIREWORKS_MODEL", "").strip() or None
    if preferred_model and preferred_model not in allowed_models:
        raise ConfigError(
            "PREFERRED_FIREWORKS_MODEL must be one of the ALLOWED_MODELS values"
        )

    return Settings(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url,
        allowed_models=allowed_models,
        input_path=os.getenv("INPUT_PATH", DEFAULT_INPUT_PATH),
        output_path=os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT_PATH),
        request_timeout_seconds=_env_float(
            "REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
            minimum=1.0,
            maximum=30.0,
        ),
        default_max_tokens=_env_int(
            "MAX_TOKENS",
            DEFAULT_MAX_TOKENS,
            minimum=16,
            maximum=2048,
        ),
        max_runtime_seconds=_env_float(
            "MAX_RUNTIME_SECONDS",
            DEFAULT_MAX_RUNTIME_SECONDS,
            minimum=1.0,
            maximum=HARD_RUNTIME_LIMIT_SECONDS,
        ),
        embedding_model_path=os.getenv(
            "EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL_PATH
        ),
        local_model_path=os.getenv("LOCAL_GGUF_PATH", DEFAULT_LOCAL_MODEL_PATH),
        enable_local_model=_env_bool("ENABLE_LOCAL_MODEL", True),
        fireworks_concurrency=_env_int(
            "FIREWORKS_CONCURRENCY",
            DEFAULT_FIREWORKS_CONCURRENCY,
            minimum=1,
            maximum=16,
        ),
        local_n_ctx=_env_int(
            "LOCAL_N_CTX", DEFAULT_LOCAL_N_CTX, minimum=512, maximum=32768
        ),
        local_n_threads=_env_int(
            "LOCAL_N_THREADS", os.cpu_count() or 4, minimum=1, maximum=32
        ),
        local_n_batch=_env_int("LOCAL_N_BATCH", 256, minimum=32, maximum=2048),
        router_confidence_threshold=_env_float(
            "ROUTER_CONFIDENCE_THRESHOLD",
            DEFAULT_ROUTER_CONFIDENCE_THRESHOLD,
            minimum=-1.0,
            maximum=1.0,
        ),
        router_margin_threshold=_env_float(
            "ROUTER_MARGIN_THRESHOLD",
            DEFAULT_ROUTER_MARGIN_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        ),
        preferred_fireworks_model=preferred_model,
    )


def _split_models(raw: str) -> list[str]:
    return [model.strip() for model in raw.split(",") if model.strip()]


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")

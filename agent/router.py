"""Semantic prompt router for token-efficient Track 1 inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
import re
from typing import Iterable

import numpy as np

from agent.config import Settings


class TaskKind(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARY = "summarisation"
    NER = "named_entity_recognition"
    DEBUGGING = "code_debugging"
    LOGIC = "logical_reasoning"
    CODE = "code_generation"


class RouteTarget(str, Enum):
    LOCAL = "local"
    FIREWORKS = "fireworks"


LOCAL_TASKS = {
    TaskKind.FACTUAL,
    TaskKind.SENTIMENT,
    TaskKind.SUMMARY,
    TaskKind.NER,
}

FIREWORKS_TASKS = {
    TaskKind.MATH,
    TaskKind.LOGIC,
    TaskKind.DEBUGGING,
    TaskKind.CODE,
}


@dataclass(frozen=True)
class RouteDecision:
    kind: TaskKind
    target: RouteTarget
    confidence: float
    margin: float
    reason: str


SEED_PROMPTS: dict[TaskKind, tuple[str, ...]] = {
    TaskKind.FACTUAL: (
        "What is the capital city of France?",
        "Who wrote Pride and Prejudice?",
        "When did the Apollo 11 moon landing happen?",
        "Define photosynthesis in one sentence.",
        "Where is the Eiffel Tower located?",
        "Which planet is known as the Red Planet?",
        "Answer this general knowledge question briefly.",
    ),
    TaskKind.MATH: (
        "Calculate 17 multiplied by 23.",
        "Solve the equation 2x + 5 = 19.",
        "What is 15 percent of 240?",
        "Find the probability of drawing two aces.",
        "Compute the average of these numbers.",
        "A word problem involving ratios, distance, time, or money.",
        "Evaluate the arithmetic expression and give the final number.",
    ),
    TaskKind.SENTIMENT: (
        "Classify the sentiment of this review as positive, negative, or neutral.",
        "Is this customer feedback happy or unhappy?",
        "Determine whether the text has positive sentiment.",
        "Label the tone of this sentence.",
        "The movie was wonderful and moving.",
        "The product broke immediately and support was terrible.",
    ),
    TaskKind.SUMMARY: (
        "Summarize the following article in two sentences.",
        "Write a concise summary of this passage.",
        "Condense the text into the main points.",
        "Give a TLDR of the paragraph.",
        "Create a brief abstract of the document.",
        "Restate the key ideas without extra detail.",
    ),
    TaskKind.NER: (
        "Extract named entities from this sentence.",
        "Identify people, organizations, locations, and dates in the text.",
        "Return all names of companies and cities mentioned.",
        "Find the named entities in the paragraph.",
        "List persons, places, and organizations.",
        "NER task with entity spans and labels.",
    ),
    TaskKind.DEBUGGING: (
        "Debug this Python traceback and explain the fix.",
        "Why does this code throw an exception?",
        "Find the bug in the function.",
        "Fix the failing test for this program.",
        "This JavaScript code has an error; repair it.",
        "Diagnose a stack trace or compiler error.",
    ),
    TaskKind.LOGIC: (
        "Given these premises, what conclusion follows?",
        "Solve this logical deduction puzzle.",
        "If all A are B and all B are C, is every A a C?",
        "Determine whether the argument is valid.",
        "Truth table reasoning with if and only if.",
        "A knights and knaves style puzzle.",
    ),
    TaskKind.CODE: (
        "Write a Python function that reverses a string.",
        "Generate code to parse a CSV file.",
        "Implement a class with these methods.",
        "Create a SQL query for this schema.",
        "Write JavaScript for the requested behavior.",
        "Produce a complete code solution.",
    ),
}


# The hashing fallback scores plain token overlap, which runs well below
# MiniLM cosine similarity; the configured thresholds assume MiniLM, so they
# are scaled down under fallback or nearly every non-lexical prompt would
# needlessly escalate to Fireworks.
FALLBACK_THRESHOLD_SCALE = 0.5


class SemanticRouter:
    """Classify prompts by cosine similarity against offline seed prompts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.encoder = _build_encoder(settings.embedding_model_path)
        self.encoder_is_fallback = isinstance(self.encoder, _HashingEncoder)
        self.seed_kinds: list[TaskKind] = []
        seed_texts: list[str] = []
        for kind, prompts in SEED_PROMPTS.items():
            for prompt in prompts:
                self.seed_kinds.append(kind)
                seed_texts.append(prompt)
        self.seed_vectors = self.encoder.encode(seed_texts)

    def route(self, prompt: str) -> RouteDecision:
        lexical_kind = _lexical_kind(prompt)
        kind, confidence, margin = self._semantic_kind(prompt)
        reason = "semantic"

        if lexical_kind is not None:
            kind = lexical_kind
            confidence = max(confidence, 0.99)
            margin = max(margin, 0.25)
            reason = "lexical+semantic"

        target = self._target_for(kind, prompt, confidence, margin, lexical_kind)
        return RouteDecision(kind, target, confidence, margin, reason)

    def _semantic_kind(self, prompt: str) -> tuple[TaskKind, float, float]:
        query = self.encoder.encode([prompt])[0]
        similarities = self.seed_vectors @ query
        scores: dict[TaskKind, float] = {kind: -1.0 for kind in TaskKind}

        for kind, score in zip(self.seed_kinds, similarities, strict=True):
            scores[kind] = max(scores[kind], float(score))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_kind, best_score = ranked[0]
        second_score = ranked[1][1]
        return best_kind, best_score, best_score - second_score

    def _target_for(
        self,
        kind: TaskKind,
        prompt: str,
        confidence: float,
        margin: float,
        lexical_kind: TaskKind | None,
    ) -> RouteTarget:
        if kind in FIREWORKS_TASKS:
            return RouteTarget.FIREWORKS

        if kind is TaskKind.FACTUAL and _looks_current_or_high_risk(prompt):
            return RouteTarget.FIREWORKS

        if not self.settings.enable_local_model:
            return RouteTarget.FIREWORKS

        confidence_floor = self.settings.router_confidence_threshold
        margin_floor = self.settings.router_margin_threshold
        if self.encoder_is_fallback:
            confidence_floor *= FALLBACK_THRESHOLD_SCALE
            margin_floor *= FALLBACK_THRESHOLD_SCALE

        if lexical_kind is None and (
            confidence < confidence_floor or margin < margin_floor
        ):
            return RouteTarget.FIREWORKS

        return RouteTarget.LOCAL


class _SentenceTransformerEncoder:
    def __init__(self, model_path: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device="cpu")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class _HashingEncoder:
    """Tiny deterministic fallback used only if the embedding model is missing."""

    dimensions = 384

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray([self._encode_one(text) for text in texts], dtype=np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[a-zA-Z0-9_+#.-]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def classify_prompt(prompt: str, settings: Settings | None = None) -> TaskKind:
    """Compatibility helper for callers that only need a category."""

    if settings is None:
        lexical_kind = _lexical_kind(prompt)
        return lexical_kind or TaskKind.FACTUAL
    return SemanticRouter(settings).route(prompt).kind


def output_budget(kind: TaskKind, prompt: str, default_max_tokens: int) -> int:
    # Unused max_tokens costs nothing (billing follows generated tokens), so
    # budgets only guard against truncation; err generous where answers can
    # legitimately run long.
    words = len(prompt.split())
    budgets = {
        TaskKind.SENTIMENT: 24,
        TaskKind.FACTUAL: 192 if _wants_explanation(prompt) else 96,
        TaskKind.NER: 192,
        TaskKind.MATH: 224,
        TaskKind.LOGIC: 224,
        TaskKind.SUMMARY: min(384, max(96, words // 3)),
        TaskKind.DEBUGGING: 768,
        TaskKind.CODE: 768,
    }
    return max(16, min(default_max_tokens, budgets[kind]))


def _wants_explanation(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:explain|describe|why|compare|discuss|elaborate)\b",
            prompt.lower(),
        )
    )


def ordered_models(
    allowed_models: list[str],
    kind: TaskKind,
    preferred_model: str | None = None,
) -> list[str]:
    """Return allowed Fireworks model IDs ordered by fit, never inventing IDs."""

    if preferred_model:
        return [preferred_model] + [
            model for model in allowed_models if model != preferred_model
        ]

    first = _select_model(allowed_models, kind)
    return [first] + [model for model in allowed_models if model != first]


def _build_encoder(model_path: str):
    try:
        return _SentenceTransformerEncoder(model_path)
    except Exception as exc:
        logging.warning(
            "embedding model %s could not be loaded; using hashing fallback: %s",
            model_path,
            exc,
        )
        return _HashingEncoder()


def _select_model(allowed_models: list[str], kind: TaskKind) -> str:
    preferences = {
        TaskKind.CODE: ("code", "coder", "qwen", "deepseek", "kimi", "gemma"),
        TaskKind.DEBUGGING: ("code", "coder", "qwen", "deepseek", "kimi", "gemma"),
        TaskKind.MATH: ("math", "reason", "qwen", "deepseek", "kimi", "gemma"),
        TaskKind.LOGIC: ("reason", "qwen", "deepseek", "kimi", "gemma"),
        TaskKind.FACTUAL: ("gemma", "llama", "qwen", "mini", "small"),
        TaskKind.SENTIMENT: ("mini", "small", "gemma", "llama", "qwen"),
        TaskKind.NER: ("mini", "small", "gemma", "llama", "qwen"),
        TaskKind.SUMMARY: ("mini", "small", "gemma", "llama", "qwen"),
    }
    # Needles are in priority order, so scan needle-first: otherwise the first
    # allowed model matching any needle wins and the preference order is moot.
    needles = preferences.get(kind, ())
    for needle in needles:
        for model in allowed_models:
            if needle in model.lower():
                return model
    return allowed_models[0]


# All lexical markers are word-bounded regexes: plain substring checks misfire
# badly ("ner" in "energy", "exception" in "exceptional") and a lexical match
# overrides the semantic router, so a false positive cannot be recovered.
_DEBUGGING_PATTERNS = (
    r"\btraceback\b",
    r"\bstack trace\b",
    r"\bdebug\w*\b",
    r"\bbugs?\b",
    r"\bfix (?:this|the|my) code\b",
    r"\bfailing tests?\b",
    r"\bexceptions?\b",
    r"\bcompiler error\b",
)

_CODE_PATTERNS = (
    r"\bwrite (?:a|an|the|some)? ?(?:\w+ )?function\b",
    r"\bwrite (?:a |the )?code\b",
    r"\bgenerate (?:a |the )?code\b",
    r"\bimplement (?:a|an|the) (?:\w+ )?(?:function|class|method)\b",
    r"\bcreate (?:a|an|the) script\b",
    r"\b(?:python|javascript|java|c\+\+) function\b",
    r"\bsql quer(?:y|ies)\b",
    r"\bcomplete the code\b",
)

_SUMMARY_PATTERNS = (
    r"\bsummar\w*\b",
    r"\btl;?dr\b",
    r"\bcondens\w*\b",
    r"\babstract of\b",
    r"\bmain points\b",
)

_NER_PATTERNS = (
    r"\bnamed entit(?:y|ies)\b",
    r"\bentit(?:y|ies)\b",
    r"\bpeople, organizations\b",
    r"\bpersons, places\b",
    r"\bner\b",
)

_SENTIMENT_PATTERNS = (
    r"\bsentiments?\b",
    r"\bpositive, negative\b",
    r"\bnegative, positive\b",
    r"\bpositive or negative\b",
    r"\bnegative or positive\b",
    r"\bclassify the review\b",
    r"\bcustomer feedback\b",
)

_MATH_PATTERNS = (
    r"\bsolv(?:e|ed|ing)\b",
    r"\bcalculat(?:e|ed|ing|ions?)\b",
    r"\bequations?\b",
    r"\bprobabilit(?:y|ies)\b",
    r"\bpercent(?:age)?s?\b",
    r"\bratios?\b",
    r"\baverage\b",
    r"\barithmetic\b",
    r"\bhow many\b",
    r"\bhow much\b",
)

_LOGIC_PATTERNS = (
    r"\blogic(?:al(?:ly)?)?\b",
    r"\bpremises?\b",
    r"\btherefore\b",
    r"\bdeduc(?:e|ed|tion|tive)\b",
    r"\btruth table\b",
    r"\bif and only if\b",
    r"\bknights and knaves\b",
    r"\bknaves?\b",
    r"\bis the argument valid\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _lexical_kind(prompt: str) -> TaskKind | None:
    text = prompt.lower()

    # Order matters. Misrouting into a Fireworks category is recoverable (the
    # remote system prompt is generic), but misrouting into a local category
    # applies a task-specific system prompt, so instruction verbs like
    # "summarize" must win over topic words like "customer feedback".
    if _matches_any(text, _DEBUGGING_PATTERNS):
        return TaskKind.DEBUGGING

    if _asks_for_code(text):
        return TaskKind.CODE

    if _matches_any(text, _SUMMARY_PATTERNS):
        return TaskKind.SUMMARY

    if _matches_any(text, _NER_PATTERNS):
        return TaskKind.NER

    if _matches_any(text, _SENTIMENT_PATTERNS):
        return TaskKind.SENTIMENT

    if _looks_mathematical(text):
        return TaskKind.MATH

    if _matches_any(text, _LOGIC_PATTERNS):
        return TaskKind.LOGIC

    if text.lstrip().startswith(
        ("who ", "what ", "when ", "where ", "which ", "define ", "name ")
    ):
        return TaskKind.FACTUAL

    return None


def _asks_for_code(text: str) -> bool:
    if "```" in text and re.search(
        r"\b(?:write|generat\w*|implement\w*|creat\w*)\b", text
    ):
        return True

    return _matches_any(text, _CODE_PATTERNS)


def _looks_mathematical(text: str) -> bool:
    if _matches_any(text, _MATH_PATTERNS):
        return True

    return bool(re.search(r"\d+\s*[-+*/^]\s*\d+", text))


def _looks_current_or_high_risk(prompt: str) -> bool:
    text = prompt.lower()
    return any(
        marker in text
        for marker in (
            "latest",
            "current",
            "currently",
            "today",
            "yesterday",
            "this year",
            "as of",
            "breaking news",
            "stock price",
            "exchange rate",
        )
    )

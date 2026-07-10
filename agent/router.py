"""Semantic prompt router for token-efficient Track 1 inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import logging
import re
from typing import Iterable

import numpy as np

from agent import deterministic
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
    DETERMINISTIC = "deterministic"


LOCAL_TASKS = {
    TaskKind.FACTUAL,
    TaskKind.SENTIMENT,
    TaskKind.SUMMARY,
}

# NER measured at 52% accuracy on the bundled Qwen2.5-1.5B model (215-case
# eval, 2026-07-10): it consistently drops one entity per multi-entity
# sentence and, in a couple of cases, hallucinated an entity lifted from its
# own system-prompt example. Factual/sentiment/summary all measured at or
# near 100% on the same model, so only NER is excluded from local routing.
FIREWORKS_TASKS = {
    TaskKind.MATH,
    TaskKind.LOGIC,
    TaskKind.DEBUGGING,
    TaskKind.CODE,
    TaskKind.NER,
}


@dataclass(frozen=True)
class RouteDecision:
    kind: TaskKind
    target: RouteTarget
    confidence: float
    margin: float
    reason: str
    # Populated only for DETERMINISTIC routes: the exact, locally computed
    # answer, so the caller never needs to re-run the solver.
    answer: str | None = None


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
        # Exact local solvers outrank everything: they only fire on prompts
        # that reduce to pure arithmetic, so a hit is always correct and
        # costs zero Fireworks tokens.
        if self.settings.enable_deterministic:
            solved = deterministic.try_solve(prompt)
            if solved is not None:
                return RouteDecision(
                    TaskKind.MATH,
                    RouteTarget.DETERMINISTIC,
                    1.0,
                    1.0,
                    "deterministic",
                    answer=solved,
                )

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
    # budgets only guard against truncation. Reasoning models (Kimi K2,
    # DeepSeek, Qwen thinking) bill chain-of-thought as completion tokens and
    # a budget that cannot fit reasoning + answer truncates the answer away,
    # so every budget is sized for reasoning headroom.
    words = len(prompt.split())
    budgets = {
        TaskKind.SENTIMENT: 320,
        TaskKind.FACTUAL: 512 if _wants_explanation(prompt) else 448,
        TaskKind.NER: 640,
        TaskKind.MATH: 1024,
        TaskKind.LOGIC: 1024,
        TaskKind.SUMMARY: min(768, max(384, words // 2)),
        TaskKind.DEBUGGING: 1536,
        TaskKind.CODE: 1536,
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
    r"\bsyntax error\b",
    r"\bruntime error\b",
    r"\binfinite loop\b",
    r"\boff[- ]by[- ]one\b",
    r"\b(?:throws?|raises?|throwing|raising) an? error\b",
)

# Softer debugging cues ("why does this fail", "returns the wrong result")
# also appear in math and logic prompts, so they only count when the prompt
# visibly contains code.
_DEBUGGING_SOFT_PATTERNS = (
    r"\bwhy (?:does|do|is|isn'?t|won'?t|doesn'?t|did)\b.{0,60}\b(?:fail\w*|work\w*|crash\w*|hang\w*|break\w*)\b",
    r"\bwhy does (?:this|the|my|it)\b.{0,80}\b(?:prints?|returns?|raises?|throws?|evaluates?)\b",
    r"\bwhat(?:'s| is) wrong\b",
    r"\breturns? the wrong\b",
    r"\bwrong (?:result|output|value|answer)s?\b",
    r"\bincorrect (?:result|output|value|answer)s?\b",
    r"\bunexpected (?:result|output|behavio\w*)\b",
    r"\bdoesn'?t (?:work|terminate|return|print|stop|compile|run)\b",
    r"\bnot work(?:ing)?\b",
    r"\bnever (?:terminates?|stops?|returns?|finishes?|ends?)\b",
    r"\bfix (?:it|this|the (?:function|loop|program|script|query|method))\b",
)

_CODE_PATTERNS = (
    r"\bwrite (?:a|an|the|some)? ?(?:\w+ )?function\b",
    r"\bwrite (?:a |the |some )?code\b",
    r"\bgenerate (?:a |the |some )?code\b",
    r"\bimplement (?:a|an|the)\b",
    r"\bwrite (?:a|an|the) (?:\w+ )?(?:program|script|method|class|query|regex|regular expression)\b",
    r"\bcreate (?:a|an|the) (?:\w+ )?(?:program|script|function|class|method|query)\b",
    r"\b(?:python|javascript|typescript|java|c\+\+|c#|go|rust|ruby|php) (?:function|program|script|class|snippet)\b",
    r"\bsql quer(?:y|ies)\b",
    r"\bcomplete the code\b",
    r"\bfill in the (?:missing|blank|rest of the)?\s*(?:code|function|implementation|body)\b",
)

_SUMMARY_PATTERNS = (
    r"\bsummar\w*\b",
    r"\btl;?dr\b",
    r"\bcondens\w*\b",
    r"\babstract of\b",
    r"\bmain points?\b",
    r"\bkey takeaways?\b",
    r"\bkey points?\b",
    r"\bone[- ]sentence (?:summary|version|overview|recap)\b",
    r"\bboil (?:this|it|the \w+) down\b",
    r"\bshorten (?:this|the)\b",
)

# Length constraints suggest summarization only when nothing stronger
# matched: "in one sentence, define X" is still factual.
_SUMMARY_WEAK_PATTERNS = (
    r"\bin (?:one|two|a single) sentences?\b",
    r"\bin (?:at most |under |no more than )?\d+ words\b",
    r"\bin \d+ words or (?:fewer|less)\b",
)

_NER_PATTERNS = (
    r"\bnamed entit(?:y|ies)\b",
    r"\bentit(?:y|ies)\b",
    r"\bpeople, organizations\b",
    r"\bpersons, places\b",
    r"\bner\b",
    r"\bproper nouns?\b",
    r"\b(?:extract|list|identify|find|pull out|return) (?:the |all |any |every )*"
    r"(?:names?|people|persons?|places?|locations?|organi[sz]ations?|companies|dates?)\b",
    r"\b(?:names?|people|persons?|places?|locations?|organi[sz]ations?|companies|cities|dates?)\b"
    r"[^.]{0,60}\bmentioned\b",
    r"\bwho, (?:what, )?where,? (?:and |or )?when\b",
)

_SENTIMENT_PATTERNS = (
    r"\bsentiments?\b",
    r"\bpositive, negative\b",
    r"\bnegative, positive\b",
    r"\bpositive or negative\b",
    r"\bnegative or positive\b",
    r"\bclassify the review\b",
    r"\bcustomer feedback\b",
    r"\btone of\b",
    r"\b(?:overall |emotional )?tone\b[^.]{0,40}\b(?:review|text|message|comment|passage|author|writer)\b",
    r"\battitude (?:expressed|conveyed|of the)\b",
    r"\bfavou?rable(?:,| or | / |/)\s*unfavou?rable\b",
    r"\bunfavou?rable(?:,| or | / |/)\s*favou?rable\b",
    r"\bhow does the (?:author|writer|reviewer|customer|speaker) feel\b",
    r"\b(?:happy|satisfied) or (?:unhappy|dissatisfied)\b",
)

_MATH_PATTERNS = (
    r"\bsolv(?:e|ed|ing)\b",
    r"\bcalculat(?:e|ed|ing|ions?)\b",
    r"\bequations?\b",
    r"\bprobabilit(?:y|ies)\b",
    r"\bpercent(?:age)?s?\b",
    r"\bratios?\b",
    r"\baverage\b",
    r"\bmean of\b",
    r"\bmedian\b",
    r"\barithmetic\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\barea of\b",
    r"\bperimeter\b",
    r"\bvolume of\b",
    r"\bradius\b",
    r"\bsum of\b",
    r"\bnext number\b",
    r"\bsequence \d|\bin the sequence\b",
    r"\bgreatest common divisor\b|\bgcd\b",
    r"\bleast common multiple\b|\blcm\b",
    r"\bsquare root\b|\bsquared\b",
    r"\bdecimal places?\b",
    r"\bconvert \d",
    r"\b(?:half|third|quarters?|three quarters) of\b",
    r"\binterest\b.*\d|\d.*\binterest\b",
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
    r"\bargument\b[^.]{0,40}\b(?:valid|invalid|sound|follows?|logical(?:ly)?)\b",
    r"\bwrong with (?:this|the|my) argument\b",
    r"\bsyllogism\b",
    r"\bvalid or invalid\b",
    r"\bcontrapositive\b",
    r"\bmodus (?:ponens|tollens)\b",
    r"\bdoes it (?:logically )?follow\b",
    r"\bwhat can (?:you|we) conclude\b",
    r"\bnecessarily\b",
    r"\b(?:taller|shorter|older|younger|faster|slower) than\b",
    r"\bfinished (?:the \w+ )?(?:before|after)\b",
    r"\bwho is the (?:shortest|tallest|oldest|youngest)\b",
    # A prompt that opens with a quantifier or conditional is almost always
    # a deduction exercise ("All of Anna's pets are cats. ...").
    r"^\s*(?:if|all|every|some)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _lexical_kind(prompt: str) -> TaskKind | None:
    text = prompt.lower()

    # Order matters: a lexical match overrides the semantic router, so
    # instruction verbs ("summarize", "extract") must win over topic words
    # ("customer feedback"), and code-context checks must run before math,
    # because code snippets are full of arithmetic-looking fragments.
    if _matches_any(text, _DEBUGGING_PATTERNS):
        return TaskKind.DEBUGGING

    if _asks_for_code(text):
        return TaskKind.CODE

    if _has_code_signals(text) and _matches_any(text, _DEBUGGING_SOFT_PATTERNS):
        return TaskKind.DEBUGGING

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

    if _matches_any(text, _SUMMARY_WEAK_PATTERNS) and len(text.split()) > 40:
        return TaskKind.SUMMARY

    if text.lstrip().startswith(
        ("who ", "what ", "when ", "where ", "which ", "define ", "name ")
    ):
        return TaskKind.FACTUAL

    return None


def _has_code_signals(text: str) -> bool:
    return bool(
        "```" in text
        or re.search(
            r"\bdef \w+|\bfunction\b|\breturn\b|\bconsole\.log\b|\bprint\s*\(|"
            r"[{};]|=>|\bclass \w+|\bimport \w+|\bselect\b.{0,60}\bfrom\b|"
            r"\bfor\s*\(|\bwhile\s*\(|\bwhile \w+ [<>=]|\bcode\b|\bfunctions?\b|"
            r"\bscripts?\b|\bquer(?:y|ies)\b|\bprograms?\b|\bsnippets?\b|"
            r"\w+\[\d+\]|=\s*['\"]",
            text,
        )
    )


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

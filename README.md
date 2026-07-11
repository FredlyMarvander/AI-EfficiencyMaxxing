# Track 1: Hybrid Token-Efficient Routing Agent

Batch AI agent for the competition harness. On startup it reads
`/input/tasks.json`, routes every prompt to the cheapest capable engine
(exact deterministic solvers at zero cost, a bundled local SLM, or Fireworks
AI for hard reasoning), and writes `/output/results.json` before exiting.

## How It Works

1. **Deterministic solvers** (`agent/deterministic.py`): prompts that reduce
   to pure arithmetic ("Calculate 17 multiplied by 23", "What is 15% of
   240?", "average of 4, 8, 15"), single-variable linear equations
   ("Solve 2x + 5 = 19 for x"), or temperature conversions ("Convert 100
   Fahrenheit to Celsius") are solved exactly in Python at zero token cost.
   The solvers only fire when the whole prompt is provably such a question;
   anything with story context, extra units, or a second variable flows to
   the LLM.
2. **Two-stage router** (`agent/router.py`): a word-boundary lexical layer
   catches explicit and hidden-style signals ("summarize", "key takeaway",
   "tone of", "why does this fail"), and a semantic layer embeds the prompt
   with `all-MiniLM-L6-v2` and scores cosine similarity against offline seed
   prompts for the eight task categories.
3. **Easy categories** (factual QA, sentiment, summarisation) can run on a
   quantized Qwen2.5-1.5B GGUF bundled in the image via llama.cpp: zero
   Fireworks tokens. Set `ENABLE_LOCAL_MODEL=0` to send them to Fireworks
   instead (costs more tokens). NER is excluded from local routing: a
   215-case eval (2026-07-10) measured it at 52% on the bundled model, which
   consistently dropped entities on multi-entity sentences, versus ~100% for
   the other three local categories, so NER always goes to Fireworks.
4. **Hard categories** (math, logic, code debugging, code generation, NER) go
   to Fireworks AI with per-category token budgets sized for reasoning models
   (whose chain-of-thought bills as completion tokens) and a category-aware
   system prompt, using a model-preference order built from `ALLOWED_MODELS`
   at runtime.
5. **Reliability**: Fireworks calls run concurrently while local inference
   gets a serial lane; rate limits and transient 5xx/429 errors retry with
   exponential backoff honoring `Retry-After`; `results.json` is written
   atomically after every task; an internal 540-second deadline guarantees
   completion before the harness's 10-minute limit; truncated answers are
   retried with a larger budget, reasoning-model think-blocks are stripped,
   and an empty answer falls back to the reasoning tail; local failures fall
   back to Fireworks, and failed remote models fall back to the next allowed
   model.

## Runtime Contract

Required environment variables (injected by the harness, never hardcoded):

- `FIREWORKS_API_KEY`: API key.
- `FIREWORKS_BASE_URL`: base URL used for every Fireworks call.
- `ALLOWED_MODELS`: comma-separated model IDs; calls are validated against
  this list before any request is sent.

Input (`/input/tasks.json`):

```json
[{"task_id":"task-1","prompt":"Classify the sentiment: I loved it."}]
```

Output (`/output/results.json`):

```json
[{"task_id":"task-1","answer":"Positive."}]
```

Optional tuning variables (see `agent/config.py` for the full list):
`FIREWORKS_CONCURRENCY` (default 4), `MAX_RUNTIME_SECONDS` (default 540),
`MAX_TOKENS` (default 2048), `ENABLE_LOCAL_MODEL` (default 1),
`ENABLE_DETERMINISTIC` (default 1), `FIREWORKS_MIN_INTERVAL` (default 0;
spaces requests for keys with tight quotas), `PREFERRED_FIREWORKS_MODEL`,
`ROUTER_CONFIDENCE_THRESHOLD`, `ROUTER_MARGIN_THRESHOLD`.

## Project Layout

```text
.
|-- main.py                 # batch entry point: parallel lanes, deadline, atomic writes
|-- agent/
|   |-- config.py           # runtime env parsing and validation
|   |-- deterministic.py    # exact zero-token solvers (arithmetic, %, averages, linear equations, temperature)
|   |-- fireworks.py        # Fireworks client: retry/backoff, usage, truncation retry
|   |-- io.py               # /input and /output JSON helpers (atomic writes)
|   |-- local_model.py      # llama.cpp GGUF wrapper with per-category prompts
|   |-- prompts.py          # compact per-category system prompts
|   |-- router.py           # lexical + semantic task routing, budgets, model order
|   `-- tokens.py           # cheap token estimation for context-window guards
|-- eval/
|   |-- eval.py             # single-config eval harness
|   |-- grading.py          # shared answer-matching heuristics
|   |-- run_modes.py        # multi-mode comparison (fireworks / det / hybrid)
|   |-- route_check.py      # offline routing sweep (no API calls)
|   |-- test_deterministic.py  # offline solver + router unit tests
|   |-- test_cases.json     # 43 quick cases
|   `-- test_cases_v2.json  # 215 cases across all eight categories
|-- Dockerfile              # linux/amd64; bundles MiniLM + Qwen2.5-1.5B GGUF
|-- docker-compose.yml
`-- requirements.txt
```

## Build and Run

Build for linux/amd64 (downloads both models into the image):

```bash
docker buildx build --platform linux/amd64 -t track1-agent .
```

Run exactly like the harness does. Copy `.env.example` to `.env` first and
fill in your key; `.env` is gitignored and never baked into the image.

macOS/Linux:

```bash
docker run --rm --platform linux/amd64 --env-file .env \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  track1-agent
```

Windows (CMD; on PowerShell replace `%CD%` with `${PWD}`):

```cmd
docker run --rm --platform linux/amd64 --env-file .env -v "%CD%\input:/input:ro" -v "%CD%\output:/output" track1-agent
```

The container exits with code `0` after writing valid JSON results. Check
token usage in the logs:

```bash
docker compose logs agent | grep tokens
```

Each task logs its route (`route task=... target=local|fireworks`) and token
usage; the final `tokens total` line is the score-relevant number.

## Evaluate Before Submitting

The eval directory is intentionally not baked into the image; mount the repo.

Offline (no API key usage): unit tests for the deterministic solvers and
lexical router, plus a routing sweep over the full 215-case set:

```bash
docker run --rm -v "$PWD:/app" track1-agent python -m eval.test_deterministic
docker run --rm --env-file .env -v "$PWD:/app" track1-agent python -m eval.route_check
```

Live comparison of routing configurations (spends Fireworks tokens; set
`FIREWORKS_MIN_INTERVAL` if your key has a tight quota):

```bash
docker run --rm --env-file .env -v "$PWD:/app" track1-agent \
  python -m eval.run_modes --modes fireworks,det,hybrid
```

It prints per-category accuracy, routing precision, local/deterministic
share, Fireworks tokens, and wall time per mode, writes JSON reports to
`eval/reports/`, and ends with a mode-by-mode comparison. Use it to decide
whether `ENABLE_LOCAL_MODEL` should stay on before submitting. On PowerShell
replace `$PWD` with `${PWD}`; on CMD use `%CD%`.

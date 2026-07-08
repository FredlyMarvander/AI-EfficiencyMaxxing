# Track 1: Hybrid Token-Efficient Routing Agent

Batch AI agent for the competition harness. On startup it reads
`/input/tasks.json`, routes every prompt to the cheapest capable engine
(a bundled local SLM at zero token cost, or Fireworks AI for hard reasoning),
and writes `/output/results.json` before exiting.

## How It Works

1. **Two-stage router** (`agent/router.py`): a word-boundary lexical layer
   catches explicit signals ("summarize", "traceback", "solve"), and a
   semantic layer embeds the prompt with `all-MiniLM-L6-v2` and scores cosine
   similarity against offline seed prompts for the eight task categories.
2. **Easy categories** (factual QA, sentiment, summarisation, NER) run on a
   quantized Qwen2.5-1.5B GGUF bundled in the image via llama.cpp: zero
   Fireworks tokens.
3. **Hard categories** (math, logic, code debugging, code generation) go to
   Fireworks AI with per-category token budgets and a model-preference order
   built from `ALLOWED_MODELS` at runtime.
4. **Reliability**: Fireworks calls run concurrently while local inference
   gets a serial lane; `results.json` is written atomically after every task;
   an internal 540-second deadline guarantees completion before the harness's
   10-minute limit; truncated answers are retried with a larger budget and
   reasoning-model think-blocks are stripped; local failures fall back to
   Fireworks, and failed remote models fall back to the next allowed model.

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
`MAX_TOKENS` (default 1024), `ENABLE_LOCAL_MODEL`, `PREFERRED_FIREWORKS_MODEL`,
`ROUTER_CONFIDENCE_THRESHOLD`, `ROUTER_MARGIN_THRESHOLD`.

## Project Layout

```text
.
|-- main.py              # batch entry point: parallel lanes, deadline, atomic writes
|-- agent/
|   |-- config.py        # runtime env parsing and validation
|   |-- fireworks.py     # Fireworks chat client: usage tracking, truncation retry
|   |-- io.py            # /input and /output JSON helpers (atomic writes)
|   |-- local_model.py   # llama.cpp GGUF wrapper with per-category prompts
|   |-- prompts.py       # compact shared system prompt
|   |-- router.py        # lexical + semantic task routing, budgets, model order
|   `-- tokens.py        # cheap token estimation for context-window guards
|-- eval/
|   |-- eval.py          # local eval harness: accuracy, routing, token report
|   `-- test_cases.json  # 43 cases across all eight categories
|-- Dockerfile           # linux/amd64; bundles MiniLM + Qwen2.5-1.5B GGUF
|-- docker-compose.yml
`-- requirements.txt
```

## Build and Run

Build for linux/amd64 (downloads both models into the image):

```bash
docker buildx build --platform linux/amd64 -t track1-agent .
```

Run exactly like the harness does:

```bash
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  track1-agent
```

The container exits with code `0` after writing valid JSON results. Check
token usage in the logs:

```bash
docker compose logs agent | grep tokens
```

Each task logs its route (`route task=... target=local|fireworks`) and token
usage; the final `tokens total` line is the score-relevant number.

## Evaluate Before Submitting

The eval directory is intentionally not baked into the image; mount it:

```bash
docker run --rm --platform linux/amd64 --env-file .env \
  -v "$PWD/eval:/app/eval" --entrypoint python \
  track1-agent -m eval.eval
```

It prints per-category accuracy, routing precision, local share, Fireworks
tokens, and wall time for 43 cases across all eight competition categories.
Use it to decide which categories belong in `LOCAL_TASKS` vs
`FIREWORKS_TASKS` (`agent/router.py`) before submitting.

# Track 1 General-Purpose AI Agent

Batch AI agent for the competition harness. On startup it reads
`/input/tasks.json`, sends each prompt to Fireworks AI through the injected
base URL, and writes `/output/results.json` before exiting.

## Runtime Contract

Required environment variables:

- `FIREWORKS_API_KEY`: API key injected by the harness.
- `FIREWORKS_BASE_URL`: Fireworks/OpenAI-compatible base URL injected by the harness.
- `ALLOWED_MODELS`: comma-separated model IDs. The code uses the first model in this list.

Input:

```json
[{"task_id":"task-1","prompt":"Classify the sentiment: I loved it."}]
```

Output:

```json
[{"task_id":"task-1","answer":"Positive."}]
```

## Token-Efficient Behavior

- Uses a concise English-only system prompt.
- Avoids greetings, preambles, and closings.
- Applies lightweight task classification for factual QA, math, sentiment,
  summarization, NER, debugging, logic, and code generation.
- Sets smaller `max_tokens` budgets for short-answer tasks and larger budgets
  for code/debugging tasks.
- Uses `temperature=0` for deterministic, direct answers.
- Uses a 30-second per-request timeout and a 10-minute overall runtime cap.

## Project Layout

```text
.
|-- main.py              # batch entry point
|-- agent/
|   |-- config.py        # runtime env parsing and validation
|   |-- fireworks.py     # requests-based Fireworks chat client
|   |-- io.py            # /input and /output JSON helpers
|   |-- prompts.py       # compact system prompt
|   `-- router.py        # task type, model ordering, token budgets
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Build and Run

Build for linux/amd64:

```bash
docker buildx build --platform linux/amd64 -t track1-agent .
```

Run locally with mounted input/output folders:

```bash
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  track1-agent
```

The container exits with code `0` after successfully writing valid JSON results.

## Run With Docker Compose

Set only your API key, then run compose. The local compose file already supplies
the Fireworks base URL and uses `minimax-m3` as the single default allowed model
for token-efficient local testing.

```powershell
$env:FIREWORKS_API_KEY="your_api_key"
docker compose up --build
```

Check token usage from the container logs:

```cmd
docker compose logs agent | findstr tokens
```

Each successful task logs `prompt`, `completion`, and `total` tokens. The final
`tokens total` line is the number to compare between model/prompt settings.

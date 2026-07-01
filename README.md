# Hybrid Token-Efficient Routing Agent

Submission for **AMD Developer Hackathon: ACT II — Track 1**.

An AI agent that completes each task using the fewest tokens possible by
deciding, in real time, whether to answer with a small **local model**
(via Ollama) or escalate to a **remote model** (via the Fireworks AI API).

## How it routes

```
task in
  │
  ├─ Layer 1: rule-based  → 0 tokens. Obvious-simple → local, obvious-hard → remote.
  │
  └─ Layer 2: try local, then check the answer's quality.
             good enough  → keep the local answer
             looks weak   → escalate to remote
```

Token counts come straight from each provider's response, and both the
local and remote tokens are tallied (so the score is honest even when a
task gets escalated).

## Project layout

```
amd-routing-agent/
├── agent/
│   ├── config.py        ← EDIT THIS ON LAUNCH DAY (model names + knobs)
│   ├── router.py        ← routing logic (the brain)
│   ├── local_model.py   ← Ollama local model
│   ├── fireworks.py     ← Fireworks AI remote model
│   └── tokens.py        ← token estimate fallback
├── api/main.py          ← FastAPI endpoint
├── eval/
│   ├── eval.py          ← local accuracy + token harness
│   └── test_cases.json  ← sample tasks
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run locally (development)

1. Install [Ollama](https://ollama.com) and pull a small model:
   ```bash
   ollama pull phi3:mini
   ```
2. Set up Python:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env      # then paste your Fireworks key into .env
   ```
3. Start the API:
   ```bash
   uvicorn api.main:app --reload
   ```
4. Test it:
   ```bash
   curl -X POST http://localhost:8000/run \
     -H "Content-Type: application/json" \
     -d '{"task": "Translate good morning to Spanish"}'
   ```

## Run the eval

```bash
python -m eval.eval
```

Prints, per task: which model was used, tokens spent, pass/fail, and the
routing reason — plus totals.

## Run with Docker (containerized submission)

```bash
docker-compose up --build
# one-time: pull the local model into the ollama container
docker-compose exec ollama ollama pull phi3:mini
```

The agent is then on `http://localhost:8000`. The pulled model is kept in
a named volume, so you only pull it once.

## Launch-day checklist (July 6)

- [ ] Put the announced **remote** model name in `agent/config.py` (`REMOTE_MODEL`).
- [ ] Put the announced **local** model name in `agent/config.py` (`LOCAL_MODEL`) and `ollama pull` it.
- [ ] Replace `eval/test_cases.json` with (a sample of) the real tasks.
- [ ] Tune the keyword lists and `quality_score()` in `agent/router.py`.
- [ ] Re-run `python -m eval.eval` and watch the token total drop.

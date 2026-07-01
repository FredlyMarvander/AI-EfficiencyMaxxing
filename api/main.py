"""FastAPI entry point. Run: uvicorn api.main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from agent.router import run_agent

app = FastAPI(title="Hybrid Token-Efficient Routing Agent")


class TaskRequest(BaseModel):
    task: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(request: TaskRequest):
    """Send one task, get the answer plus routing + token info back."""
    return run_agent(request.task)

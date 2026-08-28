"""Baseline: run the email agent on a CLOSED model (OpenAI) and measure accuracy.

This is the state of the world *before* the migration — the agent runs on a hosted
OpenAI model. The number this prints (fraction of validation questions answered
correctly, per the LLM judge) is the bar the fine-tuned open-source model has to match.

Usage (from repo root):
    OPENAI_API_KEY=... uv run scripts/baseline.py
Env:
    BASELINE_MODEL   OpenAI model to run the agent on (default: gpt-4o)
    N_VALIDATION     number of validation scenarios (default: 20)
"""

from __future__ import annotations

import asyncio
import os

import weave
from openai import AsyncOpenAI

from email_agent.agent import run_agent
from email_agent.data import load_scenarios

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "email-search-agent")

BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o")
N_VALIDATION = int(os.environ.get("N_VALIDATION", "20"))


async def main() -> None:
    # Agent Pulse tracing: every agent + judge call is captured in Weave
    # (inputs, outputs, latency, cost) so the baseline model's spend is visible.
    weave.init(WANDB_PROJECT)

    scenarios = load_scenarios(split="test", limit=N_VALIDATION, max_messages=1, shuffle=True, seed=42)
    client = AsyncOpenAI()  # reads OPENAI_API_KEY; talks to OpenAI directly

    print(f"\nRunning the email agent on the closed baseline model: {BASELINE_MODEL}")
    print(f"Evaluating on {len(scenarios)} held-out questions...\n")

    results = await asyncio.gather(
        *(run_agent(s, client=client, model_name=BASELINE_MODEL) for s in scenarios)
    )
    correct = [t.metrics.get("correct", 0.0) for t in results]
    accuracy = sum(correct) / len(correct) if correct else 0.0

    print("\n" + "=" * 48)
    print(f"Baseline model : {BASELINE_MODEL}")
    print(f"Accuracy       : {accuracy:.1%}  ({int(sum(correct))}/{len(correct)} correct)")
    print("=" * 48)
    print("This is the bar the fine-tuned open-source model needs to reach.")


if __name__ == "__main__":
    asyncio.run(main())

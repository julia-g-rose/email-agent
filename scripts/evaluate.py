"""Evaluate a hosted checkpoint (the fine-tuned model) and print its accuracy.

Runs the same agent + judge as the baseline, but against a model served over an
OpenAI-compatible endpoint — e.g. a W&B Serverless Training checkpoint
(`wandb-artifact:///entity/project/model:stepN`). Compare the number this prints to
`scripts/baseline.py` to see whether the fine-tune reached parity with the closed model.

Usage (from repo root):
    OPENAI_API_KEY=... WANDB_API_KEY=... MODEL_REF="wandb-artifact:///ENTITY/PROJECT/email-agent-001:step60" \
        uv run scripts/evaluate.py
Env:
    MODEL_REF        served model / artifact ref to evaluate (required)
    INFERENCE_BASE   OpenAI-compatible base URL (default: https://api.training.wandb.ai/v1)
    N_VALIDATION     number of validation scenarios (default: 20)
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from email_agent.agent import run_agent
from email_agent.data import load_scenarios

MODEL_REF = os.environ.get("MODEL_REF")
INFERENCE_BASE = os.environ.get("INFERENCE_BASE", "https://api.training.wandb.ai/v1")
N_VALIDATION = int(os.environ.get("N_VALIDATION", "20"))


async def main() -> None:
    if not MODEL_REF:
        raise SystemExit("Set MODEL_REF to the served checkpoint (e.g. wandb-artifact:///.../email-agent-001:step60).")

    scenarios = load_scenarios(split="test", limit=N_VALIDATION, max_messages=1, shuffle=True, seed=42)
    client = AsyncOpenAI(base_url=INFERENCE_BASE, api_key=os.environ.get("WANDB_API_KEY"))

    print(f"\nEvaluating fine-tuned checkpoint: {MODEL_REF}")
    print(f"On {len(scenarios)} held-out questions...\n")

    results = await asyncio.gather(
        *(run_agent(s, client=client, model_name=MODEL_REF) for s in scenarios)
    )
    correct = [t.metrics.get("correct", 0.0) for t in results]
    accuracy = sum(correct) / len(correct) if correct else 0.0

    print("\n" + "=" * 48)
    print(f"Checkpoint : {MODEL_REF}")
    print(f"Accuracy   : {accuracy:.1%}  ({int(sum(correct))}/{len(correct)} correct)")
    print("=" * 48)


if __name__ == "__main__":
    asyncio.run(main())

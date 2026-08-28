"""RL fine-tune the open-source model with ART + RULER on W&B Serverless Training.

Trains the agent's policy so an open-source base model (Qwen) learns to research the
inbox as well as the closed baseline. Reward is RULER (relative LLM-judge scoring of
each trajectory group); training/inference run on W&B Serverless via ServerlessBackend.
Faithful to the ART·E training loop.

Usage (from repo root):
    OPENAI_API_KEY=... WANDB_API_KEY=... uv run scripts/train.py
Env:
    BASE_MODEL   open-source base to fine-tune (default: Qwen/Qwen3.6-27B)
    RULER_MODEL  judge for RULER relative scoring (default: openai/gpt-4o)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random

import art
import weave
from art.serverless.backend import ServerlessBackend
from art.utils import iterate_dataset
from art.utils.strip_logprobs import strip_logprobs
from dotenv import load_dotenv

from email_agent.agent import EmailScenario, rollout
from email_agent.data import load_scenarios
from email_agent.rewards import RULER_MODEL, ruler_score_group

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-27B")
PROJECT = os.environ.get("WANDB_PROJECT", "email-search-agent")

TRAINING_SCENARIO_LIMIT = int(os.environ.get("TRAIN_LIMIT", "720"))
VALIDATION_SCENARIO_LIMIT = int(os.environ.get("VAL_LIMIT", "20"))

TRAINING_CONFIG = {
    "groups_per_step": 12,
    "num_epochs": 20,
    "trajectories_per_group": 4,
    "learning_rate": 1.2e-5,
    "max_steps": 60,
    "validation_step_interval": 10,
}


async def main() -> None:
    load_dotenv()
    random.seed(42)
    logging.getLogger("weave").setLevel(logging.CRITICAL)

    training_scenarios = load_scenarios(
        split="train", limit=TRAINING_SCENARIO_LIMIT, max_messages=1, shuffle=True, seed=42
    )
    validation_scenarios = load_scenarios(
        split="test", limit=VALIDATION_SCENARIO_LIMIT, max_messages=1, shuffle=True, seed=42
    )
    print(f"{len(training_scenarios)} training / {len(validation_scenarios)} validation scenarios")

    model = art.TrainableModel(name="email-agent-001", project=PROJECT, base_model=BASE_MODEL)
    backend = ServerlessBackend()
    await model.register(backend)

    # RL rollouts run the same @weave.op agent loop, so trajectories are traced too.
    # strip_logprobs keeps the large per-token logprob arrays out of the Weave payloads.
    weave.init(model.project, settings={"print_call_link": False}, global_postprocess_output=strip_logprobs)

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=TRAINING_CONFIG["groups_per_step"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch in training_iterator:
        print(f"Step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step} — {len(batch.items)} scenarios")

        train_groups = [
            art.TrajectoryGroup(
                rollout(model, EmailScenario(step=batch.step, scenario=s))
                for _ in range(TRAINING_CONFIG["trajectories_per_group"])
            )
            for s in batch.items
        ]
        finished = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=TRAINING_CONFIG["trajectories_per_group"] * len(batch.items),
        )

        judged_groups = [await ruler_score_group(g, RULER_MODEL, debug=True) for g in finished]

        if batch.step % TRAINING_CONFIG["validation_step_interval"] == 0:
            print(f"Running validation at step {batch.step}")
            val_groups = [
                art.TrajectoryGroup([rollout(model, EmailScenario(step=batch.step, scenario=s))])
                for s in validation_scenarios
            ]
            finished_val = await art.gather_trajectory_groups(
                val_groups,
                pbar_desc="gather",
                max_exceptions=TRAINING_CONFIG["trajectories_per_group"] * len(validation_scenarios),
            )
            await model.log(finished_val, split="val")

        train_result = await backend.train(
            model, judged_groups, learning_rate=TRAINING_CONFIG["learning_rate"]
        )
        await model.log(judged_groups, metrics=train_result.metrics, step=train_result.step, split="train")
        await model.delete_checkpoints("val/correct")

        print(f"Completed training step {train_result.step}")
        if batch.step >= TRAINING_CONFIG["max_steps"]:
            break


if __name__ == "__main__":
    asyncio.run(main())

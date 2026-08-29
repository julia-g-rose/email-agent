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

BASE_MODEL = os.environ.get("BASE_MODEL", "OpenPipe/Qwen3-14B-Instruct")
MODEL_NAME = os.environ.get("MODEL_NAME", "email-agent-14b")
PROJECT = os.environ.get("WANDB_PROJECT", "email-agent")
ENTITY = os.environ.get("WANDB_ENTITY", "wb-agent-team")

TRAINING_SCENARIO_LIMIT = int(os.environ.get("TRAIN_LIMIT", "720"))
VALIDATION_SCENARIO_LIMIT = int(os.environ.get("VAL_LIMIT", "20"))

TRAINING_CONFIG = {
    "groups_per_step": int(os.environ.get("GROUPS_PER_STEP", "12")),
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "20")),
    "trajectories_per_group": int(os.environ.get("TRAJ_PER_GROUP", "4")),
    "learning_rate": float(os.environ.get("LEARNING_RATE", "1.2e-5")),
    "max_steps": int(os.environ.get("MAX_STEPS", "60")),
    "validation_step_interval": int(os.environ.get("VAL_INTERVAL", "10")),
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference bars for the eval table (measured separately, temp=0, N=50).
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt-4.1-mini")
BASELINE_ACCURACY = float(os.environ.get("BASELINE_ACCURACY", "0.84"))
UNTRAINED_ACCURACY = float(os.environ.get("UNTRAINED_ACCURACY", "0.62"))


def _log_eval_table(model, scenarios, finished_val, step) -> None:
    """Log a typed EvalTable (inputs / outputs / scores) at this validation step.

    Uses `wandb.EvalTable` so the results route to the Evaluation Tables compare panel:
    inputs = the question + reference, output = the agent's answer, score = correctness.
    Logged once per validation loop, so each training step is selectable in the panel's
    native step dimension. Also logs trained accuracy against the fixed baseline bars.
    """
    run = model._get_wandb_run()
    if run is None:
        return
    import wandb

    rows = []  # ordered input -> output -> score
    correct = 0.0
    n = 0
    for scenario, group in zip(scenarios, finished_val):
        traj = group.trajectories[0] if getattr(group, "trajectories", None) else None
        answer = traj.final_answer.answer if (traj and traj.final_answer) else ""
        is_correct = float(traj.metrics.get("correct", 0.0)) if traj else 0.0
        correct += is_correct
        n += 1
        rows.append([scenario.id, scenario.question, scenario.answer, answer, is_correct])
    accuracy = correct / n if n else 0.0

    eval_table = wandb.EvalTable(
        input_columns=["scenario_id", "question", "reference_answer"],
        output_columns=["model_answer"],
        score_columns=["correct"],
        data=rows,
    )
    run.log(
        {
            "eval": eval_table,
            "eval/trained_accuracy": accuracy,
            "eval/baseline_accuracy": BASELINE_ACCURACY,  # gpt-4.1-mini bar
            "eval/untrained_accuracy": UNTRAINED_ACCURACY,  # 14B starting point
        }
    )


def _log_code(model: art.TrainableModel) -> bool:
    """Attach the repo source to the run's Code tab so it's inspectable from the run.

    Returns True once the code has been logged (the wandb run exists by then).
    """
    run = model._get_wandb_run()
    if run is None:
        return False
    run.log_code(
        root=REPO_ROOT,
        include_fn=lambda p, r=None: (
            "/.venv/" not in p
            and "/wandb/" not in p
            and "/artifacts/" not in p
            and not p.endswith(".db")
            and (p.endswith((".py", ".toml", ".md")) or p.endswith(".env.example"))
        ),
    )
    return True


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

    model = art.TrainableModel(name=MODEL_NAME, project=PROJECT, entity=ENTITY, base_model=BASE_MODEL)
    backend = ServerlessBackend()
    await model.register(backend)

    # RL rollouts run the same @weave.op agent loop, so trajectories are traced too.
    # strip_logprobs keeps the large per-token logprob arrays out of the Weave payloads.
    weave.init(f"{model.entity}/{model.project}", settings={"print_call_link": False}, global_postprocess_output=strip_logprobs)

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=TRAINING_CONFIG["groups_per_step"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
        initial_step=await model.get_step(),
    )

    code_logged = _log_code(model)
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
            _log_eval_table(model, validation_scenarios, finished_val, batch.step)

        train_result = await backend.train(
            model, judged_groups, learning_rate=TRAINING_CONFIG["learning_rate"]
        )
        await model.log(judged_groups, metrics=train_result.metrics, step=train_result.step, split="train")
        if not code_logged:
            code_logged = _log_code(model)
        await model.delete_checkpoints("val/correct")

        print(f"Completed training step {train_result.step}")
        if batch.step >= TRAINING_CONFIG["max_steps"]:
            break


if __name__ == "__main__":
    asyncio.run(main())

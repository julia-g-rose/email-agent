"""RL fine-tune the open-source model with ART + RULER on W&B Serverless Training.

Trains the agent's policy so an open-source base model (Qwen) learns to research the
inbox as well as the closed baseline. Reward combines strict answer correctness,
reference-message citation F1, and a small RULER relative-quality term;
training/inference run on W&B Serverless via ServerlessBackend.
Faithful to the ART·E training loop.

Usage (from repo root):
    OPENAI_API_KEY=... WANDB_API_KEY=... uv run scripts/train.py
Env:
    BASE_MODEL   open-source base to fine-tune (default: Qwen/Qwen3.6-27B)
    RULER_MODEL  judge for RULER relative scoring (default: openai/gpt-4o)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random

import art
import weave
from art.serverless.backend import ServerlessBackend
from art.utils import iterate_dataset
from art.utils.strip_logprobs import strip_logprobs
from dotenv import load_dotenv

from email_agent.agent import EmailScenario, JUDGE_MODEL, rollout
from email_agent.data import load_scenarios
from email_agent.rewards import (
    CORRECT_WEIGHT,
    RULER_MODEL,
    RULER_WEIGHT,
    SOURCE_WEIGHT,
    hybrid_score_group,
    source_f1,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "OpenPipe/Qwen3-14B-Instruct")
MODEL_NAME = os.environ.get("MODEL_NAME", "email-agent-14b-hybrid-reward-v1")
PROJECT = os.environ.get("WANDB_PROJECT", "email-agent")
ENTITY = os.environ.get("WANDB_ENTITY", "wb-agent-team")

TRAINING_SCENARIO_LIMIT = int(os.environ.get("TRAIN_LIMIT", "720"))
VALIDATION_SCENARIO_LIMIT = int(os.environ.get("VAL_LIMIT", "30"))

TRAINING_CONFIG = {
    "groups_per_step": int(os.environ.get("GROUPS_PER_STEP", "12")),
    "num_epochs": int(os.environ.get("NUM_EPOCHS", "20")),
    "trajectories_per_group": int(os.environ.get("TRAJ_PER_GROUP", "4")),
    "learning_rate": float(os.environ.get("LEARNING_RATE", "1.2e-5")),
    "max_steps": int(os.environ.get("MAX_STEPS", "20")),
    "validation_step_interval": int(os.environ.get("VAL_INTERVAL", "5")),
}
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "2"))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference bars measured by the standalone runs on the same 30-question,
# temperature-0 held-out set. Override only when rerunning those baselines.
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt-4.1-mini")
BASELINE_ACCURACY = float(os.environ.get("BASELINE_ACCURACY", str(22 / 30)))
UNTRAINED_ACCURACY = float(os.environ.get("UNTRAINED_ACCURACY", str(18 / 30)))


def _log_eval_table(model, scenarios, finished_val, step) -> float:
    """Log a typed EvalTable (inputs / outputs / scores) at this validation step.

    Uses `wandb.EvalTable` so the results route to the Evaluation Tables compare panel:
    inputs = the question + reference, output = the agent's answer, score = correctness.
    Logged once per validation loop, so each training step is selectable in the panel's
    native step dimension. Also logs trained accuracy against the fixed baseline bars.
    """
    run = model._get_wandb_run()
    if run is None:
        return 0.0
    import wandb

    rows = []  # ordered input -> output -> score
    correct = 0.0
    source_f1_total = 0.0
    n = 0
    for scenario, group in zip(scenarios, finished_val):
        traj = group.trajectories[0] if getattr(group, "trajectories", None) else None
        answer = traj.final_answer.answer if (traj and traj.final_answer) else ""
        is_correct = float(traj.metrics.get("correct", 0.0)) if traj else 0.0
        expected_sources = list(scenario.message_ids)
        predicted_sources = list(traj.final_answer.source_ids) if (traj and traj.final_answer) else []
        citation_f1 = source_f1(predicted_sources, expected_sources)
        correct += is_correct
        source_f1_total += citation_f1
        n += 1
        rows.append([
            scenario.id,
            scenario.question,
            scenario.answer,
            json.dumps(expected_sources),
            answer,
            json.dumps(predicted_sources),
            is_correct,
            citation_f1,
        ])
    accuracy = correct / n if n else 0.0
    mean_source_f1 = source_f1_total / n if n else 0.0

    eval_table = wandb.EvalTable(
        input_columns=["scenario_id", "question", "reference_answer", "reference_source_ids"],
        output_columns=["model_answer", "predicted_source_ids"],
        score_columns=["correct", "source_f1"],
        data=rows,
    )
    run.log(
        {
            "eval": eval_table,
            "eval/trained_accuracy": accuracy,
            "eval/correct_count": correct,
            "eval/sample_count": n,
            "eval/source_f1": mean_source_f1,
            "eval/baseline_accuracy": BASELINE_ACCURACY,  # gpt-4.1-mini bar
            "eval/untrained_accuracy": UNTRAINED_ACCURACY,  # 14B starting point
        }
    )
    return accuracy


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
    run = model._get_wandb_run()
    if run is not None:
        run.config.update(
            {
                "experiment": "hybrid-correctness-source-ruler-v1",
                "base_model": BASE_MODEL,
                "model_name": MODEL_NAME,
                "seed": 42,
                "train_limit": TRAINING_SCENARIO_LIMIT,
                "val_limit": VALIDATION_SCENARIO_LIMIT,
                "eval/baseline_accuracy": BASELINE_ACCURACY,
                "eval/untrained_accuracy": UNTRAINED_ACCURACY,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "ruler_model": RULER_MODEL,
                "correctness_judge_model": JUDGE_MODEL,
                "reward/correct_weight": CORRECT_WEIGHT,
                "reward/source_f1_weight": SOURCE_WEIGHT,
                "reward/ruler_weight": RULER_WEIGHT,
                **TRAINING_CONFIG,
            },
            allow_val_change=True,
        )
    best_val = -1.0
    stale_validations = 0
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

        judged_groups = [await hybrid_score_group(g, RULER_MODEL, debug=True) for g in finished]

        if batch.step % TRAINING_CONFIG["validation_step_interval"] == 0:
            print(f"Running validation at step {batch.step}")
            val_groups = [
                art.TrajectoryGroup(
                    [rollout(model, EmailScenario(step=batch.step, scenario=s), temperature=0.0)]
                )
                for s in validation_scenarios
            ]
            finished_val = await art.gather_trajectory_groups(
                val_groups,
                pbar_desc="gather",
                max_exceptions=TRAINING_CONFIG["trajectories_per_group"] * len(validation_scenarios),
            )
            await model.log(finished_val, split="val")
            val_accuracy = _log_eval_table(model, validation_scenarios, finished_val, batch.step)
            if val_accuracy > best_val:
                best_val = val_accuracy
                stale_validations = 0
            else:
                stale_validations += 1

        train_result = await backend.train(
            model, judged_groups, learning_rate=TRAINING_CONFIG["learning_rate"]
        )
        await model.log(judged_groups, metrics=train_result.metrics, step=train_result.step, split="train")
        if not code_logged:
            code_logged = _log_code(model)
        await model.delete_checkpoints("val/correct")

        print(f"Completed training step {train_result.step}")
        if stale_validations >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after {EARLY_STOP_PATIENCE} validation checks without improvement")
            break
        if batch.step >= TRAINING_CONFIG["max_steps"]:
            break


if __name__ == "__main__":
    asyncio.run(main())

"""Run a reproducible paired evaluation of two Serverless Training checkpoints.

The default comparison isolates the effect of the existing two-step smoke
fine-tune. It logs one W&B run with a fixed scenario manifest, checkpoint
artifact lineage, per-trial results, and a scenario-level bootstrap interval.

Usage:
    WANDB_API_KEY=... uv run python scripts/paired_evaluate.py

Environment overrides:
    CONTROL_MODEL_REF   Control checkpoint (default: email-agent-smoke-r1:step0)
    TREATMENT_MODEL_REF Treatment checkpoint (default: email-agent-smoke-r1:step2)
    N_VALIDATION        Held-out scenario count (default: 20)
    TRIALS              Trials per scenario and checkpoint (default: 3)
    EVAL_SEED           Scenario-manifest seed (default: 42)
    TEMPERATURE         Generation temperature (default: 1.0)
    JUDGE_MODEL         Correctness judge used by email_agent.agent
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import httpx
from openai import AsyncOpenAI
import wandb
import weave

from email_agent.paired_eval import (
    paired_scenario_rows,
    paired_summary,
    parse_model_ref,
    select_scenarios,
)


PROJECT = os.environ.get("WANDB_PROJECT", "email-search-agent")
ENTITY = os.environ.get("WANDB_ENTITY", "wandb")
INFERENCE_BASE = os.environ.get("INFERENCE_BASE", "https://api.training.wandb.ai/v1")
CONTROL_MODEL_REF = os.environ.get(
    "CONTROL_MODEL_REF",
    "wandb-artifact:///wandb/email-search-agent/email-agent-smoke-r1:step0",
)
TREATMENT_MODEL_REF = os.environ.get(
    "TREATMENT_MODEL_REF",
    "wandb-artifact:///wandb/email-search-agent/email-agent-smoke-r1:step2",
)
N_VALIDATION = int(os.environ.get("N_VALIDATION", "20"))
TRIALS = int(os.environ.get("TRIALS", "3"))
EVAL_SEED = int(os.environ.get("EVAL_SEED", "42"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))


class CheckpointModel(weave.Model):
    """A model identity that does not collapse different checkpoints in Weave."""

    model_ref: str
    checkpoint_step: str
    base_model: str | None = None
    code_commit: str | None = None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def count_tool_calls(traj: Any) -> int:
    count = 0
    for message in traj.messages():
        if isinstance(message, dict) and message.get("role") == "assistant":
            count += len(message.get("tool_calls") or [])
        elif getattr(message, "role", None) == "assistant":
            count += len(getattr(message, "tool_calls", None) or [])
    return count


async def evaluate_one(
    *,
    scenario: Any,
    trial: int,
    model_label: str,
    model_ref: str,
    client: AsyncOpenAI,
) -> dict[str, Any]:
    from email_agent.agent import run_agent

    try:
        traj = await run_agent(
            scenario,
            client=client,
            model_name=model_ref,
            step=trial,
            temperature=TEMPERATURE,
            trace_label=f"{model_label}-trial{trial}",
        )
        answer = traj.final_answer.answer if traj.final_answer else ""
        return {
            "scenario_id": scenario.id,
            "question": scenario.question,
            "reference": scenario.answer,
            "trial": trial,
            "model_label": model_label,
            "model_ref": model_ref,
            "answer": answer,
            "correct": float(traj.metrics.get("correct", 0.0)),
            "empty_answer": not bool(answer.strip()),
            "incomplete": traj.final_answer is None,
            "exception": False,
            "error": None,
            "judge_parse_error": bool(traj.judge_parse_error),
            "judge_reasoning": traj.judge_reasoning,
            "tool_calls": count_tool_calls(traj),
        }
    except Exception as exc:  # keep paired failures visible instead of dropping rows
        return {
            "scenario_id": scenario.id,
            "question": scenario.question,
            "reference": scenario.answer,
            "trial": trial,
            "model_label": model_label,
            "model_ref": model_ref,
            "answer": "",
            "correct": 0.0,
            "empty_answer": True,
            "incomplete": True,
            "exception": True,
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "judge_parse_error": False,
            "judge_reasoning": None,
            "tool_calls": 0,
        }


async def main() -> None:
    from email_agent.data import load_scenarios

    if TRIALS <= 0:
        raise SystemExit("TRIALS must be positive")
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is required")
    judge_model = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
    if judge_model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the configured correctness judge")

    control = parse_model_ref(CONTROL_MODEL_REF)
    treatment = parse_model_ref(TREATMENT_MODEL_REF)
    if (control.entity, control.project, control.collection) != (
        treatment.entity, treatment.project, treatment.collection
    ):
        raise SystemExit("Control and treatment must be checkpoints from the same model collection")

    all_scenarios = load_scenarios(split="test", limit=None, max_messages=1, shuffle=False)
    scenarios = select_scenarios(all_scenarios, limit=N_VALIDATION, seed=EVAL_SEED)
    scenario_ids = [int(s.id) for s in scenarios]
    commit = git_commit()
    config = {
        "experiment": "paired-checkpoint-evaluation",
        "control_model_ref": CONTROL_MODEL_REF,
        "treatment_model_ref": TREATMENT_MODEL_REF,
        "model_collection": control.collection,
        "control_checkpoint": control.alias,
        "treatment_checkpoint": treatment.alias,
        "n_validation": N_VALIDATION,
        "trials_per_scenario": TRIALS,
        "eval_seed": EVAL_SEED,
        "temperature": TEMPERATURE,
        "judge_model": judge_model,
        "scenario_ids": scenario_ids,
        "dataset": "corbt/enron_emails_sample_questions:test",
        "max_messages": 1,
        "code_commit": commit,
    }
    thread_id = os.environ.get("WANDB_ARIA_THREAD_ID")
    turn_id = os.environ.get("WANDB_ARIA_TURN_ID")
    if thread_id or turn_id:
        config["_wb_agent"] = {"thread_id": thread_id, "turn_id": turn_id}

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"paired-{control.alias}-vs-{treatment.alias}",
        job_type="eval",
        config=config,
    )
    weave.init(f"{ENTITY}/{PROJECT}")

    base_model = os.environ.get("BASE_MODEL")
    control_model_object = CheckpointModel(
        model_ref=CONTROL_MODEL_REF,
        checkpoint_step=control.alias,
        base_model=base_model,
        code_commit=commit,
    )
    treatment_model_object = CheckpointModel(
        model_ref=TREATMENT_MODEL_REF,
        checkpoint_step=treatment.alias,
        base_model=base_model,
        code_commit=commit,
    )
    control_model_object_ref = weave.publish(
        control_model_object, name=f"{control.collection}-{control.alias}"
    )
    treatment_model_object_ref = weave.publish(
        treatment_model_object, name=f"{treatment.collection}-{treatment.alias}"
    )
    run.summary["weave_control_model_ref"] = str(control_model_object_ref)
    run.summary["weave_treatment_model_ref"] = str(treatment_model_object_ref)

    # Record exact checkpoint lineage. The references remain pinned in config too.
    run.use_artifact(control.artifact_path)
    run.use_artifact(treatment.artifact_path)

    manifest_path = Path("paired_eval_manifest.json")
    manifest_path.write_text(json.dumps({
        "dataset": config["dataset"],
        "max_messages": 1,
        "seed": EVAL_SEED,
        "scenario_ids": scenario_ids,
    }, indent=2))
    manifest_artifact = wandb.Artifact("email-agent-paired-eval-manifest", type="dataset")
    manifest_artifact.add_file(str(manifest_path))
    run.log_artifact(manifest_artifact, aliases=[f"seed-{EVAL_SEED}-n-{N_VALIDATION}"])

    timeout = httpx.Timeout(1200, connect=30)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        client = AsyncOpenAI(
            base_url=INFERENCE_BASE,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        rows = []
        # Interleave checkpoints within each paired trial to reduce temporal bias.
        for scenario in scenarios:
            for trial in range(TRIALS):
                paired = await asyncio.gather(
                    evaluate_one(
                        scenario=scenario, trial=trial, model_label="step0",
                        model_ref=CONTROL_MODEL_REF, client=client,
                    ),
                    evaluate_one(
                        scenario=scenario, trial=trial, model_label="step2",
                        model_ref=TREATMENT_MODEL_REF, client=client,
                    ),
                )
                rows.extend(paired)

    summary = paired_summary(rows)
    scenario_rows = paired_scenario_rows(rows)
    control_rows = [row for row in rows if row["model_label"] == "step0"]
    treatment_rows = [row for row in rows if row["model_label"] == "step2"]
    summary["control_judge_parse_failure_rate"] = sum(
        int(row["judge_parse_error"]) for row in control_rows
    ) / len(control_rows)
    summary["treatment_judge_parse_failure_rate"] = sum(
        int(row["judge_parse_error"]) for row in treatment_rows
    ) / len(treatment_rows)
    columns = [
        "scenario_id", "question", "reference", "trial", "model_label",
        "model_ref", "answer", "correct", "empty_answer", "incomplete",
        "exception", "error", "judge_parse_error", "judge_reasoning", "tool_calls",
    ]
    table = wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in rows])
    scenario_columns = list(scenario_rows[0])
    scenario_table = wandb.Table(
        columns=scenario_columns,
        data=[[row.get(c) for c in scenario_columns] for row in scenario_rows],
    )
    run.log({"paired_eval": table, "paired_scenario_deltas": scenario_table, **summary})
    for key, value in summary.items():
        run.summary[key] = value

    no_failure_regression = all([
        float(summary["treatment_empty_answer_rate"]) <= float(summary["control_empty_answer_rate"]),
        float(summary["treatment_exception_rate"]) <= float(summary["control_exception_rate"]),
        float(summary["treatment_incomplete_rate"]) <= float(summary["control_incomplete_rate"]),
        float(summary["treatment_judge_parse_failure_rate"]) <= float(summary["control_judge_parse_failure_rate"]),
    ])
    proceed = bool(
        float(summary["accuracy_delta"]) >= 0.05
        and no_failure_regression
    )
    run.summary["no_failure_regression"] = no_failure_regression
    run.summary["proceed_to_longer_rl"] = proceed
    run.finish()

    print(json.dumps(summary, indent=2))
    print(f"Proceed to longer RL: {proceed}")


if __name__ == "__main__":
    asyncio.run(main())
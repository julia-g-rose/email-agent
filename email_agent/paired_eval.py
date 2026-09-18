"""Pure helpers for reproducible paired checkpoint evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
from typing import Iterable, Sequence


_MODEL_REF_RE = re.compile(
    r"^wandb-artifact:///([^/]+)/([^/]+)/([^:]+):([^:]+)$"
)


@dataclass(frozen=True)
class ModelRef:
    entity: str
    project: str
    collection: str
    alias: str

    @property
    def artifact_path(self) -> str:
        return f"{self.entity}/{self.project}/{self.collection}:{self.alias}"


def parse_model_ref(value: str) -> ModelRef:
    """Parse a Serverless Training artifact model reference."""
    match = _MODEL_REF_RE.fullmatch(value)
    if not match:
        raise ValueError(
            "Expected wandb-artifact:///ENTITY/PROJECT/COLLECTION:ALIAS, "
            f"got {value!r}"
        )
    return ModelRef(*match.groups())


def select_scenarios(
    scenarios: Sequence,
    *,
    limit: int,
    seed: int,
    explicit_ids: Sequence[int] | None = None,
) -> list:
    """Select one deterministic manifest, or restore an explicit manifest."""
    by_id = {int(s.id): s for s in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError("Scenario IDs must be unique")

    if explicit_ids is not None:
        missing = [int(i) for i in explicit_ids if int(i) not in by_id]
        if missing:
            raise ValueError(f"Scenario IDs not found: {missing}")
        return [by_id[int(i)] for i in explicit_ids]

    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(scenarios):
        raise ValueError(f"Requested {limit} scenarios, but only {len(scenarios)} are available")
    return random.Random(seed).sample(list(scenarios), k=limit)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_summary(
    rows: Iterable[dict],
    *,
    control_label: str = "step0",
    treatment_label: str = "step2",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, float | int]:
    """Summarize paired rows and bootstrap uncertainty by scenario.

    Repeated trials for one question are averaged before resampling, so they are
    not incorrectly treated as independent examples.
    """
    rows = list(rows)
    by_model: dict[str, list[dict]] = defaultdict(list)
    by_scenario: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = str(row["model_label"])
        scenario_id = int(row["scenario_id"])
        correct = float(row["correct"])
        by_model[label].append(row)
        by_scenario[scenario_id][label].append(correct)

    if control_label not in by_model or treatment_label not in by_model:
        raise ValueError("Both control and treatment rows are required")

    paired_deltas = []
    for scenario_id, models in by_scenario.items():
        if control_label not in models or treatment_label not in models:
            raise ValueError(f"Scenario {scenario_id} is missing a paired model result")
        control = sum(models[control_label]) / len(models[control_label])
        treatment = sum(models[treatment_label]) / len(models[treatment_label])
        paired_deltas.append(treatment - control)

    def rate(label: str, key: str) -> float:
        values = by_model[label]
        return sum(float(bool(row.get(key, False))) for row in values) / len(values)

    control_accuracy = sum(float(r["correct"]) for r in by_model[control_label]) / len(by_model[control_label])
    treatment_accuracy = sum(float(r["correct"]) for r in by_model[treatment_label]) / len(by_model[treatment_label])
    delta = sum(paired_deltas) / len(paired_deltas)

    rng = random.Random(bootstrap_seed)
    boot = []
    if bootstrap_samples > 0:
        for _ in range(bootstrap_samples):
            sample = [rng.choice(paired_deltas) for _ in paired_deltas]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        ci_low = _percentile(boot, 0.025)
        ci_high = _percentile(boot, 0.975)
    else:
        ci_low = ci_high = delta

    return {
        "scenario_count": len(paired_deltas),
        "control_trials": len(by_model[control_label]),
        "treatment_trials": len(by_model[treatment_label]),
        "control_accuracy": control_accuracy,
        "treatment_accuracy": treatment_accuracy,
        "accuracy_delta": delta,
        "accuracy_delta_ci95_low": ci_low,
        "accuracy_delta_ci95_high": ci_high,
        "control_empty_answer_rate": rate(control_label, "empty_answer"),
        "treatment_empty_answer_rate": rate(treatment_label, "empty_answer"),
        "control_exception_rate": rate(control_label, "exception"),
        "treatment_exception_rate": rate(treatment_label, "exception"),
        "control_incomplete_rate": rate(control_label, "incomplete"),
        "treatment_incomplete_rate": rate(treatment_label, "incomplete"),
    }


def paired_scenario_rows(
    rows: Iterable[dict],
    *,
    control_label: str = "step0",
    treatment_label: str = "step2",
) -> list[dict]:
    """Aggregate repeated paired trials into one diagnostic row per scenario."""
    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["scenario_id"])][str(row["model_label"])].append(row)

    result = []
    for scenario_id in sorted(grouped):
        models = grouped[scenario_id]
        if control_label not in models or treatment_label not in models:
            raise ValueError(f"Scenario {scenario_id} is missing a paired model result")
        control = models[control_label]
        treatment = models[treatment_label]

        def mean(values: list[dict], key: str) -> float:
            return sum(float(v.get(key, 0) or 0) for v in values) / len(values)

        result.append({
            "scenario_id": scenario_id,
            "question": (control[0].get("question") or treatment[0].get("question")),
            "control_accuracy": mean(control, "correct"),
            "treatment_accuracy": mean(treatment, "correct"),
            "accuracy_delta": mean(treatment, "correct") - mean(control, "correct"),
            "control_empty_answer_rate": mean(control, "empty_answer"),
            "treatment_empty_answer_rate": mean(treatment, "empty_answer"),
            "control_exception_rate": mean(control, "exception"),
            "treatment_exception_rate": mean(treatment, "exception"),
            "control_mean_tool_calls": mean(control, "tool_calls"),
            "treatment_mean_tool_calls": mean(treatment, "tool_calls"),
        })
    return result
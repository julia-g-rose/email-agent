"""Reference-aware hybrid reward for the email-search RL experiment."""

from __future__ import annotations

import os

import art
from art.rewards import ruler_score_group as _ruler_score_group

# LLM judge for RULER relative scoring (OpenAI-compatible model string).
RULER_MODEL = os.environ.get("RULER_MODEL", "openai/gpt-4o")
CORRECT_WEIGHT = float(os.environ.get("REWARD_CORRECT_WEIGHT", "0.70"))
SOURCE_WEIGHT = float(os.environ.get("REWARD_SOURCE_WEIGHT", "0.20"))
RULER_WEIGHT = float(os.environ.get("REWARD_RULER_WEIGHT", "0.10"))

__all__ = [
    "hybrid_score_group", "RULER_MODEL", "CORRECT_WEIGHT", "SOURCE_WEIGHT",
    "RULER_WEIGHT", "source_f1", "sanity_check",
]


def source_f1(predicted: list[str], expected: list[str]) -> float:
    """Set-based F1 for cited versus reference email message IDs."""
    pred, gold = set(predicted), set(expected)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


async def hybrid_score_group(group: art.TrajectoryGroup, model: str, debug: bool = False):
    """Blend strict reference correctness, citation recall, and relative RULER quality."""
    judged = await _ruler_score_group(group, model, debug=debug)
    for traj in judged.trajectories:
        ruler = max(0.0, min(1.0, float(traj.reward)))
        correct = float(traj.metrics.get("correct", 0.0))
        expected = [s for s in str(traj.metadata.get("reference_message_ids", "")).split(",") if s]
        predicted = list(traj.final_answer.source_ids) if traj.final_answer else []
        citation_f1 = source_f1(predicted, expected)
        traj.reward = (
            CORRECT_WEIGHT * correct
            + SOURCE_WEIGHT * citation_f1
            + RULER_WEIGHT * ruler
        )
        traj.metrics.update({
            "reward/correct_component": correct,
            "reward/source_f1_component": citation_f1,
            "reward/ruler_component": ruler,
            "reward/hybrid": traj.reward,
        })
    return judged


async def sanity_check() -> None:
    """Rank three answers (good / mediocre / bad) to confirm RULER is wired up."""
    base = [
        {"role": "system", "content": "You count numbers using numeric symbols."},
        {"role": "user", "content": "Count to 10."},
    ]
    group = art.TrajectoryGroup(
        trajectories=[
            art.Trajectory(messages_and_choices=[*base, {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}], reward=0),
            art.Trajectory(messages_and_choices=[*base, {"role": "assistant", "content": "one, two, three, four, five, six, seven, eight, nine, ten"}], reward=0),
            art.Trajectory(messages_and_choices=[*base, {"role": "assistant", "content": "a, b, c, d, e, f, g, h, i, j"}], reward=0),
        ]
    )
    judged = await _ruler_score_group(group, RULER_MODEL, debug=True)
    assert judged is not None
    for rank, traj in enumerate(sorted(judged.trajectories, key=lambda t: t.reward, reverse=True), 1):
        print(f"Rank {rank}: score {traj.reward:.3f} — {traj.messages()[-1]['content'][:40]}...")

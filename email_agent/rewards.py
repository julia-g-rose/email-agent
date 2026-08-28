"""RULER reward: relative scoring of a trajectory group by an LLM judge.

RL here uses RULER (from ART) instead of a hand-written reward — it ranks the
trajectories in a group against each other, so we don't have to design a scalar
reward for "good email research". Re-exported here plus a small sanity check.
"""

from __future__ import annotations

import os

import art
from art.rewards import ruler_score_group

# LLM judge for RULER relative scoring (OpenAI-compatible model string).
RULER_MODEL = os.environ.get("RULER_MODEL", "openai/gpt-4o")

__all__ = ["ruler_score_group", "RULER_MODEL", "sanity_check"]


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
    judged = await ruler_score_group(group, RULER_MODEL, debug=True)
    assert judged is not None
    for rank, traj in enumerate(sorted(judged.trajectories, key=lambda t: t.reward, reverse=True), 1):
        print(f"Rank {rank}: score {traj.reward:.3f} — {traj.messages()[-1]['content'][:40]}...")

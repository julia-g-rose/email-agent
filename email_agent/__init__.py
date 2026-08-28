"""Email-search agent (ART·E-style) — RL fine-tuning to match a closed-model baseline."""

from email_agent.agent import (
    EmailScenario,
    ProjectTrajectory,
    judge_correctness,
    rollout,
    run_agent,
)
from email_agent.data import (
    Email,
    FinalAnswer,
    Scenario,
    SearchResult,
    load_scenarios,
    read_email,
    search_emails,
)

__all__ = [
    "Email",
    "EmailScenario",
    "FinalAnswer",
    "ProjectTrajectory",
    "Scenario",
    "SearchResult",
    "judge_correctness",
    "load_scenarios",
    "read_email",
    "rollout",
    "run_agent",
    "search_emails",
]

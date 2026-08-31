# email-agent

An email-research agent that answers questions about a user's inbox by searching and
reading their mail. It's an ART·E-style agent over the Enron email corpus: given a
question, it uses `search_inbox` / `read_email` tools across up to a few turns and
returns an answer with source message IDs.

**Today it runs on a closed model (OpenAI).** The goal of this repo is to **migrate it
to an open-source model, fine-tuned with reinforcement learning to reach parity** with
the closed baseline — at a fraction of the inference cost.

## How it works

| Piece | File |
|---|---|
| Email environment — Enron dataset → SQLite + FTS5, `search_emails` / `read_email`, scenarios | `email_agent/data.py` |
| The agent — multi-turn tool-use loop + LLM correctness judge (`run_agent`, `rollout`) | `email_agent/agent.py` |
| Reward — 70% exact correctness, 20% cited-source F1, 10% RULER | `email_agent/rewards.py` |
| **Baseline** — run the agent on the closed OpenAI model, measure accuracy | `scripts/baseline.py` |
| **Train** — RL fine-tune an open-source base with ART + RULER on W&B Serverless | `scripts/train.py` |
| **Evaluate** — score a fine-tuned checkpoint on the held-out set | `scripts/evaluate.py` |

The `run_agent` loop is model-agnostic (any OpenAI-compatible endpoint), so the same
agent is used for the baseline, RL rollouts, and evaluation.

## Setup

```bash
uv venv && uv pip install -e .
cp .env.example .env    # fill in OPENAI_API_KEY and WANDB_API_KEY
```

The first run downloads the Enron dataset (`corbt/enron-emails`) and builds
`enron_emails.db` automatically (a few minutes, one time).

## The migration, step by step

```bash
# 1. Baseline: how well does the agent do on the closed OpenAI model?
OPENAI_API_KEY=... uv run scripts/baseline.py
#   -> prints the accuracy to beat (e.g. "Accuracy: 85.0%")

# 2. Fine-tune an open-source model with RL (ART + RULER) on W&B Serverless Training
OPENAI_API_KEY=... WANDB_API_KEY=... uv run scripts/train.py
#   -> trains Qwen (base_model) to research the inbox; logs to W&B; stores checkpoints

# 3. Evaluate the fine-tuned checkpoint on the same held-out set
MODEL_REF="wandb-artifact:///ENTITY/PROJECT/email-agent-001:step60" \
  OPENAI_API_KEY=... WANDB_API_KEY=... uv run scripts/evaluate.py
#   -> compare its accuracy to the baseline: did the open-source model reach parity?
```

Config knobs (env): `BASELINE_MODEL`, `BASE_MODEL`, `JUDGE_MODEL`, `RULER_MODEL`,
`N_VALIDATION`, `MAX_TURNS` — see `.env.example`.

## Recommended next experiment: hybrid reward v1

The prior `email-agent-rl-v2` run optimized RULER alone: training reward rose while
held-out correctness plateaued at 19/30. This trial makes the target metric dominant
while retaining a citation-grounding signal and a small relative-quality term.

```bash
WANDB_ENTITY=wb-agent-team WANDB_PROJECT=email-agent \
MODEL_NAME=email-agent-14b-hybrid-reward-v1 \
BASE_MODEL=OpenPipe/Qwen3-14B-Instruct \
REWARD_CORRECT_WEIGHT=0.70 REWARD_SOURCE_WEIGHT=0.20 REWARD_RULER_WEIGHT=0.10 \
VAL_LIMIT=30 VAL_INTERVAL=5 EARLY_STOP_PATIENCE=2 MAX_STEPS=20 \
uv run scripts/train.py
```

**Primary metric:** `eval/trained_accuracy` on the fixed 30-question, temperature-0
held-out set. **Success:** at least 22/30 (parity with gpt-4.1-mini). **Promising:**
at least 21/30 with no drop in `eval/source_f1`; otherwise revise the reward/data
rather than extending training. The run also logs exact correct/sample counts and
expected versus predicted source IDs so failures can be audited row by row.

## Reference

Faithfully reconstructed from OpenPipe's ART·E example
(`notebooks/art-e-reference.ipynb`), split into a runnable package with a closed-model
baseline added so the OpenAI → open-source migration can be measured end to end.

---
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - sre
  - agent-evaluation
---

# Incident Response Environment

> **An OpenEnv RL environment that turns SRE on-call triage into a measurable benchmark.**
> Agents receive a flood of alerts from a broken production system, then must
> diagnose root causes through a service dependency graph and apply targeted
> fixes — exactly what a human on-call engineer does at 3 AM.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-spec-green.svg)](https://meta-pytorch.org/OpenEnv/)
[![Tests](https://img.shields.io/badge/tests-75%20passing-brightgreen.svg)](#testing)

---

## Why this matters

Incident response is one of the highest-leverage operational tasks at every
software company on earth — and one of the worst things to do at 3 AM. The
numbers from public industry data:

| Metric | Value | Source |
|---|---|---|
| Average annual cost of incident management per company | **$1.5M** | Gartner |
| Mean time to resolve a production incident | **69 minutes** | PagerDuty State of Digital Operations |
| Companies running 24/7 on-call rotations | **>90%** of mid+ tech orgs | DevOps Research |

An RL environment that captures the **dependency tracing**, **alert correlation**,
and **prioritisation under pressure** of real incident triage gives the
agent/RL community a realistic, high-value benchmark — not a toy game, not a
synthetic puzzle.

---

## How it works in 30 seconds

```
                     ALERTS                      DEPENDENCY GRAPH
                  ┌─────────┐                  ┌──────────────────┐
   reset()  ───▶  │ 14 fire │  ───▶  agent ─▶  │ trace upstream   │
                  └─────────┘                  └──────────────────┘
                                                        │
                       ┌────────────────────────────────┘
                       ▼
                ┌─────────────┐         ┌─────────────┐
   diagnose ──▶ │ root cause? │ ──no──▶ │  symptom    │ ──▶ -1 rubric
                └─────────────┘         └─────────────┘
                       │ yes
                       ▼
                ┌─────────────┐         ┌──────────────────┐
      fix   ──▶ │ propagate   │ ──────▶ │ 8 services heal  │
                │ recovery    │         │ via cascade      │
                └─────────────┘         └──────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ all healthy?    │ ──yes─▶ episode complete, score = passed/total rubrics
              └─────────────────┘
```

When a root-cause service breaks, downstream services degrade or fail through
the dependency graph. Fixing the root cause **automatically cascades recovery**
through dependents — but the agent has to find the right one first, and there
may be multiple simultaneous root causes hidden in the alert noise.

---

## The three tasks

| | Easy | Medium | Hard |
|---|---|---|---|
| **ID** | `single_service_failure` | `multi_service_correlation` | `cascading_outage` |
| **Services** | 3 | 6 | **13** |
| **Alerts** | 3 | 5 | **14** |
| **Root causes** | 1 | 1 | **3** |
| **Max steps** | 15 | 25 | 50 |
| **Rubrics** | 7 | 7 | **13** |
| **Hardening techniques applied** | 3 / 8 | 5 / 8 | **8 / 8** |

The hard task uses **all 8 webarena-style hardening techniques** —
exploratory discovery, indirect references, chained operations, bulk-conditional
treatment, cross-feature actions, disambiguation, non-obvious paths, and
state-dependent logic. The third root cause (`session_store`, a slow memory
leak) is buried among 14 alerts at medium severity, and is only discoverable
**after** fixing the first two — pure chained-operations reasoning.

---

## Score variance proves difficulty progression

This is the strongest evidence that the env genuinely discriminates between
agents. Same env, same prompt, same Docker image — only the model changes:

| Model | Easy | Medium | Hard | Notes |
|---|---|---|---|---|
| **GPT-5.4** (frontier) | **1.000** | **1.000** | **1.000** | Solves hard in 13 steps, finds all 3 root causes |
| **GPT-4o-mini** (mid-tier) | 1.000 | 0.857 | **0.538** | Loops on hard, never discovers `session_store` |
| **Qwen 2.5 72B** (open) | 1.000 | 0.571 | ~0.4 | Misses critical-alert acknowledgement rubric |
| Random / no-action | 0.143 | 0.143 | 0.077 | Vacuous "no incorrect fixes" rubric only |

**Hard task is genuinely hard for non-frontier models.** GPT-4o-mini fixes
`primary_db` and `message_queue`, then enters a loop because 5 services are
still degraded but it can't figure out why — that's the chained-operations
technique working as designed. GPT-5.4 reasons through the dependency graph
and finds the third root cause.

---

## Action and observation space

```python
class IncidentAction(Action):
    action_type: Literal["acknowledge", "diagnose", "fix", "escalate", "check_status"]
    target_service: str
```

| Action | Effect | Strategic role |
|---|---|---|
| `acknowledge` | Marks an alert as seen | Required for **critical** alerts (rubric) |
| `diagnose` | Reveals root-cause hint for the service | **Required before fixing** (rubric) |
| `fix` | Restores a service if it's a root cause | **Penalised on symptoms** (rubric) |
| `escalate` | No-op signal action | Training signal only |
| `check_status` | Health dashboard | Orientation; safe fallback |

```python
class IncidentObservation(Observation):
    alerts: List[Dict]                # severity, service, message, acknowledged
    services: Dict[str, str]          # service_name -> "healthy" | "degraded" | "down"
    dependencies: Dict[str, List[str]] # the dependency graph
    diagnostic_results: Dict[str, str] # what diagnose() has revealed so far
    rubric_results: List[Dict]        # populated at episode end
    message: str                      # human-readable feedback
    resolved_count: int               # services healthy / total
    total_services: int
    step_number: int
    max_steps: int
```

---

## Binary rubric grading (Scaler / webarena pattern)

Each rubric is **binary** (0 or 1). Final score = `passed / total`, always in
`[0.0, 1.0]`. Reward is returned at episode end. Per-step rewards are `None`
during the episode for clean training signal.

**Per-root-cause rubrics** (scale with task complexity):

| Rubric | Pass condition |
|---|---|
| `eval_<root>_diagnosed` | Agent ran `diagnose` on the root cause service |
| `eval_<root>_fixed` | Agent ran `fix` on the root cause service |
| `eval_<root>_diagnosed_before_fix` | The diagnose happened **before** the fix (proper investigation) |

**Global rubrics** (always present):

| Rubric | Pass condition |
|---|---|
| `eval_all_services_restored` | Every service is `healthy` at episode end |
| `eval_no_incorrect_fixes` | Agent never fixed a non-root-cause service |
| `eval_critical_alerts_acknowledged` | All `critical` severity alerts were acknowledged |
| `eval_step_efficiency` | Episode resolved within 60% of max steps |

**No grader ever returns the same score for every agent** — verified across
five different models with five different scores (the Phase 2 anti-DQ check).

---

## Quick start

### Run baseline inference (matches official spec format)

```bash
# Spec env vars (per Scaler hackathon):
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export LOCAL_IMAGE_NAME="incident-response-env:latest"

# Build, then run — inference.py spawns the container via from_docker_image()
docker build -t incident-response-env .
python inference.py
```

Output is in the **mandatory** `[START] / [STEP] / [END]` log format:

```
[START] task=single_service_failure env=incident_response_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=acknowledge(database) reward=0.00 done=false error=null
[STEP] step=2 action=diagnose(database) reward=0.00 done=false error=null
[STEP] step=3 action=fix(database) reward=1.00 done=true error=null
[END] success=true steps=3 score=1.000 rewards=0.00,0.00,1.00
```

### Connect from Python

```python
from incident_response_env.client import IncidentResponseEnv
from incident_response_env.models import IncidentAction

# Pull and run from HuggingFace (judge mode)
env = await IncidentResponseEnv.from_env("mohit-1710/incident-response-env")

# Or hit the live HF Space directly
env = IncidentResponseEnv(base_url="wss://mohit-1710-incident-response-env.hf.space")

# Episode loop
result = await env.reset(task_name="cascading_outage")
result = await env.step(IncidentAction(action_type="diagnose", target_service="primary_db"))
result = await env.step(IncidentAction(action_type="fix", target_service="primary_db"))

print(result.reward, result.observation.rubric_results)
```

### Try it interactively

Open the [`/web` playground](/web) on this Space and click **Reset → Step**.
The default scenario is the easy task; use the API for medium/hard, or use the
buttons in the playground panel.

---

## What makes this submission different

1. **Real domain, not a toy.** SRE incident triage is something every tech
   company does daily. The actions, observations, and rewards map 1-to-1 to
   the actual on-call workflow — not a contrived game.

2. **Genuinely challenges frontier models.** GPT-4o-mini scores 0.54 on the
   hard task. GPT-5.4 scores 1.000. That's exactly the kind of variance the
   hackathon Phase 2 grading looks for.

3. **All 8 hardening techniques applied** (verified per task in
   `scenarios.py::techniques`). The hard task uses every technique from the
   webarena task-hardening playbook, including chained operations and
   non-obvious paths.

4. **Binary rubric grading** (the Scaler pattern). Each criterion is 0 or 1,
   final score is the average — deterministic, reproducible, and impossible
   for an agent to game by accumulating fractional rewards.

5. **75 unit tests + clean-room validator pass**. Every grader proven correct
   on initial state (all 0) and golden-path state (all 1). Pre-validation
   script passes 3/3.

6. **Lightweight infra**. Docker container uses **123 MiB** of memory and ~6%
   of one CPU on the standard 2 vCPU / 8 GB hackathon box. Inference for all
   3 tasks completes in **40 seconds** with a strong model.

---

## Repository layout

```
incident_response_env/
├── models.py              # IncidentAction, IncidentObservation, IncidentState
├── scenarios.py           # 3 task definitions: services, alerts, dependencies, techniques
├── client.py              # WebSocket client (judges connect through this)
├── server/
│   ├── app.py             # FastAPI wiring via create_app()
│   └── environment.py     # Core simulation: cascading recovery, binary rubric grading
├── tests/                 # 75 tests: grader sanity, edge cases, scenario integrity
├── inference.py           # Spec-compliant LLM agent (from_docker_image, [START]/[STEP]/[END])
├── pre_validation_script.sh  # Official pre-submission validator
├── sample_interface_script.py # Reference sample from the hackathon dashboard
├── openenv.yaml           # Environment manifest
├── Dockerfile             # Port 8000, ENABLE_WEB_INTERFACE=true
├── pyproject.toml         # [project.scripts] server = ... for `uv run server`
├── uv.lock                # Required by openenv validate
└── README.md
```

---

## Testing

```bash
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -v
```

**75 tests** covering:

- **Grader sanity** — `reset() → all rubrics 0`, `golden_path() → all rubrics 1`,
  partial progress produces intermediate scores, symptom fixes fail the
  `no_incorrect_fixes` rubric.
- **Edge cases** — invalid actions, repeated actions, episode lifecycle,
  max-steps termination, multiple resets clear state.
- **Scenario integrity** — dependency graph is a DAG, every root cause is a
  real service, every dependency reference resolves, difficulty progression is
  monotonic.

```bash
# Pre-submission validator (3/3 must pass)
bash pre_validation_script.sh https://mohit-1710-incident-response-env.hf.space .
```

---

## Built by Team Atomic

- **Mohit Kumar** (lead) — `mohitkumar001700@gmail.com`
- **Krishna Faujdar** — `fkrishna1729@gmail.com`

For the **Meta × PyTorch × Scaler OpenEnv Hackathon Round 1** (April 2026).

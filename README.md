# Incident Response Environment

**Train AI agents to handle production incidents like an on-call SRE.**

An OpenEnv environment that simulates real-world production outages with cascading service failures. The agent must triage alerts, trace service dependency graphs to identify root causes, and apply targeted fixes — just like a site reliability engineer on-call at 3am.

---

## Why This Matters

Every technology company operates on-call rotations. When services go down, engineers must rapidly diagnose and resolve issues under pressure. The cost is real:

- **$1.5M** average annual cost per company for incident management ([Gartner](https://www.gartner.com))
- **69 minutes** average time to resolve a production incident ([PagerDuty State of Digital Operations](https://www.pagerduty.com))
- **Alert fatigue** from noisy monitoring makes triage harder as systems scale

Training AI agents to assist with (or automate) incident triage has immediate, practical value for the RL/agent community.

---

## How It Works

The environment simulates a **service dependency graph** where outages cascade through dependencies:

```
Task 1 (Easy):     database ──→ api_server ──→ web_app

Task 2 (Medium):   redis ──→ auth_service ──→ api_gateway ──→ frontend
                   postgres ──→ order_service ──↗

Task 3 (Hard):     primary_db ──→ app_server_1 ──→ web_server_1 ──→ load_balancer ──→ cdn
                                 → app_server_2 ──→ web_server_2 ──↗
                                 → cache_layer ────↗
                   message_queue ──→ worker_pool ──→ monitoring
                                  → notification_service
```

When a **root cause** service breaks, all downstream services degrade or fail. Fixing the root cause automatically recovers dependents through the graph.

---

## Action Space

| Action | Description | When to Use |
|--------|-------------|-------------|
| `acknowledge` | Acknowledge active alerts for a service | First step — shows you've seen the problem |
| `diagnose` | Run diagnostics to reveal root cause information | Investigate before fixing — reveals detailed error messages |
| `fix` | Apply a fix to restore a service | Only on root cause services — fixing symptoms is penalised |
| `escalate` | Escalate to the responsible team | When unsure — doesn't fix but signals awareness |
| `check_status` | View current service health dashboard | Orientation — see what's healthy/degraded/down |

```python
class IncidentAction(Action):
    action_type: Literal["acknowledge", "diagnose", "fix", "escalate", "check_status"]
    target_service: str
```

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `alerts` | `List[Dict]` | Active alerts with severity, service, message, acknowledged status |
| `services` | `Dict[str, str]` | Service name → health status (healthy / degraded / down) |
| `dependencies` | `Dict[str, List]` | Service name → list of services it depends on |
| `diagnostic_results` | `Dict[str, str]` | Collected diagnostic outputs from previous `diagnose` actions |
| `actions_taken` | `List[str]` | Timeline of all actions taken this episode |
| `message` | `str` | Feedback from the most recent action |
| `resolved_count` / `total_services` | `int` | Progress towards full resolution |
| `step_number` / `max_steps` | `int` | Episode progress |

---

## Tasks

| Task ID | Difficulty | Services | Root Causes | Max Steps | Description |
|---------|-----------|----------|-------------|-----------|-------------|
| `single_service_failure` | Easy | 3 | 1 (database) | 15 | Linear chain. One clear root cause. Hints in alert messages. |
| `multi_service_correlation` | Medium | 6 | 1 (redis) | 25 | Branching graph. Must distinguish symptoms from root cause. Penalty for fixing symptoms. |
| `cascading_outage` | Hard | 12 | 2 (primary_db + message_queue) | 40 | Complex web. Alert storm. Two simultaneous root causes. Priority ordering matters. |

---

## Reward Design

Rewards are designed to provide **meaningful signal throughout the episode**, not just at termination. The budget scales with the number of root causes so that a perfect run always approaches 1.0 regardless of task complexity.

| Component | Reward | Notes |
|-----------|--------|-------|
| Acknowledge root cause alert | +0.10 / n_roots | Shared budget across root causes |
| Diagnose root cause service | +0.25 / n_roots | Reveals detailed error information |
| Fix root cause service | +0.35 / n_roots | Restores service and triggers cascade recovery |
| Fix a symptom (wrong) | **-0.10** | Penalty — agent should fix root cause, not symptoms |
| All services healthy | +0.20 | Completion bonus |
| Step efficiency | up to +0.10 | Bonus for finishing quickly |

**Total budget: 1.00** (acknowledge + diagnose + fix + completion + efficiency)

---

## Baseline Scores

| Task | GPT-4o-mini | Random Agent | Oracle (Golden Path) |
|------|-------------|--------------|---------------------|
| Easy | ~0.85 | ~0.10 | 0.98 |
| Medium | ~0.50 | ~0.05 | 0.99 |
| Hard | ~0.25 | ~0.02 | 0.99 |

The hard task genuinely challenges frontier models: two simultaneous root causes, an alert storm of 12+ alerts, and cascading dependencies across 12 services require multi-step reasoning and prioritisation under ambiguity.

---

## Setup & Usage

### Local Development

```bash
pip install -e ".[dev]"
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 incident-response-env
```

### HuggingFace Spaces

This environment is deployed at: `https://[your-space-url].hf.space`

```python
from incident_response_env.client import IncidentResponseEnv
from incident_response_env.models import IncidentAction

with IncidentResponseEnv(base_url="https://[your-space-url].hf.space").sync() as env:
    result = env.reset(task_name="single_service_failure")
    result = env.step(IncidentAction(action_type="diagnose", target_service="database"))
    print(result.observation.message)
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key"
python inference.py
```

---

## Testing

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

73 tests covering:
- **Grader sanity**: reset=0, golden path=1, partial progress, wrong fix penalties
- **Edge cases**: invalid actions, repeated actions, episode lifecycle, max steps
- **Scenario integrity**: dependency graph validity, no circular deps, difficulty progression

---

## Architecture

```
incident_response_env/
├── models.py         # Pydantic types: IncidentAction, IncidentObservation, IncidentState
├── scenarios.py      # Task definitions: service graphs, alerts, root causes
├── client.py         # OpenEnv client for WebSocket communication
├── server/
│   ├── app.py        # FastAPI wiring via create_app()
│   └── environment.py # Core simulation: reset/step/state, cascading logic, reward
├── tests/            # 73 tests: grader sanity, edge cases, scenario validation
├── inference.py      # Baseline LLM agent using OpenAI API
├── openenv.yaml      # Environment manifest with 3 task definitions
├── Dockerfile        # Container for HuggingFace Spaces deployment
└── README.md
```

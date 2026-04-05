"""Baseline inference script for the Incident Response environment.

Uses the OpenAI API to run an LLM agent through all three incident
response tasks. Emits structured [START]/[STEP]/[END] logs for
automated evaluation.

Required environment variables:
    API_BASE_URL  — LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (default: gpt-4o-mini)
    HF_TOKEN      — HuggingFace / API key
"""

import json
import os
import sys

from openai import OpenAI

# Use the dual-import pattern so this works both locally and in Docker
try:
    from incident_response_env.client import IncidentResponseEnv
    from incident_response_env.models import IncidentAction
except ImportError:
    from client import IncidentResponseEnv
    from models import IncidentAction


# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_STEPS_PER_TASK = 50  # Safety cap to avoid runaway episodes

SYSTEM_PROMPT = """\
You are an SRE on-call responding to a production incident.

Analyze the alerts and service statuses to identify root causes and fix them.
Acknowledge critical alerts. Diagnose services before fixing them.
There may be multiple root causes. Downstream services recover automatically
when their upstream dependencies are restored.

Respond with a JSON object ONLY (no markdown, no explanation):
{"action_type": "<action>", "target_service": "<service_name>"}

Valid actions: acknowledge, diagnose, fix, escalate, check_status
"""

TASKS = ["single_service_failure", "multi_service_correlation", "cascading_outage"]


def build_user_prompt(obs) -> str:
    """Format the environment observation into a clear prompt for the LLM."""
    # Build alert summary
    alert_lines = []
    for a in obs.alerts:
        ack = "ACK" if a.get("acknowledged") else "NEW"
        alert_lines.append(
            f"  [{ack}] {a['severity'].upper():8s} | {a['service']:20s} | {a['message']}"
        )

    # Build service status table (raw — no hints about root causes)
    svc_lines = []
    for svc, status in sorted(obs.services.items()):
        deps = obs.dependencies.get(svc, [])
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        svc_lines.append(f"  {status:10s} | {svc}{dep_str}")

    # Build diagnostic summary
    diag_lines = []
    for svc, result in obs.diagnostic_results.items():
        diag_lines.append(f"  {svc}: {result}")

    prompt = f"""INCIDENT STATUS (Step {obs.step_number}/{obs.max_steps})
Services restored: {obs.resolved_count}/{obs.total_services}
Last action result: {obs.message}

ALERTS:
{chr(10).join(alert_lines) if alert_lines else "  No alerts"}

SERVICE STATUS:
{chr(10).join(svc_lines)}

DIAGNOSTICS COLLECTED:
{chr(10).join(diag_lines) if diag_lines else "  None yet — use 'diagnose' to investigate services"}

Choose your next action. Respond with JSON only: {{"action_type": "...", "target_service": "..."}}"""

    return prompt


def parse_llm_response(text: str) -> IncidentAction:
    """Parse LLM response into an IncidentAction, with fallback handling."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return IncidentAction(
            action_type=data.get("action_type", "check_status"),
            target_service=data.get("target_service", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: check_status is always safe
        return IncidentAction(action_type="check_status", target_service="")


def run_task(task_id: str, env_url: str, client: OpenAI) -> None:
    """Run a single task and emit structured logs."""
    print(f"[START]")

    with IncidentResponseEnv(base_url=env_url).sync() as env:
        result = env.reset(task_name=task_id)
        obs = result.observation
        done = result.done
        step_count = 0
        final_score = 0.0

        while not done and step_count < MAX_STEPS_PER_TASK:
            # Build prompt from current observation
            user_prompt = build_user_prompt(obs)

            # Query the LLM
            try:
                api_params = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_completion_tokens": 256,
                }
                response = client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                llm_text = content.strip() if content else ""
            except Exception as e:
                # Retry with max_tokens for older models
                try:
                    api_params.pop("max_completion_tokens", None)
                    api_params["max_tokens"] = 256
                    response = client.chat.completions.create(**api_params)
                    content = response.choices[0].message.content
                    llm_text = content.strip() if content else ""
                except Exception:
                    llm_text = '{"action_type": "check_status", "target_service": ""}'

            # Parse into action
            action = parse_llm_response(llm_text)

            # Execute step
            result = env.step(action)
            obs = result.observation
            done = result.done
            reward = result.reward or 0.0
            step_count += 1

            # Emit structured log
            log_entry = {
                "step": step_count,
                "action_type": action.action_type,
                "target_service": action.target_service,
                "reward": reward,
                "done": done,
                "resolved": obs.resolved_count,
                "total": obs.total_services,
                "message": obs.message[:120],
            }
            print(f"[STEP] {json.dumps(log_entry)}")

            if done:
                final_score = reward
                break

        if not done:
            final_score = reward

    print(f"[END] Final Score: {final_score:.4f}, Steps taken: {step_count}")


def run_inference() -> None:
    """Run all tasks sequentially with structured logging."""
    env_url = os.getenv("OPENENV_BASE_URL", "ws://127.0.0.1:7860")
    # Ensure WebSocket protocol
    env_url = env_url.replace("http://", "ws://").replace("https://", "wss://")

    # Initialise OpenAI client with hackathon-required env vars
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", HF_TOKEN or "dummy-key"),
        base_url=API_BASE_URL,
    )

    for idx, task_id in enumerate(TASKS):
        print(f"--- Running Task {idx + 1}/{len(TASKS)}: {task_id} ---")
        run_task(task_id, env_url, client)
        print()


if __name__ == "__main__":
    run_inference()

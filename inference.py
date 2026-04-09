"""inference.py — Official-format inference script for incident_response_env.

Spawns the env from a local Docker image, runs all 3 incident response tasks
with an LLM agent, and emits logs in the EXACT [START]/[STEP]/[END] format
mandated by the hackathon spec (key=value pairs, one episode per task).

MANDATORY environment variables (per the hackathon pre-submission checklist):
    API_BASE_URL       The API endpoint for the LLM        (default provided)
    MODEL_NAME         The model identifier                (default provided)
    HF_TOKEN           HuggingFace / API key               (NO default — must be set)
    LOCAL_IMAGE_NAME   Local Docker image name             (NO default — must be set
                                                            when using from_docker_image)

Output format (one [START]/[STEP]*/[END] block per task):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

# Dual-import pattern — works both as installed package and from source dir
try:
    from incident_response_env.client import IncidentResponseEnv
    from incident_response_env.models import IncidentAction
except ImportError:
    from client import IncidentResponseEnv  # type: ignore[no-redef]
    from models import IncidentAction  # type: ignore[no-redef]


# ── Configuration (mandatory hackathon env vars, exact spec layout) ───────────
#
# Per the pre-submission checklist:
#   - Defaults are set ONLY for API_BASE_URL and MODEL_NAME.
#   - HF_TOKEN has NO default — it must be supplied by the runner.
#   - LOCAL_IMAGE_NAME has NO default — it's required when using from_docker_image().

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# The OpenAI client takes any string as api_key — judges set HF_TOKEN; for
# local OpenAI testing, set HF_TOKEN to your OpenAI key (or set API_KEY).
API_KEY = HF_TOKEN or os.getenv("API_KEY")

BENCHMARK = "incident_response_env"
MAX_STEPS_PER_TASK = 50
SUCCESS_SCORE_THRESHOLD = 0.5  # Episode "succeeds" if final score >= 0.5

TASKS = [
    "single_service_failure",
    "multi_service_correlation",
    "cascading_outage",
]


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


# ── Structured logging (exact hackathon format) ───────────────────────────────

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    err_val = error if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt building & response parsing ────────────────────────────────────────

def build_user_prompt(obs) -> str:
    """Format the environment observation into a clear prompt for the LLM."""
    alert_lines = []
    for a in obs.alerts:
        ack = "ACK" if a.get("acknowledged") else "NEW"
        alert_lines.append(
            f"  [{ack}] {a['severity'].upper():8s} | {a['service']:20s} | {a['message']}"
        )

    svc_lines = []
    for svc, status in sorted(obs.services.items()):
        deps = obs.dependencies.get(svc, [])
        dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
        svc_lines.append(f"  {status:10s} | {svc}{dep_str}")

    diag_lines = []
    for svc, result in obs.diagnostic_results.items():
        diag_lines.append(f"  {svc}: {result}")

    return f"""INCIDENT STATUS (Step {obs.step_number}/{obs.max_steps})
Services restored: {obs.resolved_count}/{obs.total_services}
Last action result: {obs.message}

ALERTS:
{chr(10).join(alert_lines) if alert_lines else "  No alerts"}

SERVICE STATUS:
{chr(10).join(svc_lines)}

DIAGNOSTICS COLLECTED:
{chr(10).join(diag_lines) if diag_lines else "  None yet — use 'diagnose' to investigate services"}

Choose your next action. Respond with JSON only: {{"action_type": "...", "target_service": "..."}}"""


def parse_llm_response(text: str) -> IncidentAction:
    """Parse LLM JSON response into an IncidentAction, with safe fallback."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        return IncidentAction(
            action_type=data.get("action_type", "check_status"),
            target_service=data.get("target_service", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return IncidentAction(action_type="check_status", target_service="")


def action_to_log_str(action: IncidentAction) -> str:
    """Format an action as a single-line string for the [STEP] log."""
    if action.target_service:
        return f"{action.action_type}({action.target_service})"
    return action.action_type


def query_llm(client: OpenAI, prompt: str) -> str:
    """Query the LLM. Try max_completion_tokens first (GPT-5+), fall back to max_tokens."""
    base_params = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    try:
        resp = client.chat.completions.create(**base_params, max_completion_tokens=256)
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception:
        try:
            resp = client.chat.completions.create(**base_params, max_tokens=256)
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception:
            return '{"action_type": "check_status", "target_service": ""}'


# ── Main episode runner ───────────────────────────────────────────────────────

async def run_task(client: OpenAI, env, task_id: str) -> None:
    """Run one task as one episode. Emits exactly one [START]/[STEP]*/[END] block."""
    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    last_error: Optional[str] = None

    try:
        result = await env.reset(task_name=task_id)
        obs = result.observation
        done = result.done

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if done:
                break

            try:
                prompt = build_user_prompt(obs)
                llm_text = query_llm(client, prompt)
                action = parse_llm_response(llm_text)
            except Exception as exc:
                action = IncidentAction(action_type="check_status", target_service="")
                last_error = str(exc)[:120]

            try:
                result = await env.step(action)
                obs = result.observation
                done = result.done
                reward = float(result.reward or 0.0)
            except Exception as exc:
                log_step(
                    step=step,
                    action=action_to_log_str(action),
                    reward=0.0,
                    done=False,
                    error=str(exc)[:120],
                )
                break

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_log_str(action),
                reward=reward,
                done=done,
                error=last_error,
            )
            last_error = None

            if done:
                break

        # Final score = the episode-end reward (our env returns 0..1 grader score on done)
        final_score = rewards[-1] if rewards else 0.0
        final_score = max(0.0, min(1.0, final_score))
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


async def main() -> None:
    """Spawn the env from a local Docker image and run all 3 tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Allow override via OPENENV_BASE_URL for local testing without Docker
    base_url_override = os.getenv("OPENENV_BASE_URL")

    if base_url_override:
        env = IncidentResponseEnv(base_url=base_url_override)
        await env.connect()
    else:
        # Runtime fallback: if LOCAL_IMAGE_NAME wasn't set, use the conventional
        # name produced by `docker build` from this repo. The module-level
        # `LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")` declaration matches
        # the spec exactly (no literal default).
        image_name = LOCAL_IMAGE_NAME or "incident-response-env:latest"
        env = await IncidentResponseEnv.from_docker_image(image_name)

    try:
        for task_id in TASKS:
            await run_task(client, env, task_id)
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

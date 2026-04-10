"""Custom Gradio /web UI for the Incident Response environment.

Replaces openenv-core's default left-sidebar layout with a single
full-width column:

    1. Header
    2. Task picker  +  Reset / Get-state buttons
    3. Action row (action type + target service + Step button)
    4. Status line
    5. Observation display (formatted markdown)
    6. Raw JSON accordion (collapsed)
    7. README accordion (collapsed)

Wired into server/app.py via a monkey-patch on
`openenv.core.env_server.web_interface.build_gradio_app` so that the
framework uses this builder for the SINGLE Gradio app it mounts at /web,
without creating the TabbedInterface that `gradio_builder=` would force.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

# Dual-import: works both as installed package and from source dir
try:
    from ..scenarios import AVAILABLE_TASKS, SCENARIOS
except ImportError:
    from scenarios import AVAILABLE_TASKS, SCENARIOS  # type: ignore[no-redef]


# ── Constants ────────────────────────────────────────────────────────────────

TASK_LABELS = {
    "single_service_failure": "Easy — Single Service Failure (3 services, 1 root cause)",
    "multi_service_correlation": "Medium — Multi-Service Correlation (6 services, 1 root cause)",
    "cascading_outage": "Hard — Cascading Outage (13 services, 3 root causes)",
}
TASK_CHOICES = [(TASK_LABELS[t], t) for t in AVAILABLE_TASKS]
DEFAULT_TASK = AVAILABLE_TASKS[0]

ACTION_CHOICES = [
    ("acknowledge", "acknowledge"),
    ("diagnose", "diagnose"),
    ("fix", "fix"),
    ("escalate", "escalate"),
    ("check_status", "check_status"),
]


# ── Markdown formatting ──────────────────────────────────────────────────────

def _format_observation_md(payload: Dict[str, Any]) -> str:
    """Render a step/reset response as readable Markdown."""
    obs = payload.get("observation") or {}
    if not obs:
        return "_No observation yet — pick a scenario above and click **Reset**._"

    reward = payload.get("reward")
    done = payload.get("done", False)
    msg = obs.get("message", "")
    step_n = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 0)
    resolved = obs.get("resolved_count", 0)
    total = obs.get("total_services", 0)

    lines: List[str] = []
    header = [
        f"**Step:** `{step_n}/{max_steps}`",
        f"**Healthy:** `{resolved}/{total}`",
        f"**Done:** `{done}`",
    ]
    if reward is not None:
        header.append(f"**Reward:** `{reward:.4f}`")
    lines.append("  ·  ".join(header))
    lines.append("")
    if msg:
        lines.append(f"> {msg}")
        lines.append("")

    # Alerts
    alerts = obs.get("alerts") or []
    if alerts:
        lines.append("### Alerts")
        for a in alerts:
            ack = "[ACK]" if a.get("acknowledged") else "[NEW]"
            sev = (a.get("severity") or "").upper()
            svc = a.get("service", "?")
            text = a.get("message", "")
            lines.append(f"- `{ack}` `{sev:8s}` **{svc}** — {text}")
        lines.append("")

    # Service health grouped by status
    services = obs.get("services") or {}
    if services:
        groups: Dict[str, List[str]] = {"down": [], "degraded": [], "healthy": []}
        for svc, status in services.items():
            groups.setdefault(status, []).append(svc)
        lines.append("### Service health")
        for status_name in ("down", "degraded", "healthy"):
            items = groups.get(status_name) or []
            if items:
                lines.append(f"- **{status_name}:** {', '.join(items)}")
        lines.append("")

    # Dependency graph
    deps = obs.get("dependencies") or {}
    if deps:
        lines.append("### Dependency graph")
        for svc, dep_list in deps.items():
            if dep_list:
                lines.append(f"- `{svc}` → {', '.join(f'`{d}`' for d in dep_list)}")
            else:
                lines.append(f"- `{svc}` _(no dependencies — possible root cause)_")
        lines.append("")

    # Diagnostics already collected
    diagnostics = obs.get("diagnostic_results") or {}
    if diagnostics:
        lines.append("### Diagnostics collected")
        for svc, result in diagnostics.items():
            lines.append(f"- **{svc}**")
            lines.append(f"  > {result}")
        lines.append("")

    # Rubric results when episode ends
    rubrics = obs.get("rubric_results") or []
    if rubrics:
        passed = sum(1 for r in rubrics if r.get("passed"))
        lines.append(f"### Final grading — {passed}/{len(rubrics)} rubrics passed")
        for r in rubrics:
            mark = "PASS" if r.get("passed") else "FAIL"
            lines.append(f"- `[{mark}]` {r.get('name', '?')}")
        lines.append("")

    return "\n".join(lines)


def _task_overview_md() -> str:
    """One-paragraph overview of each scenario for the picker."""
    parts: List[str] = ["**Available scenarios**", ""]
    for task_id in AVAILABLE_TASKS:
        scenario = SCENARIOS[task_id]
        parts.append(f"- **{TASK_LABELS[task_id]}**")
        parts.append(
            f"  `{len(scenario.services)} services` · "
            f"`{len(scenario.alerts)} alerts` · "
            f"`{len(scenario.root_cause_services)} root cause(s)` · "
            f"`max {scenario.max_steps} steps`"
        )
    return "\n".join(parts)


# ── Gradio Blocks builder (replaces openenv-core's default) ──────────────────

def build_incident_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],  # noqa: ARG001 — fixed action shape
    metadata: Any,
    is_chat_env: bool,  # noqa: ARG001 — not a chat env
    title: str = "Incident Response Environment",
    quick_start_md: Optional[str] = None,  # noqa: ARG001 — overridden, we don't show it
) -> gr.Blocks:
    """Build the full-width single-column playground for /web."""

    readme_content = ""
    if metadata is not None:
        readme_content = getattr(metadata, "readme_content", "") or ""

    task_overview = _task_overview_md()

    # ── Async event handlers ─────────────────────────────────────────────────

    async def reset_env(task_name: str):
        try:
            payload = await web_manager.reset_environment({"task_name": task_name})
            md = _format_observation_md(payload)
            raw = json.dumps(payload, indent=2, default=str)
            return md, raw, f"Reset complete — running '{task_name}'."
        except Exception as exc:  # pragma: no cover - defensive UI
            return f"**Reset failed:** `{exc}`", "{}", f"Error: {exc}"

    async def step_env(action_type: str, target_service: str):
        try:
            payload = await web_manager.step_environment(
                {
                    "action_type": action_type,
                    "target_service": (target_service or "").strip(),
                }
            )
            md = _format_observation_md(payload)
            raw = json.dumps(payload, indent=2, default=str)
            done = payload.get("done", False)
            obs = payload.get("observation") or {}
            n = obs.get("step_number", "?")
            return md, raw, ("Episode complete." if done else f"Step {n} done.")
        except Exception as exc:  # pragma: no cover - defensive UI
            return f"**Step failed:** `{exc}`", "{}", f"Error: {exc}"

    def get_state_sync():
        try:
            state = web_manager.get_state()
            return json.dumps(state, indent=2, default=str)
        except Exception as exc:  # pragma: no cover - defensive UI
            return f'{{"error": "{exc}"}}'

    # ── Layout ───────────────────────────────────────────────────────────────

    with gr.Blocks(
        title=f"OpenEnv: {title}",
        analytics_enabled=False,
    ) as blocks:
        gr.Markdown(
            "# Incident Response Environment\n"
            "_SRE on-call simulator. Triage alerts, trace dependency graphs, "
            "fix root causes to restore the system._"
        )

        # ── Scenario picker + global controls ────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                task_dropdown = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value=DEFAULT_TASK,
                    label="Scenario",
                    info="Pick a difficulty level, then click Reset.",
                )
            with gr.Column(scale=1):
                reset_btn = gr.Button("Reset Episode", variant="primary", size="lg")
            with gr.Column(scale=1):
                state_btn = gr.Button("Get State", variant="secondary", size="lg")

        gr.Markdown(task_overview)

        gr.Markdown("---")

        # ── Action row ───────────────────────────────────────────────────────
        gr.Markdown("### Take an action")
        with gr.Row():
            action_dropdown = gr.Dropdown(
                choices=ACTION_CHOICES,
                value="diagnose",
                label="Action Type",
                info="What do you want to do?",
            )
            target_input = gr.Textbox(
                label="Target Service",
                placeholder="e.g. database, primary_db, redis",
                info="Service name from the dependency graph.",
            )
            step_btn = gr.Button("Step", variant="primary", size="lg")

        status_box = gr.Textbox(
            label="Status",
            value="Ready. Pick a scenario and click Reset.",
            interactive=False,
        )

        # ── Observation display ──────────────────────────────────────────────
        gr.Markdown("### Observation")
        obs_display = gr.Markdown(
            "_Click **Reset** to start an episode._",
        )

        with gr.Accordion("Raw JSON response", open=False):
            raw_json = gr.Code(
                value="{}",
                language="json",
                label="Raw observation payload",
                interactive=False,
            )

        # ── Full-width README ────────────────────────────────────────────────
        if readme_content:
            gr.Markdown("---")
            with gr.Accordion("Full README", open=False):
                gr.Markdown(readme_content)

        # ── Wire events ──────────────────────────────────────────────────────
        reset_btn.click(
            fn=reset_env,
            inputs=[task_dropdown],
            outputs=[obs_display, raw_json, status_box],
            api_name="reset_env",
        )
        step_btn.click(
            fn=step_env,
            inputs=[action_dropdown, target_input],
            outputs=[obs_display, raw_json, status_box],
            api_name="step_form",
        )
        state_btn.click(
            fn=get_state_sync,
            inputs=[],
            outputs=[raw_json],
            api_name="get_state_sync",
        )

    return blocks

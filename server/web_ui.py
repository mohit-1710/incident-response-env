"""Custom Gradio /web UI for the Incident Response environment.

Minimal full-width playground:
    - Title only (no subtitle, no overview — that's all in the README)
    - Scenario dropdown + Reset / Get-state row
    - Action row (action type + target service + Step)
    - Status line
    - Formatted observation
    - Raw JSON (collapsed)
    - Full README (collapsed)

Wired into server/app.py via a monkey-patch on
`openenv.core.env_server.web_interface.build_gradio_app`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

# Dual-import: works both as installed package and from source dir
try:
    from ..scenarios import AVAILABLE_TASKS
except ImportError:
    from scenarios import AVAILABLE_TASKS  # type: ignore[no-redef]


TASK_CHOICES = [
    ("Easy", "single_service_failure"),
    ("Medium", "multi_service_correlation"),
    ("Hard", "cascading_outage"),
]
DEFAULT_TASK = AVAILABLE_TASKS[0]

ACTION_CHOICES = [
    "acknowledge",
    "diagnose",
    "fix",
    "escalate",
    "check_status",
]


# ── Markdown formatting ──────────────────────────────────────────────────────

def _format_observation_md(payload: Dict[str, Any]) -> str:
    """Render a step/reset response as readable Markdown."""
    obs = payload.get("observation") or {}
    if not obs:
        return "_No observation yet — pick a scenario and click Reset._"

    reward = payload.get("reward")
    done = payload.get("done", False)
    msg = obs.get("message", "")
    step_n = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 0)
    resolved = obs.get("resolved_count", 0)
    total = obs.get("total_services", 0)

    lines: List[str] = []
    header = [
        f"**Step** `{step_n}/{max_steps}`",
        f"**Healthy** `{resolved}/{total}`",
        f"**Done** `{done}`",
    ]
    if reward is not None:
        header.append(f"**Reward** `{reward:.4f}`")
    lines.append("  ·  ".join(header))
    lines.append("")
    if msg:
        lines.append(f"> {msg}")
        lines.append("")

    alerts = obs.get("alerts") or []
    if alerts:
        lines.append("**Alerts**")
        for a in alerts:
            ack = "ACK" if a.get("acknowledged") else "NEW"
            sev = (a.get("severity") or "").upper()
            svc = a.get("service", "?")
            text = a.get("message", "")
            lines.append(f"- `{ack}` `{sev:8s}` **{svc}** — {text}")
        lines.append("")

    services = obs.get("services") or {}
    if services:
        groups: Dict[str, List[str]] = {"down": [], "degraded": [], "healthy": []}
        for svc, status in services.items():
            groups.setdefault(status, []).append(svc)
        lines.append("**Service health**")
        for status_name in ("down", "degraded", "healthy"):
            items = groups.get(status_name) or []
            if items:
                lines.append(f"- **{status_name}** — {', '.join(items)}")
        lines.append("")

    deps = obs.get("dependencies") or {}
    if deps:
        lines.append("**Dependency graph**")
        for svc, dep_list in deps.items():
            if dep_list:
                lines.append(f"- `{svc}` → {', '.join(f'`{d}`' for d in dep_list)}")
            else:
                lines.append(f"- `{svc}` _(no dependencies)_")
        lines.append("")

    diagnostics = obs.get("diagnostic_results") or {}
    if diagnostics:
        lines.append("**Diagnostics collected**")
        for svc, result in diagnostics.items():
            lines.append(f"- **{svc}**")
            lines.append(f"  > {result}")
        lines.append("")

    rubrics = obs.get("rubric_results") or []
    if rubrics:
        passed = sum(1 for r in rubrics if r.get("passed"))
        lines.append(f"**Final grading — {passed}/{len(rubrics)} rubrics passed**")
        for r in rubrics:
            mark = "PASS" if r.get("passed") else "FAIL"
            lines.append(f"- `[{mark}]` {r.get('name', '?')}")
        lines.append("")

    return "\n".join(lines)


# ── Gradio Blocks builder (replaces openenv-core's default) ──────────────────

def _strip_yaml_frontmatter(text: str) -> str:
    """Remove the leading `--- ... ---` YAML block from a Markdown file.

    HF Spaces requires the frontmatter (title, emoji, sdk, app_port, etc.)
    in the actual README file, but it shouldn't be displayed as content
    in the /web playground.
    """
    if not text or not text.lstrip().startswith("---"):
        return text
    # Find the start of the frontmatter (first `---` line)
    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    rest = stripped[3:]  # past the opening ---
    # Find the next line that is exactly `---` (closing fence)
    end_idx = rest.find("\n---")
    if end_idx == -1:
        return text  # malformed; return as-is
    body = rest[end_idx + 4 :]  # past the closing --- and its newline
    return (leading + body).lstrip()


def build_incident_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],  # noqa: ARG001
    metadata: Any,
    is_chat_env: bool,  # noqa: ARG001
    title: str = "Incident Response Environment",
    quick_start_md: Optional[str] = None,  # noqa: ARG001
) -> gr.Blocks:
    """Build the minimal full-width playground for /web."""

    readme_content = ""
    if metadata is not None:
        readme_content = getattr(metadata, "readme_content", "") or ""
    readme_content = _strip_yaml_frontmatter(readme_content)

    async def reset_env(task_name: str):
        try:
            payload = await web_manager.reset_environment({"task_name": task_name})
            md = _format_observation_md(payload)
            raw = json.dumps(payload, indent=2, default=str)
            return md, raw, f"Reset — running {task_name}"
        except Exception as exc:  # pragma: no cover
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
            return md, raw, ("Episode complete" if done else f"Step {n} done")
        except Exception as exc:  # pragma: no cover
            return f"**Step failed:** `{exc}`", "{}", f"Error: {exc}"

    def get_state_sync():
        try:
            state = web_manager.get_state()
            return json.dumps(state, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            return f'{{"error": "{exc}"}}'

    with gr.Blocks(
        title=f"OpenEnv: {title}",
        analytics_enabled=False,
    ) as blocks:
        gr.Markdown("# Incident Response Environment", elem_id="ir-title")
        gr.Markdown(
            "_OpenEnv RL environment for SRE on-call triage. "
            "See the README below for the action / observation spec, scoring, and benchmarks._",
            elem_id="ir-tagline",
        )

        # ── Scenario: 3-segment radio (no dropdown toggle weirdness) ─────────
        task_radio = gr.Radio(
            choices=TASK_CHOICES,
            value=DEFAULT_TASK,
            label="Scenario",
            interactive=True,
            elem_classes=["ir-radio"],
        )
        with gr.Row(elem_classes=["ir-button-row"]):
            reset_btn = gr.Button("Reset", variant="primary", elem_classes=["ir-btn"])
            state_btn = gr.Button("Get State", variant="secondary", elem_classes=["ir-btn"])

        # ── Action row (action dropdown + target textbox) ────────────────────
        with gr.Row(equal_height=True):
            action_dropdown = gr.Dropdown(
                choices=ACTION_CHOICES,
                value="diagnose",
                label="Action",
                interactive=True,
                allow_custom_value=False,
                filterable=False,
                multiselect=False,
                scale=2,
                elem_classes=["ir-dropdown"],
            )
            target_input = gr.Textbox(
                label="Target service",
                placeholder="database",
                lines=1,
                max_lines=1,
                scale=3,
                elem_classes=["ir-textbox"],
            )
        with gr.Row(elem_classes=["ir-button-row"]):
            step_btn = gr.Button("Step", variant="primary", elem_classes=["ir-btn"])

        status_box = gr.Textbox(
            label="Status",
            value="Ready",
            interactive=False,
            lines=1,
            max_lines=1,
            elem_classes=["ir-textbox", "ir-status"],
        )

        obs_display = gr.Markdown(
            "_Click Reset to start an episode._",
            elem_classes=["ir-observation"],
        )

        with gr.Accordion("Raw JSON", open=False, elem_classes=["ir-accordion"]):
            raw_json = gr.Code(
                value="{}",
                language="json",
                interactive=False,
            )

        if readme_content:
            with gr.Accordion("README", open=False, elem_classes=["ir-accordion"]):
                gr.Markdown(readme_content)

        reset_btn.click(
            fn=reset_env,
            inputs=[task_radio],
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

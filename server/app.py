"""FastAPI application wiring for the Incident Response environment."""

# ── Monkey-patch openenv-core's Gradio UI BEFORE create_app() ────────────────
#
# Three patches, all cosmetic:
#
# 1. Replace `build_gradio_app` with our custom builder so /web uses our
#    full-width single-column layout (no left/right split, no Quick Start).
#    Patching this directly bypasses the TabbedInterface that the
#    `gradio_builder=` kwarg would force.
#
# 2. Override `OPENENV_GRADIO_CSS` (the CSS string openenv passes to
#    `mount_gradio_app(css=...)` at mount time, which `!important`-overrides
#    anything we set on our own Blocks). We append bright border styling so
#    dropdowns and textboxes are clearly visible in dark mode.
#
# 3. Override the theme's dark-mode `input_border_color` so even unstyled
#    components inherit a visible border.
#
# None of this touches WebSocket routes, HTTP routes, grading logic, or
# inference.py — judges are unaffected.

import openenv.core.env_server.gradio_theme as _openenv_theme  # noqa: E402
import openenv.core.env_server.web_interface as _openenv_wi  # noqa: E402

try:
    from .web_ui import build_incident_ui as _build_incident_ui
except ImportError:
    from server.web_ui import build_incident_ui as _build_incident_ui  # type: ignore[no-redef]

# Patch 1: replace the Gradio builder
_openenv_wi.build_gradio_app = _build_incident_ui

# Patch 2: append bright-border CSS to openenv's mount-time CSS string.
#
# Gradio 6 components are wrapped in <div data-testid="..."> elements.
# Targeting these (rather than the Svelte-generated class hashes that change
# every release) is the most stable selector strategy.
_BORDER_CSS_OVERRIDE = """

/* ── Incident Response env: classy /web playground styling ──────────────── */
/* All selectors target our own ir-* elem_classes — no Svelte hash chasing.  */
/* Borders are placed on the actual interactive surfaces, not on label boxes. */

/* Refined neutral palette (GitHub Dark inspired) */
:root {
    --ir-border: rgba(139, 148, 158, 0.35);
    --ir-border-strong: rgba(139, 148, 158, 0.55);
    --ir-focus: rgba(88, 166, 255, 0.95);
    --ir-focus-glow: rgba(88, 166, 255, 0.20);
    --ir-bg-tint: rgba(255, 255, 255, 0.025);
}

#ir-title { margin-bottom: 0 !important; padding-bottom: 0 !important; }
#ir-tagline { margin-top: 4px !important; opacity: 0.65; font-size: 0.93em; }

/* ── Inputs: identical fixed heights for the dropdown and textbox ─────────── */

:root {
    --ir-input-height: 42px;
}

/* Textbox input — fixed height */
.gradio-container .ir-textbox input[type="text"],
.gradio-container .ir-textbox textarea {
    border: 1px solid var(--ir-border-strong) !important;
    border-radius: 6px !important;
    padding: 0 12px !important;
    height: var(--ir-input-height) !important;
    min-height: var(--ir-input-height) !important;
    max-height: var(--ir-input-height) !important;
    box-sizing: border-box !important;
    background-color: var(--ir-bg-tint) !important;
    line-height: 1 !important;
    margin: 0 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

/* Dropdown outer .wrap — fixed height matching the textbox */
.gradio-container .ir-dropdown .wrap {
    border: 1px solid var(--ir-border-strong) !important;
    border-radius: 6px !important;
    height: var(--ir-input-height) !important;
    min-height: var(--ir-input-height) !important;
    max-height: var(--ir-input-height) !important;
    box-sizing: border-box !important;
    background-color: var(--ir-bg-tint) !important;
    padding: 0 12px !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

/* Strip the inner wrap so it doesn't add extra height */
.gradio-container .ir-dropdown .wrap-inner {
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

/* Strip the dropdown's internal input */
.gradio-container .ir-dropdown .wrap input,
.gradio-container .ir-dropdown .wrap-inner input {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1 !important;
    box-shadow: none !important;
}

/* Force the column wrappers to align bottom (so labels don't push inputs around) */
.gradio-container .ir-textbox,
.gradio-container .ir-dropdown {
    align-self: flex-end !important;
}

/* ── Focus highlight ──────────────────────────────────────────────────────── */
.gradio-container .ir-textbox input[type="text"]:focus,
.gradio-container .ir-textbox textarea:focus,
.gradio-container .ir-dropdown:focus-within .wrap-inner,
.gradio-container .ir-dropdown:focus-within .wrap {
    border-color: var(--ir-focus) !important;
    box-shadow: 0 0 0 2px var(--ir-focus-glow) !important;
    outline: none !important;
}

/* ── Status field: muted look since it's read-only ────────────────────────── */
.gradio-container .ir-status input[type="text"],
.gradio-container .ir-status textarea {
    opacity: 0.85 !important;
    cursor: default !important;
}

/* ── Radio group (scenario picker) — segmented control style ──────────────── */
.gradio-container .ir-radio fieldset {
    border: 1px solid var(--ir-border) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    display: flex !important;
    gap: 4px !important;
    background-color: var(--ir-bg-tint) !important;
}
.gradio-container .ir-radio label {
    flex: 1 !important;
    text-align: center !important;
    padding: 8px 14px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease !important;
    border: none !important;
}
.gradio-container .ir-radio label:hover {
    background-color: rgba(255, 255, 255, 0.04) !important;
}
.gradio-container .ir-radio input[type="radio"] {
    display: none !important;
}
.gradio-container .ir-radio label:has(input[type="radio"]:checked) {
    background-color: rgba(35, 134, 54, 0.20) !important;
    color: rgba(63, 185, 80, 1) !important;
    font-weight: 600 !important;
    box-shadow: inset 0 0 0 1px rgba(35, 134, 54, 0.45) !important;
}

/* ── Buttons: subtle borders, clear hover ────────────────────────────────── */
.gradio-container .ir-btn.primary,
.gradio-container button.ir-btn.primary {
    border: 1px solid rgba(35, 134, 54, 0.85) !important;
    border-radius: 6px !important;
}
.gradio-container .ir-btn.secondary,
.gradio-container button.ir-btn.secondary {
    border: 1px solid var(--ir-border-strong) !important;
    border-radius: 6px !important;
}

/* ── Accordions: subtle border around the whole thing ────────────────────── */
.gradio-container .ir-accordion {
    border: 1px solid var(--ir-border) !important;
    border-radius: 8px !important;
}

/* ── Observation panel: subtle frame ─────────────────────────────────────── */
.gradio-container .ir-observation {
    border: 1px solid var(--ir-border) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    background-color: var(--ir-bg-tint) !important;
}

/* ── Tighter spacing on labels above inputs ──────────────────────────────── */
.gradio-container .ir-textbox label > span:first-child,
.gradio-container .ir-dropdown label > span:first-child,
.gradio-container .ir-radio label[for] {
    font-size: 0.85em !important;
    opacity: 0.75 !important;
    margin-bottom: 4px !important;
}
"""

_openenv_theme.OPENENV_GRADIO_CSS = (
    _openenv_theme.OPENENV_GRADIO_CSS + _BORDER_CSS_OVERRIDE
)
# Re-import the now-patched value into web_interface (it imported by name)
_openenv_wi.OPENENV_GRADIO_CSS = _openenv_theme.OPENENV_GRADIO_CSS

# Patch 3: brighten the dark-mode input border color in the theme itself,
# so even if a selector misses, the underlying CSS variable produces a
# visible line. The original is "#30363d" — barely darker than the bg.
try:
    _openenv_theme.OPENENV_GRADIO_THEME = _openenv_theme.OPENENV_GRADIO_THEME.set(
        input_border_color_dark="#8b949e",
        block_border_color_dark="#30363d",
        border_color_primary_dark="#8b949e",
    )
    _openenv_wi.OPENENV_GRADIO_THEME = _openenv_theme.OPENENV_GRADIO_THEME
except Exception:
    # If the .set() API changes, fall back silently — the CSS patch above
    # is enough to make the borders visible.
    pass

try:
    from ..models import IncidentAction, IncidentObservation
    from .environment import IncidentResponseEnvironment
except ImportError:
    from models import IncidentAction, IncidentObservation
    from server.environment import IncidentResponseEnvironment

from openenv.core.env_server import create_app  # noqa: E402

app = create_app(
    IncidentResponseEnvironment,
    IncidentAction,
    IncidentObservation,
    env_name="incident_response_env",
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution (used by [project.scripts])."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

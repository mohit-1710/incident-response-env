"""FastAPI application wiring for the Incident Response environment."""

# ── Monkey-patch openenv-core's Gradio UI builder BEFORE create_app() ────────
#
# We replace `build_gradio_app` (the function `create_web_interface_app` calls
# to construct the SINGLE Gradio app it mounts at /web) with our own builder.
#
# Why monkey-patch instead of `create_app(gradio_builder=...)` ?
#   The framework, when given a `gradio_builder`, wraps the DEFAULT UI and
#   the CUSTOM UI in a `gr.TabbedInterface` — you'd see two tabs ("Playground"
#   and "Custom"). Patching `build_gradio_app` directly bypasses that and
#   makes our UI the only UI, which is what we want.
#
# This only affects the cosmetic /web playground. It does NOT touch the
# WebSocket routes, the HTTP routes, the grading logic, or anything the
# judges' inference.py runner uses.
import openenv.core.env_server.web_interface as _openenv_wi  # noqa: E402

try:
    from .web_ui import build_incident_ui as _build_incident_ui
except ImportError:
    from server.web_ui import build_incident_ui as _build_incident_ui  # type: ignore[no-redef]

_openenv_wi.build_gradio_app = _build_incident_ui

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

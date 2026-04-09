"""FastAPI application wiring for the Incident Response environment."""

# ── Monkey-patch the openenv Quick Start template BEFORE create_app() ────────
#
# The default openenv-core /web Gradio UI hard-codes a chat-style code snippet
# (`IncidentAction(message="...")`) in its Quick Start accordion, which is
# wrong for this env (our action shape is `action_type` + `target_service`,
# not `message`). The framework reads this template via the
# `get_quick_start_markdown` function in `openenv.core.env_server.web_interface`.
#
# Passing a custom `gradio_builder` doesn't help because the framework wraps
# both the default and the custom UI in a TabbedInterface — you'd see TWO
# UIs, not one. The cleanest fix is to monkey-patch the template function
# to return an empty string. When `quick_start_md` is empty/falsy, the
# Quick Start accordion is skipped entirely (see gradio_ui.py line 139:
# `if quick_start_md:`).
#
# This patch only affects the cosmetic Quick Start panel. It does NOT touch
# the playground, the README accordion, the action dropdowns, the WebSocket
# routes, the HTTP routes, or any grading logic.
import openenv.core.env_server.web_interface as _openenv_wi  # noqa: E402

_openenv_wi.get_quick_start_markdown = lambda *args, **kwargs: ""

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

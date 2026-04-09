"""FastAPI application wiring for the Incident Response environment."""

try:
    from ..models import IncidentAction, IncidentObservation
    from .environment import IncidentResponseEnvironment
except ImportError:
    from models import IncidentAction, IncidentObservation
    from server.environment import IncidentResponseEnvironment

from openenv.core.env_server import create_app

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

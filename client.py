"""OpenEnv client for connecting to the Incident Response environment.

Used by inference scripts and training code to communicate with
the environment server over WebSocket.
"""

from __future__ import annotations

from typing import Any, Dict, List

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from incident_response_env.models import (
        IncidentAction,
        IncidentObservation,
        IncidentState,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        IncidentAction,
        IncidentObservation,
        IncidentState,
    )


class IncidentResponseEnv(
    EnvClient[IncidentAction, IncidentObservation, IncidentState]
):
    """Client interface for the Incident Response environment."""

    def _step_payload(self, action: IncidentAction) -> dict:
        """Serialise an action for transmission to the server."""
        return {
            "action_type": action.action_type,
            "target_service": action.target_service,
        }

    def _parse_result(self, payload: dict) -> StepResult[IncidentObservation]:
        """Deserialise a server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=IncidentObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                alerts=obs_data.get("alerts", []),
                services=obs_data.get("services", {}),
                dependencies=obs_data.get("dependencies", {}),
                actions_taken=obs_data.get("actions_taken", []),
                diagnostic_results=obs_data.get("diagnostic_results", {}),
                message=obs_data.get("message", ""),
                resolved_count=obs_data.get("resolved_count", 0),
                total_services=obs_data.get("total_services", 0),
                step_number=obs_data.get("step_number", 0),
                max_steps=obs_data.get("max_steps", 0),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> IncidentState:
        """Deserialise a state response into a typed IncidentState."""
        return IncidentState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            accumulated_reward=payload.get("accumulated_reward", 0.0),
            root_causes_found=payload.get("root_causes_found", []),
            root_causes_fixed=payload.get("root_causes_fixed", []),
            services_healthy=payload.get("services_healthy", 0),
            services_total=payload.get("services_total", 0),
        )

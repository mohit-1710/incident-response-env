"""Pydantic models for the Incident Response environment.

Defines the typed action, observation, and state contracts
that form the API between agent and environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State


# Valid action types an on-call engineer can take
ACTION_TYPES = Literal[
    "acknowledge",   # Acknowledge an active alert
    "diagnose",      # Run diagnostics on a service to find root cause
    "fix",           # Apply a fix to a service
    "escalate",      # Escalate to a senior engineer / another team
    "check_status",  # Review current service health dashboard
]


class IncidentAction(Action):
    """An action taken by the on-call engineer during incident response.

    The agent selects an action type and a target service to act upon.
    Not every action requires a valid target (e.g. check_status), but
    the field is always present for interface consistency.
    """

    action_type: ACTION_TYPES
    target_service: str = ""


class IncidentObservation(Observation):
    """Environment observation returned after each step.

    Provides the agent with the current state of the incident:
    active alerts, service health, dependency map, and feedback
    from the last action taken. The inherited `done` and `reward`
    fields carry episode termination and scoring signals.
    """

    alerts: List[Dict[str, Any]] = []
    services: Dict[str, str] = {}
    dependencies: Dict[str, List[str]] = {}
    actions_taken: List[str] = []
    diagnostic_results: Dict[str, str] = {}
    message: str = ""
    resolved_count: int = 0
    total_services: int = 0
    step_number: int = 0
    max_steps: int = 0
    rubric_results: List[Dict[str, Any]] = []


class IncidentState(State):
    """Internal episode state exposed via the state() endpoint.

    Extends the base State (episode_id, step_count) with
    incident-specific tracking fields.
    """

    task_name: str = ""
    accumulated_reward: float = 0.0
    root_causes_found: List[str] = []
    root_causes_fixed: List[str] = []
    services_healthy: int = 0
    services_total: int = 0

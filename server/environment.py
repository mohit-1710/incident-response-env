"""Core incident response environment simulation.

Implements the OpenEnv Environment interface: reset(), step(), state.
Manages service dependency graphs, cascading failure propagation,
and deterministic reward computation for grading.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Set

# Dual-import pattern — required for both in-repo and Docker execution
try:
    from ..models import IncidentAction, IncidentObservation, IncidentState
    from ..scenarios import AVAILABLE_TASKS, SCENARIOS, Scenario
except ImportError:
    from models import IncidentAction, IncidentObservation, IncidentState
    from scenarios import AVAILABLE_TASKS, SCENARIOS, Scenario

from openenv.core.env_server import Environment


# Reward budget — scaled per scenario so golden path always reaches ~1.0.
# The per-root-cause amounts are divided by the number of root causes,
# ensuring consistent scoring regardless of scenario complexity.
BUDGET_ACKNOWLEDGE = 0.10   # Total budget for acknowledging root cause alerts
BUDGET_DIAGNOSE = 0.25      # Total budget for diagnosing root cause services
BUDGET_FIX = 0.35           # Total budget for fixing root cause services
REWARD_ALL_HEALTHY = 0.20   # Completion bonus when all services are restored
REWARD_EFFICIENCY_MAX = 0.10  # Max bonus for step efficiency
# Sum: 0.10 + 0.25 + 0.35 + 0.20 + 0.10 = 1.00

REWARD_ACKNOWLEDGE_OTHER = 0.01  # Small bonus for acknowledging non-root alerts
REWARD_FIX_SYMPTOM_PENALTY = -0.10  # Penalty for fixing a symptom, not root cause


class IncidentResponseEnvironment(
    Environment[IncidentAction, IncidentObservation, IncidentState]
):
    """Simulates SRE on-call incident triage with cascading service failures.

    The agent receives alerts about a production incident and must
    acknowledge alerts, diagnose services, and apply fixes to restore
    the system. Services form a dependency graph — fixing a root cause
    automatically recovers its downstream dependents.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = IncidentState()
        self._scenario: Optional[Scenario] = None

        # Mutable per-episode state
        self._services: Dict[str, str] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._acknowledged: Set[str] = set()
        self._diagnosed: Set[str] = set()
        self._fixed: Set[str] = set()
        self._root_causes: Set[str] = set()
        self._root_cause_alerts: Set[str] = set()
        self._actions_taken: List[str] = []
        self._diagnostic_results: Dict[str, str] = {}
        self._accumulated_reward: float = 0.0
        self._done: bool = False
        self._last_message: str = ""

        # Lookup tables built on reset
        self._service_defs: Dict[str, Any] = {}
        self._alert_map: Dict[str, Any] = {}
        self._n_root_causes: int = 1  # Set on reset, used for reward scaling

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Initialise a new incident episode.

        Args:
            seed: unused, kept for interface compatibility.
            episode_id: optional episode identifier.
            **kwargs: must include `task_name` to select the scenario.
                      Defaults to 'single_service_failure' if absent.
        """
        task_name = kwargs.get("task_name", "single_service_failure")

        if task_name not in SCENARIOS:
            # Graceful error — return an observation explaining valid tasks
            self._done = True
            return IncidentObservation(
                done=True,
                reward=0.0,
                message=(
                    f"Unknown task '{task_name}'. "
                    f"Available tasks: {AVAILABLE_TASKS}"
                ),
            )

        self._scenario = SCENARIOS[task_name]
        self._state = IncidentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
        )

        # Build service state from scenario definition
        self._services = {}
        self._dependencies = {}
        self._service_defs = {}
        for svc in self._scenario.services:
            self._services[svc.name] = svc.initial_status
            self._dependencies[svc.name] = list(svc.depends_on)
            self._service_defs[svc.name] = svc

        # Build alert lookup
        self._alert_map = {a.alert_id: a for a in self._scenario.alerts}
        self._root_causes = set(self._scenario.root_cause_services)
        self._n_root_causes = max(1, len(self._root_causes))
        self._root_cause_alerts = {
            a.alert_id for a in self._scenario.alerts if a.is_root_cause
        }

        # Reset per-episode mutable state
        self._acknowledged = set()
        self._diagnosed = set()
        self._fixed = set()
        self._actions_taken = []
        self._diagnostic_results = {}
        self._accumulated_reward = 0.0
        self._done = False
        self._last_message = (
            f"INCIDENT DETECTED: {self._scenario.name}. "
            f"{len(self._scenario.alerts)} alerts firing across "
            f"{len(self._services)} services. Begin triage."
        )

        return self._build_observation()

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Process one agent action and return the updated observation.

        Handles all action types, updates state, propagates cascading
        recovery, computes step reward, and checks for episode completion.
        """
        # Guard: episode already finished
        if self._done:
            return self._build_observation(
                message="Incident already resolved. Call reset() to start a new episode."
            )

        self._state.step_count += 1
        step_reward = 0.0

        # Dispatch action type
        action_type = action.action_type
        target = action.target_service.strip()

        if action_type == "check_status":
            step_reward, msg = self._handle_check_status()
        elif target not in self._services:
            step_reward = 0.0
            msg = (
                f"Service '{target}' not found in this incident. "
                f"Available services: {sorted(self._services.keys())}"
            )
        elif action_type == "acknowledge":
            step_reward, msg = self._handle_acknowledge(target)
        elif action_type == "diagnose":
            step_reward, msg = self._handle_diagnose(target)
        elif action_type == "fix":
            step_reward, msg = self._handle_fix(target)
        elif action_type == "escalate":
            step_reward, msg = self._handle_escalate(target)
        else:
            step_reward = 0.0
            msg = (
                f"Unknown action type '{action_type}'. "
                f"Valid types: acknowledge, diagnose, fix, escalate, check_status"
            )

        self._actions_taken.append(
            f"[Step {self._state.step_count}] {action_type} → {target}: {msg}"
        )
        self._accumulated_reward += step_reward

        # After fixing a root cause, propagate recovery through the graph
        if action_type == "fix" and target in self._services:
            self._propagate_health()

        # Check completion conditions
        all_healthy = all(s == "healthy" for s in self._services.values())
        max_steps_reached = self._state.step_count >= self._scenario.max_steps

        if all_healthy:
            # Completion bonus
            self._accumulated_reward += REWARD_ALL_HEALTHY
            # Efficiency bonus: reward for finishing quickly
            ratio = self._state.step_count / self._scenario.max_steps
            self._accumulated_reward += max(0.0, REWARD_EFFICIENCY_MAX * (1.0 - ratio))
            self._done = True
            msg = (
                "All services restored to healthy. Incident resolved. "
                f"Final score: {self._clamped_reward():.4f}"
            )
        elif max_steps_reached:
            self._done = True
            msg = (
                f"Maximum steps ({self._scenario.max_steps}) reached. "
                f"Incident partially resolved. "
                f"Final score: {self._clamped_reward():.4f}"
            )

        self._last_message = msg
        self._state.accumulated_reward = self._clamped_reward()
        self._state.root_causes_found = sorted(self._diagnosed & self._root_causes)
        self._state.root_causes_fixed = sorted(self._fixed & self._root_causes)

        healthy_count = sum(1 for s in self._services.values() if s == "healthy")
        self._state.services_healthy = healthy_count
        self._state.services_total = len(self._services)

        return self._build_observation(message=msg)

    @property
    def state(self) -> IncidentState:
        """Return the current episode state."""
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_check_status(self) -> tuple[float, str]:
        """Return a summary of current service health. No reward change."""
        healthy = [s for s, st in self._services.items() if st == "healthy"]
        degraded = [s for s, st in self._services.items() if st == "degraded"]
        down = [s for s, st in self._services.items() if st == "down"]
        return 0.0, (
            f"Service Health Dashboard — "
            f"Healthy: {healthy or 'none'}, "
            f"Degraded: {degraded or 'none'}, "
            f"Down: {down or 'none'}"
        )

    def _handle_acknowledge(self, target: str) -> tuple[float, str]:
        """Acknowledge alerts for the target service."""
        # Find alerts for this service that haven't been acknowledged
        service_alerts = [
            a for a in self._scenario.alerts
            if a.service == target and a.alert_id not in self._acknowledged
        ]
        if not service_alerts:
            if any(a.service == target for a in self._scenario.alerts):
                return 0.0, f"Alerts for '{target}' already acknowledged."
            return 0.0, f"No alerts found for service '{target}'."

        reward = 0.0
        per_root_ack = BUDGET_ACKNOWLEDGE / self._n_root_causes
        for alert in service_alerts:
            self._acknowledged.add(alert.alert_id)
            if alert.is_root_cause:
                reward += per_root_ack
            else:
                reward += REWARD_ACKNOWLEDGE_OTHER

        return reward, (
            f"Acknowledged {len(service_alerts)} alert(s) for '{target}'. "
            f"[{', '.join(a.severity.upper() for a in service_alerts)}]"
        )

    def _handle_diagnose(self, target: str) -> tuple[float, str]:
        """Run diagnostics on a service. Reveals root cause information."""
        if target in self._diagnosed:
            # Return cached result, no additional reward
            return 0.0, (
                f"Already diagnosed '{target}'. "
                f"Previous result: {self._diagnostic_results.get(target, 'N/A')}"
            )

        self._diagnosed.add(target)
        svc_def = self._service_defs[target]
        diag_output = svc_def.diagnostic_output or f"Service '{target}' diagnostics inconclusive."
        self._diagnostic_results[target] = diag_output

        reward = 0.0
        if target in self._root_causes:
            reward = BUDGET_DIAGNOSE / self._n_root_causes

        return reward, f"Diagnostics for '{target}': {diag_output}"

    def _handle_fix(self, target: str) -> tuple[float, str]:
        """Apply a fix to a service."""
        if target in self._fixed:
            return 0.0, f"Fix already applied to '{target}'. No additional action needed."

        if self._services[target] == "healthy":
            return 0.0, f"Service '{target}' is already healthy. No fix required."

        self._fixed.add(target)

        if target in self._root_causes:
            # Correct fix — restore this service
            self._services[target] = "healthy"
            return BUDGET_FIX / self._n_root_causes, (
                f"Fix applied to root cause '{target}'. "
                f"Service restored to healthy. Checking downstream dependencies..."
            )
        else:
            # Wrong fix — this is a symptom, not the cause.
            # The service might temporarily appear fixed but the root cause persists.
            # We don't change its status because the root cause is still broken.
            return REWARD_FIX_SYMPTOM_PENALTY, (
                f"Fix attempted on '{target}', but this service's issues are caused "
                f"by an upstream dependency. Fix the root cause instead. "
                f"Service remains {self._services[target]}."
            )

    def _handle_escalate(self, target: str) -> tuple[float, str]:
        """Escalate a service issue. Small reward for appropriate escalation."""
        # Escalation is valid but doesn't directly fix anything
        return 0.0, (
            f"Escalated '{target}' to the responsible team. "
            f"They will investigate, but you should continue triage."
        )

    # ------------------------------------------------------------------
    # Cascading recovery
    # ------------------------------------------------------------------

    def _propagate_health(self) -> None:
        """Propagate health status through the dependency graph.

        After a root cause is fixed, walk the graph and restore any
        service whose dependencies are now all healthy. Repeat until
        no more changes occur (handles multi-level cascades).
        """
        changed = True
        while changed:
            changed = False
            for svc_name, status in list(self._services.items()):
                if status == "healthy":
                    continue
                deps = self._dependencies.get(svc_name, [])
                if not deps:
                    continue
                # A service recovers when ALL its dependencies are healthy
                if all(self._services.get(d) == "healthy" for d in deps):
                    # Only recover if this service isn't itself a broken root cause
                    svc_def = self._service_defs[svc_name]
                    if svc_def.root_cause and svc_name not in self._fixed:
                        continue  # Still broken at its own level
                    self._services[svc_name] = "healthy"
                    changed = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamped_reward(self) -> float:
        """Return accumulated reward clamped to [0.0, 1.0]."""
        return max(0.0, min(1.0, self._accumulated_reward))

    def _build_observation(self, message: str = "") -> IncidentObservation:
        """Construct the observation returned to the agent."""
        msg = message or self._last_message
        healthy_count = sum(1 for s in self._services.values() if s == "healthy")

        return IncidentObservation(
            done=self._done,
            reward=self._clamped_reward() if self._done else self._accumulated_reward,
            alerts=[
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "service": a.service,
                    "message": a.message,
                    "acknowledged": a.alert_id in self._acknowledged,
                }
                for a in (self._scenario.alerts if self._scenario else [])
            ],
            services=dict(self._services),
            dependencies={k: list(v) for k, v in self._dependencies.items()},
            actions_taken=list(self._actions_taken),
            diagnostic_results=dict(self._diagnostic_results),
            message=msg,
            resolved_count=healthy_count,
            total_services=len(self._services),
            step_number=self._state.step_count,
            max_steps=self._scenario.max_steps if self._scenario else 0,
        )

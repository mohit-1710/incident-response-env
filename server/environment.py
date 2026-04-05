"""Core incident response environment simulation.

Implements the OpenEnv Environment interface: reset(), step(), state.
Manages service dependency graphs, cascading failure propagation,
and binary rubric-based grading where each criterion scores 0 or 1.
"""

from __future__ import annotations

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


# Per-step training signal (small, continuous) — separate from the grader
STEP_REWARD_ACKNOWLEDGE = 0.01
STEP_REWARD_DIAGNOSE = 0.02
STEP_REWARD_FIX_ROOT = 0.03
STEP_REWARD_FIX_SYMPTOM = -0.02


class IncidentResponseEnvironment(
    Environment[IncidentAction, IncidentObservation, IncidentState]
):
    """Simulates SRE on-call incident triage with cascading service failures.

    The agent receives alerts about a production incident and must
    acknowledge alerts, diagnose services, and apply fixes to restore
    the system. Services form a dependency graph — fixing a root cause
    automatically recovers its downstream dependents.

    Grading uses binary rubrics: each evaluation criterion returns 0 or 1.
    Final score = passed_rubrics / total_rubrics.
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
        self._symptom_fixes: Set[str] = set()  # Track incorrect fixes
        self._root_causes: Set[str] = set()
        self._actions_taken: List[str] = []
        self._diagnostic_results: Dict[str, str] = {}
        self._done: bool = False
        self._last_message: str = ""

        # Lookup tables built on reset
        self._service_defs: Dict[str, Any] = {}
        self._alert_map: Dict[str, Any] = {}
        self._n_root_causes: int = 1

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

        # Reset per-episode mutable state
        self._acknowledged = set()
        self._diagnosed = set()
        self._fixed = set()
        self._symptom_fixes = set()
        self._actions_taken = []
        self._diagnostic_results = {}
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
        """Process one agent action and return the updated observation."""
        # Guard: episode already finished
        if self._done:
            return self._build_observation(
                message="Incident already resolved. Call reset() to start a new episode."
            )

        self._state.step_count += 1

        # Dispatch action type
        action_type = action.action_type
        target = action.target_service.strip()

        if action_type == "check_status":
            msg = self._handle_check_status()
        elif target not in self._services:
            msg = (
                f"Service '{target}' not found in this incident. "
                f"Available services: {sorted(self._services.keys())}"
            )
        elif action_type == "acknowledge":
            msg = self._handle_acknowledge(target)
        elif action_type == "diagnose":
            msg = self._handle_diagnose(target)
        elif action_type == "fix":
            msg = self._handle_fix(target)
        elif action_type == "escalate":
            msg = self._handle_escalate(target)
        else:
            msg = (
                f"Unknown action type '{action_type}'. "
                f"Valid types: acknowledge, diagnose, fix, escalate, check_status"
            )

        self._actions_taken.append(
            f"[Step {self._state.step_count}] {action_type} → {target}: {msg}"
        )

        # After fixing a root cause, propagate recovery through the graph
        if action_type == "fix" and target in self._services:
            self._propagate_health()

        # Check completion conditions
        all_healthy = all(s == "healthy" for s in self._services.values())
        max_steps_reached = self._state.step_count >= self._scenario.max_steps

        if all_healthy or max_steps_reached:
            self._done = True
            rubrics = self._evaluate_rubrics()
            score = self._compute_grader_score(rubrics)

            if all_healthy:
                msg = (
                    f"All services restored to healthy. Incident resolved. "
                    f"Final score: {score:.4f} "
                    f"({sum(1 for r in rubrics if r['passed'])}/{len(rubrics)} rubrics passed)"
                )
            else:
                msg = (
                    f"Maximum steps ({self._scenario.max_steps}) reached. "
                    f"Final score: {score:.4f} "
                    f"({sum(1 for r in rubrics if r['passed'])}/{len(rubrics)} rubrics passed)"
                )

        self._last_message = msg

        # Update state tracking
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
    # Binary rubric evaluation (Scaler grading pattern)
    # ------------------------------------------------------------------

    def _evaluate_rubrics(self) -> List[Dict[str, Any]]:
        """Evaluate all grading rubrics. Each returns 0 (fail) or 1 (pass).

        Rubrics are dynamically generated based on the scenario's root causes
        so they scale correctly across easy/medium/hard tasks.
        """
        rubrics: List[Dict[str, Any]] = []

        # Per-root-cause rubrics
        for rc_name in self._scenario.root_cause_services:
            rubrics.append({
                "name": f"Root cause '{rc_name}' diagnosed",
                "eval_function": f"eval_{rc_name}_diagnosed",
                "score": 1.0 if rc_name in self._diagnosed else 0.0,
                "passed": rc_name in self._diagnosed,
            })
            rubrics.append({
                "name": f"Root cause '{rc_name}' fixed",
                "eval_function": f"eval_{rc_name}_fixed",
                "score": 1.0 if rc_name in self._fixed else 0.0,
                "passed": rc_name in self._fixed,
            })

        # All services restored
        all_healthy = all(s == "healthy" for s in self._services.values())
        rubrics.append({
            "name": "All services restored to healthy",
            "eval_function": "eval_all_services_restored",
            "score": 1.0 if all_healthy else 0.0,
            "passed": all_healthy,
        })

        # No incorrect fixes (agent didn't waste time fixing symptoms)
        no_bad_fixes = len(self._symptom_fixes) == 0
        rubrics.append({
            "name": "No incorrect symptom fixes attempted",
            "eval_function": "eval_no_incorrect_fixes",
            "score": 1.0 if no_bad_fixes else 0.0,
            "passed": no_bad_fixes,
        })

        # Step efficiency — completed within 60% of max steps
        if self._scenario:
            threshold = int(self._scenario.max_steps * 0.6)
            efficient = self._state.step_count <= threshold
            rubrics.append({
                "name": f"Resolved within {threshold} steps (60% of max)",
                "eval_function": "eval_step_efficiency",
                "score": 1.0 if efficient and all_healthy else 0.0,
                "passed": efficient and all_healthy,
            })

        return rubrics

    def _compute_grader_score(self, rubrics: List[Dict[str, Any]]) -> float:
        """Compute final score as passed_rubrics / total_rubrics."""
        if not rubrics:
            return 0.0
        passed = sum(1 for r in rubrics if r["passed"])
        score = passed / len(rubrics)
        self._state.accumulated_reward = score
        return score

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_check_status(self) -> str:
        """Return a summary of current service health."""
        healthy = [s for s, st in self._services.items() if st == "healthy"]
        degraded = [s for s, st in self._services.items() if st == "degraded"]
        down = [s for s, st in self._services.items() if st == "down"]
        return (
            f"Service Health Dashboard — "
            f"Healthy: {healthy or 'none'}, "
            f"Degraded: {degraded or 'none'}, "
            f"Down: {down or 'none'}"
        )

    def _handle_acknowledge(self, target: str) -> str:
        """Acknowledge alerts for the target service."""
        service_alerts = [
            a for a in self._scenario.alerts
            if a.service == target and a.alert_id not in self._acknowledged
        ]
        if not service_alerts:
            if any(a.service == target for a in self._scenario.alerts):
                return f"Alerts for '{target}' already acknowledged."
            return f"No alerts found for service '{target}'."

        for alert in service_alerts:
            self._acknowledged.add(alert.alert_id)

        return (
            f"Acknowledged {len(service_alerts)} alert(s) for '{target}'. "
            f"[{', '.join(a.severity.upper() for a in service_alerts)}]"
        )

    def _handle_diagnose(self, target: str) -> str:
        """Run diagnostics on a service. Reveals root cause information."""
        if target in self._diagnosed:
            return (
                f"Already diagnosed '{target}'. "
                f"Previous result: {self._diagnostic_results.get(target, 'N/A')}"
            )

        self._diagnosed.add(target)
        svc_def = self._service_defs[target]
        diag_output = svc_def.diagnostic_output or f"Service '{target}' diagnostics inconclusive."
        self._diagnostic_results[target] = diag_output

        return f"Diagnostics for '{target}': {diag_output}"

    def _handle_fix(self, target: str) -> str:
        """Apply a fix to a service."""
        if target in self._fixed:
            return f"Fix already applied to '{target}'. No additional action needed."

        if self._services[target] == "healthy":
            return f"Service '{target}' is already healthy. No fix required."

        self._fixed.add(target)

        if target in self._root_causes:
            self._services[target] = "healthy"
            return (
                f"Fix applied to root cause '{target}'. "
                f"Service restored to healthy. Checking downstream dependencies..."
            )
        else:
            # Track this as an incorrect symptom fix
            self._symptom_fixes.add(target)
            return (
                f"Fix attempted on '{target}', but this service's issues are caused "
                f"by an upstream dependency. Fix the root cause instead. "
                f"Service remains {self._services[target]}."
            )

    def _handle_escalate(self, target: str) -> str:
        """Escalate a service issue."""
        return (
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
                if all(self._services.get(d) == "healthy" for d in deps):
                    svc_def = self._service_defs[svc_name]
                    if svc_def.root_cause and svc_name not in self._fixed:
                        continue  # Still broken at its own level
                    self._services[svc_name] = "healthy"
                    changed = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, message: str = "") -> IncidentObservation:
        """Construct the observation returned to the agent."""
        msg = message or self._last_message
        healthy_count = sum(1 for s in self._services.values() if s == "healthy")

        # Compute rubrics and score when episode is done
        rubric_results = []
        final_reward = 0.0
        if self._done and self._scenario:
            rubric_results = self._evaluate_rubrics()
            final_reward = self._compute_grader_score(rubric_results)

        return IncidentObservation(
            done=self._done,
            reward=final_reward if self._done else None,
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
            rubric_results=rubric_results,
        )

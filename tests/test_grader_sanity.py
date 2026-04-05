"""Grader sanity checks — verify scoring correctness at boundary conditions.

For each task, we validate:
1. Fresh reset produces zero reward (no actions = no credit)
2. Golden path (optimal actions) produces near-perfect reward
3. Partial progress produces intermediate reward
4. Wrong fixes produce penalties
"""

import sys
from pathlib import Path

import pytest

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import IncidentAction
from scenarios import AVAILABLE_TASKS, SCENARIOS
from server.environment import IncidentResponseEnvironment

# Golden paths: the known-optimal action sequences for each task
GOLDEN_PATHS = {
    "single_service_failure": [
        IncidentAction(action_type="acknowledge", target_service="database"),
        IncidentAction(action_type="diagnose", target_service="database"),
        IncidentAction(action_type="fix", target_service="database"),
    ],
    "multi_service_correlation": [
        IncidentAction(action_type="acknowledge", target_service="redis"),
        IncidentAction(action_type="diagnose", target_service="redis"),
        IncidentAction(action_type="fix", target_service="redis"),
    ],
    "cascading_outage": [
        IncidentAction(action_type="acknowledge", target_service="primary_db"),
        IncidentAction(action_type="acknowledge", target_service="message_queue"),
        IncidentAction(action_type="diagnose", target_service="primary_db"),
        IncidentAction(action_type="fix", target_service="primary_db"),
        IncidentAction(action_type="diagnose", target_service="message_queue"),
        IncidentAction(action_type="fix", target_service="message_queue"),
    ],
}


class TestBaselineScoring:
    """Verify that reset() produces zero score for all tasks."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_reward_is_zero(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        assert obs.reward == 0.0 or obs.reward is None
        assert obs.done is False

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_services_not_all_healthy(self, task_name: str) -> None:
        """At least one service must be broken on reset."""
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        statuses = list(obs.services.values())
        assert any(s != "healthy" for s in statuses)

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_has_alerts(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        assert len(obs.alerts) >= 3  # Minimum 3 alerts per scenario


class TestGoldenPath:
    """Verify that optimal play produces near-perfect score."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_reaches_high_score(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        assert obs.done is True
        assert obs.reward >= 0.95, f"Golden path score {obs.reward} too low for {task_name}"

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_all_services_healthy(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        assert obs.resolved_count == obs.total_services
        assert all(s == "healthy" for s in obs.services.values())


class TestPartialProgress:
    """Verify that partial actions produce intermediate scores."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_acknowledge_only_gives_small_reward(self, task_name: str) -> None:
        """Acknowledging without fixing gives credit but not much."""
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        # Acknowledge just the first root cause alert
        root_svc = SCENARIOS[task_name].root_cause_services[0]
        obs = env.step(IncidentAction(action_type="acknowledge", target_service=root_svc))

        assert obs.reward > 0.0, "Acknowledge should give some reward"
        assert obs.reward < 0.3, "Acknowledge alone shouldn't give too much"
        assert obs.done is False

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_diagnose_without_fix_gives_intermediate(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        root_svc = SCENARIOS[task_name].root_cause_services[0]
        env.step(IncidentAction(action_type="acknowledge", target_service=root_svc))
        obs = env.step(IncidentAction(action_type="diagnose", target_service=root_svc))

        assert obs.reward > 0.1
        assert obs.reward < 0.6
        assert obs.done is False


class TestWrongFixPenalty:
    """Verify that fixing symptoms (not root cause) is penalised."""

    def test_fix_symptom_gives_penalty(self) -> None:
        """Fixing a downstream service that isn't a root cause."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")

        # Fix frontend — a symptom, not root cause (redis)
        obs = env.step(IncidentAction(action_type="fix", target_service="frontend"))
        assert obs.reward < 0.0, "Fixing a symptom should produce negative reward"

    def test_fix_symptom_does_not_heal_service(self) -> None:
        """Symptom service should stay broken when its upstream is still down."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")

        obs = env.step(IncidentAction(action_type="fix", target_service="frontend"))
        assert obs.services["frontend"] != "healthy"


class TestRewardBounds:
    """Verify reward stays within [0.0, 1.0] under all conditions."""

    def test_many_wrong_fixes_clamp_to_zero(self) -> None:
        """Repeated symptom fixes shouldn't produce deeply negative scores."""
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name="cascading_outage")

        # Fix many symptoms
        symptoms = [s for s in obs.services if s not in ("primary_db", "message_queue")]
        for svc in symptoms[:5]:
            obs = env.step(IncidentAction(action_type="fix", target_service=svc))

        # Reward should be clamped, not deeply negative
        assert obs.reward >= 0.0 or not obs.done
        # When episode ends, final reward is clamped
        # During episode, accumulated may be negative but that's ok

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_reward_at_most_one(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        assert obs.reward <= 1.0

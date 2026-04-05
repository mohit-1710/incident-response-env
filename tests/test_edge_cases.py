"""Edge case tests for the Incident Response environment.

Covers all boundary conditions: invalid inputs, repeated actions,
state management, cascading recovery, and episode lifecycle.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import IncidentAction
from scenarios import AVAILABLE_TASKS
from server.environment import IncidentResponseEnvironment


class TestInvalidActions:
    """Actions with bad input should return helpful errors, never crash."""

    def test_unknown_target_service(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type="diagnose", target_service="nonexistent"))
        assert "not found" in obs.message.lower()
        assert obs.done is False

    def test_fix_already_healthy_service(self) -> None:
        """Fixing a service that was never broken should be a no-op."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")
        # postgres is healthy in this scenario
        obs = env.step(IncidentAction(action_type="fix", target_service="postgres"))
        assert "already healthy" in obs.message.lower()

    def test_check_status_always_works(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type="check_status", target_service=""))
        assert "dashboard" in obs.message.lower()
        assert obs.done is False


class TestRepeatedActions:
    """Repeated identical actions should be idempotent."""

    def test_double_acknowledge(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")

        obs1 = env.step(IncidentAction(action_type="acknowledge", target_service="database"))
        r1 = obs1.reward
        obs2 = env.step(IncidentAction(action_type="acknowledge", target_service="database"))
        # Second ack should give no additional reward
        assert obs2.reward == r1, "Double acknowledge should not increase reward"
        assert "already acknowledged" in obs2.message.lower()

    def test_double_diagnose(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")

        obs1 = env.step(IncidentAction(action_type="diagnose", target_service="database"))
        r1 = obs1.reward
        obs2 = env.step(IncidentAction(action_type="diagnose", target_service="database"))
        assert obs2.reward == r1, "Double diagnose should not increase reward"
        assert "already diagnosed" in obs2.message.lower()

    def test_double_fix(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")

        env.step(IncidentAction(action_type="fix", target_service="frontend"))
        obs = env.step(IncidentAction(action_type="fix", target_service="frontend"))
        assert "already applied" in obs.message.lower()


class TestEpisodeLifecycle:
    """Test reset/step lifecycle and state isolation."""

    def test_step_after_done_returns_message(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type="acknowledge", target_service="database"))
        env.step(IncidentAction(action_type="diagnose", target_service="database"))
        obs = env.step(IncidentAction(action_type="fix", target_service="database"))
        assert obs.done is True

        obs2 = env.step(IncidentAction(action_type="fix", target_service="database"))
        assert obs2.done is True
        assert "already resolved" in obs2.message.lower()

    def test_multiple_resets_clean_state(self) -> None:
        """Each reset should produce a fresh episode with no leakage."""
        env = IncidentResponseEnvironment()

        # First episode: make some progress
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type="diagnose", target_service="database"))

        # Second episode: state should be clean
        obs = env.reset(task_name="multi_service_correlation")
        assert obs.reward == 0.0 or obs.reward is None
        assert obs.done is False
        assert len(obs.actions_taken) == 0
        assert len(obs.diagnostic_results) == 0
        assert "redis" in obs.services  # Different scenario loaded

    def test_reset_different_tasks(self) -> None:
        """Switching between tasks should load correct scenarios."""
        env = IncidentResponseEnvironment()

        obs1 = env.reset(task_name="single_service_failure")
        assert obs1.total_services == 3

        obs2 = env.reset(task_name="multi_service_correlation")
        assert obs2.total_services == 6

        obs3 = env.reset(task_name="cascading_outage")
        assert obs3.total_services == 12

    def test_unknown_task_name(self) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name="nonexistent_task")
        assert obs.done is True
        assert "unknown" in obs.message.lower() or "available" in obs.message.lower()

    def test_default_task_name(self) -> None:
        """Reset without task_name should default to easy task."""
        env = IncidentResponseEnvironment()
        obs = env.reset()
        assert obs.total_services == 3  # single_service_failure has 3 services
        assert obs.done is False


class TestMaxSteps:
    """Verify max_steps enforcement."""

    def test_max_steps_ends_episode(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")  # max_steps=15

        # Burn through all steps with check_status (no-op)
        for _ in range(15):
            obs = env.step(IncidentAction(action_type="check_status", target_service=""))

        assert obs.done is True
        assert "maximum steps" in obs.message.lower()
        assert 0.0 <= obs.reward <= 1.0

    def test_no_actions_scores_zero(self) -> None:
        """Just checking status repeatedly should score ~0."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")

        for _ in range(15):
            obs = env.step(IncidentAction(action_type="check_status", target_service=""))

        assert obs.reward < 0.05  # Essentially zero


class TestCascadingRecovery:
    """Verify dependency graph propagation works correctly."""

    def test_fixing_root_heals_downstream(self) -> None:
        """Fixing the root cause should auto-recover dependent services."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")

        # Fix database (root cause)
        env.step(IncidentAction(action_type="fix", target_service="database"))

        state = env.state
        assert state.services_healthy == state.services_total

    def test_fixing_symptom_does_not_heal_chain(self) -> None:
        """Fixing a downstream service won't help while root is still broken."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")

        # Try to fix api_gateway (symptom, depends on auth_service which depends on redis)
        obs = env.step(IncidentAction(action_type="fix", target_service="api_gateway"))
        assert obs.services["api_gateway"] != "healthy"

    def test_hard_task_partial_cascade(self) -> None:
        """Fixing one root cause heals its branch but not the other."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="cascading_outage")

        env.step(IncidentAction(action_type="fix", target_service="primary_db"))
        obs = env.step(IncidentAction(action_type="check_status", target_service=""))

        # primary_db branch should be healthy, message_queue branch still down
        assert obs.services["primary_db"] == "healthy"
        assert obs.services["app_server_1"] == "healthy"
        assert obs.services["message_queue"] == "down"
        assert obs.services["worker_pool"] == "down"
        assert obs.done is False


class TestStateEndpoint:
    """Verify the state() property returns correct data."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_state_has_episode_id(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)
        assert env.state.episode_id is not None
        assert len(env.state.episode_id) > 0

    def test_state_tracks_step_count(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        assert env.state.step_count == 0

        env.step(IncidentAction(action_type="check_status", target_service=""))
        assert env.state.step_count == 1

        env.step(IncidentAction(action_type="check_status", target_service=""))
        assert env.state.step_count == 2

    def test_state_tracks_task_name(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="cascading_outage")
        assert env.state.task_name == "cascading_outage"

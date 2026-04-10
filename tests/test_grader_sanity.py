"""Grader sanity checks — verify binary rubric scoring correctness.

For each task, we validate:
1. Fresh reset produces zero reward (no actions = no credit)
2. Golden path (optimal actions) produces all rubrics passing
3. Partial progress produces intermediate rubric pass rates
4. Wrong fixes cause the 'no_incorrect_fixes' rubric to fail
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import IncidentAction
from scenarios import AVAILABLE_TASKS, SCENARIOS
from server.environment import IncidentResponseEnvironment

# Golden paths: known-optimal action sequences for each task
GOLDEN_PATHS = {
    "single_service_failure": [
        IncidentAction(action_type="acknowledge", target_service="database"),
        IncidentAction(action_type="diagnose", target_service="database"),
        IncidentAction(action_type="fix", target_service="database"),
    ],
    "multi_service_correlation": [
        IncidentAction(action_type="acknowledge", target_service="redis"),
        IncidentAction(action_type="acknowledge", target_service="auth_service"),
        IncidentAction(action_type="diagnose", target_service="redis"),
        IncidentAction(action_type="fix", target_service="redis"),
    ],
    "cascading_outage": [
        IncidentAction(action_type="acknowledge", target_service="primary_db"),
        IncidentAction(action_type="acknowledge", target_service="message_queue"),
        IncidentAction(action_type="acknowledge", target_service="app_server_1"),
        IncidentAction(action_type="acknowledge", target_service="app_server_2"),
        IncidentAction(action_type="diagnose", target_service="primary_db"),
        IncidentAction(action_type="fix", target_service="primary_db"),
        IncidentAction(action_type="diagnose", target_service="message_queue"),
        IncidentAction(action_type="fix", target_service="message_queue"),
        IncidentAction(action_type="diagnose", target_service="session_store"),
        IncidentAction(action_type="fix", target_service="session_store"),
    ],
}


class TestBaselineScoring:
    """Verify that reset() produces zero score for all tasks."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_reward_is_zero(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        assert obs.reward is None or obs.reward == 0.0
        assert obs.done is False

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_services_not_all_healthy(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        assert any(s != "healthy" for s in obs.services.values())

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_reset_has_alerts(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        obs = env.reset(task_name=task_name)
        assert len(obs.alerts) >= 3


class TestGoldenPath:
    """Verify that optimal play produces all rubrics passing."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_all_rubrics_pass(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        assert obs.done is True
        # Score is mapped from [0, 1] -> (0.01, 0.99) so it stays strictly
        # inside the open interval (validator requires this).
        # All rubrics passing produces the maximum: 0.99.
        assert obs.reward == pytest.approx(0.99, abs=1e-6), (
            f"Golden path should pass all rubrics (score=0.99). "
            f"Got {obs.reward}. Failed rubrics: "
            f"{[r['name'] for r in obs.rubric_results if not r['passed']]}"
        )
        # All individual rubrics should still be 1.0 (binary)
        assert all(r["passed"] for r in obs.rubric_results)

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_all_services_healthy(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        assert obs.resolved_count == obs.total_services

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_rubric_details(self, task_name: str) -> None:
        """Each rubric should individually be 1.0."""
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)

        for rubric in obs.rubric_results:
            assert rubric["score"] == 1.0, f"Rubric '{rubric['name']}' failed"
            assert rubric["passed"] is True


class TestPartialProgress:
    """Verify that partial actions produce intermediate scores."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_diagnose_only_partial_score(self, task_name: str) -> None:
        """Diagnosing without fixing should pass some rubrics but not all."""
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)

        # Diagnose root cause but don't fix — then let max steps expire
        root_svc = SCENARIOS[task_name].root_cause_services[0]
        env.step(IncidentAction(action_type="diagnose", target_service=root_svc))

        # Burn remaining steps
        for _ in range(SCENARIOS[task_name].max_steps):
            obs = env.step(IncidentAction(action_type="check_status", target_service=""))
            if obs.done:
                break

        assert obs.done is True
        # Partial credit must land strictly inside the open interval
        assert 0.01 < obs.reward < 0.99


class TestWrongFixPenalty:
    """Verify that fixing symptoms causes the no_incorrect_fixes rubric to fail."""

    def test_symptom_fix_fails_rubric(self) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")

        # Fix symptom first, then fix root cause
        env.step(IncidentAction(action_type="fix", target_service="frontend"))
        env.step(IncidentAction(action_type="diagnose", target_service="redis"))
        obs = env.step(IncidentAction(action_type="fix", target_service="redis"))

        assert obs.done is True
        # Find the no_incorrect_fixes rubric — it should fail
        no_fix_rubric = next(
            r for r in obs.rubric_results if "incorrect" in r["name"].lower()
        )
        assert no_fix_rubric["passed"] is False
        # Symptom fix means at least one rubric fails -> score < 0.99 cap
        assert obs.reward < 0.99


class TestRewardBounds:
    """Verify reward stays strictly inside the open interval (0, 1)."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_golden_path_reward_at_max(self, task_name: str) -> None:
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)
        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)
        # Maximum possible score (all rubrics pass) is mapped to 0.99,
        # never exactly 1.0, so the validator's strict (0,1) check passes.
        assert obs.reward == pytest.approx(0.99, abs=1e-6)
        assert 0.0 < obs.reward < 1.0  # strictly inside the open interval

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_score_is_strictly_inside_open_interval(self, task_name: str) -> None:
        """No matter what actions, the final score must be in (0, 1) strictly."""
        env = IncidentResponseEnvironment()
        env.reset(task_name=task_name)
        for action in GOLDEN_PATHS[task_name]:
            obs = env.step(action)
        assert 0.0 < obs.reward, f"Score must be > 0, got {obs.reward}"
        assert obs.reward < 1.0, f"Score must be < 1, got {obs.reward}"

    def test_no_actions_scores_near_zero(self) -> None:
        """Doing nothing passes few rubrics — score lands near (but above) 0."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        for _ in range(15):
            obs = env.step(IncidentAction(action_type="check_status", target_service=""))
        assert obs.done is True
        # With the (0.01, 0.99) mapping, doing nothing gives a low but
        # strictly positive score.
        assert 0.0 < obs.reward <= 0.25
        assert obs.reward < 1.0


class TestRubricCounts:
    """Verify each task has the expected number of rubrics."""

    def test_easy_rubric_count(self) -> None:
        """Easy: 1×(diagnosed + fixed + diag_before_fix) + all_restored + no_incorrect + critical_ack + efficiency = 7."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="single_service_failure")
        for action in GOLDEN_PATHS["single_service_failure"]:
            obs = env.step(action)
        assert len(obs.rubric_results) == 7

    def test_medium_rubric_count(self) -> None:
        """Medium: 1×(diagnosed + fixed + diag_before_fix) + all_restored + no_incorrect + critical_ack + efficiency = 7."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="multi_service_correlation")
        for action in GOLDEN_PATHS["multi_service_correlation"]:
            obs = env.step(action)
        assert len(obs.rubric_results) == 7

    def test_hard_rubric_count(self) -> None:
        """Hard: 3×(diagnosed + fixed + diag_before_fix) + all_restored + no_incorrect + critical_ack + efficiency = 13."""
        env = IncidentResponseEnvironment()
        env.reset(task_name="cascading_outage")
        for action in GOLDEN_PATHS["cascading_outage"]:
            obs = env.step(action)
        assert len(obs.rubric_results) == 13

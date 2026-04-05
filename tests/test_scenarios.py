"""Scenario validity tests — verify each task definition is well-formed.

Checks structural properties of scenario data to catch misconfigurations
before they become runtime bugs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenarios import AVAILABLE_TASKS, SCENARIOS


class TestScenarioIntegrity:
    """Verify scenario data is consistent and well-formed."""

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_has_at_least_three_services(self, task_name: str) -> None:
        scenario = SCENARIOS[task_name]
        assert len(scenario.services) >= 3

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_has_at_least_three_alerts(self, task_name: str) -> None:
        scenario = SCENARIOS[task_name]
        assert len(scenario.alerts) >= 3

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_root_causes_exist_in_services(self, task_name: str) -> None:
        """Every declared root cause must correspond to an actual service."""
        scenario = SCENARIOS[task_name]
        service_names = {s.name for s in scenario.services}
        for rc in scenario.root_cause_services:
            assert rc in service_names, f"Root cause '{rc}' not in services"

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_root_cause_services_are_broken(self, task_name: str) -> None:
        """Root cause services must not start as healthy."""
        scenario = SCENARIOS[task_name]
        for svc in scenario.services:
            if svc.name in scenario.root_cause_services:
                assert svc.initial_status != "healthy", (
                    f"Root cause '{svc.name}' starts healthy — should be down or degraded"
                )

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_root_cause_alerts_exist(self, task_name: str) -> None:
        """There must be at least one alert flagged as root cause."""
        scenario = SCENARIOS[task_name]
        root_alerts = [a for a in scenario.alerts if a.is_root_cause]
        assert len(root_alerts) >= 1

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_dependencies_reference_valid_services(self, task_name: str) -> None:
        """All dependency references must point to services in the scenario."""
        scenario = SCENARIOS[task_name]
        service_names = {s.name for s in scenario.services}
        for svc in scenario.services:
            for dep in svc.depends_on:
                assert dep in service_names, (
                    f"Service '{svc.name}' depends on '{dep}' which doesn't exist"
                )

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_no_circular_dependencies(self, task_name: str) -> None:
        """Dependency graph must be a DAG — no cycles."""
        scenario = SCENARIOS[task_name]
        dep_map = {s.name: list(s.depends_on) for s in scenario.services}

        visited = set()
        in_stack = set()

        def has_cycle(node: str) -> bool:
            if node in in_stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_stack.add(node)
            for dep in dep_map.get(node, []):
                if has_cycle(dep):
                    return True
            in_stack.discard(node)
            return False

        for svc_name in dep_map:
            assert not has_cycle(svc_name), f"Circular dependency detected involving '{svc_name}'"

    @pytest.mark.parametrize("task_name", AVAILABLE_TASKS)
    def test_max_steps_is_reasonable(self, task_name: str) -> None:
        scenario = SCENARIOS[task_name]
        assert scenario.max_steps >= 10
        assert scenario.max_steps <= 100

    def test_difficulty_progression(self) -> None:
        """Harder tasks should have more services and root causes."""
        easy = SCENARIOS["single_service_failure"]
        medium = SCENARIOS["multi_service_correlation"]
        hard = SCENARIOS["cascading_outage"]

        assert len(easy.services) < len(medium.services) < len(hard.services)
        assert len(easy.root_cause_services) <= len(medium.root_cause_services) <= len(hard.root_cause_services)
        assert easy.max_steps <= medium.max_steps <= hard.max_steps

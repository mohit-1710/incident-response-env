"""Incident scenario definitions for each task difficulty level.

Each scenario is a self-contained description of a production incident:
which services exist, how they depend on each other, what broke (root causes),
and the alerts that fire as a result. The environment.py module uses these
to initialise episodes — this module is pure data, no logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Alert:
    """A single alert that fires during the incident."""

    alert_id: str
    severity: str          # "critical", "high", "medium", "low"
    service: str           # Which service triggered the alert
    message: str           # Human-readable description
    is_root_cause: bool    # Whether this alert points at an actual root cause


@dataclass(frozen=True)
class ServiceDef:
    """Definition of a service in the dependency graph."""

    name: str
    depends_on: List[str] = field(default_factory=list)
    initial_status: str = "down"    # "healthy", "degraded", "down"
    root_cause: str = ""            # Non-empty if this service is a root cause
    fix_action: str = ""            # Description of the fix required
    diagnostic_output: str = ""     # What `diagnose` reveals for this service


@dataclass(frozen=True)
class Scenario:
    """Complete incident scenario for one task."""

    task_id: str
    name: str
    description: str
    services: List[ServiceDef]
    alerts: List[Alert]
    max_steps: int
    root_cause_services: List[str]  # Names of services that are root causes


# ---------------------------------------------------------------------------
# Task 1: Single Service Failure (Easy)
# ---------------------------------------------------------------------------
# Linear chain: web_app → api_server → database
# Root cause: database has a connection timeout
# The alert messages hint strongly at the database being the issue.
# ---------------------------------------------------------------------------

SINGLE_SERVICE_FAILURE = Scenario(
    task_id="single_service_failure",
    name="Single Service Failure",
    description=(
        "Hey, just got paged — the web app is throwing 502s and the API "
        "server is returning errors on most requests. Looks like something "
        "is wrong with the backend. Can you take a look and get things "
        "back online? Should be straightforward, probably one thing causing "
        "the whole chain to fall over."
    ),
    services=[
        ServiceDef(
            name="database",
            depends_on=[],
            initial_status="down",
            root_cause="connection_timeout",
            fix_action="restart_service",
            diagnostic_output=(
                "CRITICAL: Connection pool exhausted. "
                "Max connections (100) reached. "
                "Oldest connection idle for 3600s. "
                "Restart required to clear stale connections."
            ),
        ),
        ServiceDef(
            name="api_server",
            depends_on=["database"],
            initial_status="degraded",
            diagnostic_output=(
                "ERROR: 78% of requests failing with DatabaseConnectionError. "
                "Upstream dependency 'database' is unreachable. "
                "Local health checks passing — issue is upstream."
            ),
        ),
        ServiceDef(
            name="web_app",
            depends_on=["api_server"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Serving 502 Bad Gateway for API-dependent routes. "
                "Static content serving normally. "
                "Upstream dependency 'api_server' returning errors."
            ),
        ),
    ],
    alerts=[
        Alert(
            alert_id="alert-001",
            severity="critical",
            service="database",
            message="Database connection pool exhausted — all connections in use, new requests rejected",
            is_root_cause=True,
        ),
        Alert(
            alert_id="alert-002",
            severity="high",
            service="api_server",
            message="API server error rate at 78% — upstream database dependency unreachable",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-003",
            severity="medium",
            service="web_app",
            message="Web application returning 502 errors on API-dependent routes",
            is_root_cause=False,
        ),
    ],
    max_steps=15,
    root_cause_services=["database"],
)


# ---------------------------------------------------------------------------
# Task 2: Multi-Service Correlation (Medium)
# ---------------------------------------------------------------------------
#                         ┌─ auth_service ─── redis (ROOT CAUSE)
# frontend ── api_gateway ┤
#                         └─ order_service ── postgres (healthy)
#
# Root cause: redis memory exhaustion. Auth fails, gateway degrades,
# frontend breaks. postgres and order_service are healthy but slow
# due to increased fallback traffic. Agent must NOT fix symptoms.
# ---------------------------------------------------------------------------

MULTI_SERVICE_CORRELATION = Scenario(
    task_id="multi_service_correlation",
    name="Multi-Service Correlation",
    description=(
        "Getting hammered with alerts on the e-commerce platform — frontend "
        "is broken for users, API gateway is rejecting most requests, and "
        "auth is completely down. Orders are still going through somehow but "
        "everything login-related is dead. The on-call channel is full of "
        "people saying 'just restart the frontend' but I don't think that's "
        "the actual problem. Can you trace this upstream and figure out what's "
        "actually broken? Don't waste time fixing symptoms."
    ),
    services=[
        ServiceDef(
            name="redis",
            depends_on=[],
            initial_status="down",
            root_cause="memory_exhaustion",
            fix_action="flush_and_restart",
            diagnostic_output=(
                "CRITICAL: Redis OOM — used_memory: 8.1GB / maxmemory: 8GB. "
                "Eviction policy 'noeviction' is rejecting writes. "
                "Key count: 12.4M (expected: ~2M). "
                "Memory leak suspected in session store. "
                "Requires flush of expired keys and restart."
            ),
        ),
        ServiceDef(
            name="auth_service",
            depends_on=["redis"],
            initial_status="down",
            diagnostic_output=(
                "ERROR: Cannot validate session tokens — Redis connection refused. "
                "All authenticated requests failing. "
                "Local service healthy — issue is in dependency 'redis'."
            ),
        ),
        ServiceDef(
            name="postgres",
            depends_on=[],
            initial_status="healthy",
            diagnostic_output=(
                "OK: Database operating normally. "
                "Connection pool: 23/100 active. "
                "Query latency p99: 45ms. "
                "No anomalies detected."
            ),
        ),
        ServiceDef(
            name="order_service",
            depends_on=["postgres"],
            initial_status="healthy",
            diagnostic_output=(
                "OK: Order processing functional. "
                "Slight increase in latency due to auth retries from upstream. "
                "All local dependencies healthy."
            ),
        ),
        ServiceDef(
            name="api_gateway",
            depends_on=["auth_service", "order_service"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: 62% of requests failing at auth middleware. "
                "Order endpoints responding but slow. "
                "Auth dependency 'auth_service' is down. "
                "Gateway itself is healthy — routing auth failures upstream."
            ),
        ),
        ServiceDef(
            name="frontend",
            depends_on=["api_gateway"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: User-facing errors on login, checkout, and profile pages. "
                "Public pages (home, product catalog) loading normally. "
                "Backend dependency 'api_gateway' returning mixed responses."
            ),
        ),
    ],
    alerts=[
        Alert(
            alert_id="alert-101",
            severity="critical",
            service="redis",
            message="Redis out of memory — maxmemory limit reached, write operations rejected",
            is_root_cause=True,
        ),
        Alert(
            alert_id="alert-102",
            severity="critical",
            service="auth_service",
            message="Authentication service down — cannot validate sessions, all auth requests failing",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-103",
            severity="high",
            service="api_gateway",
            message="API gateway error rate at 62% — auth middleware rejecting requests",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-104",
            severity="high",
            service="frontend",
            message="Frontend error rate spike — login and checkout flows broken for all users",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-105",
            severity="low",
            service="api_gateway",
            message="API gateway latency p99 increased from 200ms to 1.2s on order endpoints",
            is_root_cause=False,
        ),
    ],
    max_steps=25,
    root_cause_services=["redis"],
)


# ---------------------------------------------------------------------------
# Task 3: Cascading Outage (Hard)
# ---------------------------------------------------------------------------
# Two simultaneous root causes in a complex dependency web:
#
# cdn ── load_balancer ── web_server_1 ── app_server_1 ── primary_db (ROOT 1)
#                       └ web_server_2 ── app_server_2 ── primary_db
#                                       └ cache_layer ─── primary_db
# monitoring ── worker_pool ── message_queue (ROOT 2)
#                            └ notification_service
#
# primary_db is disk_full (critical), message_queue has consumer_deadlock (high)
# 12+ alerts fire including noise from monitoring flapping.
# Agent must prioritise primary_db (critical) over message_queue (high).
# ---------------------------------------------------------------------------

CASCADING_OUTAGE = Scenario(
    task_id="cascading_outage",
    name="Cascading Outage",
    description=(
        "Just got pulled into the war room — everything is on fire. The CDN "
        "is serving stale content, load balancer is returning 503s, both app "
        "servers are completely unresponsive, and the worker pool has 34K jobs "
        "backed up with an SLA breach in 13 minutes. Notification service is "
        "dead too so customers aren't even getting status updates. Someone in "
        "#infra-alerts mentioned disk space on one of the databases but I "
        "couldn't tell which one. Platform team thinks there might be TWO "
        "separate things going on — the web stack issues and the worker/queue "
        "issues might have different root causes. Start triaging by severity, "
        "I'll join in 10."
    ),
    services=[
        ServiceDef(
            name="primary_db",
            depends_on=[],
            initial_status="down",
            root_cause="disk_full",
            fix_action="cleanup_and_restart",
            diagnostic_output=(
                "CRITICAL: Disk usage at 100% on /data volume. "
                "WAL files consuming 89GB — checkpoint process stalled. "
                "All write operations blocked. Read-only queries timing out "
                "due to lock contention. Requires WAL cleanup and restart."
            ),
        ),
        ServiceDef(
            name="message_queue",
            depends_on=[],
            initial_status="down",
            root_cause="consumer_deadlock",
            fix_action="restart_consumers",
            diagnostic_output=(
                "HIGH: Consumer group 'order-processor' in deadlock state. "
                "Partition lag: 2.4M messages across 12 partitions. "
                "Consumer threads: 8/8 blocked on rebalance lock. "
                "Producers still writing — queue depth growing. "
                "Requires consumer group restart."
            ),
        ),
        ServiceDef(
            name="app_server_1",
            depends_on=["primary_db"],
            initial_status="down",
            diagnostic_output=(
                "ERROR: All database queries timing out after 30s. "
                "Connection pool: 50/50 active, all waiting on response. "
                "Thread pool exhausted. Upstream dependency 'primary_db' unresponsive."
            ),
        ),
        ServiceDef(
            name="app_server_2",
            depends_on=["primary_db"],
            initial_status="down",
            diagnostic_output=(
                "ERROR: Same as app_server_1 — database connections exhausted. "
                "Health check failing. Upstream dependency 'primary_db' unresponsive."
            ),
        ),
        ServiceDef(
            name="cache_layer",
            depends_on=["primary_db"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Cache hit rate dropped from 94% to 12%. "
                "Background refresh failing — cannot reach 'primary_db' for updates. "
                "Serving stale data for cached keys. Cache misses returning errors."
            ),
        ),
        ServiceDef(
            name="web_server_1",
            depends_on=["app_server_1"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: 91% of dynamic requests returning 503. "
                "Static assets serving normally. "
                "Upstream 'app_server_1' not responding."
            ),
        ),
        ServiceDef(
            name="web_server_2",
            depends_on=["app_server_2", "cache_layer"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Mixed failures — API routes down, cached content partially available. "
                "Dependencies 'app_server_2' down, 'cache_layer' degraded."
            ),
        ),
        ServiceDef(
            name="load_balancer",
            depends_on=["web_server_1", "web_server_2"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Both backend pools unhealthy. "
                "Active health checks failing for all upstream servers. "
                "Returning 503 to clients. Connection queue: 12,400 waiting."
            ),
        ),
        ServiceDef(
            name="cdn",
            depends_on=["load_balancer"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Origin fetch failures at 89%. "
                "Serving stale cached content where available. "
                "Edge nodes reporting 'load_balancer' origin unhealthy."
            ),
        ),
        ServiceDef(
            name="worker_pool",
            depends_on=["message_queue"],
            initial_status="down",
            diagnostic_output=(
                "ERROR: All workers idle — cannot consume from 'message_queue'. "
                "Pending job count: 34,200. "
                "Oldest job age: 47 minutes. SLA breach in 13 minutes."
            ),
        ),
        ServiceDef(
            name="notification_service",
            depends_on=["message_queue"],
            initial_status="down",
            diagnostic_output=(
                "ERROR: Cannot process notification events from 'message_queue'. "
                "Email/SMS/push queue backed up. "
                "28,000 undelivered notifications."
            ),
        ),
        ServiceDef(
            name="monitoring",
            depends_on=["worker_pool"],
            initial_status="degraded",
            diagnostic_output=(
                "WARNING: Metric ingestion delayed — worker_pool backlog. "
                "Dashboard data is 12 minutes stale. "
                "Alert evaluation running but on outdated metrics."
            ),
        ),
    ],
    alerts=[
        # Root cause alerts
        Alert(
            alert_id="alert-201",
            severity="critical",
            service="primary_db",
            message="Database disk full — /data volume at 100%, all write operations blocked",
            is_root_cause=True,
        ),
        Alert(
            alert_id="alert-202",
            severity="high",
            service="message_queue",
            message="Message queue consumer deadlock — partition lag 2.4M, all consumers blocked",
            is_root_cause=True,
        ),
        # Cascading symptom alerts
        Alert(
            alert_id="alert-203",
            severity="critical",
            service="app_server_1",
            message="App server 1 unresponsive — all database connections timed out",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-204",
            severity="critical",
            service="app_server_2",
            message="App server 2 unresponsive — database connection pool exhausted",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-205",
            severity="high",
            service="load_balancer",
            message="Load balancer returning 503 — all backend pools marked unhealthy",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-206",
            severity="high",
            service="web_server_1",
            message="Web server 1 error rate at 91% — upstream app server not responding",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-207",
            severity="high",
            service="worker_pool",
            message="Worker pool stalled — 34K pending jobs, SLA breach imminent",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-208",
            severity="medium",
            service="cdn",
            message="CDN origin fetch failure rate 89% — serving stale content",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-209",
            severity="medium",
            service="cache_layer",
            message="Cache hit rate dropped to 12% — background refresh failing",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-210",
            severity="medium",
            service="web_server_2",
            message="Web server 2 partially degraded — mixed dependency failures",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-211",
            severity="medium",
            service="notification_service",
            message="Notification delivery stopped — 28K messages queued and growing",
            is_root_cause=False,
        ),
        Alert(
            alert_id="alert-212",
            severity="low",
            service="monitoring",
            message="Monitoring data 12 minutes stale — metric ingestion worker backlogged",
            is_root_cause=False,
        ),
    ],
    max_steps=40,
    root_cause_services=["primary_db", "message_queue"],
)


# Registry of all available scenarios, keyed by task_id
SCENARIOS: Dict[str, Scenario] = {
    SINGLE_SERVICE_FAILURE.task_id: SINGLE_SERVICE_FAILURE,
    MULTI_SERVICE_CORRELATION.task_id: MULTI_SERVICE_CORRELATION,
    CASCADING_OUTAGE.task_id: CASCADING_OUTAGE,
}

AVAILABLE_TASKS = list(SCENARIOS.keys())

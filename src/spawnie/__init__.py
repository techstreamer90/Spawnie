"""
Spawnie - Model router for CLI agents.

Spawnie routes model requests to available providers. Callers request a model
(e.g., "claude-sonnet") and Spawnie finds the best available route (CLI
subscription, API key, etc.).

Setup:
    # Initialize config with defaults
    spawnie setup

    # Or programmatically
    from spawnie import setup_registry
    setup_registry()

Usage:
    from spawnie import run, list_models

    # See available models
    for model in list_models():
        print(f"{model['name']}: {model['available']} via {model['route']}")

    # Run a prompt (routes to best available provider)
    result = run("Explain quantum computing", model="claude-sonnet")
    print(result.output)

    # Run with quality levels for improved output
    result = run(
        "Design a security architecture",
        model="claude-sonnet",
        quality="hypertask",  # Dual review: self-review + external reviewer
    )

    # Quality levels:
    # - "normal": No review (fastest)
    # - "extra-clean": Self-review (agent reviews its own output)
    # - "hypertask": Dual review (self-review + external reviewer)

    # Async mode
    task_id = run("Long task", model="claude-opus", mode="async")
    result = wait_for_result(task_id)

    # Output directory mode
    result = run(
        "Analyze this and write a report",
        model="claude-sonnet",
        mode="output",
        output_dir="./output",
    )

Shell Sessions (Interactive Agents):
    from spawnie import spawn_shell, EventType

    # Spawn an agent with file system access
    session = spawn_shell(
        task="Analyze this codebase",
        model="claude-sonnet",
        working_dir=Path("./project"),
    )

    # Handle bidirectional communication
    for event in session.events():
        if event.type == EventType.QUESTION:
            session.respond(event.event_id, "your answer")
        elif event.type == EventType.DONE:
            print(f"Result: {event.data.get('result')}")
            break
"""

from .api import (
    # New API
    run,
    list_models,
    setup_registry,
    # Workflow API
    execute,
    workflow_status,
    kill,
    get_workflow_schema,
    get_workflow_guidance,
    list_workflows,
    list_tasks,
    # Shell Session API
    spawn_shell,
    # Legacy API (backwards compatible)
    spawn,
    spawn_async,
    get_result,
    get_status,
    wait_for_result,
)
from .config import SpawnieConfig
from .detection import (
    detect_cli,
    ensure_cli,
    get_installation_instructions,
    CLIStatus,
    CLINotFoundError,
)
from .models import Task, Result, QualityLevel, QUALITY_DESCRIPTIONS
from .queue import QueueManager
from .daemon import SpawnieDaemon, run_daemon
from .providers import (
    CLIProvider,
    ClaudeCLIProvider,
    CopilotCLIProvider,
    MockProvider,
    get_provider,
)
from .registry import (
    ModelRegistry,
    get_registry,
    Route,
    ModelConfig,
    ProviderConfig,
)
from .tracker import (
    Tracker,
    get_tracker,
    TaskState,
    WorkflowState,
    StepState,
    TrackerLimits,
)
from .workflow import (
    WorkflowDefinition,
    WorkflowExecutor,
    WorkflowResult,
    StepDefinition,
)
from .session import (
    ShellSession,
    SessionEvent,
    SessionStatus,
    EventType,
    list_sessions,
    get_session,
    cleanup_ended_sessions,
)

__version__ = "0.3.0"

__all__ = [
    # New API
    "run",
    "list_models",
    "setup_registry",
    # Workflow API
    "execute",
    "workflow_status",
    "kill",
    "get_workflow_schema",
    "get_workflow_guidance",
    "list_workflows",
    "list_tasks",
    # Shell Session API
    "spawn_shell",
    "ShellSession",
    "SessionEvent",
    "SessionStatus",
    "EventType",
    "list_sessions",
    "get_session",
    "cleanup_ended_sessions",
    # Legacy API
    "spawn",
    "spawn_async",
    "get_result",
    "get_status",
    "wait_for_result",
    # Configuration
    "SpawnieConfig",
    # Detection
    "detect_cli",
    "ensure_cli",
    "get_installation_instructions",
    "CLIStatus",
    "CLINotFoundError",
    # Models
    "Task",
    "Result",
    "QualityLevel",
    "QUALITY_DESCRIPTIONS",
    # Queue
    "QueueManager",
    # Daemon
    "SpawnieDaemon",
    "run_daemon",
    # Providers
    "CLIProvider",
    "ClaudeCLIProvider",
    "CopilotCLIProvider",
    "MockProvider",
    "get_provider",
    # Registry
    "ModelRegistry",
    "get_registry",
    "Route",
    "ModelConfig",
    "ProviderConfig",
    # Tracker
    "Tracker",
    "get_tracker",
    "TaskState",
    "WorkflowState",
    "StepState",
    "TrackerLimits",
    # Workflow
    "WorkflowDefinition",
    "WorkflowExecutor",
    "WorkflowResult",
    "StepDefinition",
]

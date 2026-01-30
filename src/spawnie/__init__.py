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
from .models import Task, Result
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

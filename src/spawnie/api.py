"""High-level API for Spawnie.

Spawnie is a model router that maps model requests to available providers.
Callers request a model (e.g., "claude-sonnet") and Spawnie routes to the
best available provider (CLI subscription, API, etc.).

All execution goes through workflows - even single prompts become single-step
workflows for consistent tracking and monitoring.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Literal

from .config import SpawnieConfig
from .models import Task, Result, QualityLevel
from .queue import QueueManager
from .providers import get_provider

logger = logging.getLogger("spawnie.api")


# Execution modes
Mode = Literal["blocking", "async", "output"]


def run(
    prompt: str,
    *,
    model: str = "claude-sonnet",
    mode: Mode = "blocking",
    timeout: int = 300,
    output_dir: Path | None = None,
    metadata: dict | None = None,
    quality: QualityLevel = "normal",
) -> Result | str:
    """
    Run a prompt using the best available route for the requested model.

    This is the main entry point for Spawnie. Internally, it creates a
    single-step workflow for consistent tracking and monitoring.

    Args:
        prompt: The prompt to send to the model.
        model: Model name (e.g., "claude-sonnet", "claude-opus", "gpt-4").
               Must be configured in ~/.spawnie/config.json
        mode: Execution mode:
              - "blocking": Wait for response (default)
              - "async": Return workflow_id immediately, caller polls later
              - "output": Agent writes to output_dir, blocks until done
        timeout: Maximum wait time in seconds (for blocking/output modes).
        output_dir: Directory for agent output (required for "output" mode).
        metadata: Optional metadata to attach to the task.
        quality: Quality level that controls review strategy:
              - "normal": No review, execute and return directly (fastest)
              - "extra-clean": Self-review - agent reviews its own output
              - "hypertask": Dual review - self-review + external reviewer (highest quality)

    Returns:
        - mode="blocking": Result object with output
        - mode="async": workflow_id (str) for later polling
        - mode="output": Result object (output is in output_dir)

    Raises:
        ValueError: If model not found or no route available.
        TimeoutError: If blocking/output mode times out.

    Example:
        # Simple blocking call
        result = run("Explain quantum computing", model="claude-sonnet")
        print(result.output)

        # High-quality task with dual review
        result = run(
            "Design a security architecture",
            model="claude-sonnet",
            quality="hypertask",  # Will apply self-review + external review
        )
    """
    # Import here to avoid circular imports
    from .workflow import StepDefinition, WorkflowDefinition, WorkflowExecutor

    # Handle output mode - wrap prompt with output instructions
    actual_prompt = prompt
    if mode == "output":
        if not output_dir:
            raise ValueError("output_dir required for mode='output'")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        actual_prompt = _wrap_prompt_for_output(prompt, output_dir)

    # Create a short description from the prompt (first 50 chars)
    description = prompt[:50].replace('\n', ' ').strip()
    if len(prompt) > 50:
        description += "..."

    # Create single-step workflow
    step = StepDefinition(
        name="main",
        prompt=actual_prompt,
        model=model,
        timeout=timeout,
        quality=quality,
    )

    workflow_def = WorkflowDefinition(
        name=description,
        description=description,
        steps={"main": step},
        outputs={"result": "{{steps.main.output}}"},
        timeout=timeout,
    )

    # Execute workflow
    executor = WorkflowExecutor()

    if mode == "async":
        # For async, we'd need to implement async workflow execution
        # For now, fall back to blocking and return workflow_id
        wf_result = executor.execute(workflow_def, inputs={}, timeout=timeout)
        return wf_result.workflow_id

    wf_result = executor.execute(workflow_def, inputs={}, timeout=timeout)

    # Convert WorkflowResult to Result for API compatibility
    step_result = wf_result.step_results.get("main", {})
    output = step_result.get("output", "")

    return Result(
        task_id=wf_result.workflow_id,
        status=wf_result.status,
        output=output if wf_result.status == "completed" else None,
        error=wf_result.error,
        duration_seconds=wf_result.duration_seconds,
    )


def _wrap_prompt_for_output(prompt: str, output_dir: Path) -> str:
    """Wrap prompt with instructions to write output to directory."""
    return f"""{prompt}

---
IMPORTANT: Write your output to the following directory: {output_dir.absolute()}

Create appropriate files for your response:
- For reports/analysis: Create a markdown file (e.g., report.md)
- For code: Create appropriately named source files
- For data: Create JSON/CSV files as appropriate

Do not include the output in your response - write it to files instead.
Confirm when you have written the files.
"""


# Keep old spawn() for backwards compatibility
def spawn(
    prompt: str,
    *,
    provider: str = "claude",
    model: str | None = None,
    wait: bool = True,
    timeout: int = 300,
    poll_interval: float = 0.5,
    response_dir: Path | None = None,
    queue_dir: Path | None = None,
    metadata: dict | None = None,
) -> Result | str:
    """
    Legacy spawn function - use run() for new code.

    This function bypasses the model registry and calls providers directly.
    Kept for backwards compatibility.
    """
    config = SpawnieConfig(
        provider=provider,
        model=model,
        response_dir=response_dir or Path("spawnie-responses"),
        queue_dir=queue_dir or Path(".spawnie/queue"),
        timeout=timeout,
        poll_interval=poll_interval,
    )

    config.ensure_dirs()

    task = Task(
        prompt=prompt,
        provider=provider,
        model=model,
        metadata=metadata or {},
    )

    queue = QueueManager(config.queue_dir.parent)
    task_id = queue.submit(task)

    if not wait:
        return task_id

    return _process_and_wait_legacy(task_id, config, timeout, poll_interval)


def _process_and_wait_legacy(
    task_id: str,
    config: SpawnieConfig,
    timeout: int,
    poll_interval: float,
) -> Result:
    """Legacy processing for spawn()."""
    queue = QueueManager(config.queue_dir.parent)
    provider = get_provider(config.provider)

    start_time = time.time()

    while True:
        result = queue.get_result(task_id)
        if result:
            return result

        elapsed = time.time() - start_time
        if elapsed > timeout:
            return queue.timeout(task_id, elapsed)

        task = queue.claim_next()
        if task and task.id == task_id:
            try:
                output, exit_code = provider.execute(
                    task.prompt,
                    task.model or config.model,
                )
                duration = time.time() - start_time

                if exit_code == 0:
                    return queue.complete(task_id, output, duration)
                else:
                    return queue.fail(task_id, output, duration)

            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:
                duration = time.time() - start_time
                return queue.fail(task_id, str(e), duration)

        time.sleep(poll_interval)


def spawn_async(
    prompt: str,
    *,
    provider: str = "claude",
    model: str | None = None,
    response_dir: Path | None = None,
    queue_dir: Path | None = None,
    metadata: dict | None = None,
) -> str:
    """Legacy async spawn - use run(mode="async") for new code."""
    return spawn(
        prompt,
        provider=provider,
        model=model,
        wait=False,
        response_dir=response_dir,
        queue_dir=queue_dir,
        metadata=metadata,
    )


def get_result(
    task_id: str,
    queue_dir: Path | None = None,
) -> Result | None:
    """
    Get the result for a previously spawned task.

    Args:
        task_id: The task ID returned by run(mode="async") or spawn(wait=False).
        queue_dir: Directory for queue files.

    Returns:
        The Result object if complete, None if still pending.
    """
    queue = QueueManager(queue_dir or Path(".spawnie"))
    return queue.get_result(task_id)


def get_status(
    task_id: str,
    queue_dir: Path | None = None,
) -> str | None:
    """
    Get the status of a task.

    Args:
        task_id: The task ID.
        queue_dir: Directory for queue files.

    Returns:
        Status string: "pending", "in_progress", "completed", "failed",
        or None if task not found.
    """
    queue = QueueManager(queue_dir or Path(".spawnie"))
    return queue.get_status(task_id)


def wait_for_result(
    task_id: str,
    timeout: int = 300,
    poll_interval: float = 0.5,
    queue_dir: Path | None = None,
) -> Result:
    """
    Wait for a task to complete and return its result.

    Args:
        task_id: The task ID.
        timeout: Maximum time to wait in seconds.
        poll_interval: How often to check for results.
        queue_dir: Directory for queue files.

    Returns:
        The Result object.

    Raises:
        TimeoutError: If the task doesn't complete within timeout.
    """
    queue = QueueManager(queue_dir or Path(".spawnie"))
    start_time = time.time()

    while True:
        result = queue.get_result(task_id)
        if result:
            return result

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

        time.sleep(poll_interval)


def list_models() -> list[dict]:
    """
    List all configured models and their availability.

    Returns:
        List of dicts with model info: name, description, available, route.
    """
    registry = get_registry()
    return registry.list_models()


def setup_registry():
    """
    Initialize and save the default registry configuration.

    Creates ~/.spawnie/config.json with default models and providers.
    """
    registry = get_registry()
    registry.save()
    return registry.config_path


# =============================================================================
# Workflow Execution API
# =============================================================================

def execute(
    workflow: dict | str | Path,
    inputs: dict,
    *,
    customer: str = "unknown",
    timeout: int | None = None,
) -> "WorkflowResult":
    """
    Execute a workflow definition.

    This is the main entry point for workflow execution. Customers define
    workflows in JSON, Spawnie executes them with tracking and safety.

    Args:
        workflow: Workflow definition as dict, JSON string, or path to file.
        inputs: Input values for the workflow.
        customer: Customer identifier for tracking/attribution.
        timeout: Override workflow timeout in seconds.

    Returns:
        WorkflowResult with outputs and execution details.

    Example:
        # From dict
        result = execute({
            "name": "analyze",
            "steps": {
                "analyze": {"prompt": "Analyze: {{inputs.data}}", "model": "claude-haiku"}
            },
            "outputs": {"result": "{{steps.analyze.output}}"}
        }, inputs={"data": "some text"}, customer="bam")

        # From file
        result = execute(Path("workflows/review.json"), inputs={...})
    """
    from .workflow import WorkflowDefinition, WorkflowExecutor, WorkflowResult
    from .tracker import get_tracker

    # Start tracker monitor if not running
    tracker = get_tracker()
    tracker.start_monitor()

    # Parse workflow definition
    if isinstance(workflow, dict):
        definition = WorkflowDefinition.from_dict(workflow)
    elif isinstance(workflow, Path):
        definition = WorkflowDefinition.from_file(workflow)
    elif isinstance(workflow, str):
        # Try as file path first, then as JSON
        path = Path(workflow)
        if path.exists():
            definition = WorkflowDefinition.from_file(path)
        else:
            definition = WorkflowDefinition.from_json(workflow)
    else:
        raise TypeError(f"workflow must be dict, str, or Path, not {type(workflow)}")

    # Execute
    executor = WorkflowExecutor()
    return executor.execute(definition, inputs, customer=customer, timeout=timeout)


def workflow_status(workflow_id: str | None = None) -> dict:
    """
    Get status of workflows and tasks.

    Args:
        workflow_id: Specific workflow ID, or None for overall status.

    Returns:
        Status dict with workflow/task information.
    """
    from .tracker import get_tracker

    tracker = get_tracker()

    if workflow_id:
        wf = tracker.get_workflow(workflow_id)
        if wf:
            return wf.to_dict()
        else:
            return {"error": f"Workflow not found: {workflow_id}"}

    return tracker.get_status()


def kill(target_id: str) -> dict:
    """
    Kill a running workflow or task.

    Args:
        target_id: Workflow ID (wf-xxx) or task ID (task-xxx) to kill.

    Returns:
        Status dict confirming the kill.
    """
    from .tracker import get_tracker

    tracker = get_tracker()

    if target_id.startswith("wf-"):
        tracker.kill_workflow(target_id)
        return {"status": "killed", "type": "workflow", "id": target_id}
    elif target_id.startswith("task-"):
        tracker.kill_task(target_id)
        return {"status": "killed", "type": "task", "id": target_id}
    else:
        raise ValueError(f"Unknown target type: {target_id} (expected wf-xxx or task-xxx)")


def get_workflow_schema() -> dict:
    """
    Get the JSON schema for workflow definitions.

    Customers can use this to validate their workflow definitions.
    """
    from .workflow import get_workflow_schema as _get_schema
    return _get_schema()


def get_workflow_guidance() -> str:
    """
    Get guidance text for agents on how to construct workflows.

    This returns documentation that agents can use to understand
    how to create valid workflow definitions.
    """
    from .workflow import get_workflow_guidance as _get_guidance
    return _get_guidance()


def list_workflows(customer: str | None = None) -> list[dict]:
    """
    List active workflows.

    Args:
        customer: Filter by customer, or None for all.

    Returns:
        List of workflow state dicts.
    """
    from .tracker import get_tracker

    tracker = get_tracker()
    workflows = tracker.list_workflows(customer)
    return [wf.to_dict() for wf in workflows]


def list_tasks(workflow_id: str | None = None) -> list[dict]:
    """
    List active tasks.

    Args:
        workflow_id: Filter by workflow, or None for all.

    Returns:
        List of task state dicts.
    """
    from .tracker import get_tracker

    tracker = get_tracker()
    tasks = tracker.list_tasks(workflow_id)
    return [t.to_dict() for t in tasks]


# =============================================================================
# Shell Session API
# =============================================================================

def spawn_shell(
    task: str,
    *,
    model: str = "claude-sonnet",
    provider: str | None = None,
    working_dir: Path | None = None,
    timeout: int = 3600,
) -> "ShellSession":
    """
    Spawn an interactive shell session with an agent.

    The agent runs in a shell with file system access and can communicate
    back using events (questions, progress, done).

    Args:
        task: The task/playbook for the agent to follow.
        model: Model to use (e.g., "claude-sonnet", "claude-opus").
        provider: Force specific provider ("claude", "copilot") or None for auto.
        working_dir: Working directory for the session (default: current dir).
        timeout: Maximum session duration in seconds (default: 1 hour).

    Returns:
        A ShellSession object for interacting with the agent.

    Example:
        session = spawn_shell(
            task="Analyze this codebase and create a summary",
            model="claude-sonnet",
            working_dir=Path("./my-project"),
        )

        for event in session.events():
            if event.type == EventType.QUESTION:
                answer = get_answer(event.message)
                session.respond(event.event_id, answer)
            elif event.type == EventType.PROGRESS:
                print(f"Progress: {event.message}")
            elif event.type == EventType.DONE:
                print(f"Done: {event.data.get('result')}")
                break
    """
    from .session import ShellSession

    session = ShellSession(working_dir=working_dir)
    session.start(task=task, model=model, provider=provider)
    return session

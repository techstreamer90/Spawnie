"""High-level API for Spawnie.

Spawnie is a model router that maps model requests to available providers.
Callers request a model (e.g., "claude-sonnet") and Spawnie routes to the
best available provider (CLI subscription, API, etc.).
"""

import subprocess
import time
import uuid
from pathlib import Path
from typing import Literal

from .config import SpawnieConfig
from .models import Task, Result, QualityLevel, QUALITY_DESCRIPTIONS
from .queue import QueueManager
from .providers import get_provider
from .registry import get_registry, Route, ProviderConfig


# Execution modes
Mode = Literal["blocking", "async", "output"]

# Review prompt templates based on benchmark findings
SELF_REVIEW_PROMPT = """You just completed a task. Before finalizing, critically review your own work:

ORIGINAL TASK: {original_prompt}

YOUR RESPONSE: {output}

SELF-REVIEW CHECKLIST:
1. What did you FORGET to consider? (security? cost? edge cases? alternatives?)
2. Are there CONTRADICTIONS in your response?
3. What ASSUMPTIONS are you making that might be wrong?
4. What is your CONFIDENCE level (Low/Medium/High) and why?

Provide an IMPROVED response that:
- Addresses any gaps you identified
- Acknowledges trade-offs
- States your confidence level
- Is clear about limitations

Your improved response:"""

EXTERNAL_REVIEW_PROMPT = """You are a SKEPTICAL SENIOR ARCHITECT reviewing work from another team member.

ORIGINAL TASK: {original_prompt}

THEIR RESPONSE (after self-review): {output}

Your job is to be ADVERSARIAL but constructive:
1. Find CONTRADICTIONS or inconsistencies
2. Identify ARCHITECTURAL RISKS not addressed
3. Challenge UNJUSTIFIED decisions
4. Rate overall confidence (Low/Medium/High)

Be critical. Assume something is wrong. Find it.

Your critique (max 300 words):"""

FINAL_SYNTHESIS_PROMPT = """You need to finalize your response incorporating review feedback.

ORIGINAL TASK: {original_prompt}

YOUR DRAFT (after self-review): {self_reviewed_output}

EXTERNAL REVIEWER CRITIQUE: {external_review}

Create a FINAL response that:
1. Addresses the reviewer's concerns
2. Acknowledges remaining trade-offs
3. States your confidence level with justification
4. Is honest about limitations

Your final response:"""


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

    This is the main entry point for Spawnie. It:
    1. Looks up the model in the registry
    2. Finds the best available route (CLI > API by default)
    3. Executes using that provider
    4. Optionally applies review strategies based on quality level
    5. Returns results based on mode

    Args:
        prompt: The prompt to send to the model.
        model: Model name (e.g., "claude-sonnet", "claude-opus", "gpt-4").
               Must be configured in ~/.spawnie/config.json
        mode: Execution mode:
              - "blocking": Wait for response (default)
              - "async": Return task_id immediately, caller polls later
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
        - mode="async": task_id (str) for later polling
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

        # Async call
        task_id = run("Long analysis", model="claude-opus", mode="async")
        # ... do other work ...
        result = get_result(task_id)

        # Output directory mode
        result = run(
            "Analyze this codebase and write a report",
            model="claude-sonnet",
            mode="output",
            output_dir=Path("./analysis-output"),
        )
    """
    registry = get_registry()

    # Find best route for model
    route_info = registry.get_best_route(model)
    if not route_info:
        available = [m["name"] for m in registry.list_models() if m["available"]]
        raise ValueError(
            f"No available route for model '{model}'. "
            f"Available models: {available or 'none (run spawnie setup)'}"
        )

    route, provider_config = route_info

    # Map to internal provider
    internal_provider = _get_internal_provider(provider_config)
    model_id = route.model_id  # Provider-specific model ID

    # Handle output mode - wrap prompt with output instructions
    actual_prompt = prompt
    if mode == "output":
        if not output_dir:
            raise ValueError("output_dir required for mode='output'")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        actual_prompt = _wrap_prompt_for_output(prompt, output_dir)

    # Execute based on mode
    if mode == "async":
        return _execute_async(actual_prompt, internal_provider, model_id, metadata, quality)
    else:
        result = _execute_blocking(actual_prompt, internal_provider, model_id, timeout, metadata)

        # Apply quality-level review strategies (only for successful blocking calls)
        if result.succeeded and quality != "normal":
            result = _apply_review_strategy(
                result=result,
                original_prompt=prompt,
                quality=quality,
                provider_name=internal_provider,
                model_id=model_id,
                timeout=timeout,
            )

        return result


def _get_internal_provider(provider_config: ProviderConfig) -> str:
    """Map registry provider config to internal provider name."""
    if provider_config.type == "mock":
        return "mock"

    if provider_config.type == "cli":
        if "claude" in provider_config.name:
            return "claude"
        elif "copilot" in provider_config.name:
            return "copilot"

    elif provider_config.type == "api":
        # For now, API providers not yet implemented
        # TODO: Add anthropic-api, openai-api providers
        if "anthropic" in provider_config.name:
            raise NotImplementedError(
                "Direct API providers not yet implemented. "
                "Use CLI providers or set up Claude/Copilot CLI."
            )

    return "mock"


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


def _execute_blocking(
    prompt: str,
    provider_name: str,
    model_id: str | None,
    timeout: int,
    metadata: dict | None,
) -> Result:
    """Execute a task and wait for completion."""
    from .tracker import get_tracker

    provider = get_provider(provider_name)
    tracker = get_tracker()

    # Generate task ID and register with tracker
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    model_name = model_id or provider_name

    # Create a short description from the prompt (first 50 chars)
    description = prompt[:50].replace('\n', ' ').strip()
    if len(prompt) > 50:
        description += "..."

    # Register task with tracker so it shows in monitor
    tracker.create_task(
        task_id=task_id,
        model=model_name,
        workflow_id=None,
        step=None,
        timeout=timeout,
        description=description,
    )
    tracker.start_task(task_id)

    start_time = time.time()

    try:
        output, exit_code = provider.execute(prompt, model_id)
        duration = time.time() - start_time

        if exit_code == 0:
            tracker.complete_task(task_id, output[:200] if output else None)
            return Result(
                task_id=task_id,
                status="completed",
                output=output,
                duration_seconds=duration,
            )
        else:
            tracker.fail_task(task_id, output[:200] if output else "Failed")
            return Result(
                task_id=task_id,
                status="failed",
                error=output,
                duration_seconds=duration,
            )

    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:
        duration = time.time() - start_time
        tracker.fail_task(task_id, str(e)[:200])
        return Result(
            task_id=task_id,
            status="failed",
            error=str(e),
            duration_seconds=duration,
        )


def _apply_review_strategy(
    result: Result,
    original_prompt: str,
    quality: QualityLevel,
    provider_name: str,
    model_id: str | None,
    timeout: int,
) -> Result:
    """
    Apply review strategy based on quality level.

    Quality levels:
    - "extra-clean": Self-review only (same model reviews its output)
    - "hypertask": Dual review (self-review + external reviewer)

    Based on benchmark findings:
    - Self-review catches omissions (security, cost, edge cases)
    - External review catches contradictions and architectural risks
    - Dual review combines both for highest quality
    """
    provider = get_provider(provider_name)
    start_time = time.time()
    total_duration = result.duration_seconds

    # Step 1: Self-review (for both extra-clean and hypertask)
    self_review_prompt = SELF_REVIEW_PROMPT.format(
        original_prompt=original_prompt,
        output=result.output,
    )

    try:
        self_reviewed_output, exit_code = provider.execute(self_review_prompt, model_id)
        total_duration += time.time() - start_time

        if exit_code != 0:
            # Self-review failed, return original result with note
            result.output = f"{result.output}\n\n[Self-review failed: {self_reviewed_output}]"
            result.duration_seconds = total_duration
            return result

        # For extra-clean, we're done after self-review
        if quality == "extra-clean":
            return Result(
                task_id=result.task_id,
                status="completed",
                output=self_reviewed_output,
                duration_seconds=total_duration,
            )

        # Step 2: External review (for hypertask only)
        external_review_prompt = EXTERNAL_REVIEW_PROMPT.format(
            original_prompt=original_prompt,
            output=self_reviewed_output,
        )

        start_review = time.time()
        # Use a more capable model for external review if available
        review_model = model_id  # Could upgrade to opus here
        external_review, exit_code = provider.execute(external_review_prompt, review_model)
        total_duration += time.time() - start_review

        if exit_code != 0:
            # External review failed, return self-reviewed result
            return Result(
                task_id=result.task_id,
                status="completed",
                output=self_reviewed_output,
                duration_seconds=total_duration,
            )

        # Step 3: Final synthesis incorporating external review
        synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(
            original_prompt=original_prompt,
            self_reviewed_output=self_reviewed_output,
            external_review=external_review,
        )

        start_synthesis = time.time()
        final_output, exit_code = provider.execute(synthesis_prompt, model_id)
        total_duration += time.time() - start_synthesis

        if exit_code != 0:
            # Synthesis failed, return self-reviewed result with review attached
            return Result(
                task_id=result.task_id,
                status="completed",
                output=f"{self_reviewed_output}\n\n---\nEXTERNAL REVIEW:\n{external_review}",
                duration_seconds=total_duration,
            )

        return Result(
            task_id=result.task_id,
            status="completed",
            output=final_output,
            duration_seconds=total_duration,
        )

    except Exception as e:
        # Review failed, return original result with error note
        result.output = f"{result.output}\n\n[Review process error: {e}]"
        result.duration_seconds = total_duration
        return result


def _execute_async(
    prompt: str,
    provider_name: str,
    model_id: str | None,
    metadata: dict | None,
    quality: QualityLevel = "normal",
) -> str:
    """Submit a task for async execution, return task_id."""
    config = SpawnieConfig(
        provider=provider_name,
        model=model_id,
    )
    config.ensure_dirs()

    task = Task(
        prompt=prompt,
        provider=provider_name,
        model=model_id,
        quality_level=quality,
        metadata=metadata or {},
    )

    queue = QueueManager(config.queue_dir.parent)
    return queue.submit(task)


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

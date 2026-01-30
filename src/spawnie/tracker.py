"""Real-time task and workflow tracking for Spawnie.

The tracker maintains a live view of all running tasks and workflows,
enforces timeouts, and provides observability into Spawnie's state.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import logging

logger = logging.getLogger("spawnie.tracker")

DEFAULT_TRACKER_PATH = Path.home() / ".spawnie" / "tracker.json"
DEFAULT_HISTORY_DIR = Path.home() / ".spawnie" / "history"


@dataclass
class TaskState:
    """State of a single task."""
    id: str
    workflow_id: str | None
    step: str | None
    model: str
    status: str  # "queued" | "running" | "completed" | "failed" | "timeout" | "killed"
    created_at: datetime
    description: str | None = None  # Human-readable task description
    parent_task_id: str | None = None  # Parent task ID for hierarchical display
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_at: datetime | None = None
    pid: int | None = None
    error: str | None = None
    output_preview: str | None = None  # First 200 chars of output

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "step": self.step,
            "model": self.model,
            "status": self.status,
            "description": self.description,
            "parent_task_id": self.parent_task_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "pid": self.pid,
            "error": self.error,
            "output_preview": self.output_preview,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        return cls(
            id=data["id"],
            workflow_id=data.get("workflow_id"),
            step=data.get("step"),
            model=data["model"],
            status=data["status"],
            description=data.get("description"),
            parent_task_id=data.get("parent_task_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            timeout_at=datetime.fromisoformat(data["timeout_at"]) if data.get("timeout_at") else None,
            pid=data.get("pid"),
            error=data.get("error"),
            output_preview=data.get("output_preview"),
        )


@dataclass
class StepState:
    """State of a workflow step."""
    name: str
    status: str  # "pending" | "running" | "completed" | "failed" | "skipped"
    task_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "task_id": self.task_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepState":
        return cls(
            name=data["name"],
            status=data["status"],
            task_id=data.get("task_id"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration=data.get("duration"),
            error=data.get("error"),
        )


@dataclass
class WorkflowState:
    """State of a workflow execution."""
    id: str
    name: str
    customer: str
    status: str  # "queued" | "running" | "completed" | "failed" | "timeout" | "killed"
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_at: datetime | None = None
    steps: dict[str, StepState] = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def progress(self) -> str:
        completed = sum(1 for s in self.steps.values() if s.status == "completed")
        total = len(self.steps)
        return f"{completed}/{total} steps"

    @property
    def current_step(self) -> str | None:
        for name, step in self.steps.items():
            if step.status == "running":
                return name
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "customer": self.customer,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout_at": self.timeout_at.isoformat() if self.timeout_at else None,
            "progress": self.progress,
            "current_step": self.current_step,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        steps = {name: StepState.from_dict(s) for name, s in data.get("steps", {}).items()}
        return cls(
            id=data["id"],
            name=data["name"],
            customer=data["customer"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            timeout_at=datetime.fromisoformat(data["timeout_at"]) if data.get("timeout_at") else None,
            steps=steps,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            error=data.get("error"),
        )


@dataclass
class TrackerLimits:
    """Resource limits for the tracker."""
    max_concurrent_tasks: int = 10
    max_concurrent_workflows: int = 5
    default_task_timeout: int = 300  # 5 minutes
    default_workflow_timeout: int = 1800  # 30 minutes

    def to_dict(self) -> dict:
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "default_task_timeout": self.default_task_timeout,
            "default_workflow_timeout": self.default_workflow_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackerLimits":
        return cls(
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10),
            max_concurrent_workflows=data.get("max_concurrent_workflows", 5),
            default_task_timeout=data.get("default_task_timeout", 300),
            default_workflow_timeout=data.get("default_workflow_timeout", 1800),
        )


class Tracker:
    """Central tracker for all Spawnie tasks and workflows.

    Maintains real-time state in a JSON file that can be monitored
    by external tools. Handles timeouts and provides safety limits.
    """

    @staticmethod
    def _default_stats() -> dict:
        """Return default stats structure."""
        return {
            # Counts
            "completed_today": 0,
            "failed_today": 0,
            # Runtime
            "total_runtime_seconds": 0.0,
            "avg_duration_seconds": 0.0,
            # Model usage
            "model_usage": {},  # {"claude-haiku": 5, "claude-sonnet": 3}
            # Session info
            "session_start": datetime.now().isoformat(),
            "last_activity": None,
        }

    def __init__(
        self,
        tracker_path: Path | None = None,
        history_dir: Path | None = None,
        limits: TrackerLimits | None = None,
    ):
        self.tracker_path = tracker_path or DEFAULT_TRACKER_PATH
        self.history_dir = history_dir or DEFAULT_HISTORY_DIR
        self.limits = limits or TrackerLimits()

        self.workflows: dict[str, WorkflowState] = {}
        self.tasks: dict[str, TaskState] = {}
        self.alerts: list[dict] = []
        self.stats = self._default_stats()

        self._lock = threading.RLock()
        self._monitor_thread: threading.Thread | None = None
        self._shutdown = False
        self._last_save = datetime.now()

        # Ensure directories exist
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state if present
        self._load()

    def _load(self, cleanup: bool = True):
        """Load tracker state from file."""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.limits = TrackerLimits.from_dict(data.get("limits", {}))
                self.workflows = {
                    k: WorkflowState.from_dict(v)
                    for k, v in data.get("workflows", {}).items()
                }
                self.tasks = {
                    k: TaskState.from_dict(v)
                    for k, v in data.get("tasks", {}).items()
                }
                # Merge loaded stats with defaults (for backwards compatibility)
                loaded_stats = data.get("stats", {})
                self.stats = {**self._default_stats(), **loaded_stats}
                # Ensure model_usage is a dict (could be missing in old files)
                if not isinstance(self.stats.get("model_usage"), dict):
                    self.stats["model_usage"] = {}
                self.alerts = data.get("alerts", [])

                # Clean up completed/failed items from previous runs (only on initial load)
                if cleanup:
                    self._cleanup_finished()

                logger.debug("Loaded tracker state: %d workflows, %d tasks",
                            len(self.workflows), len(self.tasks))
            except Exception as e:
                logger.warning("Could not load tracker state: %s", e)

    def reload(self):
        """Reload tracker state from file (for monitoring from another process)."""
        with self._lock:
            self._load(cleanup=False)

    def _cleanup_finished(self):
        """Move finished workflows/tasks from previous sessions to history.

        Also detects orphaned tasks where the process is no longer running.
        """
        finished_statuses = {"completed", "failed", "timeout", "killed"}

        # Clean finished workflows
        for wf_id, wf in list(self.workflows.items()):
            if wf.status in finished_statuses:
                self._archive_workflow(wf)
                del self.workflows[wf_id]

        # Clean finished or orphaned tasks
        for task_id, task in list(self.tasks.items()):
            if task.status in finished_statuses:
                self._archive_task(task)
                del self.tasks[task_id]
            # Check for orphaned running tasks (process dead)
            elif task.status == "running" and task.pid:
                if not self._is_process_alive(task.pid):
                    task.status = "failed"
                    task.error = f"Process {task.pid} no longer running (orphaned on startup)"
                    task.completed_at = datetime.now()
                    self._archive_task(task)
                    del self.tasks[task_id]
                    logger.warning("Cleaned orphaned task %s (PID %s dead)", task_id, task.pid)

        # Clean orphaned workflows (all tasks dead)
        for wf_id, wf in list(self.workflows.items()):
            if wf.status == "running":
                # Check if any step has a running task that's still alive
                has_alive_task = False
                for step in wf.steps.values():
                    if step.task_id and step.task_id in self.tasks:
                        has_alive_task = True
                        break

                if not has_alive_task and wf.current_step:
                    wf.status = "failed"
                    wf.error = "All tasks orphaned on startup"
                    wf.completed_at = datetime.now()
                    self._archive_workflow(wf)
                    del self.workflows[wf_id]
                    logger.warning("Cleaned orphaned workflow %s", wf_id)
            elif task.workflow_id and task.workflow_id not in self.workflows:
                del self.tasks[task_id]

    def _archive_workflow(self, workflow: WorkflowState):
        """Archive a completed workflow to history."""
        today = datetime.now().strftime("%Y-%m-%d")
        history_file = self.history_dir / f"{today}.jsonl"

        with open(history_file, "a", encoding="utf-8") as f:
            record = {
                "type": "workflow",
                "archived_at": datetime.now().isoformat(),
                **workflow.to_dict(),
            }
            f.write(json.dumps(record) + "\n")

    def _archive_task(self, task: TaskState):
        """Archive a completed task to history."""
        today = datetime.now().strftime("%Y-%m-%d")
        history_file = self.history_dir / f"{today}.jsonl"

        with open(history_file, "a", encoding="utf-8") as f:
            record = {
                "type": "task",
                "archived_at": datetime.now().isoformat(),
                **task.to_dict(),
            }
            f.write(json.dumps(record) + "\n")

    def save(self):
        """Save current state to tracker file with retry for Windows file locking."""
        import tempfile
        import random

        with self._lock:
            data = {
                "updated_at": datetime.now().isoformat(),
                "spawnie_pid": os.getpid(),
                "limits": self.limits.to_dict(),
                "workflows": {k: v.to_dict() for k, v in self.workflows.items()},
                "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
                "stats": {
                    **self.stats,
                    "running_workflows": sum(1 for w in self.workflows.values() if w.status == "running"),
                    "running_tasks": sum(1 for t in self.tasks.values() if t.status == "running"),
                    "queued_tasks": sum(1 for t in self.tasks.values() if t.status == "queued"),
                },
                "alerts": self.alerts[-10:],  # Keep last 10 alerts
            }

            # Retry logic for Windows file locking issues
            max_retries = 5
            last_error = None

            for attempt in range(max_retries):
                fd, tmp_path = tempfile.mkstemp(
                    suffix=".tmp",
                    prefix="tracker_",
                    dir=self.tracker_path.parent,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    Path(tmp_path).replace(self.tracker_path)
                    self._last_save = datetime.now()
                    return  # Success
                except OSError as e:
                    last_error = e
                    # Clean up temp file on failure
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    # Exponential backoff with jitter
                    if attempt < max_retries - 1:
                        delay = (0.1 * (2 ** attempt)) + (random.random() * 0.1)
                        time.sleep(delay)

            # All retries failed
            logger.warning("Failed to save tracker after %d attempts: %s", max_retries, last_error)
            raise last_error

    def _add_alert(self, level: str, message: str):
        """Add an alert to the tracker."""
        new_alert = {
            "level": level,
            "message": message,
            "at": datetime.now().isoformat(),
        }
        self.alerts = (self.alerts + [new_alert])[-100:]
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warning(message)
        else:
            logger.info(message)

    # -------------------------------------------------------------------------
    # Workflow Management
    # -------------------------------------------------------------------------

    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        customer: str,
        step_names: list[str],
        inputs: dict,
        timeout: int | None = None,
    ) -> WorkflowState:
        """Create and register a new workflow."""
        with self._lock:
            # Check limits
            running = sum(1 for w in self.workflows.values() if w.status == "running")
            if running >= self.limits.max_concurrent_workflows:
                raise RuntimeError(
                    f"Workflow limit reached ({self.limits.max_concurrent_workflows}). "
                    "Wait for running workflows to complete."
                )

            now = datetime.now()
            timeout_secs = timeout or self.limits.default_workflow_timeout

            workflow = WorkflowState(
                id=workflow_id,
                name=name,
                customer=customer,
                status="queued",
                created_at=now,
                timeout_at=now + timedelta(seconds=timeout_secs),
                steps={name: StepState(name=name, status="pending") for name in step_names},
                inputs=inputs,
            )

            self.workflows[workflow_id] = workflow
            self.save()

            logger.info("Created workflow %s (%s) for %s", workflow_id, name, customer)
            return workflow

    def start_workflow(self, workflow_id: str):
        """Mark a workflow as started."""
        with self._lock:
            if workflow_id not in self.workflows:
                raise KeyError(f"Unknown workflow: {workflow_id}")

            workflow = self.workflows[workflow_id]
            workflow.status = "running"
            workflow.started_at = datetime.now()
            self.save()

    def complete_workflow(self, workflow_id: str, outputs: dict):
        """Mark a workflow as completed."""
        with self._lock:
            if workflow_id not in self.workflows:
                return

            workflow = self.workflows[workflow_id]
            workflow.status = "completed"
            workflow.completed_at = datetime.now()
            workflow.outputs = outputs

            # Calculate duration and update stats (model tracked at task level)
            duration = 0.0
            if workflow.started_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()

            self.stats["completed_today"] += 1
            self.stats["total_runtime_seconds"] += duration
            total = self.stats["completed_today"] + self.stats["failed_today"]
            self.stats["avg_duration_seconds"] = round(self.stats["total_runtime_seconds"] / total, 2)
            self.stats["last_activity"] = datetime.now().isoformat()

            self._archive_workflow(workflow)
            del self.workflows[workflow_id]
            self.save()

            logger.info("Workflow %s completed", workflow_id)

    def fail_workflow(self, workflow_id: str, error: str):
        """Mark a workflow as failed."""
        with self._lock:
            if workflow_id not in self.workflows:
                return

            workflow = self.workflows[workflow_id]
            workflow.status = "failed"
            workflow.completed_at = datetime.now()
            workflow.error = error

            # Calculate duration and update stats (model tracked at task level)
            duration = 0.0
            if workflow.started_at:
                duration = (workflow.completed_at - workflow.started_at).total_seconds()

            self.stats["failed_today"] += 1
            self.stats["total_runtime_seconds"] += duration
            total = self.stats["completed_today"] + self.stats["failed_today"]
            if total > 0:
                self.stats["avg_duration_seconds"] = round(self.stats["total_runtime_seconds"] / total, 2)
            self.stats["last_activity"] = datetime.now().isoformat()

            self._add_alert("error", f"Workflow {workflow_id} failed: {error}")
            self._archive_workflow(workflow)
            del self.workflows[workflow_id]
            self.save()

    def kill_workflow(self, workflow_id: str, reason: str = "killed by user"):
        """Kill a workflow and all its tasks."""
        with self._lock:
            if workflow_id not in self.workflows:
                raise KeyError(f"Unknown workflow: {workflow_id}")

            workflow = self.workflows[workflow_id]
            workflow.status = "killed"
            workflow.completed_at = datetime.now()
            workflow.error = reason

            # Kill and remove all associated tasks
            for task_id, task in list(self.tasks.items()):
                if task.workflow_id == workflow_id:
                    task.status = "killed"
                    task.completed_at = datetime.now()
                    task.error = reason
                    del self.tasks[task_id]

            self._add_alert("warn", f"Workflow {workflow_id} killed: {reason}")
            self._archive_workflow(workflow)
            del self.workflows[workflow_id]
            self.save()

    # -------------------------------------------------------------------------
    # Step Management
    # -------------------------------------------------------------------------

    def start_step(self, workflow_id: str, step_name: str, task_id: str):
        """Mark a workflow step as started."""
        with self._lock:
            if workflow_id not in self.workflows:
                return

            workflow = self.workflows[workflow_id]
            if step_name in workflow.steps:
                step = workflow.steps[step_name]
                step.status = "running"
                step.started_at = datetime.now()
                step.task_id = task_id
                self.save()

    def complete_step(self, workflow_id: str, step_name: str):
        """Mark a workflow step as completed."""
        with self._lock:
            if workflow_id not in self.workflows:
                return

            workflow = self.workflows[workflow_id]
            if step_name in workflow.steps:
                step = workflow.steps[step_name]
                step.status = "completed"
                step.completed_at = datetime.now()
                if step.started_at:
                    step.duration = (step.completed_at - step.started_at).total_seconds()
                self.save()

    def fail_step(self, workflow_id: str, step_name: str, error: str):
        """Mark a workflow step as failed."""
        with self._lock:
            if workflow_id not in self.workflows:
                return

            workflow = self.workflows[workflow_id]
            if step_name in workflow.steps:
                step = workflow.steps[step_name]
                step.status = "failed"
                step.completed_at = datetime.now()
                step.error = error
                if step.started_at:
                    step.duration = (step.completed_at - step.started_at).total_seconds()
                self.save()

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def create_task(
        self,
        task_id: str,
        model: str,
        workflow_id: str | None = None,
        step: str | None = None,
        timeout: int | None = None,
        description: str | None = None,
        parent_task_id: str | None = None,
    ) -> TaskState:
        """Create and register a new task."""
        with self._lock:
            # Check limits
            running = sum(1 for t in self.tasks.values() if t.status == "running")
            if running >= self.limits.max_concurrent_tasks:
                raise RuntimeError(
                    f"Task limit reached ({self.limits.max_concurrent_tasks}). "
                    "Wait for running tasks to complete."
                )

            now = datetime.now()
            timeout_secs = timeout or self.limits.default_task_timeout

            task = TaskState(
                id=task_id,
                workflow_id=workflow_id,
                step=step,
                model=model,
                status="queued",
                description=description,
                parent_task_id=parent_task_id,
                created_at=now,
                timeout_at=now + timedelta(seconds=timeout_secs),
            )

            self.tasks[task_id] = task
            self.save()
            return task

    def start_task(self, task_id: str, pid: int | None = None):
        """Mark a task as started."""
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = "running"
            task.started_at = datetime.now()
            task.pid = pid or os.getpid()
            self.save()

    def complete_task(self, task_id: str, output_preview: str | None = None):
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.now()
            task.output_preview = output_preview[:200] if output_preview else None

            # Track model usage (counts/runtime tracked at workflow level)
            if task.model:
                model_key = task.model.lower()
                self.stats["model_usage"][model_key] = self.stats["model_usage"].get(model_key, 0) + 1
            self.stats["last_activity"] = datetime.now().isoformat()

            # Archive to history before removing
            self._archive_task(task)

            # Remove from active tracking
            del self.tasks[task_id]
            self.save()

    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = "failed"
            task.completed_at = datetime.now()
            task.error = error

            # Track model usage (counts/runtime tracked at workflow level)
            if task.model:
                model_key = task.model.lower()
                self.stats["model_usage"][model_key] = self.stats["model_usage"].get(model_key, 0) + 1
            self.stats["last_activity"] = datetime.now().isoformat()

            # Archive to history before removing
            self._archive_task(task)

            self._add_alert("error", f"Task {task_id} failed: {error[:100]}")
            del self.tasks[task_id]
            self.save()

    def kill_task(self, task_id: str, reason: str = "killed by user"):
        """Kill a specific task."""
        with self._lock:
            if task_id not in self.tasks:
                raise KeyError(f"Unknown task: {task_id}")

            task = self.tasks[task_id]
            task.status = "killed"
            task.completed_at = datetime.now()
            task.error = reason

            # Archive to history before removing
            self._archive_task(task)

            self._add_alert("warn", f"Task {task_id} killed: {reason}")
            del self.tasks[task_id]
            self.save()

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    def _is_process_alive(self, pid: int | None) -> bool:
        """Check if a process with given PID is still running."""
        if pid is None:
            return False
        try:
            # On Windows and Unix, this checks if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def check_timeouts(self) -> list[str]:
        """Check for timed-out and orphaned tasks/workflows. Returns IDs of cleaned items."""
        cleaned = []
        now = datetime.now()

        with self._lock:
            # Check workflows
            for wf_id, wf in list(self.workflows.items()):
                if wf.status != "running":
                    continue

                # Check timeout
                if wf.timeout_at and now > wf.timeout_at:
                    wf.status = "timeout"
                    wf.completed_at = now
                    wf.error = "Workflow timed out"
                    self._add_alert("error", f"Workflow {wf_id} timed out")
                    self._archive_workflow(wf)
                    del self.workflows[wf_id]
                    cleaned.append(wf_id)

            # Check tasks
            for task_id, task in list(self.tasks.items()):
                if task.status != "running":
                    continue

                should_clean = False
                reason = None

                # Check timeout
                if task.timeout_at and now > task.timeout_at:
                    should_clean = True
                    reason = "Task timed out"
                # Check if process is dead (orphan detection)
                elif task.pid and not self._is_process_alive(task.pid):
                    should_clean = True
                    reason = f"Process {task.pid} no longer running (orphaned)"

                if should_clean:
                    task.status = "timeout" if "timed out" in reason else "failed"
                    task.completed_at = now
                    task.error = reason
                    self._add_alert("warn", f"Task {task_id}: {reason}")
                    self._archive_task(task)
                    del self.tasks[task_id]
                    cleaned.append(task_id)

                    # Also clean parent workflow if task was orphaned
                    if task.workflow_id and task.workflow_id in self.workflows:
                        wf = self.workflows[task.workflow_id]
                        if wf.status == "running":
                            wf.status = "failed"
                            wf.completed_at = now
                            wf.error = f"Task {task_id} orphaned"
                            self._archive_workflow(wf)
                            del self.workflows[task.workflow_id]
                            cleaned.append(task.workflow_id)

            if cleaned:
                self.save()

        return cleaned

    def start_monitor(self, interval: float = 5.0):
        """Start background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._shutdown = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Started tracker monitor thread")

    def stop_monitor(self):
        """Stop background monitoring thread."""
        self._shutdown = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                self.check_timeouts()

                # Periodic save to update timestamps
                if (datetime.now() - self._last_save).total_seconds() > interval:
                    self.save()

            except Exception as e:
                logger.error("Monitor error: %s", e)

            time.sleep(interval)

    # -------------------------------------------------------------------------
    # Status/Query
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get current tracker status summary."""
        with self._lock:
            return {
                "updated_at": datetime.now().isoformat(),
                "workflows": {
                    "running": sum(1 for w in self.workflows.values() if w.status == "running"),
                    "queued": sum(1 for w in self.workflows.values() if w.status == "queued"),
                    "total": len(self.workflows),
                },
                "tasks": {
                    "running": sum(1 for t in self.tasks.values() if t.status == "running"),
                    "queued": sum(1 for t in self.tasks.values() if t.status == "queued"),
                    "total": len(self.tasks),
                },
                "limits": self.limits.to_dict(),
                "stats": self.stats,
                "alerts": self.alerts[-5:],
            }

    def get_workflow(self, workflow_id: str) -> WorkflowState | None:
        """Get a specific workflow's state."""
        return self.workflows.get(workflow_id)

    def get_task(self, task_id: str) -> TaskState | None:
        """Get a specific task's state."""
        return self.tasks.get(task_id)

    def list_workflows(self, customer: str | None = None) -> list[WorkflowState]:
        """List all active workflows, optionally filtered by customer."""
        with self._lock:
            workflows = list(self.workflows.values())
            if customer:
                workflows = [w for w in workflows if w.customer == customer]
            return workflows

    def list_tasks(self, workflow_id: str | None = None) -> list[TaskState]:
        """List all active tasks, optionally filtered by workflow."""
        with self._lock:
            tasks = list(self.tasks.values())
            if workflow_id:
                tasks = [t for t in tasks if t.workflow_id == workflow_id]
            return tasks


# Global tracker instance
_tracker: Tracker | None = None


def get_tracker() -> Tracker:
    """Get the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = Tracker()
    return _tracker


def reset_tracker():
    """Reset the global tracker (for testing)."""
    global _tracker
    if _tracker:
        _tracker.stop_monitor()
    _tracker = None

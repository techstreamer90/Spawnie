"""File-based queue manager for Spawnie tasks."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Task, Result


class QueueManager:
    """
    File-based queue for task management.

    Directory structure:
        .spawnie/
        ├── queue/          # Pending tasks (JSON files)
        ├── in_progress/    # Currently executing
        ├── done/           # Completed successfully
        └── failed/         # Failed tasks
    """

    def __init__(self, base_dir: Path):
        """Initialize queue manager with base directory."""
        self.base_dir = Path(base_dir)
        self.queue_dir = self.base_dir / "queue"
        self.in_progress_dir = self.base_dir / "in_progress"
        self.done_dir = self.base_dir / "done"
        self.failed_dir = self.base_dir / "failed"

        # Ensure all directories exist
        for dir_path in [self.queue_dir, self.in_progress_dir, self.done_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _task_file(self, task_id: str, directory: Path) -> Path:
        """Get the file path for a task in a given directory."""
        return directory / f"{task_id}.json"

    def _result_file(self, task_id: str, directory: Path) -> Path:
        """Get the file path for a result in a given directory."""
        return directory / f"{task_id}.result.json"

    def submit(self, task: Task) -> str:
        """
        Add a task to the queue.

        Returns:
            The task ID.
        """
        task_file = self._task_file(task.id, self.queue_dir)
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        return task.id

    def claim_next(self) -> Task | None:
        """
        Claim the next available task from the queue.

        Moves the task from queue/ to in_progress/.

        Returns:
            The claimed task, or None if queue is empty.
        """
        # Get all pending tasks, sorted by creation time (oldest first)
        pending_files = sorted(self.queue_dir.glob("*.json"))

        for task_file in pending_files:
            try:
                # Attempt atomic move (claim)
                in_progress_file = self._task_file(task_file.stem, self.in_progress_dir)

                # On Windows, we need to handle potential race conditions
                if in_progress_file.exists():
                    continue

                # Read task data
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)

                # Move to in_progress
                task_file.rename(in_progress_file)

                return Task.from_dict(task_data)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                # Another worker claimed it or file is corrupted
                continue

        return None

    def complete(self, task_id: str, output: str, duration: float = 0.0) -> Result:
        """
        Mark a task as completed and store the result.

        Args:
            task_id: The task ID.
            output: The task output.
            duration: Time taken in seconds.

        Returns:
            The Result object.
        """
        result = Result(
            task_id=task_id,
            status="completed",
            output=output,
            completed_at=datetime.now(),
            duration_seconds=duration,
        )

        # Move task file to done/
        in_progress_file = self._task_file(task_id, self.in_progress_dir)
        done_file = self._task_file(task_id, self.done_dir)

        if in_progress_file.exists():
            in_progress_file.rename(done_file)

        # Write result file
        result_file = self._result_file(task_id, self.done_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def fail(self, task_id: str, error: str, duration: float = 0.0) -> Result:
        """
        Mark a task as failed and store the result.

        Args:
            task_id: The task ID.
            error: The error message.
            duration: Time taken in seconds.

        Returns:
            The Result object.
        """
        result = Result(
            task_id=task_id,
            status="failed",
            error=error,
            completed_at=datetime.now(),
            duration_seconds=duration,
        )

        # Move task file to failed/
        in_progress_file = self._task_file(task_id, self.in_progress_dir)
        failed_file = self._task_file(task_id, self.failed_dir)

        if in_progress_file.exists():
            in_progress_file.rename(failed_file)

        # Write result file
        result_file = self._result_file(task_id, self.failed_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def timeout(self, task_id: str, duration: float = 0.0) -> Result:
        """
        Mark a task as timed out.

        Args:
            task_id: The task ID.
            duration: Time taken in seconds.

        Returns:
            The Result object.
        """
        result = Result(
            task_id=task_id,
            status="timeout",
            error="Task execution timed out",
            completed_at=datetime.now(),
            duration_seconds=duration,
        )

        # Move task file to failed/
        in_progress_file = self._task_file(task_id, self.in_progress_dir)
        failed_file = self._task_file(task_id, self.failed_dir)

        if in_progress_file.exists():
            in_progress_file.rename(failed_file)

        # Write result file
        result_file = self._result_file(task_id, self.failed_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def get_result(self, task_id: str) -> Result | None:
        """
        Get the result for a task if available.

        Args:
            task_id: The task ID.

        Returns:
            The Result object, or None if not yet complete.
        """
        # Check done directory
        result_file = self._result_file(task_id, self.done_dir)
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return Result.from_dict(json.load(f))

        # Check failed directory
        result_file = self._result_file(task_id, self.failed_dir)
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return Result.from_dict(json.load(f))

        return None

    def get_task(self, task_id: str) -> Task | None:
        """
        Get a task by ID from any directory.

        Args:
            task_id: The task ID.

        Returns:
            The Task object, or None if not found.
        """
        for directory in [self.queue_dir, self.in_progress_dir, self.done_dir, self.failed_dir]:
            task_file = self._task_file(task_id, directory)
            if task_file.exists():
                with open(task_file, "r", encoding="utf-8") as f:
                    return Task.from_dict(json.load(f))
        return None

    def get_status(self, task_id: str) -> str | None:
        """
        Get the status of a task.

        Returns:
            "pending", "in_progress", "completed", "failed", or None if not found.
        """
        if self._task_file(task_id, self.queue_dir).exists():
            return "pending"
        if self._task_file(task_id, self.in_progress_dir).exists():
            return "in_progress"
        if self._task_file(task_id, self.done_dir).exists():
            return "completed"
        if self._task_file(task_id, self.failed_dir).exists():
            return "failed"
        return None

    def list_pending(self) -> list[Task]:
        """List all pending tasks."""
        tasks = []
        for task_file in sorted(self.queue_dir.glob("*.json")):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    tasks.append(Task.from_dict(json.load(f)))
            except (json.JSONDecodeError, KeyError):
                continue
        return tasks

    def list_in_progress(self) -> list[Task]:
        """List all in-progress tasks."""
        tasks = []
        for task_file in sorted(self.in_progress_dir.glob("*.json")):
            if ".result." not in task_file.name:
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        tasks.append(Task.from_dict(json.load(f)))
                except (json.JSONDecodeError, KeyError):
                    continue
        return tasks

    def pending_count(self) -> int:
        """Get the number of pending tasks."""
        return len(list(self.queue_dir.glob("*.json")))

    def in_progress_count(self) -> int:
        """Get the number of in-progress tasks."""
        return len([f for f in self.in_progress_dir.glob("*.json") if ".result." not in f.name])

    def clear_all(self) -> None:
        """Clear all tasks from all directories. Use with caution."""
        for directory in [self.queue_dir, self.in_progress_dir, self.done_dir, self.failed_dir]:
            for f in directory.glob("*.json"):
                f.unlink()

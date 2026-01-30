"""File-based queue manager for Spawnie tasks."""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

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

    def _validate_task_id(self, task_id: str) -> None:
        """Validate task_id to prevent path traversal attacks."""
        if not task_id or not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            raise ValueError(f"Invalid task_id: {task_id}")

    def _task_file(self, task_id: str, directory: Path) -> Path:
        """Get the file path for a task in a given directory."""
        self._validate_task_id(task_id)
        return directory / f"{task_id}.json"

    def _result_file(self, task_id: str, directory: Path) -> Path:
        """Get the file path for a result in a given directory."""
        self._validate_task_id(task_id)
        return directory / f"{task_id}.result.json"

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        """Write JSON data atomically using temp file + rename."""
        # Write to temp file in same directory (ensures same filesystem for rename)
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # Atomic rename (overwrites on Unix, fails on Windows if exists)
            temp_file = Path(temp_path)
            try:
                temp_file.replace(path)  # Atomic on POSIX, works on Windows too
            except OSError:
                # On Windows, try removing target first if it exists
                if path.exists():
                    path.unlink()
                temp_file.rename(path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def submit(self, task: Task) -> str:
        """
        Add a task to the queue.

        Returns:
            The task ID.
        """
        task_file = self._task_file(task.id, self.queue_dir)
        self._atomic_write_json(task_file, task.to_dict())
        return task.id

    def claim_next(self) -> Task | None:
        """
        Claim the next available task from the queue.

        Uses atomic rename to prevent race conditions between workers.
        Moves the task from queue/ to in_progress/.

        Returns:
            The claimed task, or None if queue is empty.
        """
        # Get all pending tasks, sorted by creation time (oldest first)
        pending_files = sorted(self.queue_dir.glob("*.json"))

        for task_file in pending_files:
            try:
                in_progress_file = self._task_file(task_file.stem, self.in_progress_dir)

                # Try atomic rename first - this is the claim operation
                # If another worker already claimed it, this will fail
                try:
                    task_file.rename(in_progress_file)
                except (FileNotFoundError, FileExistsError, PermissionError):
                    # Another worker claimed it first, or file was already moved
                    continue

                # Successfully claimed - now read the task data
                try:
                    with open(in_progress_file, "r", encoding="utf-8") as f:
                        task_data = json.load(f)
                    return Task.from_dict(task_data)
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    # File is corrupted - move to failed
                    failed_file = self._task_file(task_file.stem, self.failed_dir)
                    try:
                        in_progress_file.rename(failed_file)
                    except OSError:
                        pass
                    continue

            except OSError:
                # Unexpected error, try next file
                continue

        return None

    def _write_result(self, task_id: str, result: Result, dest_dir: Path) -> None:
        """Helper to move task file and write result."""
        in_progress_file = self._task_file(task_id, self.in_progress_dir)
        dest_file = self._task_file(task_id, dest_dir)

        if in_progress_file.exists():
            try:
                in_progress_file.rename(dest_file)
            except (FileExistsError, OSError):
                # Target exists or other error - try to overwrite
                if dest_file.exists():
                    dest_file.unlink()
                try:
                    in_progress_file.rename(dest_file)
                except OSError:
                    pass

        result_file = self._result_file(task_id, dest_dir)
        self._atomic_write_json(result_file, result.to_dict())

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
        self._write_result(task_id, result, self.done_dir)
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
        self._write_result(task_id, result, self.failed_dir)
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
        self._write_result(task_id, result, self.failed_dir)
        return result

    def get_result(self, task_id: str) -> Result | None:
        """
        Get the result for a task if available.

        Args:
            task_id: The task ID.

        Returns:
            The Result object, or None if not yet complete.
        """
        for directory in [self.done_dir, self.failed_dir]:
            result_file = self._result_file(task_id, directory)
            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        return Result.from_dict(json.load(f))
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
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
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        return Task.from_dict(json.load(f))
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
        return None

    def get_status(self, task_id: str) -> str | None:
        """
        Get the status of a task.

        Returns:
            "pending", "in_progress", "completed", "failed", or None if not found.
        """
        self._validate_task_id(task_id)
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
            except (json.JSONDecodeError, KeyError, OSError):
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
                except (json.JSONDecodeError, KeyError, OSError):
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
                try:
                    f.unlink()
                except OSError:
                    pass

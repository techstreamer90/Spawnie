"""Data models for Spawnie tasks and results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import uuid


# Quality levels control how much review a task receives
QualityLevel = Literal["normal", "extra-clean", "hypertask"]

QUALITY_DESCRIPTIONS = {
    "normal": "No review - execute task and return result directly",
    "extra-clean": "Self-review - agent reviews its own output before returning",
    "hypertask": "Dual review - self-review + external reviewer for highest quality",
}


@dataclass
class Task:
    """A task to be executed by a CLI agent."""

    prompt: str
    provider: str
    model: str | None = None
    quality_level: QualityLevel = "normal"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "provider": self.provider,
            "model": self.model,
            "quality_level": self.quality_level,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create Task from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            provider=data["provider"],
            model=data.get("model"),
            quality_level=data.get("quality_level", "normal"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Result:
    """Result of a completed task."""

    task_id: str
    status: str  # "completed" | "failed" | "timeout"
    output: str | None = None
    error: str | None = None
    completed_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Result":
        """Create Result from dictionary."""
        return cls(
            task_id=data["task_id"],
            status=data["status"],
            output=data.get("output"),
            error=data.get("error"),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    @property
    def succeeded(self) -> bool:
        """Check if the task completed successfully."""
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        """Check if the task failed."""
        return self.status in ("failed", "timeout")

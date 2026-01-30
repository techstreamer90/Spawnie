"""Configuration for Spawnie."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ProviderType = Literal["claude", "copilot", "mock"]


@dataclass
class SpawnieConfig:
    """Configuration for Spawnie task spawning."""

    provider: ProviderType = "claude"
    model: str | None = None  # Sub-model: "sonnet", "opus", "haiku", etc.
    response_dir: Path = field(default_factory=lambda: Path("spawnie-responses"))
    queue_dir: Path = field(default_factory=lambda: Path(".spawnie/queue"))
    timeout: int = 300  # 5 minutes default
    poll_interval: float = 0.5  # Daemon poll interval in seconds

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.response_dir, str):
            self.response_dir = Path(self.response_dir)
        if isinstance(self.queue_dir, str):
            self.queue_dir = Path(self.queue_dir)

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        (self.queue_dir.parent / "in_progress").mkdir(parents=True, exist_ok=True)
        (self.queue_dir.parent / "done").mkdir(parents=True, exist_ok=True)
        (self.queue_dir.parent / "failed").mkdir(parents=True, exist_ok=True)

"""Claude CLI provider."""

import shutil

from ..config import CLAUDE_TIMEOUT
from .base import CLIProvider


class ClaudeCLIProvider(CLIProvider):
    """Provider for Claude CLI (claude-code)."""

    timeout = CLAUDE_TIMEOUT

    @property
    def name(self) -> str:
        return "claude"

    def detect(self) -> bool:
        """Check if Claude CLI is available."""
        return shutil.which("claude") is not None

    def get_available_models(self) -> list[str]:
        """Return available Claude models."""
        return ["sonnet", "opus", "haiku"]

    def execute(self, prompt: str, model: str | None = None) -> tuple[str, int]:
        """
        Execute a prompt using Claude CLI.

        Args:
            prompt: The prompt to execute.
            model: Optional model (sonnet, opus, haiku).

        Returns:
            A tuple of (output, exit_code).
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            return ("Claude CLI not found", 1)

        cmd = [claude_path, "--print", "--output-format", "text"]
        if model:
            cmd.extend(["--model", model])

        return self._execute_with_length_check(cmd, prompt)

"""GitHub Copilot CLI provider."""

import shutil
import subprocess

from ..config import COPILOT_TIMEOUT
from .base import CLIProvider


class CopilotCLIProvider(CLIProvider):
    """Provider for GitHub Copilot CLI."""

    timeout = COPILOT_TIMEOUT

    @property
    def name(self) -> str:
        return "copilot"

    def detect(self) -> bool:
        """Check if Copilot CLI is available."""
        # Check for standalone copilot
        if shutil.which("copilot"):
            return True

        # Check for gh copilot extension
        gh_path = shutil.which("gh")
        if gh_path:
            try:
                result = subprocess.run(
                    [gh_path, "extension", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "copilot" in result.stdout.lower():
                    return True
            except (subprocess.TimeoutExpired, OSError):
                pass

        return False

    def get_available_models(self) -> list[str]:
        """Return available Copilot models (Copilot doesn't expose model selection)."""
        return []

    def _get_copilot_command(self) -> list[str] | None:
        """Get the command to invoke Copilot CLI."""
        # Try standalone copilot first
        copilot_path = shutil.which("copilot")
        if copilot_path:
            return [copilot_path]

        # Try gh copilot
        gh_path = shutil.which("gh")
        if gh_path:
            return [gh_path, "copilot"]

        return None

    def execute(self, prompt: str, model: str | None = None) -> tuple[str, int]:
        """
        Execute a prompt using Copilot CLI.

        Note: Copilot CLI doesn't support model selection.

        Args:
            prompt: The prompt to execute.
            model: Ignored for Copilot.

        Returns:
            A tuple of (output, exit_code).
        """
        cmd_base = self._get_copilot_command()
        if not cmd_base:
            return ("Copilot CLI not found", 1)

        cmd = cmd_base + ["suggest", "-t", "shell"]
        return self._execute_with_length_check(cmd, prompt)

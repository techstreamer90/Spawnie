"""GitHub Copilot CLI provider."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import CLIProvider


class CopilotCLIProvider(CLIProvider):
    """Provider for GitHub Copilot CLI."""

    MAX_INLINE_PROMPT_LENGTH = 7000

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

        # Build command - gh copilot suggest or explain
        cmd = cmd_base + ["suggest", "-t", "shell"]

        # Handle long prompts
        if len(prompt) > self.MAX_INLINE_PROMPT_LENGTH:
            return self._execute_with_stdin(cmd, prompt)
        else:
            return self._execute_inline(cmd, prompt)

    def _execute_inline(self, cmd: list[str], prompt: str) -> tuple[str, int]:
        """Execute with prompt as command argument."""
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return (output, result.returncode)
        except subprocess.TimeoutExpired:
            return ("Command timed out", 124)
        except OSError as e:
            return (f"Failed to execute: {e}", 1)

    def _execute_with_stdin(self, cmd: list[str], prompt: str) -> tuple[str, int]:
        """Execute with prompt via stdin."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(prompt)
            temp_path = Path(f.name)

        try:
            with open(temp_path, "r", encoding="utf-8") as stdin_file:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_file,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return (output, result.returncode)
        except subprocess.TimeoutExpired:
            return ("Command timed out", 124)
        except OSError as e:
            return (f"Failed to execute: {e}", 1)
        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

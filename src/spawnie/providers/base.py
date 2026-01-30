"""Base class for CLI providers."""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import MAX_INLINE_PROMPT_LENGTH, DEFAULT_TIMEOUT


class CLIProvider(ABC):
    """Abstract base class for CLI providers."""

    # Subclasses can override these
    timeout: int = DEFAULT_TIMEOUT
    max_inline_prompt_length: int = MAX_INLINE_PROMPT_LENGTH

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def execute(self, prompt: str, model: str | None = None) -> tuple[str, int]:
        """
        Execute a prompt using the CLI.

        Args:
            prompt: The prompt to execute.
            model: Optional sub-model to use.

        Returns:
            A tuple of (output, exit_code).
        """
        pass

    @abstractmethod
    def detect(self) -> bool:
        """
        Check if the CLI is available.

        Returns:
            True if the CLI is installed and accessible.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """
        Return available sub-models for this provider.

        Returns:
            List of model names.
        """
        pass

    def _execute_inline(self, cmd: list[str], prompt: str) -> tuple[str, int]:
        """Execute with prompt as command argument."""
        cmd = cmd + [prompt]  # Don't mutate the original list

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
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
        """Execute with prompt via stdin for long prompts."""
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
                    timeout=self.timeout,
                    encoding="utf-8",
                    errors="replace",
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

    def _execute_with_length_check(self, cmd: list[str], prompt: str) -> tuple[str, int]:
        """Execute command, using stdin for long prompts."""
        if len(prompt) > self.max_inline_prompt_length:
            return self._execute_with_stdin(cmd, prompt)
        else:
            return self._execute_inline(cmd, prompt)

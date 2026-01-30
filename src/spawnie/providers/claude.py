"""Claude CLI provider."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import CLIProvider


class ClaudeCLIProvider(CLIProvider):
    """Provider for Claude CLI (claude-code)."""

    # Windows CMD has 8191 character limit, leave some room for command overhead
    MAX_INLINE_PROMPT_LENGTH = 7000

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

        Uses `claude -p <prompt>` for short prompts, or writes to a temp file
        and uses stdin for long prompts to avoid command line length limits.

        Args:
            prompt: The prompt to execute.
            model: Optional model (sonnet, opus, haiku).

        Returns:
            A tuple of (output, exit_code).
        """
        claude_path = shutil.which("claude")
        if not claude_path:
            return ("Claude CLI not found", 1)

        # Build command arguments
        cmd = [claude_path, "--print", "--output-format", "text"]

        if model:
            cmd.extend(["--model", model])

        # Handle long prompts by using stdin
        if len(prompt) > self.MAX_INLINE_PROMPT_LENGTH:
            return self._execute_with_stdin(cmd, prompt)
        else:
            return self._execute_inline(cmd, prompt)

    def _execute_inline(self, cmd: list[str], prompt: str) -> tuple[str, int]:
        """Execute with prompt as positional argument."""
        cmd.append(prompt)  # Prompt is positional, not --prompt flag

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
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
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(prompt)
            temp_path = Path(f.name)

        try:
            # Use stdin redirection
            with open(temp_path, "r", encoding="utf-8") as stdin_file:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_file,
                    capture_output=True,
                    text=True,
                    timeout=600,
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
            # Clean up temp file
            try:
                temp_path.unlink()
            except OSError:
                pass

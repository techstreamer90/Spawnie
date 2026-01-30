"""CLI detection and installation helpers."""

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CLIStatus:
    """Status of a CLI tool installation."""

    provider: str
    installed: bool
    path: str | None = None
    version: str | None = None
    models: list[str] | None = None
    error: str | None = None


class CLINotFoundError(Exception):
    """Raised when a required CLI is not found."""

    def __init__(self, provider: str, instructions: str):
        self.provider = provider
        self.instructions = instructions
        super().__init__(f"{provider} CLI not found. {instructions}")


def detect_cli(provider: str) -> CLIStatus:
    """
    Detect if a CLI tool is installed and get its details.

    Args:
        provider: The CLI provider name ("claude", "copilot", "mock").

    Returns:
        CLIStatus with installation details.
    """
    if provider == "mock":
        return CLIStatus(
            provider="mock",
            installed=True,
            path="<mock>",
            version="1.0.0",
            models=["default", "fast", "slow"],
        )

    if provider == "claude":
        return _detect_claude()

    if provider == "copilot":
        return _detect_copilot()

    return CLIStatus(
        provider=provider,
        installed=False,
        error=f"Unknown provider: {provider}",
    )


def _detect_claude() -> CLIStatus:
    """Detect Claude CLI installation."""
    # Try to find claude executable
    claude_path = shutil.which("claude")

    if not claude_path:
        return CLIStatus(
            provider="claude",
            installed=False,
            error="Claude CLI not found in PATH",
        )

    # Get version
    try:
        result = subprocess.run(
            [claude_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        version = None

    # Known Claude models
    models = ["sonnet", "opus", "haiku"]

    return CLIStatus(
        provider="claude",
        installed=True,
        path=claude_path,
        version=version,
        models=models,
    )


def _detect_copilot() -> CLIStatus:
    """Detect GitHub Copilot CLI installation."""
    # Try common copilot CLI names
    copilot_path = None
    for name in ["copilot", "github-copilot-cli", "gh copilot"]:
        copilot_path = shutil.which(name)
        if copilot_path:
            break

    # Also check for gh extension
    if not copilot_path:
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
                    copilot_path = f"{gh_path} copilot"
            except (subprocess.TimeoutExpired, OSError):
                pass

    if not copilot_path:
        return CLIStatus(
            provider="copilot",
            installed=False,
            error="GitHub Copilot CLI not found",
        )

    # Get version if possible
    version = None
    try:
        if " " in copilot_path:
            # gh copilot case
            parts = copilot_path.split()
            result = subprocess.run(
                parts + ["--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
            result = subprocess.run(
                [copilot_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        version = result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        pass

    return CLIStatus(
        provider="copilot",
        installed=True,
        path=copilot_path,
        version=version,
        models=None,  # Copilot doesn't expose model selection
    )


def ensure_cli(provider: str) -> CLIStatus:
    """
    Ensure a CLI is available, raising an error with installation instructions if not.

    Args:
        provider: The CLI provider name.

    Returns:
        CLIStatus if the CLI is installed.

    Raises:
        CLINotFoundError: If the CLI is not installed.
    """
    status = detect_cli(provider)

    if not status.installed:
        instructions = get_installation_instructions(provider)
        raise CLINotFoundError(provider, instructions)

    return status


def get_installation_instructions(provider: str) -> str:
    """
    Get platform-specific installation instructions for a CLI.

    Args:
        provider: The CLI provider name.

    Returns:
        Installation instructions as a string.
    """
    platform = sys.platform

    if provider == "claude":
        if platform == "win32":
            return """
To install Claude CLI on Windows:
1. Install Node.js (https://nodejs.org/)
2. Run: npm install -g @anthropic-ai/claude-code
3. Run: claude login

Alternative: Use winget
  winget install Anthropic.ClaudeCode
"""
        elif platform == "darwin":
            return """
To install Claude CLI on macOS:
1. Using Homebrew:
   brew install claude-code
2. Run: claude login

Alternative: Using npm
  npm install -g @anthropic-ai/claude-code
"""
        else:
            return """
To install Claude CLI on Linux:
1. Install Node.js
2. Run: npm install -g @anthropic-ai/claude-code
3. Run: claude login
"""

    if provider == "copilot":
        if platform == "win32":
            return """
To install GitHub Copilot CLI on Windows:
1. Install GitHub CLI: winget install GitHub.cli
2. Run: gh auth login
3. Run: gh extension install github/gh-copilot
"""
        elif platform == "darwin":
            return """
To install GitHub Copilot CLI on macOS:
1. Install GitHub CLI: brew install gh
2. Run: gh auth login
3. Run: gh extension install github/gh-copilot
"""
        else:
            return """
To install GitHub Copilot CLI on Linux:
1. Install GitHub CLI (https://github.com/cli/cli#installation)
2. Run: gh auth login
3. Run: gh extension install github/gh-copilot
"""

    return f"No installation instructions available for provider: {provider}"

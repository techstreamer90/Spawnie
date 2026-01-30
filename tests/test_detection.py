"""Tests for CLI detection."""

import pytest
from spawnie import detect_cli, ensure_cli, get_installation_instructions, CLINotFoundError


class TestDetectCLI:
    """Tests for detect_cli function."""

    def test_detect_mock_always_available(self):
        """Mock provider is always detected as available."""
        status = detect_cli("mock")
        assert status.installed is True
        assert status.provider == "mock"
        assert status.path == "<mock>"
        assert "default" in status.models

    def test_detect_unknown_provider(self):
        """Unknown provider returns not installed."""
        status = detect_cli("unknown_provider")
        assert status.installed is False
        assert "Unknown provider" in status.error

    def test_detect_claude(self):
        """Claude detection returns valid status structure."""
        status = detect_cli("claude")
        assert status.provider == "claude"
        # May or may not be installed depending on environment
        if status.installed:
            assert status.path is not None
            assert "sonnet" in status.models

    def test_detect_copilot(self):
        """Copilot detection returns valid status structure."""
        status = detect_cli("copilot")
        assert status.provider == "copilot"
        # May or may not be installed


class TestEnsureCLI:
    """Tests for ensure_cli function."""

    def test_ensure_mock_succeeds(self):
        """ensure_cli succeeds for mock provider."""
        status = ensure_cli("mock")
        assert status.installed is True

    def test_ensure_unknown_raises(self):
        """ensure_cli raises for unknown provider."""
        with pytest.raises(CLINotFoundError) as exc_info:
            ensure_cli("unknown_provider")
        assert "unknown_provider" in str(exc_info.value)


class TestInstallationInstructions:
    """Tests for installation instructions."""

    def test_claude_instructions_exist(self):
        """Claude has installation instructions."""
        instructions = get_installation_instructions("claude")
        assert "claude" in instructions.lower()
        assert len(instructions) > 50

    def test_copilot_instructions_exist(self):
        """Copilot has installation instructions."""
        instructions = get_installation_instructions("copilot")
        assert "copilot" in instructions.lower() or "gh" in instructions.lower()
        assert len(instructions) > 50

    def test_unknown_provider_instructions(self):
        """Unknown provider returns fallback message."""
        instructions = get_installation_instructions("unknown")
        assert "No installation instructions" in instructions

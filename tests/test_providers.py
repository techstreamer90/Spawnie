"""Tests for CLI providers."""

import pytest
from spawnie.providers import (
    get_provider,
    CLIProvider,
    ClaudeCLIProvider,
    CopilotCLIProvider,
    MockProvider,
)


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_mock_provider(self):
        """get_provider returns MockProvider for 'mock'."""
        provider = get_provider("mock")
        assert isinstance(provider, MockProvider)

    def test_get_claude_provider(self):
        """get_provider returns ClaudeCLIProvider for 'claude'."""
        provider = get_provider("claude")
        assert isinstance(provider, ClaudeCLIProvider)

    def test_get_copilot_provider(self):
        """get_provider returns CopilotCLIProvider for 'copilot'."""
        provider = get_provider("copilot")
        assert isinstance(provider, CopilotCLIProvider)

    def test_get_unknown_provider_raises(self):
        """get_provider raises ValueError for unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            get_provider("unknown")
        assert "Unknown provider" in str(exc_info.value)


class TestMockProvider:
    """Tests for MockProvider."""

    def test_name(self):
        """MockProvider has correct name."""
        provider = MockProvider()
        assert provider.name == "mock"

    def test_detect_always_true(self):
        """MockProvider is always detected."""
        provider = MockProvider()
        assert provider.detect() is True

    def test_get_available_models(self):
        """MockProvider returns mock models."""
        provider = MockProvider()
        models = provider.get_available_models()
        assert "default" in models
        assert "fast" in models
        assert "slow" in models

    def test_execute_returns_success(self):
        """MockProvider execute returns success by default."""
        provider = MockProvider()
        output, exit_code = provider.execute("Test prompt")

        assert exit_code == 0
        assert "[Mock Response]" in output
        assert "Test prompt" in output

    def test_execute_tracks_calls(self):
        """MockProvider tracks call count and last prompt."""
        provider = MockProvider()
        assert provider.call_count == 0

        provider.execute("First prompt")
        assert provider.call_count == 1
        assert provider.last_prompt == "First prompt"

        provider.execute("Second prompt", model="fast")
        assert provider.call_count == 2
        assert provider.last_prompt == "Second prompt"
        assert provider.last_model == "fast"

    def test_execute_with_model(self):
        """MockProvider includes model in output."""
        provider = MockProvider()
        output, _ = provider.execute("Test", model="fast")
        assert "fast" in output

    def test_execute_with_fail_rate(self):
        """MockProvider can simulate failures."""
        provider = MockProvider(fail_rate=1.0)  # Always fail
        output, exit_code = provider.execute("Test")

        assert exit_code == 1
        assert "Simulated failure" in output

    def test_reset(self):
        """MockProvider reset clears tracking."""
        provider = MockProvider()
        provider.execute("Test")
        assert provider.call_count == 1

        provider.reset()
        assert provider.call_count == 0
        assert provider.last_prompt is None


class TestClaudeCLIProvider:
    """Tests for ClaudeCLIProvider."""

    def test_name(self):
        """ClaudeCLIProvider has correct name."""
        provider = ClaudeCLIProvider()
        assert provider.name == "claude"

    def test_get_available_models(self):
        """ClaudeCLIProvider returns Claude models."""
        provider = ClaudeCLIProvider()
        models = provider.get_available_models()
        assert "sonnet" in models
        assert "opus" in models
        assert "haiku" in models

    def test_max_inline_prompt_length(self):
        """ClaudeCLIProvider has reasonable max prompt length."""
        assert ClaudeCLIProvider.MAX_INLINE_PROMPT_LENGTH > 1000
        assert ClaudeCLIProvider.MAX_INLINE_PROMPT_LENGTH < 8192


class TestCopilotCLIProvider:
    """Tests for CopilotCLIProvider."""

    def test_name(self):
        """CopilotCLIProvider has correct name."""
        provider = CopilotCLIProvider()
        assert provider.name == "copilot"

    def test_get_available_models_empty(self):
        """CopilotCLIProvider returns empty models (no selection supported)."""
        provider = CopilotCLIProvider()
        models = provider.get_available_models()
        assert models == []


class TestCLIProviderInterface:
    """Tests for CLIProvider interface compliance."""

    @pytest.mark.parametrize("provider_name", ["mock", "claude", "copilot"])
    def test_provider_has_required_methods(self, provider_name):
        """All providers implement required methods."""
        provider = get_provider(provider_name)

        assert hasattr(provider, "name")
        assert hasattr(provider, "execute")
        assert hasattr(provider, "detect")
        assert hasattr(provider, "get_available_models")

    @pytest.mark.parametrize("provider_name", ["mock", "claude", "copilot"])
    def test_provider_is_cli_provider(self, provider_name):
        """All providers are CLIProvider instances."""
        provider = get_provider(provider_name)
        assert isinstance(provider, CLIProvider)

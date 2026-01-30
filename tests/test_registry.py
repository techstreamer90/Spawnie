"""Tests for model registry."""

import json
import pytest
import tempfile
from pathlib import Path

from spawnie.registry import (
    ModelRegistry,
    ModelConfig,
    ProviderConfig,
    Route,
    get_registry,
    reset_registry,
)


@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        yield config_path


@pytest.fixture
def registry(temp_config):
    """Create a registry with temp config."""
    reset_registry()
    return ModelRegistry(config_path=temp_config)


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_init_creates_defaults(self, registry):
        """Registry initializes with default models and providers."""
        assert "claude-sonnet" in registry.models
        assert "claude-opus" in registry.models
        assert "claude-haiku" in registry.models
        assert "claude-cli" in registry.providers
        assert "anthropic-api" in registry.providers
        assert "mock" in registry.providers

    def test_save_creates_config_file(self, registry, temp_config):
        """save() creates config file."""
        assert not temp_config.exists()
        registry.save()
        assert temp_config.exists()

    def test_save_load_roundtrip(self, temp_config):
        """Config survives save/load cycle."""
        registry1 = ModelRegistry(config_path=temp_config)
        registry1.add_model("test-model", [{"provider": "mock", "priority": 1}], "Test")
        registry1.save()

        registry2 = ModelRegistry(config_path=temp_config)
        assert "test-model" in registry2.models
        assert registry2.models["test-model"].description == "Test"

    def test_load_preserves_custom_config(self, temp_config):
        """Loading preserves custom configuration."""
        config = {
            "providers": {
                "custom-provider": {"type": "cli", "command": "custom"}
            },
            "models": {
                "custom-model": {
                    "routes": [{"provider": "custom-provider", "priority": 1}],
                    "description": "Custom model",
                }
            },
            "preferences": {"prefer_cli": False},
        }
        with open(temp_config, "w") as f:
            json.dump(config, f)

        registry = ModelRegistry(config_path=temp_config)
        assert "custom-provider" in registry.providers
        assert "custom-model" in registry.models
        assert registry.preferences.get("prefer_cli") is False


class TestProviderAvailability:
    """Tests for provider availability checking."""

    def test_mock_always_available(self, registry):
        """Mock provider is always available."""
        assert registry.check_provider_available("mock") is True

    def test_unknown_provider_not_available(self, registry):
        """Unknown provider is not available."""
        assert registry.check_provider_available("nonexistent") is False

    def test_availability_is_cached(self, registry):
        """Availability check results are cached."""
        registry.check_provider_available("mock")
        assert "mock" in registry._availability_cache
        assert registry._availability_cache["mock"] is True


class TestRouteSelection:
    """Tests for route selection logic."""

    def test_get_best_route_returns_available(self, registry):
        """get_best_route returns an available route."""
        # Add a model with mock route
        registry.add_model("test", [{"provider": "mock", "priority": 1}])

        route_info = registry.get_best_route("test")
        assert route_info is not None
        route, provider = route_info
        assert route.provider == "mock"
        assert provider.type == "mock"

    def test_get_best_route_unknown_model(self, registry):
        """get_best_route returns None for unknown model."""
        result = registry.get_best_route("nonexistent-model")
        assert result is None

    def test_get_best_route_respects_priority(self, registry):
        """get_best_route respects route priority."""
        registry.add_model("prioritized", [
            {"provider": "mock", "priority": 2},
            {"provider": "mock", "priority": 1, "model_id": "first"},
        ])

        route_info = registry.get_best_route("prioritized")
        assert route_info is not None
        route, _ = route_info
        assert route.model_id == "first"

    def test_get_best_route_skips_unavailable(self, registry):
        """get_best_route skips unavailable providers."""
        registry.add_model("fallback-test", [
            {"provider": "nonexistent", "priority": 1},
            {"provider": "mock", "priority": 2},
        ])

        route_info = registry.get_best_route("fallback-test")
        assert route_info is not None
        route, _ = route_info
        assert route.provider == "mock"

    def test_prefer_cli_preference(self, registry):
        """CLI routes preferred when prefer_cli is True."""
        registry.preferences["prefer_cli"] = True
        # Mock is type "mock", not "cli", so this tests the preference logic
        # doesn't break when there are no CLI routes
        registry.add_model("pref-test", [{"provider": "mock", "priority": 1}])

        route_info = registry.get_best_route("pref-test")
        assert route_info is not None


class TestModelManagement:
    """Tests for model add/update."""

    def test_add_model(self, registry):
        """add_model adds a new model."""
        registry.add_model("new-model", [
            {"provider": "mock", "priority": 1, "model_id": "test"}
        ], description="A new model")

        assert "new-model" in registry.models
        model = registry.models["new-model"]
        assert model.description == "A new model"
        assert len(model.routes) == 1
        assert model.routes[0].model_id == "test"

    def test_add_model_clears_cache(self, registry):
        """add_model clears availability cache for affected routes."""
        registry._availability_cache["mock"] = False  # Manually set wrong value
        registry.add_model("cache-test", [{"provider": "mock", "priority": 1}])
        assert "mock" not in registry._availability_cache

    def test_add_provider(self, registry):
        """add_provider adds a new provider."""
        registry.add_provider("new-cli", "cli", command="new-command")

        assert "new-cli" in registry.providers
        provider = registry.providers["new-cli"]
        assert provider.type == "cli"
        assert provider.command == "new-command"


class TestListModels:
    """Tests for list_models."""

    def test_list_models_returns_all(self, registry):
        """list_models returns all models."""
        models = registry.list_models()
        names = [m["name"] for m in models]
        assert "claude-sonnet" in names
        assert "claude-opus" in names

    def test_list_models_includes_availability(self, registry):
        """list_models includes availability status."""
        registry.add_model("available-test", [{"provider": "mock", "priority": 1}])
        models = registry.list_models()

        test_model = next(m for m in models if m["name"] == "available-test")
        assert test_model["available"] is True
        assert test_model["route"] == "mock"


class TestRoute:
    """Tests for Route dataclass."""

    def test_route_to_dict(self):
        """Route serializes to dict."""
        route = Route(provider="test", priority=2, model_id="v1")
        d = route.to_dict()
        assert d["provider"] == "test"
        assert d["priority"] == 2
        assert d["model_id"] == "v1"

    def test_route_from_dict(self):
        """Route deserializes from dict."""
        route = Route.from_dict({"provider": "test", "priority": 3})
        assert route.provider == "test"
        assert route.priority == 3
        assert route.model_id is None


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_cli_provider_to_dict(self):
        """CLI provider serializes correctly."""
        provider = ProviderConfig(name="test", type="cli", command="test-cmd")
        d = provider.to_dict()
        assert d["type"] == "cli"
        assert d["command"] == "test-cmd"

    def test_api_provider_to_dict(self):
        """API provider serializes correctly."""
        provider = ProviderConfig(name="test", type="api", api_key_env="TEST_KEY")
        d = provider.to_dict()
        assert d["type"] == "api"
        assert d["api_key_env"] == "TEST_KEY"

    def test_provider_from_dict(self):
        """Provider deserializes from dict."""
        provider = ProviderConfig.from_dict("test", {
            "type": "cli",
            "command": "test-cmd",
        })
        assert provider.name == "test"
        assert provider.type == "cli"
        assert provider.command == "test-cmd"

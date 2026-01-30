"""Model registry and routing for Spawnie.

The registry maps model names to available providers/routes.
When a caller requests a model, Spawnie finds the best available route.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detection import detect_cli

logger = logging.getLogger("spawnie.registry")

DEFAULT_CONFIG_PATH = Path.home() / ".spawnie" / "config.json"


@dataclass
class Route:
    """A route to access a model via a specific provider."""
    provider: str  # e.g., "claude-cli", "anthropic-api"
    priority: int = 1  # lower = preferred
    model_id: str | None = None  # provider-specific model ID if different
    available: bool | None = None  # cached availability check

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "priority": self.priority,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Route":
        return cls(
            provider=data["provider"],
            priority=data.get("priority", 1),
            model_id=data.get("model_id"),
        )


@dataclass
class ModelConfig:
    """Configuration for a model with multiple routes."""
    name: str
    routes: list[Route] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "routes": [r.to_dict() for r in self.routes],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ModelConfig":
        routes = [Route.from_dict(r) for r in data.get("routes", [])]
        return cls(
            name=name,
            routes=routes,
            description=data.get("description", ""),
        )


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    name: str
    type: str  # "cli" or "api"
    command: str | None = None  # for CLI providers
    api_key_env: str | None = None  # for API providers
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.command:
            d["command"] = self.command
        if self.api_key_env:
            d["api_key_env"] = self.api_key_env
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "ProviderConfig":
        return cls(
            name=name,
            type=data.get("type", "cli"),
            command=data.get("command"),
            api_key_env=data.get("api_key_env"),
            extra={k: v for k, v in data.items()
                   if k not in ("type", "command", "api_key_env")},
        )


class ModelRegistry:
    """Registry of models and their available routes.

    The registry loads from ~/.spawnie/config.json and provides:
    - Model lookup by name
    - Route selection based on availability and priority
    - Provider availability checking
    """

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.models: dict[str, ModelConfig] = {}
        self.providers: dict[str, ProviderConfig] = {}
        self.preferences: dict[str, Any] = {}
        self._availability_cache: dict[str, bool] = {}

        if self.config_path.exists():
            self.load()
        else:
            self._init_defaults()

    def _init_defaults(self):
        """Initialize with sensible defaults."""
        # Default providers
        self.providers = {
            "claude-cli": ProviderConfig(
                name="claude-cli",
                type="cli",
                command="claude",
            ),
            "copilot-cli": ProviderConfig(
                name="copilot-cli",
                type="cli",
                command="gh copilot",
            ),
            "anthropic-api": ProviderConfig(
                name="anthropic-api",
                type="api",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            "openai-api": ProviderConfig(
                name="openai-api",
                type="api",
                api_key_env="OPENAI_API_KEY",
            ),
            "mock": ProviderConfig(
                name="mock",
                type="mock",
            ),
        }

        # Default models with routes
        self.models = {
            "claude-sonnet": ModelConfig(
                name="claude-sonnet",
                description="Claude Sonnet - balanced performance",
                routes=[
                    Route(provider="claude-cli", priority=1, model_id="sonnet"),
                    Route(provider="anthropic-api", priority=2, model_id="claude-sonnet-4-20250514"),
                ],
            ),
            "claude-opus": ModelConfig(
                name="claude-opus",
                description="Claude Opus - highest capability",
                routes=[
                    Route(provider="claude-cli", priority=1, model_id="opus"),
                    Route(provider="anthropic-api", priority=2, model_id="claude-opus-4-20250514"),
                ],
            ),
            "claude-haiku": ModelConfig(
                name="claude-haiku",
                description="Claude Haiku - fast and efficient",
                routes=[
                    Route(provider="claude-cli", priority=1, model_id="haiku"),
                    Route(provider="anthropic-api", priority=2, model_id="claude-haiku-3-20250414"),
                ],
            ),
        }

        self.preferences = {
            "prefer_cli": True,  # Prefer CLI (subscription) over API (pay-per-use)
        }

    def load(self):
        """Load registry from config file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load providers
        self.providers = {}
        for name, pdata in data.get("providers", {}).items():
            self.providers[name] = ProviderConfig.from_dict(name, pdata)

        # Load models
        self.models = {}
        for name, mdata in data.get("models", {}).items():
            self.models[name] = ModelConfig.from_dict(name, mdata)

        # Load preferences
        self.preferences = data.get("preferences", {})

        logger.info("Loaded registry from %s: %d models, %d providers",
                    self.config_path, len(self.models), len(self.providers))

    def save(self):
        """Save registry to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "providers": {name: p.to_dict() for name, p in self.providers.items()},
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "preferences": self.preferences,
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved registry to %s", self.config_path)

    def check_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available."""
        if provider_name in self._availability_cache:
            return self._availability_cache[provider_name]

        provider = self.providers.get(provider_name)
        if not provider:
            self._availability_cache[provider_name] = False
            return False

        available = False

        if provider.type == "mock":
            # Mock is always available
            available = True

        elif provider.type == "cli":
            # Map to detection
            if "claude" in provider_name:
                status = detect_cli("claude")
                available = status.installed
            elif "copilot" in provider_name:
                status = detect_cli("copilot")
                available = status.installed
            else:
                available = False

        elif provider.type == "api":
            # Check for API key
            import os
            if provider.api_key_env:
                available = bool(os.environ.get(provider.api_key_env))

        self._availability_cache[provider_name] = available
        return available

    def get_best_route(self, model_name: str) -> tuple[Route, ProviderConfig] | None:
        """Get the best available route for a model.

        Returns:
            Tuple of (Route, ProviderConfig) or None if no route available.
        """
        model = self.models.get(model_name)
        if not model:
            logger.warning("Unknown model: %s", model_name)
            return None

        # Sort routes by priority
        sorted_routes = sorted(model.routes, key=lambda r: r.priority)

        # Apply preferences
        if self.preferences.get("prefer_cli"):
            # Move CLI routes to front (within their priority groups)
            cli_routes = [r for r in sorted_routes
                         if self.providers.get(r.provider, ProviderConfig("", "")).type == "cli"]
            api_routes = [r for r in sorted_routes
                         if self.providers.get(r.provider, ProviderConfig("", "")).type == "api"]
            other_routes = [r for r in sorted_routes
                          if self.providers.get(r.provider, ProviderConfig("", "")).type not in ("cli", "api")]
            sorted_routes = cli_routes + other_routes + api_routes

        # Find first available route
        for route in sorted_routes:
            if self.check_provider_available(route.provider):
                provider = self.providers[route.provider]
                logger.info("Selected route for %s: %s", model_name, route.provider)
                return (route, provider)

        logger.warning("No available route for model: %s", model_name)
        return None

    def list_models(self) -> list[dict]:
        """List all models with their availability status."""
        result = []
        for name, model in self.models.items():
            route_info = self.get_best_route(name)
            result.append({
                "name": name,
                "description": model.description,
                "available": route_info is not None,
                "route": route_info[0].provider if route_info else None,
                "routes": [r.provider for r in model.routes],
            })
        return result

    def add_model(self, name: str, routes: list[dict], description: str = ""):
        """Add or update a model configuration."""
        self.models[name] = ModelConfig(
            name=name,
            description=description,
            routes=[Route.from_dict(r) for r in routes],
        )
        # Clear availability cache for affected routes
        for route in self.models[name].routes:
            self._availability_cache.pop(route.provider, None)

    def add_provider(self, name: str, provider_type: str, **kwargs):
        """Add or update a provider configuration."""
        self.providers[name] = ProviderConfig(
            name=name,
            type=provider_type,
            command=kwargs.get("command"),
            api_key_env=kwargs.get("api_key_env"),
            extra={k: v for k, v in kwargs.items()
                   if k not in ("command", "api_key_env")},
        )
        self._availability_cache.pop(name, None)


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry():
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None

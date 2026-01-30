"""CLI providers for Spawnie."""

from .base import CLIProvider
from .claude import ClaudeCLIProvider
from .copilot import CopilotCLIProvider
from .mock import MockProvider


def get_provider(provider_name: str) -> CLIProvider:
    """
    Get a CLI provider by name.

    Args:
        provider_name: The provider name ("claude", "copilot", "mock").

    Returns:
        A CLIProvider instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    providers = {
        "claude": ClaudeCLIProvider,
        "copilot": CopilotCLIProvider,
        "mock": MockProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")

    return providers[provider_name]()


__all__ = [
    "CLIProvider",
    "ClaudeCLIProvider",
    "CopilotCLIProvider",
    "MockProvider",
    "get_provider",
]

"""Base class for CLI providers."""

from abc import ABC, abstractmethod


class CLIProvider(ABC):
    """Abstract base class for CLI providers."""

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

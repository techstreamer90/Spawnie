"""Mock provider for testing."""

import random
import time

from .base import CLIProvider


class MockProvider(CLIProvider):
    """Mock provider for testing without actual CLI calls."""

    def __init__(self, delay: float = 0.1, fail_rate: float = 0.0):
        """
        Initialize mock provider.

        Args:
            delay: Simulated execution delay in seconds.
            fail_rate: Rate at which tasks should fail (0.0 to 1.0).
        """
        self.delay = delay
        self.fail_rate = fail_rate
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_model: str | None = None

    @property
    def name(self) -> str:
        return "mock"

    def detect(self) -> bool:
        """Mock is always available."""
        return True

    def get_available_models(self) -> list[str]:
        """Return mock model options."""
        return ["default", "fast", "slow"]

    def execute(self, prompt: str, model: str | None = None) -> tuple[str, int]:
        """
        Simulate executing a prompt.

        Args:
            prompt: The prompt to execute.
            model: Optional model (affects delay for "slow" model).

        Returns:
            A tuple of (output, exit_code).
        """
        self.call_count += 1
        self.last_prompt = prompt
        self.last_model = model

        # Simulate processing time
        delay = self.delay
        if model == "slow":
            delay *= 3
        elif model == "fast":
            delay /= 2

        time.sleep(delay)

        # Simulate failures
        if random.random() < self.fail_rate:
            return ("Simulated failure", 1)

        # Generate mock response
        response = f"[Mock Response]\nPrompt received ({len(prompt)} chars)\n"
        if model:
            response += f"Model: {model}\n"
        response += f"Call #{self.call_count}\n"
        response += f"Echo: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"

        return (response, 0)

    def reset(self):
        """Reset call tracking."""
        self.call_count = 0
        self.last_prompt = None
        self.last_model = None

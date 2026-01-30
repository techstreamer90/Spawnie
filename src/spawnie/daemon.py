"""Worker daemon for processing Spawnie tasks."""

import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import SpawnieConfig
from .queue import QueueManager
from .providers import get_provider


class SpawnieDaemon:
    """Daemon that polls the queue and executes tasks via CLI providers."""

    def __init__(self, config: SpawnieConfig):
        """
        Initialize the daemon.

        Args:
            config: Spawnie configuration.
        """
        self.config = config
        self.queue = QueueManager(config.queue_dir.parent)
        self.provider = get_provider(config.provider)
        self.running = False
        self._shutdown_requested = False

    def start(self):
        """Start the daemon loop."""
        self.running = True
        self._shutdown_requested = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print(f"Spawnie daemon started")
        print(f"  Provider: {self.config.provider}")
        print(f"  Model: {self.config.model or 'default'}")
        print(f"  Queue dir: {self.config.queue_dir.parent}")
        print(f"  Poll interval: {self.config.poll_interval}s")
        print()

        while self.running and not self._shutdown_requested:
            try:
                self._process_next_task()
            except Exception as e:
                print(f"Error in daemon loop: {e}", file=sys.stderr)
                time.sleep(1)  # Avoid tight loop on persistent errors

            if not self._shutdown_requested:
                time.sleep(self.config.poll_interval)

        print("Spawnie daemon stopped")

    def stop(self):
        """Stop the daemon."""
        self.running = False
        self._shutdown_requested = True

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown requested...")
        self.stop()

    def _process_next_task(self):
        """Claim and process the next task from the queue."""
        task = self.queue.claim_next()

        if not task:
            return

        print(f"Processing task {task.id[:8]}...")
        start_time = datetime.now()

        try:
            # Execute via CLI provider
            output, exit_code = self.provider.execute(
                task.prompt,
                task.model or self.config.model,
            )

            duration = (datetime.now() - start_time).total_seconds()

            if exit_code == 0:
                result = self.queue.complete(task.id, output, duration)
                print(f"  Completed in {duration:.2f}s")
            else:
                result = self.queue.fail(task.id, output, duration)
                print(f"  Failed (exit code {exit_code}) in {duration:.2f}s")

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.queue.fail(task.id, str(e), duration)
            print(f"  Error: {e}")

    def process_one(self) -> bool:
        """
        Process a single task (non-blocking).

        Returns:
            True if a task was processed, False if queue was empty.
        """
        task = self.queue.claim_next()
        if not task:
            return False

        start_time = datetime.now()

        try:
            output, exit_code = self.provider.execute(
                task.prompt,
                task.model or self.config.model,
            )
            duration = (datetime.now() - start_time).total_seconds()

            if exit_code == 0:
                self.queue.complete(task.id, output, duration)
            else:
                self.queue.fail(task.id, output, duration)

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.queue.fail(task.id, str(e), duration)

        return True


def run_daemon(config: SpawnieConfig):
    """
    Run the Spawnie daemon.

    Args:
        config: Spawnie configuration.
    """
    daemon = SpawnieDaemon(config)
    daemon.start()

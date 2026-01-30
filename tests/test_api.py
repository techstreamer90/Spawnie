"""Tests for high-level API."""

import pytest
import tempfile
from pathlib import Path

from spawnie import (
    spawn,
    spawn_async,
    get_result,
    get_status,
    wait_for_result,
    SpawnieConfig,
    Task,
    Result,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "queue_dir": base / "queue",
            "response_dir": base / "responses",
        }


class TestSpawn:
    """Tests for spawn function."""

    def test_spawn_with_mock_provider(self, temp_dirs):
        """spawn with mock provider returns result."""
        result = spawn(
            "Test prompt",
            provider="mock",
            wait=True,
            queue_dir=temp_dirs["queue_dir"],
        )

        assert isinstance(result, Result)
        assert result.status == "completed"
        assert "[Mock Response]" in result.output

    def test_spawn_returns_task_id_when_not_waiting(self, temp_dirs):
        """spawn returns task ID when wait=False."""
        task_id = spawn(
            "Test prompt",
            provider="mock",
            wait=False,
            queue_dir=temp_dirs["queue_dir"],
        )

        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID format

    def test_spawn_with_model(self, temp_dirs):
        """spawn passes model to provider."""
        result = spawn(
            "Test prompt",
            provider="mock",
            model="fast",
            wait=True,
            queue_dir=temp_dirs["queue_dir"],
        )

        assert result.status == "completed"
        assert "fast" in result.output

    def test_spawn_with_metadata(self, temp_dirs):
        """spawn accepts metadata."""
        task_id = spawn(
            "Test prompt",
            provider="mock",
            wait=False,
            queue_dir=temp_dirs["queue_dir"],
            metadata={"source": "test"},
        )

        assert isinstance(task_id, str)


class TestSpawnAsync:
    """Tests for spawn_async function."""

    def test_spawn_async_returns_task_id(self, temp_dirs):
        """spawn_async returns task ID."""
        task_id = spawn_async(
            "Test prompt",
            provider="mock",
            queue_dir=temp_dirs["queue_dir"],
        )

        assert isinstance(task_id, str)
        assert len(task_id) == 36


class TestGetResult:
    """Tests for get_result function."""

    def test_get_result_returns_none_for_pending(self, temp_dirs):
        """get_result returns None for pending task."""
        task_id = spawn(
            "Test",
            provider="mock",
            wait=False,
            queue_dir=temp_dirs["queue_dir"],
        )

        # Result not yet available (task pending in queue)
        result = get_result(task_id, queue_dir=temp_dirs["queue_dir"].parent)

        # May be None (still pending) or Result (if processed)
        # The spawn with wait=False just queues, doesn't process
        assert result is None

    def test_get_result_after_spawn_wait(self, temp_dirs):
        """get_result works after synchronous spawn."""
        result = spawn(
            "Test",
            provider="mock",
            wait=True,
            queue_dir=temp_dirs["queue_dir"],
        )

        # After waiting, we can get the result
        fetched = get_result(result.task_id, queue_dir=temp_dirs["queue_dir"].parent)
        assert fetched is not None
        assert fetched.task_id == result.task_id


class TestGetStatus:
    """Tests for get_status function."""

    def test_get_status_pending(self, temp_dirs):
        """get_status returns 'pending' for queued task."""
        task_id = spawn(
            "Test",
            provider="mock",
            wait=False,
            queue_dir=temp_dirs["queue_dir"],
        )

        status = get_status(task_id, queue_dir=temp_dirs["queue_dir"].parent)
        assert status == "pending"

    def test_get_status_completed(self, temp_dirs):
        """get_status returns 'completed' after success."""
        result = spawn(
            "Test",
            provider="mock",
            wait=True,
            queue_dir=temp_dirs["queue_dir"],
        )

        status = get_status(result.task_id, queue_dir=temp_dirs["queue_dir"].parent)
        assert status == "completed"

    def test_get_status_unknown_task(self, temp_dirs):
        """get_status returns None for unknown task."""
        status = get_status("nonexistent-task-id", queue_dir=temp_dirs["queue_dir"].parent)
        assert status is None


class TestWaitForResult:
    """Tests for wait_for_result function."""

    def test_wait_for_completed_task(self, temp_dirs):
        """wait_for_result returns immediately for completed task."""
        result = spawn(
            "Test",
            provider="mock",
            wait=True,
            queue_dir=temp_dirs["queue_dir"],
        )

        waited = wait_for_result(
            result.task_id,
            timeout=1,
            queue_dir=temp_dirs["queue_dir"].parent,
        )

        assert waited.task_id == result.task_id
        assert waited.status == "completed"


class TestSpawnieConfig:
    """Tests for SpawnieConfig."""

    def test_default_values(self):
        """SpawnieConfig has sensible defaults."""
        config = SpawnieConfig()

        assert config.provider == "claude"
        assert config.model is None
        assert config.timeout == 300
        assert config.poll_interval == 0.5

    def test_custom_values(self):
        """SpawnieConfig accepts custom values."""
        config = SpawnieConfig(
            provider="mock",
            model="fast",
            timeout=60,
            poll_interval=0.1,
        )

        assert config.provider == "mock"
        assert config.model == "fast"
        assert config.timeout == 60
        assert config.poll_interval == 0.1

    def test_path_conversion(self):
        """SpawnieConfig converts string paths to Path objects."""
        config = SpawnieConfig(
            response_dir="my-responses",
            queue_dir="my-queue",
        )

        assert isinstance(config.response_dir, Path)
        assert isinstance(config.queue_dir, Path)

    def test_ensure_dirs_creates_directories(self, temp_dirs):
        """ensure_dirs creates necessary directories."""
        config = SpawnieConfig(
            response_dir=temp_dirs["response_dir"],
            queue_dir=temp_dirs["queue_dir"],
        )

        config.ensure_dirs()

        assert temp_dirs["response_dir"].exists()
        assert temp_dirs["queue_dir"].exists()


class TestTaskModel:
    """Tests for Task model."""

    def test_task_creation(self):
        """Task can be created with required fields."""
        task = Task(prompt="Test", provider="mock")

        assert task.prompt == "Test"
        assert task.provider == "mock"
        assert task.id is not None
        assert task.created_at is not None

    def test_task_to_dict(self):
        """Task can be serialized to dict."""
        task = Task(prompt="Test", provider="mock", model="fast")
        data = task.to_dict()

        assert data["prompt"] == "Test"
        assert data["provider"] == "mock"
        assert data["model"] == "fast"
        assert "id" in data
        assert "created_at" in data

    def test_task_from_dict(self):
        """Task can be deserialized from dict."""
        task = Task(prompt="Test", provider="mock")
        data = task.to_dict()

        restored = Task.from_dict(data)

        assert restored.id == task.id
        assert restored.prompt == task.prompt
        assert restored.provider == task.provider


class TestResultModel:
    """Tests for Result model."""

    def test_result_creation(self):
        """Result can be created."""
        result = Result(
            task_id="test-id",
            status="completed",
            output="Output text",
        )

        assert result.task_id == "test-id"
        assert result.status == "completed"
        assert result.output == "Output text"

    def test_result_succeeded(self):
        """succeeded property works correctly."""
        completed = Result(task_id="1", status="completed")
        failed = Result(task_id="2", status="failed")
        timeout = Result(task_id="3", status="timeout")

        assert completed.succeeded is True
        assert failed.succeeded is False
        assert timeout.succeeded is False

    def test_result_failed(self):
        """failed property works correctly."""
        completed = Result(task_id="1", status="completed")
        failed = Result(task_id="2", status="failed")
        timeout = Result(task_id="3", status="timeout")

        assert completed.failed is False
        assert failed.failed is True
        assert timeout.failed is True

    def test_result_to_dict(self):
        """Result can be serialized to dict."""
        result = Result(
            task_id="test-id",
            status="completed",
            output="Output",
            duration_seconds=1.5,
        )
        data = result.to_dict()

        assert data["task_id"] == "test-id"
        assert data["status"] == "completed"
        assert data["output"] == "Output"
        assert data["duration_seconds"] == 1.5

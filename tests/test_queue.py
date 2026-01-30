"""Tests for queue manager."""

import pytest
import tempfile
from pathlib import Path

from spawnie import QueueManager, Task


@pytest.fixture
def temp_queue():
    """Create a temporary queue directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield QueueManager(Path(tmpdir))


class TestQueueManager:
    """Tests for QueueManager."""

    def test_submit_creates_file(self, temp_queue):
        """Submitting a task creates a queue file."""
        task = Task(prompt="Test prompt", provider="mock")
        task_id = temp_queue.submit(task)

        assert task_id == task.id
        assert (temp_queue.queue_dir / f"{task_id}.json").exists()

    def test_claim_next_moves_to_in_progress(self, temp_queue):
        """Claiming a task moves it to in_progress directory."""
        task = Task(prompt="Test prompt", provider="mock")
        task_id = temp_queue.submit(task)

        claimed = temp_queue.claim_next()

        assert claimed is not None
        assert claimed.id == task_id
        assert not (temp_queue.queue_dir / f"{task_id}.json").exists()
        assert (temp_queue.in_progress_dir / f"{task_id}.json").exists()

    def test_claim_next_empty_queue_returns_none(self, temp_queue):
        """Claiming from empty queue returns None."""
        result = temp_queue.claim_next()
        assert result is None

    def test_complete_moves_to_done(self, temp_queue):
        """Completing a task moves it to done directory."""
        task = Task(prompt="Test prompt", provider="mock")
        temp_queue.submit(task)
        temp_queue.claim_next()

        result = temp_queue.complete(task.id, "Output text", 1.5)

        assert result.status == "completed"
        assert result.output == "Output text"
        assert result.duration_seconds == 1.5
        assert (temp_queue.done_dir / f"{task.id}.json").exists()
        assert (temp_queue.done_dir / f"{task.id}.result.json").exists()

    def test_fail_moves_to_failed(self, temp_queue):
        """Failing a task moves it to failed directory."""
        task = Task(prompt="Test prompt", provider="mock")
        temp_queue.submit(task)
        temp_queue.claim_next()

        result = temp_queue.fail(task.id, "Error message", 0.5)

        assert result.status == "failed"
        assert result.error == "Error message"
        assert (temp_queue.failed_dir / f"{task.id}.json").exists()

    def test_get_result_returns_completed(self, temp_queue):
        """get_result returns result after completion."""
        task = Task(prompt="Test", provider="mock")
        temp_queue.submit(task)
        temp_queue.claim_next()
        temp_queue.complete(task.id, "Done", 1.0)

        result = temp_queue.get_result(task.id)

        assert result is not None
        assert result.status == "completed"
        assert result.output == "Done"

    def test_get_result_returns_failed(self, temp_queue):
        """get_result returns result for failed tasks."""
        task = Task(prompt="Test", provider="mock")
        temp_queue.submit(task)
        temp_queue.claim_next()
        temp_queue.fail(task.id, "Oops", 0.1)

        result = temp_queue.get_result(task.id)

        assert result is not None
        assert result.status == "failed"
        assert result.error == "Oops"

    def test_get_result_returns_none_for_pending(self, temp_queue):
        """get_result returns None for pending tasks."""
        task = Task(prompt="Test", provider="mock")
        temp_queue.submit(task)

        result = temp_queue.get_result(task.id)
        assert result is None

    def test_get_status(self, temp_queue):
        """get_status returns correct status at each stage."""
        task = Task(prompt="Test", provider="mock")

        # Not yet submitted
        assert temp_queue.get_status(task.id) is None

        # Pending
        temp_queue.submit(task)
        assert temp_queue.get_status(task.id) == "pending"

        # In progress
        temp_queue.claim_next()
        assert temp_queue.get_status(task.id) == "in_progress"

        # Completed
        temp_queue.complete(task.id, "Done", 1.0)
        assert temp_queue.get_status(task.id) == "completed"

    def test_list_pending(self, temp_queue):
        """list_pending returns all pending tasks."""
        task1 = Task(prompt="Test 1", provider="mock")
        task2 = Task(prompt="Test 2", provider="mock")
        temp_queue.submit(task1)
        temp_queue.submit(task2)

        pending = temp_queue.list_pending()

        assert len(pending) == 2
        ids = {t.id for t in pending}
        assert task1.id in ids
        assert task2.id in ids

    def test_pending_count(self, temp_queue):
        """pending_count returns correct count."""
        assert temp_queue.pending_count() == 0

        task = Task(prompt="Test", provider="mock")
        temp_queue.submit(task)
        assert temp_queue.pending_count() == 1

        temp_queue.claim_next()
        assert temp_queue.pending_count() == 0

    def test_timeout_marks_as_timeout(self, temp_queue):
        """timeout() marks task with timeout status."""
        task = Task(prompt="Test", provider="mock")
        temp_queue.submit(task)
        temp_queue.claim_next()

        result = temp_queue.timeout(task.id, 300.0)

        assert result.status == "timeout"
        assert "timed out" in result.error.lower()

    def test_clear_all(self, temp_queue):
        """clear_all removes all tasks."""
        for i in range(5):
            task = Task(prompt=f"Test {i}", provider="mock")
            temp_queue.submit(task)

        temp_queue.clear_all()

        assert temp_queue.pending_count() == 0
        assert len(list(temp_queue.queue_dir.glob("*.json"))) == 0

"""Tests for task and workflow tracking."""

import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from spawnie.tracker import (
    Tracker,
    TaskState,
    WorkflowState,
    StepState,
    TrackerLimits,
    get_tracker,
    reset_tracker,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "tracker_path": base / "tracker.json",
            "history_dir": base / "history",
        }


@pytest.fixture
def tracker(temp_dirs):
    """Create a tracker with temp directories."""
    reset_tracker()
    return Tracker(
        tracker_path=temp_dirs["tracker_path"],
        history_dir=temp_dirs["history_dir"],
    )


class TestTrackerLimits:
    """Tests for TrackerLimits."""

    def test_default_limits(self):
        """TrackerLimits has sensible defaults."""
        limits = TrackerLimits()
        assert limits.max_concurrent_tasks == 10
        assert limits.max_concurrent_workflows == 5
        assert limits.default_task_timeout == 300
        assert limits.default_workflow_timeout == 1800

    def test_limits_to_dict(self):
        """TrackerLimits serializes to dict."""
        limits = TrackerLimits(max_concurrent_tasks=5)
        d = limits.to_dict()
        assert d["max_concurrent_tasks"] == 5

    def test_limits_from_dict(self):
        """TrackerLimits deserializes from dict."""
        limits = TrackerLimits.from_dict({"max_concurrent_tasks": 20})
        assert limits.max_concurrent_tasks == 20


class TestTaskState:
    """Tests for TaskState."""

    def test_task_state_creation(self):
        """TaskState can be created."""
        task = TaskState(
            id="task-123",
            workflow_id="wf-456",
            step="analyze",
            model="claude-sonnet",
            status="running",
            created_at=datetime.now(),
        )
        assert task.id == "task-123"
        assert task.status == "running"

    def test_task_state_to_dict(self):
        """TaskState serializes to dict."""
        now = datetime.now()
        task = TaskState(
            id="task-123",
            workflow_id=None,
            step=None,
            model="test",
            status="queued",
            created_at=now,
        )
        d = task.to_dict()
        assert d["id"] == "task-123"
        assert d["status"] == "queued"
        assert d["created_at"] == now.isoformat()

    def test_task_state_from_dict(self):
        """TaskState deserializes from dict."""
        now = datetime.now()
        task = TaskState.from_dict({
            "id": "task-123",
            "model": "test",
            "status": "completed",
            "created_at": now.isoformat(),
        })
        assert task.id == "task-123"
        assert task.status == "completed"


class TestStepState:
    """Tests for StepState."""

    def test_step_state_creation(self):
        """StepState can be created."""
        step = StepState(name="analyze", status="pending")
        assert step.name == "analyze"
        assert step.status == "pending"

    def test_step_state_to_dict(self):
        """StepState serializes to dict."""
        step = StepState(name="test", status="running", duration=1.5)
        d = step.to_dict()
        assert d["name"] == "test"
        assert d["duration"] == 1.5


class TestWorkflowState:
    """Tests for WorkflowState."""

    def test_workflow_state_creation(self):
        """WorkflowState can be created."""
        wf = WorkflowState(
            id="wf-123",
            name="test-workflow",
            customer="bam",
            status="running",
            created_at=datetime.now(),
        )
        assert wf.id == "wf-123"
        assert wf.customer == "bam"

    def test_workflow_progress(self):
        """WorkflowState.progress returns correct count."""
        wf = WorkflowState(
            id="wf-123",
            name="test",
            customer="test",
            status="running",
            created_at=datetime.now(),
            steps={
                "step1": StepState(name="step1", status="completed"),
                "step2": StepState(name="step2", status="running"),
                "step3": StepState(name="step3", status="pending"),
            },
        )
        assert wf.progress == "1/3 steps"

    def test_workflow_current_step(self):
        """WorkflowState.current_step returns running step."""
        wf = WorkflowState(
            id="wf-123",
            name="test",
            customer="test",
            status="running",
            created_at=datetime.now(),
            steps={
                "step1": StepState(name="step1", status="completed"),
                "step2": StepState(name="step2", status="running"),
            },
        )
        assert wf.current_step == "step2"


class TestTrackerWorkflows:
    """Tests for workflow management in Tracker."""

    def test_create_workflow(self, tracker):
        """create_workflow creates and tracks workflow."""
        wf = tracker.create_workflow(
            workflow_id="wf-test",
            name="test-workflow",
            customer="bam",
            step_names=["step1", "step2"],
            inputs={"data": "test"},
        )

        assert wf.id == "wf-test"
        assert wf.status == "queued"
        assert "wf-test" in tracker.workflows
        assert len(wf.steps) == 2

    def test_create_workflow_enforces_limits(self, tracker):
        """create_workflow enforces concurrent workflow limit."""
        tracker.limits.max_concurrent_workflows = 2

        # Create 2 workflows
        tracker.create_workflow("wf-1", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-1")
        tracker.create_workflow("wf-2", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-2")

        # Third should fail
        with pytest.raises(RuntimeError, match="limit reached"):
            tracker.create_workflow("wf-3", "test", "bam", ["s1"], {})

    def test_start_workflow(self, tracker):
        """start_workflow updates status and timestamp."""
        tracker.create_workflow("wf-test", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-test")

        wf = tracker.workflows["wf-test"]
        assert wf.status == "running"
        assert wf.started_at is not None

    def test_complete_workflow(self, tracker, temp_dirs):
        """complete_workflow archives and removes workflow."""
        tracker.create_workflow("wf-test", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-test")
        tracker.complete_workflow("wf-test", {"result": "done"})

        assert "wf-test" not in tracker.workflows
        assert tracker.stats["completed_today"] == 1

        # Check history file
        history_files = list(temp_dirs["history_dir"].glob("*.jsonl"))
        assert len(history_files) == 1

    def test_fail_workflow(self, tracker):
        """fail_workflow sets error and archives."""
        tracker.create_workflow("wf-test", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-test")
        tracker.fail_workflow("wf-test", "Something went wrong")

        assert "wf-test" not in tracker.workflows
        assert tracker.stats["failed_today"] == 1

    def test_kill_workflow(self, tracker):
        """kill_workflow stops workflow and its tasks."""
        tracker.create_workflow("wf-test", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-test")
        tracker.create_task("task-1", "test-model", workflow_id="wf-test")
        tracker.start_task("task-1")

        tracker.kill_workflow("wf-test", "user requested")

        assert "wf-test" not in tracker.workflows
        assert "task-1" not in tracker.tasks


class TestTrackerTasks:
    """Tests for task management in Tracker."""

    def test_create_task(self, tracker):
        """create_task creates and tracks task."""
        task = tracker.create_task(
            task_id="task-test",
            model="claude-sonnet",
            workflow_id="wf-123",
            step="analyze",
        )

        assert task.id == "task-test"
        assert task.status == "queued"
        assert "task-test" in tracker.tasks

    def test_create_task_enforces_limits(self, tracker):
        """create_task enforces concurrent task limit."""
        tracker.limits.max_concurrent_tasks = 2

        tracker.create_task("task-1", "test")
        tracker.start_task("task-1")
        tracker.create_task("task-2", "test")
        tracker.start_task("task-2")

        with pytest.raises(RuntimeError, match="limit reached"):
            tracker.create_task("task-3", "test")

    def test_start_task(self, tracker):
        """start_task updates status and timestamp."""
        tracker.create_task("task-test", "test")
        tracker.start_task("task-test", pid=12345)

        task = tracker.tasks["task-test"]
        assert task.status == "running"
        assert task.started_at is not None
        assert task.pid == 12345

    def test_complete_task(self, tracker):
        """complete_task removes task from tracking."""
        tracker.create_task("task-test", "test")
        tracker.start_task("task-test")
        tracker.complete_task("task-test", "output preview")

        assert "task-test" not in tracker.tasks

    def test_fail_task(self, tracker):
        """fail_task sets error and removes task."""
        tracker.create_task("task-test", "test")
        tracker.start_task("task-test")
        tracker.fail_task("task-test", "task failed")

        assert "task-test" not in tracker.tasks

    def test_kill_task(self, tracker):
        """kill_task stops specific task."""
        tracker.create_task("task-test", "test")
        tracker.start_task("task-test")
        tracker.kill_task("task-test", "user cancelled")

        assert "task-test" not in tracker.tasks


class TestTrackerSteps:
    """Tests for step management in Tracker."""

    def test_start_step(self, tracker):
        """start_step updates step status."""
        tracker.create_workflow("wf-test", "test", "bam", ["analyze"], {})
        tracker.start_workflow("wf-test")
        tracker.start_step("wf-test", "analyze", "task-123")

        step = tracker.workflows["wf-test"].steps["analyze"]
        assert step.status == "running"
        assert step.task_id == "task-123"

    def test_complete_step(self, tracker):
        """complete_step updates step status and duration."""
        tracker.create_workflow("wf-test", "test", "bam", ["analyze"], {})
        tracker.start_workflow("wf-test")
        tracker.start_step("wf-test", "analyze", "task-123")
        time.sleep(0.01)  # Small delay for duration
        tracker.complete_step("wf-test", "analyze")

        step = tracker.workflows["wf-test"].steps["analyze"]
        assert step.status == "completed"
        assert step.duration is not None
        assert step.duration >= 0

    def test_fail_step(self, tracker):
        """fail_step sets error on step."""
        tracker.create_workflow("wf-test", "test", "bam", ["analyze"], {})
        tracker.start_workflow("wf-test")
        tracker.start_step("wf-test", "analyze", "task-123")
        tracker.fail_step("wf-test", "analyze", "step failed")

        step = tracker.workflows["wf-test"].steps["analyze"]
        assert step.status == "failed"
        assert step.error == "step failed"


class TestTrackerTimeouts:
    """Tests for timeout handling."""

    def test_check_timeouts_detects_workflow_timeout(self, tracker):
        """check_timeouts detects timed out workflows."""
        tracker.create_workflow("wf-test", "test", "bam", ["s1"], {}, timeout=1)
        tracker.start_workflow("wf-test")

        # Set timeout to past
        tracker.workflows["wf-test"].timeout_at = datetime.now() - timedelta(seconds=10)

        timed_out = tracker.check_timeouts()

        assert "wf-test" in timed_out
        assert "wf-test" not in tracker.workflows

    def test_check_timeouts_detects_task_timeout(self, tracker):
        """check_timeouts detects timed out tasks."""
        tracker.create_task("task-test", "test", timeout=1)
        tracker.start_task("task-test")

        # Set timeout to past
        tracker.tasks["task-test"].timeout_at = datetime.now() - timedelta(seconds=10)

        timed_out = tracker.check_timeouts()

        assert "task-test" in timed_out
        assert "task-test" not in tracker.tasks


class TestTrackerPersistence:
    """Tests for tracker save/load."""

    def test_save_creates_file(self, tracker, temp_dirs):
        """save() creates tracker file."""
        tracker.save()
        assert temp_dirs["tracker_path"].exists()

    def test_save_load_roundtrip(self, temp_dirs):
        """Tracker state survives save/load."""
        tracker1 = Tracker(
            tracker_path=temp_dirs["tracker_path"],
            history_dir=temp_dirs["history_dir"],
        )
        tracker1.create_workflow("wf-test", "test", "bam", ["s1"], {})
        tracker1.save()

        tracker2 = Tracker(
            tracker_path=temp_dirs["tracker_path"],
            history_dir=temp_dirs["history_dir"],
        )
        assert "wf-test" in tracker2.workflows

    def test_load_cleans_finished_workflows(self, temp_dirs):
        """Loading cleans up finished workflows from previous session."""
        # Create tracker file with completed workflow
        data = {
            "updated_at": datetime.now().isoformat(),
            "spawnie_pid": 99999,
            "limits": TrackerLimits().to_dict(),
            "workflows": {
                "wf-done": {
                    "id": "wf-done",
                    "name": "test",
                    "customer": "test",
                    "status": "completed",
                    "created_at": datetime.now().isoformat(),
                    "steps": {},
                    "inputs": {},
                    "outputs": {},
                }
            },
            "tasks": {},
            "stats": {},
        }
        with open(temp_dirs["tracker_path"], "w") as f:
            json.dump(data, f)

        tracker = Tracker(
            tracker_path=temp_dirs["tracker_path"],
            history_dir=temp_dirs["history_dir"],
        )

        # Completed workflow should be archived and removed
        assert "wf-done" not in tracker.workflows


class TestTrackerStatus:
    """Tests for status queries."""

    def test_get_status(self, tracker):
        """get_status returns summary."""
        tracker.create_workflow("wf-1", "test", "bam", ["s1"], {})
        tracker.start_workflow("wf-1")
        tracker.create_task("task-1", "test")

        status = tracker.get_status()

        assert status["workflows"]["running"] == 1
        assert status["tasks"]["queued"] == 1

    def test_list_workflows(self, tracker):
        """list_workflows returns active workflows."""
        tracker.create_workflow("wf-1", "test", "bam", ["s1"], {})
        tracker.create_workflow("wf-2", "test", "other", ["s1"], {})

        all_workflows = tracker.list_workflows()
        assert len(all_workflows) == 2

        bam_workflows = tracker.list_workflows(customer="bam")
        assert len(bam_workflows) == 1
        assert bam_workflows[0].id == "wf-1"

    def test_list_tasks(self, tracker):
        """list_tasks returns active tasks."""
        tracker.create_workflow("wf-1", "test", "bam", ["s1"], {})
        tracker.create_task("task-1", "test", workflow_id="wf-1")
        tracker.create_task("task-2", "test", workflow_id=None)

        all_tasks = tracker.list_tasks()
        assert len(all_tasks) == 2

        wf_tasks = tracker.list_tasks(workflow_id="wf-1")
        assert len(wf_tasks) == 1
        assert wf_tasks[0].id == "task-1"

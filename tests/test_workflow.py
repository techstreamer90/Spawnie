"""Tests for workflow definition and execution."""

import json
import pytest
import tempfile
from pathlib import Path

from spawnie.workflow import (
    WorkflowDefinition,
    StepDefinition,
    WorkflowContext,
    WorkflowExecutor,
    WorkflowResult,
    get_workflow_schema,
    get_workflow_guidance,
)
from spawnie.tracker import reset_tracker, Tracker
from spawnie.registry import reset_registry, ModelRegistry


@pytest.fixture
def temp_dirs():
    """Create temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "tracker_path": base / "tracker.json",
            "history_dir": base / "history",
            "config_path": base / "config.json",
        }


@pytest.fixture
def setup_spawnie(temp_dirs):
    """Set up spawnie with temp directories and mock model."""
    reset_tracker()
    reset_registry()

    # Create registry with mock model
    registry = ModelRegistry(config_path=temp_dirs["config_path"])
    registry.add_model("test-mock", [{"provider": "mock", "priority": 1}])
    registry.save()

    # Create tracker
    tracker = Tracker(
        tracker_path=temp_dirs["tracker_path"],
        history_dir=temp_dirs["history_dir"],
    )

    # Patch global instances
    import spawnie.registry
    import spawnie.tracker
    spawnie.registry._registry = registry
    spawnie.tracker._tracker = tracker

    yield {"registry": registry, "tracker": tracker}

    reset_tracker()
    reset_registry()


class TestStepDefinition:
    """Tests for StepDefinition."""

    def test_step_from_dict(self):
        """StepDefinition parses from dict."""
        step = StepDefinition.from_dict("analyze", {
            "prompt": "Analyze this",
            "model": "claude-sonnet",
            "depends": ["previous"],
            "timeout": 60,
            "retries": 2,
        })

        assert step.name == "analyze"
        assert step.prompt == "Analyze this"
        assert step.model == "claude-sonnet"
        assert step.depends == ["previous"]
        assert step.timeout == 60
        assert step.retries == 2

    def test_step_defaults(self):
        """StepDefinition has sensible defaults."""
        step = StepDefinition.from_dict("test", {
            "prompt": "Test",
            "model": "test",
        })

        assert step.depends == []
        assert step.retries == 0
        assert step.timeout is None
        assert step.condition is None


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition."""

    def test_workflow_from_dict(self):
        """WorkflowDefinition parses from dict."""
        wf = WorkflowDefinition.from_dict({
            "name": "test-workflow",
            "description": "A test workflow",
            "inputs": {"data": "string"},
            "steps": {
                "step1": {"prompt": "Do step 1", "model": "test"},
                "step2": {"prompt": "Do step 2", "model": "test", "depends": ["step1"]},
            },
            "outputs": {"result": "{{steps.step2.output}}"},
        })

        assert wf.name == "test-workflow"
        assert wf.description == "A test workflow"
        assert "data" in wf.inputs
        assert len(wf.steps) == 2
        assert wf.steps["step2"].depends == ["step1"]

    def test_workflow_from_json(self):
        """WorkflowDefinition parses from JSON string."""
        json_str = json.dumps({
            "name": "json-workflow",
            "steps": {"s1": {"prompt": "test", "model": "test"}},
        })

        wf = WorkflowDefinition.from_json(json_str)
        assert wf.name == "json-workflow"

    def test_workflow_from_file(self, tmp_path):
        """WorkflowDefinition loads from file."""
        workflow_file = tmp_path / "workflow.json"
        workflow_file.write_text(json.dumps({
            "name": "file-workflow",
            "steps": {"s1": {"prompt": "test", "model": "test"}},
        }))

        wf = WorkflowDefinition.from_file(workflow_file)
        assert wf.name == "file-workflow"

    def test_workflow_to_dict(self):
        """WorkflowDefinition serializes to dict."""
        wf = WorkflowDefinition(
            name="test",
            description="Test",
            steps={"s1": StepDefinition(name="s1", prompt="test", model="test")},
        )

        d = wf.to_dict()
        assert d["name"] == "test"
        assert "s1" in d["steps"]


class TestWorkflowValidation:
    """Tests for workflow validation."""

    def test_validate_empty_steps(self):
        """Validation catches empty steps."""
        wf = WorkflowDefinition(name="test", description="", steps={})
        errors = wf.validate()
        assert any("at least one step" in e for e in errors)

    def test_validate_missing_dependency(self):
        """Validation catches missing dependencies."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "s1": StepDefinition(name="s1", prompt="test", model="test", depends=["nonexistent"]),
            },
        )
        errors = wf.validate()
        assert any("unknown step" in e for e in errors)

    def test_validate_circular_dependency(self):
        """Validation catches circular dependencies."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "s1": StepDefinition(name="s1", prompt="test", model="test", depends=["s2"]),
                "s2": StepDefinition(name="s2", prompt="test", model="test", depends=["s1"]),
            },
        )
        errors = wf.validate()
        assert any("circular" in e.lower() for e in errors)

    def test_validate_valid_workflow(self, setup_spawnie):
        """Valid workflow passes validation."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "s1": StepDefinition(name="s1", prompt="test", model="test-mock"),
                "s2": StepDefinition(name="s2", prompt="test", model="test-mock", depends=["s1"]),
            },
        )
        errors = wf.validate()
        assert len(errors) == 0


class TestExecutionOrder:
    """Tests for execution order calculation."""

    def test_simple_order(self):
        """Simple linear workflow has correct order."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "s1": StepDefinition(name="s1", prompt="test", model="test"),
                "s2": StepDefinition(name="s2", prompt="test", model="test", depends=["s1"]),
            },
        )

        order = wf.get_execution_order()
        assert order == [["s1"], ["s2"]]

    def test_parallel_order(self):
        """Parallel steps are grouped together."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "s1": StepDefinition(name="s1", prompt="test", model="test"),
                "s2": StepDefinition(name="s2", prompt="test", model="test"),
                "s3": StepDefinition(name="s3", prompt="test", model="test", depends=["s1", "s2"]),
            },
        )

        order = wf.get_execution_order()
        # s1 and s2 can run in parallel (first group)
        # s3 runs after (second group)
        assert len(order) == 2
        assert set(order[0]) == {"s1", "s2"}
        assert order[1] == ["s3"]

    def test_complex_order(self):
        """Complex DAG has correct order."""
        wf = WorkflowDefinition(
            name="test",
            description="",
            steps={
                "a": StepDefinition(name="a", prompt="test", model="test"),
                "b": StepDefinition(name="b", prompt="test", model="test", depends=["a"]),
                "c": StepDefinition(name="c", prompt="test", model="test", depends=["a"]),
                "d": StepDefinition(name="d", prompt="test", model="test", depends=["b", "c"]),
            },
        )

        order = wf.get_execution_order()
        assert order[0] == ["a"]
        assert set(order[1]) == {"b", "c"}
        assert order[2] == ["d"]


class TestWorkflowContext:
    """Tests for WorkflowContext template resolution."""

    def test_resolve_input(self):
        """Context resolves input variables."""
        ctx = WorkflowContext(
            workflow_id="wf-1",
            inputs={"data": "hello", "count": 42},
        )

        result = ctx.resolve_template("Input: {{inputs.data}}, Count: {{inputs.count}}")
        assert result == "Input: hello, Count: 42"

    def test_resolve_step_output(self):
        """Context resolves step output variables."""
        ctx = WorkflowContext(
            workflow_id="wf-1",
            inputs={},
            step_outputs={"analyze": "analysis result"},
        )

        result = ctx.resolve_template("Previous: {{steps.analyze.output}}")
        assert result == "Previous: analysis result"

    def test_resolve_unknown_keeps_original(self):
        """Unknown variables keep original placeholder."""
        ctx = WorkflowContext(workflow_id="wf-1", inputs={})

        result = ctx.resolve_template("Unknown: {{inputs.missing}}")
        assert result == "Unknown: {{inputs.missing}}"

    def test_resolve_mixed(self):
        """Context resolves mixed variables."""
        ctx = WorkflowContext(
            workflow_id="wf-1",
            inputs={"name": "test"},
            step_outputs={"s1": "output1"},
        )

        result = ctx.resolve_template("Name: {{inputs.name}}, S1: {{steps.s1.output}}")
        assert result == "Name: test, S1: output1"


class TestWorkflowExecution:
    """Tests for workflow execution."""

    def test_execute_simple_workflow(self, setup_spawnie):
        """Execute a simple single-step workflow."""
        wf = WorkflowDefinition(
            name="simple",
            description="Simple workflow",
            steps={
                "s1": StepDefinition(name="s1", prompt="Test prompt", model="test-mock"),
            },
            outputs={"result": "{{steps.s1.output}}"},
        )

        executor = WorkflowExecutor()
        result = executor.execute(wf, inputs={}, customer="test")

        assert result.status == "completed"
        assert "result" in result.outputs
        assert "[Mock Response]" in result.outputs["result"]

    def test_execute_multi_step_workflow(self, setup_spawnie):
        """Execute a multi-step workflow with dependencies."""
        wf = WorkflowDefinition(
            name="multi",
            description="Multi-step workflow",
            steps={
                "analyze": StepDefinition(
                    name="analyze",
                    prompt="Analyze: {{inputs.data}}",
                    model="test-mock",
                ),
                "summarize": StepDefinition(
                    name="summarize",
                    prompt="Summarize: {{steps.analyze.output}}",
                    model="test-mock",
                    depends=["analyze"],
                ),
            },
            outputs={"summary": "{{steps.summarize.output}}"},
        )

        executor = WorkflowExecutor()
        result = executor.execute(wf, inputs={"data": "test data"}, customer="test")

        assert result.status == "completed"
        assert "analyze" in result.step_results
        assert "summarize" in result.step_results
        assert result.step_results["analyze"]["status"] == "completed"
        assert result.step_results["summarize"]["status"] == "completed"

    def test_execute_with_inputs(self, setup_spawnie):
        """Workflow receives and uses inputs."""
        wf = WorkflowDefinition(
            name="with-inputs",
            description="",
            inputs={"message": "string"},
            steps={
                "echo": StepDefinition(
                    name="echo",
                    prompt="Echo this: {{inputs.message}}",
                    model="test-mock",
                ),
            },
            outputs={},
        )

        executor = WorkflowExecutor()
        result = executor.execute(wf, inputs={"message": "Hello World"}, customer="test")

        assert result.status == "completed"
        # Check the prompt was resolved correctly
        assert "Hello World" in result.step_results["echo"]["output"]

    def test_execute_records_duration(self, setup_spawnie):
        """Execution records duration."""
        wf = WorkflowDefinition(
            name="timed",
            description="",
            steps={"s1": StepDefinition(name="s1", prompt="test", model="test-mock")},
            outputs={},
        )

        executor = WorkflowExecutor()
        result = executor.execute(wf, inputs={}, customer="test")

        assert result.duration_seconds >= 0
        assert result.started_at is not None
        assert result.completed_at is not None


class TestWorkflowSchema:
    """Tests for workflow schema and guidance."""

    def test_get_workflow_schema(self):
        """get_workflow_schema returns valid schema."""
        schema = get_workflow_schema()

        assert schema["type"] == "object"
        assert "steps" in schema["properties"]
        assert "name" in schema["required"]

    def test_get_workflow_guidance(self):
        """get_workflow_guidance returns documentation."""
        guidance = get_workflow_guidance()

        assert "Workflow Definition Guide" in guidance
        assert "{{inputs." in guidance
        assert "{{steps." in guidance
        assert "depends" in guidance


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_result_to_dict(self):
        """WorkflowResult serializes to dict."""
        from datetime import datetime

        result = WorkflowResult(
            workflow_id="wf-123",
            name="test",
            status="completed",
            outputs={"result": "done"},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            duration_seconds=1.5,
        )

        d = result.to_dict()
        assert d["workflow_id"] == "wf-123"
        assert d["status"] == "completed"
        assert d["duration_seconds"] == 1.5

"""Workflow definition and execution for Spawnie.

Workflows are declarative definitions of multi-step LLM tasks.
Customers (like BAM) define workflows in JSON, Spawnie executes them.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

from .tracker import get_tracker, WorkflowState
from .registry import get_registry
from .providers import get_provider
from .api import SELF_REVIEW_PROMPT, EXTERNAL_REVIEW_PROMPT, FINAL_SYNTHESIS_PROMPT

logger = logging.getLogger("spawnie.workflow")


# JSON Schema for workflow definitions
WORKFLOW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["name", "steps"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Workflow name/identifier",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description",
        },
        "inputs": {
            "type": "object",
            "description": "Input parameter definitions",
            "additionalProperties": {
                "type": "string",
                "description": "Parameter type hint",
            },
        },
        "steps": {
            "type": "object",
            "description": "Workflow steps keyed by name",
            "additionalProperties": {
                "type": "object",
                "required": ["prompt", "model"],
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt template with {{variable}} placeholders",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (from registry)",
                    },
                    "depends": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps that must complete before this one",
                    },
                    "output": {
                        "type": "string",
                        "description": "Name for this step's output",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Step timeout in seconds",
                    },
                    "retries": {
                        "type": "integer",
                        "description": "Number of retries on failure",
                        "default": 0,
                    },
                    "condition": {
                        "type": "string",
                        "description": "Condition expression (skip if false)",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["normal", "extra-clean", "hypertask"],
                        "default": "normal",
                        "description": "Quality level: normal (fast), extra-clean (self-review), hypertask (dual review)",
                    },
                },
            },
        },
        "outputs": {
            "type": "object",
            "description": "Output mappings from step outputs",
            "additionalProperties": {
                "type": "string",
            },
        },
        "timeout": {
            "type": "integer",
            "description": "Overall workflow timeout in seconds",
        },
    },
}


@dataclass
class StepDefinition:
    """Definition of a single workflow step."""
    name: str
    prompt: str
    model: str
    depends: list[str] = field(default_factory=list)
    output: str | None = None
    timeout: int | None = None
    retries: int = 0
    condition: str | None = None
    quality: str = "normal"  # "normal" | "extra-clean" | "hypertask"

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "StepDefinition":
        return cls(
            name=name,
            prompt=data["prompt"],
            model=data["model"],
            depends=data.get("depends", []),
            output=data.get("output"),
            timeout=data.get("timeout"),
            retries=data.get("retries", 0),
            condition=data.get("condition"),
            quality=data.get("quality", "normal"),
        )


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str
    steps: dict[str, StepDefinition]
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    timeout: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        steps = {
            name: StepDefinition.from_dict(name, step_data)
            for name, step_data in data.get("steps", {}).items()
        }
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            timeout=data.get("timeout"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowDefinition":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: Path) -> "WorkflowDefinition":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": {
                name: {
                    "prompt": step.prompt,
                    "model": step.model,
                    "depends": step.depends,
                    "output": step.output,
                    "timeout": step.timeout,
                    "retries": step.retries,
                    "condition": step.condition,
                }
                for name, step in self.steps.items()
            },
            "outputs": self.outputs,
            "timeout": self.timeout,
        }

    def validate(self) -> list[str]:
        """Validate the workflow definition. Returns list of errors."""
        errors = []

        # Check for empty steps
        if not self.steps:
            errors.append("Workflow must have at least one step")

        # Check dependencies exist
        step_names = set(self.steps.keys())
        for name, step in self.steps.items():
            for dep in step.depends:
                if dep not in step_names:
                    errors.append(f"Step '{name}' depends on unknown step '{dep}'")

        # Check for circular dependencies (only if no missing dependencies)
        if not any("unknown step" in e for e in errors):
            visited = set()
            rec_stack = set()

            def has_cycle(step_name: str) -> bool:
                if step_name not in self.steps:
                    return False  # Skip unknown steps
                visited.add(step_name)
                rec_stack.add(step_name)
                for dep in self.steps[step_name].depends:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
                rec_stack.remove(step_name)
                return False

            for step_name in self.steps:
                if step_name not in visited:
                    if has_cycle(step_name):
                        errors.append("Workflow has circular dependencies")
                        break

        # Check models exist in registry
        registry = get_registry()
        for name, step in self.steps.items():
            route = registry.get_best_route(step.model)
            if not route:
                errors.append(f"Step '{name}' uses unavailable model '{step.model}'")

        return errors

    def get_execution_order(self) -> list[list[str]]:
        """Get steps in execution order (list of parallel groups)."""
        # Topological sort with parallel grouping
        completed = set()
        order = []

        while len(completed) < len(self.steps):
            # Find all steps whose dependencies are satisfied
            ready = []
            for name, step in self.steps.items():
                if name not in completed:
                    if all(dep in completed for dep in step.depends):
                        ready.append(name)

            if not ready:
                # This shouldn't happen if validate() passed
                raise RuntimeError("Unable to resolve step dependencies")

            order.append(ready)
            completed.update(ready)

        return order


@dataclass
class WorkflowContext:
    """Runtime context for workflow execution."""
    workflow_id: str
    inputs: dict[str, Any]
    step_outputs: dict[str, str] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    def resolve_template(self, template: str) -> str:
        """Resolve {{variable}} placeholders in a template string."""
        def replace(match):
            path = match.group(1).strip()
            value = self._resolve_path(path)
            if value is None:
                return match.group(0)  # Keep original if not found
            return str(value)

        return re.sub(r'\{\{(.+?)\}\}', replace, template)

    def _resolve_path(self, path: str) -> Any:
        """Resolve a dotted path like 'steps.analyze.output' or 'inputs.data'."""
        parts = path.split(".")

        if parts[0] == "inputs" and len(parts) > 1:
            return self.inputs.get(parts[1])

        if parts[0] == "steps" and len(parts) >= 2:
            step_name = parts[1]
            if step_name in self.step_outputs:
                return self.step_outputs[step_name]

        if parts[0] in self.variables:
            return self.variables[parts[0]]

        return None


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_id: str
    name: str
    status: str  # "completed" | "failed" | "timeout" | "killed"
    outputs: dict[str, Any]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    error: str | None = None
    step_results: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "outputs": self.outputs,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "step_results": self.step_results,
        }


class WorkflowExecutor:
    """Executes workflow definitions."""

    def __init__(self):
        self.tracker = get_tracker()
        self.registry = get_registry()

    def execute(
        self,
        definition: WorkflowDefinition,
        inputs: dict[str, Any],
        customer: str = "unknown",
        timeout: int | None = None,
    ) -> WorkflowResult:
        """Execute a workflow definition with given inputs.

        Args:
            definition: The workflow definition to execute.
            inputs: Input values for the workflow.
            customer: Customer identifier for tracking.
            timeout: Override workflow timeout.

        Returns:
            WorkflowResult with outputs or error information.
        """
        # Validate
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        # Create workflow tracking
        workflow_id = f"wf-{uuid.uuid4().hex[:12]}"
        step_names = list(definition.steps.keys())

        self.tracker.create_workflow(
            workflow_id=workflow_id,
            name=definition.name,
            customer=customer,
            step_names=step_names,
            inputs=inputs,
            timeout=timeout or definition.timeout,
        )

        # Execute
        started_at = datetime.now()
        context = WorkflowContext(workflow_id=workflow_id, inputs=inputs)
        step_results = {}

        try:
            self.tracker.start_workflow(workflow_id)

            # Get execution order (parallel groups)
            execution_order = definition.get_execution_order()

            for parallel_group in execution_order:
                # For now, execute sequentially within groups
                # TODO: Add actual parallel execution
                for step_name in parallel_group:
                    step = definition.steps[step_name]

                    # Check condition
                    if step.condition:
                        condition_result = self._evaluate_condition(step.condition, context)
                        if not condition_result:
                            self.tracker.complete_step(workflow_id, step_name)
                            step_results[step_name] = {"status": "skipped", "reason": "condition false"}
                            continue

                    # Execute step with retries
                    result = self._execute_step(
                        workflow_id=workflow_id,
                        step=step,
                        context=context,
                        retries=step.retries,
                    )

                    step_results[step_name] = result

                    if result["status"] == "failed":
                        raise RuntimeError(f"Step '{step_name}' failed: {result.get('error')}")

                    # Store output
                    if step.output:
                        context.step_outputs[step_name] = result["output"]
                    else:
                        context.step_outputs[step_name] = result["output"]

            # Build outputs
            outputs = {}
            for output_name, output_template in definition.outputs.items():
                outputs[output_name] = context.resolve_template(output_template)

            # Complete workflow
            completed_at = datetime.now()
            self.tracker.complete_workflow(workflow_id, outputs)

            return WorkflowResult(
                workflow_id=workflow_id,
                name=definition.name,
                status="completed",
                outputs=outputs,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                step_results=step_results,
            )

        except Exception as e:
            completed_at = datetime.now()
            error_msg = str(e)
            self.tracker.fail_workflow(workflow_id, error_msg)

            return WorkflowResult(
                workflow_id=workflow_id,
                name=definition.name,
                status="failed",
                outputs={},
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                error=error_msg,
                step_results=step_results,
            )

    def _execute_step(
        self,
        workflow_id: str,
        step: StepDefinition,
        context: WorkflowContext,
        retries: int,
    ) -> dict:
        """Execute a single step with retries."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"

        # Resolve prompt template
        prompt = context.resolve_template(step.prompt)

        # Get provider for model
        route_info = self.registry.get_best_route(step.model)
        if not route_info:
            return {"status": "failed", "error": f"No route for model: {step.model}"}

        route, provider_config = route_info

        # Map to internal provider
        if provider_config.type == "mock":
            internal_provider = "mock"
        elif provider_config.type == "cli" and "claude" in provider_config.name:
            internal_provider = "claude"
        elif provider_config.type == "cli" and "copilot" in provider_config.name:
            internal_provider = "copilot"
        else:
            internal_provider = "mock"

        provider = get_provider(internal_provider)

        # Create description from step name and prompt preview
        prompt_preview = prompt[:30].replace('\n', ' ')
        description = f"Step '{step.name}': {prompt_preview}..."

        # Track task
        self.tracker.create_task(
            task_id=task_id,
            model=step.model,
            workflow_id=workflow_id,
            step=step.name,
            timeout=step.timeout,
            description=description,
        )
        self.tracker.start_step(workflow_id, step.name, task_id)
        self.tracker.start_task(task_id)

        # Execute with retries
        last_error = None
        for attempt in range(retries + 1):
            try:
                output, exit_code = provider.execute(prompt, route.model_id)

                if exit_code == 0:
                    # Apply quality-level review if specified
                    if step.quality != "normal" and output:
                        output = self._apply_step_review(
                            output=output,
                            original_prompt=prompt,
                            quality=step.quality,
                            provider=provider,
                            model_id=route.model_id,
                        )

                    self.tracker.complete_task(task_id, output[:200] if output else None)
                    self.tracker.complete_step(workflow_id, step.name)
                    return {
                        "status": "completed",
                        "output": output,
                        "attempts": attempt + 1,
                    }
                else:
                    last_error = output or f"Exit code {exit_code}"

            except Exception as e:
                last_error = str(e)

            if attempt < retries:
                logger.warning("Step %s attempt %d failed, retrying: %s",
                              step.name, attempt + 1, last_error)

        # All retries failed
        self.tracker.fail_task(task_id, last_error or "Unknown error")
        self.tracker.fail_step(workflow_id, step.name, last_error or "Unknown error")
        return {
            "status": "failed",
            "error": last_error,
            "attempts": retries + 1,
        }

    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """Evaluate a simple condition expression."""
        # For now, just check if referenced values are truthy
        # e.g., "steps.analyze.output" checks if that output exists and is non-empty
        resolved = context.resolve_template(f"{{{{{condition}}}}}")
        return bool(resolved and resolved != f"{{{{{condition}}}}}")

    def _apply_step_review(
        self,
        output: str,
        original_prompt: str,
        quality: str,
        provider,
        model_id: str | None,
    ) -> str:
        """
        Apply quality-level review to step output.

        Based on benchmark findings:
        - extra-clean (self-review): catches omissions like security, cost, edge cases
        - hypertask (dual review): catches both omissions AND contradictions
        """
        try:
            # Step 1: Self-review (for both extra-clean and hypertask)
            self_review_prompt = SELF_REVIEW_PROMPT.format(
                original_prompt=original_prompt,
                output=output,
            )
            reviewed_output, exit_code = provider.execute(self_review_prompt, model_id)

            if exit_code != 0:
                logger.warning("Self-review failed, returning original output")
                return output

            # For extra-clean, return self-reviewed output
            if quality == "extra-clean":
                return reviewed_output

            # Step 2: External review (for hypertask only)
            external_prompt = EXTERNAL_REVIEW_PROMPT.format(
                original_prompt=original_prompt,
                output=reviewed_output,
            )
            external_review, exit_code = provider.execute(external_prompt, model_id)

            if exit_code != 0:
                logger.warning("External review failed, returning self-reviewed output")
                return reviewed_output

            # Step 3: Final synthesis
            synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(
                original_prompt=original_prompt,
                self_reviewed_output=reviewed_output,
                external_review=external_review,
            )
            final_output, exit_code = provider.execute(synthesis_prompt, model_id)

            if exit_code != 0:
                # Return self-reviewed with external review attached
                return f"{reviewed_output}\n\n---\nEXTERNAL REVIEW:\n{external_review}"

            return final_output

        except Exception as e:
            logger.error("Review process error: %s", e)
            return output  # Return original on error


def get_workflow_schema() -> dict:
    """Get the JSON schema for workflow definitions."""
    return WORKFLOW_SCHEMA


def get_workflow_guidance() -> str:
    """Get guidance text for agents on how to construct workflows."""
    return """
# Spawnie Workflow Definition Guide

Workflows are declarative JSON definitions that describe multi-step LLM tasks.

## Basic Structure

```json
{
  "name": "my-workflow",
  "description": "What this workflow does",

  "inputs": {
    "data": "string",
    "options": "object"
  },

  "steps": {
    "step1": {
      "prompt": "Analyze this: {{inputs.data}}",
      "model": "claude-haiku"
    },
    "step2": {
      "prompt": "Based on: {{steps.step1.output}}\\nDo more analysis.",
      "model": "claude-sonnet",
      "depends": ["step1"]
    }
  },

  "outputs": {
    "result": "{{steps.step2.output}}"
  }
}
```

## Template Variables

Use `{{path}}` syntax to reference:
- `{{inputs.name}}` - Input parameters
- `{{steps.stepname.output}}` - Output from a previous step

## Step Properties

- `prompt` (required): The prompt template
- `model` (required): Model from registry (e.g., "claude-sonnet", "claude-haiku")
- `depends`: Array of step names that must complete first
- `output`: Name for this step's output (defaults to step name)
- `timeout`: Step-specific timeout in seconds
- `retries`: Number of retry attempts on failure (default: 0)
- `condition`: Skip step if condition is falsy

## Execution Order

Steps run in dependency order. Steps with no dependencies (or satisfied dependencies)
can potentially run in parallel.

## Available Models

Use `spawnie models` to see available models and their routes.
Common models: claude-sonnet, claude-opus, claude-haiku

## Example: Code Review Workflow

```json
{
  "name": "code-review",
  "description": "Multi-perspective code review",

  "inputs": {
    "code": "string",
    "language": "string"
  },

  "steps": {
    "security": {
      "prompt": "Review this {{inputs.language}} code for security issues:\\n```\\n{{inputs.code}}\\n```",
      "model": "claude-sonnet",
      "output": "security_review"
    },
    "performance": {
      "prompt": "Review this {{inputs.language}} code for performance issues:\\n```\\n{{inputs.code}}\\n```",
      "model": "claude-sonnet",
      "output": "perf_review"
    },
    "summary": {
      "prompt": "Summarize these code reviews:\\n\\nSecurity:\\n{{steps.security.output}}\\n\\nPerformance:\\n{{steps.performance.output}}",
      "model": "claude-haiku",
      "depends": ["security", "performance"],
      "output": "summary"
    }
  },

  "outputs": {
    "security": "{{steps.security.output}}",
    "performance": "{{steps.performance.output}}",
    "summary": "{{steps.summary.output}}"
  }
}
```
"""

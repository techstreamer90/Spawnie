# Spawnie

Model router and workflow orchestrator for CLI agents. Routes model requests to the best available provider (CLI subscriptions, API fallback) to minimize costs.

## Agent Quick Reference

```bash
# Run a prompt
spawnie run "Your prompt here" -m claude-sonnet

# Execute a workflow
spawnie workflow path/to/workflow.json -i key=value

# Monitor in real-time (TUI)
spawnie monitor

# Check status
spawnie status
spawnie status --watch  # Live updates

# Kill a workflow
spawnie kill wf-abc123
```

### TUI Monitor Keybindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh |
| `k` | Kill first running workflow |
| `s` | Save screenshot (SVG) |
| `Ctrl+P` | Command palette |

The monitor shows real-time workflow progress, active tasks, and alerts. Screenshots are saved as SVG files in the current directory.

## Features

- **Model Router**: Request a model (e.g., "claude-sonnet"), Spawnie finds the best available provider
- **CLI Priority**: Uses CLI subscriptions (Claude CLI, Copilot CLI) before API fallback
- **Workflow Orchestration**: Define multi-step workflows in JSON with dependencies
- **Real-time Tracking**: Monitor all running tasks and workflows via TUI or CLI
- **Screenshot Capture**: Save TUI state as SVG for debugging/documentation

## Installation

```bash
pip install spawnie
# Or from source
pip install -e C:/spawnie

# For TUI monitor
pip install spawnie[monitor]
# Or: pip install textual
```

## Workflows

### Workflow JSON Format

```json
{
  "name": "workflow-name",
  "description": "What this workflow does",

  "inputs": {
    "param1": "string",
    "param2": "string"
  },

  "steps": {
    "step1": {
      "prompt": "Do something with {{inputs.param1}}",
      "model": "claude-haiku"
    },
    "step2": {
      "prompt": "Process: {{steps.step1.output}}",
      "model": "claude-sonnet",
      "depends": ["step1"]
    }
  },

  "outputs": {
    "result": "{{steps.step2.output}}"
  }
}
```

### Template Variables

- `{{inputs.name}}` - Reference workflow inputs
- `{{steps.step_name.output}}` - Reference output from a previous step

### Step Dependencies

Use `"depends": ["step1", "step2"]` to ensure steps run in order. Steps without dependencies run in parallel.

### Execute Workflows

```bash
# CLI
spawnie workflow workflow.json -i topic="quantum computing"

# Multiple inputs
spawnie workflow workflow.json -i topic="AI" -i depth="detailed"

# JSON inputs
spawnie workflow workflow.json --inputs-json '{"topic": "AI", "count": 5}'
```

```python
# Python
from spawnie import execute

result = execute(
    "workflow.json",
    inputs={"topic": "quantum computing"},
    customer="my-app",
)
print(result.status)       # "completed" or "failed"
print(result.outputs)      # {"result": "..."}
print(result.step_results) # Per-step details
```

## Python API

```python
from spawnie import run, list_models, execute

# List available models
for model in list_models():
    if model["available"]:
        print(f"{model['name']} via {model['route']}")

# Run a single prompt
result = run("Explain quantum computing", model="claude-sonnet")
print(result.output)
print(result.status)           # "completed" or "failed"
print(result.duration_seconds) # Execution time

# Async mode - returns task ID immediately
task_id = run("Long task", model="claude-opus", mode="async")

# Execute a workflow
result = execute("workflow.json", inputs={"data": "..."})
```

## Real-time Monitoring

### TUI Monitor (Recommended)

```bash
spawnie monitor
```

Opens a terminal UI showing:
- Active workflows with step progress
- Running tasks
- Recent alerts
- Live stats (completed/failed today)

Press `s` to save a screenshot as SVG.

### CLI Status

```bash
# Summary
spawnie status

# Watch mode (updates every 2s)
spawnie status --watch

# Specific workflow details
spawnie status wf-abc123

# Kill a workflow or task
spawnie kill wf-abc123
```

## Model Registry

Spawnie routes model requests to the best available provider:

```
claude-sonnet → claude-cli (priority 1) → anthropic-api (priority 2)
claude-opus   → claude-cli (priority 1) → anthropic-api (priority 2)
claude-haiku  → claude-cli (priority 1) → anthropic-api (priority 2)
```

CLI subscriptions are preferred (no per-token cost). API is fallback.

### Configuration

Config lives at `~/.spawnie/config.json`:

```json
{
  "providers": {
    "claude-cli": {"type": "cli", "command": "claude"},
    "anthropic-api": {"type": "api", "api_key_env": "ANTHROPIC_API_KEY"}
  },
  "models": {
    "claude-sonnet": {
      "routes": [
        {"provider": "claude-cli", "priority": 1},
        {"provider": "anthropic-api", "priority": 2}
      ]
    }
  }
}
```

## CLI Reference

```bash
# Setup
spawnie setup              # Initialize with defaults
spawnie config             # Show config summary
spawnie config --show      # Show full config file
spawnie models             # List models and availability

# Run prompts
spawnie run "prompt" -m MODEL           # Blocking (default)
spawnie run "prompt" -m MODEL --mode async   # Returns task ID

# Workflows
spawnie workflow FILE -i key=value      # Execute workflow
spawnie guidance                        # Show workflow JSON guide

# Monitoring
spawnie monitor               # TUI monitor (recommended)
spawnie status                # CLI status summary
spawnie status --watch        # CLI watch mode
spawnie status WORKFLOW_ID    # Specific workflow
spawnie kill TARGET_ID        # Kill workflow or task

# Diagnostics
spawnie detect claude         # Check Claude CLI availability
spawnie detect copilot        # Check Copilot CLI availability
```

## File Structure

```
~/.spawnie/
├── config.json       # Model registry and provider config
├── tracker.json      # Real-time state (workflows, tasks, alerts)
└── history/          # Archived completed workflows
    └── 2024-01-15.jsonl
```

## License

MIT

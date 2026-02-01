# Spawnie

Model router and workflow orchestrator for CLI agents. Routes model requests to the best available provider (CLI subscriptions, API fallback) to minimize costs.

**For Agents**: Spawnie lets you call LLMs without API keys by routing to CLI tools (Claude CLI, GitHub Copilot) that use existing subscriptions. Request a model like `claude-sonnet` and Spawnie finds an available provider automatically.

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
from spawnie import run, list_models, execute, get_result, wait_for_result

# List available models
for model in list_models():
    if model["available"]:
        print(f"{model['name']} via {model['route']}")

# Run a single prompt (blocking by default)
result = run("Explain quantum computing", model="claude-sonnet")
print(result.output)
print(result.status)           # "completed" or "failed"
print(result.duration_seconds) # Execution time

# Async mode - returns task ID immediately
task_id = run("Long task", model="claude-opus", mode="async")

# Poll for result (returns None if not ready)
result = get_result(task_id)

# Or wait with timeout
result = wait_for_result(task_id, timeout=300)

# Execute a workflow
result = execute("workflow.json", inputs={"data": "..."})
```

### Error Handling

```python
result = run("prompt", model="claude-sonnet")

if result.status == "completed":
    print(result.output)
elif result.status == "failed":
    print(f"Error: {result.error}")
elif result.status == "timeout":
    print("Task timed out")

# Result object fields:
# - task_id: str
# - status: "completed" | "failed" | "timeout"
# - output: str | None (on success)
# - error: str | None (on failure)
# - duration_seconds: float
# - completed_at: datetime | None
```

### Testing with Mock Provider

Use the mock provider for testing without making real LLM calls:

```python
from spawnie import run
from spawnie.registry import get_registry

# Add a test model that uses mock provider
registry = get_registry()
registry.add_model("test-model", [{"provider": "mock", "priority": 1}])

# Now use it
result = run("test prompt", model="test-model")
# Returns: "[Mock Response]\nPrompt received (11 chars)..."
```

## Shell Sessions (Interactive Agents)

Shell sessions spawn agents with full file system access. Two modes available:

| Mode | Flag | Use Case |
|------|------|----------|
| **Interactive** | `--new-window` / `-n` | Opens new terminal for direct user interaction |
| **Background** | `--background` / `-b` | Programmatic control with real-time event logging |

### Interactive Mode (New Window)

Opens a new terminal window where you can chat directly with the agent:

```bash
# Spawn interactive agent in new window
spawnie shell "Help me refactor this codebase" -m claude-sonnet --new-window

# Short form
spawnie shell "Review my code for bugs" -m claude-haiku -n

# Example: Cleanup assistant
spawnie shell "Help me clean up old sessions and temp files. Run 'spawnie sessions --all' to start." -m claude-sonnet -n
```

The agent receives your prompt, responds, and stays interactive for follow-up conversation.

### Background Mode (Programmatic Control)

Starts a session in the background with real-time event logging. Ideal for orchestration by other agents:

```bash
# Start background session - returns session ID immediately
spawnie shell "Monitor the system and report status" -m claude-haiku --background
# Output: session-abc123

# With JSON output
spawnie shell "Your task" -m claude-haiku -b --json
# Output: {"session_id": "session-abc123", "status": "running", ...}

# Poll for events (non-blocking)
spawnie session-events session-abc123 --json

# Respond to questions
spawnie session-respond session-abc123 evt-id "your answer"

# Example: Monitoring agent
spawnie shell "Periodically report progress using 'spawnie progress'. Signal completion with 'spawnie done'." -m claude-haiku -b
```

### Event Polling (Background Mode)

```bash
# Check session events
spawnie session-events <session-id> --json

# Output:
{
  "session_id": "session-abc123",
  "status": "running",
  "events": [
    {"type": "progress", "message": "Agent online", "event_id": "evt-1"},
    {"type": "question", "message": "Which files?", "event_id": "evt-2"}
  ],
  "pending_questions": [...]
}

# Respond to a question
spawnie session-respond session-abc123 evt-2 "All Python files"
```

### Python API (Background Mode)

```python
from spawnie import spawn_shell, EventType

session = spawn_shell(
    task="Analyze this codebase and propose a refactor plan",
    model="claude-sonnet",
    working_dir=Path("./my-project"),
)

# Event loop - handle agent communication
for event in session.events():
    if event.type == EventType.QUESTION:
        print(f"Agent asks: {event.message}")
        answer = input("Your answer: ")
        session.respond(event.event_id, answer)

    elif event.type == EventType.PROGRESS:
        print(f"Progress: {event.message}")

    elif event.type == EventType.DONE:
        print(f"Done! Result: {event.data.get('result')}")
        break

    elif event.type == EventType.ERROR:
        print(f"Error: {event.message}")
        break
```

### Inside the Session (Agent Side)

The agent running in the shell uses CLI commands to communicate:

```bash
# Ask the orchestrator a question (blocks until answered)
answer=$(spawnie ask "Should I include the legacy modules?")

# Report progress (non-blocking)
spawnie progress "Analyzing module 3 of 10" --percent 30

# Spawn a "dark" subtask (runs in background, no shell)
result=$(spawnie run "Summarize this file" -m claude-haiku --dark)

# Signal completion
spawnie done --result "./output/plan.md" --message "Refactor plan complete"
```

### Session Management

```bash
# List active sessions
spawnie sessions

# List all sessions including ended
spawnie sessions --all --json

# Kill a session
spawnie session-kill <session-id>

# Clean up old sessions (older than 24 hours)
spawnie sessions --cleanup --max-age 24
```

### Event Types

| Event | Description | Blocks? |
|-------|-------------|---------|
| `question` | Agent needs input to proceed | Yes (waits for response) |
| `progress` | Informational status update | No |
| `done` | Task complete, session ends | N/A |
| `error` | Something went wrong | N/A |

## Real-time Monitoring

### TUI Monitor (Recommended)

```bash
spawnie monitor
```

Opens a terminal UI showing:
- Active workflows with step progress (hierarchical task display)
- Running tasks with parent/child relationships
- Recent alerts
- Live stats: completed/failed today, success rate, total runtime, model usage

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

### Provider Availability

```python
from spawnie import detect_cli, list_models

# Check specific CLI
status = detect_cli("claude")
print(status.installed)  # True/False
print(status.version)    # Version string if installed

# List all models with availability
for m in list_models():
    print(f"{m['name']}: available={m['available']}, route={m['route']}")
```

If no provider is available for a model, `run()` raises `ValueError` with available alternatives:

```python
try:
    result = run("prompt", model="claude-sonnet")
except ValueError as e:
    print(e)  # "No available route for model 'claude-sonnet'. Available models: ['test-mock']"
```

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
  },
  "preferences": {
    "prefer_cli": true
  }
}
```

### Built-in Constants

These values are defined in `spawnie.config`:

| Constant | Value | Description |
|----------|-------|-------------|
| `CLAUDE_TIMEOUT` | 600s | Claude CLI execution timeout (10 min) |
| `COPILOT_TIMEOUT` | 300s | Copilot CLI execution timeout (5 min) |
| `DEFAULT_TIMEOUT` | 300s | Default for other providers |
| `MAX_INLINE_PROMPT_LENGTH` | 7000 | Prompts longer than this use stdin |
| `AVAILABILITY_CACHE_TTL` | 60s | How long provider availability is cached |

Provider availability is checked and cached. If you install a CLI mid-session, it will be detected within 60 seconds.

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
spawnie run "prompt" -m MODEL -q extra-clean # With quality level

# Workflows
spawnie workflow FILE -i key=value      # Execute workflow
spawnie guidance                        # Show workflow JSON guide

# Workflows Library
spawnie workflows list                  # List library workflows
spawnie workflows show NAME             # Show workflow definition
spawnie workflows run NAME -i key=val   # Run from library

# Monitoring
spawnie monitor               # TUI monitor (recommended)
spawnie status                # CLI status summary
spawnie status --watch        # CLI watch mode
spawnie status WORKFLOW_ID    # Specific workflow
spawnie kill TARGET_ID        # Kill workflow or task

# Shell Sessions
spawnie shell "task" -m MODEL -n          # Interactive (new window)
spawnie shell "task" -m MODEL -b          # Background (programmatic)
spawnie shell "task" -m MODEL -b --json   # Background with JSON output
spawnie session-events SESSION_ID --json  # Poll events from background session
spawnie session-respond SESSION_ID EVT_ID "answer"  # Respond to question
spawnie sessions                          # List active sessions
spawnie sessions --all                    # List all sessions
spawnie session-kill SESSION_ID           # Kill a session
spawnie sessions --cleanup --max-age 24   # Clean up old sessions

# Diagnostics
spawnie detect claude         # Check Claude CLI availability
spawnie detect copilot        # Check Copilot CLI availability
```

## Workflows Library

Store reusable workflows in `~/.spawnie/workflows/`:

```bash
# List available workflows
spawnie workflows list

# Show workflow definition
spawnie workflows show my-workflow

# Run a workflow from the library
spawnie workflows run my-workflow -i param=value
```

Workflows are JSON files. See the `Workflow JSON Format` section above for the schema.

## Quality Levels

Control output quality with the `-q` flag:

```bash
# Normal - single pass (default, fastest)
spawnie run "prompt" -m claude-sonnet -q normal

# Extra-clean - self-review pass (catches errors)
spawnie run "prompt" -m claude-sonnet -q extra-clean

# Hypertask - dual expert review (highest quality)
spawnie run "prompt" -m claude-sonnet -q hypertask
```

| Level | Passes | Use Case |
|-------|--------|----------|
| `normal` | 1 | Quick tasks, exploration |
| `extra-clean` | 2 | Production code, important outputs |
| `hypertask` | 3+ | Critical decisions, complex analysis |

## File Structure

```
~/.spawnie/
├── config.json       # Model registry and provider config
├── tracker.json      # Real-time state (workflows, tasks, alerts)
├── workflows/        # Reusable workflow library
│   └── my-workflow.json
└── history/          # Archived completed workflows
    └── 2024-01-15.jsonl
```

## License

All Rights Reserved. See [LICENSE](LICENSE) for details.

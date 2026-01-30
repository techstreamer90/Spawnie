"""CLI commands for Spawnie."""

import argparse
import json
import sys
import time
from pathlib import Path

from . import (
    run,
    list_models,
    setup_registry,
    detect_cli,
    get_status,
    get_result,
    get_registry,
    SpawnieConfig,
    run_daemon,
    # Workflow API
    execute,
    workflow_status,
    kill,
    get_workflow_guidance,
    list_workflows,
    list_tasks,
    get_tracker,
)
from .api import spawn


def cmd_setup(args):
    """Initialize Spawnie configuration."""
    config_path = setup_registry()
    print(f"Created config at: {config_path}")
    print()
    print("Default models configured:")
    for model in list_models():
        status = "available" if model["available"] else "not available"
        route = model["route"] or "no route"
        print(f"  {model['name']}: {status} ({route})")
    return 0


def cmd_models(args):
    """List available models."""
    models = list_models()

    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    print("Available models:")
    print()
    for model in models:
        status = "OK" if model["available"] else "--"
        route = model["route"] or "no route"
        routes = ", ".join(model["routes"])
        print(f"  [{status}] {model['name']}")
        print(f"       {model['description']}")
        print(f"       Routes: {routes}")
        if model["available"]:
            print(f"       Using: {route}")
        print()

    return 0


def cmd_run(args):
    """Run a prompt."""
    result = run(
        args.prompt,
        model=args.model,
        mode=args.mode,
        timeout=args.timeout,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    if args.mode == "async":
        # result is task_id
        print(f"Task submitted: {result}")
        print("Use 'spawnie status <task-id>' to check status")
        return 0

    # result is Result object
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        if result.output:
            print()
            print("Output:")
            print(result.output)
        if result.error:
            print()
            print("Error:")
            print(result.error)

    return 0 if result.succeeded else 1


def cmd_workflow(args):
    """Execute a workflow."""
    # Load workflow from file
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Workflow file not found: {workflow_path}")
        return 1

    # Parse inputs
    inputs = {}
    if args.input:
        for inp in args.input:
            if "=" in inp:
                key, value = inp.split("=", 1)
                inputs[key] = value
            else:
                print(f"Invalid input format: {inp} (expected key=value)")
                return 1

    if args.inputs_json:
        inputs.update(json.loads(args.inputs_json))

    print(f"Executing workflow: {workflow_path}")
    print(f"Customer: {args.customer}")
    print(f"Inputs: {inputs}")
    print()

    result = execute(
        workflow_path,
        inputs=inputs,
        customer=args.customer,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print()

        if result.step_results:
            print("Steps:")
            for step_name, step_result in result.step_results.items():
                status = step_result.get("status", "unknown")
                print(f"  {step_name}: {status}")

        if result.outputs:
            print()
            print("Outputs:")
            for name, value in result.outputs.items():
                preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  {name}: {preview}")

        if result.error:
            print()
            print(f"Error: {result.error}")

    return 0 if result.status == "completed" else 1


def cmd_status(args):
    """Show tracker status."""
    if args.workflow_id:
        # Specific workflow
        status = workflow_status(args.workflow_id)
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            if "error" in status:
                print(status["error"])
                return 1
            print(f"Workflow: {status['id']}")
            print(f"Name: {status['name']}")
            print(f"Customer: {status['customer']}")
            print(f"Status: {status['status']}")
            print(f"Progress: {status['progress']}")
            if status.get('current_step'):
                print(f"Current: {status['current_step']}")
            print()
            print("Steps:")
            for name, step in status.get('steps', {}).items():
                print(f"  {name}: {step['status']}")
        return 0

    # Overall status
    status = workflow_status()

    if args.json:
        print(json.dumps(status, indent=2, default=str))
        return 0

    print("=== Spawnie Status ===")
    print(f"Updated: {status['updated_at']}")
    print()
    print(f"Workflows: {status['workflows']['running']} running, {status['workflows']['queued']} queued")
    print(f"Tasks: {status['tasks']['running']} running, {status['tasks']['queued']} queued")
    print()
    print(f"Limits: {status['limits']['max_concurrent_workflows']} workflows, {status['limits']['max_concurrent_tasks']} tasks")
    print()
    print(f"Today: {status['stats'].get('completed_today', 0)} completed, {status['stats'].get('failed_today', 0)} failed")

    if status.get('alerts'):
        print()
        print("Alerts:")
        for alert in status['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")

    # Watch mode
    if args.watch:
        try:
            while True:
                time.sleep(args.interval)
                print("\033[2J\033[H", end="")  # Clear screen
                status = workflow_status()
                print("=== Spawnie Status (watching) ===")
                print(f"Updated: {status['updated_at']}")
                print()
                print(f"Workflows: {status['workflows']['running']} running, {status['workflows']['queued']} queued")
                print(f"Tasks: {status['tasks']['running']} running, {status['tasks']['queued']} queued")

                # Show active workflows
                workflows = list_workflows()
                if workflows:
                    print()
                    print("Active Workflows:")
                    for wf in workflows:
                        print(f"  {wf['id']}: {wf['name']} ({wf['status']}) - {wf['progress']}")

                # Show active tasks
                tasks = list_tasks()
                if tasks:
                    print()
                    print("Active Tasks:")
                    for t in tasks:
                        print(f"  {t['id']}: {t['model']} ({t['status']})")

                print()
                print("Press Ctrl+C to stop watching")
        except KeyboardInterrupt:
            print("\nStopped watching")

    return 0


def cmd_kill(args):
    """Kill a workflow or task."""
    try:
        result = kill(args.target_id)
        print(f"Killed {result['type']}: {result['id']}")
        return 0
    except KeyError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1


def cmd_guidance(args):
    """Show workflow guidance for agents."""
    print(get_workflow_guidance())
    return 0


def cmd_detect(args):
    """Check CLI availability."""
    status = detect_cli(args.provider)

    print(f"Provider: {status.provider}")
    print(f"Installed: {'Yes' if status.installed else 'No'}")

    if status.installed:
        if status.path:
            print(f"Path: {status.path}")
        if status.version:
            print(f"Version: {status.version}")
        if status.models:
            print(f"Models: {', '.join(status.models)}")
    else:
        if status.error:
            print(f"Error: {status.error}")

        from .detection import get_installation_instructions
        print("\nInstallation instructions:")
        print(get_installation_instructions(args.provider))

    return 0 if status.installed else 1


def cmd_daemon(args):
    """Run the daemon."""
    config = SpawnieConfig(
        provider=args.provider,
        model=args.model,
        queue_dir=Path(args.queue_dir) if args.queue_dir else Path(".spawnie/queue"),
        poll_interval=args.poll_interval,
    )

    run_daemon(config)
    return 0


def cmd_submit(args):
    """Submit a task (legacy)."""
    result_or_id = spawn(
        args.prompt,
        provider=args.provider,
        model=args.model,
        wait=args.wait,
        timeout=args.timeout,
    )

    if args.wait:
        result = result_or_id
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print()
        if result.output:
            print("Output:")
            print(result.output)
        if result.error:
            print("Error:")
            print(result.error)
        return 0 if result.succeeded else 1
    else:
        print(f"Task submitted: {result_or_id}")
        print("Use 'spawnie status <task-id>' to check status")
        return 0


def cmd_task_status(args):
    """Check task status (legacy)."""
    status = get_status(args.task_id)

    if status is None:
        print(f"Task not found: {args.task_id}")
        return 1

    print(f"Task: {args.task_id}")
    print(f"Status: {status}")

    if status in ("completed", "failed"):
        result = get_result(args.task_id)
        if result:
            print(f"Duration: {result.duration_seconds:.2f}s")
            if result.output:
                print("\nOutput:")
                print(result.output[:500])
                if len(result.output) > 500:
                    print("... (truncated)")
            if result.error:
                print("\nError:")
                print(result.error)

    return 0


def cmd_monitor(args):
    """Launch the TUI monitor."""
    try:
        from .monitor import run_monitor
    except ImportError:
        print("Monitor requires 'textual' package.")
        print("Install with: pip install spawnie[monitor]")
        print("         or: pip install textual")
        return 1

    run_monitor()
    return 0


def cmd_config(args):
    """Show or edit configuration."""
    registry = get_registry()

    if args.show:
        print(f"Config path: {registry.config_path}")
        print()
        if registry.config_path.exists():
            with open(registry.config_path, "r") as f:
                print(f.read())
        else:
            print("(no config file - using defaults)")
        return 0

    if args.add_model:
        name, provider = args.add_model.split(":", 1) if ":" in args.add_model else (args.add_model, "claude-cli")
        registry.add_model(name, [{"provider": provider, "priority": 1}])
        registry.save()
        print(f"Added model: {name} via {provider}")
        return 0

    # Default: show summary
    print(f"Config: {registry.config_path}")
    print(f"Models: {len(registry.models)}")
    print(f"Providers: {len(registry.providers)}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="spawnie",
        description="Spawnie - Model router and workflow orchestrator for CLI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # setup command
    subparsers.add_parser("setup", help="Initialize Spawnie configuration")

    # models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a single prompt")
    run_parser.add_argument("prompt", help="The prompt to run")
    run_parser.add_argument("-m", "--model", default="claude-sonnet", help="Model to use")
    run_parser.add_argument("--mode", choices=["blocking", "async", "output"], default="blocking")
    run_parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    run_parser.add_argument("--output-dir", help="Output directory (for mode=output)")
    run_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # workflow command
    wf_parser = subparsers.add_parser("workflow", help="Execute a workflow")
    wf_parser.add_argument("workflow", help="Path to workflow JSON file")
    wf_parser.add_argument("-i", "--input", action="append", help="Input as key=value")
    wf_parser.add_argument("--inputs-json", help="Inputs as JSON string")
    wf_parser.add_argument("-c", "--customer", default="cli", help="Customer identifier")
    wf_parser.add_argument("--timeout", type=int, help="Workflow timeout")
    wf_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # status command
    status_parser = subparsers.add_parser("status", help="Show tracker status")
    status_parser.add_argument("workflow_id", nargs="?", help="Specific workflow ID")
    status_parser.add_argument("--watch", "-w", action="store_true", help="Watch mode")
    status_parser.add_argument("--interval", type=float, default=2.0, help="Watch interval")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # kill command
    kill_parser = subparsers.add_parser("kill", help="Kill a workflow or task")
    kill_parser.add_argument("target_id", help="Workflow (wf-xxx) or task (task-xxx) ID")

    # guidance command
    subparsers.add_parser("guidance", help="Show workflow guidance for agents")

    # monitor command
    subparsers.add_parser("monitor", help="Launch real-time TUI monitor")

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Check CLI availability")
    detect_parser.add_argument("provider", choices=["claude", "copilot", "mock"])

    # config command
    config_parser = subparsers.add_parser("config", help="Show or edit configuration")
    config_parser.add_argument("--show", action="store_true", help="Show full config")
    config_parser.add_argument("--add-model", help="Add model (name:provider)")

    # daemon command (legacy)
    daemon_parser = subparsers.add_parser("daemon", help="Run the task daemon")
    daemon_parser.add_argument("--provider", choices=["claude", "copilot", "mock"], default="claude")
    daemon_parser.add_argument("--model", help="Model to use")
    daemon_parser.add_argument("--queue-dir", help="Queue directory")
    daemon_parser.add_argument("--poll-interval", type=float, default=0.5)

    # submit command (legacy)
    submit_parser = subparsers.add_parser("submit", help="Submit a task (legacy)")
    submit_parser.add_argument("prompt", help="The prompt to submit")
    submit_parser.add_argument("--provider", choices=["claude", "copilot", "mock"], default="claude")
    submit_parser.add_argument("--model", help="Model to use")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for result")
    submit_parser.add_argument("--timeout", type=int, default=300)

    # task-status command (legacy)
    task_status_parser = subparsers.add_parser("task-status", help="Check task status (legacy)")
    task_status_parser.add_argument("task_id", help="The task ID to check")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "setup": cmd_setup,
        "models": cmd_models,
        "run": cmd_run,
        "workflow": cmd_workflow,
        "status": cmd_status,
        "kill": cmd_kill,
        "guidance": cmd_guidance,
        "monitor": cmd_monitor,
        "detect": cmd_detect,
        "config": cmd_config,
        "daemon": cmd_daemon,
        "submit": cmd_submit,
        "task-status": cmd_task_status,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

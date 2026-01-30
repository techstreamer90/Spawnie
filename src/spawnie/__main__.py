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
from .session import (
    ShellSession,
    ask_question,
    report_progress,
    signal_done,
    signal_error,
    get_current_session,
    list_sessions,
    get_session,
    cleanup_ended_sessions,
    EventType,
)


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
        quality=args.quality,
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


# =============================================================================
# Shell Session Commands (for agent communication)
# =============================================================================

def cmd_ask(args):
    """Ask a question and wait for response (used inside a session)."""
    session_info = get_current_session()
    if not session_info:
        print("Error: Not running inside a Spawnie session", file=sys.stderr)
        print("This command should be used by an agent running in a shell session.", file=sys.stderr)
        return 1

    try:
        answer = ask_question(args.question, timeout=args.timeout)
        print(answer)
        return 0
    except TimeoutError:
        print(f"Error: No response received within {args.timeout}s", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_progress(args):
    """Report progress (used inside a session)."""
    session_info = get_current_session()
    if not session_info:
        print("Error: Not running inside a Spawnie session", file=sys.stderr)
        return 1

    try:
        data = {}
        if args.percent is not None:
            data["percent"] = args.percent
        if args.step:
            data["step"] = args.step

        report_progress(args.message, data=data if data else None)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_done(args):
    """Signal completion (used inside a session)."""
    session_info = get_current_session()
    if not session_info:
        print("Error: Not running inside a Spawnie session", file=sys.stderr)
        return 1

    try:
        signal_done(result=args.result, message=args.message)
        print("Session complete.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_shell(args):
    """Start an interactive shell session."""
    session = ShellSession(
        working_dir=Path(args.working_dir) if args.working_dir else None,
    )

    print(f"Starting shell session: {session.session_id}")
    print(f"Working directory: {session.working_dir}")
    print(f"Model: {args.model}")
    print()

    session.start(
        task=args.task,
        model=args.model,
        provider=args.provider,
    )

    print("Session started. Listening for events...")
    print("Press Ctrl+C to stop.")
    print()

    try:
        for event in session.events(timeout=args.timeout):
            timestamp = event.timestamp.strftime("%H:%M:%S")

            if event.type == EventType.QUESTION:
                print(f"[{timestamp}] QUESTION: {event.message}")
                if args.interactive:
                    answer = input("Your answer: ")
                    session.respond(event.event_id, answer)
                else:
                    print("  (Non-interactive mode - no response sent)")
                    print(f"  Event ID: {event.event_id}")

            elif event.type == EventType.PROGRESS:
                print(f"[{timestamp}] PROGRESS: {event.message}")
                if event.data:
                    print(f"  Data: {event.data}")

            elif event.type == EventType.DONE:
                print(f"[{timestamp}] DONE: {event.message}")
                if event.data.get("result"):
                    print(f"  Result: {event.data['result']}")
                break

            elif event.type == EventType.ERROR:
                print(f"[{timestamp}] ERROR: {event.message}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Killing session...")
        session.kill()
    except TimeoutError as e:
        print(f"\nTimeout: {e}")
        session.kill()

    print(f"\nSession ended. Status: {session.status}")
    return 0


def cmd_sessions(args):
    """List and manage shell sessions."""
    if args.cleanup:
        cleaned = cleanup_ended_sessions(max_age_hours=args.max_age)
        print(f"Cleaned up {cleaned} old sessions")
        return 0

    sessions = list_sessions(include_ended=args.all)

    if not sessions:
        print("No sessions found.")
        return 0

    if args.json:
        print(json.dumps([s.to_dict() for s in sessions], indent=2, default=str))
        return 0

    print("Shell Sessions:")
    print()
    for s in sessions:
        status_icon = {
            "starting": "...",
            "running": ">>>",
            "done": "[OK]",
            "error": "[!!]",
            "killed": "[X]",
        }.get(s.status, "???")

        print(f"  {status_icon} {s.session_id}")
        print(f"       Status: {s.status}")
        print(f"       Started: {s.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if s.ended_at:
            print(f"       Ended: {s.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if s.pid:
            print(f"       PID: {s.pid}")
        if s.result:
            print(f"       Result: {s.result[:50]}...")
        if s.error:
            print(f"       Error: {s.error[:50]}...")
        print()

    return 0


def cmd_session_kill(args):
    """Kill a specific session."""
    session = get_session(args.session_id)
    if not session:
        print(f"Session not found: {args.session_id}")
        return 1

    session.kill()
    print(f"Killed session: {args.session_id}")
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
    run_parser.add_argument(
        "-q", "--quality",
        choices=["normal", "extra-clean", "hypertask"],
        default="normal",
        help="Quality level: normal (no review), extra-clean (self-review), hypertask (dual review)"
    )

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

    # ==========================================================================
    # Shell Session Commands
    # ==========================================================================

    # shell command - start interactive session
    shell_parser = subparsers.add_parser("shell", help="Start an interactive shell session")
    shell_parser.add_argument("task", help="The task/playbook for the agent")
    shell_parser.add_argument("-m", "--model", default="claude-sonnet", help="Model to use")
    shell_parser.add_argument("--provider", choices=["claude", "copilot"], help="Force specific provider")
    shell_parser.add_argument("-d", "--working-dir", help="Working directory for the session")
    shell_parser.add_argument("--timeout", type=int, default=3600, help="Session timeout in seconds")
    shell_parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode (answer questions)")

    # sessions command - list sessions
    sessions_parser = subparsers.add_parser("sessions", help="List shell sessions")
    sessions_parser.add_argument("--all", "-a", action="store_true", help="Include ended sessions")
    sessions_parser.add_argument("--json", action="store_true", help="Output as JSON")
    sessions_parser.add_argument("--cleanup", action="store_true", help="Clean up old sessions")
    sessions_parser.add_argument("--max-age", type=int, default=24, help="Max age in hours for cleanup")

    # session-kill command - kill a session
    session_kill_parser = subparsers.add_parser("session-kill", help="Kill a shell session")
    session_kill_parser.add_argument("session_id", help="Session ID to kill")

    # ask command - ask question (used inside session)
    ask_parser = subparsers.add_parser("ask", help="Ask orchestrator a question (inside session)")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--timeout", type=int, default=300, help="Timeout waiting for response")

    # progress command - report progress (used inside session)
    progress_parser = subparsers.add_parser("progress", help="Report progress (inside session)")
    progress_parser.add_argument("message", help="Progress message")
    progress_parser.add_argument("--percent", type=int, help="Completion percentage (0-100)")
    progress_parser.add_argument("--step", help="Current step name")

    # done command - signal completion (used inside session)
    done_parser = subparsers.add_parser("done", help="Signal task completion (inside session)")
    done_parser.add_argument("--result", help="Path to result file")
    done_parser.add_argument("--message", help="Completion message")

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
        # Shell session commands
        "shell": cmd_shell,
        "sessions": cmd_sessions,
        "session-kill": cmd_session_kill,
        "ask": cmd_ask,
        "progress": cmd_progress,
        "done": cmd_done,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

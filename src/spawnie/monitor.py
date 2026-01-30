"""
Spawnie Monitor - Real-time TUI for workflow visualization.

A terminal interface that shows:
- Active workflows and their steps
- Running tasks
- Status updates in real-time
- Basic interaction (kill, refresh)
"""

from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Header, Footer, Static, Rule, Label
from textual.reactive import reactive
from textual.timer import Timer

from .tracker import get_tracker, WorkflowState, TaskState


# Status symbols
STATUS_SYMBOLS = {
    "queued": "⏳",
    "pending": "⏳",
    "running": "🔄",
    "completed": "🟢",
    "failed": "🔴",
    "timeout": "⏰",
    "killed": "💀",
    "skipped": "⏭️",
}


def status_symbol(status: str) -> str:
    """Get emoji symbol for status."""
    return STATUS_SYMBOLS.get(status, "❓")


def render_workflows(workflows: list[WorkflowState]) -> str:
    """Render workflows to rich markup string."""
    if not workflows:
        return "[dim]No active workflows[/dim]"

    lines = []
    for wf in workflows:
        symbol = status_symbol(wf.status)

        # Header line
        lines.append(f"{symbol} [bold]{wf.name}[/bold] ({wf.id})")
        lines.append(f"   Customer: {wf.customer}  |  Status: {wf.status}  |  {wf.progress}")

        # Steps
        lines.append("   [bold]Steps:[/bold]")
        for name, step in wf.steps.items():
            step_sym = status_symbol(step.status)
            duration = f"{step.duration:.1f}s" if step.duration else ""
            task_info = f"({step.task_id})" if step.task_id and step.status == "running" else ""
            error_info = f" [red]Error: {step.error[:30]}...[/red]" if step.error else ""

            lines.append(f"      {step_sym} {name:20} {step.status:12} {duration:8} {task_info}{error_info}")

        # Error if any
        if wf.error:
            lines.append(f"   [red]Error: {wf.error}[/red]")

        lines.append("─" * 60)

    return "\n".join(lines)


def render_tasks(tasks: list[TaskState]) -> str:
    """Render tasks to rich markup string."""
    if not tasks:
        return "[dim]No active tasks[/dim]"

    lines = []
    for task in tasks:
        symbol = status_symbol(task.status)
        # Show description if available, otherwise fall back to ID
        if task.description:
            desc = task.description[:50] + "..." if len(task.description) > 50 else task.description
            lines.append(f"   {symbol} [bold]{task.model}[/bold] - {desc}")
            lines.append(f"      ID: {task.id}  Status: {task.status}")
        else:
            wf_info = f"[{task.workflow_id}:{task.step}]" if task.workflow_id else ""
            lines.append(f"   {symbol} {task.id:20} {task.model:15} {task.status:10} {wf_info}")

    return "\n".join(lines)


def render_stats(status: dict) -> str:
    """Render stats to rich markup string."""
    wf = status.get("workflows", {})
    tasks = status.get("tasks", {})
    stats = status.get("stats", {})

    return (
        f"Workflows: {wf.get('running', 0)} running, {wf.get('queued', 0)} queued  |  "
        f"Tasks: {tasks.get('running', 0)} running, {tasks.get('queued', 0)} queued  |  "
        f"Today: {stats.get('completed_today', 0)} completed, {stats.get('failed_today', 0)} failed"
    )


def render_alerts(alerts: list[dict]) -> str:
    """Render alerts to rich markup string."""
    if not alerts:
        return "[dim]No recent alerts[/dim]"

    lines = []
    for alert in alerts[-5:]:
        level = alert.get("level", "info")
        msg = alert.get("message", "")
        time = alert.get("at", "")[:19]

        if level == "error":
            lines.append(f"   🔴 [{time}] {msg}")
        elif level == "warn":
            lines.append(f"   🟡 [{time}] {msg}")
        else:
            lines.append(f"   ⚪ [{time}] {msg}")

    return "\n".join(lines)


class SpawnieMonitor(App):
    """Main Spawnie Monitor application."""

    CSS = """
    Screen {
        background: $surface;
    }

    .section-label {
        color: $primary;
        margin-top: 1;
        text-style: bold;
    }

    #stats-panel {
        background: $primary-background;
        padding: 1;
    }

    #workflows-panel {
        padding: 1;
    }

    #tasks-panel {
        padding: 1;
    }

    #alerts-panel {
        padding: 1;
    }

    #main-container {
        height: 100%;
        padding: 1;
    }

    #workflows-container {
        height: auto;
        max-height: 60%;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("k", "kill", "Kill selected"),
        ("s", "screenshot", "Screenshot"),
        ("+", "faster", "Faster refresh"),
        ("-", "slower", "Slower refresh"),
    ]

    TITLE = "Spawnie Monitor"
    SUB_TITLE = "Real-time workflow visualization"

    refresh_interval = reactive(1.0)
    last_update = reactive("")
    refresh_count = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracker = get_tracker()
        self._timer: Timer | None = None

    def watch_refresh_interval(self, value: float) -> None:
        """Handle refresh interval changes - recreate timer with new interval."""
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(value, self.refresh_data)

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            # Stats at top
            yield Static(id="stats-panel")

            yield Rule()

            # Workflows section
            yield Label("Active Workflows", classes="section-label")
            with ScrollableContainer(id="workflows-container"):
                yield Static(id="workflows-panel")

            yield Rule()

            # Tasks section
            yield Label("Active Tasks", classes="section-label")
            yield Static(id="tasks-panel")

            yield Rule()

            # Alerts section
            yield Label("Recent Alerts", classes="section-label")
            yield Static(id="alerts-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self._timer = self.set_interval(self.refresh_interval, self.refresh_data)
        self.refresh_data()

    def on_unmount(self) -> None:
        """Cleanup when app closes."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def refresh_data(self) -> None:
        """Refresh all data from tracker - uses update() to avoid flicker."""
        try:
            # Reload state from disk (other processes may have updated it)
            self.tracker.reload()

            status = self.tracker.get_status()
            workflows = self.tracker.list_workflows()
            tasks = self.tracker.list_tasks()
            alerts = status.get("alerts", [])

            # Update panels in-place using update() - no flicker!
            self.query_one("#stats-panel", Static).update(render_stats(status))
            self.query_one("#workflows-panel", Static).update(render_workflows(workflows))
            self.query_one("#tasks-panel", Static).update(render_tasks(tasks))
            self.query_one("#alerts-panel", Static).update(render_alerts(alerts))

            self.refresh_count += 1
            self.last_update = datetime.now().strftime("%H:%M:%S")
            self.sub_title = f"Last update: {self.last_update} (#{self.refresh_count})"

        except Exception as e:
            self.notify(f"Error refreshing: {e}", severity="error")

    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_data()
        self.notify("Refreshed")

    def action_kill(self) -> None:
        """Kill the first running workflow (simple implementation)."""
        workflows = self.tracker.list_workflows()
        running = [w for w in workflows if w.status == "running"]

        if running:
            wf = running[0]
            try:
                self.tracker.kill_workflow(wf.id, "Killed from monitor")
                self.notify(f"Killed workflow {wf.id}")
                self.refresh_data()
            except Exception as e:
                self.notify(f"Error killing: {e}", severity="error")
        else:
            self.notify("No running workflows to kill", severity="warning")

    def action_screenshot(self) -> None:
        """Take a screenshot of the monitor."""
        path = self.save_screenshot(path=".")
        self.notify(f"Screenshot saved: {path}")

    def action_faster(self) -> None:
        """Increase refresh rate (decrease interval)."""
        self.refresh_interval = max(0.5, self.refresh_interval - 0.5)
        self.notify(f"Refresh interval: {self.refresh_interval}s")

    def action_slower(self) -> None:
        """Decrease refresh rate (increase interval)."""
        self.refresh_interval = min(10.0, self.refresh_interval + 0.5)
        self.notify(f"Refresh interval: {self.refresh_interval}s")


def run_monitor():
    """Run the Spawnie monitor."""
    app = SpawnieMonitor()
    app.run()


if __name__ == "__main__":
    run_monitor()

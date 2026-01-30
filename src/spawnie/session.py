"""Shell session management for interactive agents.

A ShellSession provides bidirectional communication between an orchestrator
and a spawned agent running in a shell with file system access.

Communication Protocol:
- Session directory: ~/.spawnie/sessions/<session-id>/
- events.jsonl: Agent writes events (question, progress, done, error)
- responses.jsonl: Orchestrator writes responses to questions
- status.json: Current session state

Event Types:
- question: Agent needs input (blocks until response)
- progress: Informational update
- done: Task complete, session ends
- error: Something went wrong, session ends
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Generator, Any

from .config import DEFAULT_TIMEOUT


class EventType(str, Enum):
    """Types of events an agent can emit."""
    QUESTION = "question"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"


@dataclass
class SessionEvent:
    """An event emitted by an agent."""
    type: EventType
    message: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEvent":
        return cls(
            type=EventType(data["type"]),
            message=data["message"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data["event_id"],
        )


@dataclass
class SessionResponse:
    """A response from the orchestrator to an agent question."""
    event_id: str  # ID of the question being answered
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionResponse":
        return cls(
            event_id=data["event_id"],
            answer=data["answer"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class SessionStatus:
    """Current status of a session."""
    session_id: str
    status: str  # "starting", "running", "done", "error", "killed"
    pid: int | None = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionStatus":
        return cls(
            session_id=data["session_id"],
            status=data["status"],
            pid=data.get("pid"),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            result=data.get("result"),
            error=data.get("error"),
        )


def get_sessions_dir() -> Path:
    """Get the sessions directory."""
    return Path.home() / ".spawnie" / "sessions"


def get_session_dir(session_id: str) -> Path:
    """Get the directory for a specific session."""
    return get_sessions_dir() / session_id


class ShellSession:
    """
    Manages an interactive shell session with an agent.

    The session spawns a shell, sets up communication channels,
    and provides methods to interact with the running agent.
    """

    def __init__(
        self,
        session_id: str | None = None,
        working_dir: Path | None = None,
        shell: str | None = None,
    ):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.shell = shell or self._detect_shell()

        self.session_dir = get_session_dir(self.session_id)
        self.events_file = self.session_dir / "events.jsonl"
        self.responses_file = self.session_dir / "responses.jsonl"
        self.status_file = self.session_dir / "status.json"

        self._process: subprocess.Popen | None = None
        self._events_read_position = 0
        self._status = SessionStatus(session_id=self.session_id, status="starting")

    def _detect_shell(self) -> str:
        """Detect the appropriate shell for the platform."""
        if sys.platform == "win32":
            # Prefer PowerShell, fall back to cmd
            if os.environ.get("COMSPEC"):
                return os.environ["COMSPEC"]
            return "powershell.exe"
        else:
            # Use user's shell or default to bash
            return os.environ.get("SHELL", "/bin/bash")

    def _setup_session_dir(self) -> None:
        """Create the session directory structure."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty files
        self.events_file.touch()
        self.responses_file.touch()

        # Write initial status
        self._save_status()

    def _save_status(self) -> None:
        """Save current status to file."""
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(self._status.to_dict(), f, indent=2)

    def _load_status(self) -> SessionStatus:
        """Load status from file."""
        if self.status_file.exists():
            with open(self.status_file, "r", encoding="utf-8") as f:
                return SessionStatus.from_dict(json.load(f))
        return self._status

    def start(
        self,
        task: str,
        model: str = "claude-sonnet",
        provider: str | None = None,
    ) -> None:
        """
        Start the shell session with an initial task.

        Args:
            task: The task/playbook for the agent
            model: Model to use (e.g., "claude-sonnet")
            provider: Specific provider ("claude", "copilot") or None for auto
        """
        self._setup_session_dir()

        # Build environment with session info
        env = os.environ.copy()
        env["SPAWNIE_SESSION_ID"] = self.session_id
        env["SPAWNIE_SESSION_DIR"] = str(self.session_dir)
        env["SPAWNIE_MODEL"] = model
        if provider:
            env["SPAWNIE_PROVIDER"] = provider

        # Build the command to run in the shell
        # The agent will be started with instructions on how to communicate
        agent_instructions = self._build_agent_instructions(task, model)

        # Determine the CLI command based on provider
        if provider == "copilot" or (not provider and "copilot" in model.lower()):
            cli_cmd = self._build_copilot_command(agent_instructions)
        else:
            cli_cmd = self._build_claude_command(agent_instructions, model)

        # Start the shell with the command
        if sys.platform == "win32":
            # Windows: use cmd /c or powershell -Command
            if "powershell" in self.shell.lower():
                shell_cmd = [self.shell, "-NoExit", "-Command", cli_cmd]
            else:
                shell_cmd = [self.shell, "/K", cli_cmd]
        else:
            # Unix: use shell -c
            shell_cmd = [self.shell, "-c", cli_cmd]

        self._process = subprocess.Popen(
            shell_cmd,
            cwd=self.working_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        self._status.status = "running"
        self._status.pid = self._process.pid
        self._save_status()

    def _build_agent_instructions(self, task: str, model: str) -> str:
        """Build the instructions prompt for the agent."""
        return f"""{task}

---
IMPORTANT - Communication Protocol:

You are running in a Spawnie shell session. Use these commands to communicate:

1. To ASK a question and wait for an answer:
   spawnie ask "Your question here"
   The answer will be returned as stdout.

2. To report PROGRESS (non-blocking):
   spawnie progress "Status update message"

3. When you are DONE:
   spawnie done --result "path/to/result/file.md"
   Or: spawnie done --message "Summary of what was accomplished"

4. To spawn a DARK (background) subtask:
   spawnie run "subtask prompt" -m {model} --dark

Your working directory is: {self.working_dir}
Session ID: {self.session_id}

Begin your work now.
"""

    def _build_claude_command(self, instructions: str, model: str) -> str:
        """Build the Claude CLI command."""
        # Escape for shell
        escaped = instructions.replace("'", "'\\''")
        model_flag = ""
        if model:
            # Extract sub-model (e.g., "claude-sonnet" -> "sonnet")
            sub_model = model.replace("claude-", "") if model.startswith("claude-") else model
            model_flag = f"--model {sub_model}"

        return f"claude --print {model_flag} '{escaped}'"

    def _build_copilot_command(self, instructions: str) -> str:
        """Build the Copilot CLI command."""
        escaped = instructions.replace("'", "'\\''")
        return f"gh copilot suggest -t shell '{escaped}'"

    def events(
        self,
        poll_interval: float = 0.5,
        timeout: int | None = None,
    ) -> Generator[SessionEvent, None, None]:
        """
        Generator that yields events from the agent.

        Args:
            poll_interval: How often to check for new events (seconds)
            timeout: Maximum time to wait for events (None = no timeout)

        Yields:
            SessionEvent objects as they are emitted by the agent
        """
        start_time = time.time()

        while True:
            # Check for timeout
            if timeout and (time.time() - start_time) > timeout:
                self._status.status = "error"
                self._status.error = "Session timed out"
                self._save_status()
                raise TimeoutError(f"Session {self.session_id} timed out after {timeout}s")

            # Check if process has ended
            if self._process and self._process.poll() is not None:
                # Process ended - check for any final events
                for event in self._read_new_events():
                    yield event

                # Update status based on how it ended
                if self._status.status == "running":
                    exit_code = self._process.returncode
                    if exit_code == 0:
                        self._status.status = "done"
                    else:
                        self._status.status = "error"
                        self._status.error = f"Process exited with code {exit_code}"
                    self._status.ended_at = datetime.now()
                    self._save_status()
                return

            # Read new events
            for event in self._read_new_events():
                yield event

                # If done or error, stop
                if event.type in (EventType.DONE, EventType.ERROR):
                    self._status.status = event.type.value
                    if event.type == EventType.DONE:
                        self._status.result = event.data.get("result") or event.message
                    else:
                        self._status.error = event.message
                    self._status.ended_at = datetime.now()
                    self._save_status()
                    return

            time.sleep(poll_interval)

    def _read_new_events(self) -> list[SessionEvent]:
        """Read any new events from the events file."""
        events = []

        if not self.events_file.exists():
            return events

        with open(self.events_file, "r", encoding="utf-8") as f:
            f.seek(self._events_read_position)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        events.append(SessionEvent.from_dict(data))
                    except json.JSONDecodeError:
                        continue
            self._events_read_position = f.tell()

        return events

    def respond(self, event_id: str, answer: str) -> None:
        """
        Send a response to an agent's question.

        Args:
            event_id: The event_id of the question being answered
            answer: The answer text
        """
        response = SessionResponse(event_id=event_id, answer=answer)

        with open(self.responses_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(response.to_dict()) + "\n")

    def send_input(self, text: str) -> None:
        """
        Send input to the shell's stdin.

        Args:
            text: Text to send (will add newline if not present)
        """
        if self._process and self._process.stdin:
            if not text.endswith("\n"):
                text += "\n"
            self._process.stdin.write(text)
            self._process.stdin.flush()

    def kill(self) -> None:
        """Kill the session immediately."""
        if self._process:
            try:
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    os.kill(self._process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        self._status.status = "killed"
        self._status.ended_at = datetime.now()
        self._save_status()

    def is_alive(self) -> bool:
        """Check if the session is still running."""
        if not self._process:
            return False
        return self._process.poll() is None

    @property
    def status(self) -> str:
        """Get the current status string."""
        return self._load_status().status

    def cleanup(self) -> None:
        """Clean up session files."""
        self.kill()

        import shutil
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)


# =============================================================================
# Functions for use INSIDE a session (by the agent via CLI commands)
# =============================================================================

def get_current_session() -> tuple[str, Path] | None:
    """
    Get the current session info from environment.

    Returns:
        Tuple of (session_id, session_dir) or None if not in a session.
    """
    session_id = os.environ.get("SPAWNIE_SESSION_ID")
    session_dir = os.environ.get("SPAWNIE_SESSION_DIR")

    if session_id and session_dir:
        return (session_id, Path(session_dir))
    return None


def emit_event(event: SessionEvent) -> None:
    """
    Emit an event from inside a session.

    Called by CLI commands like `spawnie ask`, `spawnie progress`, etc.
    """
    session_info = get_current_session()
    if not session_info:
        raise RuntimeError("Not running inside a Spawnie session")

    _, session_dir = session_info
    events_file = session_dir / "events.jsonl"

    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict()) + "\n")


def wait_for_response(event_id: str, timeout: int = 300, poll_interval: float = 0.5) -> str:
    """
    Wait for a response to a question.

    Args:
        event_id: The event_id of the question
        timeout: Maximum time to wait in seconds
        poll_interval: How often to check for response

    Returns:
        The answer text

    Raises:
        TimeoutError: If no response within timeout
    """
    session_info = get_current_session()
    if not session_info:
        raise RuntimeError("Not running inside a Spawnie session")

    _, session_dir = session_info
    responses_file = session_dir / "responses.jsonl"

    start_time = time.time()
    seen_responses = set()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"No response received within {timeout}s")

        if responses_file.exists():
            with open(responses_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and line not in seen_responses:
                        seen_responses.add(line)
                        try:
                            data = json.loads(line)
                            if data.get("event_id") == event_id:
                                return data["answer"]
                        except json.JSONDecodeError:
                            continue

        time.sleep(poll_interval)


def ask_question(question: str, timeout: int = 300) -> str:
    """
    Ask a question and wait for the response.

    This is the implementation behind `spawnie ask`.

    Args:
        question: The question to ask
        timeout: Maximum time to wait for response

    Returns:
        The answer text
    """
    event = SessionEvent(
        type=EventType.QUESTION,
        message=question,
    )
    emit_event(event)
    return wait_for_response(event.event_id, timeout=timeout)


def report_progress(message: str, data: dict | None = None) -> None:
    """
    Report progress (non-blocking).

    This is the implementation behind `spawnie progress`.
    """
    event = SessionEvent(
        type=EventType.PROGRESS,
        message=message,
        data=data or {},
    )
    emit_event(event)


def signal_done(result: str | None = None, message: str | None = None) -> None:
    """
    Signal that the task is complete.

    This is the implementation behind `spawnie done`.
    """
    event = SessionEvent(
        type=EventType.DONE,
        message=message or "Task completed",
        data={"result": result} if result else {},
    )
    emit_event(event)


def signal_error(error: str) -> None:
    """
    Signal that an error occurred.

    This is the implementation behind `spawnie error`.
    """
    event = SessionEvent(
        type=EventType.ERROR,
        message=error,
    )
    emit_event(event)


# =============================================================================
# Session listing and management
# =============================================================================

def list_sessions(include_ended: bool = False) -> list[SessionStatus]:
    """
    List all sessions.

    Args:
        include_ended: Whether to include ended sessions

    Returns:
        List of SessionStatus objects
    """
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []

    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            status_file = session_dir / "status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        status = SessionStatus.from_dict(json.load(f))
                    if include_ended or status.status in ("starting", "running"):
                        sessions.append(status)
                except (json.JSONDecodeError, KeyError):
                    continue

    return sorted(sessions, key=lambda s: s.started_at, reverse=True)


def get_session(session_id: str) -> ShellSession | None:
    """
    Get an existing session by ID.

    Returns a ShellSession object that can be used to interact with
    a previously started session.
    """
    session_dir = get_session_dir(session_id)
    if not session_dir.exists():
        return None

    session = ShellSession(session_id=session_id)
    return session


def cleanup_ended_sessions(max_age_hours: int = 24) -> int:
    """
    Clean up old ended sessions.

    Args:
        max_age_hours: Remove sessions older than this

    Returns:
        Number of sessions cleaned up
    """
    import shutil

    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return 0

    now = datetime.now()
    cleaned = 0

    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            status_file = session_dir / "status.json"
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        status = SessionStatus.from_dict(json.load(f))

                    # Only clean up ended sessions
                    if status.status in ("done", "error", "killed"):
                        ended_at = status.ended_at or status.started_at
                        age = now - ended_at
                        if age.total_seconds() > max_age_hours * 3600:
                            shutil.rmtree(session_dir, ignore_errors=True)
                            cleaned += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    return cleaned

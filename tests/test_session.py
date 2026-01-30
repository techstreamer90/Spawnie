"""Tests for shell session functionality."""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from spawnie.session import (
    ShellSession,
    SessionEvent,
    SessionResponse,
    SessionStatus,
    EventType,
    get_sessions_dir,
    get_session_dir,
    emit_event,
    get_current_session,
    list_sessions,
    cleanup_ended_sessions,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """All expected event types exist."""
        assert EventType.QUESTION == "question"
        assert EventType.PROGRESS == "progress"
        assert EventType.DONE == "done"
        assert EventType.ERROR == "error"


class TestSessionEvent:
    """Tests for SessionEvent dataclass."""

    def test_event_creation(self):
        """SessionEvent can be created with defaults."""
        event = SessionEvent(type=EventType.QUESTION, message="Test?")
        assert event.type == EventType.QUESTION
        assert event.message == "Test?"
        assert event.data == {}
        assert event.event_id is not None

    def test_event_to_dict(self):
        """SessionEvent serializes to dict."""
        event = SessionEvent(
            type=EventType.PROGRESS,
            message="50% done",
            data={"percent": 50},
        )
        d = event.to_dict()
        assert d["type"] == "progress"
        assert d["message"] == "50% done"
        assert d["data"]["percent"] == 50
        assert "timestamp" in d
        assert "event_id" in d

    def test_event_from_dict(self):
        """SessionEvent deserializes from dict."""
        d = {
            "type": "done",
            "message": "Complete",
            "data": {"result": "output.md"},
            "timestamp": "2024-01-15T10:30:00",
            "event_id": "abc123",
        }
        event = SessionEvent.from_dict(d)
        assert event.type == EventType.DONE
        assert event.message == "Complete"
        assert event.data["result"] == "output.md"
        assert event.event_id == "abc123"


class TestSessionResponse:
    """Tests for SessionResponse dataclass."""

    def test_response_creation(self):
        """SessionResponse can be created."""
        response = SessionResponse(event_id="evt123", answer="Yes")
        assert response.event_id == "evt123"
        assert response.answer == "Yes"

    def test_response_roundtrip(self):
        """SessionResponse serializes and deserializes."""
        response = SessionResponse(event_id="evt456", answer="No")
        d = response.to_dict()
        restored = SessionResponse.from_dict(d)
        assert restored.event_id == response.event_id
        assert restored.answer == response.answer


class TestSessionStatus:
    """Tests for SessionStatus dataclass."""

    def test_status_creation(self):
        """SessionStatus can be created with defaults."""
        status = SessionStatus(session_id="sess-123", status="running")
        assert status.session_id == "sess-123"
        assert status.status == "running"
        assert status.pid is None

    def test_status_roundtrip(self):
        """SessionStatus serializes and deserializes."""
        status = SessionStatus(
            session_id="sess-456",
            status="done",
            pid=12345,
            result="output.md",
        )
        d = status.to_dict()
        restored = SessionStatus.from_dict(d)
        assert restored.session_id == status.session_id
        assert restored.status == status.status
        assert restored.pid == status.pid
        assert restored.result == status.result


class TestShellSession:
    """Tests for ShellSession class."""

    def test_session_creation(self):
        """ShellSession can be created."""
        session = ShellSession()
        assert session.session_id.startswith("session-")
        assert session.working_dir == Path.cwd()

    def test_session_with_custom_id(self):
        """ShellSession can be created with custom ID."""
        session = ShellSession(session_id="my-session-123")
        assert session.session_id == "my-session-123"

    def test_session_with_working_dir(self):
        """ShellSession respects working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = ShellSession(working_dir=Path(tmpdir))
            assert session.working_dir == Path(tmpdir)

    def test_session_dir_structure(self):
        """Session directory has expected structure."""
        session = ShellSession(session_id="test-session")
        assert session.session_dir == get_session_dir("test-session")
        assert session.events_file == session.session_dir / "events.jsonl"
        assert session.responses_file == session.session_dir / "responses.jsonl"
        assert session.status_file == session.session_dir / "status.json"


class TestGetCurrentSession:
    """Tests for get_current_session function."""

    def test_not_in_session(self):
        """Returns None when not in a session."""
        # Clear any existing env vars
        os.environ.pop("SPAWNIE_SESSION_ID", None)
        os.environ.pop("SPAWNIE_SESSION_DIR", None)
        assert get_current_session() is None

    def test_in_session(self):
        """Returns session info when env vars are set."""
        os.environ["SPAWNIE_SESSION_ID"] = "test-sess"
        os.environ["SPAWNIE_SESSION_DIR"] = "/tmp/test"
        try:
            result = get_current_session()
            assert result is not None
            session_id, session_dir = result
            assert session_id == "test-sess"
            assert session_dir == Path("/tmp/test")
        finally:
            os.environ.pop("SPAWNIE_SESSION_ID", None)
            os.environ.pop("SPAWNIE_SESSION_DIR", None)


class TestListSessions:
    """Tests for list_sessions function."""

    def test_list_empty(self):
        """Returns empty list when no sessions."""
        # This might return existing sessions, so just check it doesn't error
        sessions = list_sessions()
        assert isinstance(sessions, list)

    def test_list_with_ended(self):
        """Can include ended sessions."""
        sessions = list_sessions(include_ended=True)
        assert isinstance(sessions, list)


class TestSessionFileOperations:
    """Tests for session file operations."""

    def test_emit_event_requires_session(self):
        """emit_event raises error when not in session."""
        os.environ.pop("SPAWNIE_SESSION_ID", None)
        os.environ.pop("SPAWNIE_SESSION_DIR", None)

        event = SessionEvent(type=EventType.PROGRESS, message="Test")
        with pytest.raises(RuntimeError, match="Not running inside"):
            emit_event(event)

    def test_emit_event_writes_to_file(self):
        """emit_event writes to events file when in session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            events_file = session_dir / "events.jsonl"
            events_file.touch()

            os.environ["SPAWNIE_SESSION_ID"] = "test-emit"
            os.environ["SPAWNIE_SESSION_DIR"] = str(session_dir)

            try:
                event = SessionEvent(type=EventType.PROGRESS, message="50% done")
                emit_event(event)

                # Check file contents
                with open(events_file, "r") as f:
                    line = f.readline()
                    data = json.loads(line)
                    assert data["type"] == "progress"
                    assert data["message"] == "50% done"
            finally:
                os.environ.pop("SPAWNIE_SESSION_ID", None)
                os.environ.pop("SPAWNIE_SESSION_DIR", None)

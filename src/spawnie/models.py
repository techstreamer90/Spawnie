"""Data models for Spawnie tasks and results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import uuid


# Quality levels control how much review a task receives
QualityLevel = Literal["normal", "extra-clean", "hypertask"]

QUALITY_DESCRIPTIONS = {
    "normal": "No review - execute task and return result directly",
    "extra-clean": "Self-review - agent reviews its own output before returning",
    "hypertask": "Dual review - self-review + external reviewer for highest quality",
}

# Review prompt templates based on benchmark findings
SELF_REVIEW_PROMPT = """You just completed a task. Before finalizing, critically review your own work:

ORIGINAL TASK: {original_prompt}

YOUR RESPONSE: {output}

SELF-REVIEW CHECKLIST:
1. What did you FORGET to consider? (security? cost? edge cases? alternatives?)
2. Are there CONTRADICTIONS in your response?
3. What ASSUMPTIONS are you making that might be wrong?
4. What is your CONFIDENCE level (Low/Medium/High) and why?

SOURCE VERIFICATION (CRITICAL):
If your response makes claims about code, files, or implementations:
- You MUST read the actual source files before making claims
- Do NOT speculate about what code does - READ IT
- Do NOT estimate complexity without seeing the actual code
- Any claim about a specific file/line MUST be verified by reading it

Provide an IMPROVED response that:
- Addresses any gaps you identified
- Acknowledges trade-offs
- States your confidence level
- Is clear about limitations
- VERIFIES all code-related claims against actual source

Your improved response:"""

EXTERNAL_REVIEW_PROMPT = """You are a SKEPTICAL SENIOR ARCHITECT reviewing work from another team member.

ORIGINAL TASK: {original_prompt}

THEIR RESPONSE (after self-review): {output}

Your job is to be ADVERSARIAL but constructive:
1. Find CONTRADICTIONS or inconsistencies
2. Identify ARCHITECTURAL RISKS not addressed
3. Challenge UNJUSTIFIED decisions
4. Rate overall confidence (Low/Medium/High)

SOURCE VERIFICATION (CRITICAL):
Before critiquing, you MUST verify claims against actual code:
- If the response mentions specific files/lines, READ THEM to verify
- If complexity is estimated, CHECK the actual implementation
- If risks are claimed, CONFIRM they exist in the source
- Do NOT accept or reject claims without reading the relevant code
- Call out any unverified speculation in the original response

Be critical. Assume something is wrong. Find it. VERIFY with source code.

Your critique (max 300 words):"""

FINAL_SYNTHESIS_PROMPT = """You need to finalize your response incorporating review feedback.

ORIGINAL TASK: {original_prompt}

YOUR DRAFT (after self-review): {self_reviewed_output}

EXTERNAL REVIEWER CRITIQUE: {external_review}

Create a FINAL response that:
1. Addresses the reviewer's concerns
2. Acknowledges remaining trade-offs
3. States your confidence level with justification
4. Is honest about limitations

VERIFICATION REQUIREMENT:
- All claims about code MUST be verified by reading actual source files
- If you haven't read the code, say "UNVERIFIED" next to the claim
- Do NOT present speculation as fact
- Complexity estimates require reading the actual implementation

Your final response:"""


@dataclass
class Task:
    """A task to be executed by a CLI agent."""

    prompt: str
    provider: str
    model: str | None = None
    quality_level: QualityLevel = "normal"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "provider": self.provider,
            "model": self.model,
            "quality_level": self.quality_level,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Create Task from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            provider=data["provider"],
            model=data.get("model"),
            quality_level=data.get("quality_level", "normal"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Result:
    """Result of a completed task."""

    task_id: str
    status: str  # "completed" | "failed" | "timeout"
    output: str | None = None
    error: str | None = None
    completed_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Result":
        """Create Result from dictionary."""
        return cls(
            task_id=data["task_id"],
            status=data["status"],
            output=data.get("output"),
            error=data.get("error"),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    @property
    def succeeded(self) -> bool:
        """Check if the task completed successfully."""
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        """Check if the task failed."""
        return self.status in ("failed", "timeout")

"""Structured findings format for code quality analysis.

Provides a schema and validation for findings from quality check workflows.
Enables consistent, actionable output that can be programmatically processed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import json


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"  # Must fix - broken functionality, security issues
    HIGH = "high"          # Should fix soon - significant issues
    MEDIUM = "medium"      # Should fix - moderate issues
    LOW = "low"            # Nice to have - minor improvements


class Confidence(str, Enum):
    """Confidence levels for findings."""
    CERTAIN = "certain"      # Verified, definitely exists
    LIKELY = "likely"        # Strong evidence, probably exists
    POSSIBLE = "possible"    # Some evidence, might exist
    SPECULATIVE = "speculative"  # Educated guess, needs verification


@dataclass
class CodeReference:
    """Reference to a specific location in code."""
    file: str
    line: int | None = None
    end_line: int | None = None
    code_snippet: str | None = None  # The actual code at this location

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "code_snippet": self.code_snippet,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeReference":
        return cls(
            file=data["file"],
            line=data.get("line"),
            end_line=data.get("end_line"),
            code_snippet=data.get("code_snippet"),
        )


@dataclass
class Evidence:
    """Evidence supporting a finding."""
    files_checked: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    code_refs: list[CodeReference] = field(default_factory=list)
    reasoning: str | None = None  # Explanation of how conclusion was reached

    def to_dict(self) -> dict:
        return {
            "files_checked": self.files_checked,
            "search_queries": self.search_queries,
            "code_refs": [ref.to_dict() for ref in self.code_refs],
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            files_checked=data.get("files_checked", []),
            search_queries=data.get("search_queries", []),
            code_refs=[CodeReference.from_dict(r) for r in data.get("code_refs", [])],
            reasoning=data.get("reasoning"),
        )


@dataclass
class SuggestedFix:
    """A suggested fix for a finding."""
    file: str
    description: str
    before: str | None = None  # Code before change
    after: str | None = None   # Code after change (the fix)
    diff: str | None = None    # Unified diff format

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "description": self.description,
            "before": self.before,
            "after": self.after,
            "diff": self.diff,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SuggestedFix":
        return cls(
            file=data["file"],
            description=data["description"],
            before=data.get("before"),
            after=data.get("after"),
            diff=data.get("diff"),
        )


@dataclass
class Finding:
    """A single finding from code analysis."""
    id: str
    title: str
    description: str
    category: str  # e.g., "security", "documentation", "consistency", "performance"
    severity: Severity
    confidence: Confidence
    evidence: Evidence
    suggested_fixes: list[SuggestedFix] = field(default_factory=list)
    verified: bool = False  # Has this been verified by a second pass?
    depends_on: list[str] = field(default_factory=list)  # IDs of findings this depends on
    blocks: list[str] = field(default_factory=list)  # IDs of findings this blocks
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "evidence": self.evidence.to_dict(),
            "suggested_fixes": [fix.to_dict() for fix in self.suggested_fixes],
            "verified": self.verified,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            category=data["category"],
            severity=Severity(data["severity"]),
            confidence=Confidence(data["confidence"]),
            evidence=Evidence.from_dict(data.get("evidence", {})),
            suggested_fixes=[SuggestedFix.from_dict(f) for f in data.get("suggested_fixes", [])],
            verified=data.get("verified", False),
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            tags=data.get("tags", []),
        )


@dataclass
class FindingsReport:
    """Complete findings report from a code quality analysis."""
    workflow_id: str
    generated_at: datetime
    codebase_path: str
    summary: str
    findings: list[Finding]
    stats: dict = field(default_factory=dict)  # Counts by severity, category, etc.

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "generated_at": self.generated_at.isoformat(),
            "codebase_path": self.codebase_path,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FindingsReport":
        return cls(
            workflow_id=data["workflow_id"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            codebase_path=data["codebase_path"],
            summary=data["summary"],
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            stats=data.get("stats", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "FindingsReport":
        return cls.from_dict(json.loads(json_str))

    def get_by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def get_by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def get_actionable(self) -> list[Finding]:
        """Get findings that are actionable (have suggested fixes and high confidence)."""
        return [
            f for f in self.findings
            if f.suggested_fixes and f.confidence in (Confidence.CERTAIN, Confidence.LIKELY)
        ]

    def compute_stats(self) -> dict:
        """Compute statistics for this report."""
        stats = {
            "total": len(self.findings),
            "by_severity": {},
            "by_category": {},
            "by_confidence": {},
            "verified": sum(1 for f in self.findings if f.verified),
            "with_fixes": sum(1 for f in self.findings if f.suggested_fixes),
        }

        for severity in Severity:
            count = sum(1 for f in self.findings if f.severity == severity)
            if count > 0:
                stats["by_severity"][severity.value] = count

        for finding in self.findings:
            stats["by_category"][finding.category] = stats["by_category"].get(finding.category, 0) + 1

        for confidence in Confidence:
            count = sum(1 for f in self.findings if f.confidence == confidence)
            if count > 0:
                stats["by_confidence"][confidence.value] = count

        self.stats = stats
        return stats


# JSON Schema for validation (can be used with jsonschema library)
FINDING_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["id", "title", "description", "category", "severity", "confidence"],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z]+-[0-9]+$"},
        "title": {"type": "string", "maxLength": 100},
        "description": {"type": "string"},
        "category": {
            "type": "string",
            "enum": ["security", "documentation", "consistency", "performance",
                     "dead_code", "error_handling", "testing", "architecture"]
        },
        "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "confidence": {"type": "string", "enum": ["certain", "likely", "possible", "speculative"]},
        "evidence": {
            "type": "object",
            "properties": {
                "files_checked": {"type": "array", "items": {"type": "string"}},
                "search_queries": {"type": "array", "items": {"type": "string"}},
                "code_refs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "code_snippet": {"type": "string"}
                        }
                    }
                },
                "reasoning": {"type": "string"}
            }
        },
        "suggested_fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["file", "description"],
                "properties": {
                    "file": {"type": "string"},
                    "description": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "diff": {"type": "string"}
                }
            }
        },
        "verified": {"type": "boolean"},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "blocks": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}}
    }
}


def get_findings_prompt_instructions() -> str:
    """Get instructions for prompting agents to output structured findings."""
    return '''
OUTPUT FORMAT REQUIREMENTS:

You MUST output your findings as a JSON array. Each finding must follow this exact structure:

```json
[
  {
    "id": "category-001",
    "title": "Short descriptive title (max 100 chars)",
    "description": "Detailed explanation of the issue",
    "category": "security|documentation|consistency|performance|dead_code|error_handling|testing|architecture",
    "severity": "critical|high|medium|low",
    "confidence": "certain|likely|possible|speculative",
    "evidence": {
      "files_checked": ["path/to/file1.py", "path/to/file2.py"],
      "search_queries": ["pattern I searched for"],
      "code_refs": [
        {
          "file": "path/to/file.py",
          "line": 42,
          "end_line": 45,
          "code_snippet": "def problematic_function():\\n    # actual code here"
        }
      ],
      "reasoning": "I found this because X, which indicates Y"
    },
    "suggested_fixes": [
      {
        "file": "path/to/file.py",
        "description": "What the fix does",
        "before": "old code",
        "after": "new code"
      }
    ],
    "verified": false,
    "tags": ["relevant", "tags"]
  }
]
```

IMPORTANT RULES:
1. ALWAYS include actual code snippets, not just line numbers
2. ALWAYS explain your reasoning in the evidence
3. Set confidence to "certain" only if you verified the issue exists
4. Set confidence to "speculative" if you're guessing without verification
5. Include before/after code in suggested_fixes when possible
6. Use consistent category names from the enum above
'''

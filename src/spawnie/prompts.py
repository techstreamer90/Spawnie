"""Base prompts for response routing."""

from pathlib import Path


RESPONSE_ROUTING_PROMPT = """
When you complete this task, save your output to: {response_path}

Format your response as plain text unless otherwise specified.
Do not include any preamble or explanation about saving the file.
Just complete the task and ensure the output is written to the specified path.
"""


def wrap_prompt(
    prompt: str,
    task_id: str,
    response_dir: Path,
    include_routing: bool = True,
) -> str:
    """
    Wrap a user prompt with response routing instructions.

    Args:
        prompt: The original user prompt.
        task_id: The task ID.
        response_dir: Directory where responses should be saved.
        include_routing: Whether to include file routing instructions.

    Returns:
        The wrapped prompt.
    """
    if not include_routing:
        return prompt

    response_path = response_dir / f"{task_id}.txt"
    routing = RESPONSE_ROUTING_PROMPT.format(response_path=response_path)

    return f"{prompt}\n\n---\n{routing}"


def create_analysis_prompt(
    code: str,
    instruction: str = "Analyze this code and provide insights.",
) -> str:
    """
    Create a code analysis prompt.

    Args:
        code: The code to analyze.
        instruction: Analysis instruction.

    Returns:
        Formatted prompt.
    """
    return f"""{instruction}

```
{code}
```
"""


def create_generation_prompt(
    description: str,
    language: str = "python",
    context: str | None = None,
) -> str:
    """
    Create a code generation prompt.

    Args:
        description: What to generate.
        language: Programming language.
        context: Optional context code.

    Returns:
        Formatted prompt.
    """
    prompt = f"Generate {language} code for: {description}"

    if context:
        prompt += f"\n\nContext:\n```{language}\n{context}\n```"

    prompt += "\n\nProvide only the code without explanations."

    return prompt


def create_review_prompt(
    code: str,
    focus: str | None = None,
) -> str:
    """
    Create a code review prompt.

    Args:
        code: The code to review.
        focus: Optional focus area (e.g., "security", "performance").

    Returns:
        Formatted prompt.
    """
    prompt = "Review the following code"
    if focus:
        prompt += f", focusing on {focus}"
    prompt += ":\n\n"
    prompt += f"```\n{code}\n```\n\n"
    prompt += "Provide specific, actionable feedback."

    return prompt

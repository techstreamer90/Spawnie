# Workflow Process Analysis

## What Information Is Available After a Workflow Run

### Currently Stored
1. **Workflow-level outputs** - Final outputs for each declared output in workflow.json
2. **Task records** with:
   - `output_preview` - Truncated output (first ~500 chars?)
   - Status, timing, model used
   - Error message if failed
   - No full output preserved

### NOT Stored (Critical Gaps)
1. **Full task outputs** - Only preview saved, full response lost
2. **Agent reasoning/exploration** - What files did each agent read? What did it try?
3. **Tool calls made** - No record of glob/grep/read operations
4. **Failed attempts** - If agent tried something that didn't work, no record
5. **Confidence levels** - Agent might be guessing vs. certain

---

## Analysis: Could I Fix All Issues From This Report?

### Issues I COULD Fix (Have Enough Info)
| Issue | Why Fixable |
|-------|-------------|
| `--dark` flag missing | Report says where to add it (line numbers given) |
| Add UTF-8 encoding | Already fixed during this session |
| Extract Serializable base class | Pattern is clear, files listed |
| Add missing loggers | Files listed, pattern is standard |

### Issues I COULD NOT Fix Without More Research
| Issue | What's Missing |
|-------|----------------|
| Command injection risk | Which inputs exactly? What sanitization? Need to read session.py |
| Silent session failures | Where exactly? Need to trace code flow |
| Dead prompts.py | Is it truly unused? Need to grep for imports |
| Error handling patterns | What's the "standard" pattern? Need examples from codebase |
| Test results empty | Why did run_tests fail? No diagnostic info |

### Issues That Are VAGUE/UNACTIONABLE
| Issue | Problem |
|-------|---------|
| "Inconsistent error handling" | Which files? What pattern to use? |
| "~175 lines duplicated" | Where exactly? What's the duplication? |
| "Refactor Tracker class" | Big undertaking - needs design discussion |
| "Security: Command injection" | Need threat model, not just "sanitize" |

---

## What Additional Information Would Help

### 1. File References with Context
Instead of: "Fix error handling in session.py:256-271"
Want: The actual code snippet + suggested fix + before/after

### 2. Evidence Trail
For each finding, want:
- What files were read to reach this conclusion?
- What search queries were run?
- What was the reasoning chain?

### 3. Verification Status
- Did the agent actually verify the issue exists?
- Did it try a fix and see if it works?
- Confidence level (certain/likely/speculative)

### 4. Dependency Analysis
- Which fixes depend on other fixes?
- What's the safe order to apply changes?
- Are there conflicts between recommendations?

### 5. Test Coverage
- Which issues have existing tests that would catch regressions?
- Which fixes need new tests?

### 6. Full Intermediate Outputs
- Complete output from each step (not truncated)
- Enable drilling down into specific findings

---

## Why run_tests Produced No Output

The step prompt asked to "find and run the test suite" but:
1. Agent might not have found tests
2. Agent might have run tests but output was empty
3. Agent might have hit an error we can't see

**No way to diagnose** because:
- No log of what commands were attempted
- No record of what files were searched
- No error message captured

---

## Recommendations for Process Improvement

### Short-term Fixes

1. **Store full task outputs** (not just preview)
   - Add `output` field to archived tasks
   - Or write to separate files: `~/.spawnie/outputs/{task_id}.txt`

2. **Add structured findings format**
   ```json
   {
     "finding": "Missing --dark flag",
     "severity": "critical",
     "evidence": {
       "files_checked": ["__main__.py", "README.md"],
       "search_queries": ["--dark", "dark mode"],
       "code_refs": [{"file": "__main__.py", "line": 763}]
     },
     "suggested_fix": {
       "file": "__main__.py",
       "diff": "..."
     },
     "confidence": 0.95,
     "verified": true
   }
   ```

3. **Require code snippets in reports**
   - Prompt should say "Include the actual code for each issue"
   - Not just line numbers, but the problematic code

4. **Add verification step**
   - After finding issues, have agent verify each one
   - "Can you reproduce issue X? Show evidence."

### Medium-term Improvements

5. **Tool call logging**
   - Record every file read, search, command
   - Store in task metadata

6. **Multi-pass validation**
   - First pass: Find issues
   - Second pass: Different agent verifies findings
   - Discard low-confidence items

7. **Interactive clarification**
   - If agent is unsure, it should ask
   - "I found possible dead code in prompts.py. Should I verify by checking all imports?"

### Long-term Vision

8. **Diff-based fixes**
   - Output should include actual patches
   - `git apply` ready diffs for each fix

9. **Test generation**
   - For each fix, generate a test that would catch regression

10. **Dependency graph**
    - Show which fixes are safe to apply independently
    - Which need to be applied together

---

## Conclusion

The current workflow produces a **good summary** but not enough **actionable detail** to confidently fix issues without additional research. The main gaps are:

1. **Lost information** - Full agent outputs truncated
2. **No evidence trail** - Can't verify findings
3. **Vague recommendations** - "Fix error handling" not actionable
4. **No verification** - Agent might be wrong

To make this production-ready, we need structured output with evidence, full logs, and verification steps.

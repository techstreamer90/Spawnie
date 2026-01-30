# Seed: BAM Model of Spawnie

This is a BAM model (digital twin) of the Spawnie workflow orchestrator.

## The Key Insight

**Reality is part of the model.**

Each node in the sketch has a `source` reference containing:
- `file`: Path to actual source code
- `line`: Line number where the code starts
- `hash`: SHA256 hash of the file

When reality changes (code is modified), the hash mismatches signal **drift**.
Agents can only modify the sketch if they can prove the change matches reality.

## Purpose

This model serves as:
1. **Verifiable documentation** - The sketch points to real code with hashes
2. **Shared context for agents** - Agents query the model, not raw files
3. **Drift detection** - Automatic detection when reality diverges from model
4. **Declarative development** - Agents converge reality toward the sketch

## Structure

```
seed/
├── project.json           # Project metadata
├── model/
│   ├── sketch.json        # Graph structure (nodes, edges, types)
│   └── colors/            # Detailed properties per node
│       ├── modules.json   # Python modules
│       ├── classes.json   # Classes and their methods
│       ├── concepts.json  # Architectural patterns
│       └── cli.json       # CLI commands
├── sources/               # Source data (if any)
└── intermediate/          # Intermediate extraction format
```

## The Model

### Node Types

| Type | Description | Count |
|------|-------------|-------|
| Module | Python module (file) | 10 |
| Class | Python class | 10 |
| Function | Standalone function | 4 |
| Dataclass | Data structure | 5 |
| CLICommand | CLI command | 6 |
| Concept | Architectural pattern | 5 |

### Edge Types

| Type | Description |
|------|-------------|
| IMPORTS | Module imports another |
| DEFINES | Module defines class/function |
| INHERITS | Class inherits from another |
| CALLS | Function/method calls another |
| USES | Class uses another class |
| IMPLEMENTS | CLI command implemented by handler |
| EMBODIES | Code embodies a concept |

### Key Concepts Captured

1. **Tasks as Single-Step Workflows** - Unified tracking
2. **Model Routing** - Abstract model names → concrete providers
3. **Parallel Step Execution** - Concurrent independent steps
4. **Orphan Detection** - Cleanup dead processes
5. **Quality Levels** - Review strategies (normal/extra-clean/hypertask)

## Usage

### Verify the model

```bash
# Check that model matches reality
python verify.py

# With custom source root
python verify.py /path/to/spawnie
```

Output:
```
=== BAM Model Verification ===

VERIFIED (40 nodes):
  [OK] cls-tracker -> src/spawnie/tracker.py
  [OK] cls-workflow-executor -> src/spawnie/workflow.py
  ...

=== Summary ===
Total nodes with source: 40
  Verified: 40
  Drifted:  0
  Missing:  0

[SUCCESS] Model matches reality!
```

### Query the model

```python
import json
from pathlib import Path

# Load sketch
sketch = json.loads(Path("model/sketch.json").read_text())

# Find all classes with their source locations
classes = [n for n in sketch["nodes"] if n["type"] == "Class"]
for c in classes:
    src = c.get("source", {})
    print(f"{c['label']}: {src.get('file')}:{src.get('line')}")

# Find what Tracker uses
tracker_edges = [e for e in sketch["edges"]
                 if e["from"] == "cls-tracker" and e["type"] == "USES"]
```

### Enrich with color

```python
# Load color
modules = json.loads(Path("model/colors/modules.json").read_text())

# Get details for tracker module
tracker_info = modules["nodes"]["mod-tracker"]
print(f"LOC: {tracker_info['loc']}")
print(f"Public API: {tracker_info['public_api']}")
```

## Next Steps

1. **Python extractor for BAM** - Auto-generate this from source code
2. **Keep model in sync** - Update when Spawnie changes
3. **Agent integration** - Spawnie workflows that query this model
4. **Cross-references** - Link to other BAM models

## The Vision

Reality is part of the model. The sketch contains source references (file paths, line numbers, hashes) that point directly to code. This creates a closed loop:

```
        ┌─────────────────────────────┐
        │      BAM Model (Sketch)     │
        │         + Color             │
        │                             │
        │  ┌───────────────────────┐  │
        │  │   Source References   │──┼──┐
        │  │   (files + hashes)    │  │  │ Points to
        │  └───────────────────────┘  │  │
        └──────────────┬──────────────┘  │
                       │                 │
          Agents modify│                 ▼
                       │    ┌─────────────────────────────┐
                       │    │        Real World           │
                       └───►│    (code, systems)          │
                            └─────────────────────────────┘
                                         │
                    verify.py checks ────┘
                    hashes match
```

**Workflow:**
1. Agent reads sketch to understand desired state
2. Agent modifies real code to implement changes
3. Agent updates sketch with new source references + hashes
4. `verify.py` confirms model matches reality

The model cannot drift silently - hash mismatches immediately signal that reality has changed.

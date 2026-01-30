# Notes for Future Claude

This document captures the key insights and current state of this project.

## The Core Insight

**Reality is part of the model.**

Traditional documentation describes code but drifts over time. This model embeds reality directly via source references with cryptographic hashes. When code changes, `verify.py` detects the drift immediately.

## The Two Layers

The model has (or should have) two layers:

### 1. Intent Layer (what we want)
Nodes WITHOUT a `source` field represent desired state - things that should exist but don't yet.

```json
{
  "id": "cls-bam-extractor",
  "type": "Class",
  "label": "BAMExtractor",
  "description": "Auto-generates BAM models from Python source"
}
```

This is the GOAL. An agent's job is to implement it.

### 2. Reality Layer (what we have)
Nodes WITH a `source` field represent current state - things that exist with verified locations.

```json
{
  "id": "cls-tracker",
  "type": "Class",
  "label": "Tracker",
  "source": {
    "file": "src/spawnie/tracker.py",
    "line": 215,
    "hash": "c4561408b1272866a4f8423d66b811cb7cd5575e59e90397d8f42adef3f0f165"
  }
}
```

This is REALITY. The hash proves it matches the actual code.

### The Delta

The difference between intent and reality is **the work to be done**.

- Intent node with no source → needs to be implemented
- Reality node with hash mismatch → code changed, model needs update
- Reality node marked "deprecated" → needs to be removed

## What's Built

1. **sketch.json v2.0** - 40 nodes with source references, 5 abstract concepts
2. **verify.py** - Checks all hashes, reports drift
3. **Color files** - Detailed properties in `model/colors/*.json`

## What's Missing

1. **Intent nodes** - Currently all nodes describe existing code. No "goals" yet.
2. **Update tooling** - No automated way for agents to update model after code changes
3. **Extraction tooling** - No way to auto-generate model from source code
4. **Agent integration** - No Spawnie workflow that uses this model as context

## The Vision (from our conversation)

```
        ┌─────────────────────────────┐
        │      BAM Model (Sketch)     │
        │                             │
        │  Intent: what should exist  │
        │  Reality: what does exist   │
        │           (via hashes)      │
        └──────────────┬──────────────┘
                       │
          Agents read intent,
          implement in reality,
          update source refs
                       │
                       ▼
        ┌─────────────────────────────┐
        │        Real World           │
        │    (code, systems)          │
        └─────────────────────────────┘
```

The model is not documentation. It's the **source of truth** that agents converge reality towards.

## Key Files

- `model/sketch.json` - Graph structure with source references
- `model/colors/*.json` - Detailed properties per node type
- `verify.py` - Reality verification script
- `README.md` - Public documentation

## How to Continue

1. **Add intent**: Create nodes without `source` for things you want built
2. **Implement**: Build the code those nodes describe
3. **Link reality**: Add `source` with file/line/hash once implemented
4. **Verify**: Run `python verify.py` to confirm model matches code

## Context

This model lives inside Spawnie at `spawnie/bam/`. The model describes itself.
Reality contains its own model - self-documenting in the most literal sense.

Source root is `..` (parent directory), so `src/spawnie/tracker.py` resolves
to the actual tracker.py in this repo.

Built: 2026-01-31 (Friday night session)
Moved into reality: 2026-01-31

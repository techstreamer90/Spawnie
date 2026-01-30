#!/usr/bin/env python3
"""
Verify that the BAM model matches reality.

This script checks that source references in sketch.json
point to actual code that still exists with matching hashes.
"""

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_model(model_dir: Path, source_root: Path) -> dict:
    """Verify sketch.json against reality."""
    sketch_path = model_dir / "sketch.json"

    if not sketch_path.exists():
        return {"error": f"Sketch not found: {sketch_path}"}

    sketch = json.loads(sketch_path.read_text())

    results = {
        "verified": [],
        "drifted": [],
        "missing": [],
        "no_source": [],
    }

    for node in sketch.get("nodes", []):
        node_id = node.get("id")
        source = node.get("source")

        if not source:
            results["no_source"].append(node_id)
            continue

        file_path = source_root / source.get("file", "")
        expected_hash = source.get("hash")

        if not file_path.exists():
            results["missing"].append({
                "id": node_id,
                "file": str(file_path),
            })
            continue

        actual_hash = sha256_file(file_path)

        if actual_hash == expected_hash:
            results["verified"].append({
                "id": node_id,
                "file": source.get("file"),
            })
        else:
            results["drifted"].append({
                "id": node_id,
                "file": source.get("file"),
                "expected": expected_hash[:16] + "...",
                "actual": actual_hash[:16] + "...",
            })

    return results


def print_results(results: dict) -> bool:
    """Print verification results. Returns True if all verified."""
    verified = results.get("verified", [])
    drifted = results.get("drifted", [])
    missing = results.get("missing", [])
    no_source = results.get("no_source", [])

    print(f"\n=== BAM Model Verification ===\n")

    if verified:
        print(f"VERIFIED ({len(verified)} nodes):")
        for v in verified:
            print(f"  [OK] {v['id']} -> {v['file']}")

    if drifted:
        print(f"\nDRIFTED ({len(drifted)} nodes) - reality changed:")
        for d in drifted:
            print(f"  [DRIFT] {d['id']} -> {d['file']}")
            print(f"          expected: {d['expected']}")
            print(f"          actual:   {d['actual']}")

    if missing:
        print(f"\nMISSING ({len(missing)} nodes) - files not found:")
        for m in missing:
            print(f"  [MISSING] {m['id']} -> {m['file']}")

    if no_source:
        print(f"\nNO SOURCE ({len(no_source)} nodes) - abstract concepts:")
        for n in no_source:
            print(f"  [ABSTRACT] {n}")

    total = len(verified) + len(drifted) + len(missing)
    print(f"\n=== Summary ===")
    print(f"Total nodes with source: {total}")
    print(f"  Verified: {len(verified)}")
    print(f"  Drifted:  {len(drifted)}")
    print(f"  Missing:  {len(missing)}")
    print(f"  Abstract: {len(no_source)}")

    all_ok = len(drifted) == 0 and len(missing) == 0
    if all_ok:
        print("\n[SUCCESS] Model matches reality!")
    else:
        print("\n[WARNING] Model has drifted from reality.")

    return all_ok


if __name__ == "__main__":
    import sys

    # Default paths - relative to this script's location
    bam_dir = Path(__file__).parent.resolve()
    model_dir = bam_dir / "model"

    # Read source_root from sketch.json
    sketch = json.loads((model_dir / "sketch.json").read_text())
    source_root_str = sketch.get("source_root", "..")
    source_root = (bam_dir / source_root_str).resolve()

    # Allow override via args
    if len(sys.argv) > 1:
        source_root = Path(sys.argv[1]).resolve()

    results = verify_model(model_dir, source_root)

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    success = print_results(results)
    sys.exit(0 if success else 1)

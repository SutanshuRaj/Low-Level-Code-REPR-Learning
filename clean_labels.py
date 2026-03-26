#!/usr/bin/env python3
"""
Clean labeled dataset by removing rare classes and optionally rebalancing.

Usage:
    python clean_labels.py                    # Clean rare classes
    python clean_labels.py --merge-complexity # Also merge 'high' into 'medium'
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def clean_labels(input_path: str, output_path: str, merge_complexity: bool = False):
    """Remove rare classes and optionally merge complexity levels."""

    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} functions")

    # Classes to remove from side_effects (too rare to learn)
    RARE_SIDE_EFFECTS = {"assertions", "network"}

    # Track changes
    changes = {"side_effects_cleaned": 0, "complexity_merged": 0}

    for item in data:
        labels = item.get("labels", {})

        # Clean side_effects
        if "side_effects" in labels:
            original = set(labels["side_effects"])
            cleaned = [se for se in labels["side_effects"] if se not in RARE_SIDE_EFFECTS]

            # If all effects were rare, default to "none"
            if not cleaned:
                cleaned = ["none"]

            if set(cleaned) != original:
                changes["side_effects_cleaned"] += 1

            labels["side_effects"] = cleaned

        # Optionally merge complexity
        if merge_complexity and "complexity" in labels:
            if labels["complexity"] == "high":
                labels["complexity"] = "medium"
                changes["complexity_merged"] += 1

    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nChanges made:")
    print(f"  - Side effects cleaned (removed rare): {changes['side_effects_cleaned']}")
    if merge_complexity:
        print(f"  - Complexity 'high' merged to 'medium': {changes['complexity_merged']}")

    # Print new distribution
    print("\n=== New Side Effect Distribution ===")
    se_counts = Counter()
    combo_counts = Counter()
    for item in data:
        effects = item['labels'].get('side_effects', ['none'])
        for e in effects:
            se_counts[e] += 1
        combo_counts[tuple(sorted(effects))] += 1

    for k, v in sorted(se_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\n  Unique combinations: {len(combo_counts)} (was 22)")

    if merge_complexity:
        print("\n=== New Complexity Distribution ===")
        cx_counts = Counter(item['labels'].get('complexity', 'medium') for item in data)
        for k, v in sorted(cx_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    print(f"\nSaved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean labeled dataset")
    parser.add_argument("--input", "-i", default="data/processed/labeled_functions.json")
    parser.add_argument("--output", "-o", default="data/processed/labeled_functions.json")
    parser.add_argument("--merge-complexity", action="store_true",
                        help="Merge 'high' complexity into 'medium'")

    args = parser.parse_args()
    clean_labels(args.input, args.output, args.merge_complexity)

if __name__ == "__main__":
    main()

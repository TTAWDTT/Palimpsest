"""Matched-source changes with class-balanced, source bootstrap intervals."""

import random
from collections import defaultdict


def paired_change(original, processed) -> dict[str, object]:
    common = set(original) & set(processed)
    by_label = defaultdict(list)
    for source in sorted(common):
        label = original[source]["label"]
        if processed[source]["label"] != label:
            raise ValueError(f"Paired source label mismatch: {source}")
        first = float(original[source]["score"]) > 0
        second = float(processed[source]["score"]) > 0
        truth = label == "FAKE"
        by_label[label].append(
            (int(second == truth) - int(first == truth), int(first != second))
        )
    if set(by_label) != {"REAL", "FAKE"}:
        raise ValueError("Paired evaluation requires both source classes")

    class_change = {
        label: sum(change for change, _ in values) / len(values)
        for label, values in by_label.items()
    }
    class_flip = {
        label: sum(flip for _, flip in values) / len(values)
        for label, values in by_label.items()
    }
    rng = random.Random(20260924)
    bootstrap = []
    for _ in range(2000):
        samples = {
            label: rng.choices(values, k=len(values))
            for label, values in by_label.items()
        }
        bootstrap.append(
            sum(
                sum(change for change, _ in samples[label]) / len(samples[label])
                for label in ("REAL", "FAKE")
            )
            / 2
        )
    bootstrap.sort()
    return {
        "paired_sources": len(common),
        "paired_real_sources": len(by_label["REAL"]),
        "paired_fake_sources": len(by_label["FAKE"]),
        "class_accuracy_change": class_change,
        "balanced_accuracy_change": sum(class_change.values()) / 2,
        "balanced_accuracy_change_ci95": [bootstrap[49], bootstrap[1950]],
        "class_decision_flip_rate": class_flip,
        "source_macro_decision_flip_rate": sum(class_flip.values()) / 2,
        "unpaired_original_sources": len(set(original) - common),
        "unpaired_processed_sources": len(set(processed) - common),
    }

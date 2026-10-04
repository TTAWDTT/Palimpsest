"""Origin classification metrics: FAKE-positive logits, strict score > 0.

Bootstrap sampling is by source and class, preserving the published local
protocol and its seed. This module has no dataset or detector dependency.
"""

from __future__ import annotations

import random
from collections import defaultdict

LABELS = {"REAL": 0, "FAKE": 1}


def auc(scores_and_labels: list[tuple[float, int]]) -> float:
    sorted_pairs = sorted(scores_and_labels)
    positive_count = sum(label for _, label in sorted_pairs)
    negative_count = len(sorted_pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both classes")

    positive_rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while (
            end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]
        ):
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label for _, label in sorted_pairs[index:end]
        )
        index = end

    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def evaluate(rows: list[dict[str, str]], detector: str) -> dict[str, float | int]:
    pairs: list[tuple[float, int]] = []
    source_correct: dict[str, list[int]] = defaultdict(list)
    source_fake_decisions: dict[str, list[int]] = defaultdict(list)
    source_labels: dict[str, int] = {}
    class_correct = [0, 0]
    class_counts = [0, 0]

    for row in rows:
        label = LABELS[row["label"]]
        score = float(row[detector])
        source = row["src"]
        if source in source_labels and source_labels[source] != label:
            raise ValueError(f"Conflicting origin labels for source {source}")
        source_labels[source] = label
        correct = int((score > 0) == bool(label))
        pairs.append((score, label))
        source_correct[source].append(correct)
        source_fake_decisions[source].append(int(score > 0))
        class_correct[label] += correct
        class_counts[label] += 1

    source_accuracy = [[], []]
    source_pair_disagreement = [[], []]
    for source, correctness in source_correct.items():
        source_label = source_labels[source]
        source_accuracy[source_label].append(sum(correctness) / len(correctness))
        decisions = source_fake_decisions[source]
        fake_decisions = sum(decisions)
        if len(decisions) > 1:
            source_pair_disagreement[source_label].append(
                2
                * fake_decisions
                * (len(decisions) - fake_decisions)
                / (len(decisions) * (len(decisions) - 1))
            )

    real_accuracy = class_correct[0] / class_counts[0]
    fake_accuracy = class_correct[1] / class_counts[1]
    source_real_accuracy = sum(source_accuracy[0]) / len(source_accuracy[0])
    source_fake_accuracy = sum(source_accuracy[1]) / len(source_accuracy[1])
    bootstrap_rng = random.Random(20260923)
    bootstrap_ba = []
    for _ in range(2000):
        sampled_real = bootstrap_rng.choices(
            source_accuracy[0], k=len(source_accuracy[0])
        )
        sampled_fake = bootstrap_rng.choices(
            source_accuracy[1], k=len(source_accuracy[1])
        )
        bootstrap_ba.append(
            (
                sum(sampled_real) / len(sampled_real)
                + sum(sampled_fake) / len(sampled_fake)
            )
            / 2
        )
    bootstrap_ba.sort()
    return {
        "images": len(rows),
        "real_images": class_counts[0],
        "fake_images": class_counts[1],
        "sources": len(source_labels),
        "real_sources": len(source_accuracy[0]),
        "fake_sources": len(source_accuracy[1]),
        "auc": auc(pairs),
        "balanced_accuracy_at_zero": (real_accuracy + fake_accuracy) / 2,
        "real_accuracy_at_zero": real_accuracy,
        "fake_accuracy_at_zero": fake_accuracy,
        "source_macro_balanced_accuracy_at_zero": (
            source_real_accuracy + source_fake_accuracy
        )
        / 2,
        "source_macro_balanced_accuracy_ci95": [bootstrap_ba[49], bootstrap_ba[1950]],
        "source_macro_pair_disagreement": (
            (sum(source_pair_disagreement[0]) / len(source_pair_disagreement[0]))
            + (sum(source_pair_disagreement[1]) / len(source_pair_disagreement[1]))
        )
        / 2
        if source_pair_disagreement[0] and source_pair_disagreement[1]
        else None,
    }

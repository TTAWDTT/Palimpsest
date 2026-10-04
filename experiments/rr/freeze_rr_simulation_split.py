"""Record RR sources already used for simulation development and reserve the rest."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import REPO_ROOT

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.rr.analyze_rr_simulation_inputs import MANIFEST, load_pairs
from experiments.rr.rr_oracle_codec_control import (
    select_sources as select_oracle_sources,
)
from experiments.rr.rr_simulator_v0 import split_sources


BASE = REPO_ROOT
EVALUATION = BASE / "work" / "rr_simulator_crop_1000_evaluation.json"
DIAGNOSTIC = BASE / "work" / "rr_simulation_diagnostic.json"
ORACLE = BASE / "work" / "rr_oracle_codec_control.json"
OUTPUT = DATA_ROOT / "manifests/rr_simulation_source_registry.csv"
SUMMARY = BASE / "work" / "rr_simulation_source_registry.json"
MANUALLY_INSPECTED = {
    "real/real_004508",
    "ai/normal_009005",
    "real/real_001659",
    "ai/Medical_&_Public_Health_000583",
    "real/real_001546",
    "real/real_003985",
    "real/real_007961",
}
PREVIOUS_PILOTS = (
    "d3_rr_pilot.csv",
    "benford_rr_pilot.csv",
    "fourier_knn_rr_pilot.csv",
    "d3_rr_smoke.csv",
    "rr_largest_sanity.csv",
)


def sample_diagnostic_sources(
    sources, sample_per_class: int, max_pixels: int
) -> list[str]:
    selected = []
    for label in ("ai", "real"):
        candidates = []
        for source, group in sources.items():
            if not source.startswith(label + "/"):
                continue
            if any(
                int(row["width"]) * int(row["height"]) > max_pixels
                for row in group.values()
            ):
                continue
            rank = hashlib.sha256(
                f"rr-simulation-diagnostic-20260924/{source}".encode()
            ).digest()
            candidates.append((rank, source))
        selected.extend(source for _, source in sorted(candidates)[:sample_per_class])
    return selected


def digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def pilot_sources() -> set[str]:
    sources = set()
    for filename in PREVIOUS_PILOTS:
        with (BASE / "work" / filename).open(newline="", encoding="utf-8") as handle:
            sources.update(row["src"] for row in csv.DictReader(handle))
    validation = json.loads(
        (BASE / "work" / "bfree_crop_first_validation.json").read_text(encoding="utf-8")
    )
    for case in validation["cases"]:
        parts = Path(case["filename"]).parts
        sources.add(f"{parts[1]}/{Path(parts[2]).stem}")
    return sources


def inspected_anomalies(sources) -> set[str]:
    anomalies = set()
    for source, group in sources.items():
        if "redigital" not in group:
            continue
        transformed = group["redigital"]
        if (int(transformed["width"]), int(transformed["height"])) == (700, 700):
            anomalies.add(source)
        if group["original"]["sha256"] == transformed["sha256"]:
            anomalies.add(source)
    return anomalies


def main() -> None:
    sources = load_pairs()
    calibration, _ = split_sources(sources)
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    development = list(
        dict.fromkeys(row["source"] for row in evaluation["per_source_features"])
    )
    if digest(development) != evaluation["validation_source_ids_sha256"]:
        raise ValueError("simulator development source fingerprint mismatch")
    diagnostic_config = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))[
        "sample_descriptors"
    ]["sampling"]
    diagnostic = sample_diagnostic_sources(
        sources,
        diagnostic_config["per_class"],
        diagnostic_config["max_pixels_per_image"],
    )
    if digest(diagnostic) != diagnostic_config["selected_source_ids_sha256"]:
        raise ValueError("diagnostic source fingerprint mismatch")
    oracle_config = json.loads(ORACLE.read_text(encoding="utf-8"))
    oracle = select_oracle_sources(
        sources,
        oracle_config["selected_sources_per_class"],
        oracle_config["max_pixels"],
    )
    if digest(oracle) != oracle_config["selected_source_ids_sha256"]:
        raise ValueError("oracle source fingerprint mismatch")
    inspected = (
        set(diagnostic)
        | set(oracle)
        | MANUALLY_INSPECTED
        | pilot_sources()
        | inspected_anomalies(sources)
    )
    development_set = set(development)
    if development_set & calibration.keys():
        raise ValueError("calibration/development overlap")
    status_by_source = {}
    for source, group in sources.items():
        if "redigital" not in group:
            status = "incomplete_pair"
        elif source in calibration:
            status = "calibration"
        elif source in development_set:
            status = "development"
        elif source in inspected:
            status = "prior_exploration"
        else:
            status = "reserved_simulation_check"
        status_by_source[source] = status
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "label", "status"))
        writer.writeheader()
        writer.writerows(
            {
                "source": source,
                "label": source.split("/", 1)[0],
                "status": status_by_source[source],
            }
            for source in sorted(status_by_source)
        )
    summary = {
        "scope": "RR test internal simulator development registry; reserved is not an external blind detector test",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "registry_path": str(OUTPUT),
        "registry_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "counts": dict(Counter(status_by_source.values())),
        "prior_exploration_source_count": len(inspected),
        "development_source_count": len(development_set),
        "reserved_source_ids_sha256": digest(
            sorted(
                source
                for source, status in status_by_source.items()
                if status == "reserved_simulation_check"
            )
        ),
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

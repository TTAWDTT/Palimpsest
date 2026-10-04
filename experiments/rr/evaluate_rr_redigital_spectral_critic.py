"""Compare small RR redigital simulations with source-disjoint two-sample critics."""

from __future__ import annotations

from palimpsest.paths import REPO_ROOT

import hashlib
import json

from experiments.rr.evaluate_rr_simulator_critic import NUMERIC_FEATURES, evaluate


BASE = REPO_ROOT
INPUT = BASE / "work" / "rr_redigital_420_spectral_residual_pilot.json"
OUTPUT = BASE / "work" / "rr_redigital_420_spectral_residual_critic.json"
SPECTRAL_FEATURES = (
    "mid_spectrum_log_power_ratio",
    "high_spectrum_log_power_ratio",
)


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    if data["sources"] != 506 or len(data["per_source_features"]) != 506:
        raise ValueError("unexpected development source count")
    result = {
        "scope": "source-disjoint critics on RR-internal 128-pixel proxy features",
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "conditions": {},
    }
    for variant in (
        "color",
        "white_noise",
        "spectral_residual",
        "color_no_proxy_jpeg",
        "white_noise_no_proxy_jpeg",
        "spectral_residual_no_proxy_jpeg",
    ):
        rows = [
            {"source": row["source"], "real": row["real"], "simulated": row[variant]}
            for row in data["per_source_features"]
        ]
        result["conditions"][variant] = {
            "five_features": evaluate(rows, False, NUMERIC_FEATURES),
            "five_plus_spectrum": evaluate(
                rows, False, NUMERIC_FEATURES + SPECTRAL_FEATURES
            ),
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

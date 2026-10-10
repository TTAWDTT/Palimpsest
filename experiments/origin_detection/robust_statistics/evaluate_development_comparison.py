"""Descriptive cached baselines on the issued third-iteration selection sources."""

import json
import tomllib

from palimpsest.contracts import Origin
from palimpsest.evaluation.cached import CachedRun, ExpectedImage, evaluate_cached_method
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.baselines.evaluate_rr_suite import FILES, PROVENANCE
from experiments.origin_detection.ordinal_statistics.run_iteration import CONFIG, OUTPUT, load_cache, write_json


def main():
    path = OUTPUT / "baseline_comparison.json"
    if path.exists():
        raise FileExistsError("Preserve cached baseline comparison")
    with CONFIG.open("rb") as stream:
        rows, _, _ = load_cache(tomllib.load(stream))
    conditions = {"stylegan2_orig": "original", "recap_mac": "mac_iphone", "recap_monitor": "lg_blackfly"}
    result = {"scope": "Post-hoc descriptive cached baseline comparison; same development selection source sets; no inference or model selection",
              "features_receipt_sha256": file_sha256(OUTPUT / "features.json"),
              "analysis_code_sha256": file_sha256(__import__("pathlib").Path(__file__)), "domains": {}}
    for domain in ("rr", "chimera"):
        selected = [r for r in rows if r["role"] == "selection" and r["domain"] == domain]
        expected = [ExpectedImage(r["src"].split("/", 1)[1], r["condition"],
                                  Origin.AI if r["label"] == "FAKE" else Origin.NATURAL, r["relative_path"]) for r in selected]
        source_set = {r.source for r in expected}
        outputs = {}
        runs = FILES if domain == "rr" else {"B-Free": ("chimera_bfree_full.csv", "24012eacbd52b3d23c698c8f631edbf688e1560b836e859173413460f67e0d6e")}
        for method, (filename, sha) in runs.items():
            mapping = {c: c for c in ("original", "transfer", "redigital")} if domain == "rr" else conditions
            outputs[method], _ = evaluate_cached_method(
                expected, [CachedRun(WORK_DIR / filename, condition_map=mapping, expected_sha256=sha,
                                     provenance=PROVENANCE[method] if domain == "rr" else {"protocol": "official B-Free fixed five-crop batch1"})],
                selected_sources=source_set, method=method,
                score_kind="logit" if method != "Benford-RF" else "probability_margin")
        result["domains"][domain] = {"sources": len(source_set), "images": len(expected), "methods": outputs}
    write_json(path, result)
    print(json.dumps({d: v["sources"] for d, v in result["domains"].items()}))


if __name__ == "__main__":
    main()

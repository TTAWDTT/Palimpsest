"""Compare immutable single-image predictions on identical Chimera sources."""

from collections import defaultdict
import json

from palimpsest.contracts import Origin
from palimpsest.evaluation.cached import CachedRun, ExpectedImage, evaluate_cached_method
from palimpsest.evaluation.classification import evaluate
from palimpsest.evaluation.pairing import paired_change
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.robust_statistics.chimera.run_algorithm import MANIFEST, CONDITIONS
from experiments.origin_detection.robust_statistics.rr.protocol import write_json

OUTPUT = WORK_DIR / "robust_statistics/chimera_paired_projection"
OLD = WORK_DIR / "robust_statistics/chimera_first_iteration"


def main():
    result_path = OUTPUT / "comparison.json"
    if result_path.exists():
        raise FileExistsError("Fixed external comparison exists; no overwrite")
    receipt_path = OUTPUT / "frozen_rule.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if file_sha256(MANIFEST) != "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24":
        raise ValueError("External inventory changed")
    old_path = OLD / "frozen_rule.json"
    if file_sha256(old_path) != "fea8deea24980f8176daa6f9fdb58d883df138f5ca6bcd42b89c1bfe95da6005":
        raise ValueError("Prior diagnostic receipt changed")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    rows = read_rows(MANIFEST)
    expected = [ExpectedImage(r["src"], CONDITIONS[r["condition"]],
                              Origin.AI if r["label"] == "FAKE" else Origin.NATURAL, r["filename"]) for r in rows]
    if len(expected) != 3600 or len({r.source for r in expected}) != 1200:
        raise ValueError("External comparison counts differ")
    labels = {r.source: "FAKE" if r.label == Origin.AI else "REAL" for r in expected}
    scene_sources = defaultdict(set)
    for source in labels:
        scene_sources[source.split("/", 1)[0]].add(source)
    runs = {
        "projection": (OUTPUT / "frozen_rule.csv", receipt["csv_sha256"], receipt["algorithm"]["threshold"]),
        "previous_color": (OLD / "frozen_rule.csv", old["csv_sha256"], old["algorithm"]["threshold"]),
        "B-Free": (WORK_DIR / "chimera_bfree_full.csv",
                   "24012eacbd52b3d23c698c8f631edbf688e1560b836e859173413460f67e0d6e", 0.0),
    }
    reports, predictions = {}, {}
    for method, (path, fingerprint, threshold) in runs.items():
        reports[method], predictions[method] = evaluate_cached_method(
            expected, [CachedRun(path, condition_map=CONDITIONS, expected_sha256=fingerprint)],
            method=method, threshold=threshold, score_kind="logit" if method == "B-Free" else "statistical_score")
    by_condition = {}
    for condition in CONDITIONS.values():
        margins = {method: {source: {"src": source, "label": labels[source], "score": prediction.margin}
                            for source, prediction in predictions[method][condition].items()}
                   for method in predictions}
        by_condition[condition] = {
            "projection_minus_previous_color": paired_change(margins["previous_color"], margins["projection"]),
            "projection_minus_B-Free": paired_change(margins["B-Free"], margins["projection"]),
        }
    strata = {}
    for scene, sources in sorted(scene_sources.items()):
        strata[scene] = {method: {condition: evaluate(
            [{"src": source, "label": labels[source], "score": mapping[source].margin}
             for source in sorted(sources)], "score")
            for condition, mapping in predictions[method].items()} for method in predictions}
    write_json(result_path, {
        "scope": "fixed RR parameters on previously exposed external dataset; no independent blind test",
        "sources": len(labels), "images": len(expected), "methods": reports,
        "paired_method_differences": by_condition, "posthoc_scene_strata": strata,
        "method_difference_semantics": "candidate BA minus comparator BA on identical class/source sets; "
                                       "paired_change is reused across methods; no causal intervention or novelty evidence",
        "inputs": {str(p): file_sha256(p) for p in (MANIFEST, receipt_path, old_path)},
        "analysis_code_sha256": file_sha256(REPO_ROOT / "experiments/origin_detection/paired_projection/chimera/evaluate_comparison.py"),
    })
    for condition, value in by_condition.items():
        print(condition, {method: v["balanced_accuracy_change"] for method, v in value.items()}, flush=True)


if __name__ == "__main__":
    main()

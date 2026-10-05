"""One cached-score entry for RR baselines and digital propagation controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from palimpsest.contracts import Origin
from palimpsest.evaluation.cached import (
    CachedRun,
    ExpectedImage,
    classification_table,
    evaluate_cached_method,
)
from palimpsest.evaluation.channel_fidelity import (
    compare_detector_response,
    paired_score_error_gain,
)
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR
from experiments.origin_detection.propagation_controls.rr.protocol import (
    CONFIG,
    CONTROLS,
    CONTROL_ROOT,
    CURRENT_INDEX,
    CURRENT_SCORES,
    RESULT_ROOT,
    expected_images,
    load_inputs,
)

CONDITIONS = ("original", "transfer", "redigital")
FILES = {
    "B-Free": (
        "rr_bfree_complete.csv",
        "0e4b82ea9a3134ea464cc89575303a76de4d9420056f34bb2601731351570a9b",
    ),
    "D3": (
        "d3_rr_full.csv",
        "706fe0ee5c8a0af1cc89643ec1d1901fe5070e2b85f41703415cc613f3520455",
    ),
    "Benford-RF": (
        "benford_rr_full.csv",
        "adebcfed28465c013358a3c75331443dd29201806ec242b6bab1c381707e535d",
    ),
}
PROVENANCE = {
    "B-Free": {
        "hardware": "RTX 4060 Laptop GPU",
        "protocol": "official weights, five crops, batch1, TF32 off, tile64, crop-first above 8MP",
        "summary": "rr_bfree_complete_summary.json",
    },
    "D3": {
        "hardware": "RTX 4060 Laptop GPU",
        "protocol": "official weights, batch1 patch shuffle, global seed418; differs from author batch128",
        "summary": "d3_rr_full.json",
    },
    "Benford-RF": {
        "hardware": "CPU (historical record does not identify exact CPU model)",
        "protocol": "paper-inspired adaptation; fixed Benford DCT features and RF100 trained on RR train originals",
        "summary": "benford_rr_full.json",
    },
}


def load_expected() -> tuple[list[ExpectedImage], set[str], dict]:
    manifest = DATA_ROOT / "manifests/rr_test_files.csv"
    if (
        file_sha256(manifest)
        != "1b5ca7632034680bbffcb8f74524ee4225ab5bb11aeead79f7fce244928501e5"
    ):
        raise ValueError("RR image manifest fingerprint mismatch")
    audit = WORK_DIR / "rr_bfree_evaluation.json"
    if (
        file_sha256(audit)
        != "ecac9870f085dc46be9c81cb89252ef01a3f5a38bdbf9c8b570ac8e0175e92ce"
    ):
        raise ValueError("Original overlap audit fingerprint mismatch")
    excluded = set(
        json.loads(audit.read_text(encoding="utf-8"))["trainval_overlap_audit"][
            "excluded_source_ids"
        ]
    )
    trainval = DATA_ROOT / "manifests/rr_trainval_files.csv"
    with trainval.open(encoding="utf-8-sig", newline="") as stream:
        trainval_hashes = {row["sha256"] for row in csv.DictReader(stream)}
    images, actual_overlaps = [], set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            source = f"{row['label']}/{row['source_id']}"
            if row["condition"] == "original" and row["sha256"] in trainval_hashes:
                actual_overlaps.add(source)
            if source not in excluded:
                images.append(
                    ExpectedImage(
                        source,
                        row["condition"],
                        Origin.AI if row["label"] == "ai" else Origin.NATURAL,
                        row["filename"],
                    )
                )
    if actual_overlaps != excluded or len(excluded) != 14:
        raise ValueError("Independent original SHA overlap recomputation disagrees")
    counts = Counter(image.condition for image in images)
    if counts != {"original": 16986, "transfer": 16986, "redigital": 16985}:
        raise ValueError(f"Independent RR inventory counts differ: {counts}")
    return (
        images,
        excluded,
        {
            "test_manifest_sha256": file_sha256(manifest),
            "trainval_manifest_sha256": file_sha256(trainval),
            "overlap_audit_sha256": file_sha256(audit),
            "excluded_source_groups": sorted(excluded),
            "scope": "exclude known exact train/val overlaps only; near duplicates unexamined",
            "condition_counts": dict(counts),
        },
    )


def run_provenance(method: str) -> dict:
    values = dict(PROVENANCE[method])
    path = WORK_DIR / values["summary"]
    values["summary_sha256"] = file_sha256(path)
    values["recorded_summary"] = json.loads(path.read_text(encoding="utf-8"))
    return values


def assert_historical_metric_parity(name: str, report: dict) -> None:
    path = (
        WORK_DIR
        / {
            "B-Free": "rr_bfree_evaluation.json",
            "D3": "d3_rr_full.json",
            "Benford-RF": "benford_rr_full.json",
        }[name]
    )
    previous = json.loads(path.read_text(encoding="utf-8"))
    old = (
        previous["exact_overlap_excluded"]["conditions"]
        if name == "B-Free"
        else previous["test"]
    )
    for condition, metrics in report["conditions"].items():
        if metrics != old[condition]["metrics"]:
            raise ValueError(f"Historical metric parity failed: {name}/{condition}")


def evaluate_baselines(expected: list[ExpectedImage], excluded: set[str]) -> dict:
    methods = {}
    for name, (filename, fingerprint) in FILES.items():
        print(f"Audit and aggregate cached {name}", flush=True)
        report, _ = evaluate_cached_method(
            expected,
            [
                CachedRun(
                    WORK_DIR / filename,
                    condition_map={name: name for name in CONDITIONS},
                    expected_sha256=fingerprint,
                    provenance=run_provenance(name),
                )
            ],
            method=name,
            score_kind="margin" if name == "Benford-RF" else "logit",
            excluded_sources=excluded,
        )
        assert_historical_metric_parity(name, report)
        report["historical_metric_parity"] = True
        methods[name] = report
    return methods


def verify_controls(current: dict) -> dict:
    fingerprints = {}
    for control in CONTROLS:
        path = RESULT_ROOT / f"{control}.index.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        if index["config_sha256"] != file_sha256(CONFIG):
            raise ValueError("Control configuration changed after materialization")
        if index["current_index_sha256"] != file_sha256(CURRENT_INDEX):
            raise ValueError(
                "Current simulation index changed after control preparation"
            )
        seen = set()
        for row in index["images"]:
            if row["source"] in seen or row["source"] not in current:
                raise ValueError("Control index contains duplicated/unexpected source")
            seen.add(row["source"])
            if file_sha256(CONTROL_ROOT / row["filename"]) != row["sha256"]:
                raise ValueError(f"Materialized control changed: {row['source']}")
        if seen != current.keys():
            raise ValueError("Control image coverage differs")
        fingerprints[control] = file_sha256(path)
    return fingerprints


def evaluate_propagation() -> dict:
    config, sources, current, roles = load_inputs()
    fingerprints = verify_controls(current)
    expected = expected_images(sources, current, include_controls=True)
    runs = [
        CachedRun(
            WORK_DIR / FILES["B-Free"][0],
            condition_map={"original": "original", "transfer": "released_transfer"},
            input_conditions=CONDITIONS,
            expected_sha256=FILES["B-Free"][1],
            provenance=run_provenance("B-Free"),
        ),
        CachedRun(
            CURRENT_SCORES,
            fixed_condition="current_simulation",
            expected_sha256="edffa3c1ddb357e8be8eed8a3bb786972abeb4976cd6e013d24e70ecdd153b7d",
            provenance={
                "protocol": "legacy class-balanced full-pool independent geometry/JPEG donors"
            },
        ),
    ]
    for control in CONTROLS:
        provenance_path = RESULT_ROOT / f"{control}.run.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance["status"] != "complete"
            or provenance["index_sha256"] != fingerprints[control]
            or provenance["config_sha256"] != file_sha256(CONFIG)
            or provenance["weight_sha256"]
            != "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
        ):
            raise ValueError(
                "Control inference provenance differs from prepared inputs"
            )
        if provenance["manifest_sha256"] != file_sha256(
            RESULT_ROOT / f"{control}.manifest.csv"
        ):
            raise ValueError("Inference manifest fingerprint differs")
        runs.append(
            CachedRun(
                RESULT_ROOT / f"{control}.csv",
                fixed_condition=control,
                expected_sha256=provenance["output_csv_sha256"],
                provenance=provenance,
            )
        )
    report, predictions = evaluate_cached_method(
        expected, runs, method="B-Free", selected_sources=set(current)
    )
    labels = {
        source: Origin.AI if source.startswith("ai/") else Origin.NATURAL
        for source in current
    }
    real = predictions["released_transfer"]
    unchanged = [
        source for source in current if current[source]["geometry_mode"] == "resize"
    ]
    unchanged_score_difference = max(
        abs(
            predictions["current_simulation"][source].score
            - predictions["matched_resize_jpeg"][source].score
        )
        for source in unchanged
    )
    if unchanged_score_difference > 1e-5:
        raise ValueError(
            "Byte-identical resize inputs do not reproduce legacy B-Free scores"
        )
    response = {
        condition: compare_detector_response(
            predictions["original"], real, predictions[condition], labels
        )
        for condition in ("current_simulation", *CONTROLS)
    }
    comparisons = {
        comparator: paired_score_error_gain(
            real,
            predictions["current_simulation"],
            predictions[comparator],
            labels,
            seed=config["bootstrap_seed"],
            replicates=config["bootstrap_replicates"],
        )
        for comparator in CONTROLS
    }
    sha_groups = {}
    for source, group in sources.items():
        sha_groups.setdefault(group["original"]["sha256"], []).append(source)
    within_development = [
        group
        for group in sha_groups.values()
        if sum(source in current for source in group) > 1
    ]
    cross_calibration = [
        group
        for group in sha_groups.values()
        if any(source in current for source in group)
        and any(roles.get(source) == "calibration" for source in group)
    ]
    overlapping_development = {
        source for group in cross_calibration for source in group if source in current
    }
    sensitivity_labels = {
        source: label
        for source, label in labels.items()
        if source not in overlapping_development
    }
    sensitivity = {}
    if overlapping_development:
        for comparator in CONTROLS:
            select = lambda condition: {
                source: predictions[condition][source] for source in sensitivity_labels
            }
            sensitivity[comparator] = paired_score_error_gain(
                select("released_transfer"),
                select("current_simulation"),
                select(comparator),
                sensitivity_labels,
                seed=config["bootstrap_seed"],
                replicates=config["bootstrap_replicates"],
            )
    strata = {}
    for mode in ("resize", "center_crop"):
        subset = {
            source: label
            for source, label in labels.items()
            if current[source]["geometry_mode"] == mode
        }
        selected = lambda condition: {
            source: predictions[condition][source] for source in subset
        }
        strata[mode] = {
            "sources": len(subset),
            "current_vs_matched": paired_score_error_gain(
                selected("released_transfer"),
                selected("current_simulation"),
                selected("matched_resize_jpeg"),
                subset,
                seed=config["bootstrap_seed"],
                replicates=config["bootstrap_replicates"],
            ),
        }
    diagnostics_path = RESULT_ROOT / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics["config_sha256"] != file_sha256(CONFIG):
        raise ValueError("Diagnostic configuration differs")
    feature_rows = {
        condition: {}
        for condition in ("released_transfer", "current_simulation", *CONTROLS)
    }
    for row in diagnostics["measurements"]:
        if "features" in row:
            if row["source"] in feature_rows[row["condition"]]:
                raise ValueError("Duplicated diagnostic source")
            feature_rows[row["condition"]][row["source"]] = row
    if any(rows.keys() != current.keys() for rows in feature_rows.values()):
        raise ValueError("Diagnostic coverage differs")
    distributions = {}
    for condition in ("current_simulation", *CONTROLS):
        real_rows, sim_rows = feature_rows["released_transfer"], feature_rows[condition]
        values = {}
        for feature in next(iter(real_rows.values()))["features"]:
            first = np.sort([row["features"][feature] for row in real_rows.values()])
            second = np.sort([row["features"][feature] for row in sim_rows.values()])
            values[feature] = float(np.abs(first - second).mean())
        distributions[condition] = values
    return {
        "scope": "RR internal development; 500 AI and 500 natural sources; not independent physical or external-platform validation",
        "config": config,
        "config_sha256": file_sha256(CONFIG),
        "control_index_sha256": fingerprints,
        "diagnostics_sha256": file_sha256(diagnostics_path),
        "methods": {"B-Free": report},
        "detector_response": response,
        "current_score_error_gain_over_controls": comparisons,
        "source_split_exact_original_audit": {
            "development_duplicate_original_groups": within_development,
            "calibration_development_duplicate_original_groups": cross_calibration,
            "overlapping_development_sources": sorted(overlapping_development),
            "meaning": "original-file SHA equality only; no near-duplicate claim",
        },
        "sensitivity_excluding_calibration_original_duplicates": sensitivity,
        "geometry_strata": strata,
        "feature_wasserstein": distributions,
        "matched_control_byte_parity": diagnostics["matched_control_byte_parity"],
        "unchanged_resize_score_max_absolute_difference": unchanged_score_difference,
    }


def markdown_report(result: dict) -> str:
    lines = [
        "# RR 统一评测与传播对照",
        "",
        "此报告由缓存逐图分数生成，不启动推理或训练。BA 与真假类正确率使用固定阈值；配对变化为处理后减原图，同源缺失单独记录在 JSON。",
        "",
    ]
    if "baselines" in result:
        lines += [
            "## 全量 baseline",
            "",
            classification_table(result["baselines"]),
            "",
            "三方法均为 50,957 张，逐项历史指标一致。B-Free/D3 为本机 GPU 单图协议；Benford-RF 是 CPU 适配版，不能按毫秒数作同硬件排名。",
            "",
        ]
    if "propagation" in result:
        propagation = result["propagation"]
        lines += [
            "## 同源开发对照",
            "",
            classification_table(propagation["methods"]),
            "",
            "每个条件为相同的 1,000 个开发来源；RR 内部开发证据，不是留出测试或真实物理验证。",
            "",
            "| 模拟 | 对真实分数变化的MAE | Pearson | 真实失败复现召回/精确率 |",
            "|---|---:|---:|---:|",
        ]
        for name, row in propagation["detector_response"].items():
            pearson = (
                f"{row['score_change_pearson']:.4f}"
                if row["score_change_pearson"] is not None
                else "—"
            )
            recall = (
                f"{row['real_failure_recall']:.2%}"
                if row["real_failure_recall"] is not None
                else "—"
            )
            precision = (
                f"{row['simulated_failure_precision']:.2%}"
                if row["simulated_failure_precision"] is not None
                else "—"
            )
            lines.append(
                f"| {name} | {row['score_change_mae']:.4f} | {pearson} | {recall}/{precision} |"
            )
        lines += [
            "",
            "### 当前模拟相对对照的分数误差改善",
            "",
            "正值表示当前模拟的误差更小；区间由真假类别分层、同源配对 bootstrap 计算。",
        ]
        for name, row in propagation["current_score_error_gain_over_controls"].items():
            lines += [
                "",
                f"- 相对 {name}：{row['class_balanced_mae_gain']:+.4f}，95% CI [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]。",
            ]
        overlaps = propagation["source_split_exact_original_audit"]
        lines += [
            "",
            "### 原图字节重复与敏感性",
            "",
            f"开发集内部重复原图组：{len(overlaps['development_duplicate_original_groups'])}；校准/开发跨角色重复组：{len(overlaps['calibration_development_duplicate_original_groups'])}。来源ID不相交并不保证图像字节不重叠。主分析保留预先冻结的1,000来源，另排除跨角色重复的开发来源作敏感性分析。",
        ]
        for name, row in propagation[
            "sensitivity_excluding_calibration_original_duplicates"
        ].items():
            lines += [
                "",
                f"- 排除后 {row['sources']} 来源，相对 {name}：{row['class_balanced_mae_gain']:+.4f}，95% CI [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]。",
            ]
        lines += ["", "### 几何消融与未改动样本核对", ""]
        for mode, row in propagation["geometry_strata"].items():
            gain = row["current_vs_matched"]
            lines.append(
                f"- {mode}（{row['sources']}来源）：相对 matched resize 的误差改善 {gain['class_balanced_mae_gain']:+.4f}，95% CI [{gain['ci95'][0]:+.4f}, {gain['ci95'][1]:+.4f}]。"
            )
        lines += [
            "",
            f"674个未改动来源的新旧 B-Free 分数最大绝对差：{propagation['unchanged_resize_score_max_absolute_difference']:.8f}。",
            "",
            "### 五特征分布距离 W1",
            "",
            "同一列可以比较不同模拟，不能跨单位加总。",
            "",
            "| 模拟 | 亮度相关 | RGB均值变化 | 梯度比 | 宽比例 | 高比例 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, values in propagation["feature_wasserstein"].items():
            numbers = [
                values[key]
                for key in (
                    "coarse_luminance_correlation",
                    "mean_rgb_absolute_change",
                    "coarse_gradient_energy_ratio",
                    "width_ratio",
                    "height_ratio",
                )
            ]
            lines.append(
                f"| {name} | " + " | ".join(f"{v:.5f}" for v in numbers) + " |"
            )
    lines += [
        "",
        "## 证据边界",
        "",
        "- 排除 14 个已知逐字节重叠组；近重复未全面审计。",
        "- RR 再数字化没有逐图物理方式标签；不能把合并结果称为拍屏准确率。",
        "- D3 是 batch1 随机置乱，预训练内容重叠未知；Benford 为论文启发的适配版。",
        "- simulation 只做数字传播对照；固定检测器反应一致不证明物理正确或训练有用。",
        "- 新旧运行时间与缓存状态不同，传播对照的毫秒数是记录值，不能据此认定模拟图令模型加速。",
        "",
        "全部文件指纹、来源覆盖、置信区间、参数及运行协议见同次生成的 JSON。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=("baselines", "propagation", "all"), default="baselines"
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=WORK_DIR / "rr_evaluation_suite"
    )
    args = parser.parse_args()
    expected, excluded, audit = load_expected()
    result = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_audit": audit,
        "entry": "experiments.origin_detection.baselines.evaluate_rr_suite",
        "entry_sha256": file_sha256(Path(__file__)),
        "shared_evaluator_sha256": file_sha256(
            REPO_ROOT / "src/palimpsest/evaluation/cached.py"
        ),
    }
    if args.scope in ("baselines", "all"):
        result["baselines"] = evaluate_baselines(expected, excluded)
    if args.scope in ("propagation", "all"):
        print("Audit and aggregate matched propagation controls", flush=True)
        result["propagation"] = evaluate_propagation()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_prefix.with_suffix(".md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(f"Saved {args.output_prefix}.json and .md", flush=True)


if __name__ == "__main__":
    main()

"""Audit source identity and assign roles; never decode candidate images."""

import csv
import hashlib
from collections import Counter, defaultdict

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT
from .protocol import CONFIG, MANIFEST, PINS, REGISTRY, RESULT, TRAINVAL, read_csv, settings, write_json


def assign_roles(images, registry, trainval, config):
    groups, filenames, digest_sources = defaultdict(dict), set(), defaultdict(set)
    for row in images:
        name, condition, label = row["filename"], row["condition"], row["label"]
        source = f"{label}/{row['source_id']}"
        if label not in ("ai", "real") or condition not in config["conditions"]:
            raise ValueError("Invalid label/condition")
        if name in filenames or condition in groups[source]:
            raise ValueError("Duplicate filename or source/condition")
        if name != f"{condition}/{label}/{name.rsplit('/', 1)[-1]}" or not row["sha256"]:
            raise ValueError("Filename disagrees with label/condition or missing SHA")
        filenames.add(name)
        groups[source][condition] = row
        digest_sources[row["sha256"]].add(source)
    statuses = {}
    for row in registry:
        if row["source"] in statuses or row["source"].split("/", 1)[0] != row["label"]:
            raise ValueError("Duplicate or mislabeled registry source")
        statuses[row["source"]] = row["status"]
    train_hashes = {row["sha256"] for row in trainval}
    direct_overlap = {
        source for source, rows in groups.items()
        if "original" in rows and rows["original"]["sha256"] in train_hashes
    }
    # The old registry intentionally omitted the 14 already excluded sources.
    # Accept only omissions independently explained by original SHA overlap.
    missing_registry = groups.keys() - statuses.keys()
    if statuses.keys() - groups.keys() or not missing_registry <= direct_overlap:
        raise ValueError("Registry/inventory source mismatch")
    statuses.update({s: "excluded_known_trainval_overlap" for s in missing_registry})
    parents = {source: source for source in groups}

    def find(source):
        while parents[source] != source:
            parents[source] = parents[parents[source]]
            source = parents[source]
        return source

    for sharing in digest_sources.values():
        members = sorted(sharing)
        for source in members[1:]:
            left, right = find(members[0]), find(source)
            parents[max(left, right)] = min(left, right)
    components = defaultdict(list)
    for source in groups:
        components[find(source)].append(source)
    excluded, candidates, used_components = {}, defaultdict(list), set()
    for component, members in components.items():
        if len({source.split("/", 1)[0] for source in members}) != 1:
            raise ValueError("Exact duplicate component has conflicting origin labels")
        if set(members) & direct_overlap:
            excluded.update({s: "exact_trainval_overlap_component" for s in members})
            continue
        complete = [s for s in members if set(groups[s]) == set(config["conditions"])]
        for source in set(members) - set(complete):
            excluded[source] = "incomplete_pair"
        eligible = sorted(s for s in complete if statuses[s] == config["development_status"])
        if eligible:
            representative = eligible[0]
            candidates[representative.split("/", 1)[0]].append(representative)
    assignments = {}
    for label in ("ai", "real"):
        ordered = sorted(candidates[label], key=lambda s: hashlib.sha256(
            f"{config['seed']}:{s}".encode()).hexdigest())
        count = config["sources_per_class"]
        if len(ordered) < count:
            raise ValueError(f"Not enough distinct development groups for {label}")
        offset = 0
        for role in ("fit", "selection", "threshold"):
            size = config[f"{role}_per_class"]
            for source in ordered[offset:offset + size]:
                assignments[source] = role
                used_components.add(find(source))
            offset += size
        if offset != count:
            raise ValueError("Configured role counts do not sum to selected count")
    passports = []
    for source in sorted(groups):
        component = find(source)
        role = assignments.get(source)
        if role is None:
            if source in excluded:
                role = "excluded"
            elif component in used_components:
                role = "excluded_duplicate_of_algorithm_development"
            elif statuses[source] == "reserved_simulation_check":
                if any(statuses[s] != "reserved_simulation_check" for s in components[component]):
                    role = "excluded_duplicate_of_historical_exploration"
                else:
                    role = "eligible_internal_holdout_not_opened"
            else:
                role = "unused_historically_exposed"
        passports.append({
            "source_group": source, "label": source.split("/", 1)[0],
            "component": component, "historical_role": statuses[source], "algorithm_role": role,
            "exclusion_reason": excluded.get(source, role if role.startswith("excluded") else ""),
        })
    selected = []
    for source in sorted(assignments):
        for condition in config["conditions"]:
            selected.append({**groups[source][condition], "source_group": source,
                             "algorithm_role": assignments[source], "component": find(source)})
    summary = {
        "source_groups": len(groups), "development_sources": len(assignments),
        "registry_omissions_verified_as_overlap": sorted(missing_registry),
        "development_images": len(selected), "direct_trainval_overlaps": sorted(direct_overlap),
        "duplicate_components": {c: sorted(m) for c, m in components.items() if len(m) > 1},
        "role_counts": dict(Counter(p["algorithm_role"] for p in passports)),
        "development_role_class_counts": dict(Counter(
            f"{role}/{source.split('/', 1)[0]}" for source, role in assignments.items())),
        "near_duplicates": "not exhaustively audited",
        "independence": "post-hoc project-internal roles; no new external blind test",
    }
    return passports, selected, summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if (RESULT / "source_audit.json").exists():
        raise FileExistsError("Source roles already issued; use a new protocol for changed roles")
    for path, fingerprint in PINS.items():
        if file_sha256(path) != fingerprint:
            raise ValueError(f"Historical input changed: {path}")
    passports, selected, summary = assign_roles(
        read_csv(MANIFEST), read_csv(REGISTRY), read_csv(TRAINVAL), settings())
    if len(summary["direct_trainval_overlaps"]) != 14:
        raise ValueError("Historical 14-overlap audit disagrees")
    RESULT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT / "source_roles.csv", passports)
    write_csv(RESULT / "development_images.csv", selected)
    summary["inputs"] = {str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p):
                         file_sha256(p) for p in (CONFIG, MANIFEST, REGISTRY, TRAINVAL)}
    summary["development_manifest_sha256"] = file_sha256(RESULT / "development_images.csv")
    summary["source_roles_sha256"] = file_sha256(RESULT / "source_roles.csv")
    write_json(RESULT / "source_audit.json", summary)
    print(summary["development_role_class_counts"], flush=True)
    print(summary["role_counts"], flush=True)


if __name__ == "__main__":
    main()

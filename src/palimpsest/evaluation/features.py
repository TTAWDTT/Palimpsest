"""Shared feature-cache identity/coverage checks and ordered source-role views."""

import numpy as np

IDENTITY_FIELDS = ("src", "label", "role", "domain", "scene", "condition", "sha256")


def validate_feature_cache(rows, inventory, names, *, variants=("raw",), bounds=(-1, 1)):
    expected = {r["filename"]: r for r in inventory}
    if not expected or len(expected) != len(inventory):
        raise ValueError("Empty/duplicate feature inventory")
    seen, groups = set(), {}
    for r in inventory:
        metadata = tuple(r[k] for k in ("label", "role", "domain", "scene"))
        prior, conditions = groups.setdefault(r["src"], (metadata, set()))
        if prior != metadata or r["condition"] in conditions or r["label"] not in ("REAL", "FAKE"):
            raise ValueError("Conflicting paired source inventory")
        conditions.add(r["condition"])
    for r in rows:
        key = (r["filename"], r["variant"])
        if key in seen or r["filename"] not in expected or r["variant"] not in variants:
            raise ValueError("Unexpected/duplicate feature identity")
        seen.add(key)
        if any(r[k] != expected[r["filename"]][k] for k in IDENTITY_FIELDS):
            raise ValueError("Feature identity differs from inventory")
        values = np.asarray([float(r[n]) for n in names])
        if not np.isfinite(values).all() or np.any(values < bounds[0]) or np.any(values > bounds[1]):
            raise ValueError("Invalid feature values")
    if seen != {(name, variant) for name in expected for variant in variants}:
        raise ValueError("Incomplete feature cache")
    return rows


def feature_views(rows, names, role, *, processed, variant="raw", reference="original"):
    groups = {}
    for r in rows:
        if r["role"] == role and r["variant"] == variant and (r["condition"] != reference) == processed:
            key = f"{r['domain']}/{r['scene']}/{r['condition']}"
            groups.setdefault(key, []).append(r)
    output = {}
    for key, records in sorted(groups.items()):
        records.sort(key=lambda r: r["src"])
        labels = np.array([r["label"] == "FAKE" for r in records], dtype=int)
        if set(labels) != {0, 1} or len({r["src"] for r in records}) != len(records):
            raise ValueError("Each view needs unique sources and both classes")
        output[key] = records, np.asarray([[float(r[n]) for n in names] for r in records]), labels
    return output

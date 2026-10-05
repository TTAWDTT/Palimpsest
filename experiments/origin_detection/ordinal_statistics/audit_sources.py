"""Chimera exact-component role audit for joint development, not a blind test."""

from collections import Counter, defaultdict
import hashlib

from palimpsest.data.source_groups import exact_source_components

CONDITIONS = {"stylegan2_orig": "original", "recap_mac": "mac_iphone", "recap_monitor": "lg_blackfly"}


def chimera_inventory(manifest, image_audit, external_digests, config):
    audits = {r["filename"]: r for r in image_audit}
    if len(audits) != len(image_audit) or len(manifest) != len(audits):
        raise ValueError("Chimera inventory coverage/uniqueness differs")
    grouped, seen = defaultdict(dict), set()
    for row in manifest:
        name = row["filename"]
        if name in seen or name not in audits:
            raise ValueError("Chimera duplicate/missing filename")
        seen.add(name)
        audit = audits[name]
        if any(row[k] != audit[k] for k in ("src", "condition", "label")):
            raise ValueError("Chimera audit identity/label mismatch")
        if row["label"] not in ("FAKE", "REAL") or row["condition"] not in CONDITIONS:
            raise ValueError("Invalid Chimera condition/label")
        source, condition = "chimera/" + row["src"], CONDITIONS[row["condition"]]
        if condition in grouped[source]:
            raise ValueError("Chimera duplicate source condition")
        scene = row["src"].split("/", 1)[0]
        if scene not in ("cat", "church", "horse"):
            raise ValueError("Unregistered Chimera scene")
        grouped[source][condition] = {"filename": name, "src": source, "condition": condition,
                                      "label": row["label"], "scene": scene, "domain": "chimera",
                                      "sha256": audit["sha256"], "width": audit["width"], "height": audit["height"]}
    if seen != set(audits):
        raise ValueError("Missing Chimera audit records")
    for rows in grouped.values():
        if set(rows) != set(CONDITIONS.values()) or len({r["label"] for r in rows.values()}) != 1:
            raise ValueError("Incomplete/conflicting Chimera source")
    components = exact_source_components({s: {r["sha256"] for r in rows.values()} for s, rows in grouped.items()})
    members = defaultdict(list)
    for source, component in components.items():
        members[component].append(source)
    excluded, cells = {}, defaultdict(list)
    for component, sources in members.items():
        firsts = [grouped[s]["original"] for s in sources]
        if len({r["label"] for r in firsts}) != 1:
            raise ValueError("Chimera exact component has conflicting labels")
        reason = None
        if any(r["sha256"] in external_digests for s in sources for r in grouped[s].values()):
            reason = "exact_overlap_with_rr_or_rr_trainval"
        elif len({r["scene"] for r in firsts}) != 1:
            reason = "cross_scene_exact_component"
        if reason:
            excluded.update({s: reason for s in sources})
            continue
        representative = min(sources)
        excluded.update({s: "duplicate_representative_used" for s in sources if s != representative})
        first = grouped[representative]["original"]
        cells[first["scene"], first["label"]].append(representative)
    if len(cells) != 6 or any(len(v) < config["minimum_chimera_sources_per_cell"] for v in cells.values()):
        raise ValueError("Insufficient distinct Chimera sources in registered cells")
    assigned = {}
    for cell, sources in cells.items():
        ordered = sorted(sources, key=lambda s: hashlib.sha256(f"{config['seed']}:{s}".encode()).hexdigest())
        a = int(len(sources) * config["chimera_role_fractions"][0])
        b = a + int(len(sources) * config["chimera_role_fractions"][1])
        for role, subset in zip(("fit", "selection", "threshold"), (ordered[:a], ordered[a:b], ordered[b:])):
            assigned.update({s: role for s in subset})
    inventory = [{**row, "role": assigned[s]} for s in sorted(assigned) for row in grouped[s].values()]
    summary = {"sources": len(grouped), "used_sources": len(assigned), "excluded": excluded,
               "exact_duplicate_components": {c: v for c, v in members.items() if len(v) > 1},
               "role_scene_class_counts": dict(Counter(f"{role}/{grouped[s]['original']['scene']}/{grouped[s]['original']['label']}"
                                                       for s, role in assigned.items())),
               "scope": "Previously exposed Chimera is development; encoded-file identity only; near duplicates unverified"}
    return inventory, summary

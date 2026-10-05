"""Exact encoded-file identity components; not a near-duplicate detector."""

from collections import defaultdict


def exact_source_components(source_digests):
    """Return stable source -> component using any shared digest, transitively."""
    if not source_digests or any(not digests or any(not d for d in digests)
                                 for digests in source_digests.values()):
        raise ValueError("Empty source/digest inventory")
    parents = {source: source for source in source_digests}

    def find(source):
        while parents[source] != source:
            parents[source] = parents[parents[source]]
            source = parents[source]
        return source

    shared = defaultdict(list)
    for source, digests in source_digests.items():
        for digest in set(digests):
            shared[digest].append(source)
    for members in shared.values():
        for source in members[1:]:
            a, b = find(members[0]), find(source)
            parents[max(a, b)] = min(a, b)
    return {source: find(source) for source in sorted(parents)}

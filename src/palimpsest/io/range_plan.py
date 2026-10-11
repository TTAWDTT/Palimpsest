"""Metadata-only HTTP range coalescing; does not fetch or decode files."""


def coalesce_ranges(spans, *, maximum_gap, maximum_span):
    """Merge disjoint inclusive spans under a byte gap and total span cap.

    Rejects duplicate/overlapping spans and individual entries above the cap.
    Output covers every original byte,possibly including bounded extra gaps.
    """
    if (not isinstance(maximum_gap, int) or isinstance(maximum_gap, bool) or maximum_gap < 0
            or not isinstance(maximum_span, int) or isinstance(maximum_span, bool) or maximum_span < 1):
        raise ValueError('Invalid range limits')
    ordered = sorted(spans)
    if not ordered: raise ValueError('Empty byte ranges')
    prior = -1
    for start, stop in ordered:
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in (start, stop))
                or start < 0 or stop < start or start <= prior or stop-start+1 > maximum_span):
            raise ValueError('Invalid/overlapping/oversized byte range')
        prior = stop
    output = []; start, stop = ordered[0]
    for next_start, next_stop in ordered[1:]:
        if next_start-stop-1 <= maximum_gap and next_stop-start+1 <= maximum_span:
            stop = next_stop
        else:
            output.append((start, stop)); start, stop = next_start, next_stop
    output.append((start, stop))
    return output

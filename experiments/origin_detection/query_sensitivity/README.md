# Query perturbation sensitivity: prospective development protocol

Registered before computing sensitivity features or new selection scores.
No goal API, new image collection, encoder training or new layer search.

## Signal and prior work

For each native query independently, use its existing raw and
`jpeg90_444_after_resize256` features. Restore `(unit+1)/2` storage and compute
`1-cosine` at the final768 and block12 LN2 CLS1024 representations. This is two
numbers, not a same-source original/processed pair input. JPEG query formation
includes AREA long-side256 resizing without upsampling, followed by Pillow
Q90/4:4:4 encoding. It does not reproduce the new paper's defocus perturbation
or its optimal-layer search. RIGID/MINDER and intermediate-layer sensitivity
are prior work; RINE and our earlier fixed block12 experiment are already
recorded. Reusing the latter cache avoids another GPU extraction.

The raw query may itself be a published transfer or recapture image. The
perturbed image is formed from that query. Native source/condition labels
only organize fitting/evaluation and are never detector features.

## Roles, fixed methods and scope

Use only the earlier 6300 native images / 2100 development sources, with
unchanged fit1260 / threshold420 / selection420 sources. Each source has
three native conditions. Two numeric features per native image:
fit3780, threshold1260, selection1260 rows. Four fixed heads: average
logistic loss (temperature0), same-source soft maximum (temperature.1),
the latter with existing label-compatible wrong-source grouping, and
balanced source-null labels. Ridge.01, scale floor.001, 500 iterations,
gradient tolerance1e-5; no parameter/layer selection. Reuse the existing
source-risk solver and class-aware global threshold policy. Null fitting
uses fit-only pseudo labels; final threshold labels remain true, so this
is a pipeline control, not a deployment permutation test.

Evaluate native raw conditions only: 15 BA/AUC views including scene and
domain aggregates, and 10 matched processing pairs, each head. No additional
Q70/Q60 query robustness is tested: the existing cache lacks perturbations
of those queries. Scores cannot be compared as an identical panel with
the previous60-view/130-pair experiments, nor satisfy the full acceptance
criterion by omitting harder query conditions. RR and Chimera are exposed
development; independent pixels and RR reserved remain sealed.

## Gates and receipts

Before fitting: parent CSV/receipt/source pins, complete filename/variant
identity, same-query metadata, finite nondegenerate embeddings, known-angle
and wrong-shift plant controls. Streaming preparation writes only two
features per native query and hashes; it does not rerun a baseline. Two-size
same-code preparation pilot and fold-size fitting pilot establish conditional
cost before a full run. Attached sessions only; output progress is completed
numeric rows. Freeze code after executing pins.

Before reporting: role counts and source disjointness, saved-rule/native
score equality, unique finite selection scores, exact integer/Fraction BA
and matched changes, source bootstrap with the existing seed/repetitions.
Report worst class changes and pair support. AUC or small drift alone cannot
substitute for accuracy or processing stability. Receipt transcription and
negative controls are software checks, not independent device evidence.

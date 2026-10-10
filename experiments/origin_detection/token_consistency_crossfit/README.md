# Token statistics + source consistency — prospective iteration53

Registered after iteration52 and before this calculation. The unchanged frozen
PECore/CuRe tokens give clipped-center+normalized-RMS2048 features. Iteration52
mixed/source minimum domain BA85.56%, worst decline5.56pp; mixed/mean86.11%/5pp.
Test whether the added class information supports the established source-score
variance penalty without losing BA. This combines existing representation and
regularization; no new invariance theorem, encoder training, or new pixels.

Use the signed52 numeric cache, its exact role/variant records and null mapping.
1260fit/420threshold/420selection sources, raw/Q90 only in fit. Five fit-only
source folds from stage50, seed20261007 within true class/domain/scene. All six
views of a source stay together. Cache loaded as numeric rows; moments re-fit
inside each train fold. Fixed strengths0,.1,1,10; source temperature.1,ridge.01,
scale-floor.001,LBFGS500,gradient<=1e-5,CPU BLAS one thread.

Reuse exact stage50 OOF selector: all30 scope BA>=80%, minimize worst of35
declines, then maximize BA, then smallest strength. If infeasible, highest min
BA fallback, not acceptance. Evaluate all strengths for truth and signed30
balanced source-null separately, with pseudo labels for null OOF evaluation.
True labels stratify folds. Final outer threshold uses unchanged24 true-label
weak views in both paths; null is not a permutation p-value.

Four final rules: zero reference, selected genuine source, selected wrong
consistency groups (classification sources stay genuine), selected source-null.
Zero must reproduce all5040 signed52 mixed/source decisions/margins to1e-12
before any nonzero outer score is evaluated. No selection-driven strength scan.
Final60groups/130pairs plus raw processed null AUC intervals; 80% and2pp user
criterion remains unmet until demonstrated with independent real data.

Before real fit: shared variance/gradient and source-fold tests, new artificial
CV score/1/8-decline control, numeric group merger rejection. Then measure one
actual6048x2048 train fold at strength1, without looking at its held scores.
Record duration and project40fits at identical fold sizes; initial fit used
only for cost, not method choice. Estimate is conditional on strength-dependent
solver iterations; if pilot>120s, do not launch the40fit grid and restructure.
Record progress in completed strength OOF artifacts (five fits each). Workers remain attached.

Pin code/controls/cache before fitting; preserve failures. Final integer/Fraction
and OOF audit use existing standalone tools, wrong BA/score/fold negative controls,
emitall plus wrong quote. Never call these checks independent scientific evidence:
same encoder/data/selection have been repeatedly exposed. Existing near-content
candidate and pretraining-overlap uncertainty remain. External pixels stay sealed.

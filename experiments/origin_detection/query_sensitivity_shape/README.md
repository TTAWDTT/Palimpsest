# Fit-only shape diagnostic, before new outer scores

The two sensitivity coordinates from query_sensitivity are fixed and immutable.
Current linear heads failed absolute BA and processing stability. Test whether
a conventional nonlinear boundary merits further work; do not acquire features
or evaluate threshold/selection sources here. This is not a new invariance
principle or a performance claim for the full detector.

Keep the five existing deterministic source folds (source_folds default seed
20261007). Each fold trains on four fit-source folds (1008 sources/3024 native
rows) and scores the held fit-source fold (252 sources/756 native rows).
No threshold calibration is performed: only pooled held-out ranking/AUC is
diagnosed. Original, transfer/recapture condition labels never enter a model.

Fixed comparison: source-risk linear temperature.1/ridge.01 versus existing
source-level bootstrapped ExtraTrees (64 trees/depth6/minimum12 records/seed
20261006) and the same trees with row-level bootstrap. Native exported scores
must match sklearn. No hyperparameter, feature, perturbation or layer sweep.
True and balanced source-null fits both get their own held labels; null OOF
evaluates pseudo labels, unlike the previous outer true-label calibration.
Fit-only null assignment is generated once and reused without external labels.

Before main computation: existing known solver/forest tests and same full-fold
pilot for all three truth readouts; condition budgets cover 30 fits, exclude
input/tests/serialization, and do not imply image inference speed. Attached
sessions only, completed fold/model outputs are progress. All 3780 held scores
per model are saved with source-fold IDs and original fit roles. No new selection
scores or deployment head follows automatically; AUC cannot accept80%/2pp.

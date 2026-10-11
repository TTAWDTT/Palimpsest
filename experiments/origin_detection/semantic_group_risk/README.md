# Semantic group risk — prospective iteration37

Register before fitting. Iteration17 frozen CLIP source risk remains the high
aggregate-BA development reference, but declines8.75pp. Iteration13 evaluated
entropy group risk and paired score penalties on weaker499 hand-written features;
that failure does not establish the same outcome on the768 CLIP representation.
This is reuse of established objectives, not a novel causal-invariance theorem.

Reuse signed CLIP pixels/vectors; no new encoder, extraction, simulation, final
neural training, external pixels or RR reserved. Same1260fit/420threshold/420selection
sources and weakraw/Q90 fitting. Explicit fit traversal checked against the shared
weighted source matrix. Domainhalf/equal source/view masses; fit-only standardization.

Four registered objectives: mean, group, pair, both. Same public group_margin
implementation: ridge.01, scale-floor.001, LBFGS500 gradient<=1e-5, group temperature
.1 when active and squared same-source score penalty strength.1 when active.
Groups are domain/scene/condition/encoding/true class48; no cell selection. Pairs
anchor original/raw to the other five weak views of each source,6300fit pairs.
Temperature0/pair0 reproduces the earlier mean objective. Require all5040 selection
decisions to match the signed iteration17 weak/mean reference before continuing;
measure scalar discrepancy too, but do not require bit-exact solver coefficients.

Both-objective controls: wrong-source circular matches in the same class/domain/
scene/condition/encoding (reuse signed prior helper), and exactly balanced stage30
pseudo-label assignment with genuine source pairs. Report four raw processed AUC
intervals for the pseudo-label control; above-chance intervals stay explicit.
Calibrate only true threshold-role labels on24weak views, including null rules.

First rerun known shortcut reversal, analytic-vs-finite-difference gradient,
constant-log2 and serialization controls in tests/detection/test_group_margin.py.
Coverage/finite/signed-cache checks precede the six heads. Evaluate60metric cells
and130same-source comparisons per head; separate integer audit with deliberately
corrupted BA and selected-number transcription controls follow. Startup-free live
decode/encoder/readout timing only if a candidate warrants it. No accuracy or
stability claim from optimizer success. Exposed development; bootstrap extrema
intervals selected after exposure, not simultaneous or final validation.

Run: `python -m experiments.origin_detection.semantic_group_risk.run_iteration`.
Outputs: `work/robust_statistics/semantic_group_risk`; refuse overwrite.

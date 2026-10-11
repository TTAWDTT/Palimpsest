# Frozen semantic class moments — prospective iteration38

Register before moment fitting. Iteration36 full/source improves Chimera
minimum scene BA to95%, but RR minimum domain81.67% and worst drop5.56pp
still fail stability; its native E2E P95<=2MP is69.41ms. Iteration17 CLIP
remains a faster high-aggregate reference. Test whether class covariance
geometry retains useful signal missed by a fitted separating plane.

Two fixed representations: original signed CLIP768 and released adapted
CuRe raw patchmean1024. No representation normalization changes, no
encoder fine-tuning, new pixels, simulation or source-conditioned inference.
For each: pooled, diagonal and class_full Gaussian moments, plus balanced
class_full label-null matching iteration30. Eight registered heads, not a
post-hoc oracle choice. Classical heads credited to discriminant analysis
and Kotyan et al arxiv2608.18523v1; no novelty or paper replication claim.
Our support, weighting, regularization and calibration differ from that paper.

Use only weak raw/Q90 fit7560 rows,1260 sources, domains half mass then
equal sources/views. Fit-only standardization floor.001, covariance ridge.1,
shrinkage.5, priors.5, same public GaussianRule implementation. Threshold
uses420 source/2520 rows in24 weak groups; selection420source5040 rows,
60 metrics130 paired changes. Raw/Q90/Q70/Q60; seed20261006;2000 source
bootstrap repetitions. Same inventory2100 sources6300 files; CLIP parent
20160 and CuRe15120 rows must pass existing signed coverage audits.
No strength/ridge/dimension/threshold search or concatenation. RR reserved
and independent external pixels remain sealed. Exposed RR/Chimera development.

Before fit, run existing planted equal-mean/different-variance density
oracle and serialization/invalid-state controls. Gaussian output must
match single-row scoring exactly; inverses residual<=1e-7, positive
covariance and finite scores. Balanced assignments must exactly match
stage30. Preserve four processed-raw AUC intervals for each null including
any interval above.5. Integer/Fraction audit + changed BA refusal and
emitall transcription + wrong twin afterwards. These share input margins;
they do not establish independent scientific robustness. Selected worst
pair CIs are post-hoc and not multiplicity-adjusted. Live speed only if
a new fixed candidate warrants it. Goal stays OPEN if accuracy, stability,
speed or independent real evidence is absent.

Run `python -m experiments.origin_detection.semantic_gaussian.run_iteration`.

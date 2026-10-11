# Token nonlinear source readout — prospective iteration55

Registered after54: mixed token linear source consistency reaches minBA84.44%,
max decline3.75pp. Exact finite threshold oracle finds no80%/2pp common state.
Test whether nonlinear boundaries can preserve extra token class information
with source variance regularization. Reuse known random Fourier Gaussian mapping
from14/19, not a new kernel principle. No inference of physical invariance.

Signed52 input is center+normalized RMS2048; append fixed512 RFF coordinates
(256 Gaussian frequencies,seed20261006,gamma1/2048), head2560. Mean/scales for
mapper learned solely within each CV training fold from its source weights;
classification moments also fold-local. Mapper never sees held-source features
for fitting, or threshold/selection moments. Fullfit mapper then frozen.
Ridge on re-standardized coordinates is not the original Gaussian RKHS norm.

Reuse exact source CV five folds and strengths0,.1,1,10,source temperature.1,
ridge.01,scale-floor.001,LBFGS2000,gradient1e-5. Full truth and signed balanced
source-null grids, no reuse of linear54 OOF scores.30 OOF BA groups/35 declines,
same80% feasibility-first selector. Truth labels only stratify folds, null OOF
uses pseudo labels. Outer24threshold views still use real labels for both.
Four final heads zero/selected/wrong consistency source/source-null,60groups
and130paired comparisons per head. Genuine classification sources unchanged.

Zero is a nonlinear baseline, so it should not match linear54. Its portable
scores must exactly equal its mapped head on all5040selection rows; save/load
checks before real execution. Single/batch equality enforced by score_rule.
No bandwidth/count/seed/strength scan after outer results. No extra encoder
passes, new original pixels, RRreserved or independent data unsealed.

Before real work: known same-first/second-moment nonlinear classification,
mapper train-fold moments/unseen extreme input no mutation, head calibration,
portable load/exact scores; shared kernel approximation and source CV controls.
Then actual6048x2048 fold0 strength10 costpilot, no held/outer scoring. If>120s
or fails numerical gate, restructure before full40CVfits. Projection40*measured
fit assumes same dimensions/count,conditional on solver iterations; loading,
four final fits and evaluation excluded. Attached worker, progress stored per
completed strength, preserve failure and signed source. Run lint before sealing.

Final standalone OOF/Fraction, altered fold/score/BA, emitall/wrong quote and null
intervals. Independent validation absent until a fixed candidate actually passes;
near-content and pretraining uncertainties retained. Goal criterion unchanged.

# Radial conditional score calibration — prospective iteration48

Register before real fitting. QuAD (Guillaro et al., CVPRW2026, main10-page
text plus formulas/tables and five calibration scripts read) uses measured
IQA-conditioned Gaussian detector-score distributions and sums retrieved
near-duplicate evidence. Its multi-image retrieval input differs from our
single-image requirement. We test only the conditional scalar density idea.
Encoder log norm is an observed covariate, NOT validated IQA or a calibrated
physical degradation parameter. This is not a QuAD reproduction.

Use signed36 complete1024 CuRe vectors and its frozen full/source and
full/source_null StableRules. No encoder/head refitting. The scalar is saved
rule.score minus its existing threshold; second covariate log L2 norm.
All normalizations estimated on fit only (weighted .001 scale floor).
Fit class-conditional mean a*q+b and log variance alpha*q+beta by weighted
Gaussian NLL plus1e-4 coefficient ridge; log-var clipped[-30,30] with
matching derivative, LBFGS300, gradient<=1e-5. No hyperparameter scan.
Base head already saw these fit sources; scalar MLE is in-sample calibration,
not cross-fitting. Selection remains repeated exposed development.

Four controls: no-radius constant Gaussian; genuine radial conditional;
wrong_radius conditional (shuffle fit q within domain, scene, true class,
condition, variant with seed20261006; query q remains actual); source_null
conditional on signed30 pseudo-labels and36 pseudo-label head. Shuffling
preserves broad class/processing marginals, breaks exact paired scalar/radius
associations; not a permutation p-value or proof of physical causation.

Same1260/420/420 source roles, fit7560 raw/Q90, threshold2520/24views,
selection5040/four variants, 60 metric groups/130 comparisons. Threshold
class_threshold fixed before results, strict positive marginFAKE, tiesREAL.
Single query takes only full feature vector and fixed rule, no source or
condition, no retrieval or target batch adaptation. Query radius and scalar
cannot recreate missing information; Gaussian tails may extrapolate poorly.

Before real fit: analytical equal-variance Gaussian LLR; finite-difference
NLL gradients including clipping; exact single/batch/save-load; tiny known
labels fitted; invalid dimensions/norm/NaN/missing class and planted wrong
LLR refused. Pin implementation and controls before computation. Signed36
code/cache/head SHA and signed30 assignment must match. After complete:
integer/Fraction and corrupt BA audit, emitall/wrong quote; all four null
raw AUC intervals retained. No selective scene or threshold adoption.

Goal BA>=80%, decline<=2pp, independent real evidence remains OPEN. No new
external pixels, RR reserved use, final encoder training or live GPU worker.
Append results, do not rewrite this signed protocol after fitting.

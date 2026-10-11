# Angular versus radial CuRe information — prospective iteration42

Register before drift statistics or head fitting. AIDGN (Jin et al.,
arXiv2210.15836v1) assumes class-conditional angular invariance and models
norm shift; its trained backbone and norm-dependent objective differ from
our frozen readout. L2 normalization alone is not an AIDGN reproduction.
Read main pp1–6 and reference context p7; proofs/supplement pp8–15 remain
reading debt. No physical angular invariance theorem is asserted.

Use only signed iteration36 full1024 CuRe features, weak raw/Q90 fit and
calibration, four selection variants. No new encoder/pixels or stronger
support from iteration41. Two fixed query maps: unit direction f/||f||;
unit direction plus log(||f||). Positive global feature rescaling is removed
only by the first map; adding radius explicitly abandons this invariance.
Nonzero finite vectors required. No epsilon, clipping or parameter search.
Single image only; no inference access to source/condition labels.

Six fixed heads: each map sourceT.1, meanT0, balanced source_null following
stage30. Ridge.01, scale_floor.001, source equal/domain half/view equal;
LBFGS500/gradient<=1e-5. Fit7560, threshold2520/24groups, selection5040;
1260/420/420 sources unchanged. Seed20261006/bootstrap2000. Compare signed
iteration36 raw source/mean without refitting. Repeated exposed development,
not independent test. Neither RR reserved nor external images opened.

Before real arrays: plants (3,4)->(.6,.8), positive scale1/8,1,8 unchanged;
negative scale reverses direction; radius changes by log8; exact single/
batch/save-load; invalid zero/NaN/dimension rejected; altered coordinate
detected; planted classifier labels recovered through map+same source fitter.
Fit-only drift audit uses same source/condition raw/Q90 pairs with original
raw reference, records norm ratio, cosine, best scalar and nonradial residual;
known radial/orthogonal examples and conflicting pair keys must pass first.
These diagnostics do not choose maps or hyperparameters.

Controls saved before real computation with pinned implementation/tests.
Full signed input coverage and exact mapped traversal required; balanced
assignment hash unchanged. Preserve all four null raw-processed AUC CIs,
integer/Fraction audit and deliberately wrong BA, emitall/wrong quote twin.
Selected-worst CIs uncorrected. Live timing only if candidate warrants it.
Goal stays OPEN without adequate absolute ability, little decline, speed
and independent real-processing evidence.

Run `audit_drift`, then `run_iteration` in CPU runtime. Mapping is shared in
`palimpsest.detection.representations.angular_features`; no changes to
already signed feature extraction directories.

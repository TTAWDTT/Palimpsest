# Quadratic source-centroid modes with source risk — prospective iteration39

Register before basis fitting. Gaussian moment rules in iteration38 printed
CLIP pooled/QDA12.5pp drop and CuRe pooled12.5pp/QDA10pp. Do not tune their
covariance/ridge to the selected failures. Instead separate fixed fit-only
second-order signal extraction from a discriminative source-view objective.
Classical whitening/eigenmodes/quadratic expansions are not claimed novel.

Two signed representations: CLIP768 and CuRe1024 raw full. Fit-only global
weighted center/scale floor.001. Average each source's six weak raw/Q90
views to source centroids; class covariances have equal class priors and
source masses. Whiten pooled centroid covariance with shrinkage.5/ridge.1,
retain32 eigenvectors of largest absolute whitened covariance difference.
Fix signs by largest-magnitude coordinate. No dimension, spectrum, covariance,
normalization or ridge search. Query squared projections plus unchanged
raw original vector; inference uses one file, no source ID or paired input.

For each representation: full+quadratic sourceT.1; full+quadratic meanT0;
quadratic-only sourceT.1; full+quadratic balanced-null sourceT.1 (map AND
head fit on matching stage30 pseudo labels). Eight heads. Reuse existing
source-risk fitter/ridge.01 and fixed threshold class-min policy.
1260fit sources7560 weak rows,420threshold2520weak rows24groups,
420selection5040views60metrics130same-source changes. Variants raw/Q90/Q70/Q60;
domains half/source equal/view equal; seed20261006/bootstrap2000. Source-centroid
fit averaging does not consume paired inputs at deployment. Exposed RR/Chimera;
no new pixels/encoder training, RR reserved or independent external reads.

Before scientific fitting: planted equal-mean/different-variance axis must
retain squared signal with exact16 ratio; repeated fit views source count4;
constant axis zero; single/batch and serialization exact; wrong direction
changes output and conflicting source labels/NaN/dimension counts refused.
Whitening residual<=1e-7 and positive covariance. Signed caches audited;
selection feature transforms must repeat exactly. Eight heads integer/Fraction
audit and changedBA refusal, quote arithmetic+wrong twin after completion.
Four raw processed AUC intervals per null, anomalies retained. Selected
worstpair CIs uncorrected. Fit/transform cost is not native latency; live
speed only if worthwhile. Accuracy, stability, speed and independent real
evidence jointly required; goal remains OPEN otherwise.

Run `python -m experiments.origin_detection.semantic_covariance_risk.run_iteration`.

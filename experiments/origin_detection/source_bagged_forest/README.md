# Source-bagged randomized forest — prospective iteration 34

Register before any fit or selection. Test nonlinear local partitions on the
unchanged signed CLIP-L+DINO1536 cache. This is a conventional trained forest
on frozen neural features, not a hand-crafted-only algorithm, new tree principle,
physical inverse or claim of certified propagation invariance.

Fixed comparison: 128 ExtraTreeClassifier trees, gini, max_depth12,
min_samples_leaf12 records, max_features sqrt, seed20261006, serial fitting.
Source-bagged samples 1260 source IDs with replacement within domain/class,
keeping all six original/processed raw/Q90 views of each selected source.
Row-bagged control instead resamples individual records within domain/class.
Each tree has exactly .5 total sample weight per domain, equal class/domain
mass; source multiplicity is identical on all source views only for source bagging.
Class labels/domains/source IDs are FIT-only. No condition gate at query.
A balanced-source-null source-bagged forest repeats the entire fitting pipeline
with stage30 truth-orthogonal source sham labels. No depth/tree/leaf/seed grid.
The 12-record leaf setting is NOT a minimum of 12 independent sources.

All weakfit data1260sources7560views, threshold420sources24weak groups,
selection420sources5040scores60groups130pairs. Same global threshold maximizes
worst class accuracy, calibrated separately on true threshold labels for every
candidate including null. Four raw processed null AUC CIs are recorded even
when anomalies occur. Bootstrap is model construction, not a p-value or external
validation. Same exposed development scope; no RRreserved/external pixels.

Save plain JSON node arrays/probabilities; no pickle. Export inference casts
query to float32 exactly as sklearn trees; verify serial exported probabilities
against native predict_proba before selection, error <=1e-15. Batch/single exact,
known finite tree boundary/leaf arithmetic, XOR forest recovery, source versus
row bootstrap weight invariants, invalid/conflicting labels and serialization
controls required before run. Integer confusion counts with BA+0.01 rejection
and emitall wrong quote after run. Only live speed if accuracy/decline warrant.

Prior: author workshop Extra-Trees2006 algorithm and averaging, not original
40page article (record fetch timed out). Clustered RF2026 main1–10partial reading
emphasizes independent groups, honest disjoint split/evaluation/correlation data,
weighted least squares and strong smoothness/density assumptions. We implement
none of those covariance weights or honest theorem; only source bootstrap.
No test covariate adaptation, no theorem import. Candidate failure keeps goal open.

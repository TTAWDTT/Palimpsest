# CuRe source-consistency cross-fit — prospective iteration50

Register before CV or selection. Iteration49 fixed class-group paired loss
still drops5pp. Stage21 fixed source-score variance strengths on CLIP; it did
not optimize deployment via fit-only source CV. Test the existing convex
source-variance objective on full CuRe with an explicit fit-only selector.
This is algorithm development using established regularization/CV principles,
not a new loss or physical invariance guarantee. No blur or encoder training.

Signed36 full1024 vectors, same1260fit/420threshold/420selection sources.
Five folds by SHA order seed20261007 within original domain/scene/true class;
all six weak views of a source stay together. Fold assignment is immutable
and shared by real and pseudo-label paths. Each train fold has432RR/576Chimera
sources, holds108RR/144Chimera. No threshold or selection record enters CV.

Fixed strengths0,.1,1,10; fit_consistent_source_risk reused, source temperature.1,
ridge.01,scale-floor.001,LBFGS500,gradient<=1e-5. All15 same-source weak-view
differences represented by its covariance, no directional/basis truncation.
CV predictions at fixed zero threshold, one held-source prediction/row, pool
all7560 OOF records. Exact30BA groups/35 physical-or-encoding comparisons.
Among strengths whose minimum all-scope OOFBA>=80%, choose smallest worst
paired drop, then higher BA, then smaller strength. If none, highest minBA,
then decline, then smaller strength; fallback does not mean target passed.
This weak-view CV cannot certify new Q70/Q60 or device generalization.

Run this entire selection for truth and signed30 pseudo-label targets;
pseudo-label CV uses pseudo labels. True source labels only stratify folds.
Then fit chosen strength on allfit; final class_threshold uses unchanged true
threshold-role labels/24views for both paths. Null label code/hash immutable.
Final60metrics/130pairs on selection computed only after strength frozen.
Four final records: zero reference, chosen real, chosen real with wrong
consistency groups, chosen pseudo-label. Wrong groups reuse fixed47 helper,
only variance grouping changes; source classification grouping stays genuine.
Also evaluate fullfit strength0 and require all5040 decisions match36full/source
before continuing. Save all OOF decisions/fold/strength artifacts, not just winner.

Before real CV: known source-fold order invariance/no leakage, role/duplicate
refusal, exact1/8 planted paired decline, collapse-rejecting selector, and
existing variance/gradient/zero-reference tests. Pin code and controls before
real calculation; pilot synthetic1024 fit for budget before40CVfits. No
post-selection strength scan, no independent data or RR reserved opened.
Current fit/threshold near-content candidate remains disclosed, no role edits.
Integer/Fraction,wrongBA,all4null raw AUC intervals,emitall/wrongquote after
complete. Live image timing only if fixed candidate warrants it. Final goal
BA>=80%,decline<=2pp,independent real validation remains OPEN.

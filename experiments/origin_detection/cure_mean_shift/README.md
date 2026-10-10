# Class-conditional mean shifts — prospective iteration43

Register before any real mean/projection statistics. Read ISR ICML2022
(Wang et al., PMLR162:23018–23033) main pp1–9, proof/implementation pp13–16,
references11–12 context; p10 references owed. ISR-Mean projects out varying
class-conditional environmental means. Infinite samples, linear injective
mixing, identifiable environment variation are theorem conditions. Our
paired finite processing means do not establish these assumptions. Not paper
reproduction or a new invariant subspace principle. Earlier stage20 removed
individual pair covariance directions in CLIP64; this retains full CuRe1024
except at most40 conditional-mean directions and does not equate the methods.

Same signed iteration36 full1024, raw/Q90 weak support only. Fit standardize
weighted mean/SD floor.001. Within domain/scene/class, each of six processing/
encoding views has the same fit-source set. Subtract original raw class mean
from other five means; four domain/scene strata×two classes×five shifts=40.
SVD of40×1024 matrix, remove row space with relative singular floor1e-8.
Refuse empty retained space or failed orthogonality/projection constraints.
No dimension/threshold/strength search. Zero shift is identity (not refusal).
This enforces finite estimated mean equalities, not per-image invariance.

Four fixed heads: honest sourceT.1 and meanT0; honest-map source_null where
both mean fitting and head labels use balanced stage30 source assignment;
wrong-processing source control. Wrong processing randomly permutes six
view names independently per source within domain/scene/class, seed20261006,
preserving one of every view per source and the full classification rows.
Only the projected-shift estimator sees corrupted view names; actual risk
source groups/threshold labels remain honest. A mere permutation of whole
groups would leave the row space unchanged and is not a useful corruption.

Ridge.01, source/domain/view weights unchanged, LBFGS500/gradient<=1e-5;
1260/420/420 sources,7560 fit,2520 threshold24groups,5040 selection4variants.
Bootstrap2000. No new pixels/encoder training; RRreserved remains sealed.
Fit-only construction; source/scene/condition never query inputs.

Before real data: planted stable class axis and varying nuisance axis;
projection removes nuisance, retains planted class and allows same fitter to
recover labels; signal-aligned shift demonstrates projection may lose class;
zero-shift identity, coordinate-swapped corruption caught, invalid values and
incomplete/conflicting source panels refused, serialization/single/batch exact.
Class-mean residual/orthogonality<=1e-8. Head+map collapse into native1024
linear rule requires margin error<=1e-9 and all decisions identical. Paired
means do not exploit pairing beyond completeness; no claim of individual
source preservation. Software receipts pinned before real run.

Integer/Fraction and BA+.01 refusal, null four raw AUC intervals, emitall and
wrong quote twin preserved. Exposed development, selected-worst CI uncorrected;
live speed only for warranted fixed candidate. Goal OPEN until adequate
absolute ability, almost no decline, live speed and independent real evidence.

# Paired Gaussian kernel — prospective iteration45

Registered before new fit statistics/heads. The released B-Free iteration44 is
still running. This separate CPU experiment reuses only signed CLIP stage16
features;no new neural inference,external pixels or simulation.

Question: do local nonlinear scores plus matched-source smoothness preserve
class separation better than global linear consistency (21) or global deletion
(20/43)? Manifold regularization is known prior art,not a new principle;its
distribution/geodesic assumptions do not certify screen-camera invariance.
Our fully supervised true-source graph and restricted source-centroid span
differ from the author's semi-supervised neighbor graph/full representer.

Same1260fit/420threshold/420selection sources. Weak raw/Q90 six views/source,
7560/2520/5040 records,24 calibration views,60 metrics/130 pairs. Unit CLIP
stored coordinates unchanged. All1260 fit-source six-view centroids are anchors;
Gaussian k=exp(-squared_distance/(2*width));width is the median strictly positive
unordered squared anchor distance (reject nonpositive/nonfinite median).
No width/anchor/encoder/ridge sweep;selection contributes no anchor or scale.

Objective: equal-domain/source/view weighted squared class-target error,
targets−1/+1;plus strength times weighted within-source score variance;plus
.01*alpha'Kanchors*alpha +1e-6*||alpha||². Unpenalized intercept;weighted
centering eliminates it. Fixed strength0 and1,one balanced source_null at1,
one wrong-source variance at1. Wrong-source independently permutes IDs for
each weak view within domain/scene/class;true labels/classification weights
and anchor construction stay unchanged. Null changes only fit labels using
the signed stage30 map;threshold labels are real as in existing controls.

Before real features: known Fraction solution in a two-anchor four-record
panel,variance shrinkage vs0,finite/complete-class source refusal;known Gaussian
distances;portable load and single/batch exact margins;changed scalar gate.
Software receipt and source/code pins required before cache opening. Solve
symmetric positive-definite system;relative residual<=1e-8. No posthoc tolerance
change. All queries receive only final feature vector,never source/scene/view.

Use existing fixed class threshold24 and source bootstrap2000,seed20261006.
Save three genuine/zero/null rules and wrong control,full scores and hashes,
integer/Fraction audit and wrongBA,emitall/wrong twin. Exact live readout/cache
and decode-inclusive timing only if worthwhile development gates pass.
No claim of kernel novelty,physical guarantee or final neural/non-neural purity.
User criterion BA>=80%,drop<=2pp,independent real validation remains OPEN.

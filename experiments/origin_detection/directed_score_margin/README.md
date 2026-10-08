# Processing-order signed margin: prospective comparison

Registered after67 failed and before this real-data fit. Reuse complete3920
features, all source roles and the same five directions/20 terms. Change only
the last readout penalty: source entropy-smoothed logistic risk/ridge plus
lambda times mean squared positive `y_signed*(f_before-f_after)` on declared
processing edges. Pairwise difference/ranking and squared hinge are prior
methods, e.g. [Ranking SVM](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf)
section4.1; this is a task-specific adaptation, not a new ranking principle.

## Pair graph and scope

Current training-fold records explicitly provide domain/scene/source/condition/
variant; prediction receives no records/original image/process label. Graph:
original to each of2 physical/released processed conditions at each of3 fit
variants, plus weaker to stronger encoding in each of3 conditions. This is
15 edges/source,11340 edges for756-source inner train and18900 forfull1260.
Each source has node-weight mass divided equally over its edges, totaledge
weight1. Labels for loss come from the supplied training target, including
source-null, not secretly from true labels stored in metadata.

Wrong-source control uses the already registered within-domain/scene/true-class/
condition/variant source permutation as virtual pair groups; source risk,
banks and moments keep the true source groups. Cross-source edge count must
be recorded. It is not a null wrong-label training model.

If all positive directed score violations are zero on known same-label edges,
a common fixed threshold cannot turn a correct before decision into a wrong
after decision, including the REAL tie convention. Finite-penalty optimization
does not ensure zero loss, and arbitrary small mean loss can still flip every
near-threshold case. This property covers known scored edges, not unseen
physical image channels or fixed accuracy80%. Keep the absoluteBA gate and
independent real-data requirement. No new simulation/encoder training/pixels.

## Procedure, versions and output

Seven/extra known answer checks cover one-sided improvement versus degradation,
tiny-loss flip counterexample, analytic gradient, zero-strength equality,
old-prefix-safe bank construction, serialization, and training-context row
alignment/exclusion. Contextual source-panel dispatch adds an explicit opt-in;
ordinary methods keep numeric signatures. Historical64-67 source audits must
use their executed Git version, as recorded in the interface revision ledger.

Run current controls, real6804/2268/2268 cold/warm pilot, then fixed lambda
0/.1/1/10 whole source-held calibration CV, and four outer heads with60/130
metrics/pairs. Prior source masks and seed/null assignment stay fixed. Pilot
<=120s percall, conditional40CV budget recorded; rates/parameter costs variable.
Shared driver handles roles, calibration, portable scores and receipts; no
forked evaluation loop. Zero head must match67 zero scores if numerical
identity holds; otherwise investigate before claiming a matched objective
comparison. Existing QuantileScoreRule kind denotes serialization layout,
the recorded readout method identifies this different training penalty.

The bank construction path also fits an unused zero-strength source head;
initialization calls, basis/LP/aux/map and actual directed readout calls are
separately reported. Baseline and nonzero penalty are separate optimizations.
Outputs stay in work/robust_statistics/directed_score_margin; source-exposed
results cannot close the independent scientific acceptance.

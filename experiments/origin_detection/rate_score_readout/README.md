# Rate-aware readout: protocol registered before fitting

RR/Chimera are repeatedly exposed development data. No independent validation
has passed. The first pilot measured cost only; the full campaign has not run
at this protocol revision, and no performance success is claimed.

The69 post-exposure threshold diagnostic found no joint80%/2pp state among5041
fixed-score states. This motivates changing the readout, not adopting an oracle
cutoff. Reuse3920 inputs/five-score20-term bank and1260/420/420 roles. No image
extraction, simulation, neural training, external pixels or RR reserved.

Use class-balanced weighted error proxies `e_i=sigmoid(-y_i*f_i/.25)`, where
the signed label is derived from supplied integer0/1 labels, including null.
Each fit-only domain/scene/condition/variant group contributes half mass per
class, with scene aggregates. `E_g` is its weighted proxy error. Penalize
`mean(max(E_g-.2,0)^2)+mean(max(E_after-E_before-.02,0)^2)` over45 groups/75
processing/encoding comparisons. Retain the same source entropy-logistic/ridge
base. One fixed baseline initialization, L-BFGS-B,2000 iterations, finite
stationary-gradient<=1e-5 and nonincrease gate. This is nonconvex, not a globally
optimal constrained solver. Sigmoid at zero gives half error while actual tie
classification is REAL; proxy feasibility does not prove hard feasibility.

Register strengths0/10/100/1000, cold100/warm10, before any real fit. Same
three train/onecal/oneheld source folds; paired BA/drop calibration from69
including reported fallback precedes exact OOF evaluation/strength selection.
Final four heads retain60/130 queries. All cal support is raw/Q90/Q60; query
includesQ70. Wrong-source control is expected identical: aggregate rate
differences do not depend on matching order, and true source-risk/bank/maps
remain unchanged. It is not a causal pairing intervention. Null bases/head
use supplied pseudo labels, finalcal true labels as in previous campaigns.

Credit conventional rate proxies/penalty methods; not a new optimization
principle or reproduction of Cotter's proxy-Lagrangian algorithm. Primary
records: [JMLR2019](https://www.jmlr.org/papers/v20/18-616.html), introduction
pp2–5 and related-work pp6–9 targeted read; [official TFCO](https://github.com/google-research/tensorflow_constrained_optimization)
README proxy/rate helpers reviewed. The paper explicitly warns surrogate
constraints may fail true constraints. Full59-page/appendix read, two-player
theorems, author recency/citation chains remain owed; no literature-completeness
or novelty claim. [ICML2019](https://proceedings.mlr.press/v97/cotter19b.html)
primary abstract identifies constraint generalization with separate datasets;
our CV roles test a different procedure and do not inherit that theorem.

Before full run: known class-balanced groups/complete fit-role refusal,
finite-difference analytic gradient including bias, soft/hard tie mismatch,
zero parity, wrong structural equality, cold uint8 parity and serialization;
old bank/record-aware/calibration controls. Measure same-code cold/warm pilot
under single CPU thread, per-call120s gate; projected40CV cost is conditional.
Save controls/pins/pilot/rules/OOF/bank entries, then independently recount hard
rates. Do not expand strengths after viewing selection.

Command: `.venv/Scripts/python.exe -m experiments.origin_detection.rate_score_readout.run_iteration --record-controls`,
then `--pilot`, then without flags. Outputs private work/robust_statistics/rate_score_readout.

Pre-full review added the necessary software counterexample: zero sigmoid
penalty/soft BA about85% can coexist with hard BA70%. Tests also explicitly
assert each zero-score REAL/FAKE decision, harmful processing direction and
improvement without extra rate penalty. Production optimizer code is unchanged.
Initial four-test controls and20.464s cold/2.604s warm pilot are preserved in
private work/robust_statistics/rate_score_readout_precheck_v1, tied to Git013b0a4.
New controls/pilot must use this strengthened test/protocol version; old
receipts are not relabeled as having passed the extra tests.

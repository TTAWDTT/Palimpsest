# Split basis/readout sources: protocol before real fitting

Repeatedly exposed RR/Chimera development only. No independent validation or
new performance result. After70 all20 true fit penalties were zero and all
four11340-row OOF predictions exact, test source-disjoint supervision stages.
This is a conventional sample-split heuristic, not Cotter's two-player algorithm
or a claim that source reuse was proven to cause failure.

Keep3920 inputs, five directions/20 terms, same rate loss, source-logistic/ridge,
strength0/10/100/1000 and paired calibration. Inside each supplied fit call,
split sources equally: stratify by(domain,scene,supplied label); order by SHA256
of compact UTF8 JSON `[20261008,domain,src]`. Take first floor(n/2) sources per
stratum, alternating floor/ceil across sorted odd strata so total halves are
equal. Refuse fewer than2 per stratum, odd total, conflicting source labels,
non-fit roles or incomplete9-view processing panels. No seed search; all views
of a source stay together. Null uses supplied pseudo labels for partition and
learning, not metadata truth.

On756 inner training sources,378 sources/3402 rows learn all directions and
bank score moments; disjoint378/3402 learn final source/rate readout. On1260
final training sources,630/5670 per stage. Every group gives half rate weight
per class even if null split classes are unequal. Both stages remain strictly
inside fit role; existing252 inner cal/252 held and420 final threshold/420
selection sources retain their roles. Current data offer all three conditions
and raw/Q90/Q60 fit/cal; queryQ70 remains outside that support.

Zero is now the sample-split source-risk baseline, not the old full-source
zero. Do not claim old exact zero parity or that any change uniquely proves
overfitting: each stage has half as many sources, and both the learned bank and
readout change. Wrong-source remains a structural identity control. Portable
saved QuantileScoreRule kind describes layout, not the training objective.
Single-start nonconvex stationary gates and hard OOF selection remain unchanged.
Soft feasibility remains distinct from hard BA/drop; final null true-label cal
and degenerate allREAL result, if any, must be shown as failure.

Before full run: row-order invariant/balanced/disjoint source split with odd
strata and invalid-role/class tests; explicit basis arrays exclude head sources,
head arrays exclude basis sources; cold uint8 parity, wrong identity, serialized
prediction, analytic rate gradient and soft/hard counterexamples; inherited
record/calibration checks. Save member keys for independent reconstruction from
OOF source IDs/labels and fold roles. Auditor must explicitly understand solver
records3402/sources378, full context6804/756 and basis3402/378, not silently
rename solver counts to old full-input counts.

Measure same-code cold100/warm10, one CPU thread,120s-percall gate; conditional
40CV cost only. Full run is four strengths x fivefold x two labels plus four
outer heads. Existing cached pixels unchanged; no neural training, simulation,
external verification images or RR reserved. Work private robust_statistics/split_rate_score.

Relevant credit: [Cotter ICML2019](https://proceedings.mlr.press/v97/cotter19b.html)
section3 Definition1 separates model/proxy training and exact constraint player
data; section4.2 additionally requires strong convexity/Lipschitz conditions,
and section5 practical algorithms lack those guarantees. Here we instead split
basis/head supervision, with no players or hard-constraint enforcement. The
main9-page text has now been read; supplement proofs and full59-page JMLR are
still owed, no completed novelty review claimed.

Command: `.venv/Scripts/python.exe -m experiments.origin_detection.split_rate_score.run_iteration`
with `--record-controls`, then `--pilot`, then without flags. Preserve old70 source and receipts.

Pre-pilot review found and fixed metadata-dependent cache reuse: a stage
template key now contains full numeric input hash plus ordered source/processing
identity and basis membership. Identical child numeric bank arrays may share a
bank; stage-template and bank counts are reported separately. A renamed-source
planted control forces different basis rows with the same full numeric context.
`training_fit.py` now belongs to the source pins. Known literal split members
including odd strata and supplied null labels were verified with .NET SHA256
on explicit compact JSON preimages; the auditor's expected fixture no longer
comes from the producer splitter. Initial13 controls under Git1c21889 are
preserved as precheck_v1; they do not certify these added gates or a real fit.

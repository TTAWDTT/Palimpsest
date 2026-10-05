# Joint RGB representation — prospective fifteenth iteration

Hypothesis: independent channel histograms omit RGB dependence that may survive
some propagation. No guarantee under physical recapture; this is a conventional
RTPO-inspired adaptation, not a claimed new invariant or faithful paper replication.

Before real extraction: two known-answer tests must pass. The first locks strict
product order (all <= and any <), incomparable and equal neighbors; demonstrates
fixed-grid component-monotone preservation and a positive-mixing counterexample.
The second has identical per-channel marginals but different joint residual counts.

Fixed preprocessing: long edge256 AREA; pad minimum20; scale factors1,2,4.
Each scale uses eight3x3 neighbors,45 dominated/incomparable bins and9 tie bins;
three normalized RGB second-difference values at the same location give63
simultaneous-sign-reversal orbits, reusing the sixth-round normalization exactly.
351 dimensions; no channel permutation folding, filenames/labels/donor parameters
in extraction. These preprocessing choices do not inherit rank invariance.

Freeze two representations: new351 alone and prior499+351=850. Each gets ERM
or group+pair logit with thirteenth-round fixed ridge0.01/pair0.1/temperature0.1;
reuse fourteenth-round ordered fitting and class threshold, no bandwidth/seed scan.
One hybrid/both source-label-shuffled control. All splits and tests stay exposed
development; RR reserved sealed.1260fit sources,420threshold,420selection.
Extract6300 signed real images to raw/Q90 rows and selection-only Q70,
13860 feature records. Preflight60images measures cost before full extraction.

Review raw/Q90/Q70 class correctness, source-paired decline, AUC bootstrap and
exact rate-boundary audit. Historical55%/5pp gates are screening only;
provisional final80%/2pp plus independent real data and decode-inclusive latency
are still required. No detector finalization/neural training in this round.
Stop this fixed candidate configuration if it fails, continue overall research.

```powershell
.venv/Scripts/python.exe -m pytest tests/detection/test_color_relations.py -q
.venv/Scripts/python.exe -m experiments.origin_detection.color_relations.run_iteration --pilot 60
.venv/Scripts/python.exe -m experiments.origin_detection.color_relations.run_iteration --prepare
.venv/Scripts/python.exe -m experiments.origin_detection.color_relations.run_iteration
```

# Truth-orthogonal sham control — prospective thirtieth iteration

Register before fit. Stages26–28 used the SAME seed and within-domain/scene
source permutation, so their RR null anomalies are correlated observations.
The fit-only metadata audit found RR truth/null phi0.0592593, not a causal
explanation of held-out scores or a finding of leakage.

Use the unchanged stage27 CLIP-L/14+DINO-S/14 descriptor1536, fit1260sources
weakraw/Q90=7560records. Assign half0/half1 sham targets within EACH
domain/scene/true-class stratum, seed20261006. All source views share one
target. Exact contingency independence from true class is required; refuse
odd/absent classes, never silently drop data. True labels are used ONLY to
construct a negative control, not detector inference. This constrained control
is not an ordinary permutation significance test.

Fit ONE source-risk head temperature.1/ridge.01/scale_floor.001/L-BFGS500/
gradient1e-5. Do not refit/select/modify stage27 real-label method, scan seeds,
replace its old null, or claim the new control retroactively clears it.
Threshold uses unchanged24 raw/Q90 true-label calibration views, selection
5040 scores/60metricgroups/130pairs. Also retain the threshold-free four raw
processed AUC intervals. This intentionally matches the existing calibration
protocol; true-label threshold exposure means it is not a wholly label-blind
pipeline. The fit truth/sham orthogonality certificate cannot establish
held-out independence, absence of dataset cues, or real deployment stability.

Before fit: exact small-stratum controls, view/order invariance, odd/missing
class and selection/conflicting-metadata refusal; pin all source/cache/code.
After fit: save assignment and SHA, fit diagnostics, every null interval,
integer confusion audit and deliberately corrupted BA refusal. No new pixels,
RR reserved, neural encoder training, device data, or speed measurement.

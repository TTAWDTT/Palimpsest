# Finite-view hinge — prospective twenty-sixth iteration

Question: does exact maximum hinge per source improve fixed-threshold stability
over the smoothed logistic source risk? This is established robust linear/SVM
machinery, not a new principle or a physical uncertainty-set theorem.

Registered before fitting this campaign:

- Reuse signed20160 original CLIP-L/14 final768 features; no new pixel inference.
  Same2100 sources,6300 images,fit1260/threshold420/selection420.
- Fit raw/Q90 only7560 views. Each domain half weight, equal source and view
  weights within domain. Pixel/source metadata is not a deployment input.
- Standardize by fit-only weighted mean/standard deviation, floor.001.
- Solve `sum(source_weight * slack_source) + .001*sum(abs(w))`, subject
  `y*(w*z+bias) >= 1-slack_source` at every signed fit view. Bias free, slack>=0.
  Compare mean hinge with per-view slack using the same fixed L1 penalty,
  and maximum-source hinge with source labels shuffled within domain/scene.
- No penalty/temperature sweep, no new feature kernel, no encoder training.
  This changes surrogate and regularization simultaneously; it cannot isolate
  hinge alone versus earlier L2 logistic heads. The mean/max hinge contrast is
  the matched test for source aggregation within this experiment.
- SciPy HiGHS-IPM, primal/dual tolerance1e-8,IPM1e-9. Known symmetric margin,
  replication/source weight, invalid source-label controls; recompute primal,
  dual feasibility and objective gap<=1e-6, deliberately corrupt primal/dual.
  These are finite-precision software checks, not an exact mathematical proof.
- Whole-fit maximum-source cost pilot,120second solver budget. Refuse timeout,
  infeasibility, unboundedness, residual failure or worse-than-constant objective;
  preserve failure. Do not claim a failed solve as evidence against mechanism.
- Successful cost pilot is reused for the max/source full evaluation; no refit.
  Other heads same budget. Threshold fixed via24 raw/Q90 views,selection all4
  variants5040 margins. Same BA,class rates,AUC,source flips/CI and integer audit.
- `StableRule` explicit center/scale/weights/bias scoring format is reused;
  legacy ridge field stores the fixed L1 coefficient; diagnostic explicitly
  identifies the LP objective rather than pretending it is L2 ridge fitting.
- A linear score over a finite feature hull attains extreme margins at observed
  vertices. This does not assert that unseen propagation lies in that hull,
  that convex feature mixtures are physical photographs, or that LP margin at
  fit automatically survives threshold recalibration and external data.
- Current development reference17:85% minimum domain,83.75% scene,8.75pp worst
  same-source drop. Goal remains sufficient accuracy,almost no decline,and fast
  live decoding pipeline with independent real validation. RR reserved sealed.

Run `source_hinge.run_iteration --pilot` after software controls; then default.
No final algorithm is certified by this pre-registration.

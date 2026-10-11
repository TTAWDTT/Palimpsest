# Compact complementary features — prospective twenty-eighth iteration

Registered before this fit. Stage27 reduced worst same-source decline to5pp
but minimum domainBA82.22% and live P9552.95ms/39.51ms still fail the overall goal.
Test whether smaller semantics keep the complementarity at lower cost.

- Reuse signed CLIP-B/16 final512 (stage22) and DINOv2-S/14 CLS384+patchmean384
  (stage18), all20160 image/variant vectors, independently normalized;join1280.
  No new pixels,layer/severity/search,neural encoder training or source metadata
  in deployment. This is conventional frozen-feature fusion,not a new principle.
- Same2100 sources6300real images;fit1260 weakraw/Q90=7560records,
  threshold420/24views,selection420/fourencodings5040scores. Same ridge.01,
  temperature.1 vs0,scale.001,L-BFGS500/gradient1e-5.
- Fixed three heads joint/source,joint/mean,joint/source_null. Existing22
  base-alone and27 weakDINO are matching component diagnostics,not refit.
- Source shuffle within domain/scene. Preserve all four null AUC intervals,
  including nonuniform abnormalities;no new seed chosen to erase an anomaly.
- Validate every parent hash/code/weight/inventory and exact metadata coverage,
  known pair schema tests before fitting;60metricgroups130sourcepairs perhead,
  integer audit/wrongBA and emitall quotes after fitting.
- If useful discrimination and decline remain competitive with27,run exact
  same120nativefile/3repeat live pair benchmark. Verify score/cache parity1e-12
  and record decoding+two preprocessors+two encoders+readout,not cached fit time.
  Accuracy80%/2pp and predict10/E2E50ms remain provisional working screens,
  independent real validation still required. Failed candidates do not end goal.
- No RR reserved unsealing,simulation or final neural training. Prior27 is
  development only with a small RR null anomaly;do not treat it as an accepted
  external reference or select this experiment's parameters after its results.

Run compact_complement.run_iteration,then audit saved scores. Live timing uses
frozen_readout.benchmark_pair_encoder --compact;the original large-pair timing
source is archived in Git31735d8 with its recorded script hash.

# Feature-only processing gate — prospective twenty-ninth iteration

Registered before this fit. Hypothesis: a shared boundary may mix original and
processed observations;separate origin experts with an estimated condition gate
could help. Mixtures of experts/probability pooling are established prior work.

- Signed stage24 finalCLIP768 columns plus stage18 DINO CLS/patchmean768,
  concat1536 as stage27. No new pixels,layer/strength sweep,encoder training,
  RR reserved access or use of physical process labels at inference.
- Fit1260 sources7560raw/Q90views,domain/source equal weights. Original expert
  sees original-condition2views/source;processed expert sees both actual
  processed-condition4views/source. Both conventional source-risk heads,T.1,
  ridge.01,scale.001,L-BFGS500/gradient1e-5. No source crosses fit/threshold/selection.
- Conventional linear logistic gate gets binarycondition labels only onfit:
  original0/processed1. Sample weights rebalance these two labels to50/50;
  no AI label predicts the gate. Meanrisk T0,otherwise same fixed optimization.
  The learned gate is a processing cue,not a calibrated wild-channel posterior
  or exact inference of device/physical method. OriginalQ90still labelledoriginal;
  processed means authors' condition,not every possible later JPEG event.
- Final probability `(1-p_gate)*sigmoid(raw_score)+p_gate*sigmoid(processed_score)`,
  stable logit storage. Same24threshold views fit one final threshold;selection
  fourencodings5040margins. No percondition deployment threshold.
- Candidates soft_gate,uniform_half using identical origin experts,soft_gate_null
  with origin labels source-shuffled within domain/scene and same fixed gate.
  Only feature vectors enter score. No oraclecondition score used for selection.
- Gate accuracy/AUC on exposed development reported separately and not considered
  origin-detection success. Preserve all null intervals/anomalies,not justuniform
  refusal. Known probability/endpoints/extreme logits/singlebatch/artifact and
  invalidschema controls before fit; signed cache and exact source masks checked.
- Compare60groups130same-source pairs,bothclass rates,BA,AUC,flip/bootstrap2000,
  integer audit/wrongBA and emitall. Current tradeoff27:82.22% domain,91.25%
  Chimera scene,5ppdrop. Overall goal still high absolute ability,almost no
  decline,live speed and independent real validation;provisional80%/2pp,10/50ms.
- Fitting cost and a successful gate do not certify detector stability. If accuracy
  and decline meaningfully improve,bind fixedrule before live cost/external tests.
  Failed head does not end goal. Neural training and simulation remain paused.

Prior reading scope is in sources/2026-10-06_condition_mixture_review.json.
Run condition_mixture.run_iteration after software controls;preserve failures.

# Complementary frozen encoders — prospective twenty-seventh iteration

Question: do contrastive global semantics and self-supervised CLS/patch-mean
features provide complementary class evidence under real propagation? Combining
pretrained representations is established practice, not a new ensemble principle.

Registered before this fit:

- Reuse signed CLIP-L/14 final768 and DINOv2-S/14 CLS384+patchmean384 caches,
  all20160 matching image/variant vectors. Each component is independently unit
  normalized as before; concatenate1536. No layer/kernel/weight search or new
  pixel extraction, no encoder training.
- Same2100 sources/6300 development images;fit1260 sources weakraw/Q90=7560
  views;threshold420 sources24 views;selection420 sources4 variants5040 scores.
  DINO stage18 used three fit encodings;add weakDINO/source as matched baseline.
- Candidates weakDINO/source, joint/source, joint/mean, joint/source_null. Same
  ridge.01,T.1/0,scale.001,L-BFGS500/gradient1e-5 as stage17/24/25. Source/scene
  metadata only for fitting/assessment,never as deployment input.
- Parent code/weights/cache/inventory hashes and complete metadata join required.
  Known vector-composition/wrong dimension/duplicate names controls first;
  no successful defaults and no sealed RR images.
- Evaluate60 groups130same-source pairs,BA,both class rates,AUC,flip/bootstrapCI;
  source-label-shuffle within domain/scene, independent integer audit and wrongBA.
  Fixed24threshold calibration,score>threshold AI and tiesREAL unchanged.
- Reference17:minimum domain85%,scene83.75%,worstdrop8.75pp. Joint candidate
  must improve useful absolute evidence and decline; accuracy alone is not goal.
  If it meaningfully improves,live sequential two-encoder parity and same120file
  decode-inclusive cost are required. Cached fitting cost cannot establish speed.
- Joint computation has higher cost than one encoder. This tradeoff is explicit;
  frozen_feature_pair live composition is reusable but does not certify fastness.
  Classification on observed development cannot certify independent real
  generalization. RR reserved sealed;simulation/finalneuraltraining paused.

Prior sources already audited in the stage16/18 frozen-encoder research records:
CLIP's contrastive semantic pretraining and DINOv2 self-supervised patch features.
Those original recognition results are not propagation-robust AI detection proof.

Command: `complementary_encoders.run_iteration` after known controls. Failures
are preserved;they do not end the overall user objective.

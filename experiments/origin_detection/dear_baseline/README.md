# DEAR-r — prospective fixed baseline and mechanism diagnostic

Read PMLR ICML2026 paper main method, results and implementation appendix;
inspect pinned author code 5e0dc665eee24b632c03be09a27a05319abc9e7f.
Use one official DEAR-r weight, SHA430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97,
CC BY-NC4.0 plus declared OpenRAIL-M restrictions. This is an author neural
baseline, not our newly trained model, RAD implementation or novel algorithm.

Question: does spatial forensic channel selection preserve discrimination in
our physical/digital conditions where frozen global representations still fall?
Pruning and classifier refinement change together; no causal isolation of either.

No target fitting, threshold calibration, resizing, cropping or OOM fallback.
FP32 native-resolution ToTensor/ImageNet normalization, logit>0 means AI.
Strict load and exact synthetic repeat gate before inference. No batch bucketing.
Complete checkpoint/source SHA verified, 820/2048 gate channels expected.

Selection-only fixed420sources1260images; raw and same Q90/Q70/Q60 encoding
variants. All are exposed development, not a new external test. Evaluate both
classes,60groups130same-source pairs, CIs and flips. Compare saved seventeenth
reference on identical views. Do not rerun B-Free/D3/Benford full baselines or
open RR reserved. Fixed60image cost pilot stratifies domain/class/condition plus
six largest files by metadata. Full run only after pilot succeeds; retain any
failure without altering native inference. No optional diagnostic inpaint data
download or new encoder training in this round.

If useful, measure the same120native files withdecode+preprocess+forward and
cached-score checks. Inference timing during vector generation is not that
benchmark. Failure of this baseline continues the broader algorithm goal.

## Native pilot failure and prospective aligned execution amendment

Native60-image pilot failed CUDA OOM on large files;last progress30images120views.
No complete predictions were saved, no accuracy result and no fallback resizing.
The failed native code pins are preserved separately.

Before new inference, test aligned convolutional execution. From the inspected
ResNet stride0=1, conv7/pool3 plus3/4/6/3 bottlenecks gives support113pixels and
stride8. BN in eval is pointwise; no global spatial operation precedes pooling.
Use64pixel halos (aligned tostride8),64x64final cells per tile; retain each global
output cell once, sumfeatures inFP64, castFP32, then apply saved gate and head.
This is not resize, patch voting, mean oftile logits or exactbitwise equality.

Three synthetic shapes97x133,191x197,645x653 compare full andtiled logits with
absolute tolerance1e-4;featuremean atol1e-3,rtol1e-5. Undersized halo8 must fail
feature equivalence. Real pilot must separately compare feasible native images
and all decision signs. Do not widen tolerances after results. Tile geometry
planted controls must cover everyglobal outputcell exactly once. Tiled run uses
distinct output directories and explicitly reports execution variant. If parity
fails, stop this variant, retain failure, continue overall method development.

Metadata-only amendment before real parity:500kpixel cap omitted both camera
conditions (585225 and1052676pixels), so no real parity was run. Use1.2MP cap,
choose largest feasible file per domain/class/condition (12files);score tolerance
and decision gate unchanged. This cap limits only numerical comparison memory,
not the tiled pilot/full evaluation denominator or inference resolution.

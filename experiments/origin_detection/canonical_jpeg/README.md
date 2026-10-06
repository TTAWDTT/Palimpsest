# Fixed detection canonicalization — prospective iteration 35

Register before extraction or selection. Recent CuRe inference code at
8f7832c83dcd3f4f89de955103e94a64ddc3cdc9 applies JPEG70 after resizing all
queries, with an additional PNG-only JPEG96. This exposes a method-specific
normalization as well as trained response representation. It does not prove
which part causes the paper's robustness. No CuRe replication or new mechanism
claim; its LoRA/base/head are not used here.

Test one fixed common detection preprocessing: AREA longedge at most256,
PIL JPEGquality70 subsampling2, then unchanged official CLIP-L/14 JIT fp16
vision. No quality/crop/scale parameter search, no filename-based decisions.
Raw/Q90/Q70/Q60 test inputs are each normalized independently; existing
augmented JPEG inputs undergo a SECOND encoding. This is not idempotent,
cannot recover deleted information and may weaken true/AI separation.
It is an inference operation, not new simulation research or final encoder training.

Same6300images20160features; fit1260sources7560weak canonical raw/Q90 views,
threshold420sources24weakgroups, selection420sources5040scores60groups130pairs.
Conventional source-riskT.1 and mean-risk0 heads, ridge.01, scale-floor.001,
LBFGS500 grad<=1e-5. Matching source-risk balanced label-null fromstage30.
Threshold uses real threshold labels including null. Four rawprocessed AUC CIs.
Do not tune the JPEG quality after selection; no reserved/external pixels.

Synthetic flat image, geometry/channel checks, deterministic repeat and known
non-idempotence before runtime. Original canonical raw feature must equal the
signed old JPEG70 feature for the same file, byte-exact all768coordinates.
Pilot60files must pass this mapping and wrong+.001 rejection before full run;
full6300raw-to-Q70 mapping checked after full extraction. Runtime compares
wrapper against original encoder on identical canonical pixels on3syntheticshapes.
No identity parity for re-encoded Q90/Q70/Q60 is expected; record their actual
loss, do not copy raw scores. All weights/code/cache/roles pinned.
Independent integer/BA error and quote arithmetic after evaluation. A changed
method needs matching live decode+JPEG+CLIP speed if metrics warrant it.
Existing stage17 is an exposed development reference, not newly re-fit here.

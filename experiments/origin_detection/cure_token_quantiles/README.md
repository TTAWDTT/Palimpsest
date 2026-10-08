# Token projection shape: admission pilot

Registered before target feature extraction. Existing mean, clipped center,
normalized diagonal spread and normalized covariance cannot distinguish all
token distributions. Exact equal-moment rational examples motivate checking
additional shape; they are not class labels or evidence of camera invariance.

Reuse the frozen CuRe post-LN576x1024 bag and its fixed1024x32 projection
(seed20261008). For each projected column, center at median, divide by MAD;
if MAD is zero use RMS about median, and if constant output zero. Both scales
are homogeneous under positive scaling; no fixed epsilon is inserted.
Store ten linear-interpolated quantiles .05/.1/.2/.3/.4/.6/.7/.8/.9/.95:
320 new coordinates, appended to the unchanged3600 descriptor. No extra
neural forward, new projection selection or new neural training.

This finite shape profile borrows projection/one-dimensional-distribution
ideas from sliced Wasserstein literature; it does not compute the continuous
SW distance or inherit injectivity/kernel theorems. Per-column normalization
also discards location/scale; original information remains in the parent
prefix. Finite directions can miss nullspace changes, and extreme quantiles
can change under small contamination. Bag permutation/positive scale/shift
properties are not screen-camera invariance of a transformer.

Run software controls on known equal-old-statistic bags, permutation/positive
affine transformations, MAD-zero/constant cases and blind projection plant.
Then ONLY reuse the old60 exposed selection native files as an engineering
witness:240 query variants, repeat60 raw, all3600 parent values and author
probabilities must be bitexact. Shared numeric_extension handles image hashes,
materialization and cache storage. This entrypoint only runs the engineering
pilot, without classification or full extraction. A later full-run protocol
needs measured budget and proper source
roles. A successful pilot establishes software integration, not accuracy.

# CuRe token statistics — prospective iteration52

Register before new extraction. Existing full mean readouts50/51 remain above
2pp decline. Inspect actual released PECore-L14 source: ln_post produces577
tokens; CLS is excluded and576 patches averaged. Read-only hook captures
these patches in that same single forward;require exact author mean and saved
36full means/probabilities. No encoder/LoRA/preprocessing changes or new pixels.

Fixed bag map in FP64: coordinate median anchor a; distances||t_i-a||;radius
median distance;f_i=min(1,radius/distance),zero distance uses1. Clip each token
to a+f_i*(t_i-a),then average;normalized population RMS around that clipped
mean is the second1024-coordinate descriptor. Zero spread gives zeros. This
is median-anchored radial winsorization,not a solved Huber M-estimator or new
robust estimation theory. Restricted feature translation/positive scale
equivariance does not imply image processing invariance. Patches are dependent
and attention can spread corruption;salient minority tokens can be removed.
No spatial ordering is added by bag statistics;no segmentation claim.

Save3072 coordinates: original author mean1024,clipped center1024,unitspread1024.
Five fixed heads:clipped/source,spread/source,mixed/source,mixed/mean,
mixed/source_null;mix=clipped+spread2048. SourceT.1/meanT0,ridge.01,floor.001,
LBFGS500/gradient<=1e-5. Source roles1260/420/420 unchanged;weak raw/Q90 fit7560,
threshold2520/24views,selection5040/4variants,60metrics/130pairs. Null30hash
unchanged,true threshold labels included fornull. Original mean reference
reuses36 result after full vector/probability parity,not baseline rerun.

Memory: metadataCSV + FP64.npy memmap,not millions of Python numeric strings.
Lazy Mapping rows and metadata-only ChainMap null labels shared by fit/evaluate.
Decoder/source reference read streaming;compact original means in FP32 before
encoder loading,exact cast from36FP32 storage. .partial files untrusted until
all coverage/finiteness/hashes and receipt complete. No resume implemented;
failure preserves partial evidence,never silently overwrites it.

Before real extraction: analytical clipping,constant/outlier majority,
same-mean different-spread,positive feature-affine/permutation examples,
invalid-input and numeric-storage identity controls. GPU three artificial
images(flat/channel/spatial) before pilot,repeat vectors/scalars and direct
hook-disabled original parity;changed mean coordinate rejected. Then the
same60 previously exposed36pilot native files and all4variants:original means
and probabilities bit-exact;repeated feature vectors for60raw files exact.
Pilot median extraction<=150ms engineering budget,not user deployment target.
Only after cost/runtime/code/data/parity passed full6300/15120 extraction.
Runtime packages/weight/source hashes recorded,one attached GPU worker.

Full3072cache signed and audited before head fitting. All null raw AUC CIs,
integer/wrongBA,emitall/wrongquote retained. Independent data/RRreserved sealed;
development near-content doubt stays explicit. Goal still80%/2pp/independent
real validation;live deployment parity/timing needed if candidate warrants it.

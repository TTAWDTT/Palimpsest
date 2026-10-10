# Frozen CuRe vectors with conventional readouts — prospective iteration36

Register before pixel extraction or fitting. Official scalar CuRe baseline
already scored the exposed selection queue: worst drop22.5pp and minimum
domain BA56.67%. Post-exposure per-group labeled threshold oracle still has
minimum domain71.67%, so merely recalibrating that scalar cannot meet the
provisional80% gate on this finite queue. This is not a population Bayes limit.

Test information retained before the head: release's1024 post-LN patchmean and
128 response subspace (f-mu)@W. Use unchanged released LoRA/base/projection,
author file preprocessing including PNG branch, batch1 fp16, torch4/OpenCV1
threads. No encoder/projection training, restoration or new simulation.
Store raw unbounded vectors, not unit normalization, arctan or sigmoid.
Feature float32/float16 values cast exactly to float64 for CSV. No layer, quality,
crop, subspace dimension or strength search. No concatenation with other models.

Same6300real images and2100source registry; fit1260source weakraw/Q90 six
views/source=7560, threshold420source2520weakviews/24groups, selection420source
5040views in raw/Q90/Q70/Q60. Total15120feature rows. Keep source-weighted
domainhalf mass, same ridge.01, scale-floor.001, LBFGS500 gradient<=1e-5,
seed20261006. Two representation choices times meanT0/sourceT.1, plus matching
balanced label-null sourceT.1 for both representations: six fixed heads.
Balanced assignment must match stage30 exactly; true threshold labels calibrate
each head including null. Four raw processed AUC intervals per null.
No RR reserved/new external pixels. Repeatedly exposed development, no novelty
or successful independent robustness claim from this representation substitution.

First software controls: known unbounded signed vectors retain exact FP32/FP16,
old default bounded validator refuses them; invalid vectors refused. Synthetic
file wrapper features repeat exactly and original-head probabilities match
direct author forward on flat/channel/spatial images. Pilot60selectionfiles
stratified domain/scene/class/condition plus largest images, all four query
variants; all240 reconstructed original-head probabilities must equal the
already signed CuRe baseline scores bit-exact. Deliberately changed probability
must be refused. Code/weights/source/preprocessor/CSV/registry pinned.
Full extraction requires successful identical pilot; original-head probabilities
must exactly match all5040 old selection entries; no copying vectors or scores.

After exact coverage/identity/finite-vector validation, fit six heads, evaluate
60metric groups/130same-source pairs each, audit integer rates and deliberately
wrong BA, then quote arithmetic. Worst-pair intervals are selected post hoc
and uncorrected. Live decode+preprocessor+encoder+readout speed only if a new
candidate warrants it. Author scalar latency is not this new head's latency.

Commands: `prepare_features --pilot`; `prepare_features --prepare`; `run_iteration`.

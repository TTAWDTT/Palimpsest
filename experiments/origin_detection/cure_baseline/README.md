# Official CuRe fixed development baseline

Registered before native model inference. Author source commit
8f7832c83dcd3f4f89de955103e94a64ddc3cdc9, adapter SHA256
17cabb334563c15abc8827db87dc272a4aef10f4279025faee1ef8d32c444ae1,
PE-Core-L14-336 base revision bafb0f76541d399057e980a25947f67acec76575,
SHA256 0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8.
Author code/weights external; verify Git blobs, file hashes and strict model load.
Use exact author file preprocessing including PNG-only JPEG96, common JPEG70,
outer CUDA fp16 autocast, CPU torch threads4, batch1, probability>=.5 fake.
The shared strict-margin evaluator maps an exact .5 tie to positive subnormal;
retain the original probability in scores. No threshold calibration/training.

Same exposed selection queue420sources/1260realfiles; four query variants
raw/Q90-444/Q70-420/Q60-420, latter three AREA longedge<=256 then PIL JPEG.
Author preprocessor receives original file for raw and actual .jpg for variants;
do not add PNG-specific reencoding to synthetic JPEG variants. No fit/threshold
pixels in accuracy evaluation or model adaptation, no RR reserved or new external
validation pixels. Same60metrics130pairs.
Pretrained backbone/adapter overlap unknown; this is a development baseline,
not independent final validation or reproduction of the paper's RR queue.

Gate first: tie software tests; exact fixed source blobs; official synthetic
preprocessing repeat/flat/dimensions/PNG branch; model strict load; synthetic
model repeat; compare adapter probability with direct author expression.
Then pilot60files stratified by domain/class/condition/scene plus large images,
check finite probabilities, unique filenames and explicit variants. Pilot checks
execution only, no parameter choices. Full inference requires identical pins
and successful pilot. Native JPEG scores must exactly repeat when recomputed.

Report AUC, fixed-threshold BA and per-class rates, exact source-paired drops
and intervals, class flips; independent integer-count audit and planted BA error.
After full scores, benchmark the existing120nativefile timing queue three times
at batch1, compare exact saved probabilities, hash files before timer, include
decode+author preprocessing+GPU+head, exclude initialization/warmup. Record
CPU threads4 as author protocol, not identical threads1 to custom benchmarks;
separate <=2MP and above2MP, no phone/localization/runtime transfer claims.

Prospective timing-inventory correction before CuRe model execution: the existing
120file timing queue contains32fit,14threshold and74selection native files.
The latter probabilities come from the accuracy run; the other46 files are
predicted once without timing to bind expected outputs, then used ONLY for
timing/repeat parity. Their labels are never used or added to accuracy metrics.
Record all120 reference probabilities and the role counts in the timing receipt.

`run_iteration --pilot` and then `run_iteration`; final `benchmark_files`.
No author methods code altered, no quality sweep, no result-guided threshold.

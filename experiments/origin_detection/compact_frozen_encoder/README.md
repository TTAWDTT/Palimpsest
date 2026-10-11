# Compact frozen encoder — prospective eighteenth iteration

Motivation fixed before stage17 results: stage16 neural representation recovered
absolute class evidence but failed scene stability and latency. Test one smaller
existing pretrained encoder, not a newly trained backbone. Official DINOv2-S/14
commit7764ea0f912e53c92e82eb78a2a1631e92725fc8 and88,283,115-byte checkpoint
SHA256b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9.
Code/card reviewed; generic classification evidence is not AI/capture validation.

Official constructor and strict state loading, unchanged float32, TF32disabled,
no xFormers, native PyTorch attention. Official eval shortside256/center224 and
ImageNet normalization, stored orientation. Extract CLS384 and mean of normalized
patch tokens384 once; independently L2-unit/affine storage. Compare CLS vs CLS+mean,
each mean loss vs source-temperature.1, all ridge.01. No size,seed,temperature or
precision sweep. One combined/source source-label-shuffled null. Not WaRPAD or
DINO-Detect reproduction, and model-family choice is not novel research.

Same6300 images and fixed fit/threshold/selection sources; raw/Q90/Q70 for every
image, Q60 only1260 selection images:20160vectors. Fit uses fit-role three real
conditions x three encodings; threshold fixed24raw/Q90 views only. Q60 diagnostic
has fresh encoding parameters, not independent content. Full per-scene/classes,
source-paired changes for all encoding/physical views, CI and flips. Require known
preprocess/descriptor controls, official torchvision equivalence and strict weight
coverage/repeat before a60-imagecostpilot and full extraction. If equivalence,
coverage or repeat fail, refuse cache/fitting. Keep source registry unchanged.

Choose a development diagnostic candidate by greatest minimum scene/domain BA,
then smallest maximum scene drop, then CLS for cost, then mean loss. This choice
is not acceptance; provisional BA80%/drop2pp across scopes, independent real
verification and decode-inclusive speed must still be checked. Benchmark one
chosen rule with120 fixed native files and3repeats, preserving>2MP split and
SHA checks outside timer. No detector outputs select extra data or parameters.

RRreserved stays sealed; no full old baseline rerun; no simulation or backbone
training. Failed candidate does not end overall user goal.

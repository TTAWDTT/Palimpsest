# Source-view risk — prospective seventeenth iteration

Motivation: stage16 recovered absolute class evidence, but aggregate4.58pp hid
10pp cat/Mac/Q70 decline. Compare harder encoded training and source-wise risk
separately, preserving one pretrained backbone. This is a MaxUp-inspired
adaptation, not a new maximum-loss principle, full-network training or guaranteed
physical invariance. Definitionread: https://openaccess.thecvf.com/content/CVPR2021/papers/Gong_MaxUp_Lightweight_Adversarial_Training_With_Data_Augmentation_Improves_Neural_Network_CVPR_2021_paper.pdf

For signed margins ys*f(x), logistic losses lsv, fixed temperatureT=.1:
Rs=T*log(sum_v p(v|s)*exp(lsv/T)). Fit weighted source meanRs+ridge*||w||²,
ridge=.01 (the same convention as stage16). Mean-loss control is T=0 with
conditional mean, not an inverse-temperature singularity. It is convex in the
linear head; smoothed risk is between mean and finite-view max. Source-risk
weights use domain half/half and sources equal within each domain. No source or
condition tags enter deployed features or scoring. Standardize fit moments only.
L-BFGSmax500,gradient max<=1e-5;failure refuses interpretation.

Fixed2x2: weak(raw/Q90) vs strong(raw/Q90/Q70) fit augmentation; mean loss vs
source risk. Weak/mean is existing stage16clip/erm, not refitted. One strong/source
source-label-shuffled null (within domain/scene), plus existingclip/both as frozen
reference, not an additional searched model. All use CLIP768 only. No seed,
temperature,backbone,size or ridge sweep. Threshold calibration stays24raw/Q90
views for all candidates, without Q70/Q60 calibration, to separate training
augmentation from threshold changes.

Reusedsigned6300 real images; additional cache: Q70 for all6300, Q60 only1260
selection images =7560 vectors. Regenerated1260 selectionQ70 vectors must match
stage16exactly before accepting new extraction. Strongfit1260sources x9states
=11340rows, weakfit7560. Threshold420sources,selection420. Q70is already exposed
development diagnostic; newly declared Q60is encoding-only, not independent
sources. Same resize256 then JPEG4:2:0 before unchanged CLIP224 crop.

Before extraction/fitting: known equal-view risk, hard-view weighting, two-step
finite-difference gradients, permutation and source-label consistency/refusal
controls.60-imagecostpilot; no plots or new data after outcomes to tune this trial.
Evaluate all four encodings, per-domain and per-scene both classes; matched
physical changes within each encoding and all encoding-pair changes. Report
confidence intervals and decision flips; don't hide worse scene behind net BA.
Historical55%/5pp gates are screens only. Provisional80%/2pp plus independent
real data and decode-inclusive speed still needed; stage16speed does not close
this goal. Keep RRreserved sealed, simulation/finalencoder training paused.
Stop failed configuration, continue overall goal.

# Source score consistency — prospective twenty-first iteration

Stage20 reduced average representation drift but lost class evidence or kept
large accuracy decline. Retain full CLIP768 and source-risk classifier17;
penalize fitted score variance across matched source views without truncation.
This reuses known consistency regularization (paired-score8/group-margin13)
with the source-risk17 objective;not a novel loss principle or unseen guarantee.

Same1260fit sources,raw/Q90,six real-condition/encoding views each. Equal
domain/source/views;within-source conditional means form normalized-feature
variance gram. Minimize source-temperature.1 logistic risk + ridge.01||w||²
+ strength*w'Gram*w. Fixed strengths0,.1,1,not a selection-dependent sweep.
Zero reuses existing fitter exactly and must match17weak/source saved margins.
Consistency term retains label-risk term;report both and classwise failures.

One wrong-source control at.1: within domain/class and each condition/encoding
view independently permute source IDs;honest classification-risk source groups
remain unchanged,only consistency gram uses scrambled groups. One source-label
shuffled null at.1. Fit normalization remains fit-only;all controls same counts.
Threshold24raw/Q90 views,selection420sources/four encodings,60groups130pairs.
Known weighted variance/gradient tests precede fit-only pilot and fullrun.
No backbone training or new images;RRreserved stays sealed.

Choose one honest candidate greatest minimum domain/scene BA,then smaller worst
scene drop,then lower strength. Provisional80%/2pp,absolute perclass evidence,
pairedCI/flips,independent real validation and decode-inclusive speed required.
No post-hoc strength search. Failure continues the research goal.

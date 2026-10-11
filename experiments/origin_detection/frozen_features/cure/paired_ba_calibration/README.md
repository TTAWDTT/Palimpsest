# Matched BA/drop calibration: prospective protocol

Registered after68 failed and before this real calibration/fitting. Reuse67's
complete3920 features, five directions/20 terms and source-variance objective,
lambda0/.1/1/10. Change only the global threshold policy; do not use68's different
directed loss. No encoder/new images, no selection oracle threshold.

Calibration rows exclusively have threshold role (or the fit-internal cal
fold temporarily marked threshold). Include36 primitive groups plus9 scene
aggregates=45 groups and75 matched processing/encoding comparisons. Evaluate
all unique score cutoffs and strictFAKE `score>threshold`, tiesREAL. Integer
BA/drop units use the LCM of2×classcounts; grouped identical scores flip together.

If any candidate hasminimum BA>=.8 andmaximum matched drop<=.02, choose greatest
minimumBA, then smallerdrop, greaterminimumclassrate, smaller|threshold| and
smallerthreshold. If onlyBA>=.8 feasible, choose smallestmaximumdrop first,
then greatestminimumBA/classrate and the same ties. If noBA-feasible cutoff,
maximize minimumBA thenminimizedrop/classrate/ties. All fallback states are
reported, not called successful acceptance. Role checks and pair identities
must hold before selection. Primary difference is policy, not data orlambda.

Whole source-held CV includes this new calibration before parameter selection;
three train folds/onecal/oneheld remain756/252/252 sources. Final calibration
uses the frozen420 threshold sources; outer420 sources are evaluated once by
the registered four-head protocol60/130. Null innercal usespseudo labels;
final thresholdtrue labels. Matching the policy's internal target cannot
prove an external80%/2pp result or guarantee future processing stability.

Knownperfectpanel/ties/independent brute-force Fraction selector, role refusal,
old risk/bank components and same-code cold/warm pilot precede the full run.
Measured120s-percall gate andconditional40CV budget stay; no classification
metric fromselection is used to choose the new cutoff orappend parameter grid.
Record cutoff state count, feasibility/fallback, exactcalBA/drop, software and
source pins. Outputs in work/robust_statistics/paired_ba_calibration.

# Matched local score transport — prospective thirty-third iteration

Register before selection evaluation; independent of ongoingstage32 results.
Stage31 class supports did not remove decline. Test whether matched sources
help estimate a LOCAL score change instead of forming a global invariant
projection or replacing the classifier with class-neighbor votes.

Reuse signedstage27CLIP-L+DINO1536. Base original-only source-risk head is
the unchanged original expert in stage29 soft_gate_rule.json (fit1260sources,
raw/Q90original2520views,T.1,ridge.01). Do not refit this real-label head.
For each FIT source store its six weak views and the mean base score of its
two original views. Query feature metric is fixed .5CLIP/.25DINOCLS/.25mean.
For each source pick its closest view;pick k=5 nearest DISTINCT sources
globally without query labels or condition. Output:

`base(query)+mean_i[original_anchor_score(i)-base(closest_view_i)]`.

This estimates a score-level nuisance correction from known fit pairs;it is
not a physical inverse operator, optimal transport coupling, restored pixels,
or new kernel-regression principle. Original JPEG augmentation is still an
original authorcondition. Unknown strong processing may not match any fitview;
neighbors can be wrongclass and correction can hallucinate origin evidence.
No query metadata or transductive testbatch adaptation is permitted.

Fixed comparisons: original_head identity baseline, matched_correction,
shuffled_anchor_correction. Shuffle original anchors among sources within
domain/scene,seed20261006 (not within trueclass), keeping base/index unchanged;
this alignment control is NOT a label-null pipeline because base head uses
real labels. Additional balanced_null_correction fits ONE original-only head
with the same truth-orthogonal sham labels asstage30, then constructs its own
anchors/closest-view scores,matching the mechanism. No k/gain/encoder/metric
or anchor search;full correctiongain1,identitygain0,uniformmean5neighbors.

Same24weakthreshold views420sources;selection420sources5040scores,
60metricgroups130pairs. Threshold calibration remains true-label exposed
even for null;four rawprocessed AUC intervals mandatory. All pairs arefitonly,
source counts/6views/2original anchors exact. Save rules,indexblobs/hashes,
parentcode/cache/inventory and reusedbase-headhash. Knownhandcase mustrecover
knownadditive score offset,zerooffsetidentity,two-source neighbor selection,
single/batchbit equality,finite/schema/coverage and artifactintegrity controls.
Independent integer counts and incorrectBA plus quote arithmetic after run.

Field mapping/expected-risk prior: FLDA JMLR2016 main1-6read,including explicit
label conditional-independence and transferdistribution assumptions. That work
fits target likelihood and minimizes expected transferloss;we do neither and
have matched author fitviews. No population theorem is imported for this local
finite correction. No new pixels/RRreserved/simulation/finalneuraltraining.
Only benchmark live decoding+twoencoders+neighbormapping if decision metrics
warrant it. Candidate failure does not end the research goal.

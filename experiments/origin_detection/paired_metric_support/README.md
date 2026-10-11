# Full paired metric plus source supports — prospective iteration47

Register before fitting any real metric or evaluating its scores. Frozen CuRe
1024full signed36cache; no new pixels or encoder training. Cross-camera matching
primary: Liao etal,CVPR2015 XQDA,PDFpp3-6Eq1-9;person identity likelihood ratios
and supervised subspace are prior art,not our binary AI detector or a physical
robustness theorem. This round uses only a conventional genuine-pair covariance
Mahalanobis metric,not a reproduction of full XQDA/LOMO/KISSME.

Fit1260sources/7560weak raw,Q90 records,domain/source/view weights. Fit weighted
center and per-coordinate standard deviation floor.001. In standardized space,
compute mean outer product of all15unordered distinct view differences/source,
weight each source by its fit weight. No centering of the difference vectors
across sources. Full matrix C retained;no rank truncation or eigenvalue search.
Whitening W=(C+.01*trace(C)/1024*I)^(-1/2);if covariance trace is zero,refuse.
Require finite positive eigenvalues and inverse-root numerical identity<=1e-8.

Four fixed heads,each stores six views/source and k5different sources/class:
standardized Euclidean,pairedmetric,pairedmetric/source_null,wrongsource_metric.
Use existing SourceSupportRule minview Euclidean distance and averagekclass
distance;save only transformed FIT supports plus full fixed query map. Wrong
metric reuses stage45 within(domain,scene,class,condition,variant)source shuffles;
support bags themselves remain the correctly grouped sources,so only nuisance
metric is changed. Balancedsource_null30 changes fit support labels only;metric
is entirely label blind. Query reads only its feature vector,never source/class
domain/condition/other query images. No post-hoc k,ridge,scale or support search.

Class-threshold calibrated on existing24weak threshold views420sources;selection
5040scores/60groups/130pairs,all four variants includingQ70/Q60. Stage36source
and31support are historical references,not independent tests. Report null raw
processing AUC intervals,finite/identity and singlebatch exact parity,loadsave,
integer and wrongBA,emitall and wrong quote. No oracle thresholds from the recent
score feasibility diagnostic are used. RRreserved/external pixels stay sealed.

Before real metric: independent explicit unordered-pair covariance toy,known
diagonal whitening,planted source/finite/trace failures,portable rule roundtrip
and artifact corruption,query batch exact parity. Reuse existing distinct-source
and view-duplication controls. Timing synthetic1024D fit/query only to check
feasibility,not a live speed claim. Real covariance/heads are serialized on CPU
after controls,with BFree GPU extraction still attached;do not start another
GPU helper. BA>=80%,drop<=2pp and independent validation remain the final goal.

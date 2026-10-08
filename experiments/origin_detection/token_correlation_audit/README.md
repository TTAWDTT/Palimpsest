# Token covariance information control

Analytic feature bags chosen while iteration55 runs. This is a mechanism control,
not a detector or selection-data experiment. Two four-token bags have identical
mean, clipped center and normalized diagonal RMS, but opposite cross-coordinate
correlation. Any deterministic readout of the existing equal descriptors must
also be equal. Covariance can distinguish them; no implication that real AI
images are separable or that recapture preserves their covariance.

Run `python -m experiments.origin_detection.token_correlation_audit.audit_information_loss`.
Known matrices and an incorrect same-covariance claim are checked. No pixels,
labels, learned projections or extra encoders are used; output is append-only.

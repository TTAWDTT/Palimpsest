# Reconstruction cancellation — finite mechanism audit

Before exact arithmetic, register25 offsets {-2,-1,0,1,2}/7 in each
coordinate. Synthetic point(3/7,0), manifold y=0, projection(x,y)=(x,0),
R=P+constant delta. Compare two composed residuals using Fraction and an
integer coordinate route; require25 exact matches. Horizontal perturbation
must cancel; any nonzero vertical perturbation must not. Deliberately replace
nonzero second error with zero; refusal required. No fitted values or pixels.

This examines a missing premise in the printed interpretation of
Qi et al CVPR2026 Eq1 and5–7. Constant delta has perfect spatial correlation,
but R(x) need not lie in the manifold. Counterexample does not invalidate
their image experiments or certify any physical propagation classifier.
No general no-go claim. Protocol receipt in work/robust_statistics/did_review.

Run `python -m experiments.origin_detection.reconstruction_audit.audit_cancellation`.

# Frozen-feature consistency audit — no real data

Audit the printed DCPTv1 frozen-encoder setting before choosing its loss as a
candidate. PDF2604.10102v1,Eq1–4 computes feature cosine before the trainable
head. For fixed encoder and fixed views,feature loss is constant in head
parameters. It cannot change the head gradient. This is a logical implication
of the printed setting,not a reproduction of uninspected author code/results.

Exact rational quadratic toy: adding a positive constant preserves differences,
stationary minimizer and objective ordering. CPU float64 PyTorch toy uses fixed
nonidentical feature pairs and a trainable2x2linear head;CE gradient with/without
feature cosine must be bit-exact. A trainable feature changes cosine gradient
and is deliberately rejected by the zero-gradient control. A constant uniform
head also makes detached-clean symmetricKL zero with zero head gradient;
classification loss is not claimed to have that same degeneracy.

Before generating the scientific toy receipt,run independent exact rational
controls and a planted nonconstant-term refusal. Then use the runtime overlay
with CPU tensors only,save script/test/PDF/runtime fingerprints and numerical
results. No CUDA computation,encoder weights,real images,head campaign training
or external communication. Owe author code and actual ablation protocol.

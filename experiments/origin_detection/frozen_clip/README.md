# Frozen CLIP representation — prospective sixteenth iteration

Question: can pretrained visual semantics provide class evidence surviving
recapture/recompression that fifteen statistical readouts did not? This is an
existing neural backbone with a fitted conventional classifier, not a non-neural
algorithm, novelty claim, retrained backbone or final detector.

Use existing local official CLIP ViT-L/14 archive with pinned SHA. No weight
download. Execute original archive.visual TorchScript directly, no eager
reconstruction. All encoder parameters frozen; one synthetic witness must give
finite768-dimensional features and exactly repeat. This is a software repeatability
check, not independent scientific validation. The preregistered eager-vs-JIT
parity control FAILED (max abs .0078125 > .002), before any real image decoded;
its pins are preserved in encoder_attempt1.json. Rebuilding was rejected instead
of widening tolerance; this prospective protocol supersedes reconstruction.
Two known-answer tests lock bicubic
short-side224/rounded center crop, CLIP normalization and unit-vector normalization.
Stored features are (L2unit+1)/2 in [0,1]; fit standardization removes this affine
storage. No labels, device tags or condition routing are encoder inputs.

Freeze CLIP768 and previous hybrid499+CLIP768=1267; ERM and group+pair heads,
plus source-shuffled hybrid/both control. Same seed/ridge/pair/temperature,
splits, threshold protocol and exact rate audits as stages13–15. No backbone,
layer, crop-size or regularization sweep. Fit6300 signed real images, raw/Q90
ordinary and selection-onlyQ70 =13860 vectors. Q90/Q70 are applied AFTER the
same long-edge256 preprocessing used by prior rounds, BEFORE CLIP short-side224
center crop. Decode stored orientation.60-image pilot first; costs include CUDA
synchronization. PIL decode is not included in reported encoder timing, so this
is not the final end-to-end latency evidence.

All RR/Chimera data remain exposed development; reserved stays sealed. Frozen
pretraining content overlap unknown. Historical55%/5pp gates only screen;
provisional80%/2pp and independent real data plus decode-inclusive latency still
needed. Candidate failure continues overall goal. No final source-model training.

```powershell
.venv/Scripts/python.exe -m pytest tests/detection/test_frozen_clip.py -q
$env:PYTHONPATH = 'src;.'
 .venv/Scripts/python.exe -c "import sys,runpy; sys.path.append('../envs/bfree/Lib/site-packages'); sys.argv=['run','--pilot','60']; runpy.run_module('experiments.origin_detection.frozen_clip.run_iteration',run_name='__main__')"
 .venv/Scripts/python.exe -c "import sys,runpy; sys.path.append('../envs/bfree/Lib/site-packages'); sys.argv=['run','--prepare']; runpy.run_module('experiments.origin_detection.frozen_clip.run_iteration',run_name='__main__')"
.venv/Scripts/python.exe -m experiments.origin_detection.frozen_clip.run_iteration
```

Runtime: repository NumPy/Pillow/OpenCV/classical environment takes precedence;
the existing bfree site-packages is appended only for CUDA torch and dependencies.
No environment modification. Runtime module paths and versions are recorded.

Runtime correction before real pixels: the official JIT first vs second optimized
execution differed by max .0078125; later optimized repeats agreed. Five
nonoptimized executions had four exact adjacent repeats. Production explicitly
uses optimized_execution(False) for both controls and extraction. Shape/finite/
exact repeat gates stay unchanged. See jit_repeat_audit.json; this also means the
earlier eager-vs-JIT discrepancy cannot establish an architectural bug.

# Released B-Free full representation — prospective iteration44

Register before new feature pilots/extraction/fitting. User final criterion:
BA>=80%, same-source decline<=2pp, and independent real validation; live
decode/preprocess speed must also be reported. No candidate yet meets it.

Read B-Free CVPR2025 main pp4–7 and supplement pp1–3, released wrapper and
normalization; remaining paper pages/citation graph owed. Author end-to-end
fine-tuned DINOv2-reg is distinct from earlier original pretrained DINO/CLIP
and CuRe. We freeze the released B-Free parameters, then fit only conventional
heads; not a new encoder or reproduction of authors' training. No original
RR/ReWIND full baseline rerun. Forward during feature capture retains the
author score as a software witness, not an additional baseline campaign.

Pin local vendor c6a9f898782fb466b29af01f21960b67415afb0e, source blobs/config,
weight SHA5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947,
and installed timm/Torch preprocessing. Original504 patch-space five crops;
stored RGB orientation, official normalization, tile64 and audited crop-first
>8MP path. FP32 CUDA batch1, TF32off, Torch4/OpenCV1; no AMP, resizing changes
or crop-count changes. Read-only pre-head hook captures5×768 pooled vectors;
mean on device then exact float64 cast. No L2/bounding, crop variance or new
JPEG policy. Linear-head algebra checked within1e-5; original forward score
and single repeat must be bit-exact before interpreting features.

First known five-crop averages and invalid finite/schema controls. Then flat,
channel-separated, spatial synthetic images (small wrap/large crop cases),
repeated extraction vs hook-disabled direct official scoring. Deliberately
changed score must be caught. Small raw60-file pilot chosen from signed CuRe
pilot's already exposed selection filenames, repeated raw features/scalar,
native SHA/dimensions and coverage checks. Cost budget pilot<=500ms median
feature pass (initial load/hash excluded); refuse full extraction otherwise.
This is an engineering gate, not measured speed or a phone requirement.

Full same6300files,15120vectors:7560fit/2520threshold weak raw/Q90;
5040selection four variants. Source roles1260/420/420 unchanged. Strong Q70
and Q60 remain evaluation-only. Immutable pilot/code/runtime and native
digests required. Progress is logged incrementally; CSV is written after full
coverage in an attached session only;
no daemon or background helper. Preserve failure, do not overwrite receipts.

Three fixed768D heads: sourceT.1,meanT0,balanced source_null assignment30.
Domain half/source/view equal,ridge.01/floor.001,LBFGS500/gradient<=1e-5,
seed20261006/bootstrap2000.24weak threshold groups,60 metrics/130pairs.
No selection-dependent architecture/width/ridge/Q sweep. Queries only final
pixels; filenames used for audit,never classifier input. Pilot establishes
interface/cost only; complete heads establish development evidence only.

Exact query/cached margin controls, null four raw processed AUC intervals,
integer/Fraction and corrupt BA, emitall/wrong twin preserved. Independent
external pixels unopened; RR reserved sealed; repeated RR/Chimera development
and B-Free pretraining overlap not fully audited. No released weight/source
vendoring into public repository. Goal remains OPEN through candidate failure.

# ImageNet-ES 物理干预

[返回实验目录](../README.md)

真实光圈/ISO 变化、响应拟合和跨内容留出。

所有 Python 命令从仓库根目录以 `python -m experiments.imagenet_es.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [audit_imagenet_es_remote_index.py](audit_imagenet_es_remote_index.py) | Audit ImageNet-ES's official remote ZIP central directory via HTTP Range. |
| [evaluate_imagenet_es_aperture_pairs.py](evaluate_imagenet_es_aperture_pairs.py) | External JPEG-level aperture intervention check on fixed-source pairs. |
| [evaluate_imagenet_es_iso_aperture.py](evaluate_imagenet_es_iso_aperture.py) | Check aperture/ISO interventions at the JPEG layer on frozen sources. |
| [evaluate_imagenet_es_tone_holdout.py](evaluate_imagenet_es_tone_holdout.py) | Apply a frozen effective JPEG law to never-fit ImageNet-ES source classes. |
| [extract_imagenet_es_control_pairs.py](extract_imagenet_es_control_pairs.py) | Fetch a bounded set of official ImageNet-ES same-source aperture pairs. |
| [fit_imagenet_es_effective_tone.py](fit_imagenet_es_effective_tone.py) | Falsify a one-parameter-linked effective JPEG tone law on screen pairs. |
| [freeze_imagenet_es_aperture_registry.py](freeze_imagenet_es_aperture_registry.py) | Freeze the 20 ImageNet-ES aperture-development groups with local file hashes. |

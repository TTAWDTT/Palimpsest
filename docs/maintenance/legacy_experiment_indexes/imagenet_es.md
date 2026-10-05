> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# ImageNet-ES 物理干预

[返回实验目录](../../../experiments/README.md)

真实光圈/ISO 变化、响应拟合和跨内容留出。

所有 Python 命令从仓库根目录以 `python -m experiments.imagenet_es.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [audit_imagenet_es_remote_index.py](../../../experiments/data_preparation/imagenet_es/audit_remote_index.py) | Audit ImageNet-ES's official remote ZIP central directory via HTTP Range. |
| [evaluate_imagenet_es_aperture_pairs.py](../../../experiments/screen_capture/exposure_response/imagenet_es/evaluate_aperture_pairs.py) | External JPEG-level aperture intervention check on fixed-source pairs. |
| [evaluate_imagenet_es_iso_aperture.py](../../../experiments/screen_capture/exposure_response/imagenet_es/evaluate_iso_aperture.py) | Check aperture/ISO interventions at the JPEG layer on frozen sources. |
| [evaluate_imagenet_es_tone_holdout.py](../../../experiments/screen_capture/exposure_response/imagenet_es/evaluate_tone_holdout.py) | Apply a frozen effective JPEG law to never-fit ImageNet-ES source classes. |
| [extract_imagenet_es_control_pairs.py](../../../experiments/data_preparation/imagenet_es/prepare_control_pairs.py) | Fetch a bounded set of official ImageNet-ES same-source aperture pairs. |
| [fit_imagenet_es_effective_tone.py](../../../experiments/screen_capture/exposure_response/imagenet_es/fit_effective_tone.py) | Falsify a one-parameter-linked effective JPEG tone law on screen pairs. |
| [freeze_imagenet_es_aperture_registry.py](../../../experiments/data_preparation/imagenet_es/prepare_registry_aperture_registry.py) | Freeze the 20 ImageNet-ES aperture-development groups with local file hashes. |

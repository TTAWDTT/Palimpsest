> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# Dragotti 拍屏

[返回实验目录](../../../experiments/README.md)

跨相机配对、几何及衍射频率预算。

所有 Python 命令从仓库根目录以 `python -m experiments.dragotti.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [analyze_dragotti_probe.py](../../../experiments/screen_capture/geometry/dragotti/audit_pairs.py) | Audit a small same-content, cross-camera Dragotti recapture probe. |
| [build_dragotti_audit_bundle.py](../../../experiments/data_preparation/dragotti/prepare_audit_bundle.py) | Freeze the small verified Dragotti Range probe as one machine-readable record. |
| [check_dragotti_diffraction_alias_budget.py](../../../experiments/screen_capture/aliasing/dragotti/audit_diffraction_alias_budget.py) | Conditional diffraction/alias budget for Dragotti EOS600D recapture. |
| [merge_dragotti_cross_scene.py](../../../experiments/screen_capture/geometry/dragotti/evaluate_cross_scene.py) | Package verified Dragotti cross-scene probe records for inspection. |
| [probe_dragotti_alias.py](../../../experiments/screen_capture/aliasing/dragotti/run_alias.py) | Conservative screen-lattice alias probe on verified Dragotti color-chart crops. |
| [probe_dragotti_diffraction_otf.py](../../../experiments/screen_capture/aliasing/dragotti/run_diffraction_otf.py) | Conditional ideal diffraction cutoff for a published recapture setting. |
| [probe_dragotti_predicted_alias_spectrum.py](../../../experiments/screen_capture/aliasing/dragotti/run_predicted_alias_spectrum.py) | Exploratory screen-grid alias-band probe on verified real recaptures. |
| [probe_dragotti_range.py](../../../experiments/data_preparation/dragotti/prepare_range_members.py) | Extract a few verified members from Dragotti's public ZIP via HTTP Range. |

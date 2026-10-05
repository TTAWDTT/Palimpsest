> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# 投影相机

[返回实验目录](../../../experiments/README.md)

CompenNet/CompenNet++ 几何、光度和跨设置实验。

所有 Python 命令从仓库根目录以 `python -m experiments.projector.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [audit_compennet_full.py](../../../experiments/data_preparation/compennet/audit_full.py) | Audit the complete official CompenNet projector-camera benchmark archive. |
| [benchmark_projector_forward_speed.py](../../../experiments/projection_capture/chart_response/compennet/benchmark_forward_speed.py) | CPU simulation-kernel timing on one 256px CompenNet input. |
| [compare_compennetpp_light_setups.py](../../../experiments/projection_capture/chart_response/compennet/evaluate_light_setups.py) | Compare true structured-light decodes on two cloud_np CompenNet++ setups. |
| [decode_compennetpp_sl_one_setup.py](../../../experiments/projection_capture/structured_light/compennetpp/prepare_sl_one_setup.py) | Decode one real projector-camera Gray-code capture sequence. |
| [diagnose_projector_residuals.py](../../../experiments/projection_capture/chart_response/compennet/evaluate_residuals.py) | Diagnose where uniform-chart prediction fails on real projected textures. |
| [download_compennet.ps1](../../../experiments/data_preparation/compennet/prepare.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [evaluate_compennetpp_cross_setup_display.py](../../../experiments/projection_capture/raw_response/compennetpp/evaluate_cross_setup_display.py) | Does the source-image display transform transfer to a second real setup? |
| [evaluate_compennetpp_display_fit_identifiability.py](../../../experiments/projection_capture/raw_response/compennetpp/evaluate_display_fit_identifiability.py) | Separate display-placement optimization from photometric-model error. |
| [evaluate_compennetpp_geometry_surrogates.py](../../../experiments/projection_capture/structured_light/compennetpp/evaluate_geometry_surrogates.py) | Hold out camera tiles when fitting effective projector-camera geometry. |
| [evaluate_compennetpp_neutral_tone.py](../../../experiments/projection_capture/raw_response/compennetpp/evaluate_neutral_tone.py) | Predefined gray-ramp tone hypotheses on frozen CompenNet++ test pairs. |
| [evaluate_compennetpp_raw_forward_pilot.py](../../../experiments/projection_capture/raw_response/compennetpp/evaluate_raw_forward_pilot.py) | Pilot forward projection test in native camera coordinates. |
| [extract_compennetpp_sl_one_setup.py](../../../experiments/data_preparation/compennetpp/prepare_sl_one_setup.py) | Extract 42 structured-light source/camera pairs via CRC-verified ZIP Range. |
| [probe_compennet_range.py](../../../experiments/data_preparation/compennet/run_range.py) | Read official CompenNet ZIP directory over verified HTTP Range only. |
| [probe_compennetpp_range.py](../../../experiments/data_preparation/compennetpp/run_range.py) | Inspect author CompenNet++ ZIP central directory over HTTP Range only. |
| [probe_compennetpp_raw_reference_pairs.py](../../../experiments/data_preparation/compennetpp/run_raw_reference_pairs.py) | Range/CRC audit of a few same-setup projector and raw-camera pairs. |
| [probe_compennetpp_structured_light.py](../../../experiments/data_preparation/compennetpp/run_structured_light.py) | CRC-probe structured-light inputs and unwarped captures without 11 GB ZIP. |
| [probe_projector_channel_basis.py](../../../experiments/projection_capture/chart_response/compennet/run_channel_basis.py) | Falsify an additive projector-channel/surface-mixing hypothesis. |
| [probe_projector_channel_transfer.py](../../../experiments/projection_capture/chart_response/compennet/run_channel_transfer.py) | Transfer pure-channel tone curves while recalibrating four target charts. |
| [probe_projector_chart_model.py](../../../experiments/projection_capture/chart_response/compennet/run_chart_model.py) | Test chart-only projector forward models on held-out textured inputs. |
| [probe_projector_full_lut.py](../../../experiments/projection_capture/chart_response/compennet/run_full_lut.py) | High-capacity pixelwise 3D-LUT control for the chart-only forward probe. |
| [probe_projector_transfer.py](../../../experiments/projection_capture/chart_response/compennet/run_transfer.py) | Can a chart-calibrated effective projector response transfer to another setup? |

> 迁移前分类的历史索引。当前入口见 [实验工作流](../../../experiments/README.md)；本文件的数量和分组描述保留为历史记录。

# Chimera 拍屏

[返回实验目录](../../../experiments/README.md)

外部真实配对、固定检测器迁移、频谱与过程反证。

所有 Python 命令从仓库根目录以 `python -m experiments.chimera.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [analyze_chimera_highband_confounders.py](../../../experiments/screen_capture/spectral_structure/chimera/evaluate_highband_confounders.py) | Test whether native high-band excess tracks source content and scene. |
| [analyze_chimera_score_contraction.py](../../../experiments/origin_detection/score_transfer/chimera/evaluate_score_contraction.py) | Separate score compression from common offset on frozen Chimera development. |
| [audit_chimera_archive.py](../../../experiments/data_preparation/chimera/audit_archive.py) | Audit official Chimera Zenodo data archive without extracting it wholesale. |
| [check_chimera_content_pairs.py](../../../experiments/data_preparation/chimera/audit_content_pairs.py) | Check released Chimera filename pairing with image-content pHashes. |
| [download_chimera_data_ranges.py](../../../experiments/data_preparation/chimera/prepare_data_ranges.py) | Download the public Zenodo data archive with checked independent byte ranges. |
| [evaluate_chimera_bfree.py](../../../experiments/origin_detection/score_transfer/chimera/evaluate_bfree.py) | Evaluate source-paired B-Free performance on verified Chimera recaptures. |
| [evaluate_chimera_composite_photometry.py](../../../experiments/screen_capture/photometry/chimera/evaluate_composite_photometry.py) | Calibrate a shared RGB response after fixed post-crop geometry alignment. |
| [evaluate_chimera_effective_blur_proxy.py](../../../experiments/screen_capture/photometry/chimera/evaluate_effective_blur_proxy.py) | Test a frozen Gaussian effective-blur proxy after geometry/color calibration. |
| [evaluate_chimera_fixed_publication_geometry.py](../../../experiments/screen_capture/photometry/chimera/evaluate_fixed_publication_geometry.py) | Test whether one fixed post-crop affine warp transfers to unseen content. |
| [evaluate_chimera_simulation_score_transfer.py](../../../experiments/origin_detection/score_transfer/chimera/evaluate_simulation_score_transfer.py) | Compare fixed B-Free source-score changes on real vs composite proxy recaptures. |
| [evaluate_chimera_size_control.py](../../../experiments/origin_detection/score_transfer/chimera/evaluate_size_control.py) | Compare native and 256x256 Chimera recaptures under fixed B-Free weights. |
| [evaluate_chimera_virtual_screen_mac.py](../../../experiments/screen_capture/source_to_rgb/chimera/evaluate_virtual_mac.py) | Evaluate one frozen virtual forward chain against paired real Chimera Mac recaptures. |
| [fit_chimera_virtual_screen_effective_mac.py](../../../experiments/screen_capture/source_to_rgb/chimera/fit_virtual_effective_mac.py) | Fit only effective exposure/blur to real RGB observables on calibration IDs. |
| [freeze_chimera_simulation_split.py](../../../experiments/data_preparation/chimera/prepare_registry_simulation_split.py) | Freeze source-level, stratified Chimera splits before simulator calibration. |
| [materialize_chimera_composite_simulation.py](../../../experiments/screen_capture/photometry/chimera/prepare_composite_simulation.py) | Materialize the frozen geometry/color/blur proxy on development sources. |
| [materialize_chimera_virtual_screen_mac.py](../../../experiments/screen_capture/source_to_rgb/chimera/prepare_virtual_mac.py) | Render a frozen, unfitted screen-camera hypothesis on Chimera development. |
| [prepare_chimera_bfree_manifest.py](../../../experiments/data_preparation/chimera/prepare_bfree_manifest.py) | Decode all source and genuine recapture images and sign an inference manifest. |
| [prepare_chimera_size_control.py](../../../experiments/data_preparation/chimera/prepare_size_control.py) | Create a frozen 256x256 recapture control for Chimera B-Free evaluation. |
| [probe_chimera_native_bandpower.py](../../../experiments/screen_capture/spectral_structure/chimera/run_native_bandpower.py) | Contrast native recapture high-band power with pure digital resampling. |
| [probe_chimera_native_channel_coherence.py](../../../experiments/screen_capture/spectral_structure/chimera/run_native_channel_coherence.py) | Compare high-band RGB channel coherence in real recapture and digital controls. |
| [probe_chimera_native_frequency.py](../../../experiments/screen_capture/spectral_structure/chimera/run_native_frequency.py) | Search for repeatable native-pixel spectral peaks in true screen recaptures. |
| [probe_chimera_registration.py](../../../experiments/screen_capture/geometry/chimera/run_registration.py) | Probe content registration of Chimera published recaptures at common size. |
| [probe_chimera_residual_spectrum.py](../../../experiments/screen_capture/spectral_structure/chimera/run_residual_spectrum.py) | Probe unfitted real-minus-proxy spatial residuals on development sources. |
| [probe_chimera_screen_observables.py](../../../experiments/screen_capture/geometry/chimera/run_observables.py) | Measure preregistered output observables on nonreserved Chimera pairs. |
| [test_chimera_luma_sharpen_hypothesis.py](../../../experiments/screen_capture/photometry/chimera/evaluate_luma_sharpen_hypothesis.py) | Falsify simple post-tone luma sharpening against native Chimera recaptures. |

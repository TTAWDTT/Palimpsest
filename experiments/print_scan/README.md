# 打印、扫描与拍纸

[返回实验目录](../README.md)

DFD、DESCAN、CSGC、DIV2K-SCAN 的数据审计和模型验证。

所有 Python 命令从仓库根目录以 `python -m experiments.print_scan.<脚本名>` 运行。

| 脚本 | 用途（脚本原始说明） |
|---|---|
| [analyze_descan_print_signatures.py](analyze_descan_print_signatures.py) | Predefined, content-gated DESCAN print/scan observations; no model fit. |
| [audit_csgc2400.py](audit_csgc2400.py) | Full-entry audit of official CSGC 2400-spi paired archive. |
| [audit_descan_near_duplicates.py](audit_descan_near_duplicates.py) | Screen DESCAN Valid/Test clean patches for possible near duplicates. |
| [audit_descan_pairs.py](audit_descan_pairs.py) | Audit paired DESCAN-18K validation/test archives without extracting them. |
| [audit_dfd_all_image_metadata.py](audit_dfd_all_image_metadata.py) | Collect per-image header metadata from the fully verified DFD B/W archive. |
| [audit_dfd_color_fullpage_metadata.py](audit_dfd_color_fullpage_metadata.py) | Extract and inspect one complete DFD color TIFF per printer folder. |
| [audit_dfd_color_inventory.py](audit_dfd_color_inventory.py) | Inventory the verified DFD color archive without extracting 25 GB of scans. |
| [audit_dfd_color_links.py](audit_dfd_color_links.py) | Audit how cropped PNGs relate by name to DFD color whole-sheet TIFFs. |
| [audit_dfd_color_non_tiff_fullpages.py](audit_dfd_color_non_tiff_fullpages.py) | Verify DFD color's six exceptional whole-page PNG scans and encoded DPI. |
| [audit_dfd_color_scanner_pairs.py](audit_dfd_color_scanner_pairs.py) | Check image-content pairing for DFD's six exceptional full-page PNGs. |
| [audit_dfd_halftone.py](audit_dfd_halftone.py) | Audit DFD's public B/W scanned-sheet archive without unpacking it. |
| [audit_div2k_scan_100_pairs.py](audit_div2k_scan_100_pairs.py) | Full CRC, content-pair, and metadata audit for official DIV2K-SCAN XR test. |
| [audit_div2k_scan_test.py](audit_div2k_scan_test.py) | Audit public DIV2K-SCAN iPhone XR test archive and three source probes. |
| [check_print_scan_convergence.py](check_print_scan_convergence.py) | Check the prototype's internal fine-grid convergence on a virtual patch. |
| [compare_dfd_print_scan_prototype.py](compare_dfd_print_scan_prototype.py) | One fixed-condition sanity comparison, not a calibrated printer validation. |
| [csgc_bounded_power_tone.py](csgc_bounded_power_tone.py) | Diagnostic: can one bounded reflectance-plus-power curve explain CSGC? |
| [csgc_cross_spi_12_id_audit.py](csgc_cross_spi_12_id_audit.py) | Range audit 12 predetermined CSGC holdout IDs across 4800/9600 SPI. |
| [csgc_cross_spi_12_id_forward.py](csgc_cross_spi_12_id_forward.py) | Fixed-parameter CSGC forward transfer on 12 cross-SPI digital templates. |
| [csgc_cross_spi_forward_probe.py](csgc_cross_spi_forward_probe.py) | Small falsification probe: transfer 2400-spi composite parameters to other SPI. |
| [csgc_effective_channel.py](csgc_effective_channel.py) | Estimate a *composite* blur/linear-tone diagnostic on known CSGC pairs. |
| [csgc_forward_residuals.py](csgc_forward_residuals.py) | Probe held-out CSGC residual structure of the fixed forward prototype. |
| [csgc_nonlinear_tone.py](csgc_nonlinear_tone.py) | Test a bounded monotone *composite* transfer on held-out CSGC templates. |
| [csgc_residual_conditional_diagnostic.py](csgc_residual_conditional_diagnostic.py) | Diagnostic of which fixed-input structures explain CSGC forward residuals. |
| [dfd_resampling_peak_probe.py](dfd_resampling_peak_probe.py) | Check whether publication-like resizing can erase genuine print lattices. |
| [download_descan_calibration.py](download_descan_calibration.py) | Download and verify the public paired DESCAN-18K validation/test archives. |
| [download_dfd_color.ps1](download_dfd_color.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [download_div2k_valid_hr.ps1](download_div2k_valid_hr.ps1) | PowerShell 下载或续跑入口；会产生外部 I/O，须显式运行。 |
| [evaluate_csgc_forward.py](evaluate_csgc_forward.py) | Run physical-coordinate binary print-scan prototype on 850 held-out codes. |
| [index_csgc_multi_zip.py](index_csgc_multi_zip.py) | Index all three CSGC official ZIPs from verified HTTP Range tail fragments. |
| [probe_color_print_scan_descan.py](probe_color_print_scan_descan.py) | Fixed-parameter color-print prototype versus selected real DESCAN tiles. |
| [probe_csgc_cross_resolution.py](probe_csgc_cross_resolution.py) | Range-extract one CSGC source/scan pair per SPI and compare templates. |
| [probe_csgc_more_ids.py](probe_csgc_more_ids.py) | Spot-check additional cross-SPI CSGC pair mappings by HTTP Range + CRC. |
| [probe_dfd_color_fullpage_tile.py](probe_dfd_color_fullpage_tile.py) | Measure one color tile and blank-paper control in an 800-ppi DFD full scan. |
| [probe_dfd_color_matched_printers.py](probe_dfd_color_matched_printers.py) | Same-named color chart/driver setting across two HP CLJ5550 printer IDs. |
| [probe_dfd_color_multitile_transfer.py](probe_dfd_color_multitile_transfer.py) | Pre-fixed four-color ROI spectral peaks across D5/D6 same nominal print setup. |
| [probe_dfd_color_sample_spectra.py](probe_dfd_color_sample_spectra.py) | Exploratory 2D spectra of one selected DFD color scan per printer folder. |
| [probe_dfd_lattice_forward_transfer.py](probe_dfd_lattice_forward_transfer.py) | Calibrate one effective square halftone lattice on D5; predict D6 peaks. |
| [probe_dfd_scanner_pair.py](probe_dfd_scanner_pair.py) | Extract two scanner-labeled copies of one DFD print for a visual audit. |
| [probe_dfd_tile_control.py](probe_dfd_tile_control.py) | Compare a printed uniform tile with blank paper in one DFD scan. |
| [probe_dfd_tone_lattice.py](probe_dfd_tone_lattice.py) | Observed peak stability across gray tiles in two matched DFD P3 scans. |
| [probe_div2k_print_hypotheses_100.py](probe_div2k_print_hypotheses_100.py) | Fixed, uncalibrated print->camera hypotheses on audited DIV2K-SCAN pairs. |
| [probe_div2k_scan_originals.py](probe_div2k_scan_originals.py) | CRC-verified HTTP Range extraction of original DIV2K validation images. |
| [probe_photo_paper_div2k.py](probe_photo_paper_div2k.py) | Fixed continuous-tone print hypothesis on the three verified DIV2K-SCAN pairs. |
| [probe_print_camera_div2k.py](probe_print_camera_div2k.py) | Fixed-parameter, non-calibrated print->paper->camera probe on three real pairs. |
| [verify_csgc_pair_content.py](verify_csgc_pair_content.py) | Check that CSGC's same-number source/scan entries are content paired. |

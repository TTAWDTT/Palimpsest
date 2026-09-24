# Print-scan forward model: measurement contract

Status: research design plus restricted monochrome and color structural
prototypes in `origin_simulation/print_scan.py` and
`origin_simulation/color_print_scan.py`. A separate continuous-tone photo-paper
surface hypothesis lives in `origin_simulation/photo_paper.py`: effective
exposure spot, developed dye density, physical dye spread and frozen paper
granularity. Neither color surface has device-specific ICC/RIP, spectral,
chemical or optical calibration. Publication-stage crop, resampling and codec
are separately represented in `origin_simulation/publication.py`. A composite channel has been fitted on 100
CSGC binary codes and evaluated on 850 unseen codes, but the printer, paper,
scanner and publication-stage effects have not been individually calibrated.

The DFD 800-ppi gray scan control shows why these stages must remain separate:
16 genuine printed 512-pixel tone patches have a median annular strongest-peak
share of 0.211; fourfold Lanczos downsampling of those same physical regions
reduces it to 0.007. A weak peak in a released aligned crop therefore cannot
identify its underlying RIP family without physical and publication scale.

## Conditional chain

Given known digital source `x` and a recorded device configuration, model:

`x -> print color management / tone curve -> raster image processor (RIP)
halftoning -> toner or ink deposition -> paper reflectance and subsurface
scatter -> scanner illumination / optics -> scanner sampling -> scanner ISP
-> output encoding`.

The camera-of-print branch now shares the print and paper stages through
`simulate_color_print_surface()`, then uses scene illumination, paper pose
and the restricted camera optics/RAW/ISP in `origin_simulation/print_camera.py`
instead of the flatbed scanner. A geometric warp of a scanner image is not a
physical substitute for that branch. The camera model has point + ambient
lighting and Lambertian paper only; real lab photo paper, specular BRDF,
phone ISP and original camera settings remain uncalibrated. Public DIV2K-SCAN
iPhone XR test PNGs are real print-phone captures but have already undergone
perspective correction/alignment, so they constrain only the published output.
The alternate `simulate_photo_paper_surface()` can feed the same camera chain.
This is a separate **hypothesis**, since DIV2K-SCAN says only "professional
photo lab" and does not identify its print technology. A physically valid
family cannot be selected by tuning its final brightness or texture to three
published phone PNGs.

For a monochrome controlled patch, a minimal model can be written as:

`u = T_driver(x)`

`h(r) = 1[spot(R_angle r / p) < u(r)]`

`d(r) = D_print(h(r), theta_deposition)`

`rho(r) = R_paper_ink(d(r), theta_paper)`

`y[m,n] = Q_scanner{integral rho(r) * L(r) * PSF_scanner(r-r_mn) dr + noise}`.

Here `p` is the physical halftone lattice period, `D_print` may include
mechanical dot growth or loss, `R_paper_ink` includes paper scattering, and
`r_mn` is the scanner sample position. In a color implementation, channel
separation, channel-specific screen angles/registration and spectral
reflectance must be represented or kept as explicit unknown composite terms.

## Identifiability and status of parameters

| Parameter | Evidence needed | What cannot be inferred from a final TIFF |
|---|---|---|
| Driver / RIP mode, print resolution | Spool settings or trusted per-print metadata; controlled mode changes | Exact halftone algorithm from one spectral peak |
| Halftone lattice and angle | High-resolution scan of uniform digital tone patches; multiple locations and tones | Whether the strongest FFT peak is the fundamental rather than a harmonic |
| Tone transfer and dot gain | Known input gray values, blank paper, solid ink, repeated measured patches | Unique separation of RIP transfer, mechanical dot gain and paper optical dot gain |
| Paper scatter | Reflectance point/line-spread measurement or well-constrained composite MTF | A material-specific paper PSF from one compressed scan |
| Scanner PSF, pitch and orientation | Scanner calibration target, true dpi, raw scan frame, rotated rescans | Physical scanner MTF after registration and 1024-pixel cropping |
| Scanner color/noise pipeline | Same sheet scanned repeatedly with locked settings, ideally linear data | RAW scanner noise from an 8-bit TIFF alone |

## Evidence roles and splitting

- **DFD grey halftone sheets**: mixed 600/800-ppi full-sheet physical scans with
  printer/driver/setting tokens. Use fixed uniform patches to constrain
  observable lattice orientation/frequency and compare device/configuration
  changes. If original P3 chart or exact print job is absent, DFD cannot by
  itself provide full `x -> y` calibration. The D5/D6 matched-model pair can
  serve as a *conditional* printer-unit holdout only if matching filename
  settings and paper/scanner conditions are verified. A single sheet per
  setting does not estimate within-setting repeatability.
  In the first locally verified D5 scan, a fixed 512-pixel ROI inside three
  different uniform gray tiles has a 45-degree spectral-peak/annular-median
  ratio of approximately 176–318; a blank-paper ROI has ratio 2.69 at the
  same bin. These ratios are *local scanner-output observations*, not printer
  line-screen specifications. Across ten D5/D6 tone tiles each, eight lighter
  tiles retain a ~141 lpi strongest peak, while the two darkest tiles per
  device have a weak ~83 lpi dominant peak. Therefore a peak estimator needs
  a tone-dependent confidence/negative-control rule. The full archive audit
  is recorded separately.
- **DESCAN-18K**: original digital page / physically scanned page pairs. The
  public 1024-pixel registered crops support task-level and frequency/color
  checks across scanner IDs, but remove full-page geometry and do not publish
  per-crop scan dpi. Separate source pages between calibration and test to
  prevent shared-layout/content leakage; keep scanner 1/2 test held out.
- **CSGC random binary codes**: 950 exact 100×100 references printed at 600
  dpi and prepared real scans at nominal 2400/4800/9600 spi. The 2400 archive
  is fully CRC-checked; 16 cross-resolution IDs have been range-checked.
  This supports a `binary_direct` printer-raster branch with known digital
  input. It is not a halftoned gray photograph. The published crops have
  TIFF DPI=72; use author acquisition settings and the 4/8/16 scale relation,
  not TIFF DPI, to set physical scale. It remains unclear whether the three
  resolution conditions re-scan the same physical sheets. Calibrate on code
  IDs 1–100 and keep 101–950 unseen; do not use these random codes as a final
  natural-vs-AI origin evaluation. With a single composite spatial/tonal fit,
  the actual forward prototype scores median Pearson 0.884 and RMSE 0.108 on
  the 850 held-out 2400-spi codes. The output has a systematic positive tone
  residual, especially over white reference bits. Twelve newly selected
  holdout IDs have also been probed across three SPI conditions without
  retuning; this is a small conditional transfer
  check, not identification of scanner resolution as the only changed cause.
  A 3x3 linear-neighborhood residual diagnostic reduces held-out median RMSE
  to 0.100, versus 0.0955 for a full local pattern table. This suggests
  repeatable spatial response missing from the Gaussian prototype but does
  not distinguish printer interactions from scanner resampling, registration,
  or archive post-processing. Local-neighborhood statistical PI models are
  already published; this diagnostic must not be presented as a new method.
- **VIPPrint**: printed/scanned natural/GAN face task data. Use only after
  verifying exact digital/scan pairing, archive completeness, and generator/
  printer splits. Its older face-GAN domain cannot certify modern general
  image-origin performance.
- **RRDataset redigital**: use only as a terminal downstream comparison until
  the four physical routes receive per-image labels; it cannot calibrate the
  print-scan stages by route.

## Falsifiable checks before detector training

1. At a fixed printer, known chart and scanner, change only driver/RIP mode.
   Verify whether the simulator predicts both the 2-D lattice shift and the
   tone/contrast change, not just a single fitted FFT bin. Include blank paper
   as a negative control for printed-dot peaks.
2. Scan the same physical sheet at two actual scanner resolutions and at a
   small rotation. The dot lattice stays in paper coordinates; the observed
   sampled frequency/angle must transform with scanner pitch/orientation.
   Digitally resizing one saved scan does not constitute this physical check.
3. Repeatedly scan one unchanged sheet to estimate scanner noise/registration
   variance independently of printer deposition variance.
4. Hold out a printer unit, driver setting, scanner, source page and full
   capture session as distinct generalization axes. A one-dimensional spectral
   match on calibration data is insufficient for process validity.
5. Compare paired real/synthetic residuals after geometric alignment and
   common terminal encoding. Report tone curves, lattice vector peaks,
   edge/MTF, patch variance and detector-score transfer separately, with
   confidence intervals by *physical print or source page*, not by correlated
   crops from one sheet.

Research basis: [DFD official dataset](https://dfd.inf.tu-dresden.de/dataset/),
[DESCAN-18K official repository](https://github.com/mlvc-lab/DESCAN-18K),
[paper-PSF optical dot gain study](https://www.jstage.jst.go.jp/article/nig1987/35/4/35_4_189/_article/-char/en),
and [electrophotographic printer model](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/jist/47/5/art00013).

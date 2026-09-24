# Projector-to-surface-to-camera forward model contract

Status: process design with real-pair access underway. No projector forward
module has passed device calibration. This contract separates what can be
measured from the public CompenNet PNGs from what requires native captures.

## Physical order

`digital source -> video compositor/fit -> projector color response and
spatiotemporal modulator -> projection optics and pose -> screen irradiance
-> surface reflectance / ambient light -> camera optics -> exposure/CFA/ISP
-> publication crop/warp/encoding`.

For one wavelength band and a planar Lambertian screen, a restricted model is

`L_out(r,t) = rho(r)/pi * [E_ambient(r,t) + E_projector(r,t)]`.

`E_projector` depends jointly on projector pixel coordinate, DMD/LED timing,
point-spread, ray angle, throw distance and vignetting. The camera then
integrates `L_out` over its pixel aperture, wavelength sensitivity and row
exposure interval before CFA and ISP. A colored surface multiplies incident
projected light in each spectral band; it is not a post-hoc additive texture.
This three-band expression is only an effective RGB approximation to an
unknown spectral BRDF. On a glossy or curved screen, the Lambertian model
must fail or be replaced. Projector black still contributes a nonzero light
floor, while ambient light persists even when projected input is black.

Texas Instruments' [DLP display guide](https://www.ti.com/lit/an/dlpa059h/dlpa059h.pdf)
describes the DMD image and throw geometry. The [DLP optical-engine
study](https://www.mdpi.com/2304-6732/10/5/559) separates light source, DMD,
projection lens and illumination uniformity. A [DLP temporal-imaging
experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC3824265/) shows that
micromirror time slots can interact with rolling-shutter exposure. Therefore
temporal color/brightness stripes, if present, must depend on exposure phase,
shutter scan and projector schedule. They must not be sampled as an image-wide
independent random overlay.

## Parameter, evidence and identifiability

| Stage | Independent evidence | Public CompenNet photometric 256 px data |
|---|---|---|
| Digital fit and pixel lattice | Exact frame sent to projector, native output mode | Common input `train`/`test` is known, but original 800x600 DMD frame is not retained |
| Projector tone/black floor | Flat RGB chart including black, locked brightness/contrast | 126 reference inputs and their captured outputs can constrain **composite** response |
| Projection PSF/pose | Dot/edge chart at multiple distances and native camera frames | 256 px `cam/warp` hides the original PSF and projective geometry |
| Surface reflectance/ambient | Same surface under projector black/white and ambient changes, ideally spectral/BRDF chart | Different surfaces and light/position groupings plus reference output constrain combined modulation |
| Camera response | RAW and locked exposure/white balance; repeat frames | Released PNG and published manual settings do not identify CFA, shot noise or ISP |
| DLP time | High-speed photodiode or rolling-shutter captures with known row timing | One still 256 px warped output cannot uniquely infer DMD bit-plane schedule |
| Publication | Original camera frame plus published crop/warp | `cam/warp` is explicitly post-registered; its interpolation must be modeled separately |

The official [CompenNet paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_End-To-End_Projector_Photometric_Compensation_CVPR_2019_paper.html)
uses 24 setups, 500 training inputs and 200 held-out test inputs per setup;
the [repository](https://github.com/BingyaoHuang/CompenNet) states that
camera images are warped to the projector view. These pairs can falsify a
surface-aware **effective radiometric** simulator. A successful color/texture
match does not identify the projector, surface and camera factors separately.

The larger author's CompenNet++ public ZIP remote directory contains both
`cam/raw` and warped branches. A CRC-verified sample of `cam/raw/test`
is 640x480 RGB PNG with no readable image metadata, and the author's archive
README defines it as unwarped camera output: **not Bayer sensor RAW**. The
sample is a curved textured screen with large saturated areas; a plain SIFT
homography failed, so geometrical validation needs an explicit nonplanar
surface model or captured structured-light reference. The entire 11 GB ZIP
has not yet been downloaded or fully CRC audited. A single unwarped PNG
still cannot identify the DLP time schedule or true sensor response.

## Falsification order

1. Audit full package integrity, paired IDs and image dimensions. Split by
   source input ID; source images repeat across settings. Hold out entire
   surfaces or light/position setups separately.
2. Fit projector black floor, global response and a *single fixed* surface
   gain from reference colors on a calibration setting. Freeze them before
   predicting the same setting's 200 textured test inputs.
3. Compare held-out tone curves, residual spectra, spatial error by surface
   material, and color interactions against `resize + tone curve`, per-pixel
   affine and post-hoc texture controls. Use the exact same publication warp.
4. If native unwarped PNGs are verified, test chart corner projection and
   border/edge MTF under altered position. Do not call this RAW/ISP validation.
5. Only a separately controlled projector/camera grid with DLP timing,
   camera row timing and RAW allows a claim about temporal or sensor stages.

RRDataset's mixed `redigital` images lack per-image route/device labels.
Use RR only for terminal downstream robustness after these route-specific
checks; its JPEG subsampling is not a projector label.

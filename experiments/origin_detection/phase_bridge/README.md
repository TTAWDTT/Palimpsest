# DCT sign and global Fourier phase — numerical scope check

Exploratory mechanism witness, not detector development data or prospective
performance evidence. CPTFormer motivates complex image phase using JPEG block
DCT quantization. A surviving real DCT coefficient retains its sign (phase0/pi);
that fact alone does NOT establish preservation of the reconstructed block's
complex global Fourier phase. No claim that all author empirical improvements
are invalid, or that every Fourier phase changes under JPEG.

One8x8 orthonormal inverseDCT block: coeff00=1024,01=6.1,02=9.9,others0.
Quantization entries01=11,02=10,others1 (only nonzero DC entry is unaffected).
Round/dequantize to1024,11,10;all three original nonzero signs survive. Pixels
stay inside[0,255];use floating block operator,NO integerpixelrounding,color
transform,chroma subsampling,fullJPEGfileorphysicalcapture simulation claim.
Observe Fourier index(0,1) before and after. Cross-check SciPy IDCT+NumPyFFT
against explicit cosinebasis+scalarcomplexDFT. They share the chosen witness,
so this is two arithmetic paths,not independent investigator verification.
Planted zero globalphasechange must be rejected. Save exact inputs,library
versions,scriptSHA,numericalresiduals and scope. No theorem certificate.

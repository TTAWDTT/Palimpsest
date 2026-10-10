# Congruence distance versus fixed-reference decisions

Before using SPD distance as an image readout, check the statement it actually
supports. The affine-invariant distance preserves d(BAB^T,BCB^T) for common
invertible B, not d(A,BCB^T) with A fixed. A two-class diagonal covariance example
preserves pair distances when both references transform, yet flips a REAL query
when only inference data transform. No EEG or image data are involved.

Known diagonal ratios checked with Fraction, Decimal50/100 and SciPy generalized
eigenvalues. Wrong 'unchanged decision/BA' controls must refuse. This does not
prove a physical camera channel or rule out useful covariance classification.

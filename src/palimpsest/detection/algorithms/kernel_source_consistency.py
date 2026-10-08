"""Fit-local random Fourier readout with established source consistency.

This tests nonlinear information in a fixed representation. Kernel approximation
or small score variance is not a guarantee of correct propagated-image decisions.
"""

from dataclasses import replace

import numpy as np

from .kernel_readout import fit_fourier_map, KernelRule
from .consistent_source_risk import fit_consistent_source_risk


def fit_kernel_source_consistency(values, labels, weights, sources, *, feature_names,
                                  strength, consistency_groups=None, frequency_count=256,
                                  seed=20261006, manifest_sha=''):
    """Learn mapping moments only from the supplied training fold."""
    x = np.asarray(values, float)
    mapper = fit_fourier_map(x, weights, feature_names=feature_names,
                            frequency_count=frequency_count, seed=seed, scale_floor=.001)
    transformed = np.c_[x, mapper.transform(x)]
    names = tuple(feature_names)+mapper.output_names
    head, diagnostic = fit_consistent_source_risk(transformed, labels, weights, sources,
        strength=strength, consistency_groups=consistency_groups, feature_names=names,
        temperature=.1, ridge=.01, scale_floor=.001, maximum_iterations=2000,
        gradient_tolerance=1e-5, manifest_sha=manifest_sha)
    rule = KernelRule(mapper, head, concatenate_input=True)
    diagnostic.update({'input_dimensions':x.shape[1], 'mapped_dimensions':transformed.shape[1],
                       'kernel_gamma':mapper.gamma, 'frequency_count':frequency_count,
                       'mapping_seed':seed, 'mapping_moments_scope':'Supplied training fold only'})
    return rule, diagnostic


def calibrate_kernel_rule(rule, views, calibrate):
    """Use the existing head threshold policy on exactly mapped threshold views."""
    mapped = {}
    for key, (records, matrix, labels) in views.items():
        transformed = rule.mapper.transform(matrix)
        if rule.concatenate_input:transformed = np.c_[matrix, transformed]
        mapped[key] = records, transformed, labels
    head, diagnostic = calibrate(rule.readout, mapped)
    return replace(rule, readout=head), diagnostic

"""Explicit training-row context for algorithms that need processing pair graphs.

Context is never inferred from feature values or supplied at prediction time.
Legacy numeric fitters continue to receive their existing five/six arguments.
"""

from dataclasses import dataclass

from .calibrated_readout_campaign import CampaignMethod


@dataclass(frozen=True)
class RecordAwareCampaignMethod(CampaignMethod):
    record_aware: bool = True


def bank_budgets(method):
    expected = getattr(method, 'expected_banks', 12)
    pilot = getattr(method, 'pilot_banks', 1)
    if (any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (expected, pilot))
            or expected < pilot):
        raise ValueError('Invalid declared numeric bank budget')
    return expected, pilot


@dataclass(frozen=True)
class BudgetedRecordAwareCampaignMethod(RecordAwareCampaignMethod):
    expected_banks: int = 12
    pilot_banks: int = 1

    def __post_init__(self):
        bank_budgets(self)


def fit_training_rows(fit, values, labels, weights, sources, parameter, *, records,
                      record_aware=False, wrong=None):
    if len(records) != len(values) or any(r['role'] != 'fit' for r in records):
        raise ValueError('Fitter context must contain exactly its training rows')
    kwargs = {'records': tuple(records)} if record_aware else {}
    if wrong is None:
        return fit(values, labels, weights, sources, parameter, **kwargs)
    return fit(values, labels, weights, sources, parameter, wrong, **kwargs)

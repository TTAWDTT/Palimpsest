"""Known population controls cover dependencies hidden by scalar phase energy."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.origin_detection.non_neural.generator_mechanisms.audit_polyphase import moment_controls
from palimpsest.evaluation.polyphase import linear_population_moments, periodic_upsampling_matrix


SPEC = Path(__file__).resolve().parents[2] / 'configs/evaluation/polyphase_operator_controls.json'


def test_known_populations_and_each_planted_corruption():
    result = moment_controls(json.loads(SPEC.read_text(encoding='utf-8')))
    assert result['positive_fixtures'] == result['negative_controls_caught'] == 7
    cases = {c['name']:c['exact'] for c in result['fixtures']}
    filtered = cases['repeat_then_two_tap_mean']
    assert filtered['difference_energies'] == ['1/2','1/2']
    assert filtered['variances'] == ['1/2','1']
    balanced = cases['stationary_covariance_non_gaussian']
    assert balanced['second_order_stationary']
    assert balanced['centered_fourth_moments'] == ['1','59/27']


@pytest.mark.parametrize('mutation', ('duplicate','drop','tolerance','wrong_fourth'))
def test_spec_mutations_are_refused(mutation):
    spec = deepcopy(json.loads(SPEC.read_text(encoding='utf-8')))
    if mutation == 'duplicate': spec['fixtures'][1]['name']=spec['fixtures'][0]['name']
    if mutation == 'drop': spec['fixtures'].pop()
    if mutation == 'tolerance': spec['atol']=1.0
    if mutation == 'wrong_fourth': spec['fixtures'][3]['expected']['centered_fourth_moments'][1]='1'
    with pytest.raises(ValueError): moment_controls(spec)


@pytest.mark.parametrize('kernel,stride,length', (([1],0,4),([],2,4),([float('nan')],2,4),([1],True,4),([1],1.5,4)))
def test_invalid_operator_refused(kernel,stride,length):
    with pytest.raises(ValueError): periodic_upsampling_matrix(kernel,stride,length)


@pytest.mark.parametrize('covariance', (np.array([[1,1],[0,1]]),np.array([[1,2],[2,1]]),np.array([[np.nan,0],[0,1]])))
def test_invalid_population_covariance_refused(covariance):
    with pytest.raises(ValueError): linear_population_moments(np.eye(2),np.zeros(2),covariance)

"""Known trig score,portable/batch parity,immutable arrays and input refusal."""

import math
import numpy as np
import pytest
from palimpsest.detection.algorithms.kernel_readout import FourierMap,KernelRule
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.compiled_kernel import CompiledKernelRule


def test_known_compiled_score_and_portable_parity():
    mapper=FourierMap(('x','y'),(1,2),(2,3),((1,2),(-3,4)),7,.5)
    names=mapper.feature_names+mapper.output_names
    weights=(.1,.2,.4,-.5,.7,.3)
    head=StableRule(names,(0,)*6,(1,)*6,weights,.25,0,.01,threshold=.6)
    portable=KernelRule(mapper,head,True);compiled=CompiledKernelRule(portable)
    expected=.1*5+.2*(-1)+(.4-.5*math.cos(-10)+.3*math.sin(-10))/math.sqrt(2)+.25
    assert abs(compiled.score([[5,-1]])[0]-expected)<1e-14
    matrix=np.array([[5,-1],[1,2],[0,0]])
    np.testing.assert_array_equal(compiled.score(matrix),portable.score(matrix))
    np.testing.assert_array_equal(compiled.score(matrix),[compiled.score([r])[0] for r in matrix])
    assert compiled.threshold==.6 and compiled.feature_names==('x','y')
    with pytest.raises(ValueError):compiled._frequencies[0,0]=0
    for wrong in ([[1]],[[np.nan,0]],[1,2]):
        with pytest.raises(ValueError):compiled.score(wrong)

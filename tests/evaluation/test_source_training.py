"""Unequal source/view counts retain equal domain/source masses."""

import numpy as np
import pytest
from palimpsest.evaluation.source_training import weighted_source_arrays


def test_known_domain_and_source_mass_and_fit_roles():
    rows=[]
    for domain,count in [('a',1),('b',2)]:
        for source in range(count):
            for condition in ['original','physical']:
                for variant in ['raw','encoded']:
                    rows.append(dict(domain=domain,src=str(source),scene='all',condition=condition,
                                     variant=variant,role='fit',label='FAKE' if source else 'REAL',known=source))
    rows.append({**rows[0],'role':'selection','known':np.nan})
    args=dict(expected_source_counts={'a':1,'b':2},seed=7)
    x,y,weights,groups=weighted_source_arrays(rows,('known',),('raw','encoded'),**args)
    assert len(x)==12 and np.isclose(weights.sum(),1)
    np.testing.assert_allclose(np.bincount(groups,weights=weights),[.5,.25,.25])
    assert np.array_equal(y,[0]*8+[1]*4)
    repeated=weighted_source_arrays(rows[::-1],('known',),('raw','encoded'),**args)
    for first,second in zip((x,y,weights,groups),repeated):np.testing.assert_array_equal(first,second)
    with pytest.raises(ValueError,match='Incomplete'):
        weighted_source_arrays(rows[1:],('known',),('raw','encoded'),**args)

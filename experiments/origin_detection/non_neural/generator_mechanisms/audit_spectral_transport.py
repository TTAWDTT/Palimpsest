"""One-shot finite spectral transport controls, no image classification.

Use covariance from known periodic operators and separately transform the
postprocessing operator to check filter, shift and noise assumptions.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.evaluation.polyphase import (
    frequency_covariance,
    periodic_upsampling_matrix,
    spectral_coherence_magnitude,
)


def _check(actual, expected, name):
    if not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
        raise ValueError(f"Known spectral answer mismatch: {name}")


def controls():
    started=perf_counter()
    a=periodic_upsampling_matrix([1,1],2,4)
    covariance=a@a.T
    size=len(covariance)
    spectrum=frequency_covariance(covariance)
    _check(spectrum[2,6],-1j,'base C26')
    shifted=np.roll(covariance,(1,1),axis=(0,1))
    shift_spectrum=frequency_covariance(shifted)
    _check(shift_spectrum[2,6],1j,'shifted C26')
    _check(np.abs(shift_spectrum),np.abs(spectrum),'shifted full magnitudes')
    b=periodic_upsampling_matrix([0.5,0.5],1,size)
    filtered=b@covariance@b.T
    filtered_spectrum=frequency_covariance(filtered)
    transfer=np.fft.fft([0.5,0.5]+[0.0]*(size-2))
    _check(filtered_spectrum,transfer[:,None]*spectrum*transfer.conj()[None,:],'filter law')
    _check(filtered_spectrum[2,6],-0.5,'filtered C26')
    white=covariance+np.eye(size)
    white_spectrum=frequency_covariance(white)
    _check(white_spectrum,spectrum+np.eye(size),'independent white noise')
    _check(spectral_coherence_magnitude(covariance,2,6),1,'base coherence')
    _check(spectral_coherence_magnitude(white,2,6),0.5,'white noise coherence')
    nonstationary=covariance+np.diag([2.0,0.0]*4)
    nonstationary_spectrum=frequency_covariance(nonstationary)
    _check(nonstationary_spectrum[2,6],1-1j,'nonstationary noise C26')
    # Fixed lowpass with frequencies0,1,7; its real circular impulse response.
    mask=np.zeros(size); mask[[0,1,7]]=1
    kernel=np.fft.ifft(mask).real
    lowpass=periodic_upsampling_matrix(kernel,1,size)
    low_covariance=lowpass@covariance@lowpass.T
    low_spectrum=frequency_covariance(low_covariance)
    off_diagonal=low_spectrum-np.diag(np.diag(low_spectrum))
    _check(off_diagonal,0,'lowpass erases off-diagonal covariance')
    expected_energy=(4+np.sqrt(2))/8
    _check(np.trace(low_covariance)/size,expected_energy,'nonzero retained lowpass energy')
    balanced=periodic_upsampling_matrix([1,2/3,0,-1/3,0,2/3],2,3)
    balanced_spectrum=frequency_covariance(balanced@balanced.T)
    _check(balanced_spectrum-np.diag(np.diag(balanced_spectrum)),0,'balanced covariance')
    # Each false statement is explicitly checked and must throw.
    wrongs=(('shift sign',shift_spectrum[2,6],-1j),
            ('missing conjugation',filtered_spectrum,transfer[:,None]*spectrum*transfer[None,:]),
            ('noise invariant coherence',spectral_coherence_magnitude(white,2,6),1),
            ('nonstationary noise immunity',nonstationary_spectrum[2,6],-1j),
            ('lowpass keeps C26',low_spectrum[2,6],-1j))
    caught=[]
    for name,actual,wrong in wrongs:
        try: _check(actual,wrong,name)
        except ValueError: caught.append(name)
        else: raise ValueError(f'Wrong spectral statement survived: {name}')
    value=lambda z:{'real':float(np.real(z)),'imag':float(np.imag(z)),'magnitude':float(abs(z))}
    return {'base_C26':value(spectrum[2,6]),'shift_C26':value(shift_spectrum[2,6]),
            'filter_C26':value(filtered_spectrum[2,6]),'white_C26':value(white_spectrum[2,6]),
            'nonstationary_noise_C26':value(nonstationary_spectrum[2,6]),
            'base_coherence':spectral_coherence_magnitude(covariance,2,6),
            'white_coherence':spectral_coherence_magnitude(white,2,6),
            'lowpass_max_off_diagonal':float(np.max(abs(off_diagonal))),
            'lowpass_mean_square':float(np.trace(low_covariance)/size),
            'balanced_max_off_diagonal':float(np.max(abs(balanced_spectrum-np.diag(np.diag(balanced_spectrum))))),
            'negative_controls_caught':len(caught),'corruptions':caught,
            'elapsed_seconds':perf_counter()-started,'scope':'Known finite centered population covariance; no single-image estimator, AI specificity, JPEG or recapture validation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=controls()
    result['code_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
    import palimpsest.evaluation.polyphase as module
    result['operator_code_sha256']=sha256(Path(module.__file__).read_bytes()).hexdigest()
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()

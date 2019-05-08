import os
import numpy as np
import librosa
import scipy
import math
import gammatone
import librosa.display
import matplotlib.pyplot as plt
import gammatone.gtgram
import six

__all__ = ['power_to_db',
           'load_sound_gtccs',
           'gammaspectrogram',
           'gtcc']
		   
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db   : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
    """   
    
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def load_sound_gtccs(file_paths, n_channels):

    X,sr = librosa.load(file_paths, sr=11025, res_type='kaiser_fast')
    gammaspec = gammatone.gtgram.gtgram(wave = X, fs = sr, window_time = 0.025, hop_time = 0.010, channels = n_channels, f_min = 20)             
    S = power_to_db(gammaspec)     
    dct_type=2
    norm='ortho'
    gtccs = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_channels]
    return gtccs


	
def load_sound_mfccs(file_paths, n_mfcc):

    X,sr = librosa.load(file_paths, sr=11025, res_type='kaiser_fast')

    mfccs = librosa.feature.mfcc(y=X,sr=sr,n_mfcc=n_mfcc)
    
#    mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sr,n_mfcc=40).T,axis=0)                                 
    return mfccs

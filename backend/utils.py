import numpy as np

def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std

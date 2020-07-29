import numpy as np
from madmom.features.onsets import OnsetPeakPickingProcessor


def peak_picking(odf, thr):
    proc = OnsetPeakPickingProcessor(threshold=thr, smooth=0.05, pre_max=0.01, post_max=0.01, combine=0.0, delay=0.0, online=False, fps=100)
    if isinstance(odf, list):
        Onset_times_pred = [proc(fn) for fn in odf]
    else:
        Onset_times_pred = proc(odf)
    return Onset_times_pred


def thresholding(Y, thr):
    if isinstance(Y, list):
        Y_bin = []
        for k in range(len(Y)):
            Y_bin.append(np.asarray(Y[k] > thr, dtype=int))
    else:
        Y_bin = np.asarray(Y > thr, dtype=int)
    return Y_bin

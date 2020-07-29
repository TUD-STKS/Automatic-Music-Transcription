import numpy as np
import mir_eval
from ..post_processing.binarize_output import thresholding


def _midi_to_frequency(p):
    return 440. * (2 ** ((p-69)/12))


def get_mir_eval_rows(y, fps=100.):
    time_t = np.arange(len(y)) / fps
    freq_hz = [_midi_to_frequency(np.asarray(np.nonzero(row))).ravel() for row in y]
    return time_t, freq_hz


def determine_threshold(Y_true, Y_pred, threshold):
    measures = []
    for thr in threshold:
        Y_pred_bin = thresholding(Y_pred, thr)
        measures.append(eval_multipitch_tracking(Y_true=Y_true, Y_pred=Y_pred_bin))
    return measures


def eval_multipitch_tracking(Y_true, Y_pred):
    if isinstance(Y_true, list):
        measures = []
        for y_true, y_pred in zip(Y_true, Y_pred):
            ref_time, ref_freqs = get_mir_eval_rows(y=y_true)
            est_time, est_freqs = get_mir_eval_rows(y=y_pred)
            measures.append(mir_eval.multipitch.metrics(ref_time, ref_freqs, est_time, est_freqs))
        measures = np.mean(measures, axis=0)
    else:
        ref_time, ref_freqs = get_mir_eval_rows(y=Y_true)
        est_time, est_freqs = get_mir_eval_rows(y=Y_pred)
        measures = mir_eval.multipitch.metrics(ref_time, ref_freqs, est_time, est_freqs)
    return measures

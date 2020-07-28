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
        tmp_frame_measures = []
        for y_true, y_pred in zip(Y_true, Y_pred_bin):
            ref_time, ref_freqs = get_mir_eval_rows(y=y_true)
            est_time, est_freqs = get_mir_eval_rows(y=y_pred)
            tmp_frame_measures.append(mir_eval.multipitch.metrics(ref_time, ref_freqs, est_time, est_freqs))
        measures.append(np.mean(tmp_frame_measures, axis=0))
    return measures


def eval_multipitch_tracking(Pitches_ref, Pitches_res):
    if isinstance(Pitches_ref, list):
        measures = []
        for res_pitches, ref_pitches in zip(Pitches_res, Pitches_ref):
            measures.append(OnsetEvaluation(detections=res_pitches, annotations=ref_pitches))
        measures = OnsetSumEvaluation(measures)
    else:
        measures = OnsetEvaluation(detections=Pitches_res, annotations=Pitches_ref)
    return measures
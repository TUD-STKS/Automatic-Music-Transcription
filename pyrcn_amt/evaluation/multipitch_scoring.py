import numpy as np
from sklearn.metrics import accuracy_score
import mir_eval
from ..post_processing.binarize_output import thresholding


def _midi_to_frequency(p):
    return 440. * (2 ** ((p-69)/12))


def get_mir_eval_rows(y, fps=100.):
    time_t = np.arange(len(y)) / fps
    freq_hz = [_midi_to_frequency(np.asarray(np.nonzero(row))).ravel() for row in y]
    return time_t, freq_hz


def get_mir_eval_intervals(y, fps=100.):
    intervals_t = []
    freqs_hz = []
    for k in range(y.shape[1]):
        idx_onsets = np.argwhere(np.diff(y[:, k], axis=0) > 0)
        idx_offsets = np.argwhere(np.diff(y[:, k], axis=0) < 0)
        if len(idx_onsets) < len(idx_offsets) and idx_offsets[0] == 0:
            idx_offsets = idx_offsets[1:, :]
        elif len(idx_onsets) != len(idx_offsets):
            print("This should not occur!!!")
        for i in range(len(idx_onsets)):
            intervals_t.append([(idx_onsets[i][0]) / fps, idx_offsets[i][0] / fps])
            freqs_hz.append(_midi_to_frequency(k))
    return np.asarray(intervals_t), np.asarray(freqs_hz)


def determine_threshold(Y_true, Y_pred, threshold):
    measures = []
    for thr in threshold:
        Y_pred_bin = thresholding(Y_pred, thr)
        measures.append(eval_multipitch_tracking(Y_true=Y_true, Y_pred=Y_pred_bin))
        # measures.append(eval_music_transcription(Y_true=Y_true, Y_pred=Y_pred_bin))
    return measures


def determine_prediction_threshold(Y_true, Y_pred, threshold):
    measures = []
    for thr in threshold:
        Y_pred_bin = thresholding(Y_pred, thr)
        measures.append(eval_note_prediction(Y_true=Y_true, Y_pred=Y_pred_bin))
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


def eval_music_transcription(Y_true, Y_pred):
    if isinstance(Y_true, list):
        all_measures = []
        for y_true, y_pred in zip(Y_true, Y_pred):
            ref_intervals, ref_pitches = get_mir_eval_intervals(y=y_true)
            est_intervals, est_pitches = get_mir_eval_intervals(y=y_pred)
            all_measures.append(mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches))
        measures = [None] * len(all_measures[0].keys())
        keys = list(all_measures[0].keys())
        for k in range(len(measures)):
            measures[k] = np.mean([d[keys[k]] for d in all_measures])
    else:
        ref_intervals, ref_pitches = get_mir_eval_intervals(y=Y_true)
        est_intervals, est_pitches = get_mir_eval_intervals(y=Y_pred)
        all_measures = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)
        measures = [None] * len(all_measures.keys())
        keys = list(all_measures[0].keys())
        for k in range(len(measures)):
            measures[k] = all_measures[keys[k]]
    return measures


def eval_note_prediction(Y_true, Y_pred):
    if isinstance(Y_true, list):
        measures = []
        for y_true, y_pred in zip(Y_true, Y_pred):
            measures.append(accuracy_score(y_true=y_true, y_pred=y_pred))
        measures = np.mean(measures, axis=0)
    else:
        measures = accuracy_score(y_true=Y_true, y_pred=Y_pred)
    return measures

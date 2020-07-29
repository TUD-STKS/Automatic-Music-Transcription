import numpy as np
from madmom.evaluation.onsets import OnsetEvaluation, OnsetSumEvaluation
from ..post_processing.binarize_output import peak_picking


def determine_peak_picking_threshold(odf, threshold, Onset_times_ref):
    measures = []
    for thr in threshold:
        Onset_times_res = peak_picking(odf, thr)
        measures.append(eval_onset_detection(Onset_times_ref, Onset_times_res))
    return measures


def eval_onset_detection(Onset_times_ref, Onset_times_res):
    if isinstance(Onset_times_ref, list) or isinstance(Onset_times_ref, np.ndarray):
        measures = []
        for res_onsets, ref_onsets in zip(Onset_times_res, Onset_times_ref):
            measures.append(OnsetEvaluation(detections=res_onsets, annotations=ref_onsets))
        measures = OnsetSumEvaluation(measures)
    else:
        measures = OnsetEvaluation(detections=Onset_times_res, annotations=Onset_times_ref)
    return measures
from madmom.evaluation.onsets import OnsetEvaluation, OnsetSumEvaluation
from ..post_processing.binarize_output import thresholding


def determine_threshold(Y, threshold, Pitches_ref):
    measures = []
    for thr in threshold:
        Y_res = thresholding(Y, thr)
        measures.append(eval_multipitch_tracking(Pitches_ref, Y_res))
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
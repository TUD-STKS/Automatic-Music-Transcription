import numpy as np
from madmom.utils import quantize_events, quantize_notes
from madmom.audio.signal import smooth


def discretize_onset_labels(onset_labels: list, fps: float = 100., target_widening=True, length=None):
    onset_targets = quantize_events(events=onset_labels, fps=fps, length=length)
    if target_widening:
        onset_targets = smooth(onset_targets, np.asarray([0.5, 1.0, 0.5]))
    return onset_targets


def discretize_boundary_labels(boundary_labels: list, fps: float = 100., target_widening=True, length=None):
    boundary_targets = quantize_events(events=boundary_labels, fps=fps, length=length)
    if target_widening:
        boundary_targets = np.minimum(smooth(boundary_targets, np.asarray([0.5, 1.0, 0.5])), 1)
    return boundary_targets


def discretize_offset_labels(offset_labels: list, fps: float = 100., target_widening=True, length=None):
    offset_targets = quantize_events(events=offset_labels, fps=fps, length=length)
    if target_widening:
        offset_targets = smooth(offset_targets, np.asarray([0.5, 1.0, 0.5]))
    return offset_targets


def discretize_beat_labels(beat_labels: list, fps: float = 100., target_widening=True, length=None):
    beat_targets = quantize_events(events=beat_labels, fps=fps, length=length)
    if target_widening:
        beat_targets = smooth(beat_targets, np.asarray([0.5, 1.0, 0.5]))
    return beat_targets


def discretize_notes(note_labels: list, fps: float = 100., num_pitches=128, target_widening=True, length=None):
    note_targets = quantize_notes(notes=note_labels, fps=fps, num_pitches=num_pitches, length=length)
    if target_widening:
        note_targets = smooth(note_targets, np.array([0.25, 0.5, 0.25]))
    return note_targets

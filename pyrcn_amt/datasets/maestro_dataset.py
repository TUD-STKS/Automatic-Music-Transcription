import os
import csv
import numpy as np
from madmom.utils import combine_events
from madmom.io import midi


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the MAESTRO Dataset
    :param dataset_path:
    :return:
    """
    raise NotImplementedError


def load_dataset(dataset_path: str = None):
    """
    This function aims to load the MAESTRO Dataset.
    :param dataset_path:
    :return:
    """
    dataset_path = os.path.normpath(dataset_path)
    train_input = []
    train_output = []
    val_input = []
    val_output = []
    test_input = []
    test_output = []
    try:
        with open(os.path.join(dataset_path, "maestro-v2.0.0.csv"), encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for line in reader:
                if line["split"] == "train":
                    train_input.append(os.path.normpath(os.path.join(dataset_path, line["audio_filename"])))
                    train_output.append(os.path.normpath(os.path.join(dataset_path, line["midi_filename"])))
                elif line["split"] == "validation":
                    val_input.append(os.path.normpath(os.path.join(dataset_path, line["audio_filename"])))
                    val_output.append(os.path.normpath(os.path.join(dataset_path, line["midi_filename"])))
                elif line["split"] == "test":
                    test_input.append(os.path.normpath(os.path.join(dataset_path, line["audio_filename"])))
                    test_output.append(os.path.normpath(os.path.join(dataset_path, line["midi_filename"])))
                else:
                    raise("Unknown split: {0}".format(line["split"]))
    except FileNotFoundError:
        raise("Dataset not found: {0}".format(os.path.join(dataset_path, "maestro-v2.0.0.csv")))

    return list(zip(train_input, train_output)), list(zip(val_input, val_output)), list(zip(test_input, test_output))


def get_pitch_labels(filename: str = None):
    """
    This function returns the pitch labels.
    :param filename:
    :return:
    """
    try:
        # ‘onset time’ ‘note number’ ‘duration’ ‘velocity’ ‘channel’
        notes = midi.load_midi(filename=filename)[:, :3]
        return notes
    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


def get_onset_labels(filename: str = None):
    """
    This function returns the onset labels.
    :param filename:
    :return:
    """
    try:
        # ‘onset time’ ‘note number’ ‘duration’ ‘velocity’ ‘channel’
        onset_labels = midi.load_midi(filename=filename)[:, 0]
        return combine_events(list(dict.fromkeys(onset_labels)), 0.03, combine='mean')
    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


def get_offset_labels(filename: str = None):
    """
    This function returns the offset labels.
    :param filename:
    :return:
    """
    try:
        # ‘onset time’ ‘note number’ ‘duration’ ‘velocity’ ‘channel’
        onset_labels_and_duration = midi.load_midi(filename=filename)[:, [0, 2]]
        return combine_events(list(dict.fromkeys(np.sum(onset_labels_and_duration, axis=1))), 0.03, combine='mean')
    except FileNotFoundError:
        raise("File not found: {0}".format(filename))

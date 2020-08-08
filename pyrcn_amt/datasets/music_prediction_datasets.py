import os
import csv
import numpy as np
from joblib import load


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the MAPS Dataset
    :param dataset_path:
    :return:
    """
    raise NotImplementedError


def load_piano_midi_dataset(dataset_path: str = None, mode: int = 4, validation: bool = True):
    """
    This function aims to load a small dataset for polyphonic music generation.
    :param dataset_path:
    :param mode:
    :param validation:
    :return:
    """
    if mode == 1:
        dataset_path = os.path.normpath(os.path.join(dataset_path, "JSB_Chorales.pickle"))
    elif mode == 2:
        dataset_path = os.path.normpath(os.path.join(dataset_path, "MuseData.pickle"))
    elif mode == 3:
        dataset_path = os.path.normpath(os.path.join(dataset_path, "Nottingham.pickle"))
    elif mode == 4:
        dataset_path = os.path.normpath(os.path.join(dataset_path, "Piano-midi.de.pickle"))

    dataset = load(dataset_path)
    training_set = dataset['train']
    validation_set = dataset['valid']
    test_set = dataset['test']

    if validation:
        return (training_set, validation_set), test_set
    else:
        training_set = training_set + validation_set
        return training_set, test_set


def get_pitch_labels(filename: str = None):
    """
    This function returns the pitch labels.
    :param filename:
    :return:
    """
    try:
        notes = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for label in reader:
                start_time = float(label['OnsetTime'])
                end_time = float(label['OffsetTime'])
                note = int(label['MidiPitch'])
                notes.append([start_time, note, end_time - start_time])
        return np.array(notes)
    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


def get_onset_labels(filename: str = None):
    """
    This function returns the onset labels.
    :param filename:
    :return:
    """
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            onset_labels = []
            for label in reader:
                onset_labels.append(float(label['OnsetTime']))
        return list(dict.fromkeys(onset_labels))

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


def get_offset_labels(filename: str = None):
    """
    This function returns the offset labels.
    :param filename:
    :return:
    """
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            offset_labels = []
            for label in reader:
                offset_labels.append(float(label['OffsetTime']))
        return list(dict.fromkeys(offset_labels))

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))

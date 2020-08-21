import os
import csv
import numpy as np
from madmom.utils import combine_events


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the MAPS Dataset
    :param dataset_path:
    :return:
    """
    raise NotImplementedError


def load_dataset(dataset_path: str = None, fold_id: int = 1, validation: bool = True, configuration: int = 1):
    """
    This function aims to load the MAPS Dataset.
    :param dataset_path:
    :param fold_id:
    :param validation:
    :param configuration:
    :return:
    """
    dataset_path = os.path.normpath(dataset_path)
    if configuration == 1:
        config_path = "sigtia-conf1-splits"
    elif configuration == 2:
        config_path = "sigtia-conf2-splits"
    else:
        raise ValueError

    if fold_id == 0:
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_1', 'train')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            training_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_1', 'valid')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            validation_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_1', 'test')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            test_set = content
    elif fold_id == 1:
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_2', 'train')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            training_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_2', 'valid')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            validation_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_2', 'test')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            test_set = content
    elif fold_id == 2:
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_3', 'train')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            training_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_3', 'valid')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            validation_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_3', 'test')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            test_set = content
    elif fold_id == 3:
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_4', 'train')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            training_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_4', 'valid')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            validation_set = content
        with open(os.path.join(dataset_path, 'mapsSplits', config_path, 'fold_4', 'test')) as f:
            content = f.readlines()
            content = [(os.path.join(dataset_path, x.strip() + ".wav"),
                        os.path.join(dataset_path, x.strip() + ".txt")) for x in content]
            test_set = content
    else:
        raise ValueError

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
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            offset_labels = []
            for label in reader:
                offset_labels.append(float(label['OffsetTime']))
        return combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))

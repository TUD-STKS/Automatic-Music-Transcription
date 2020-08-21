import os
import csv
import numpy as np
from madmom.utils import combine_events


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the MusicNet Dataset
    :param dataset_path:
    :return:
    """
    from six.moves import urllib
    import tarfile
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

        data = urllib.request.urlopen('https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz')
        with open(os.path.join(dataset_path, 'musicnet.tar.gz'), 'wb') as f:
            print("Downloading...")
            # stream the download to disk (it might not fit in memory!)
            while True:
                chunk = data.read(16 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        print("Extracting...")
        tar = tarfile.open(os.path.join(dataset_path, 'musicnet.tar.gz'), "r:gz")
        tar.extractall(path=dataset_path)
        tar.close()


def load_dataset(dataset_path: str = None):
    """
    This function aims to load the MusicNet Dataset.
    :param dataset_path:
    :return:
    """
    dataset_path = os.path.normpath(dataset_path)
    train_input = [os.path.join(dataset_path, 'train_data', f) for f in os.listdir(os.path.join(dataset_path, 'train_data'))]
    train_output = [os.path.join(dataset_path, 'train_labels', f) for f in os.listdir(os.path.join(dataset_path, 'train_labels'))]
    test_input = [os.path.join(dataset_path, 'test_data', f) for f in os.listdir(os.path.join(dataset_path, 'test_data'))]
    test_output = [os.path.join(dataset_path, 'test_labels', f) for f in os.listdir(os.path.join(dataset_path, 'test_labels'))]
    return list(zip(train_input, train_output)), list(zip(test_input, test_output))


def get_pitch_labels(filename: str = None):
    """
    This function returns the pitch labels.
    :param filename:
    :return:
    """
    try:
        notes = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = float(label['start_time']) / 44100.
                end_time = float(label['end_time']) / 44100.
                note = int(label['note'])
                notes.append([start_time, note, end_time - start_time])
        return np.array(notes)
    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


def get_instrument_labels(filename: str = None):
    """
    This function returns the instrument labels.
    :param filename:
    :return:
    """
    try:
        instruments = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = float(label['start_time']) / 44100.
                end_time = float(label['end_time']) / 44100.
                instrument = int(label['instrument'])
                instruments.append([start_time, instrument, end_time - start_time])
        return np.array(instruments)
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
            reader = csv.DictReader(f, delimiter=',')
            onset_labels = []
            for label in reader:
                onset_labels.append(float(label['start_time']) / 44100.)

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
            reader = csv.DictReader(f, delimiter=',')
            offset_labels = []
            for label in reader:
                offset_labels.append(float(label['end_time']) / 44100.)
        return combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))

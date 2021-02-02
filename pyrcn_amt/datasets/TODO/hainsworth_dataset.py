import os
from sklearn.model_selection import KFold
import csv
from collections import deque


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the Hainsworth Dataset
    :param dataset_path:
    :return:
    """
    raise NotImplementedError


def load_dataset(dataset_path: str = None, fold_id: int = 0):
    """
    This function aims to load the Hainsworth Dataset.
    :param dataset_path:
    :param fold_id:
    :return:
    """
    dataset_path = os.path.normpath(dataset_path)
    file_names = []
    for fid_wav, fid_annotation in zip(os.listdir(os.path.join(dataset_path, "wav")), os.listdir(os.path.join(dataset_path, "annotations"))):
        file_names.append((os.path.join(dataset_path, "wav", fid_wav), os.path.join(dataset_path, "annotations", fid_annotation)))

    kf = KFold(n_splits=8)
    training_set = []
    test_set = []
    k = 0
    for train_index, test_index in kf.split(X=file_names):
        if k == fold_id:
            for idx in train_index:
                training_set.append(file_names[idx])
            for idx in test_index:
                test_set.append(file_names[idx])
            break
        k = k + 1
    return training_set, test_set


def get_beat_labels(filename: str = None):
    """
    This function returns the beat labels.
    :param filename:
    :return:
    """
    try:
        with open(filename, 'r') as f:
            content = f.readlines()
            beat_labels = [float(x.strip()) for x in content]
        return beat_labels

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))


if __name__ == "__main__":
    train_files, test_files = load_dataset(dataset_path=r"C:\Temp\beat_tracking_datasets\hains")
    beat_labels = get_beat_labels(filename=train_files[0][1])
    exit(0)

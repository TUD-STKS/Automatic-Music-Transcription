import os
from madmom.utils import combine_events


def download_dataset(dataset_path: str = None):
    """
    This function aims to download the Boeck Onset Dataset
    :param dataset_path:
    :return:
    """
    raise NotImplementedError


def load_dataset(dataset_path: str = None, fold_id: int = 0, validation=True):
    """
    This function aims to load the Boeck Onset Dataset.
    :param dataset_path:
    :param fold_id:
    :param validation:
    :return:
    """
    dataset_path = os.path.normpath(dataset_path)
    folds = []
    for _, _, file_names in os.walk(os.path.join(dataset_path, "onsets_splits")):
        for file_name in sorted(file_names):
            with open(os.path.join(dataset_path, "onsets_splits", file_name)) as f:
                content = f.readlines()
                content = [(os.path.join(dataset_path, "onsets_audio", x.strip() + ".flac"),
                            os.path.join(dataset_path, "onsets_annotations", x.strip() + ".onsets")) for x in content]
                folds.append(content)
    curr_folds = folds[fold_id:] + folds[:fold_id]
    test_set = curr_folds[-1]
    if validation:
        training_set = [item for sublist in curr_folds[:-2] for item in sublist]
        validation_set = curr_folds[-2]
        return (training_set, validation_set), test_set
    else:
        training_set = [item for sublist in curr_folds[:-1] for item in sublist]
        return training_set, test_set


def get_onset_labels(filename: str = None):
    """
    This function returns the onset labels.
    :param filename:
    :return:
    """
    try:
        with open(filename, 'r') as f:
            content = f.readlines()
            onset_labels = [float(x.strip()) for x in content]

        return combine_events(onset_labels, 0.03, combine='mean')

    except FileNotFoundError:
        raise("File not found: {0}".format(filename))

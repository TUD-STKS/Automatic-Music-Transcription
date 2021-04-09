import os.path

import madmom
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class BoeckOnsetCorpus:
    """Corpus object for the Boeck onset datset
    """

    audio_extension = ".flac"
    label_extension = ".onsets"
    split_extension = ".fold"

    def __init__(self, audio_dir, label_dir, split_dir):
        """Corpus(transcription_dir, audio_dir)
        Create a corpus object with transcriptions and audio files
        in the specified create_directories
        """

        self.audio_dir = os.path.realpath(audio_dir)
        self.label_dir = os.path.realpath(label_dir)
        self.split_dir = os.path.realpath(split_dir)

    def get_audio_dir(self):
        "get_audio_dir() - Return audio directory"
        return self.audio_dir

    def get_label_dir(self):
        "get_label_dir() - Return label directory"
        return self.label_dir

    def get_split_dir(self):
        "get_label_dir() - Return audio directory"
        return self.label_dir

    def get_onset_events(self, utterance):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        return the onset times in seconds

        If fs is None, returns sample numbers from TIMIT documentation,
        otherwise converts to seconds as per fs

        Returns:
        starts, stops, phones
        """

        # Construct location of phoneme transcription file
        onsets = madmom.io.load_onsets(os.path.join(self.label_dir, utterance))
        return madmom.utils.combine_events(onsets, 0.03, combine='mean')

    def get_labels(self, utterance, fps=100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        onset_times = self.get_onset_events(utterance=utterance)
        onset_targets = madmom.utils.quantize_events(events=onset_times, fps=fps, length=n_frames)
        onset_targets = madmom.audio.signal.smooth(onset_targets, np.asarray([0.5, 1.0, 0.5]))
        return onset_targets

    def get_audiofilename(self, utterance):
        """get_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.audio_dir, utterance + self.audio_extension)

    def get_labelfilename(self, utterance):
        """get_labelfilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.label_dir, utterance + self.label_extension)

    def get_utterances(self, fold=0):
        """get_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)

        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """
        fold_name = os.path.join(self.split_dir, "8-fold_cv_random_" + str(fold) + self.split_extension)
        utterances = np.loadtxt(fname=fold_name, dtype=str)
        return utterances

    def get_unlabeled_utterances(self):
        """get_unlabeled_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)

        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """
        fold_name = os.path.join(self.split_dir, "unlabeled" + self.split_extension)
        utterances = np.loadtxt(fname=fold_name, dtype=str, comments=None)
        return utterances


if __name__ == "__main__":
    corpus = BoeckOnsetCorpus(audio_dir=r"Z:\Projekt-Musik-Datenbank\OnsetDetektion\onsets_audio",
                              label_dir=r"Z:\Projekt-Musik-Datenbank\OnsetDetektion\onsets_annotations",
                              split_dir = r"Z:\Projekt-Musik-Datenbank\OnsetDetektion\onsets_splits")
    training_utterances = np.concatenate(([corpus.get_utterances(fold=fold_id) for fold_id in [0, 1, 2, 3, 4, 5]]))
    validation_utterances = corpus.get_utterances(fold=6)
    test_utterances = corpus.get_utterances(fold=7)
    exit(0)

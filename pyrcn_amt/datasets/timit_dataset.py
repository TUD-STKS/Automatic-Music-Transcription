import os
import numpy as np
import csv
import madmom


class TIMITCorpus:
    """Corpus object for the TIMIT corpus
    """

    audio_extension = ".flac"
    label_extension = ".onsets"

    # List of phoneme symbols used in phone-level transcriptions
    phones = ["b", "d", "g", "p", "t", "k", "dx", "q",
              "jh", "ch", "s", "sh", "z", "zh", "f", "th",
              "v", "dh", "m", "n", "ng", "em", "en", "eng",
              "nx", "l", "r", "w", "y", "hh", "hv", "el",
              "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay",
              "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er",
              "ax", "ix", "axr", "ax-h", "pau", "epi",
              # closure portion of stops
              "bcl", "dcl", "gcl", "pcl", "tcl", "kcl",
              # silence
              "h#"]

    silence = "h#"

    # Create dictionary to map phonemes to numbers, e.g. phone2int['g'] = 2
    phone2int = {
        "iy": 0,
        "ih": 1,
        "ix": 1,
        "eh": 2,
        "ey": 3,
        "ae": 4,
        "aa": 5,
        "ao": 5,
        "aw": 6,
        "ay": 7,
        "ah": 8,
        "ax": 8,
        "ax-h": 8,
        "oy": 9,
        "ow": 10,
        "uh": 11,
        "uw": 12,
        "ux": 12,
        "er": 13,
        "axr": 13,
        "jh": 14,
        "ch": 15,
        "b": 16,
        "bcl": 16,
        "d": 17,
        "dcl": 17,
        "g": 18,
        "gcl": 18,
        "p": 19,
        "pcl": 19,
        "t": 20,
        "tcl": 20,
        "k": 21,
        "kcl": 21,
        "dx": 22,
        "s": 23,
        "sh": 24,
        "zh": 24,
        "z": 25,
        "f": 26,
        "th": 27,
        "v": 28,
        "dh": 29,
        "m": 30,
        "em": 30,
        "n": 31,
        "en": 31,
        "nx": 31,
        "ng": 32,
        "eng": 32,
        "l": 33,
        "el": 33,
        "r": 34,
        "w": 35,
        "y": 36,
        "hh": 37,
        "hv": 37,
        "h#": 38,
        "q": 38,
        "pau": 38,
        "epi": 38,
        }
    # phone2int = dict([(key, val) for val, key in enumerate(phones)])

    def __init__(self, transcription_dir, audio_dir):
        """Corpus(transcription_dir, audio_dir)
        Create a corpus object with transcriptions and audio files
        in the specified create_directories
        """

        self.transcription_dir = os.path.realpath(transcription_dir)
        self.audio_dir = os.path.realpath(audio_dir)

    def get_audio_dir(self):
        "get_audio_dir() - Return audio directory"
        return self.audio_dir

    def get_label_dir(self):
        "get_audio_dir() - Return label directory"
        return self.transcription_dir

    @classmethod
    def get_phonemes(cls):
        """"get_phonemes()
        Return phone names used in transcriptions.
        See TIMIT doc/phoncode.doc for details on meanings
        """
        return cls.phones

    def get_phoneme_transcription(self, utterance, fs=None):
        """get_phone_transcription(utterance)
        Given a relative path to a specific utterance, e.g.
            train/dr1/jcjf0/sa1  (training directory, dialect region
                1, speaker jcjf0, sentence sa1 - see TIMIT docs)
        return the phonemes and samples/timing with which they are associated.

        If fs is None, returns sample numbers from TIMIT documentation,
        otherwise converts to seconds as per fs

        Returns:
        starts, stops, phones
        """

        # Construct location of phoneme transcription file
        phones = []
        with open(os.path.join(self.transcription_dir, utterance), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            if fs is not None:
                for label in reader:
                    start_time = np.around(float(label[0]) / fs, 2)
                    end_time = np.around((float(label[1]) - 1) / fs, 2)
                    phone = self.phone2int[label[2]]
                    phones.append([start_time, phone, end_time - start_time])
            else:
                for label in reader:
                    start_time = int(label[0])
                    end_time = int(label[1]) - 1
                    phone = self.phone2int[label[2]]
                    phones.append([start_time, phone, end_time - start_time])
        return np.asarray(phones)

    def get_phoneme_labels(self, utterance, fps=100, fs=16000, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        phoneme_times = self.get_phoneme_transcription(utterance=utterance, fs=fs)
        phoneme_targets = madmom.utils.quantize_notes(notes=phoneme_times, fps=fps, num_pitches=39, length=n_frames)

        return np.argmax(phoneme_targets, axis=1)

    def get_silence(self):
        "get_silence() - Return label for silence/noise"
        return self.silence

    def get_audiofilename(self, utterance):
        """get_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """

        audfile = os.path.join(self.audio_dir, utterance + ".wav")
        return audfile

    def get_labelfilename(self, utterance):
        """get_labelfilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.transcription_dir, utterance + ".phn")

    def get_relative_audiofilename(self, utterance):
        """get_relative_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g.
            train/dr1/jcjf0/sa1
        construct a relative pathname from the root of the corpus
        audio data to the audio file associated with it:
            train/dr1/jcjf0/sa1.wav
        """
        return utterance + ".wav"

    def get_utterances(self, utttype="train"):
        """get_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)

        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """

        utterances = []  # list of utterances

        # Traverse audio directory
        targetdir = os.path.join(self.audio_dir, utttype)

        # Walk iterates over lists of subdirectories and files in
        # a directory tree.
        for directory, _subdirs, files in os.walk(targetdir):
            # Get rid of root and remove file separator
            reldir = directory.replace(self.audio_dir, "")[1:]

            for f in files:
                if f.lower().endswith(".wav"):
                    # Found one, strip off extension and add to list
                    uttid, _ext = os.path.splitext(f)
                    utterances.append(os.path.join(reldir, uttid))

        # Return the list as a numpy array which will let us use
        # more sophisticated indexing (e.g. using a list variable
        # of indices
        return np.asarray(utterances)

    def set_feature_extractor(self, feature_extractor):
        """set_feature_extractor(feature_extractor)
        After passing in an object capable of extracting features with
        a call to get_features(fname) with a filename relative to the
        audio root, one can use get_features in this class to retrieve
        features
        """
        if not hasattr(feature_extractor, 'transform'):
            raise RuntimeError(
                'Specified feature_extractor does not support transform')

        self.feature_extractor = feature_extractor


if __name__ == "__main__":
    corpus = TIMITCorpus(transcription_dir="Z:\TIMIT", audio_dir="Z:\TIMIT")
    training_utterances = corpus.get_utterances(utttype='train')
    test_utterances = corpus.get_utterances(utttype='test')




import os
import csv
import numpy as np
import madmom


class MusicNET:
    """Corpus object for the MusicNET dataset
    """

    audio_extension = ".flac"
    label_extension = ".csv"

    def __init__(self, audio_dir, label_dir):
        """Corpus(transcription_dir, audio_dir)
        Create a corpus object with transcriptions and audio files
        in the specified create_directories
        """

        self.audio_dir = os.path.realpath(audio_dir)
        self.label_dir = os.path.realpath(label_dir)

    def get_audio_dir(self):
        "get_audio_dir() - Return audio directory"
        return self.audio_dir

    def get_label_dir(self):
        "get_label_dir() - Return label directory"
        return self.label_dir

    def get_note_events(self, utterance, fs=None):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the note events with start and stop in seconds

        If fs is None, returns note events with start and stop in samples

        Returns:
        start, note, duration
        """
        notes = []
        with open(os.path.join(self.label_dir, utterance), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            if fs is not None:
                for label in reader:
                    start_time = float(label['start_time']) / fs
                    end_time = float(label['end_time']) / fs
                    note = int(label['note'])
                    notes.append([start_time, note, end_time - start_time])
            else:
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    note = int(label['note'])
                    notes.append([start_time, note, end_time - start_time])
        return np.array(notes)

    def get_instrument_events(self, utterance, fs=None):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the instrument events with start and stop in seconds

        If fs is None, returns instrument events with start and stop in samples

        Returns:
        start, note, duration
        """
        instruments = []
        with open(os.path.join(self.label_dir, utterance), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            if fs is not None:
                for label in reader:
                    start_time = float(label['start_time']) / fs
                    end_time = float(label['end_time']) / fs
                    instrument = int(label['instrument'])
                    instruments.append([start_time, instrument, end_time - start_time])
            else:
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    instruments.append([start_time, instrument, end_time - start_time])
        return np.array(instruments)

    def get_onset_events(self, utterance, fs=None):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the instrument events with start and stop in seconds

        If fs is None, returns instrument events with start and stop in samples

        Returns:
        start, note, duration
        """
        onset_labels = []
        with open(os.path.join(self.label_dir, utterance), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            if fs is not None:
                for label in reader:
                    onset_labels.append(float(label['start_time']) / fs)
                return madmom.utils.combine_events(list(dict.fromkeys(onset_labels)), 0.03, combine='mean')
            else:
                for label in reader:
                    onset_labels.append(int(label['start_time']))
                return madmom.utils.combine_events(list(dict.fromkeys(onset_labels)), 1323, combine='mean')

    def get_offset_events(self, utterance, fs=None):
        """get_offset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the instrument events with start and stop in seconds

        If fs is None, returns instrument events with start and stop in samples

        Returns:
        start, note, duration
        """
        offset_labels = []
        with open(os.path.join(self.label_dir, utterance), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            if fs is not None:
                for label in reader:
                    offset_labels.append(float(label['end_time']) / fs)
                return madmom.utils.combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')
            else:
                for label in reader:
                    offset_labels.append(int(label['end_time']))
                return madmom.utils.combine_events(list(dict.fromkeys(offset_labels)), 1323, combine='mean')

    def get_note_labels(self, utterance, fps=100, fs=44100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        note_times = self.get_note_events(utterance=utterance, fs=fs)
        note_targets = madmom.utils.quantize_notes(notes=note_times, fps=fps, num_pitches=128, length=n_frames)
        note_targets = madmom.audio.signal.smooth(note_targets, np.array([0.25, 0.5, 0.25]))
        return note_targets

    def get_instrument_labels(self, utterance, fps=100, fs=44100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        instrument_times = self.get_instrument_events(utterance=utterance, fs=fs)
        instrument_targets = madmom.utils.quantize_notes(notes=instrument_times, fps=fps, num_pitches=11, length=n_frames)
        instrument_targets = madmom.audio.signal.smooth(instrument_targets, np.array([0.25, 0.5, 0.25]))
        return instrument_targets

    def get_onset_labels(self, utterance, fps=100, fs=44100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        onset_times = self.get_onset_events(utterance=utterance, fs=fs)
        onset_targets = madmom.utils.quantize_events(events=onset_times, fps=fps, length=n_frames)
        onset_targets = madmom.audio.signal.smooth(onset_targets, np.asarray([0.5, 1.0, 0.5]))
        return onset_targets

    def get_offset_labels(self, utterance, fps=100, fs=44100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        offset_times = self.get_offset_events(utterance=utterance, fs=fs)
        offset_targets = madmom.utils.quantize_events(events=offset_times, fps=fps, length=n_frames)
        offset_targets = madmom.audio.signal.smooth(offset_targets, np.asarray([0.5, 1.0, 0.5]))
        return offset_targets

    def get_audiofilename(self, utterance, fold="train"):
        """get_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.audio_dir, fold + "_data", utterance + ".wav")

    def get_labelfilename(self, utterance, fold="train"):
        """get_labelfilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.label_dir, fold + "_labels", utterance + ".csv")

    def get_utterances(self, fold="train"):
        """get_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)

        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """
        audio_path = os.path.join(self.audio_dir, fold + "_data")
        utterances = np.asarray([f[:-4] for f in os.listdir(audio_path)])
        return utterances


if __name__ == "__main__":
    corpus = MusicNET(audio_dir=r"Z:\Projekt-Musik-Datenbank\musicNET", label_dir=r"Z:\Projekt-Musik-Datenbank\musicNET")
    training_utterances = corpus.get_utterances(fold='train')
    utt = corpus.get_audiofilename(training_utterances[0], fold="train")
    test_utterances = corpus.get_utterances(fold='test')

import os
import csv
import numpy as np
import madmom


class MAPSDataset:
    """Corpus object for the MusicNET dataset
    """

    audio_extension = ".wav"
    label_extension = ".txt"
    split_extension = ""

    def __init__(self, audio_dir, label_dir, split_dir, configuration: int = 1):
        """Corpus(transcription_dir, audio_dir)
        Create a corpus object with transcriptions and audio files
        in the specified create_directories
        """

        self.audio_dir = os.path.realpath(audio_dir)
        self.label_dir = os.path.realpath(label_dir)
        self.split_dir = os.path.realpath(split_dir)
        self.configuration = configuration

    def get_audio_dir(self):
        "get_audio_dir() - Return audio directory"
        return self.audio_dir

    def get_label_dir(self):
        "get_label_dir() - Return label directory"
        return self.label_dir

    def get_note_events(self, utterance):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the note events with start and stop in seconds

        Returns:
        start, note, duration
        """
        notes = []
        with open(utterance, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for label in reader:
                start_time = float(label['OnsetTime'])
                end_time = float(label['OffsetTime'])
                note = int(label['MidiPitch'])
                notes.append([start_time, note, end_time - start_time])
        return np.array(notes)

    def get_onset_events(self, utterance):
        """get_onset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the instrument events with start and stop in seconds

        If fs is None, returns instrument events with start and stop in samples

        Returns:
        start, note, duration
    """
        onset_labels = []
        with open(utterance, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for label in reader:
                start_time = float(label['OnsetTime'])
                note = int(label['MidiPitch'])
                onset_labels.append([start_time, note])
        return madmom.utils.combine_events(list(dict.fromkeys(onset_labels)), 0.03, combine='mean')

    def get_offset_events(self, utterance):
        """get_offset_events(utterance)
        Given a file name of a specific utterance, e.g.
            ah_development_guitar_2684_TexasMusicForge_Dandelion_pt1
        returns the instrument events with start and stop in seconds

        If fs is None, returns instrument events with start and stop in samples

        Returns:
        start, note, duration
        """
        offset_labels = []
        with open(utterance, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for label in reader:
                start_time = float(label['OnsetTime'])
                note = int(label['MidiPitch'])
                offset_labels.append([start_time, note])
        return madmom.utils.combine_events(list(dict.fromkeys(offset_labels)), 0.03, combine='mean')

    def get_note_labels(self, utterance, fps=100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        note_times = self.get_note_events(utterance=utterance)
        note_targets = madmom.utils.quantize_notes(notes=note_times, fps=fps, num_pitches=128, length=n_frames)
        note_targets = madmom.audio.signal.smooth(note_targets, np.array([0.25, 0.5, 0.25]))
        return note_targets

    def get_onset_labels(self, utterance, fps=100, n_frames=None):
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

    def get_offset_labels(self, utterance, fps=100, n_frames=None):
        """get_labels(utterance)
        Similar to get_phoneme_transciption, but assumes that a feature
        extractor has been set and uses the feature sample rate to align
        the phoneme transcription to feature frames

        Returns list
        start - start frames
        stop - stop frames
        phonemes - phonemes[idx] is between start[idx]  and stop[idx]

        """
        offset_times = self.get_offset_events(utterance=utterance)
        offset_targets = madmom.utils.quantize_events(events=offset_times, fps=fps, length=n_frames)
        offset_targets = madmom.audio.signal.smooth(offset_targets, np.asarray([0.5, 1.0, 0.5]))
        return offset_targets

    def get_audiofilename(self, utterance):
        """get_audiofilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.audio_dir, utterance + ".wav")

    def get_labelfilename(self, utterance, fold="train"):
        """get_labelfilename(utterance)
        Given a relative path to a specific utterance, e.g.
            tra[in/dr1/jcjf0/sa1
        construct a full pathname to the audio file associated with it:
            C:\Data\corpora\timit\wav16\train/dr1/jcjf0\sa1.wav
        """
        return os.path.join(self.label_dir, fold + "_labels", utterance + ".csv")

    def get_utterances(self, fold=0, split="train"):
        """get_utterances(utttype)
        Return list of train or test utterances (as specified by utttype)

        e.g.  get_utterances('train')
        returns a list:
            [train/dr1/jcjf0/sa1, train/dr1/jcjf0/sa2, ...
             train/dr8/mtcs0/sx352]
        """
        fold_name = os.path.join(self.split_dir, "sigtia-conf" + str(self.configuration) +"-splits", "fold_" + str(fold),
                                 split)
        utterances = np.loadtxt(fname=fold_name, dtype=str)
        return utterances


if __name__ == "__main__":
    corpus = MAPSDataset(audio_dir=r"Z:\Projekt-Musik-Datenbank\MultiPitch-Tracking",
                         label_dir=r"Z:\Projekt-Musik-Datenbank\MultiPitch-Tracking",
                         split_dir=r"Z:\Projekt-Musik-Datenbank\MultiPitch-Tracking\mapsSplits",
                         configuration=1)
    training_utterances = corpus.get_utterances(fold=1, split="train")
    utt = corpus.get_audiofilename(training_utterances[0])
    test_utterances = corpus.get_utterances(fold=1, split="test")
    utt = corpus.get_audiofilename(test_utterances[0])

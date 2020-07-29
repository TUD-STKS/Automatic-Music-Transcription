import os
import numpy as np
from madmom.io.audio import load_audio_file, LoadAudioFileError
import soundfile as sf
from librosa.core import resample

from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, SpectrogramDifferenceProcessor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def parse_feature_settings(feature_settings):
    feature_settings['mono'] = bool(feature_settings['mono'])
    feature_settings['fs'] = float(feature_settings['fs'])
    feature_settings['normalize'] = bool(feature_settings['normalize'])
    try:
        feature_settings['frame_size'] = eval(feature_settings['frame_size'])
    except TypeError:
        try:
            feature_settings['frame_size'] = [int(feature_settings['frame_size'])]
        except ValueError:
            raise

    feature_settings['fps'] = float(feature_settings['fps'])
    if feature_settings['filterbank'] == 'LogarithmicFilterbank':
        feature_settings['filterbank'] = LogarithmicFilterbank
    else:
        TypeError("Unknown type of filterbank")

    feature_settings['num_bands'] = int(feature_settings['num_bands'])
    feature_settings['fmin'] = float(feature_settings['fmin'])
    feature_settings['fmax'] = float(feature_settings['fmax'])
    feature_settings['norm_filters'] = bool(feature_settings['norm_filters'])
    if feature_settings['logarithm'] == 'log10':
        feature_settings['logarithm'] = np.log10
    elif feature_settings['logarithm'] == 'log':
        feature_settings['logarithm'] = np.log10
    else:
        TypeError("Unknown type of logarithm")
    feature_settings['mul'] = float(feature_settings['mul'])
    feature_settings['add'] = float(feature_settings['add'])
    feature_settings['diff_ratio'] = float(feature_settings['diff_ratio'])
    feature_settings['positive_diffs'] = bool(feature_settings['positive_diffs'])
    if 'minmax' in feature_settings['scaler']:
        scaler_settings = feature_settings['scaler'].split('_')
        feature_settings['scaler'] = MinMaxScaler(feature_range=(scaler_settings[1], scaler_settings[2]))
    elif 'zscore' in feature_settings['scaler']:
        feature_settings['scaler'] = StandardScaler()
    elif eval(feature_settings['scaler']) is None:
        feature_settings['scaler'] = None
    else:
        TypeError("Unknown type of feature normalization")

    return feature_settings


def create_processors(feature_settings: dict):
    if feature_settings['mono']:
        sig = SignalProcessor(num_channels=1, sample_rate=feature_settings['fs'])
    else:
        raise

    multi = ParallelProcessor([])
    for frame_size in feature_settings['frame_size']:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=feature_settings['fps'])
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(filterbank=feature_settings['filterbank'],
                                            num_bands=feature_settings['num_bands'], fmin=feature_settings['fmin'],
                                            fmax=feature_settings['fmax'],
                                            norm_filters=feature_settings['norm_filters'], unique_filters=True)
        spec = LogarithmicSpectrogramProcessor(log=feature_settings['logarithm'], mul=feature_settings['mul'],
                                               add=feature_settings['add'])
        diff = SpectrogramDifferenceProcessor(diff_ratio=feature_settings['diff_ratio'],
                                              positive_diffs=feature_settings['positive_diffs'], stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor([frames, stft, filt, spec, diff]))
    pre_processor = SequentialProcessor([sig, multi, np.hstack])

    scaler = feature_settings['scaler']
    return pre_processor, scaler


def load_sound_file(file_name: os.path, feature_settings: dict):
    try:
        if feature_settings['mono']:
            s, sr = load_audio_file(filename=file_name, sample_rate=feature_settings['fs'], num_channels=1)
        else:
            s, sr = load_audio_file(filename=file_name, sample_rate=feature_settings['fs'])
    except LoadAudioFileError:
        s, sr = sf.read(file=file_name)
        if s.ndim > 1 and feature_settings['mono']:
            s = s[:, 0]
        if sr != feature_settings['fs']:
            s = resample(y=s, orig_sr=sr, target_sr=feature_settings['fs'])
    except FileNotFoundError:
        raise
    if feature_settings['normalize']:
        s = s / np.max(np.abs(s))
    return s


def extract_features(s: np.ndarray, pre_processor, scaler):
    X = pre_processor.process(data=s)
    if scaler is not None:
        U = scaler.fit_transform(X=X, y=None)
    else:
        U = X - 1.0

    return U

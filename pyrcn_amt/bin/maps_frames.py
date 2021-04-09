import numpy as np
from scipy.spatial.distance import cosine
import argparse
import os
import librosa
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    SpectrogramDifferenceProcessor

from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from joblib import dump, load, Parallel, delayed

from pyrcn.echo_state_network import ESNRegressor
from pyrcn.base import InputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression

import yaml

from pyrcn_amt.datasets.maps_dataset import MAPSDataset
from pyrcn_amt.feature_extraction.feature_extractor import FeatureExtractor
from pyrcn_amt.evaluation.multipitch_scoring import determine_threshold
from pyrcn_amt.post_processing.binarize_output import thresholding
from pyrcn_amt.evaluation.multipitch_scoring import get_mir_eval_rows


def create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100.):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))])

    sig = SignalProcessor(num_channels=1, sample_rate=sr)
    multi = ParallelProcessor([])
    for frame_size in frame_sizes:
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps_hz)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
                                            norm_filters=True, unique_filters=True)
        spec = LogarithmicSpectrogramProcessor(log=np.log10, mul=5, add=1)
        diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor([frames, stft, filt, spec, diff]))
    feature_extractor = FeatureExtractor(SequentialProcessor([sig, multi, np.hstack]))

    feature_extraction_pipeline = Pipeline([("audio_loading", audio_loading),
                                            ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


def train_maps_frames(config_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    feature_extraction_pipeline = create_feature_extraction_pipeline(sr=44100, frame_sizes=[2048], fps_hz=100)
    dump(feature_extraction_pipeline,
         os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())

    corpus = MAPSDataset(audio_dir=experiment_settings["in_folder"],
                         label_dir=experiment_settings["in_folder"],
                         split_dir=os.path.join(experiment_settings["in_folder"], "mapsSplits"),
                         configuration=3)

    losses = []
    for k in [1, 2, 3, 4]:
        training_files = corpus.get_utterances(fold=k, split="train")
        validation_files = corpus.get_utterances(fold=k, split="valid")

        tmp_losses = Parallel(n_jobs=1)(delayed(opt_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                              training_files, validation_files, experiment_settings)
                                        for params in ParameterGrid(param_grid=param_grid))
        losses.append(tmp_losses)
    dump(losses, filename=os.path.join(experiment_settings["out_folder"], 'losses.lst'))


def validate_maps_frames(config_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    feature_extraction_pipeline = create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096],
                                                                     fps_hz=100)
    dump(feature_extraction_pipeline,
         os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())

    corpus = MAPSDataset(audio_dir=experiment_settings["in_folder"],
                         label_dir=experiment_settings["in_folder"],
                         split_dir=os.path.join(experiment_settings["in_folder"], "mapsSplits"),
                         configuration=3)

    scores = []
    for params in ParameterGrid(param_grid=param_grid):
        tmp_scores = Parallel(n_jobs=1)(delayed(score_function)(base_esn, params, feature_extraction_pipeline, corpus,
                                                                k, experiment_settings)
                                        for k in range(4))
        scores.append(tmp_scores)
    dump(scores, filename=os.path.join(experiment_settings["out_folder"], 'scores.lst'))


def test_maps_frames(config_file, in_file, out_file):
    with open(config_file, 'r') as file:
        settings = yaml.full_load(stream=file)

    experiment_settings = settings['experiment']
    param_grid = settings['param_grid']

    # Make Paths
    if not os.path.isdir(experiment_settings['out_folder']):
        os.mkdir(experiment_settings['out_folder'])
    if not os.path.isdir(os.path.join(experiment_settings['out_folder'], 'models')):
        os.mkdir(os.path.join(experiment_settings['out_folder'], 'models'))

    # replicate config file and store results there
    copyfile(config_file, os.path.join(experiment_settings['out_folder'], 'config.yaml'))

    try:
        feature_extraction_pipeline = load(
            os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))
    except FileNotFoundError:
        feature_extraction_pipeline = \
            create_feature_extraction_pipeline(sr=44100, frame_sizes=[1024, 2048, 4096], fps_hz=100)
        dump(feature_extraction_pipeline,
             os.path.join(experiment_settings['out_folder'], 'models', 'feature_extraction_pipeline.joblib'))

    base_esn = ESNRegressor(input_to_node=InputToNode(), node_to_node=NodeToNode(), regressor=IncrementalRegression())
    base_esn.set_params(**param_grid)
    f_name = os.path.join(experiment_settings["out_folder"], "models", "esn_200000_True.joblib")
    try:
        esn = load(f_name)
    except FileNotFoundError:
        corpus = MAPSDataset(audio_dir=experiment_settings["in_folder"],
                             label_dir=experiment_settings["in_folder"],
                             split_dir=os.path.join(experiment_settings["in_folder"], "mapsSplits"),
                             configuration=3)
        training_files = corpus.get_utterances(fold=1, split="train")
        Parallel(n_jobs=1)(delayed(train_esn)(base_esn, params, feature_extraction_pipeline, corpus, training_files,
                                              experiment_settings)
                           for params in ParameterGrid(param_grid=param_grid))
        esn = load(f_name)

    U = feature_extraction_pipeline.transform(in_file)
    y_pred = esn.predict(X=U)
    y_pred_bin = thresholding(y_pred, 0.3)
    est_time, est_freqs = get_mir_eval_rows(y=y_pred_bin)
    with open(out_file, 'w') as f:
        for t in range(len(est_time)):
            f.write('{0}\t'.format(est_time[t]))
            notes = est_freqs[t]
            for note in notes:
                f.write('{0}\t'.format(note))
            f.write('\n')


def train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings):
    print(params)
    esn = clone(base_esn)
    esn.set_params(**params)

    for fids in training_utterances[:-1]:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, n_frames=U.shape[0])
        esn.partial_fit(X=U, y=y, postpone_inverse=True)
    U = feature_extraction_pipeline.transform(corpus.get_audiofilename(training_utterances[-1])).T
    y = corpus.get_labels(corpus.get_labelfilename(training_utterances[-1]), fps=100, n_frames=U.shape[0])
    esn.partial_fit(X=U, y=y, postpone_inverse=False)
    serialize = False
    if serialize:
        dump(esn, os.path.join(experiment_settings["out_folder"],
                               "models",
                               "esn_" + str(esn.input_to_node.hidden_layer_size) +
                               '_' + str(esn.node_to_node.bi_directional) + '.joblib'))
    return esn


def opt_function(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, validation_utterances,
                 experiment_settings):
    esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_utterances, experiment_settings)

    #  Validation
    train_loss = []
    for fids in training_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        train_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred)])

    validation_loss = []
    for fids in validation_utterances:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        y_true = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        y_true[y_true < 1] = 0
        y_pred = esn.predict(X=U)
        validation_loss.append([cosine(y_true, y_pred), mean_squared_error(y_true, y_pred)])

    return [np.mean(train_loss, axis=0), np.mean(validation_loss, axis=0)]


def score_function(base_esn, params, feature_extraction_pipeline, corpus, k, experiment_settings):
    training_files = corpus.get_utterances(fold=k, split="train")
    validation_files = corpus.get_utterances(fold=k, split="valid")

    try:
        f_name = os.path.join(experiment_settings["out_folder"],
                              "models",
                              "esn_" + str(params["node_to_node__hidden_layer_size"]) +
                              "_" + str(params['node_to_node__hidden_layer_size']) + ".joblib")
        esn = load(f_name)
    except FileNotFoundError:
        esn = train_esn(base_esn, params, feature_extraction_pipeline, corpus, training_files, experiment_settings)

    # Training set
    Y_pred_train = []
    Pitch_times_train = []
    for fids in training_files:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        pitch_labels = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        Pitch_times_train.append(pitch_labels)
        y_pred = esn.predict(X=U)
        Y_pred_train.append(y_pred)
    train_scores = determine_threshold(Y_true=Pitch_times_train, Y_pred=Y_pred_train,
                                       threshold=np.linspace(start=0.1, stop=0.4, num=16))

    # Test set
    Y_pred_validation = []
    Pitch_times_validation = []
    for fids in validation_files:
        U = feature_extraction_pipeline.transform(corpus.get_audiofilename(fids)).T
        pitch_labels = corpus.get_note_labels(corpus.get_labelfilename(fids), fps=100, fs=44100, n_frames=U.shape[0])
        Pitch_times_validation.append(pitch_labels)
        y_pred = esn.predict(X=U)
        Y_pred_validation.append(y_pred)
    validation_scores = determine_threshold(Y_true=Pitch_times_validation, Y_pred=Y_pred_validation,
                                            threshold=np.linspace(start=0.1, stop=0.4, num=16))
    return train_scores, validation_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Echo State Network')
    parser.add_argument('-inf',  type=str)

    in_file = r"Z:\Projekt-Musik-Datenbank\musicNET\train_data\1727.wav"
    out_file = r"C:\Users\Steiner\Documents\Python\Automatic-Music-Transcription\1727.f0"
    args = parser.parse_args()
    train_maps_frames(args.inf)
